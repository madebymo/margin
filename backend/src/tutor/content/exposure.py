"""Pure deterministic allocation over a versioned item bank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from tutor.schemas.assessment import (
    AssessmentItem,
    AssessmentSurface,
    ChoiceAnswerSpec,
    ContentExposureState,
    ExposureRecord,
    ItemBankDocument,
    ItemReservation,
    LessonBundleReservation,
)
from tutor.content.item_bank import bundle_leakage_problems


class AllocationError(LookupError):
    """Raised when trusted inventory cannot satisfy a reservation."""


@dataclass(frozen=True)
class AllocationResult:
    """One reserved item and the new immutable exposure state."""

    item: AssessmentItem
    reservation: ItemReservation
    state: ContentExposureState


@dataclass(frozen=True)
class BundleAllocationResult:
    """A complete disjoint lesson bundle and the new immutable state."""

    bundle: LessonBundleReservation
    items: tuple[AssessmentItem, ...]
    state: ContentExposureState


class ItemAllocator:
    """Deterministically reserves unexposed families from a validated bank."""

    def __init__(self, bank: ItemBankDocument) -> None:
        self._bank = bank
        self._items = {
            (item.item_id, item.revision): item
            for item in bank.items
        }
        self._latest_revisions = {
            item_id: max(
                item.revision for item in bank.items if item.item_id == item_id
            )
            for item_id in {item.item_id for item in bank.items}
        }

    def reserve_item(
        self,
        state: ContentExposureState,
        *,
        kc_id: str,
        surface: AssessmentSurface,
        exclude_families: Iterable[str] = (),
        visible_texts: Iterable[str] = (),
    ) -> AllocationResult:
        """Reserve the first stable eligible item and return a new state.

        A family already reserved anywhere in the episode is ineligible, even
        before it is displayed.  This makes concurrent generation unable to
        accidentally select the same mathematical family for another surface.
        """
        excluded = state.used_family_ids | frozenset(exclude_families)
        eligible = [
            item
            for item in self._bank.items
            if item.kc_id == kc_id
            and surface in item.eligible_surfaces
            and item.family_id not in excluded
            and item.revision == self._latest_revisions[item.item_id]
        ]
        candidates = sorted(
            eligible,
            key=lambda item: (
                item.allocation_order is None,
                item.allocation_order if item.allocation_order is not None else 0,
                isinstance(item.answer, ChoiceAnswerSpec),
                item.family_id,
                item.item_id,
                item.revision,
            ),
        )
        visible = tuple(visible_texts)
        if visible:
            # Any positive leak or indeterminate equivalence result makes the
            # family ineligible.  The verifier gate therefore fails closed and
            # deterministic ordering selects the next safe family.
            candidates = [
                item
                for item in candidates
                if self.answer_separated(item, visible)
            ]
        if not candidates:
            raise AllocationError(
                f"no unexposed answer-separated {surface.value} item family "
                f"remains for {kc_id}"
            )
        item = candidates[0]
        reservation = ItemReservation(
            item_id=item.item_id,
            revision=item.revision,
            family_id=item.family_id,
            kc_id=item.kc_id,
            surface=surface,
        )
        new_state = state.model_copy(
            update={"reservations": [*state.reservations, reservation]}
        )
        return AllocationResult(item=item, reservation=reservation, state=new_state)

    def reserve_lesson_bundle(
        self,
        state: ContentExposureState,
        kc_id: str,
        *,
        visible_texts: Iterable[str] = (),
    ) -> BundleAllocationResult:
        """Atomically reserve worked, guided, and three independent check-in families."""
        current = state
        visible = tuple(visible_texts)
        allocations: list[AllocationResult] = []
        for surface in (
            AssessmentSurface.WORKED_EXAMPLE,
            AssessmentSurface.GUIDED_WIDGET,
            AssessmentSurface.CHECKIN,
            AssessmentSurface.CHECKIN,
            AssessmentSurface.CHECKIN,
        ):
            allocated = self.reserve_item(
                current,
                kc_id=kc_id,
                surface=surface,
                visible_texts=visible,
            )
            allocations.append(allocated)
            current = allocated.state
        bundle = LessonBundleReservation(
            worked_example=allocations[0].reservation,
            guided_widget=allocations[1].reservation,
            checkins=[
                allocations[2].reservation,
                allocations[3].reservation,
                allocations[4].reservation,
            ],
        )
        return BundleAllocationResult(
            bundle=bundle,
            items=tuple(allocation.item for allocation in allocations),
            state=current,
        )

    def qualify_episode(
        self,
        state: ContentExposureState,
        *,
        kc_ids: Sequence[str],
        target_kc: str,
        visible_texts: Iterable[str] = (),
    ) -> None:
        """Prove the remaining inventory can support one bounded episode.

        Qualification is deliberately a dry run over immutable exposure-state
        copies.  It reserves the maximum content the current policies can make
        mandatory: three diagnostic observations, a complete lesson plus two
        post-remediation checks for every routeable KC, and both target
        capstones.  A successful check changes neither the caller's ledger nor
        allocator state.

        The released bank contract gives every family exactly one surface, but
        this method still uses the real allocator rather than counting rows so
        family retirement, latest revisions, allocation order, and dynamic
        answer-separation all participate in the proof.
        """
        ordered_kcs = tuple(dict.fromkeys(kc_ids))
        if target_kc not in ordered_kcs:
            raise AllocationError("episode target is outside its qualified KC closure")

        current = state
        visible = tuple(visible_texts)
        try:
            for kc_id in ordered_kcs:
                for _ in range(3):
                    allocation = self.reserve_item(
                        current,
                        kc_id=kc_id,
                        surface=AssessmentSurface.DIAGNOSTIC,
                        visible_texts=visible,
                    )
                    current = allocation.state

                bundle = self.reserve_lesson_bundle(
                    current,
                    kc_id,
                    visible_texts=visible,
                )
                current = bundle.state
                for _ in range(2):
                    allocation = self.reserve_item(
                        current,
                        kc_id=kc_id,
                        surface=AssessmentSurface.CHECKIN,
                        visible_texts=visible,
                    )
                    current = allocation.state

            for _ in range(2):
                allocation = self.reserve_item(
                    current,
                    kc_id=target_kc,
                    surface=AssessmentSurface.CAPSTONE,
                    visible_texts=visible,
                )
                current = allocation.state
        except AllocationError as exc:
            raise AllocationError(
                "remaining reviewed inventory cannot support a complete bounded episode"
            ) from exc

    @staticmethod
    def answer_separated(
        item: AssessmentItem,
        visible_texts: Iterable[str],
    ) -> bool:
        """Whether prior learner-visible content is proven separate from truth.

        ``bundle_leakage_problems`` reports both proven leaks and indeterminate
        parser/worker outcomes.  Treating either as unsafe prevents a timeout or
        unsupported expression from silently becoming mastery-bearing content.
        """
        return not bundle_leakage_problems(tuple(visible_texts), [item])

    def reservation_answer_separated(
        self,
        reservation: ItemReservation,
        visible_texts: Iterable[str],
    ) -> bool:
        """Apply the same gate to an item reserved earlier in a lesson bundle."""
        return self.answer_separated(self.item_for(reservation), visible_texts)

    def item_for(self, reservation: ItemReservation) -> AssessmentItem:
        """Resolve a reservation against the pinned bank."""
        try:
            item = self._items[(reservation.item_id, reservation.revision)]
        except KeyError as exc:
            raise AllocationError(
                f"reserved item {reservation.item_id}@{reservation.revision} "
                f"is absent from bank {self._bank.bank_version}"
            ) from exc
        if item.family_id != reservation.family_id or item.kc_id != reservation.kc_id:
            raise AllocationError("reservation identity does not match the pinned item")
        return item

    def record_exposure(
        self,
        state: ContentExposureState,
        reservation: ItemReservation,
        *,
        hints_seen: int | None = None,
        solution_exposed: bool | None = None,
        answer_revealed: bool | None = None,
    ) -> ContentExposureState:
        """Monotonically insert or update learner-visible exposure facts.

        Replaying the same facts is idempotent.  Later hints or a subsequently
        shown solution can only increase exposure; callers cannot erase them.
        """
        key = (reservation.item_id, reservation.revision)
        reserved = {
            (record.item_id, record.revision): record
            for record in state.reservations
        }
        if key not in reserved or reserved[key] != reservation:
            raise AllocationError("cannot expose an item that was not reserved")
        previous = next(
            (
                record
                for record in state.exposures
                if (record.item_id, record.revision) == key
            ),
            None,
        )
        if previous is not None:
            if hints_seen is not None and hints_seen < previous.hints_seen:
                raise AllocationError("hints_seen cannot decrease")
        record = ExposureRecord(
            **reservation.model_dump(),
            hints_seen=(
                hints_seen
                if hints_seen is not None
                else previous.hints_seen if previous else 0
            ),
            solution_exposed=bool(solution_exposed)
            or bool(previous and previous.solution_exposed),
            answer_revealed=bool(answer_revealed)
            or bool(previous and previous.answer_revealed),
        )
        if previous == record:
            return state
        if previous is None:
            exposures = [*state.exposures, record]
        else:
            # Exposure ledger sequence is durable provenance. A monotonic
            # update must replace its original row in place rather than move
            # it to the end; otherwise an equivalent checkpoint can no longer
            # reconcile with the append-only transition ledger after restart.
            exposures = [
                record
                if (existing.item_id, existing.revision) == key
                else existing
                for existing in state.exposures
            ]
        return state.model_copy(update={"exposures": exposures})

    def update_exposure(
        self,
        state: ContentExposureState,
        reservation: ItemReservation,
        *,
        hints_seen: int | None = None,
        solution_exposed: bool | None = None,
        answer_revealed: bool | None = None,
    ) -> ContentExposureState:
        """Named alias for the monotonic exposure upsert used by session code."""
        return self.record_exposure(
            state,
            reservation,
            hints_seen=hints_seen,
            solution_exposed=solution_exposed,
            answer_revealed=answer_revealed,
        )
