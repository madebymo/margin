"""RAG-based pack ingestion: pedagogical resources -> draft PedagogyPacks.

This is the plan's offline pack builder ("retrieval + LLM + human review"):
load a local corpus of pedagogical resources (.md/.txt), retrieve the most
relevant excerpts per KC with a dependency-free BM25 index, and have the LLM
compile them into schema-valid PedagogyPack drafts (review_status =
llm_generated). Runs at authoring time — never in the session hot path.

Safety properties:
- Misconception/metaphor ids are built server-side from model-proposed slugs
  (``m.<kc>.<slug>`` / ``met.<slug>``), so the membership-validation contract
  used by error analysis stays intact.
- KCs whose drafts fail validation are skipped and reported — never fabricated.
- Human-authored template packs always override drafts (see packs/loader.py).

Usage:
    python -m tutor.packs.ingest --sources ~/corpus --dry-run
    python -m tutor.packs.ingest --sources ~/corpus                       # all KCs
    python -m tutor.packs.ingest --sources ~/corpus --kcs kc.der.chain_rule
"""

import argparse
import logging
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from tutor.llm import prompts
from tutor.llm.client import LLMClient, LLMError
from tutor.packs.loader import GENERATED_DIR
from tutor.schemas.common import ReviewStatus, WidgetType
from tutor.schemas.kc import GraphDocument, KCNode
from tutor.schemas.pedagogy import Metaphor, Misconception, PedagogyPack
from tutor.seed.load_seed import load_graph

load_dotenv()

logger = logging.getLogger("tutor.packs")

_VALID_WIDGETS = {widget.value for widget in WidgetType}


# -- corpus ---------------------------------------------------------------------


@dataclass
class Chunk:
    """One retrievable excerpt of a source document."""

    source: str
    text: str


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """Split text into paragraph-aligned chunks of at most ``max_chars``."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def load_corpus(sources_dir: Path | str) -> list[Chunk]:
    """Load all .md/.txt files under a directory into chunks."""
    chunks: list[Chunk] = []
    for path in sorted(Path(sources_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for piece in chunk_text(text):
            chunks.append(Chunk(source=path.name, text=piece))
    return chunks


# -- retrieval -------------------------------------------------------------------


_STOPWORDS = frozenset(
    "the a an of to is in for and or with then that this as on by it its at be "
    "are was from into when their they you your often use used".split()
)


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    ]


class Bm25Index:
    """Dependency-free BM25 over corpus chunks (deterministic, offline)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._token_counts = [Counter(_tokenize(chunk.text)) for chunk in chunks]
        self._doc_lens = [sum(counts.values()) for counts in self._token_counts]
        total = sum(self._doc_lens)
        self._avg_len = (total / len(chunks)) if chunks else 0.0
        document_frequency: Counter = Counter()
        for counts in self._token_counts:
            document_frequency.update(counts.keys())
        n = len(chunks)
        self._idf = {
            token: math.log((n - df + 0.5) / (df + 0.5) + 1)
            for token, df in document_frequency.items()
        }

    def search(
        self, query: str, top_k: int = 4, min_ratio: float = 0.25
    ) -> list[tuple[Chunk, float]]:
        """Rank chunks for a query.

        Only positive-scoring chunks are returned, and weak matches below
        ``min_ratio`` of the best score are dropped — an irrelevant excerpt
        in the prompt is worse than none.
        """
        query_tokens = _tokenize(query)
        scored: list[tuple[Chunk, float]] = []
        for index, counts in enumerate(self._token_counts):
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token)
                if not frequency:
                    continue
                idf = self._idf.get(token, 0.0)
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * self._doc_lens[index] / (self._avg_len or 1.0)
                )
                score += idf * frequency * (self._k1 + 1) / denominator
            if score > 0:
                scored.append((self._chunks[index], score))
        scored.sort(key=lambda pair: -pair[1])
        if not scored:
            return []
        cutoff = scored[0][1] * min_ratio
        return [pair for pair in scored[:top_k] if pair[1] >= cutoff]


def kc_query(node: KCNode) -> str:
    """Retrieval query for one KC: name, id words, description, examples."""
    tail_words = node.id.rsplit(".", 1)[1].replace("_", " ")
    examples = " ".join(node.canonical_examples)
    return f"{node.name} {tail_words} {node.description} {examples}"


# -- pack compilation --------------------------------------------------------------


def _slug(value: object, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")[:40]
    return slug or fallback


def _validated_pack(node: KCNode, data: dict, sources: list[str]) -> PedagogyPack:
    """Normalize a model draft into a schema-valid pack (ids built server-side)."""
    kc_tail = node.id.rsplit(".", 1)[1]
    misconceptions: list[Misconception] = []
    for index, raw in enumerate(list(data.get("misconceptions", []))[:4]):
        if not isinstance(raw, dict):
            continue
        slug = _slug(raw.get("slug", ""), f"misconception_{index}")
        misconceptions.append(
            Misconception(
                id=f"m.{kc_tail}.{slug}",
                description=str(raw.get("description", "")).strip(),
                error_signature=str(raw.get("error_signature", "")).strip(),
                remediation_hint=str(raw.get("remediation_hint", "")).strip(),
            )
        )
    if not misconceptions:
        raise LLMError("no usable misconceptions in pack draft")

    metaphors: list[Metaphor] = []
    for index, raw in enumerate(list(data.get("metaphors", []))[:2]):
        if not isinstance(raw, dict):
            continue
        slug = _slug(raw.get("slug", ""), f"metaphor_{index}")
        affinity = [
            widget
            for widget in raw.get("widget_affinity", [])
            if widget in _VALID_WIDGETS
        ] or ["live_input"]
        metaphors.append(
            Metaphor(
                id=f"met.{slug}",
                description=str(raw.get("description", "")).strip(),
                widget_affinity=affinity,
            )
        )

    error_patterns = [
        str(pattern).strip()
        for pattern in list(data.get("error_patterns", []))[:4]
        if str(pattern).strip()
    ]
    return PedagogyPack(
        kc_id=node.id,
        misconceptions=misconceptions,
        metaphors=metaphors,
        error_patterns=error_patterns,
        sources=sorted(set(sources)),
        review_status=ReviewStatus.LLM_GENERATED,
    )


def build_pack(
    node: KCNode,
    client: LLMClient,
    index: Bm25Index,
    top_k: int = 4,
    max_attempts: int = 2,
) -> PedagogyPack | None:
    """Retrieve excerpts and compile one pack; None (never fabricated) on failure."""
    hits = index.search(kc_query(node), top_k=top_k)
    excerpts = [(chunk.source, chunk.text) for chunk, _ in hits]
    sources = [source for source, _ in excerpts]
    for _ in range(max_attempts):
        try:
            data = client.complete_json(
                system=prompts.PACK_SYSTEM,
                user=prompts.pack_user(node, excerpts),
                tag=f"pack:{node.id}",
            )
            return _validated_pack(node, data, sources)
        except (LLMError, ValidationError) as exc:
            logger.warning("pack draft failed for %s: %s", node.id, exc)
    return None


# -- pipeline ------------------------------------------------------------------------


@dataclass
class IngestReport:
    """Outcome of one ingestion run."""

    generated: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    chunk_count: int = 0
    source_count: int = 0


def run_ingest(
    graph: GraphDocument,
    sources_dir: Path | str,
    out_dir: Path | str,
    client: LLMClient,
    kc_ids: set[str] | None = None,
    top_k: int = 4,
) -> IngestReport:
    """Compile draft packs for the selected KCs (default: all graph nodes)."""
    chunks = load_corpus(sources_dir)
    index = Bm25Index(chunks)
    report = IngestReport(
        chunk_count=len(chunks), source_count=len({chunk.source for chunk in chunks})
    )
    resolved_out = Path(out_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)
    nodes = [
        node
        for node in sorted(graph.nodes, key=lambda n: n.id)
        if kc_ids is None or node.id in kc_ids
    ]
    for node in nodes:
        pack = build_pack(node, client, index, top_k=top_k)
        if pack is None:
            report.skipped.append((node.id, "generation or validation failed"))
            continue
        path = resolved_out / f"{node.id.replace('.', '_')}.json"
        path.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")
        report.generated.append(node.id)
    return report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, help="directory of .md/.txt resources")
    parser.add_argument("--out", default=str(GENERATED_DIR), help="output directory")
    parser.add_argument("--kcs", nargs="*", default=None, help="restrict to these kc ids")
    parser.add_argument("--top-k", type=int, default=4, help="excerpts per KC")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="retrieval preview only: no LLM calls, no files written",
    )
    args = parser.parse_args(argv)

    graph = load_graph()
    kc_ids = set(args.kcs) if args.kcs else None
    unknown = (kc_ids or set()) - graph.node_ids()
    if unknown:
        print(f"unknown kc ids: {sorted(unknown)}", file=sys.stderr)
        return 1

    if args.dry_run:
        chunks = load_corpus(args.sources)
        index = Bm25Index(chunks)
        print(f"corpus: {len(chunks)} chunks from {len({c.source for c in chunks})} files")
        for node in sorted(graph.nodes, key=lambda n: n.id):
            if kc_ids is not None and node.id not in kc_ids:
                continue
            hits = index.search(kc_query(node), top_k=args.top_k)
            summary = ", ".join(f"{chunk.source}({score:.1f})" for chunk, score in hits)
            print(f"{node.id}: {summary or '(no relevant excerpts)'}")
        return 0

    from tutor.llm.factory import build_client

    try:
        client = build_client(args.provider)
    except LLMError as exc:
        print(f"LLM client unavailable: {exc}", file=sys.stderr)
        return 1

    report = run_ingest(
        graph, args.sources, args.out, client, kc_ids=kc_ids, top_k=args.top_k
    )
    print(
        f"generated {len(report.generated)} pack(s) into {args.out} "
        f"(corpus: {report.chunk_count} chunks / {report.source_count} files)"
    )
    for kc_id, reason in report.skipped:
        print(f"skipped {kc_id}: {reason}", file=sys.stderr)
    return 0 if report.generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
