const GOAL_FALLBACKS = [
  {
    id: "integral-substitution",
    target_kc: "kc.int.u_substitution",
    title: "Integrals using substitution",
    description: "Recognize a composite integrand and reverse the chain rule.",
  },
  {
    id: "derivative-chain-rule",
    target_kc: "kc.der.chain_rule",
    title: "Derivatives using the chain rule",
    description: "Differentiate a composition of functions.",
  },
  {
    id: "fundamental-theorem",
    target_kc: "kc.int.ftc",
    title: "The Fundamental Theorem of Calculus",
    description: "Connect accumulation, derivatives, and definite integrals.",
  },
  {
    id: "product-quotient-rule",
    target_kc: "kc.der.product_quotient",
    title: "Product and quotient rules",
    description: "Differentiate products and quotients of functions.",
  },
  {
    id: "solve-quadratics",
    target_kc: "kc.alg.solve_quadratic",
    title: "Solve quadratic equations",
    description: "Choose and apply a reliable method for a quadratic equation.",
  },
];

export const fallbackGoals = Object.freeze(
  GOAL_FALLBACKS.map((goal) => Object.freeze({ ...goal })),
);

export function humanizeIdentifier(value) {
  if (!value) return "";
  const finalPart = String(value).split(".").at(-1) ?? String(value);
  return finalPart
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function normalizeGoals(payload) {
  const rows = Array.isArray(payload)
    ? payload
    : payload?.goals ?? payload?.items ?? [];
  if (!Array.isArray(rows)) return [];
  return rows
    .map((row) => {
      const id = row.goal_id ?? row.id ?? row.target_kc;
      const targetKc = row.target_kc ?? row.kc_id ?? id;
      if (!id || !targetKc) return null;
      return {
        id: String(id),
        target_kc: String(targetKc),
        title: row.title ?? row.label ?? humanizeIdentifier(targetKc),
        description: row.description ?? row.summary ?? "",
        available: row.available !== false,
      };
    })
    .filter((goal) => goal && goal.available !== false);
}

const ROLLOUT_STATUSES = new Set([
  "available",
  "not_selected",
  "paused",
  "content_unavailable",
]);
const ROLLOUT_PERCENTAGES = new Set([0, 5, 25, 100]);

export function normalizeGoalCatalog(payload) {
  const goals = normalizeGoals(payload);
  const raw = payload && !Array.isArray(payload) ? payload.rollout : null;
  const status = ROLLOUT_STATUSES.has(raw?.status)
    ? raw.status
    : goals.length > 0
      ? "available"
      : "content_unavailable";
  const percentage = ROLLOUT_PERCENTAGES.has(raw?.percentage)
    ? raw.percentage
    : 100;
  const reason = typeof raw?.reason === "string" ? raw.reason.trim() : "";
  return {
    goals,
    rollout: { status, reason, percentage },
  };
}

export function catalogEmptyMessage(rollout) {
  const reason = rollout?.reason?.trim();
  switch (rollout?.status) {
    case "not_selected":
      return {
        title: "Pilot access is expanding",
        body:
          reason ||
          "This browser is not included in the current pilot cohort yet.",
      };
    case "paused":
      return {
        title: "New sessions are temporarily paused",
        body:
          reason ||
          "The current pilot is paused while the tutoring service is checked.",
      };
    case "content_unavailable":
      return {
        title: "Reviewed goals are being prepared",
        body:
          reason ||
          "New sessions will appear when every goal and prerequisite is ready.",
      };
    default:
      return {
        title: "No new goals are available",
        body: reason || "Please check availability again soon.",
      };
  }
}

function segmentText(segment) {
  if (typeof segment === "string") return segment;
  if (!segment || typeof segment !== "object") return "";
  if (typeof segment.text === "string") return segment.text;
  if (typeof segment.expression === "string") return segment.expression;
  if (typeof segment.latex === "string") return segment.latex;
  if (typeof segment.math === "string") return segment.math;
  if (segment.type === "blank" || segment.kind === "blank") {
    return segment.label || "___";
  }
  return "";
}

function visibleText(value) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(segmentText).filter(Boolean).join(" ");
  if (!value || typeof value !== "object") return "";
  if (typeof value.text === "string") return value.text;
  if (Array.isArray(value.segments)) return visibleText(value.segments);
  return "";
}

function transcriptRows(raw) {
  if (Array.isArray(raw.transcript)) return raw.transcript;
  if (Array.isArray(raw.messages)) return raw.messages;
  if (Array.isArray(raw.interactions)) return raw.interactions;
  return [];
}

function normalizeTranscriptEntry(entry, index) {
  const roleValue = entry.role ?? entry.speaker ?? "";
  const kindValue = entry.kind ?? entry.type ?? "message";
  const role =
    roleValue === "student" || roleValue === "user" || kindValue === "you"
      ? "student"
      : roleValue === "system" || kindValue === "error"
        ? "system"
        : "tutor";
  const text =
    visibleText(entry.text) ||
    visibleText(entry.content) ||
    visibleText(entry.message) ||
    visibleText(entry.prompt);
  const key =
    entry.interaction_key ??
    entry.pending_key ??
    entry.key ??
    entry.id ??
    `entry-${index}`;
  const promptSegments = Array.isArray(entry.prompt_segments)
    ? entry.prompt_segments
    : Array.isArray(entry.content)
      ? entry.content
      : [];
  const contentBlocks = Array.isArray(entry.content_blocks)
    ? entry.content_blocks
        .filter((block) => block && typeof block === "object")
        .map((block) => ({
          kind: String(block.kind ?? "text"),
          text: typeof block.text === "string" ? block.text : "",
          segments: Array.isArray(block.segments) ? block.segments : [],
        }))
    : [];
  return {
    id: String(entry.id ?? entry.sequence ?? `${key}-${index}`),
    key: String(key),
    role,
    kind: String(kindValue),
    text,
    content_blocks: contentBlocks,
    prompt_segments: promptSegments,
    widget: entry.widget ?? entry.interaction?.widget ?? null,
    widget_status: entry.widget_status ?? entry.status ?? null,
    widget_attempt_number: entry.widget_attempt_number ?? null,
    raw: entry,
  };
}

function normalizePending(raw, transcript) {
  const value =
    raw.pending ??
    raw.pending_interaction ??
    (raw.pending_kind
      ? {
          kind: raw.pending_kind,
          kc_id: raw.pending_kc,
          key: raw.pending_key,
        }
      : null);
  if (!value) return null;
  const finalInteraction = [...transcript].reverse().find((entry) =>
    ["probe", "checkin", "capstone", "widget"].includes(entry.kind),
  );
  const key =
    value.pending_key ??
    value.interaction_key ??
    value.key ??
    finalInteraction?.key ??
    null;
  const kind = value.kind ?? value.type ?? raw.pending_kind ?? "answer";
  const widget =
    value.widget ??
    value.interaction?.widget ??
    (finalInteraction?.key === key ? finalInteraction.widget : null);
  const rawHint = value.hint ?? {};
  const hintAvailable = rawHint.available ?? value.can_hint ?? raw.can_hint ?? true;
  const hintTotal = Number(rawHint.total ?? value.hint_total ?? 0);
  const hintNextIndex = Number(
    rawHint.next_index ?? value.hint_index ?? value.hints_given ?? 0,
  );
  return {
    key: key == null ? null : String(key),
    kind: String(kind),
    kc_id: value.kc_id ?? value.kc ?? raw.pending_kc ?? null,
    skill_label:
      value.skill_label ??
      value.skill_name ??
      value.kc_label ??
      humanizeIdentifier(value.kc_id ?? value.kc ?? raw.pending_kc),
    input_mode:
      value.input_mode ??
      (widget || String(kind).includes("widget") ? "widget" : "text"),
    can_hint: Boolean(hintAvailable),
    hint: {
      available: Boolean(hintAvailable),
      next_index: Number.isFinite(hintNextIndex) ? hintNextIndex : 0,
      total: Number.isFinite(hintTotal) ? hintTotal : 0,
      next_reveals_answer: Boolean(
        rawHint.next_reveals_answer ?? value.next_hint_reveals_answer ?? false,
      ),
    },
    hint_index: Number.isFinite(hintNextIndex) ? hintNextIndex : 0,
    prompt: visibleText(value.prompt ?? value.content),
    prompt_segments: Array.isArray(value.prompt_segments)
      ? value.prompt_segments
      : [],
    choice_options: Array.isArray(value.choice_options)
      ? value.choice_options.map(String)
      : [],
    placeholder: value.placeholder ?? "Type your answer",
    widget,
  };
}

function stringList(...candidates) {
  const value = candidates.find((candidate) => Array.isArray(candidate)) ?? [];
  return value
    .map((item) =>
      typeof item === "string"
        ? humanizeIdentifier(item)
        : item?.label ?? item?.title ?? humanizeIdentifier(item?.kc_id ?? item?.id),
    )
    .filter(Boolean);
}

function normalizeSummary(raw) {
  const value = raw.learner_summary ?? raw.mastery_summary ?? raw.summary ?? {};
  return {
    confirmed_strengths: stringList(
      value.confirmed_strengths,
      value.strengths,
      value.mastered,
    ),
    confirmed_gaps: stringList(
      value.confirmed_gaps,
      value.gaps,
      value.needs_practice,
    ),
    uncertain: stringList(
      value.uncertain,
      value.uncertain_skills,
      value.unresolved,
    ),
  };
}

const PLAN_STEP_LABELS = Object.freeze({
  teach_confirmed_gap: "Teaching a confirmed gap",
  verify_uncertain: "Checking an uncertain skill",
  practice_target: "Practicing the goal skill",
});

const PHASE_PROGRESS_LABELS = Object.freeze({
  intake: "Preparing your session",
  diagnose: "Checking what you already know",
  plan: "Building your learning path",
  teach: "Working through your learning path",
  capstone: "Solving an unseen goal problem",
  done: "Learning path complete",
  stopped: "Session ended",
});

function progressLabel(value, phase) {
  const candidate =
    value.label ?? value.current_step_label ?? value.plan_step ?? null;
  if (candidate) {
    const text = String(candidate);
    return (
      PLAN_STEP_LABELS[text] ??
      (/^[a-z0-9_.-]+$/.test(text) ? humanizeIdentifier(text) : text)
    );
  }
  return PHASE_PROGRESS_LABELS[phase] ?? phaseLabel(phase);
}

function normalizeProgress(raw, phase) {
  const value = raw.progress ?? {};
  const completed = Number(value.completed ?? value.current ?? value.step ?? 0);
  const total = Number(value.total ?? value.steps ?? 0);
  const explicitPercent = Number(value.percent ?? value.percentage);
  const probesUsed = Number(
    value.probes_used ??
      value.diagnosis_probes_used ??
      value.diagnosis_used ??
      0,
  );
  const probeBudget = Number(
    value.probe_budget ??
      value.diagnosis_probe_budget ??
      value.diagnosis_budget,
  );
  const percent = Number.isFinite(explicitPercent)
    ? explicitPercent
    : total > 0
      ? (completed / total) * 100
      : phase === "diagnose" && Number.isFinite(probeBudget) && probeBudget > 0
        ? (probesUsed / probeBudget) * 100
        : 0;
  const barKind =
    total > 0 || Number.isFinite(explicitPercent)
      ? "path"
      : phase === "diagnose" &&
          Number.isFinite(probeBudget) &&
          probeBudget > 0
        ? "diagnosis"
        : null;
  return {
    label: progressLabel(value, phase),
    completed: Number.isFinite(completed) ? completed : 0,
    total: Number.isFinite(total) ? total : 0,
    percent: Math.max(0, Math.min(100, Number.isFinite(percent) ? percent : 0)),
    probes_used: Number.isFinite(probesUsed) ? probesUsed : null,
    probe_budget: Number.isFinite(probeBudget) ? probeBudget : null,
    bar_kind: barKind,
    bar_label:
      barKind === "diagnosis" ? "Diagnosis budget used" : "Learning path progress",
  };
}

function unwrap(payload) {
  return payload?.view ?? payload?.session_view ?? payload?.session ?? payload ?? {};
}

export function normalizeSessionView(payload) {
  const raw = unwrap(payload);
  const transcript = transcriptRows(raw).map(normalizeTranscriptEntry);
  const targetKc =
    raw.goal?.target_kc ??
    raw.goal?.kc_id ??
    raw.target?.kc_id ??
    raw.target_kc ??
    null;
  const phase = String(raw.phase ?? raw.status ?? "intake").toLowerCase();
  const effectiveMode =
    raw.content_mode?.effective ??
    raw.content_mode_effective ??
    raw.effective_content_mode ??
    (raw.llm_enabled ? "llm_coaching" : "curated");
  return {
    session_id: raw.session_id ?? raw.id ?? null,
    revision: Number(raw.revision ?? raw.version ?? 0),
    phase,
    terminal: raw.terminal ?? (phase === "done" || phase === "stopped"),
    durability: raw.durability ?? (raw.persistence_enabled ? "durable" : "memory_only"),
    goal: {
      id: raw.goal?.goal_id ?? raw.goal?.id ?? targetKc,
      target_kc: targetKc,
      title:
        raw.goal?.title ??
        raw.goal?.label ??
        raw.target?.label ??
        humanizeIdentifier(targetKc),
      description: raw.goal?.description ?? "",
    },
    content_mode: {
      requested:
        raw.content_mode?.requested ??
        raw.content_mode_requested ??
        effectiveMode,
      effective: effectiveMode,
      fallback_reason:
        raw.content_mode?.fallback_reason ?? raw.content_fallback_reason ?? null,
    },
    transcript,
    pending: normalizePending(raw, transcript),
    progress: normalizeProgress(raw, phase),
    learner_summary: normalizeSummary(raw),
    last_action: raw.last_action ?? payload?.result ?? payload?.action_result ?? null,
    raw,
  };
}

export function phaseLabel(phase) {
  const labels = {
    intake: "Getting ready",
    diagnose: "Finding your starting point",
    plan: "Building your path",
    teach: "Learning",
    capstone: "Goal problem",
    done: "Complete",
    stopped: "Stopped",
  };
  return labels[String(phase).toLowerCase()] ?? humanizeIdentifier(phase);
}

export function isWidgetPending(pending) {
  if (!pending) return false;
  if (pending.input_mode != null) {
    return pending.input_mode === "widget";
  }
  return Boolean(pending.widget || String(pending.kind).includes("widget"));
}

export function pendingAcceptsText(pending) {
  if (!pending || isWidgetPending(pending)) return false;
  return !["none", "message", "read_only"].includes(String(pending.input_mode));
}

export function canPreserveAnswerDraft(previousView, nextView) {
  const previousPending = previousView?.pending;
  const nextPending = nextView?.pending;
  return Boolean(
    previousPending?.key &&
      previousPending.key === nextPending?.key &&
      pendingAcceptsText(previousPending) &&
      pendingAcceptsText(nextPending),
  );
}
