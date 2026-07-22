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

const DEFAULT_COACHING_CAPABILITY = Object.freeze({
  available: false,
  provider: "openai",
  model: "gpt-5.6",
  model_label: "GPT-5.6",
  reason: "GPT-5.6 coaching is not configured on this server.",
});

function contentModeRows(payload) {
  if (!payload || typeof payload !== "object") return [];
  const rows =
    payload.supported_content_modes ??
    payload.content_modes ??
    payload.capabilities?.supported_content_modes ??
    payload.capabilities?.content_modes;
  return Array.isArray(rows) ? rows : [];
}

function coachingRecord(payload) {
  if (!payload || typeof payload !== "object") return null;
  const direct =
    payload.coaching ??
    payload.content_mode_capabilities?.llm_coaching ??
    payload.capabilities?.coaching;
  if (direct && typeof direct === "object") return direct;
  if (typeof direct === "boolean") return { available: direct };
  if (typeof payload.coaching_available === "boolean") {
    return {
      available: payload.coaching_available,
      provider: payload.coaching_provider,
      model: payload.coaching_model,
      reason: payload.coaching_unavailable_reason,
    };
  }
  return contentModeRows(payload).find((row) => {
    const id = typeof row === "string" ? row : row?.id ?? row?.mode ?? row?.name;
    return id === "llm_coaching";
  }) ?? null;
}

function coachingModelLabel(model, provider) {
  if (/^gpt-?5\.6(?:$|[-_])/i.test(model)) return "GPT-5.6";
  if (model) return model;
  return provider === "openai" ? "OpenAI" : humanizeIdentifier(provider);
}

/**
 * Normalize an optional server declaration that genuine coaching is configured.
 *
 * Older catalogs expose no content-mode capability. They deliberately remain
 * curated-only rather than optimistically requesting a mode that the server
 * would immediately downgrade.
 */
export function normalizeCoachingCapability(...payloads) {
  for (const payload of payloads) {
    const record = coachingRecord(payload);
    if (record == null) continue;
    const isString = typeof record === "string";
    const available = isString
      ? record === "llm_coaching"
      : Boolean(record.available ?? record.configured ?? record.enabled ?? true);
    const provider = isString
      ? "openai"
      : String(record.provider ?? "openai").toLowerCase();
    const model = isString ? "gpt-5.6" : String(record.model ?? "gpt-5.6");
    const reason = isString
      ? ""
      : typeof record.reason === "string"
        ? record.reason.trim()
        : "";
    return {
      available,
      provider,
      model,
      model_label: coachingModelLabel(model, provider),
      reason:
        reason ||
        (available
          ? ""
          : "GPT-5.6 coaching is not configured on this server."),
    };
  }
  return { ...DEFAULT_COACHING_CAPABILITY };
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
  const rawAttribution =
    entry.generated_by && typeof entry.generated_by === "object"
      ? entry.generated_by
      : entry.attribution && typeof entry.attribution === "object"
        ? entry.attribution
        : null;
  const isCoach = ["coach", "coaching", "llm_coaching"].includes(
    String(kindValue).toLowerCase(),
  );
  const attribution =
    rawAttribution || isCoach
      ? {
          provider: String(rawAttribution?.provider ?? entry.provider ?? "openai"),
          model: String(rawAttribution?.model ?? entry.model ?? "gpt-5.6"),
          policy_version:
            (rawAttribution?.policy_version ?? rawAttribution?.prompt_version) == null
              ? null
              : String(
                  rawAttribution.policy_version ?? rawAttribution.prompt_version,
                ),
          focus:
            rawAttribution?.focus == null ? null : String(rawAttribution.focus),
        }
      : null;
  const coachLabel = attribution
    ? `${coachingModelLabel(attribution.model, attribution.provider)} coach`
    : null;
  return {
    id: String(entry.id ?? entry.sequence ?? `${key}-${index}`),
    key: String(key),
    role,
    kind: String(kindValue),
    text,
    content_blocks: contentBlocks,
    prompt_segments: promptSegments,
    widget: entry.widget ?? entry.interaction?.widget ?? null,
    widget_state: entry.widget_state ?? null,
    widget_status: entry.widget_status ?? entry.status ?? null,
    widget_attempt_number: entry.widget_attempt_number ?? null,
    generated_by: attribution,
    attribution,
    coach_label: coachLabel,
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
  const legacyWidget =
    value.widget ??
    value.interaction?.widget ??
    (finalInteraction?.key === key ? finalInteraction.widget : null);
  const typedInput = normalizePendingInput(value.input);
  const widget = typedInput?.widget ?? legacyWidget;
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
      typedInput?.input_mode ??
      value.input_mode ??
      (widget || String(kind).includes("widget") ? "widget" : "text"),
    input: typedInput?.input ?? null,
    answer_kind: typedInput?.answer_kind ?? null,
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
    choice_options:
      typedInput?.choice_options ??
      (Array.isArray(value.choice_options)
        ? value.choice_options.map(String)
        : []),
    label: typedInput?.label ?? value.label ?? "Your answer",
    placeholder:
      typedInput?.placeholder ?? value.placeholder ?? "Type your answer",
    help_text: typedInput?.help_text ?? value.help_text ?? "",
    max_length: typedInput?.max_length ?? value.max_length ?? 256,
    widget,
    widget_state:
      typedInput?.widget_state ??
      value.widget_state ??
      value.widget_current_state ??
      null,
  };
}

function normalizePendingInput(input) {
  if (!input || typeof input !== "object") return null;
  if (input.type === "text" || input.type === "legacy_text") {
    const maxLength = normalizeMaxLength(input.max_length);
    return {
      input_mode: "text",
      input: {
        type: String(input.type),
        ...(input.type === "text" ? { answer_kind: String(input.answer_kind) } : {}),
        label: String(input.label ?? "Your answer"),
        placeholder: String(input.placeholder ?? "Type your answer"),
        help_text: String(input.help_text ?? ""),
        max_length: maxLength,
      },
      answer_kind: input.answer_kind == null ? null : String(input.answer_kind),
      label: String(input.label ?? "Your answer"),
      placeholder: String(input.placeholder ?? "Type your answer"),
      help_text: String(input.help_text ?? ""),
      max_length: maxLength,
      choice_options: [],
      widget: null,
      widget_state: null,
    };
  }
  if (input.type === "legacy_choice") {
    const options = Array.isArray(input.options) ? input.options.map(String) : [];
    return {
      input_mode: "choice",
      input: {
        type: "legacy_choice",
        label: String(input.label ?? "Choose an answer"),
        options,
      },
      label: String(input.label ?? "Choose an answer"),
      placeholder: "",
      help_text: "",
      max_length: 256,
      choice_options: options,
      widget: null,
      widget_state: null,
    };
  }
  if (input.type === "mapping_v1") {
    const rows = Array.isArray(input.rows)
      ? input.rows.map((row) => ({
          entry_id: String(row.entry_id),
          label: String(row.label),
          spoken_text: String(row.spoken_text),
          selected_option_id:
            row.selected_option_id == null ? null : String(row.selected_option_id),
        }))
      : [];
    const options = Array.isArray(input.options)
      ? input.options.map((option) => ({
          entry_id: String(option.entry_id),
          label: String(option.label),
          spoken_text: String(option.spoken_text),
        }))
      : [];
    const prompt = String(input.prompt ?? "");
    return {
      input_mode: "widget",
      input: { type: "mapping_v1", prompt, rows, options },
      label: "Guided matching practice",
      placeholder: "",
      help_text: "Match every row to one option.",
      max_length: 256,
      choice_options: [],
      widget: {
        widget_type: "mapping_v1",
        interaction_version: "mapping_v1",
        prompt,
        presentation: {
          prompt,
          rows: rows.map(({ selected_option_id: _selected, ...row }) => row),
          options,
        },
      },
      widget_state: {
        rows: rows.map((row) => ({
          id: row.entry_id,
          value: row.selected_option_id ?? "",
        })),
      },
    };
  }
  if (input.type === "slider_v1") {
    const presentation = {
      prompt: String(input.prompt ?? ""),
      label: String(input.label ?? "Value"),
      help_text: String(input.help_text ?? ""),
      minimum: Number(input.minimum),
      maximum: Number(input.maximum),
      step: Number(input.step),
      initial_value: Number(input.initial_value),
      value_label: String(input.value_label ?? "Selected value"),
      ...(input.result_template == null
        ? {}
        : { result_template: String(input.result_template) }),
    };
    const currentValue = Number(input.current_value);
    return {
      input_mode: "widget",
      input: {
        type: "slider_v1",
        ...presentation,
        current_value: currentValue,
      },
      label: presentation.label,
      placeholder: "",
      help_text: presentation.help_text,
      max_length: 256,
      choice_options: [],
      widget: {
        widget_type: "slider_v1",
        interaction_version: "slider_v1",
        prompt: presentation.prompt,
        presentation,
      },
      widget_state: { value: currentValue },
    };
  }
  return null;
}

function normalizeMaxLength(value) {
  const parsed = Number(value ?? 256);
  if (!Number.isFinite(parsed)) return 256;
  return Math.min(256, Math.max(1, Math.trunc(parsed)));
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
    release_id: raw.release_id ?? null,
    release_digest: raw.release_digest ?? null,
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
