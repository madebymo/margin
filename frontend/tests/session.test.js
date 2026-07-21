import { describe, expect, it } from "vitest";

import {
  catalogEmptyMessage,
  canPreserveAnswerDraft,
  isWidgetPending,
  normalizeGoalCatalog,
  normalizeGoals,
  normalizeSessionView,
  pendingAcceptsText,
} from "../src/session.js";

describe("authoritative session normalization", () => {
  it("normalizes a wrapped v2 snapshot without exposing unknown server fields", () => {
    const view = normalizeSessionView({
      view: {
        session_id: "session-1",
        revision: 7,
        phase: "diagnose",
        durability: "durable",
        goal: {
          goal_id: "chain-rule",
          target_kc: "kc.der.chain_rule",
          title: "Chain rule",
        },
        transcript: [
          {
            sequence: 1,
            role: "tutor",
            kind: "probe",
            interaction_key: "probe-1",
            content: [
              { type: "text", text: "Differentiate" },
              { type: "math", latex: "x^2" },
              { type: "blank" },
            ],
          },
        ],
        pending: {
          key: "probe-1",
          kind: "probe",
          kc_id: "kc.der.power_rule",
          skill_name: "Using the power rule",
          input_mode: "text",
          can_hint: false,
          hint: {
            available: false,
            next_index: 3,
            total: 3,
            next_reveals_answer: false,
          },
        },
        progress: {
          phase: "diagnose",
          current_skill: "Using the power rule",
          plan_step: "Confirming the target",
          diagnosis_probes_used: 2,
          diagnosis_probe_budget: 5,
        },
        learner_summary: {
          confirmed_strengths: ["kc.alg.exponent_rules"],
          confirmed_gaps: [],
          uncertain_skills: ["Power rule"],
        },
      },
    });

    expect(view).toMatchObject({
      session_id: "session-1",
      revision: 7,
      phase: "diagnose",
      durability: "durable",
      goal: { title: "Chain rule" },
      pending: {
        key: "probe-1",
        skill_label: "Using the power rule",
        can_hint: false,
      },
      progress: {
        label: "Confirming the target",
        probes_used: 2,
        probe_budget: 5,
        percent: 40,
      },
    });
    expect(view.transcript[0].text).toBe("Differentiate x^2 ___");
    expect(view.transcript[0].prompt_segments).toHaveLength(3);
    expect(view.learner_summary.confirmed_strengths).toEqual(["Exponent Rules"]);
    expect(view.learner_summary.uncertain).toEqual(["Power Rule"]);
    expect(pendingAcceptsText(view.pending)).toBe(true);
  });

  it("preserves safe choice options without an expected-answer field", () => {
    const view = normalizeSessionView({
      session_id: "choice-session",
      pending: {
        key: "choice-1",
        kind: "probe",
        kc_id: "kc.alg.solve_linear",
        input_mode: "choice",
        prompt: "Choose the equivalent form.",
        choice_options: ["option-a", "option-b"],
        hint: {
          available: true,
          next_index: 2,
          total: 3,
          next_reveals_answer: true,
        },
      },
    });

    expect(view.pending.choice_options).toEqual(["option-a", "option-b"]);
    expect(view.pending.hint).toEqual({
      available: true,
      next_index: 2,
      total: 3,
      next_reveals_answer: true,
    });
    expect(view.pending).not.toHaveProperty("expected_choice_id");
  });

  it("recognizes a widget pending item and suppresses the unrelated text answer", () => {
    const view = normalizeSessionView({
      session_id: "session-2",
      phase: "teach",
      pending: {
        key: "widget-1",
        kind: "widget",
        input_mode: "widget",
        widget: {
          widget_type: "mapping_v1",
          left: ["x"],
          right: ["1"],
        },
        widget_state: {
          rows: [{ id: "x", value: "1" }],
        },
      },
    });

    expect(isWidgetPending(view.pending)).toBe(true);
    expect(pendingAcceptsText(view.pending)).toBe(false);
    expect(view.pending.widget_state).toEqual({
      rows: [{ id: "x", value: "1" }],
    });
  });

  it("retains every safe guided-practice snapshot in transcript order", () => {
    const widget = {
      widget_type: "slider_v1",
      interaction_version: "slider_v1",
      presentation: {
        prompt: "Choose a value.",
        label: "Value",
        help_text: "Use the arrow keys.",
        minimum: -2,
        maximum: 2,
        step: 1,
        initial_value: 0,
        value_label: "Selected value",
      },
    };
    const view = normalizeSessionView({
      session_id: "widget-history",
      transcript: [
        {
          sequence: 4,
          role: "student",
          kind: "widget_attempt",
          interaction_key: "guided-1",
          text: "Guided-practice attempt 1 submitted.",
          widget,
          widget_state: { value: -1 },
          widget_status: "attempted",
          widget_attempt_number: 1,
        },
        {
          sequence: 6,
          role: "student",
          kind: "widget_attempt",
          interaction_key: "guided-1",
          text: "Guided-practice attempt 2 submitted.",
          widget,
          widget_state: { value: 1 },
          widget_status: "solved",
          widget_attempt_number: 2,
        },
      ],
    });

    expect(view.transcript.map((entry) => entry.widget_attempt_number)).toEqual([
      1,
      2,
    ]);
    expect(view.transcript.map((entry) => entry.widget_state)).toEqual([
      { value: -1 },
      { value: 1 },
    ]);
    expect(view.transcript.every((entry) => entry.widget === widget)).toBe(true);
  });

  it("normalizes the typed text contract and pinned release identity", () => {
    const view = normalizeSessionView({
      session_id: "typed-text",
      release_id: "release.product-rule.v1",
      release_digest: "a".repeat(64),
      pending: {
        key: "probe-typed",
        kind: "probe",
        kc_id: "kc.der.power_rule",
        input: {
          type: "text",
          answer_kind: "finite_set",
          label: "Your set",
          placeholder: "For example: {-2, 2}",
          help_text: "Put distinct values inside braces.",
          max_length: 128,
        },
      },
    });

    expect(view.release_id).toBe("release.product-rule.v1");
    expect(view.release_digest).toBe("a".repeat(64));
    expect(view.pending).toMatchObject({
      input_mode: "text",
      answer_kind: "finite_set",
      label: "Your set",
      placeholder: "For example: {-2, 2}",
      help_text: "Put distinct values inside braces.",
      max_length: 128,
    });
  });

  it("bounds a malformed legacy maximum length before rendering", () => {
    const oversized = normalizeSessionView({
      session_id: "legacy-max",
      pending: {
        key: "legacy-max-pending",
        input: {
          type: "legacy_text",
          label: "Your answer",
          placeholder: "Type it",
          help_text: "Keep it short.",
          max_length: 10000,
        },
      },
    });
    const nonnumeric = normalizeSessionView({
      session_id: "legacy-invalid-max",
      pending: {
        key: "legacy-invalid-max-pending",
        input: {
          type: "legacy_text",
          max_length: "not-a-number",
        },
      },
    });

    expect(oversized.pending.max_length).toBe(256);
    expect(nonnumeric.pending.max_length).toBe(256);
  });

  it("derives resumable mapping and slider controls from only typed public fields", () => {
    const mapping = normalizeSessionView({
      session_id: "typed-mapping",
      pending: {
        key: "mapping-1",
        kind: "guided_widget",
        input: {
          type: "mapping_v1",
          prompt: "Match each expression.",
          rows: [
            {
              entry_id: "row.a",
              label: "x^2",
              spoken_text: "x squared",
              selected_option_id: "option.a",
            },
            {
              entry_id: "row.b",
              label: "x^3",
              spoken_text: "x cubed",
              selected_option_id: null,
            },
          ],
          options: [
            {
              entry_id: "option.a",
              label: "2*x",
              spoken_text: "two x",
            },
            {
              entry_id: "option.b",
              label: "3*x^2",
              spoken_text: "three x squared",
            },
          ],
        },
      },
    });
    const slider = normalizeSessionView({
      session_id: "typed-slider",
      pending: {
        key: "slider-1",
        kind: "guided_widget",
        input: {
          type: "slider_v1",
          prompt: "Choose the exponent.",
          label: "Exponent",
          help_text: "Use the arrow keys.",
          minimum: 0,
          maximum: 5,
          step: 1,
          initial_value: 0,
          current_value: 2,
          value_label: "Selected exponent",
          result_template: "Exponent: {value}",
        },
      },
    });

    expect(mapping.pending.input_mode).toBe("widget");
    expect(mapping.pending.widget_state).toEqual({
      rows: [
        { id: "row.a", value: "option.a" },
        { id: "row.b", value: "" },
      ],
    });
    expect(mapping.pending.widget).not.toHaveProperty("scoring");
    expect(mapping.pending.widget).not.toHaveProperty("correct_pairs");
    expect(slider.pending.widget_state).toEqual({ value: 2 });
    expect(slider.pending.widget.presentation).not.toHaveProperty("target");
    expect(pendingAcceptsText(slider.pending)).toBe(false);
  });

  it("treats an explicit text delivery as text even for guided-widget evidence", () => {
    const view = normalizeSessionView({
      session_id: "session-text-guided",
      phase: "teach",
      pending: {
        key: "guided-1",
        kind: "guided_widget",
        input_mode: "math",
        prompt: "Differentiate x^3.",
      },
    });

    expect(isWidgetPending(view.pending)).toBe(false);
    expect(pendingAcceptsText(view.pending)).toBe(true);
  });

  it("turns policy identifiers into phase-appropriate student progress labels", () => {
    const teaching = normalizeSessionView({
      session_id: "session-teach",
      phase: "teach",
      progress: {
        plan_step: "verify_uncertain",
        diagnosis_probes_used: 4,
        diagnosis_probe_budget: 5,
      },
    });
    const diagnosis = normalizeSessionView({
      session_id: "session-diagnose",
      phase: "diagnose",
      progress: {
        diagnosis_probes_used: 2,
        diagnosis_probe_budget: 5,
      },
    });

    expect(teaching.progress).toMatchObject({
      label: "Checking an uncertain skill",
      percent: 0,
      bar_kind: null,
    });
    expect(diagnosis.progress).toMatchObject({
      label: "Checking what you already know",
      percent: 40,
      bar_kind: "diagnosis",
      bar_label: "Diagnosis budget used",
    });
  });

  it("preserves a retry draft only for the same text interaction", () => {
    const prior = normalizeSessionView({
      session_id: "session-retry",
      pending: {
        key: "check-1",
        kind: "checkin",
        input_mode: "math",
      },
    });
    const samePending = normalizeSessionView({
      session_id: "session-retry",
      revision: 2,
      pending: {
        key: "check-1",
        kind: "checkin",
        input_mode: "math",
      },
    });
    const advanced = normalizeSessionView({
      session_id: "session-retry",
      revision: 3,
      pending: {
        key: "check-2",
        kind: "checkin",
        input_mode: "math",
      },
    });
    const widget = normalizeSessionView({
      session_id: "session-retry",
      revision: 2,
      pending: {
        key: "check-1",
        kind: "widget",
        input_mode: "widget",
        widget: { widget_type: "mapping_v1" },
      },
    });

    expect(canPreserveAnswerDraft(prior, samePending)).toBe(true);
    expect(canPreserveAnswerDraft(prior, advanced)).toBe(false);
    expect(canPreserveAnswerDraft(prior, widget)).toBe(false);
  });

  it("keeps only released goals returned by the server", () => {
    expect(
      normalizeGoals({
        goals: [
          {
            goal_id: "ready",
            target_kc: "kc.int.ftc",
            title: "FTC",
            available: true,
          },
          {
            goal_id: "hidden",
            target_kc: "kc.int.u_substitution",
            available: false,
          },
        ],
      }),
    ).toEqual([
      {
        id: "ready",
        target_kc: "kc.int.ftc",
        title: "FTC",
        description: "",
        available: true,
      },
    ]);
  });

  it("preserves an explicitly empty reviewed catalog", () => {
    expect(normalizeGoals({ catalog_version: 1, goals: [] })).toEqual([]);
  });

  it("keeps rollout admission separate from content readiness", () => {
    const catalog = normalizeGoalCatalog({
      catalog_version: 1,
      goals: [],
      rollout: {
        status: "not_selected",
        reason: "This browser is outside the current 5% cohort.",
        percentage: 5,
      },
    });

    expect(catalog).toEqual({
      goals: [],
      rollout: {
        status: "not_selected",
        reason: "This browser is outside the current 5% cohort.",
        percentage: 5,
      },
    });
    expect(catalogEmptyMessage(catalog.rollout)).toEqual({
      title: "Pilot access is expanding",
      body: "This browser is outside the current 5% cohort.",
    });
  });

  it("uses distinct intake messages for a pause and incomplete content", () => {
    expect(
      catalogEmptyMessage({
        status: "paused",
        reason: "New starts are paused.",
        percentage: 25,
      }),
    ).toEqual({
      title: "New sessions are temporarily paused",
      body: "New starts are paused.",
    });
    expect(
      catalogEmptyMessage({
        status: "content_unavailable",
        reason: "Reviewed coverage is incomplete.",
        percentage: 100,
      }),
    ).toEqual({
      title: "Reviewed goals are being prepared",
      body: "Reviewed coverage is incomplete.",
    });
  });
});
