import { describe, expect, it } from "vitest";

import {
  promptSegmentAnnouncement,
  promptSegmentsAnnouncement,
  transcriptEntryAnnouncement,
} from "../src/accessibility.js";

describe("screen-reader transcript announcements", () => {
  it("uses reviewed speech for math, tables, and plots", () => {
    expect(
      promptSegmentsAnnouncement([
        { kind: "text", text: "Differentiate" },
        { kind: "math", expression: "x^2", spoken_text: "x squared" },
        { kind: "blank", label: "Derivative" },
        {
          kind: "table",
          caption: "Values",
          spoken_text: "The values increase by two each row.",
        },
        {
          kind: "plot",
          title: "Line graph",
          spoken_text: "A line rises from zero to four.",
        },
      ]),
    ).toBe(
      "Differentiate x squared Derivative The values increase by two each row. " +
        "A line rises from zero to four.",
    );
  });

  it("falls back safely for legacy math and labels an unlabeled blank", () => {
    expect(promptSegmentAnnouncement({ kind: "math", expression: "2*x" })).toBe(
      "2*x",
    );
    expect(promptSegmentAnnouncement({ kind: "blank" })).toBe("Answer blank");
  });

  it("announces structured blocks instead of an empty plain-text fallback", () => {
    const announcement = transcriptEntryAnnouncement({
      role: "tutor",
      text: "",
      content_blocks: [
        {
          kind: "worked_example",
          text: "Apply the rule one step at a time.",
          segments: [
            {
              kind: "math",
              expression: "2*x",
              spoken_text: "two x",
            },
          ],
        },
      ],
    });

    expect(announcement).toBe(
      "Tutor update: Worked example. Apply the rule one step at a time. two x",
    );
  });

  it("labels learner responses and ignores empty entries", () => {
    expect(
      transcriptEntryAnnouncement({ role: "student", text: "x^2" }),
    ).toBe("Your response: x^2");
    expect(transcriptEntryAnnouncement({ role: "tutor", text: "  " })).toBe("");
  });
});
