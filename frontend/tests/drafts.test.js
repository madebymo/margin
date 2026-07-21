import { describe, expect, it } from "vitest";

import {
  answerDraftScope,
  clearAnswerDraft,
  clearWidgetDraft,
  readAnswerDraft,
  readWidgetDraft,
  widgetDraftScope,
  writeAnswerDraft,
  writeWidgetDraft,
} from "../src/drafts.js";

function memoryStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.get(key) ?? null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
    removeItem(key) {
      values.delete(key);
    },
  };
}

const scope = Object.freeze({ sessionId: "session-1", pendingKey: "check-1" });

describe("pending answer draft recovery", () => {
  it("restores a draft only for the exact session and pending interaction", () => {
    const storage = memoryStorage();
    expect(writeAnswerDraft(scope, "2*x", storage)).toBe(true);
    expect(readAnswerDraft(scope, storage)).toBe("2*x");

    expect(
      readAnswerDraft(
        { sessionId: "session-1", pendingKey: "check-2" },
        storage,
      ),
    ).toBe("");
    expect(readAnswerDraft(scope, storage)).toBe("");
  });

  it("does not let one interaction clear a newer interaction's draft", () => {
    const storage = memoryStorage();
    const newer = { sessionId: "session-1", pendingKey: "check-2" };
    writeAnswerDraft(newer, "x^2", storage);

    expect(clearAnswerDraft(scope, storage)).toBe(false);
    expect(readAnswerDraft(newer, storage)).toBe("x^2");
  });

  it("rejects oversized and malformed records without throwing", () => {
    const storage = memoryStorage();
    expect(writeAnswerDraft(scope, "x".repeat(257), storage)).toBe(false);
    expect(readAnswerDraft(scope, storage)).toBe("");

    storage.setItem("tutor.v2.answer-draft.v1", "not-json");
    expect(readAnswerDraft(scope, storage)).toBe("");
  });

  it("clears an empty draft and tolerates unavailable storage", () => {
    const storage = memoryStorage();
    writeAnswerDraft(scope, "x", storage);
    expect(writeAnswerDraft(scope, "", storage)).toBe(true);
    expect(readAnswerDraft(scope, storage)).toBe("");

    const unavailable = {
      getItem() {
        throw new Error("blocked");
      },
      removeItem() {
        throw new Error("blocked");
      },
      setItem() {
        throw new Error("blocked");
      },
    };
    expect(readAnswerDraft(scope, unavailable)).toBe("");
    expect(writeAnswerDraft(scope, "x", unavailable)).toBe(false);
    expect(clearAnswerDraft(scope, unavailable)).toBe(false);
  });

  it("derives an exact storage scope only from active session views", () => {
    expect(
      answerDraftScope({
        session_id: "session-1",
        pending: { key: "check-1" },
      }),
    ).toEqual(scope);
    expect(answerDraftScope({ session_id: "session-1", pending: null })).toBeNull();
  });
});

describe("pending guided-practice draft recovery", () => {
  it("round-trips bounded slider and mapping state for the exact interaction", () => {
    const storage = memoryStorage();
    expect(writeWidgetDraft(scope, { value: 3 }, storage)).toBe(true);
    expect(readWidgetDraft(scope, storage)).toEqual({ value: 3 });

    const mapping = {
      rows: [
        { id: "row.a", value: "option.b" },
        { id: "row.b", value: "" },
      ],
    };
    expect(writeWidgetDraft(scope, mapping, storage)).toBe(true);
    expect(readWidgetDraft(scope, storage)).toEqual(mapping);
  });

  it("drops stale, malformed, and oversized widget state without throwing", () => {
    const storage = memoryStorage();
    writeWidgetDraft(scope, { value: 2 }, storage);
    expect(
      readWidgetDraft(
        { sessionId: "session-1", pendingKey: "guided-2" },
        storage,
      ),
    ).toBeNull();
    expect(readWidgetDraft(scope, storage)).toBeNull();

    expect(writeWidgetDraft(scope, { value: Number.NaN }, storage)).toBe(false);
    expect(
      writeWidgetDraft(
        scope,
        {
          rows: Array.from({ length: 13 }, (_, index) => ({
            id: `row.${index}`,
            value: "",
          })),
        },
        storage,
      ),
    ).toBe(false);
    storage.setItem("tutor.v2.widget-draft.v1", "not-json");
    expect(readWidgetDraft(scope, storage)).toBeNull();
  });

  it("does not let an older interaction clear a newer widget draft", () => {
    const storage = memoryStorage();
    const newer = { sessionId: "session-1", pendingKey: "guided-2" };
    writeWidgetDraft(newer, { value: 4 }, storage);

    expect(clearWidgetDraft(scope, storage)).toBe(false);
    expect(readWidgetDraft(newer, storage)).toEqual({ value: 4 });
    expect(widgetDraftScope({
      session_id: "session-1",
      pending: { key: "guided-2" },
    })).toEqual(newer);
  });
});
