import { describe, expect, it, vi } from "vitest";

import { ApiError } from "../src/api.js";
import {
  clearPendingRecovery,
  MutationCoordinator,
  PENDING_RECOVERY_KEY,
  readPendingRecovery,
} from "../src/mutations.js";

function memoryStorage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

describe("mutation coordinator", () => {
  it("reads the authoritative snapshot from the API error contract", () => {
    const session = { session_id: "session-1", revision: 3 };
    const error = new ApiError("stale", {
      status: 409,
      code: "stale_interaction",
      payload: { session },
    });

    expect(error.view).toBe(session);
    expect(error.retryable).toBe(false);
  });

  it("keeps one mutation in flight", async () => {
    let release;
    const pending = new Promise((resolve) => {
      release = resolve;
    });
    const coordinator = new MutationCoordinator(() => pending, () => "request-1");

    const first = coordinator.execute({ type: "answer", answer: "2x" });
    await expect(
      coordinator.execute({ type: "request_hint" }),
    ).rejects.toThrow(/already being checked/);
    release({ ok: true });
    await expect(first).resolves.toEqual({ ok: true });
  });

  it("reuses a stable request id after retryable failures", async () => {
    const send = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError("temporary", { status: 503, code: "commit_failed" }),
      )
      .mockResolvedValueOnce({ revision: 2 });
    const requestId = vi.fn().mockReturnValueOnce("request-a");
    const coordinator = new MutationCoordinator(send, requestId);
    const action = { type: "answer", answer: "x^2" };

    await expect(coordinator.execute(action)).rejects.toThrow("temporary");
    await expect(coordinator.execute(action)).resolves.toEqual({ revision: 2 });

    expect(send).toHaveBeenCalledTimes(2);
    expect(send.mock.calls[0][0].request_id).toBe("request-a");
    expect(send.mock.calls[1][0].request_id).toBe("request-a");
    expect(requestId).toHaveBeenCalledTimes(1);
  });

  it("does not reuse ids after a semantic conflict", async () => {
    const send = vi
      .fn()
      .mockRejectedValueOnce(
        new ApiError("stale", { status: 409, code: "stale_interaction" }),
      )
      .mockResolvedValueOnce({ revision: 3 });
    const requestId = vi
      .fn()
      .mockReturnValueOnce("request-a")
      .mockReturnValueOnce("request-b");
    const coordinator = new MutationCoordinator(send, requestId);
    const action = { type: "answer", answer: "x" };

    await expect(coordinator.execute(action)).rejects.toThrow("stale");
    await coordinator.execute(action);

    expect(send.mock.calls[0][0].request_id).toBe("request-a");
    expect(send.mock.calls[1][0].request_id).toBe("request-b");
  });

  it("persists only a create recovery proof across a retryable response loss", async () => {
    const storage = memoryStorage();
    const requestId = "f147b21b-1630-4eb0-8f1a-17dc0900acb3";
    const secretContext = "private coaching context";
    const send = vi.fn().mockRejectedValue(
      new ApiError("connection lost", { status: 0, code: "network_error" }),
    );
    const coordinator = new MutationCoordinator(send, () => requestId, {
      recoveryOperation: "create",
      recoveryStorage: storage,
    });

    await expect(
      coordinator.execute({ goal_id: "goal.one", context: secretContext }),
    ).rejects.toThrow("connection lost");

    const raw = storage.getItem(PENDING_RECOVERY_KEY);
    expect(JSON.parse(raw)).toEqual({
      schema_version: 1,
      operation: "create",
      request_id: requestId,
    });
    expect(raw).not.toContain(secretContext);
    expect(readPendingRecovery(storage)).toEqual({
      schema_version: 1,
      operation: "create",
      request_id: requestId,
    });
  });

  it("clears a matching recovery proof only after a confirmed response", async () => {
    const storage = memoryStorage();
    const requestId = "7d384134-50a4-4a74-8102-d330a6e602f7";
    const send = vi.fn().mockResolvedValue({ session_id: "session-2" });
    const coordinator = new MutationCoordinator(send, () => requestId, {
      recoveryOperation: "reset",
      recoveryStorage: storage,
    });

    await coordinator.execute({ expected_revision: 2, pending_key: "item-2" });

    expect(readPendingRecovery(storage)).toBeNull();
  });

  it("retains recovery proof when failure has no confirmed HTTP response", async () => {
    const storage = memoryStorage();
    const requestId = "340aed3e-e041-4166-ae1b-d02784c262dd";
    const coordinator = new MutationCoordinator(
      vi.fn().mockRejectedValue(new Error("ambiguous local failure")),
      () => requestId,
      { recoveryOperation: "reset", recoveryStorage: storage },
    );

    await expect(
      coordinator.execute({ expected_revision: 1, pending_key: "item-1" }),
    ).rejects.toThrow("ambiguous local failure");

    expect(readPendingRecovery(storage)?.request_id).toBe(requestId);
  });

  it("does not rotate a token when recovery storage is unavailable", async () => {
    const send = vi.fn();
    const coordinator = new MutationCoordinator(
      send,
      () => "ec9b69e8-b001-4c5f-97fd-e5eb7520ba67",
      { recoveryOperation: "create", recoveryStorage: null },
    );

    await expect(coordinator.execute({ goal_id: "goal.one" })).rejects.toThrow(
      /cannot safely preserve session recovery state/,
    );
    expect(send).not.toHaveBeenCalled();
  });

  it("lets bootstrap clear only the exact recovered proof", () => {
    const storage = memoryStorage();
    const requestId = "4f7261bf-5ce4-44e5-8f83-da783e37b6cc";
    storage.setItem(
      PENDING_RECOVERY_KEY,
      JSON.stringify({
        schema_version: 1,
        operation: "reset",
        request_id: requestId,
      }),
    );

    clearPendingRecovery(
      { operation: "create", request_id: requestId },
      storage,
    );
    expect(readPendingRecovery(storage)?.operation).toBe("reset");

    clearPendingRecovery(
      { operation: "reset", request_id: requestId },
      storage,
    );
    expect(readPendingRecovery(storage)).toBeNull();
  });
});
