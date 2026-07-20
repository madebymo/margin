import { newRequestId } from "./api.js";

export const PENDING_RECOVERY_KEY = "tutor.v2.pending-recovery.v1";
const RECOVERY_OPERATIONS = new Set(["create", "reset"]);
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function availableStorage(storage) {
  if (storage !== undefined) return storage;
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

export function readPendingRecovery(storage) {
  const target = availableStorage(storage);
  if (!target) return null;
  try {
    const value = JSON.parse(target.getItem(PENDING_RECOVERY_KEY));
    if (
      value?.schema_version !== 1 ||
      !RECOVERY_OPERATIONS.has(value.operation) ||
      typeof value.request_id !== "string" ||
      !UUID_PATTERN.test(value.request_id)
    ) {
      target.removeItem(PENDING_RECOVERY_KEY);
      return null;
    }
    return Object.freeze({
      schema_version: 1,
      operation: value.operation,
      request_id: value.request_id,
    });
  } catch {
    try {
      target.removeItem(PENDING_RECOVERY_KEY);
    } catch {
      // Storage may be disabled. Mutation delivery still works in memory.
    }
    return null;
  }
}

function writePendingRecovery(recovery, storage) {
  const target = availableStorage(storage);
  if (!target) return false;
  try {
    target.setItem(PENDING_RECOVERY_KEY, JSON.stringify(recovery));
    return true;
  } catch {
    return false;
  }
}

export function clearPendingRecovery(expected = null, storage) {
  const target = availableStorage(storage);
  if (!target) return;
  try {
    const current = readPendingRecovery(target);
    if (
      expected &&
      current &&
      (current.operation !== expected.operation ||
        current.request_id !== expected.request_id)
    ) {
      return;
    }
    target.removeItem(PENDING_RECOVERY_KEY);
  } catch {
    // Storage cleanup is best effort and contains no student-authored text.
  }
}

function stable(value) {
  if (Array.isArray(value)) {
    return `[${value.map(stable).join(",")}]`;
  }
  if (value && typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${stable(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

export class MutationCoordinator {
  constructor(send, requestId = newRequestId, options = {}) {
    this.send = send;
    this.requestId = requestId;
    this.active = false;
    this.retry = null;
    this.recoveryOperation = options.recoveryOperation ?? null;
    this.recoveryStorage = options.recoveryStorage;
    if (
      this.recoveryOperation !== null &&
      !RECOVERY_OPERATIONS.has(this.recoveryOperation)
    ) {
      throw new Error("Unknown recovery operation.");
    }
  }

  async execute(action) {
    if (this.active) {
      throw new Error("Another response is already being checked.");
    }
    const fingerprint = stable(action);
    const requestId =
      this.retry?.fingerprint === fingerprint ? this.retry.requestId : this.requestId();
    const recovery = this.recoveryOperation
      ? {
          schema_version: 1,
          operation: this.recoveryOperation,
          request_id: requestId,
        }
      : null;
    const pendingRecovery = recovery
      ? readPendingRecovery(this.recoveryStorage)
      : null;
    if (
      pendingRecovery &&
      (pendingRecovery.operation !== recovery.operation ||
        pendingRecovery.request_id !== recovery.request_id)
    ) {
      throw new Error(
        "A previous session change still needs recovery. Reload this page before trying again.",
      );
    }
    if (recovery && !writePendingRecovery(recovery, this.recoveryStorage)) {
      throw new Error(
        "This browser cannot safely preserve session recovery state. Enable session storage and retry.",
      );
    }
    this.active = true;
    try {
      const result = await this.send({ ...action, request_id: requestId });
      this.retry = null;
      if (recovery) clearPendingRecovery(recovery, this.recoveryStorage);
      return result;
    } catch (error) {
      this.retry = error?.retryable ? { fingerprint, requestId } : null;
      const definitiveClientResponse =
        Number.isInteger(error?.status) &&
        error.status >= 400 &&
        error.status < 500 &&
        !error.retryable;
      if (recovery && definitiveClientResponse) {
        clearPendingRecovery(recovery, this.recoveryStorage);
      }
      throw error;
    } finally {
      this.active = false;
    }
  }

  clearRetry() {
    this.retry = null;
  }
}
