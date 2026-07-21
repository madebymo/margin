const DRAFT_STORAGE_KEY = "tutor.v2.answer-draft.v1";
const DRAFT_SCHEMA_VERSION = 1;
const MAX_DRAFT_LENGTH = 256;
const WIDGET_DRAFT_STORAGE_KEY = "tutor.v2.widget-draft.v1";
const WIDGET_DRAFT_SCHEMA_VERSION = 1;
const MAX_WIDGET_ROWS = 12;

function resolveStorage(storage) {
  if (storage !== undefined) return storage;
  try {
    return globalThis.sessionStorage ?? null;
  } catch {
    return null;
  }
}

function validScope(scope) {
  return Boolean(
    scope &&
      typeof scope.sessionId === "string" &&
      scope.sessionId.length > 0 &&
      typeof scope.pendingKey === "string" &&
      scope.pendingKey.length > 0,
  );
}

function parseStoredDraft(storage) {
  const target = resolveStorage(storage);
  if (!target) return { target: null, draft: null };
  try {
    const draft = JSON.parse(target.getItem(DRAFT_STORAGE_KEY));
    const valid =
      draft?.schema_version === DRAFT_SCHEMA_VERSION &&
      typeof draft.session_id === "string" &&
      draft.session_id.length > 0 &&
      typeof draft.pending_key === "string" &&
      draft.pending_key.length > 0 &&
      typeof draft.value === "string" &&
      draft.value.length > 0 &&
      draft.value.length <= MAX_DRAFT_LENGTH;
    if (!valid) {
      target.removeItem(DRAFT_STORAGE_KEY);
      return { target, draft: null };
    }
    return { target, draft };
  } catch {
    try {
      target.removeItem(DRAFT_STORAGE_KEY);
    } catch {
      // Storage may become unavailable between reads; draft recovery is optional.
    }
    return { target, draft: null };
  }
}

export function readAnswerDraft(scope, storage) {
  if (!validScope(scope)) return "";
  const { target, draft } = parseStoredDraft(storage);
  if (!draft) return "";
  if (
    draft.session_id !== scope.sessionId ||
    draft.pending_key !== scope.pendingKey
  ) {
    try {
      target?.removeItem(DRAFT_STORAGE_KEY);
    } catch {
      // A stale draft can be ignored even when storage cleanup is unavailable.
    }
    return "";
  }
  return draft.value;
}

export function writeAnswerDraft(scope, value, storage) {
  const target = resolveStorage(storage);
  if (!target || !validScope(scope)) return false;
  if (typeof value !== "string" || value.length === 0) {
    clearAnswerDraft(scope, target);
    return true;
  }
  if (value.length > MAX_DRAFT_LENGTH) return false;
  try {
    target.setItem(
      DRAFT_STORAGE_KEY,
      JSON.stringify({
        schema_version: DRAFT_SCHEMA_VERSION,
        session_id: scope.sessionId,
        pending_key: scope.pendingKey,
        value,
      }),
    );
    return true;
  } catch {
    return false;
  }
}

export function clearAnswerDraft(scope = null, storage) {
  const target = resolveStorage(storage);
  if (!target) return false;
  try {
    if (scope && validScope(scope)) {
      const { draft } = parseStoredDraft(target);
      if (
        draft &&
        (draft.session_id !== scope.sessionId ||
          draft.pending_key !== scope.pendingKey)
      ) {
        return false;
      }
    }
    target.removeItem(DRAFT_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

export function answerDraftScope(view) {
  const sessionId = view?.session_id;
  const pendingKey = view?.pending?.key;
  if (!sessionId || !pendingKey) return null;
  return { sessionId: String(sessionId), pendingKey: String(pendingKey) };
}

function normalizeWidgetDraftState(state) {
  if (!state || typeof state !== "object" || Array.isArray(state)) return null;
  if (Object.hasOwn(state, "value")) {
    const value = Number(state.value);
    return Number.isFinite(value) ? { value } : null;
  }
  if (!Array.isArray(state.rows) || state.rows.length > MAX_WIDGET_ROWS) {
    return null;
  }
  const rows = [];
  const ids = new Set();
  for (const row of state.rows) {
    const id = typeof row?.id === "string" ? row.id : "";
    const value = typeof row?.value === "string" ? row.value : "";
    if (
      !id ||
      id.length > 64 ||
      value.length > 64 ||
      ids.has(id)
    ) {
      return null;
    }
    ids.add(id);
    rows.push({ id, value });
  }
  return rows.length >= 2 ? { rows } : null;
}

function parseStoredWidgetDraft(storage) {
  const target = resolveStorage(storage);
  if (!target) return { target: null, draft: null };
  try {
    const draft = JSON.parse(target.getItem(WIDGET_DRAFT_STORAGE_KEY));
    const state = normalizeWidgetDraftState(draft?.state);
    const valid =
      draft?.schema_version === WIDGET_DRAFT_SCHEMA_VERSION &&
      typeof draft.session_id === "string" &&
      draft.session_id.length > 0 &&
      typeof draft.pending_key === "string" &&
      draft.pending_key.length > 0 &&
      state !== null;
    if (!valid) {
      target.removeItem(WIDGET_DRAFT_STORAGE_KEY);
      return { target, draft: null };
    }
    return { target, draft: { ...draft, state } };
  } catch {
    try {
      target.removeItem(WIDGET_DRAFT_STORAGE_KEY);
    } catch {
      // Storage may become unavailable between reads; recovery is optional.
    }
    return { target, draft: null };
  }
}

export function readWidgetDraft(scope, storage) {
  if (!validScope(scope)) return null;
  const { target, draft } = parseStoredWidgetDraft(storage);
  if (!draft) return null;
  if (
    draft.session_id !== scope.sessionId ||
    draft.pending_key !== scope.pendingKey
  ) {
    try {
      target?.removeItem(WIDGET_DRAFT_STORAGE_KEY);
    } catch {
      // A stale draft can be ignored even when cleanup is unavailable.
    }
    return null;
  }
  return draft.state;
}

export function writeWidgetDraft(scope, state, storage) {
  const target = resolveStorage(storage);
  const normalized = normalizeWidgetDraftState(state);
  if (!target || !validScope(scope) || normalized === null) return false;
  try {
    target.setItem(
      WIDGET_DRAFT_STORAGE_KEY,
      JSON.stringify({
        schema_version: WIDGET_DRAFT_SCHEMA_VERSION,
        session_id: scope.sessionId,
        pending_key: scope.pendingKey,
        state: normalized,
      }),
    );
    return true;
  } catch {
    return false;
  }
}

export function clearWidgetDraft(scope = null, storage) {
  const target = resolveStorage(storage);
  if (!target) return false;
  try {
    if (scope && validScope(scope)) {
      const { draft } = parseStoredWidgetDraft(target);
      if (
        draft &&
        (draft.session_id !== scope.sessionId ||
          draft.pending_key !== scope.pendingKey)
      ) {
        return false;
      }
    }
    target.removeItem(WIDGET_DRAFT_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}

export const widgetDraftScope = answerDraftScope;
