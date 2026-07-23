"use strict";

export class ApiError extends Error {
  constructor(message, { status = 0, code = "request_failed", payload = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.payload = payload;
    this.view = payload?.view ?? payload?.session_view ?? payload?.session ?? null;
    this.retryable = status === 0 || status === 429 || status >= 500;
  }
}

function errorMessage(payload, response) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (
    detail &&
    typeof detail.message === "string" &&
    detail.message.trim()
  ) {
    return detail.message;
  }
  if (typeof payload?.message === "string" && payload.message.trim()) {
    return payload.message;
  }
  return response.statusText || "Request failed";
}

export async function request(path, { method = "GET", body, signal } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      credentials: "same-origin",
      headers: {
        Accept: "application/json",
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      throw error;
    }
    throw new ApiError("The tutor could not be reached. Check your connection and retry.", {
      status: 0,
      code: "network_error",
    });
  }

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const code =
      payload?.code ??
      payload?.error_code ??
      (typeof payload?.detail === "object" ? payload.detail.code : null) ??
      `http_${response.status}`;
    throw new ApiError(errorMessage(payload, response), {
      status: response.status,
      code,
      payload,
    });
  }
  return payload;
}

// Kept for the legacy endpoints while existing v1 sessions finish.
export function api(path, body, options = {}) {
  return request(path, {
    method: body === undefined ? "GET" : "POST",
    body,
    signal: options.signal,
  });
}

export const apiV2 = Object.freeze({
  goals(signal) {
    return request("/api/v2/goals", { signal });
  },
  capabilities(signal) {
    return request("/api/v2/capabilities", { signal });
  },
  current(signal) {
    return request("/api/v2/sessions/current", { signal });
  },
  create(body, signal) {
    return request("/api/v2/sessions", { method: "POST", body, signal });
  },
  recover(body, signal) {
    return request("/api/v2/sessions/recover", { method: "POST", body, signal });
  },
  action(sessionId, body, signal) {
    return request(`/api/v2/sessions/${encodeURIComponent(sessionId)}/actions`, {
      method: "POST",
      body,
      signal,
    });
  },
  reset(body, signal) {
    return request("/api/v2/sessions/current/reset", {
      method: "POST",
      body,
      signal,
    });
  },
});

export function newRequestId() {
  const secureCrypto = globalThis.crypto;
  if (secureCrypto?.randomUUID) {
    return secureCrypto.randomUUID();
  }
  if (!secureCrypto?.getRandomValues) {
    throw new Error("Secure randomness is required to start or change a session.");
  }
  const bytes = secureCrypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((value) => value.toString(16).padStart(2, "0"));
  return [
    hex.slice(0, 4).join(""),
    hex.slice(4, 6).join(""),
    hex.slice(6, 8).join(""),
    hex.slice(8, 10).join(""),
    hex.slice(10, 16).join(""),
  ].join("-");
}
