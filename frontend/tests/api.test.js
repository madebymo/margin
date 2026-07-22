import { afterEach, describe, expect, it, vi } from "vitest";

import { newRequestId, request } from "../src/api.js";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("API error messages", () => {
  it("falls back when a nested error message is blank", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: vi.fn().mockResolvedValue({
          detail: { message: "   " },
        }),
      }),
    );

    await expect(request("/test")).rejects.toMatchObject({
      name: "ApiError",
      message: "Bad Request",
      status: 400,
      code: "http_400",
    });
  });
});

describe("request id generation", () => {
  it("uses the platform UUID generator when available", () => {
    const randomUUID = vi
      .fn()
      .mockReturnValue("239917d8-6390-444f-b274-2c6f450f3e88");
    vi.stubGlobal("crypto", { randomUUID });

    expect(newRequestId()).toBe("239917d8-6390-444f-b274-2c6f450f3e88");
    expect(randomUUID).toHaveBeenCalledOnce();
  });

  it("builds an RFC 4122 UUIDv4 with getRandomValues", () => {
    vi.stubGlobal("crypto", {
      getRandomValues(bytes) {
        bytes.set([...Array(16).keys()]);
        return bytes;
      },
    });

    expect(newRequestId()).toBe("00010203-0405-4607-8809-0a0b0c0d0e0f");
  });

  it("fails closed when cryptographic randomness is unavailable", () => {
    vi.stubGlobal("crypto", undefined);

    expect(() => newRequestId()).toThrow(/Secure randomness is required/);
  });
});
