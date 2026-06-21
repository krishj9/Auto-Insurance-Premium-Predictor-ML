import type { FeatureSchemaResponse, PredictRequest, PredictResponse } from "../types/api";

const API_BASE_URL_FROM_ENV = (
  import.meta as unknown as { env?: { VITE_API_BASE_URL?: string } }
).env?.VITE_API_BASE_URL;

export const BASE_URL = API_BASE_URL_FROM_ENV && API_BASE_URL_FROM_ENV.trim().length > 0
  ? API_BASE_URL_FROM_ENV
  : "http://localhost:8000";

const REQUEST_TIMEOUT_MS = 10_000;
const MAX_RESPONSE_BYTES = 1_000_000;

export async function getFeatureSchema(): Promise<FeatureSchemaResponse> {
  return requestJson<FeatureSchemaResponse>("/feature-schema", {
    method: "GET",
  });
}

export async function predictPremium(payload: PredictRequest): Promise<PredictResponse> {
  return requestJson<PredictResponse>("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(`${BASE_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });

    const contentLengthHeader = response.headers.get("content-length");
    if (contentLengthHeader !== null) {
      const contentLength = Number(contentLengthHeader);
      if (Number.isFinite(contentLength) && contentLength > MAX_RESPONSE_BYTES) {
        throw new Error("Response is too large to process safely.");
      }
    }

    const responseText = await response.text();
    if (responseText.length > MAX_RESPONSE_BYTES) {
      throw new Error("Response is too large to process safely.");
    }

    let parsed: unknown = null;
    if (responseText.length > 0) {
      try {
        parsed = JSON.parse(responseText) as unknown;
      } catch {
        throw new Error("Received non-JSON response from API.");
      }
    }

    if (!response.ok) {
      const message = extractErrorMessage(parsed) ?? `Request failed with status ${response.status}.`;
      throw new Error(message);
    }

    return parsed as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Request timed out. Please try again.");
    }
    if (error instanceof Error) {
      throw error;
    }
    throw new Error("Unexpected API error.");
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function extractErrorMessage(payload: unknown): string | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const record = payload as Record<string, unknown>;
  if (typeof record.detail === "string") {
    return record.detail;
  }

  const error = record.error;
  if (typeof error === "object" && error !== null) {
    const errorRecord = error as Record<string, unknown>;
    if (typeof errorRecord.message === "string") {
      return errorRecord.message;
    }
  }

  return null;
}
