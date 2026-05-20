// API client for the ScholarMind FastAPI backend.

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_API_BASE ||
  "https://web-production-442bd.up.railway.app";

export type Source = {
  index: number;
  paper_id: string;
  page: number | string;
  preview: string;
};

export type Plan = {
  reasoning: string;
  tool: string;
  arguments: Record<string, unknown>;
};

export type Timings = {
  planner_ms: number;
  tool_ms: number;
  synthesis_ms: number;
  total_ms: number;
};

export type ChatResponse = {
  answer: string;
  sources: Source[];
  plan: Plan;
  timings: Timings;
  metadata: Record<string, unknown>;
  trace_id: string | null;
};

export type Info = {
  name: string;
  description: string;
  num_papers: number;
  embedding_model: string;
  llm_model: string;
  tools_available: string[];
};

const RENDER_COLD_START_TIMEOUT_MS = 60_000; // Render free tier can take ~50s to wake

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxAttempts = 3,
): Promise<Response> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        RENDER_COLD_START_TIMEOUT_MS,
      );
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timeoutId);
      return res;
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err));
      const isAbort =
        lastError.name === "AbortError" ||
        lastError.message.includes("aborted");
      // Only retry on network / timeout errors, not on HTTP error responses
      if (!isAbort && !lastError.message.includes("fetch")) break;
      if (attempt < maxAttempts) {
        // Exponential back-off: 2s, 4s
        await new Promise((r) => setTimeout(r, 2 ** attempt * 1000));
      }
    }
  }
  throw lastError ?? new Error("Request failed");
}

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetchWithRetry(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const errBody = await res.json();
      detail = errBody.detail || JSON.stringify(errBody);
    } catch {
      // ignore json parse errors on error responses
    }
    throw new Error(`API ${res.status}: ${detail}`);
  }

  return res.json();
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetchWithRetry(`${API_BASE}${path}`, { method: "GET" });
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

/** Ask the agent a question — returns cited answer + plan + sources. */
export async function askAgent(query: string): Promise<ChatResponse> {
  return postJSON<ChatResponse>("/chat", { query });
}

/** Get backend metadata. */
export async function getInfo(): Promise<Info> {
  return getJSON<Info>("/info");
}