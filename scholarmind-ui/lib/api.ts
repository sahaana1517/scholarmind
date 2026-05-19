// API client for the ScholarMind FastAPI backend.

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

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

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const errBody = await res.json();
      detail = errBody.detail || JSON.stringify(errBody);
    } catch {}
    throw new Error(`API ${res.status}: ${detail}`);
  }

  return res.json();
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
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