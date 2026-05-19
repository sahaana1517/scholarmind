"use client";

import { useState } from "react";
import { askAgent, type ChatResponse } from "@/lib/api";

export default function Home() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await askAgent(query);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <div className="max-w-3xl mx-auto px-6 py-12">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-3xl font-bold mb-2">ScholarMind</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Agentic RAG over 30 research papers — ask anything about retrieval,
            knowledge graphs, transformers, or LLM agents.
          </p>
        </header>

        {/* Query input */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What is retrieval augmented generation?"
              disabled={loading}
              className="flex-1 px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Thinking..." : "Ask"}
            </button>
          </div>
        </form>

        {/* Loading state */}
        {loading && (
          <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6 mb-6">
            <div className="flex items-center gap-3 text-gray-600 dark:text-gray-400">
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
              <span>Agent planning, retrieving, and synthesizing...</span>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
            <p className="text-red-700 dark:text-red-300 font-mono text-sm">{error}</p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="space-y-6">
            {/* Plan */}
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <div className="text-xs font-semibold uppercase text-blue-700 dark:text-blue-400 mb-1">
                Agent Plan
              </div>
              <div className="text-sm">
                <span className="text-gray-600 dark:text-gray-400">Tool: </span>
                <code className="px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/40 text-blue-900 dark:text-blue-300 font-mono text-xs">
                  {result.plan.tool}
                </code>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 mt-1">
                {result.plan.reasoning}
              </p>
            </div>

            {/* Answer */}
            <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
              <div className="text-xs font-semibold uppercase text-gray-500 dark:text-gray-400 mb-3">
                Answer
              </div>
              <div className="prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap leading-relaxed">
                {result.answer}
              </div>
            </div>

            {/* Sources */}
            {result.sources.length > 0 && (
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
                <div className="text-xs font-semibold uppercase text-gray-500 dark:text-gray-400 mb-3">
                  Sources
                </div>
                <ul className="space-y-3">
                  {result.sources.map((s) => (
                    <li
                      key={s.index}
                      className="flex gap-3 text-sm border-l-2 border-gray-200 dark:border-gray-700 pl-3"
                    >
                      <span className="font-mono text-xs text-gray-500 dark:text-gray-400 pt-0.5">
                        [{s.index}]
                      </span>
                      <div className="flex-1">
                        <div className="font-mono text-xs text-gray-600 dark:text-gray-400 mb-1">
                          Paper {s.paper_id} · p.{s.page}
                        </div>
                        <p className="text-gray-700 dark:text-gray-300">
                          {s.preview}...
                        </p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Timings */}
            <div className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              Planner: {result.timings.planner_ms.toFixed(0)}ms ·
              Tool: {result.timings.tool_ms.toFixed(0)}ms ·
              Synthesis: {result.timings.synthesis_ms.toFixed(0)}ms ·
              Total: {result.timings.total_ms.toFixed(0)}ms
              {result.trace_id && (
                <>
                  {" · "}
                  <span title={result.trace_id}>
                    Trace: {result.trace_id.slice(0, 8)}...
                  </span>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}