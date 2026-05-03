import React from "react";
import type { PromptIssue } from "@/types/prompt";

type Props = {
  text: string;
  issues: PromptIssue[];
};

const issueClassMap: Record<string, string> = {
  ambiguity: "bg-yellow-200 underline decoration-yellow-600",
  weak_action: "bg-orange-200 underline decoration-orange-600",
  redundancy: "bg-blue-200 underline decoration-blue-600",
  too_short: "bg-red-100 underline decoration-red-600",
  meta_prompt_drift: "bg-red-200 underline decoration-red-700",
};

export default function PromptHighlighter({ text, issues }: Props) {
  const inlineIssues = issues
    .filter((issue) => issue.start !== null && issue.end !== null)
    .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));

  if (inlineIssues.length === 0) {
    return <p className="whitespace-pre-wrap text-sm text-slate-700">{text || <span className="text-slate-400">No prompt entered.</span>}</p>;
  }

  const parts: React.ReactNode[] = [];
  let cursor = 0;

  inlineIssues.forEach((issue) => {
    const start = issue.start ?? 0;
    const end = issue.end ?? start;

    if (start < cursor) return;

    if (cursor < start) {
      parts.push(<span key={`plain-${cursor}`}>{text.slice(cursor, start)}</span>);
    }

    parts.push(
      <mark
        key={issue.id}
        className={`rounded px-1 ${issueClassMap[issue.type] ?? "bg-slate-200 underline"}`}
        title={`${issue.message} Suggestion: ${issue.suggestion}`}
      >
        {text.slice(start, end)}
      </mark>
    );

    cursor = end;
  });

  if (cursor < text.length) {
    parts.push(<span key={`plain-${cursor}`}>{text.slice(cursor)}</span>);
  }

  return <p className="whitespace-pre-wrap leading-7 text-sm text-slate-700">{parts}</p>;
}
