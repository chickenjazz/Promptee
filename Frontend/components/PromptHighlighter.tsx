import React from "react";
import type { PromptIssue } from "@/types/prompt";

type Props = {
  text: string;
  issues: PromptIssue[];
  theme?: "light" | "dark";
};

const issueClassMapLight: Record<string, string> = {
  ambiguity: "bg-yellow-200 underline decoration-yellow-600 text-slate-800",
  weak_action: "bg-orange-200 underline decoration-orange-600 text-slate-800",
  redundancy: "bg-blue-200 underline decoration-blue-600 text-slate-800",
  too_short: "bg-red-100 underline decoration-red-600 text-slate-800",
  meta_prompt_drift: "bg-red-200 underline decoration-red-700 text-slate-800",
};

const issueClassMapDark: Record<string, string> = {
  ambiguity: "bg-yellow-900/50 underline decoration-yellow-500 text-yellow-100",
  weak_action: "bg-orange-900/50 underline decoration-orange-500 text-orange-100",
  redundancy: "bg-blue-900/50 underline decoration-blue-500 text-blue-100",
  too_short: "bg-red-900/50 underline decoration-red-500 text-red-100",
  meta_prompt_drift: "bg-red-950 underline decoration-red-500 text-red-100",
};

export default function PromptHighlighter({ text, issues, theme = "light" }: Props) {
  const isDark = theme === "dark";
  const issueClassMap = isDark ? issueClassMapDark : issueClassMapLight;

  const inlineIssues = issues
    .filter((issue) => issue.start !== null && issue.end !== null)
    .sort((a, b) => (a.start ?? 0) - (b.start ?? 0));

  if (inlineIssues.length === 0) {
    return (
      <p className={`whitespace-pre-wrap text-sm italic ${isDark ? "text-slate-500" : "text-slate-400"}`}>
        No text highlighted for issues.
      </p>
    );
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
        className={`rounded px-1 ${issueClassMap[issue.type] ?? (isDark ? "bg-slate-700 underline text-slate-200" : "bg-slate-200 underline text-slate-800")}`}
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

  return (
    <p className={`whitespace-pre-wrap leading-7 text-sm ${isDark ? "text-slate-300" : "text-slate-700"}`}>
      {parts}
    </p>
  );
}
