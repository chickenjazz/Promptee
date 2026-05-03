import React from "react";
import type { PromptIssue, PromptIssueSeverity } from "@/types/prompt";

type Props = {
  issues: PromptIssue[];
  theme?: "light" | "dark";
};

const severityClassLight: Record<PromptIssueSeverity, string> = {
  low: "text-slate-500",
  medium: "text-amber-600",
  high: "text-red-600",
};

const severityClassDark: Record<PromptIssueSeverity, string> = {
  low: "text-sky-300",
  medium: "text-amber-300",
  high: "text-rose-300",
};

export default function IssuePanel({ issues, theme = "light" }: Props) {
  const isDark = theme === "dark";

  if (!issues.length) {
    return (
      <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-600"}`}>
        No major prompt issues detected.
      </p>
    );
  }

  const severityClass = isDark ? severityClassDark : severityClassLight;

  return (
    <div className="space-y-3">
      {issues.map((issue) => (
        <div
          key={issue.id}
          className={`rounded-lg border p-3 shadow-sm ${isDark
            ? "border-slate-700 bg-slate-800"
            : "border-slate-200 bg-white"
            }`}
        >
          <div className="flex items-center justify-between gap-2">
            <h4
              className={`font-semibold text-sm capitalize ${isDark ? "text-slate-100" : "text-slate-800"
                }`}
            >
              {issue.type.replaceAll("_", " ")}
            </h4>
            <span
              className={`text-[10px] font-bold uppercase tracking-wider ${severityClass[issue.severity]
                }`}
            >
              {issue.severity}
            </span>
          </div>
          {issue.span && (
            <p
              className={`mt-1 text-xs ${isDark ? "text-slate-300" : "text-slate-600"
                }`}
            >
              Highlighted:{" "}
              <code
                className={`rounded px-1 ${isDark ? "bg-slate-700 text-slate-200" : "bg-slate-100 text-slate-800"
                  }`}
              >
                {issue.span}
              </code>
            </p>
          )}
          <p
            className={`mt-2 text-xs ${isDark ? "text-slate-300" : "text-slate-700"
              }`}
          >
            {issue.message}
          </p>
          <p
            className={`mt-1 text-xs ${isDark ? "text-slate-400" : "text-slate-500"
              }`}
          >
            Suggestion: {issue.suggestion}
          </p>
        </div>
      ))}
    </div>
  );
}
