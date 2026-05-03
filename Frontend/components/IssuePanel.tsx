import React from "react";
import type { PromptIssue, PromptIssueSeverity } from "@/types/prompt";

type Props = {
  issues: PromptIssue[];
};

const severityClass: Record<PromptIssueSeverity, string> = {
  low: "text-slate-500",
  medium: "text-amber-600",
  high: "text-red-600",
};

export default function IssuePanel({ issues }: Props) {
  if (!issues.length) {
    return <p className="text-sm text-slate-600">No major prompt issues detected.</p>;
  }

  return (
    <div className="space-y-3">
      {issues.map((issue) => (
        <div key={issue.id} className="rounded-lg border border-slate-200 bg-white p-3 shadow-sm">
          <div className="flex items-center justify-between gap-2">
            <h4 className="font-semibold text-sm capitalize text-slate-800">
              {issue.type.replaceAll("_", " ")}
            </h4>
            <span className={`text-[10px] font-bold uppercase tracking-wider ${severityClass[issue.severity]}`}>
              {issue.severity}
            </span>
          </div>
          {issue.span && (
            <p className="mt-1 text-xs text-slate-600">
              Highlighted: <code className="rounded bg-slate-100 px-1">{issue.span}</code>
            </p>
          )}
          <p className="mt-2 text-xs text-slate-700">{issue.message}</p>
          <p className="mt-1 text-xs text-slate-500">Suggestion: {issue.suggestion}</p>
        </div>
      ))}
    </div>
  );
}
