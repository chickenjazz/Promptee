import React from "react";

type Props = {
  recommendations: string[];
  institutionalGuideline?: string;
  theme?: "light" | "dark";
};

export default function RecommendationPanel({ recommendations, institutionalGuideline, theme = "light" }: Props) {
  const isDark = theme === "dark";

  if (!recommendations.length && !institutionalGuideline) {
    return (
      <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-600"}`}>
        No tutor guidance available yet.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {recommendations.length > 0 && (
        <div
          className={`rounded-lg border p-4 shadow-sm ${
            isDark ? "border-slate-700 bg-slate-800" : "border-slate-200 bg-white"
          }`}
        >
          <h3
            className={`text-sm font-bold ${
              isDark ? "text-slate-100" : "text-slate-800"
            }`}
          >
            Actionable Recommendations
          </h3>
          <ol
            className={`mt-2 list-decimal space-y-1 pl-5 text-xs ${
              isDark ? "text-slate-300" : "text-slate-700"
            }`}
          >
            {recommendations.map((recommendation, index) => (
              <li key={index}>{recommendation}</li>
            ))}
          </ol>
        </div>
      )}

      {institutionalGuideline && (
        <div
          className={`rounded-lg border p-4 shadow-sm ${
            isDark ? "border-sky-900 bg-slate-800" : "border-blue-100 bg-blue-50"
          }`}
        >
          <h3
            className={`text-sm font-bold ${
              isDark ? "text-sky-300" : "text-blue-900"
            }`}
          >
            Educational Guideline
          </h3>
          <p
            className={`mt-2 text-xs ${
              isDark ? "text-sky-100/90" : "text-blue-900/90"
            }`}
          >
            {institutionalGuideline}
          </p>
        </div>
      )}
    </div>
  );
}
