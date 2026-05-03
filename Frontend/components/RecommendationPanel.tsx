import React from "react";

type Props = {
  recommendations: string[];
  institutionalGuideline?: string;
};

export default function RecommendationPanel({ recommendations, institutionalGuideline }: Props) {
  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="text-sm font-bold text-slate-800">Actionable Recommendations</h3>
        <ol className="mt-2 list-decimal space-y-1 pl-5 text-xs text-slate-700">
          {recommendations.map((recommendation, index) => (
            <li key={index}>{recommendation}</li>
          ))}
        </ol>
      </div>

      {institutionalGuideline && (
        <div className="rounded-lg border border-blue-100 bg-blue-50 p-4 shadow-sm">
          <h3 className="text-sm font-bold text-blue-900">Educational Guideline</h3>
          <p className="mt-2 text-xs text-blue-900/90">{institutionalGuideline}</p>
        </div>
      )}
    </div>
  );
}
