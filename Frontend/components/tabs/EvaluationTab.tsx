import React from 'react';
import { Sparkles, ChevronDown, Download, FileText } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { OptimizedData } from '@/app/page';

export default function EvaluationTab({ optimizedData }: { optimizedData: OptimizedData | null }) {
  const chartData = [
    { 
      subject: 'Clarity', 
      raw: optimizedData ? Math.round(optimizedData.raw_score?.clarity * 100) : 45, 
      optimized: optimizedData ? Math.round(optimizedData.optimized_score?.clarity * 100) : 92 
    },
    { 
      subject: 'Specificity', 
      raw: optimizedData ? Math.round(optimizedData.raw_score?.specificity * 100) : 30, 
      optimized: optimizedData ? Math.round(optimizedData.optimized_score?.specificity * 100) : 88 
    },
  ];

  const improvementScore = optimizedData ? `+${Math.round(optimizedData.improvement_score * 100)}%` : '+58%';
  const clarityDelta = optimizedData ? optimizedData.optimized_score?.clarity_delta : 0;
  const specificityDelta = optimizedData ? optimizedData.optimized_score?.specificity_delta : 0;
  const topCriterion = clarityDelta > specificityDelta ? 'Clarity' : 'Specificity';
  const topDelta = Math.max(clarityDelta, specificityDelta);
  const topImprovement = optimizedData ? `+${Math.round(topDelta * 100)}% improvement` : '+58% improvement';


  return (
    <div className="max-w-6xl mx-auto px-6 py-8 animate-in fade-in">
      <h1 className="text-2xl font-bold mb-1">Evaluation Dashboard</h1>
      <p className="text-slate-500 text-sm mb-8">Transparent scoring, reproducibility, and auditability</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-blue-600 text-white rounded-xl p-6 shadow-md relative overflow-hidden">
          <h3 className="text-blue-100 font-medium mb-1 relative z-10">Total Improvement Score</h3>
          <div className="text-5xl font-bold mb-2 relative z-10">ΔQ = {improvementScore}</div>
          <div className="text-xs text-blue-200 relative z-10">Q_opt - Q_raw</div>
          <Sparkles className="absolute right-4 bottom-4 w-24 h-24 text-white opacity-10" />
        </div>
        <div className="bg-green-500 text-white rounded-xl p-6 shadow-md relative overflow-hidden">
          <h3 className="text-green-100 font-medium mb-1 relative z-10">Top Improved Criterion</h3>
          <div className="text-3xl font-bold mb-2 relative z-10">{topCriterion}</div>
          <div className="text-sm text-green-100 relative z-10">{topImprovement}</div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm mb-8">
        <h3 className="font-bold text-lg mb-6 text-slate-800">Multi-Criteria Comparison</h3>
        <div className="h-80 w-full flex justify-center items-center">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
              barGap={8}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis
                dataKey="subject"
                tick={{ fill: '#64748b', fontSize: 14, fontWeight: 500 }}
                axisLine={false}
                tickLine={false}
                dy={10}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                dx={-10}
              />
              <Tooltip
                cursor={{ fill: '#f8fafc' }}
                contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
              />
              <Bar name="Raw Prompt" dataKey="raw" fill="#ef4444" radius={[4, 4, 0, 0]} maxBarSize={60} />
              <Bar name="Optimized Prompt" dataKey="optimized" fill="#22c55e" radius={[4, 4, 0, 0]} maxBarSize={60} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Custom Legend */}
        <div className="flex justify-center space-x-6 mt-6 pt-4 border-t border-slate-100">
          <div className="flex items-center text-sm font-medium text-slate-600">
            <span className="w-3 h-3 rounded-full bg-red-500 mr-2 shadow-sm"></span> Raw Prompt
          </div>
          <div className="flex items-center text-sm font-medium text-slate-600">
            <span className="w-3 h-3 rounded-full bg-green-500 mr-2 shadow-sm"></span> Optimized Prompt
          </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm mb-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="font-bold text-lg text-slate-800">Heuristic Rubric & MCDM Weights</h3>
          <ChevronDown className="w-5 h-5 text-slate-400" />
        </div>
        <p className="text-sm text-slate-500 mb-6">Current weighting method: <strong>Weighted Sum Model (WSM)</strong></p>

        <div className="space-y-6">
          {[
            { name: 'Clarity (w₁)', desc: 'Role definition and instruction transparency', val: 50 },
            { name: 'Specificity (w₂)', desc: 'Constraint precision and requirement detail', val: 50 },
          ].map(w => (
            <div key={w.name} className="border border-slate-100 p-4 rounded-lg bg-slate-50">
              <div className="flex justify-between mb-2">
                <div>
                  <div className="font-medium text-sm text-slate-800">{w.name}</div>
                  <div className="text-xs text-slate-500">{w.desc}</div>
                </div>
                <div className="bg-blue-100 text-blue-800 text-xs font-bold px-3 py-1 rounded-full h-min">{w.val}%</div>
              </div>
              <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden mt-3">
                <div className="h-full bg-blue-600" style={{ width: `${w.val}%` }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h3 className="font-bold text-lg mb-4 text-slate-800">Export & Reporting</h3>
        <div className="flex space-x-4">
          <button className="bg-blue-600 text-white px-5 py-2.5 rounded-md font-medium flex items-center hover:bg-blue-700 text-sm">
            <Download className="w-4 h-4 mr-2" /> Download Report (PDF)
          </button>
          <button className="bg-slate-700 text-white px-5 py-2.5 rounded-md font-medium flex items-center hover:bg-slate-800 text-sm">
            <FileText className="w-4 h-4 mr-2" /> Export Run Logs (CSV)
          </button>
        </div>
      </div>
    </div>
  );
}