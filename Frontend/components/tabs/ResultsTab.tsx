import React, { useState } from 'react';
import { Search, Eye, ChevronDown, ChevronUp } from 'lucide-react';

export default function ResultsTab() {
  // State to track which row is currently expanded
  const [expandedId, setExpandedId] = useState<string | null>('EXP-2026-001'); // Defaulting the first one to open for demonstration

  const toggleRow = (id: string) => {
    setExpandedId(prev => prev === id ? null : id);
  };

  // Expanded mock data to include full raw and optimized prompts
  const runs = [
    {
      id: 'EXP-2026-001',
      date: '2026-02-18',
      rawSnippet: 'Write a function to calculate factorial...',
      fullRaw: 'Write a function to calculate factorial',
      fullOpt: 'You are an expert software engineer. Create a Python function to calculate factorial with error handling...',
      delta: '+58%',
      score: 0.89,
      accept: 'High'
    },
    {
      id: 'EXP-2026-002',
      date: '2026-02-17',
      rawSnippet: 'Create a binary search algorithm...',
      fullRaw: 'Create a binary search algorithm for an array of numbers.',
      fullOpt: 'You are a computer science professor. Provide a highly optimized Python implementation of binary search. Ensure the array is sorted before searching and include time complexity in the comments.',
      delta: '+45%',
      score: 0.85,
      accept: 'Medium'
    },
    {
      id: 'EXP-2026-003',
      date: '2026-02-16',
      rawSnippet: 'Write a sorting function...',
      fullRaw: 'Write a sorting function',
      fullOpt: 'Implement the Merge Sort algorithm in Python. Include clear variable names, type hints, and a brief explanation of the divide-and-conquer strategy used.',
      delta: '+62%',
      score: 0.91,
      accept: 'High'
    },
    {
      id: 'EXP-2026-004',
      date: '2026-02-15',
      rawSnippet: 'Build a REST API endpoint...',
      fullRaw: 'Build a REST API endpoint to get users',
      fullOpt: 'Act as a Senior Backend Developer. Create a secure FastAPI GET endpoint at "/users". Implement pagination (limit/offset), error handling for database connection failures, and return a structured JSON response.',
      delta: '+51%',
      score: 0.87,
      accept: 'High'
    },
    {
      id: 'EXP-2026-005',
      date: '2026-02-14',
      rawSnippet: 'Create a web form validation...',
      fullRaw: 'Create a web form validation for email and password',
      fullOpt: 'Write a JavaScript utility function for form validation. The email must conform to RFC 5322. The password must be at least 8 characters long, contain an uppercase letter, a number, and a special character. Return specific error messages for each failure condition.',
      delta: '+55%',
      score: 0.88,
      accept: 'Medium'
    },
  ];

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 animate-in fade-in">
      <h1 className="text-2xl font-bold mb-1">Results History</h1>
      <p className="text-slate-500 text-sm mb-6">Historical experiments with output integrity metrics</p>

      {/* Search and Filter Controls */}
      <div className="flex space-x-4 mb-6">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-2.5 text-slate-400 w-5 h-5" />
          <input type="text" placeholder="Search by Run ID or prompt..." className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-md text-sm focus:outline-none focus:border-blue-500" />
        </div>
        <select className="border border-slate-200 rounded-md px-4 py-2 text-sm focus:outline-none focus:border-blue-500 bg-white">
          <option>Sort by Date (Newest)</option>
        </select>
      </div>

      {/* Results Table */}
      <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm mb-8">
        <div className="bg-blue-50 text-blue-800 text-xs px-6 py-3 border-b border-slate-200">
          Showing <strong>5</strong> experiments
        </div>
        <table className="w-full text-left text-sm text-slate-600">
          <thead className="bg-slate-50 border-b border-slate-200 text-xs uppercase font-bold text-slate-500">
            <tr>
              <th className="px-6 py-4">Run ID</th>
              <th className="px-6 py-4">Date</th>
              <th className="px-6 py-4">Raw Prompt Snippet</th>
              <th className="px-6 py-4">ΔQ</th>
              <th className="px-6 py-4">BERT Score</th>
              <th className="px-6 py-4">Acceptance</th>
              <th className="px-6 py-4 text-center">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {runs.map((run) => (
              <React.Fragment key={run.id}>
                {/* Main Visible Row */}
                <tr className={`hover:bg-slate-50 transition-colors ${expandedId === run.id ? 'bg-slate-50' : ''}`}>
                  <td className="px-6 py-4 font-mono text-xs text-slate-800 font-medium">{run.id}</td>
                  <td className="px-6 py-4">{run.date}</td>
                  <td className="px-6 py-4 truncate max-w-xs">{run.rawSnippet}</td>
                  <td className="px-6 py-4 font-bold text-green-500">{run.delta}</td>
                  <td className="px-6 py-4">{run.score}</td>
                  <td className="px-6 py-4">
                    <span className={`px-2 py-1 rounded text-xs font-bold ${run.accept === 'High' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                      {run.accept}
                    </span>
                  </td>
                  <td className="px-6 py-4 flex justify-center">
                    <button
                      onClick={() => toggleRow(run.id)}
                      className="text-blue-600 hover:text-blue-800 transition-colors flex items-center p-1 rounded-md hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      aria-label={expandedId === run.id ? "Collapse details" : "Expand details"}
                    >
                      <Eye className="w-4 h-4 mr-1" />
                      {expandedId === run.id ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                  </td>
                </tr>

                {/* Expandable Details Row */}
                {expandedId === run.id && (
                  <tr className="bg-slate-50 border-b border-slate-200">
                    <td colSpan={7} className="px-8 py-6">
                      <div className="flex flex-col md:flex-row gap-6 animate-in slide-in-from-top-2 duration-200">

                        {/* Raw Prompt Detail */}
                        <div className="flex-1 flex flex-col">
                          <h4 className="font-semibold text-slate-800 text-sm mb-3">Raw Prompt</h4>
                          <div className="bg-white border border-slate-200 rounded-lg p-4 text-sm text-slate-600 flex-1 shadow-sm">
                            {run.fullRaw}
                          </div>
                        </div>

                        {/* Optimized Prompt Detail */}
                        <div className="flex-1 flex flex-col">
                          <h4 className="font-semibold text-slate-800 text-sm mb-3">Optimized Prompt</h4>
                          <div className="bg-white border border-slate-200 rounded-lg p-4 text-sm text-slate-600 flex-1 shadow-sm">
                            {run.fullOpt}
                          </div>
                        </div>

                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {/* Metrics Explanation Footer */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 shadow-sm">
        <h4 className="font-bold text-yellow-800 mb-2">Why Output Integrity Metrics?</h4>
        <p className="text-sm text-yellow-700 leading-relaxed">
          <strong>BERTScore:</strong> Uses contextual embeddings from BERT to evaluate semantic similarity between the optimized output and reference outputs. Unlike traditional n-gram matching metrics, BERTScore captures deeper semantic meaning and context, ensuring the system preserves the user's original intent while improving prompt quality and structure.
        </p>
      </div>
    </div>
  );
}