import React, { useState, useEffect } from 'react';
import { Search, Eye, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';

export default function ResultsTab({ user, isActive }: { user: { id: number, username: string }, isActive?: boolean }) {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [runs, setRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isActive) return;
    
    const fetchHistory = async () => {
      setLoading(true);
      try {
        const res = await fetch(`http://127.0.0.1:8000/history/${user.id}`);
        const data = await res.json();
        setRuns(data.history || []);
      } catch (err) {
        console.error('Failed to fetch history', err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [user.id, isActive]);

  const toggleRow = (id: number) => {
    setExpandedId(prev => prev === id ? null : id);
  };

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
          {loading ? 'Loading...' : <>Showing <strong>{runs.length}</strong> experiments</>}
        </div>
        
        {loading ? (
          <div className="flex justify-center p-8">
            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
          </div>
        ) : (
          <table className="w-full text-left text-sm text-slate-600">
            <thead className="bg-slate-50 border-b border-slate-200 text-xs uppercase font-bold text-slate-500">
              <tr>
                <th className="px-6 py-4">Date</th>
                <th className="px-6 py-4">Raw Prompt Snippet</th>
                <th className="px-6 py-4">ΔQ</th>
                <th className="px-6 py-4">BERT Score</th>
                <th className="px-6 py-4">Acceptance</th>
                <th className="px-6 py-4 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {runs.map((run, idx) => (
                <React.Fragment key={idx}>
                  <tr className={`hover:bg-slate-50 transition-colors ${expandedId === idx ? 'bg-slate-50' : ''}`}>
                    <td className="px-6 py-4">{new Date(run.timestamp).toLocaleString()}</td>
                    <td className="px-6 py-4 truncate max-w-xs">{run.raw_prompt.slice(0, 40)}...</td>
                    <td className="px-6 py-4 font-bold text-green-500">+{(run.improvement_score * 100).toFixed(0)}%</td>
                    <td className="px-6 py-4">{(run.optimized_score?.candidate_quality || 0).toFixed(2)}</td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${run.improvement_score > 0.4 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'}`}>
                        {run.improvement_score > 0.4 ? 'High' : 'Medium'}
                      </span>
                    </td>
                    <td className="px-6 py-4 flex justify-center">
                      <button
                        onClick={() => toggleRow(idx)}
                        className="text-blue-600 hover:text-blue-800 transition-colors flex items-center p-1 rounded-md hover:bg-blue-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <Eye className="w-4 h-4 mr-1" />
                        {expandedId === idx ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                    </td>
                  </tr>

                  {expandedId === idx && (
                    <tr className="bg-slate-50 border-b border-slate-200">
                      <td colSpan={6} className="px-8 py-6">
                        <div className="flex flex-col md:flex-row gap-6 animate-in slide-in-from-top-2 duration-200">
                          <div className="flex-1 flex flex-col">
                            <h4 className="font-semibold text-slate-800 text-sm mb-3">Raw Prompt</h4>
                            <div className="bg-white border border-slate-200 rounded-lg p-4 text-sm text-slate-600 flex-1 shadow-sm whitespace-pre-wrap">
                              {run.raw_prompt}
                            </div>
                          </div>
                          <div className="flex-1 flex flex-col">
                            <h4 className="font-semibold text-slate-800 text-sm mb-3">Optimized Prompt</h4>
                            <div className="bg-white border border-slate-200 rounded-lg p-4 text-sm text-slate-600 flex-1 shadow-sm whitespace-pre-wrap">
                              {run.optimized_prompt}
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                    No experiments found. Run some optimizations in the Demo tab!
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>

      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6 shadow-sm">
        <h4 className="font-bold text-yellow-800 mb-2">Why Output Integrity Metrics?</h4>
        <p className="text-sm text-yellow-700 leading-relaxed">
          <strong>BERTScore:</strong> Uses contextual embeddings from BERT to evaluate semantic similarity between the optimized output and reference outputs. Unlike traditional n-gram matching metrics, BERTScore captures deeper semantic meaning and context, ensuring the system preserves the user's original intent while improving prompt quality and structure.
        </p>
      </div>
    </div>
  );
}