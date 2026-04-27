import React, { useState } from 'react';
import { Sparkles, ShieldAlert, CheckCircle2, FileText, Lock, Loader2, RefreshCw, BarChart3 } from 'lucide-react';
import { OptimizedData } from '@/app/page';

export default function DemoTab({ user, onSignIn, optimizedData, setOptimizedData }: { user: any, onSignIn: () => void, optimizedData: OptimizedData | null, setOptimizedData: (data: OptimizedData | null) => void }) {
  const [rawPrompt, setRawPrompt] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [copied, setCopied] = useState(false);

  // State to manage which LLM Output is currently being viewed
  const [llmView, setLlmView] = useState<'raw' | 'optimized'>('optimized');

  // --- Backend Simulation Logic ---
  const handleOptimize = async () => {
    if (!rawPrompt.trim()) return;
    setStatus('loading');
    setOptimizedData(null);
    setLlmView('optimized'); // Reset view to optimized upon new run

    try {
      const response = await fetch('http://127.0.0.1:8000/optimize_prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: rawPrompt })
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Derive some explainable actions based on metrics
      const actions: string[] = [];
      if (data.optimized_score?.clarity_delta > 0) actions.push(`Improved clarity by +${(data.optimized_score.clarity_delta * 100).toFixed(0)}%`);
      if (data.optimized_score?.specificity_delta > 0) actions.push(`Increased specificity by +${(data.optimized_score.specificity_delta * 100).toFixed(0)}%`);
      if (data.optimized_score?.structural_bonus > 0) actions.push("Added structural formatting elements");
      if (actions.length === 0) actions.push("Re-worded for better processing");

      const newData: OptimizedData = {
        ...data,
        actions
      };

      setOptimizedData(newData);
      setStatus('success');

      // Simulate saving to history if logged in
      if (user) {
        const history = JSON.parse(localStorage.getItem('promptee_history') || '[]');
        history.unshift({
          id: `EXP-${new Date().getFullYear()}-${Math.floor(Math.random() * 1000)}`,
          date: new Date().toISOString().split('T')[0],
          raw: rawPrompt,
          delta: `+${((data.improvement_score || 0) * 100).toFixed(0)}%`,
          score: (data.optimized_score?.candidate_quality || 0).toFixed(2),
          accept: 'High'
        });
        localStorage.setItem('promptee_history', JSON.stringify(history));
      }
    } catch (err) {
      console.error(err);
      setStatus('error');
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(optimizedData?.optimized_prompt || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 animate-in fade-in">
      <h1 className="text-2xl font-bold mb-1">Prompt Optimization Demo</h1>
      <p className="text-slate-500 text-sm mb-6">Experience real-time prompt engineering with multi-criteria evaluation</p>

      {/* Main Action Bar */}
      <div className="bg-white p-4 rounded-lg border border-slate-200 mb-6 flex justify-between items-center shadow-sm">
        <button
          onClick={handleOptimize}
          disabled={status === 'loading' || !rawPrompt.trim()}
          className="bg-slate-200 text-slate-500 px-6 py-2 rounded-md font-medium text-sm flex items-center transition-all focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 data-[active=true]:bg-blue-600 data-[active=true]:text-white data-[active=true]:hover:bg-blue-700"
          data-active={rawPrompt.trim().length > 0 && status !== 'loading'}
          aria-live="polite"
        >
          {status === 'loading' ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Sparkles className="w-4 h-4 mr-2" />}
          {status === 'loading' ? 'Processing via Qwen...' : 'Run Optimization'}
        </button>
        {status === 'error' && <span className="text-red-500 text-sm font-medium">Optimization failed. Try again.</span>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* 1. INPUT STATE */}
        <div className="bg-white border border-slate-200 rounded-lg flex flex-col shadow-sm focus-within:ring-1 focus-within:ring-blue-500">
          <div className="p-4 border-b border-slate-100 font-bold text-sm bg-slate-50 flex justify-between items-center">
            Raw Prompt Input
            <button onClick={() => setRawPrompt('')} className="text-slate-400 hover:text-slate-600" aria-label="Clear prompt"><RefreshCw className="w-4 h-4" /></button>
          </div>
          <div className="p-4 flex-1">
            <label htmlFor="raw-prompt" className="text-xs font-medium text-slate-700 mb-2 block">Enter your prompt</label>
            <textarea
              id="raw-prompt"
              value={rawPrompt}
              onChange={(e) => setRawPrompt(e.target.value)}
              className="w-full h-40 border border-slate-200 rounded-md p-3 text-sm focus:outline-none focus:border-blue-500 resize-none text-slate-700 placeholder:text-slate-400"
              placeholder="Write a function to calculate factorial..."
            />
          </div>
          <div className="bg-blue-50 p-4 border-t border-blue-100 flex items-start text-blue-800 text-xs">
            <ShieldAlert className="w-4 h-4 mr-2 flex-shrink-0 mt-0.5" />
            <p><strong>Privacy Notice:</strong> Do not input sensitive personal data.</p>
          </div>
        </div>

        {/* 2. OPTIMIZED PROMPT OUTPUT */}
        <div className="bg-white border border-slate-200 rounded-lg flex flex-col shadow-sm relative">
          <div className="p-4 border-b border-slate-100 font-bold text-sm bg-slate-50">Optimized Prompt</div>

          {status === 'idle' && (
            <div className="flex-1 p-4">
              <textarea className="w-full h-full border border-slate-200 rounded-md p-3 text-sm bg-slate-50 text-slate-400 resize-none" readOnly placeholder="Optimized prompt will appear here..." />
            </div>
          )}

          {status === 'loading' && (
            <div className="flex-1 flex flex-col items-center justify-center p-8">
              <Loader2 className="w-8 h-8 text-blue-500 animate-spin mb-4" />
              <div className="space-y-2 w-full max-w-[200px]">
                <div className="h-2 bg-slate-200 rounded animate-pulse"></div>
                <div className="h-2 bg-slate-200 rounded animate-pulse w-5/6"></div>
                <div className="h-2 bg-slate-200 rounded animate-pulse w-4/6"></div>
              </div>
            </div>
          )}

          {status === 'success' && optimizedData && (
            <>
              <div className="p-4 flex-1 overflow-auto animate-in fade-in">
                <textarea className="w-full h-32 border border-slate-200 rounded-md p-3 text-sm bg-slate-50 text-slate-700 resize-none font-mono text-xs focus:outline-none" readOnly value={optimizedData.optimized_prompt} aria-label="Optimized Prompt Output" />
                <div className="mt-4">
                  <h4 className="text-xs font-bold mb-2 text-slate-800">Explainable Actions</h4>
                  <ul className="space-y-2 text-xs text-slate-600">
                    {optimizedData.actions?.map((act: string, i: number) => (
                      <li key={i} className="flex items-start"><CheckCircle2 className="text-green-500 w-4 h-4 mr-2 flex-shrink-0" /> {act}</li>
                    ))}
                  </ul>
                </div>
              </div>
              <div className="p-4 border-t border-slate-100">
                <button onClick={copyToClipboard} className="w-full bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium py-2 rounded-md text-sm transition-colors flex justify-center items-center focus:ring-2 focus:ring-slate-400">
                  {copied ? <CheckCircle2 className="w-4 h-4 mr-2 text-green-600" /> : <FileText className="w-4 h-4 mr-2" />}
                  {copied ? 'Copied to Clipboard!' : 'Copy Optimized Prompt'}
                </button>
              </div>
            </>
          )}
        </div>

        {/* 3. LLM OUTPUT PREVIEW (With Toggle) */}
        <div className="bg-white border border-slate-200 rounded-lg flex flex-col shadow-sm">
          <div className="p-4 border-b border-slate-100 font-bold text-sm bg-slate-50">LLM Output</div>
          <div className="p-4 flex-1 flex flex-col">

            {/* Toggle Buttons */}
            <div className="flex bg-slate-100 rounded-md p-1 mb-4">
              <button
                onClick={() => setLlmView('raw')}
                disabled={status !== 'success'}
                className={`flex-1 text-xs py-2 rounded font-medium transition-colors ${llmView === 'raw' && status === 'success' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-700 disabled:opacity-50 disabled:hover:text-slate-500'}`}
              >
                Raw Prompt Output
              </button>
              <button
                onClick={() => setLlmView('optimized')}
                disabled={status !== 'success'}
                className={`flex-1 text-xs py-2 rounded font-medium transition-colors ${llmView === 'optimized' && status === 'success' ? 'bg-blue-600 text-white shadow-sm' : 'text-slate-500 hover:text-slate-700 disabled:opacity-50 disabled:hover:text-slate-500'}`}
              >
                Optimized Output
              </button>
            </div>

            {/* Content Area */}
            {status === 'success' && optimizedData ? (
              <div className="bg-slate-50 border border-slate-200 p-3 rounded-md flex-1 font-mono text-xs text-slate-700 overflow-auto animate-in fade-in">
                <pre className="whitespace-pre-wrap font-inherit m-0">
                  {llmView === 'raw' ? optimizedData.external_llm_response_raw : optimizedData.external_llm_response_optimized}
                </pre>
              </div>
            ) : (
              <textarea className="w-full flex-1 border border-slate-200 rounded-md p-3 text-sm bg-slate-50 text-slate-400 resize-none" readOnly placeholder="LLM output will appear here after optimization..." />
            )}
          </div>
          {status === 'success' && (
            <div className="p-4 border-t border-slate-100 bg-green-50/50 text-green-800 text-xs flex items-center">
              <Sparkles className="w-4 h-4 mr-2" />
              <p><strong>Powered by:</strong> External Large Language Model (GPT-4 / Claude / etc.)</p>
            </div>
          )}
        </div>
      </div>

      {/* AUTH LOCKED STATE & METRICS */}
      {status === 'idle' || status === 'loading' || status === 'error' ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center shadow-sm animate-in fade-in">
          <div className="w-16 h-16 bg-slate-50 text-slate-400 rounded-full flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-bold mb-2 text-slate-700">
            {status === 'loading' ? 'Calculating Metrics...' : 'Metrics Awaiting Run'}
          </h3>
          <p className="text-slate-500 text-sm max-w-md mx-auto">
            {status === 'loading'
              ? 'Analyzing heuristic criteria against multi-dimensional benchmarks...'
              : 'Enter a raw prompt and click "Run Optimization" above to generate your real-time multi-criteria performance metrics.'}
          </p>
        </div>
      ) : !user ? (
        <div className="bg-white border border-slate-200 rounded-xl p-12 text-center shadow-sm animate-in fade-in">
          <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4"><Lock className="w-8 h-8" /></div>
          <h3 className="text-xl font-bold mb-2">Full Evaluation Locked</h3>
          <p className="text-slate-500 text-sm max-w-md mx-auto mb-6">Access complete multi-criteria evaluation metrics, comparison charts, and experiment history by creating a free account.</p>
          <button onClick={onSignIn} className="bg-blue-600 text-white px-6 py-2.5 rounded-md font-medium hover:bg-blue-700 transition-colors">Sign In / Sign Up</button>
        </div>
      ) : (
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm animate-in fade-in slide-in-from-bottom-4 duration-500">
          <h3 className="text-lg font-bold mb-6 border-b border-slate-100 pb-4">Metrics Snapshot</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-6 mb-8">
            {[
              { 
                label: 'Clarity', 
                raw: optimizedData ? Math.round(optimizedData.raw_score?.clarity * 100) : 45, 
                opt: optimizedData ? Math.round(optimizedData.optimized_score?.clarity * 100) : 92, 
                delta: optimizedData ? `+${Math.round(optimizedData.optimized_score?.clarity_delta * 100)}%` : '+47%' 
              },
              { 
                label: 'Specificity', 
                raw: optimizedData ? Math.round(optimizedData.raw_score?.specificity * 100) : 30, 
                opt: optimizedData ? Math.round(optimizedData.optimized_score?.specificity * 100) : 88, 
                delta: optimizedData ? `+${Math.round(optimizedData.optimized_score?.specificity_delta * 100)}%` : '+58%' 
              },
            ].map(metric => (
              <div key={metric.label} className="flex flex-col space-y-3">
                <div className="flex justify-between items-center text-sm">
                  <span className="font-medium text-slate-700">{metric.label}</span>
                  <span className="text-green-500 font-bold">{metric.delta}</span>
                </div>

                {/* Raw Bar */}
                <div>
                  <div className="text-[11px] text-slate-500 mb-1.5">Raw: {metric.raw}%</div>
                  <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-red-400 rounded-full" style={{ width: `${metric.raw}%` }}></div>
                  </div>
                </div>

                {/* Optimized Bar */}
                <div>
                  <div className="text-[11px] text-slate-500 mb-1.5">Optimized: {metric.opt}%</div>
                  <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div className="h-full bg-green-500 rounded-full" style={{ width: `${metric.opt}%` }}></div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-md text-sm text-yellow-800">
            <strong>Why this exists:</strong> According to the Garbage In, Garbage Out (GIGO) theory, sound optimization requires explicit, testable criteria at the prompt interface to prevent noisy supervision signals.
          </div>
        </div>
      )}
    </div>
  );
}