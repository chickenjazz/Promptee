import React, { useState, useRef, useEffect } from 'react';
import { Sparkles, ShieldAlert, CheckCircle2, FileText, Loader2, RefreshCw, BarChart3, TrendingUp, Zap, ThumbsUp, ThumbsDown } from 'lucide-react';
import { OptimizedData } from '@/app/page';
import PromptHighlighter from '@/components/PromptHighlighter';
import IssuePanel from '@/components/IssuePanel';
import RecommendationPanel from '@/components/RecommendationPanel';

// Local component for collapsible sidebar sections
function SidebarSection({
  title,
  children,
  defaultOpen = true,
  count,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  count?: number;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section className="border-b border-slate-200">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center justify-between px-5 py-4 text-left hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-600">
            {title}
          </span>
          {typeof count === "number" && (
            <span className="rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-semibold text-slate-700">
              {count}
            </span>
          )}
        </div>
        <span className="text-slate-400 font-mono text-lg leading-none">
          {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <div className="px-5 pb-4">
          {children}
        </div>
      )}
    </section>
  );
}

export default function DemoTab({ user, onSignIn, optimizedData, setOptimizedData }: { user: any, onSignIn: () => void, optimizedData: OptimizedData | null, setOptimizedData: (data: OptimizedData | null) => void }) {
  const [rawPrompt, setRawPrompt] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<'none' | 'like' | 'dislike'>('none');
  // External-LLM benchmarking is off by default — it adds 1-3s of network round-trip
  // to every request. Users opt in when they want the side-by-side comparison.
  const [runBenchmark, setRunBenchmark] = useState(false);

  // Auto-resize optimized prompt textarea and scroll to card
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const optimizedCardRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (status === 'success') {
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
      }
      if (optimizedCardRef.current) {
        optimizedCardRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, [optimizedData?.optimized_prompt, status]);

  // State to manage which LLM Output is currently being viewed
  const [llmView, setLlmView] = useState<'raw' | 'optimized'>('optimized');

  // --- Backend Simulation Logic ---
  const handleOptimize = async () => {
    if (!rawPrompt.trim()) return;
    setStatus('loading');
    setFeedback('none');
    setOptimizedData(null);
    setLlmView('optimized'); // Reset view to optimized upon new run

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';
      const response = await fetch(`${apiBase}/optimize_prompt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: rawPrompt, benchmark: runBenchmark })
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

  // Compute metrics for the evaluation report
  const rawOverall = optimizedData ? Math.round(optimizedData.raw_score?.raw_quality * 100) : 0;
  const optOverall = optimizedData ? Math.round(optimizedData.optimized_score?.candidate_quality * 100) : 0;
  const clarityRaw = optimizedData ? Math.round(optimizedData.raw_score?.clarity * 100) : 0;
  const clarityOpt = optimizedData ? Math.round(optimizedData.optimized_score?.clarity * 100) : 0;
  const specificityRaw = optimizedData ? Math.round(optimizedData.raw_score?.specificity * 100) : 0;
  const specificityOpt = optimizedData ? Math.round(optimizedData.optimized_score?.specificity * 100) : 0;

  return (
    <div className="min-h-screen bg-slate-100 animate-in fade-in">
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_420px]">
        {/* ========================================================= */}
        {/* MAIN WORKSPACE (LEFT) */}
        {/* ========================================================= */}
        <main className="min-w-0 pl-20 pr-4 py-6">
          <div className="w-full space-y-4">

            <div className="mb-6">
              <h1 className="text-2xl font-bold mb-1 text-slate-800">Prompt Optimization Demo</h1>
              <p className="text-slate-500 text-sm">Experience real-time prompt engineering with multi-criteria evaluation</p>
            </div>

            {/* 1. RAW PROMPT INPUT */}
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
                  className="w-full min-h-[140px] max-h-[320px] resize-y overflow-auto border border-slate-200 rounded-md p-3 text-sm focus:outline-none focus:border-blue-500 text-slate-700 placeholder:text-slate-400"
                  placeholder="Write a function to calculate factorial..."
                />
              </div>
              {/* Run Optimization Button */}
              <div className="px-4 pb-4">
                <button
                  onClick={handleOptimize}
                  disabled={status === 'loading' || !rawPrompt.trim()}
                  className="w-full bg-slate-200 text-slate-500 px-6 py-2.5 rounded-md font-medium text-sm flex items-center justify-center transition-all focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 data-[active=true]:bg-blue-600 data-[active=true]:text-white data-[active=true]:hover:bg-blue-700"
                  data-active={rawPrompt.trim().length > 0 && status !== 'loading'}
                  aria-live="polite"
                >
                  {status === 'loading' ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Sparkles className="w-4 h-4 mr-2" />}
                  {status === 'loading' ? 'Processing via Qwen...' : 'Run Optimization'}
                </button>
                <label className="mt-2 flex items-center justify-center gap-2 text-xs text-slate-600 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={runBenchmark}
                    onChange={(e) => setRunBenchmark(e.target.checked)}
                    disabled={status === 'loading'}
                    className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                  />
                  Compare with external LLM <span className="text-slate-400">(adds 1–3s)</span>
                </label>
                {status === 'error' && <p className="text-red-500 text-xs font-medium mt-2 text-center">Optimization failed. Try again.</p>}
              </div>
              <div className="bg-blue-50 p-4 border-t border-blue-100 flex items-start text-blue-800 text-xs">
                <ShieldAlert className="w-4 h-4 mr-2 flex-shrink-0 mt-0.5" />
                <p><strong>Privacy Notice:</strong> Do not input sensitive personal data.</p>
              </div>
            </div>

            {/* 2. OPTIMIZED PROMPT OUTPUT */}
            <div ref={optimizedCardRef} className="scroll-mt-6" aria-hidden="true" />
            <div 
              className={`rounded-xl border border-slate-200 bg-white p-0 shadow-sm flex flex-col relative overflow-hidden ${status === 'success' ? 'animate-in slide-in-from-bottom-12 fade-in duration-700 ease-out' : ''}`}
            >
              <div className="p-4 border-b border-slate-100 font-bold text-sm bg-slate-50 flex justify-between items-center">
                Optimized Prompt
                {status === 'success' && optimizedData?.rewrite_metadata?.archetype && (
                  <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wider">
                    {optimizedData.rewrite_metadata.archetype} / {optimizedData.rewrite_metadata.modularity}
                  </span>
                )}
              </div>

              {status === 'idle' && (
                <div className="flex-1 p-4">
                  <textarea className="w-full min-h-[140px] max-h-[360px] resize-y overflow-auto border border-slate-200 rounded-md p-3 text-sm bg-slate-50 text-slate-400" readOnly placeholder="Optimized prompt will appear here..." />
                </div>
              )}

              {status === 'loading' && (
                <div className="flex-1 flex flex-col items-center justify-center p-8 min-h-[200px]">
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
                  <div className="p-4 flex-1 animate-in fade-in">
                    <textarea
                      ref={textareaRef}
                      className="w-full min-h-[140px] resize-none overflow-hidden border border-slate-200 rounded-md p-3 text-sm bg-slate-50 text-slate-700 font-mono focus:outline-none"
                      readOnly
                      value={optimizedData.optimized_prompt}
                      aria-label="Optimized Prompt Output"
                    />
                  </div>
                  <div className="px-4 pb-4 flex gap-2">
                    <button onClick={copyToClipboard} className="flex-1 bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium py-2 rounded-md text-sm transition-colors flex justify-center items-center focus:ring-2 focus:ring-slate-400">
                      {copied ? <CheckCircle2 className="w-4 h-4 mr-2 text-green-600" /> : <FileText className="w-4 h-4 mr-2" />}
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                    <button
                      onClick={() => setFeedback('like')}
                      className={`flex items-center justify-center font-medium px-4 py-2 rounded-md text-sm transition-colors focus:ring-2 focus:ring-slate-400 ${feedback === 'like' ? 'bg-green-100 text-green-700 shadow-inner' :
                          'bg-green-50 text-green-600 hover:bg-green-100'
                        }`}
                      aria-label="Like"
                    >
                      <ThumbsUp className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setFeedback('dislike')}
                      className={`flex items-center justify-center font-medium px-4 py-2 rounded-md text-sm transition-colors focus:ring-2 focus:ring-slate-400 ${feedback === 'dislike' ? 'bg-red-100 text-red-700 shadow-inner' :
                          'bg-red-50 text-red-600 hover:bg-red-100'
                        }`}
                      aria-label="Dislike"
                    >
                      <ThumbsDown className="w-4 h-4" />
                    </button>
                  </div>
                </>
              )}
            </div>

            {/* 3. LLM OUTPUT COMPARISON */}
            <div className="bg-white border border-slate-200 rounded-lg flex flex-col shadow-sm">
              <div className="p-4 border-b border-slate-100 font-bold text-sm bg-slate-50">LLM Output</div>
              <div className="p-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Raw Prompt Output */}
                  <div className="flex flex-col">
                    <div className="text-xs font-semibold text-slate-600 mb-2 flex items-center">
                      <span className="w-2 h-2 rounded-full bg-red-400 mr-2"></span>
                      Raw Prompt Output
                    </div>
                    {status === 'success' && optimizedData?.external_llm_response_raw ? (
                      <div className="bg-slate-50 border border-slate-200 p-3 rounded-md flex-1 font-mono text-xs text-slate-700 overflow-auto min-h-[150px] max-h-[300px] animate-in fade-in">
                        <pre className="whitespace-pre-wrap font-inherit m-0">
                          {optimizedData.external_llm_response_raw}
                        </pre>
                      </div>
                    ) : (
                      <textarea className="w-full flex-1 min-h-[150px] border border-slate-200 rounded-md p-3 text-sm bg-slate-50 text-slate-400 resize-none" readOnly placeholder={status === 'success' ? "Enable 'Compare with external LLM' to view raw prompt output." : "Raw prompt LLM output will appear here..."} />
                    )}
                  </div>

                  {/* Optimized Prompt Output */}
                  <div className="flex flex-col">
                    <div className="text-xs font-semibold text-slate-600 mb-2 flex items-center">
                      <span className="w-2 h-2 rounded-full bg-green-500 mr-2"></span>
                      Optimized Prompt Output
                    </div>
                    {status === 'success' && optimizedData?.external_llm_response_optimized ? (
                      <div className="bg-slate-50 border border-slate-200 p-3 rounded-md flex-1 font-mono text-xs text-slate-700 overflow-auto min-h-[150px] max-h-[300px] animate-in fade-in">
                        <pre className="whitespace-pre-wrap font-inherit m-0">
                          {optimizedData.external_llm_response_optimized}
                        </pre>
                      </div>
                    ) : (
                      <textarea className="w-full flex-1 min-h-[150px] border border-slate-200 rounded-md p-3 text-sm bg-slate-50 text-slate-400 resize-none" readOnly placeholder={status === 'success' ? "Enable 'Compare with external LLM' to view optimized prompt output." : "Optimized prompt LLM output will appear here..."} />
                    )}
                  </div>
                </div>
              </div>
              {status === 'success' && (
                <div className="p-4 border-t border-slate-100 bg-green-50/50 text-green-800 text-xs flex items-center">
                  <Sparkles className="w-4 h-4 mr-2" />
                  <p><strong>Powered by:</strong> External Large Language Model (GPT-4 / Claude / etc.)</p>
                </div>
              )}
            </div>

          </div>
        </main>

        {/* ========================================================= */}
        {/* SIDEBAR (RIGHT) */}
        {/* ========================================================= */}
        <aside className="hidden xl:block xl:sticky xl:top-0 h-screen bg-white border-l border-slate-200 flex flex-col text-slate-600 shadow-sm">
          <div className="h-full overflow-y-auto custom-scrollbar pb-6">

            {/* Header */}
            <div className="px-5 py-4 border-b border-slate-200 bg-white sticky top-0 z-10">
              <h2 className="text-sm font-bold text-slate-800 flex items-center">
                <BarChart3 className="w-4 h-4 mr-2 text-blue-600" />
                Evaluation Report
              </h2>
            </div>

            {status !== 'success' || !optimizedData ? (
              <div className="flex flex-col items-center justify-center p-8 text-center mt-10">
                <div className="w-14 h-14 bg-slate-50 text-slate-400 rounded-full flex items-center justify-center mx-auto mb-4 border border-slate-100">
                  <BarChart3 className="w-7 h-7" />
                </div>
                <h3 className="text-sm font-bold mb-1 text-slate-700">
                  {status === 'loading' ? 'Evaluating...' : 'Awaiting Results'}
                </h3>
                <p className="text-slate-500 text-xs max-w-[200px]">
                  {status === 'loading'
                    ? 'Analyzing prompt quality metrics...'
                    : 'Run optimization to see the evaluation report.'}
                </p>
                {status === 'loading' && <Loader2 className="w-5 h-5 text-blue-500 animate-spin mt-4" />}
              </div>
            ) : (
              <div className="animate-in fade-in">

                {/* Score Comparison */}
                <section className="px-5 py-4 border-b border-slate-200">
                  <h4 className="text-[10px] font-semibold text-slate-500 mb-3 uppercase tracking-wider">Overall Score Comparison</h4>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-red-50 border border-red-100 rounded-lg p-3 text-center">
                      <div className="text-[10px] font-semibold text-red-500 uppercase tracking-wider mb-1">Raw Prompt</div>
                      <div className="text-3xl font-extrabold text-red-500">{rawOverall}%</div>
                    </div>
                    <div className="bg-green-50 border border-green-100 rounded-lg p-3 text-center">
                      <div className="text-[10px] font-semibold text-green-600 uppercase tracking-wider mb-1">Optimized</div>
                      <div className="text-3xl font-extrabold text-green-600">{optOverall}%</div>
                    </div>
                  </div>
                </section>

                {/* Metrics Snapshot */}
                <section className="px-5 py-4 border-b border-slate-200">
                  <h4 className="text-[10px] font-semibold text-slate-500 mb-3 uppercase tracking-wider">Metrics Snapshot</h4>
                  <div className="space-y-4">
                    {[
                      { label: 'Clarity', raw: clarityRaw, opt: clarityOpt },
                      { label: 'Specificity', raw: specificityRaw, opt: specificityOpt },
                    ].map(metric => (
                      <div key={metric.label} className="bg-slate-50 border border-slate-100 rounded-lg p-3">
                        <div className="text-xs font-semibold text-slate-700 mb-2">{metric.label}</div>
                        {/* Combined bar visualization */}
                        <div className="relative h-4 w-full bg-slate-200 rounded-full overflow-hidden">
                          {/* Raw bar (red) */}
                          <div
                            className="absolute top-0 left-0 h-full bg-red-400 rounded-full transition-all duration-700 ease-out"
                            style={{ width: `${metric.raw}%`, zIndex: metric.raw <= metric.opt ? 2 : 1 }}
                          ></div>
                          {/* Optimized bar (green) */}
                          <div
                            className="absolute top-0 left-0 h-full bg-green-500 rounded-full transition-all duration-700 ease-out"
                            style={{ width: `${metric.opt}%`, zIndex: metric.opt < metric.raw ? 2 : 1 }}
                          ></div>
                        </div>
                        {/* Labels */}
                        <div className="flex justify-between mt-1.5">
                          <span className="text-[10px] font-medium text-red-500">{metric.raw}%</span>
                          <span className="text-[10px] font-medium text-green-600">{metric.opt}%</span>
                        </div>
                      </div>
                    ))}
                    {/* Legend */}
                    <div className="flex justify-center space-x-4 pt-1">
                      <div className="flex items-center text-[10px] text-slate-500">
                        <span className="w-2.5 h-2.5 rounded-full bg-red-400 mr-1.5"></span> Raw
                      </div>
                      <div className="flex items-center text-[10px] text-slate-500">
                        <span className="w-2.5 h-2.5 rounded-full bg-green-500 mr-1.5"></span> Optimized
                      </div>
                    </div>
                  </div>
                </section>

                {/* Key Improvements */}
                <section className="px-5 py-4 border-b border-slate-200">
                  <h4 className="text-[10px] font-semibold text-slate-500 mb-3 uppercase tracking-wider">Key Improvements</h4>
                  <div className="bg-slate-50 border border-slate-100 rounded-lg p-3">
                    <ul className="space-y-2">
                      {optimizedData.actions?.map((act: string, i: number) => (
                        <li key={i} className="flex items-start text-xs text-slate-600">
                          <Zap className="w-3.5 h-3.5 mr-2 text-amber-500 flex-shrink-0 mt-0.5" />
                          {act}
                        </li>
                      ))}
                      {optimizedData.improvement_score > 0 && (
                        <li className="flex items-start text-xs text-slate-600">
                          <TrendingUp className="w-3.5 h-3.5 mr-2 text-green-500 flex-shrink-0 mt-0.5" />
                          Overall quality improved by +{Math.round(optimizedData.improvement_score * 100)}%
                        </li>
                      )}
                    </ul>
                  </div>
                </section>

                {/* Detected Issues Dropdown */}
                <SidebarSection
                  title="Detected Issues"
                  count={(optimizedData.issues ?? []).length}
                >
                  <div className="space-y-4">
                    {/* Highlighted raw prompt preview */}
                    <div>
                      <h4 className="text-[10px] font-bold text-slate-500 mb-2 uppercase tracking-wider">Diagnostics Preview</h4>
                      <div className="rounded-md border border-slate-100 bg-slate-50 p-3 text-slate-700">
                        <PromptHighlighter
                          text={optimizedData.raw_prompt}
                          issues={optimizedData.issues ?? []}
                          theme="light"
                        />
                      </div>
                      {optimizedData.validation && optimizedData.validation.status === 'invalid' && (
                        <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3 text-xs text-red-800">
                          <strong className="text-red-700">Rewrite validation flagged issues:</strong>{' '}
                          {optimizedData.validation.issues.map((i) => i.type).join(', ')}.
                          Falling back to the raw prompt where applicable.
                        </div>
                      )}
                    </div>

                    {/* Issue List */}
                    <div>
                      <h4 className="text-[10px] font-bold text-slate-500 mb-2 uppercase tracking-wider">Details</h4>
                      <IssuePanel issues={optimizedData.issues ?? []} theme="light" />
                    </div>
                  </div>
                </SidebarSection>

                {/* Tutor Guidance Dropdown */}
                <SidebarSection title="Tutor Guidance" defaultOpen={false}>
                  <RecommendationPanel
                    recommendations={optimizedData.recommendations ?? []}
                    institutionalGuideline={optimizedData.institutional_guideline}
                    theme="light"
                  />
                </SidebarSection>

              </div>
            )}
          </div>
        </aside>

      </div>
    </div>
  );
}