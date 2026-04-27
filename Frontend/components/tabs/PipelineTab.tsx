import React, { useState } from 'react';
import { Monitor, ArrowRight, Database, BarChart3, Cpu, Sparkles, ChevronDown, ChevronUp } from 'lucide-react';

export default function PipelineTab() {
  // State to manage accordion open/close
  const [isOfflineOpen, setIsOfflineOpen] = useState(false);
  const [isInferenceOpen, setIsInferenceOpen] = useState(false);

  return (
    <div className="max-w-6xl mx-auto px-6 py-8 animate-in fade-in">
      <h1 className="text-2xl font-bold mb-2">Pipeline & System Architecture</h1>
      <p className="text-slate-500 text-sm mb-8">Visual mapping of thesis concepts to implementation modules</p>

      {/* System Flow Diagram */}
      <div className="bg-white border border-slate-200 rounded-xl p-8 mb-6 shadow-sm">
        <h3 className="font-bold mb-6 text-slate-700">System Flow</h3>
        <div className="flex flex-col lg:flex-row items-center justify-between space-y-4 lg:space-y-0 lg:space-x-4">
          <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg text-center w-full lg:w-48 h-40 flex flex-col justify-center items-center">
            <Monitor className="w-6 h-6 text-blue-500 mb-2" />
            <div className="font-bold text-sm mb-1">UI (Presentation)</div>
            <div className="text-xs text-slate-500">React-based interface</div>
          </div>
          <ArrowRight className="text-slate-300 hidden lg:block" />
          <div className="bg-green-50 border border-green-200 p-6 rounded-lg text-center w-full lg:w-48 h-40 flex flex-col justify-center items-center">
            <Database className="w-6 h-6 text-green-500 mb-2" />
            <div className="font-bold text-sm mb-1">FastAPI Backend</div>
            <div className="text-xs text-slate-500">RESTful API handling</div>
          </div>
          <ArrowRight className="text-slate-300 hidden lg:block" />
          <div className="bg-purple-50 border border-purple-200 p-6 rounded-lg text-center w-full lg:w-48 h-40 flex flex-col justify-center items-center">
            <BarChart3 className="w-6 h-6 text-purple-500 mb-2" />
            <div className="font-bold text-sm mb-1">Heuristic Scoring</div>
            <div className="text-xs text-slate-500">MCDM evaluation</div>
          </div>
          <ArrowRight className="text-slate-300 hidden lg:block" />
          <div className="bg-yellow-50 border border-yellow-200 p-6 rounded-lg text-center w-full lg:w-48 h-40 flex flex-col justify-center items-center shadow-md border-b-4 border-yellow-400">
            <Cpu className="w-6 h-6 text-yellow-600 mb-2" />
            <div className="font-bold text-sm mb-1">Optimization Engine</div>
            <div className="text-xs text-slate-600">Qwen + LoRA fine-tuned</div>
          </div>
          <ArrowRight className="text-slate-300 hidden lg:block" />
          <div className="bg-red-50 border border-red-200 p-6 rounded-lg text-center w-full lg:w-48 h-40 flex flex-col justify-center items-center">
            <Sparkles className="w-6 h-6 text-red-500 mb-2" />
            <div className="font-bold text-sm mb-1">External LLM</div>
            <div className="text-xs text-slate-500">Output generation</div>
          </div>
        </div>
      </div>

      {/* Expandable Accordions */}
      <div className="space-y-4 mb-8">

        {/* Offline Training Module */}
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden transition-all duration-200">
          <div
            onClick={() => setIsOfflineOpen(!isOfflineOpen)}
            className="p-4 flex justify-between items-center cursor-pointer hover:bg-slate-50 select-none"
          >
            <span className="font-bold text-slate-800">Offline Training Module</span>
            {isOfflineOpen ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
          </div>

          {isOfflineOpen && (
            <div className="p-6 border-t border-slate-100 bg-white space-y-6 animate-in slide-in-from-top-2 duration-200">
              <div>
                <h4 className="font-medium text-slate-900 mb-2">Quantized Low-Rank Adaptation (QLoRA)</h4>
                <p className="text-sm text-slate-600 leading-relaxed">Efficient fine-tuning using 4-bit NF4 quantization to reduce memory footprint while maintaining model performance.</p>
              </div>
              <div>
                <h4 className="font-medium text-slate-900 mb-2">Direct Preference Optimization (DPO)</h4>
                <p className="text-sm text-slate-600 leading-relaxed">Aligns the model with human preferences by directly optimizing the policy to prefer better prompt structures over worse ones, without requiring a separate reward model.</p>
              </div>
              <div>
                <h4 className="font-medium text-slate-900 mb-2">Training Dataset</h4>
                <p className="text-sm text-slate-600 leading-relaxed">Curated preference pairs showing examples of raw vs. optimized prompts with quality annotations across multiple criteria.</p>
              </div>
            </div>
          )}
        </div>

        {/* Inference-Time Evaluation Module */}
        <div className="bg-white border border-slate-200 rounded-lg overflow-hidden transition-all duration-200">
          <div
            onClick={() => setIsInferenceOpen(!isInferenceOpen)}
            className="p-4 flex justify-between items-center cursor-pointer hover:bg-slate-50 select-none"
          >
            <span className="font-bold text-slate-800">Inference-Time Evaluation Module</span>
            {isInferenceOpen ? <ChevronUp className="w-5 h-5 text-slate-400" /> : <ChevronDown className="w-5 h-5 text-slate-400" />}
          </div>

          {isInferenceOpen && (
            <div className="p-6 border-t border-slate-100 bg-white space-y-6 animate-in slide-in-from-top-2 duration-200">
              <div>
                <h4 className="font-medium text-slate-900 mb-2">Real-Time Heuristic Scoring</h4>
                <p className="text-sm text-slate-600 leading-relaxed">Evaluates incoming prompts using predefined rubrics for clarity, specificity, structure, and semantic preservation.</p>
              </div>
              <div>
                <h4 className="font-medium text-slate-900 mb-2">Weighted Sum Model (WSM)</h4>
                <p className="text-sm text-slate-600 leading-relaxed">Combines individual criterion scores using configurable weights to produce an overall quality score (Q).</p>
              </div>
              <div>
                <h4 className="font-medium text-slate-900 mb-2">Diff Generation</h4>
                <p className="text-sm text-slate-600 leading-relaxed">Highlights specific changes made during optimization with explanations for transparency.</p>
              </div>
            </div>
          )}
        </div>

      </div>

      {/* Architecture Rationale */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
        <h4 className="font-bold text-yellow-800 mb-2">Why This Architecture?</h4>
        <p className="text-sm text-yellow-700 leading-relaxed">
          Evaluators need to see the separation between inference-time heuristic evaluation and the offline parameter-space adaptation that consolidates prompt-derived gains. This architecture ensures that improvements are not just heuristic-based rewrites, but are grounded in learned preferences from the fine-tuned model, creating a robust optimization pipeline.
        </p>
      </div>
    </div>
  );
}