import React from 'react';
import { ArrowRight, AlertTriangle, Sparkles, FileText, BarChart3 } from 'lucide-react';

export default function OverviewTab({ onTryDemo }: { onTryDemo: () => void }) {
  return (
    <div className="animate-in fade-in duration-500">
      {/* Hero Section */}
      <div className="bg-blue-600 text-white py-24 text-center px-4">
        <h1 className="text-4xl md:text-5xl font-bold max-w-4xl mx-auto leading-tight mb-6">
          Prompt Optimization Pipeline Using Multi-Criteria Heuristic Evaluation and Transformer Fine-Tuning Algorithms
        </h1>
        <p className="text-blue-100 text-lg max-w-2xl mx-auto mb-8">
          Mitigating prompt brittleness through structured evaluation and Direct Preference Optimization (DPO)
        </p>
        <button onClick={onTryDemo} className="bg-white text-blue-600 px-6 py-3 rounded-md font-medium flex items-center mx-auto hover:bg-blue-50 transition-colors">
          Try the Demo <ArrowRight className="ml-2 w-4 h-4" />
        </button>
      </div>

      {/* How It Works Section */}
      <div className="max-w-6xl mx-auto px-6 py-16">
        <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0 md:space-x-4">
          {[
            { id: 1, title: 'Input Raw Prompt', desc: 'User provides initial prompt' },
            { id: 2, title: 'Multi-Criteria Heuristic Evaluation (Q_raw)', desc: 'Evaluate prompt using MCDM methods' },
            { id: 3, title: 'Transformation via LoRA/DPO', desc: 'Fine-tuned Qwen Model optimizes prompt' },
            { id: 4, title: 'Post-Evaluation (Q_opt)', desc: 'Evaluate optimized prompt quality' },
            { id: 5, title: 'LLM Output Generation', desc: 'Generate improved output using external LLM' }
          ].map((step, idx) => (
            <React.Fragment key={step.id}>
              <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm flex-1 w-full text-center md:text-left">
                <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold mb-4 mx-auto md:mx-0">{step.id}</div>
                <h3 className="font-bold text-sm mb-2">{step.title}</h3>
                <p className="text-xs text-slate-500">{step.desc}</p>
              </div>
              {idx < 4 && <ArrowRight className="text-slate-300 hidden md:block w-6 h-6 flex-shrink-0" />}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Why It Matters Section */}
      <div className="bg-slate-100 py-16">
        <div className="max-w-4xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-12">Why It Matters</h2>
          <div className="bg-white rounded-xl p-8 border border-slate-200 shadow-sm space-y-8">
            <div className="flex items-start">
              <div className="bg-orange-100 p-3 rounded-lg mr-4"><AlertTriangle className="text-orange-600 w-6 h-6" /></div>
              <div>
                <h3 className="font-bold text-lg mb-2">The Problem: Prompt Brittleness</h3>
                <p className="text-slate-600 text-sm">The widespread adoption of AI often encounters repetitive or inaccurate outputs due to "prompt brittleness". Small variations in prompt wording can lead to drastically different results.</p>
              </div>
            </div>
            <div className="flex items-start">
              <div className="bg-green-100 p-3 rounded-lg mr-4"><Sparkles className="text-green-600 w-6 h-6" /></div>
              <div>
                <h3 className="font-bold text-lg mb-2">The Solution: Structured Evaluation</h3>
                <p className="text-slate-600 text-sm">This pipeline introduces structured evaluation signals to measurably guide prompt improvement, reducing reliance on ad hoc trial-and-error.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Key Features Section (Newly Added) */}
      <div className="bg-white py-20">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-center mb-16 text-slate-900">Key Features</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-10 text-center">
            {/* Feature 1 */}
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mb-6">
                <FileText className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-3">Multi-Criteria Evaluation</h3>
              <p className="text-sm text-slate-600 leading-relaxed max-w-xs mx-auto">
                Assess prompts across Clarity, Specificity, Structural Completeness, and Semantic Preservation
              </p>
            </div>

            {/* Feature 2 */}
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-purple-50 text-purple-600 rounded-full flex items-center justify-center mb-6">
                <Sparkles className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-3">Transformer Fine-Tuning</h3>
              <p className="text-sm text-slate-600 leading-relaxed max-w-xs mx-auto">
                Leveraging QLoRA and DPO for efficient parameter-space adaptation
              </p>
            </div>

            {/* Feature 3 */}
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 bg-green-50 text-green-600 rounded-full flex items-center justify-center mb-6">
                <BarChart3 className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-slate-900 mb-3">Transparent Metrics</h3>
              <p className="text-sm text-slate-600 leading-relaxed max-w-xs mx-auto">
                Reproducible results with BLEU and ROUGE-L scoring for output integrity
              </p>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}