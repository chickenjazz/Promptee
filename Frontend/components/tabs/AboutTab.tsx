import React from 'react';
import { FileText, Code, Target, AlertTriangle, CheckCircle2 } from 'lucide-react';

export default function AboutTab() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 animate-in fade-in space-y-8">

      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-1 text-slate-900">About Promptee</h1>
        <p className="text-slate-500 text-sm">Documentation, methodology, and software quality parameters</p>
      </div>

      {/* Abstract */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex items-center mb-6">
          <div className="bg-blue-50 text-blue-600 p-2.5 rounded-lg mr-4">
            <FileText className="w-5 h-5" />
          </div>
          <h2 className="font-bold text-lg text-slate-900">Abstract</h2>
        </div>
        <div className="text-sm text-slate-600 space-y-4 leading-relaxed">
          <p>Promptee is a prompt optimization pipeline that addresses the critical challenge of <em>prompt brittleness</em> in large language model (LLM) applications. By combining multi-criteria heuristic evaluation with transformer fine-tuning algorithms, this system provides a structured, reproducible approach to prompt engineering.</p>
          <p>The system evaluates prompts across four key dimensions: Clarity, Specificity, Structural Completeness, and Semantic Preservation. Using Quantized Low-Rank Adaptation (QLoRA) and Direct Preference Optimization (DPO), a fine-tuned Qwen model learns to transform raw prompts into optimized versions that consistently produce better LLM outputs.</p>
          <p>This research addresses the widespread adoption of AI in the Philippines, where repetitive or inaccurate outputs hinder productivity. By providing measurable quality metrics (BLEU, ROUGE-L) and transparent evaluation criteria, Promptee offers a scientifically grounded alternative to ad hoc trial-and-error prompt development.</p>
        </div>
      </div>

      {/* Key Definitions */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h2 className="font-bold text-lg mb-6 text-slate-900">Key Definitions</h2>
        <div className="space-y-5">
          <div>
            <div className="text-sm text-slate-800 mb-1">Prompt Brittleness</div>
            <div className="text-sm text-slate-500">The tendency for small variations in prompt wording to produce drastically different or degraded outputs from language models.</div>
          </div>
          <div>
            <div className="text-sm text-slate-800 mb-1">Multi-Criteria Decision Making (MCDM)</div>
            <div className="text-sm text-slate-500">A systematic approach to evaluating alternatives based on multiple, often conflicting criteria with explicit weighting.</div>
          </div>
          <div>
            <div className="text-sm text-slate-800 mb-1">Heuristic Evaluation</div>
            <div className="text-sm text-slate-500">Rule-based assessment using predefined criteria and rubrics to score prompt quality objectively.</div>
          </div>
          <div>
            <div className="text-sm text-slate-800 mb-1">Direct Preference Optimization (DPO)</div>
            <div className="text-sm text-slate-500">A training technique that aligns language models to human preferences by directly optimizing the policy without requiring a separate reward model.</div>
          </div>
          <div>
            <div className="text-sm text-slate-800 mb-1">GIGO Theory</div>
            <div className="text-sm text-slate-500">"Garbage In, Garbage Out" - the principle that low-quality inputs inevitably produce low-quality outputs, necessitating rigorous input validation.</div>
          </div>
        </div>
      </div>

      {/* Core Methodologies */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex items-center mb-6">
          <div className="bg-purple-50 text-purple-600 p-2.5 rounded-lg mr-4">
            <Code className="w-5 h-5" />
          </div>
          <h2 className="font-bold text-lg text-slate-900">Core Methodologies</h2>
        </div>
        <div className="space-y-6">
          {[
            { title: 'Multi-Criteria Decision Making (MCDM)', desc: 'Weighted Sum Model applied to four evaluation criteria to produce aggregate quality scores' },
            { title: 'Quantized Low-Rank Adaptation (QLoRA)', desc: 'Efficient fine-tuning of large language models using 4-bit quantization and low-rank matrices' },
            { title: 'Direct Preference Optimization (DPO)', desc: 'Alignment technique that directly optimizes model policy based on human preference data' },
            { title: 'Transformer Architecture', desc: 'Qwen-based model fine-tuned specifically for prompt engineering tasks' },
          ].map((item, i) => (
            <div key={i} className="pl-4 border-l-4 border-blue-500">
              <div className="font-medium text-sm text-slate-900 mb-1">{item.title}</div>
              <div className="text-sm text-slate-500">{item.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Software Quality & Evaluation Frameworks */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex items-center mb-6">
          <div className="bg-emerald-50 text-emerald-600 p-2.5 rounded-lg mr-4">
            <Target className="w-5 h-5" />
          </div>
          <h2 className="font-bold text-lg text-slate-900">Software Quality & Evaluation Frameworks</h2>
        </div>

        <div className="space-y-6">
          <div>
            <h3 className="font-bold text-slate-900 mb-3 text-sm">ISO 25010</h3>
            <ul className="space-y-2">
              <li className="flex items-start text-sm text-slate-500"><CheckCircle2 className="w-4 h-4 text-emerald-500 mr-2 flex-shrink-0 mt-0.5" /> Functional Suitability - Does the system optimize prompts effectively?</li>
              <li className="flex items-start text-sm text-slate-500"><CheckCircle2 className="w-4 h-4 text-emerald-500 mr-2 flex-shrink-0 mt-0.5" /> Usability - Is the interface accessible and intuitive?</li>
              <li className="flex items-start text-sm text-slate-500"><CheckCircle2 className="w-4 h-4 text-emerald-500 mr-2 flex-shrink-0 mt-0.5" /> Performance Efficiency - Does optimization complete in reasonable time?</li>
            </ul>
          </div>
          <div>
            <h3 className="font-bold text-slate-900 mb-3 text-sm">Technology Acceptance Model (TAM)</h3>
            <ul className="space-y-2">
              <li className="flex items-start text-sm text-slate-500"><CheckCircle2 className="w-4 h-4 text-emerald-500 mr-2 flex-shrink-0 mt-0.5" /> Perceived Usefulness - Do users believe it improves their prompts?</li>
              <li className="flex items-start text-sm text-slate-500"><CheckCircle2 className="w-4 h-4 text-emerald-500 mr-2 flex-shrink-0 mt-0.5" /> Perceived Ease of Use - Is the system learnable and approachable?</li>
              <li className="flex items-start text-sm text-slate-500"><CheckCircle2 className="w-4 h-4 text-emerald-500 mr-2 flex-shrink-0 mt-0.5" /> Intention to Use - Would users integrate this into their workflow?</li>
            </ul>
          </div>
        </div>
      </div>

      {/* System Limitations & Scope */}
      <div className="bg-[#fffdf2] border border-amber-200 rounded-xl p-6 shadow-sm">
        <div className="flex items-center mb-6">
          <div className="bg-amber-100 text-amber-700 p-2.5 rounded-lg mr-4">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <h2 className="font-bold text-lg text-slate-900">System Limitations & Scope</h2>
        </div>
        <ul className="space-y-4">
          {[
            'System is confined to domain-specific datasets and may not generalize to all prompt types',
            'Training is optimized for Windows environments with CUDA support',
            'Requires substantial computational resources for fine-tuning (GPU with ≥16GB VRAM recommended)',
            'Quality improvements depend on the diversity and quality of the preference pair dataset',
            'Not suitable for real-time applications requiring sub-second response times'
          ].map((item, idx) => (
            <li key={idx} className="flex items-start text-sm text-amber-800">
              <span className="w-5 h-5 bg-amber-500 text-white rounded-full flex items-center justify-center text-[10px] font-bold mr-3 flex-shrink-0 mt-0.5">
                {idx + 1}
              </span>
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* Technical Stack */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h2 className="font-bold text-lg mb-6 text-slate-900">Technical Stack</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

          <div>
            <h3 className="font-semibold text-slate-900 mb-3 text-sm">Frontend</h3>
            <ul className="space-y-2 text-sm text-slate-500">
              <li>• React 18 with TypeScript</li>
              <li>• Tailwind CSS for styling</li>
              <li>• Recharts for data visualization</li>
              <li>• React Router for navigation</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-slate-900 mb-3 text-sm">Backend</h3>
            <ul className="space-y-2 text-sm text-slate-500">
              <li>• FastAPI (Python 3.10+)</li>
              <li>• Hugging Face Transformers</li>
              <li>• PyTorch with CUDA support</li>
              <li>• Qwen model with QLoRA/DPO</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-slate-900 mb-3 text-sm">Evaluation</h3>
            <ul className="space-y-2 text-sm text-slate-500">
              <li>• NLTK for BLEU scoring</li>
              <li>• Rouge for ROUGE-L metrics</li>
              <li>• Custom MCDM implementation</li>
              <li>• NumPy for numerical computation</li>
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-slate-900 mb-3 text-sm">Infrastructure</h3>
            <ul className="space-y-2 text-sm text-slate-500">
              <li>• Windows 10/11 (CUDA 12.x)</li>
              <li>• NVIDIA GPU (≥16GB VRAM)</li>
              <li>• Docker for containerization</li>
              <li>• Git for version control</li>
            </ul>
          </div>

        </div>
      </div>

      {/* Citation Box */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 shadow-sm">
        <h2 className="font-bold text-lg mb-4 text-slate-900">Citation</h2>
        <div className="bg-white border border-slate-200 p-5 rounded-md text-sm font-mono text-slate-600 italic">
          [Author Name]. (2026). Promptee: Prompt Optimization Pipeline Using Multi-Criteria Heuristic Evaluation and Transformer Fine-Tuning Algorithms. [Institution Name], Philippines.
        </div>
      </div>

    </div>
  );
}