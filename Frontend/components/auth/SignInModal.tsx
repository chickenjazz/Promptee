import React, { useState } from 'react';
import { Loader2, X } from 'lucide-react';

export default function SignInModal({ onClose, onSuccess }: { onClose: () => void, onSuccess: (user: { email: string }) => void }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  // --- Backend Simulation Logic ---
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('loading');
    setErrorMsg('');

    // Front-end validation
    if (password.length < 6) {
      setStatus('error');
      setErrorMsg('Password must be at least 6 characters.');
      return;
    }

    // Simulate Network Request
    setTimeout(() => {
      setStatus('idle');
      onSuccess({ email }); // Automatically "logs in" the user
    }, 1200);
  };

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-[100] p-4" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div className="bg-white rounded-xl w-full max-w-sm p-6 shadow-2xl relative animate-in zoom-in-95 duration-200">
        <button onClick={onClose} className="absolute top-4 right-4 p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-full transition-colors" aria-label="Close modal">
          <X className="w-5 h-5" />
        </button>

        <h2 id="modal-title" className="text-xl font-bold mb-6 text-slate-900">Sign In to Promptee</h2>

        <form onSubmit={handleSubmit} className="space-y-4" noValidate>
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-slate-700 mb-1">Email Address</label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
              className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow"
              aria-invalid={status === 'error' ? 'true' : 'false'}
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-slate-700 mb-1">Password</label>
            <input
              id="password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full border border-slate-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-shadow"
            />
          </div>

          {/* Accessible Error Announcer */}
          <div aria-live="assertive" className="min-h-[20px]">
            {status === 'error' && <p className="text-sm text-red-600 font-medium">{errorMsg}</p>}
          </div>

          <button
            type="submit"
            disabled={status === 'loading' || !email || !password}
            className="w-full bg-blue-600 text-white py-2.5 rounded-md hover:bg-blue-700 font-medium mt-2 disabled:bg-blue-400 disabled:cursor-not-allowed flex items-center justify-center transition-colors focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            {status === 'loading' ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Continue'}
          </button>

          <div className="text-center text-sm text-slate-500 mt-4">
            For demo purposes, any email/password works.
          </div>
        </form>
      </div>
    </div>
  );
}