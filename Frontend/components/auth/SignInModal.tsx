import React, { useState } from 'react';
import { Loader2, X } from 'lucide-react';

export default function SignInModal({ onClose, onSuccess }: { onClose: () => void, onSuccess: (user: { id: number, username: string }) => void }) {
  const [isSignUp, setIsSignUp] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [errorMsg, setErrorMsg] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('loading');
    setErrorMsg('');

    if (password.length < 6) {
      setStatus('error');
      setErrorMsg('Password must be at least 6 characters.');
      return;
    }

    try {
      if (isSignUp) {
        const res = await fetch('http://127.0.0.1:8000/signup', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Sign up failed');
        
        const signInRes = await fetch('http://127.0.0.1:8000/signin', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        });
        const signInData = await signInRes.json();
        if (!signInRes.ok) throw new Error(signInData.detail || 'Sign in failed');
        
        setStatus('idle');
        onSuccess({ id: signInData.user_id, username: signInData.username });
      } else {
        const res = await fetch('http://127.0.0.1:8000/signin', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Sign in failed');
        
        setStatus('idle');
        onSuccess({ id: data.user_id, username: data.username });
      }
    } catch (err: any) {
      setStatus('error');
      setErrorMsg(err.message);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center z-[100] p-4" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div className="bg-white rounded-xl w-full max-w-sm p-6 shadow-2xl relative animate-in zoom-in-95 duration-200">
        <button onClick={onClose} className="absolute top-4 right-4 p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-full transition-colors" aria-label="Close modal">
          <X className="w-5 h-5" />
        </button>

        <h2 id="modal-title" className="text-xl font-bold mb-6 text-slate-900">{isSignUp ? 'Create an Account' : 'Sign In to Promptee'}</h2>

        <form onSubmit={handleSubmit} className="space-y-4" noValidate>
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-slate-700 mb-1">Username</label>
            <input
              id="username"
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Your username"
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

          <div aria-live="assertive" className="min-h-[20px]">
            {status === 'error' && <p className="text-sm text-red-600 font-medium">{errorMsg}</p>}
          </div>

          <button
            type="submit"
            disabled={status === 'loading' || !username || !password}
            className="w-full bg-blue-600 text-white py-2.5 rounded-md hover:bg-blue-700 font-medium mt-2 disabled:bg-blue-400 disabled:cursor-not-allowed flex items-center justify-center transition-colors focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            {status === 'loading' ? <Loader2 className="w-5 h-5 animate-spin" /> : (isSignUp ? 'Sign Up' : 'Continue')}
          </button>

          <div className="text-center text-sm text-slate-500 mt-4">
            {isSignUp ? 'Already have an account? ' : 'Need an account? '}
            <button type="button" onClick={() => setIsSignUp(!isSignUp)} className="text-blue-600 font-medium hover:underline">
              {isSignUp ? 'Sign In' : 'Sign Up'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}