'use client';

import React, { useState, useEffect, useCallback } from 'react';
import OverviewTab from '@/components/tabs/OverviewTab';
import DemoTab from '@/components/tabs/DemoTab';
import PipelineTab from '@/components/tabs/PipelineTab';
import AboutTab from '@/components/tabs/AboutTab';

import ResultsTab from '@/components/tabs/ResultsTab';
import SignInModal from '@/components/auth/SignInModal';
import SplashScreen from '@/components/SplashScreen';
import type { PromptIssue, RewriteMetadata, ValidationResult } from '@/types/prompt';

export interface ScoreResult {
  raw_quality: number;
  candidate_quality: number;
  clarity: number;
  specificity: number;
  ambiguity_penalty: number;
  redundancy_penalty: number;
  length_penalty: number;
  structural_bonus: number;
  quality_improvement: number;
  clarity_delta: number;
  specificity_delta: number;
  semantic_preservation: number;
  rejected: boolean;
  final_score: number;
}

export interface OptimizationResponse {
  raw_prompt: string;
  optimized_prompt: string;
  raw_score: ScoreResult;
  optimized_score: ScoreResult;
  external_llm_response_raw: string;
  external_llm_response_optimized: string;
  improvement_score: number;
  rewrite_metadata: RewriteMetadata;
  issues: PromptIssue[];
  recommendations: string[];
  institutional_guideline: string;
  validation: ValidationResult;
}

export interface OptimizedData extends OptimizationResponse {
  actions: string[];
}

export default function PrompteeApp() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [user, setUser] = useState<{ email: string } | null>(null);
  const [showSignIn, setShowSignIn] = useState(false);
  const [isInitializing, setIsInitializing] = useState(true);
  const [showSplash, setShowSplash] = useState(true);
  const [optimizedData, setOptimizedData] = useState<OptimizedData | null>(null);

  // Simulated Backend: Check LocalStorage on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('promptee_user');
    if (savedUser) setUser(JSON.parse(savedUser));
    setIsInitializing(false);
  }, []);

  const handleSignOut = () => {
    localStorage.removeItem('promptee_user');
    setUser(null);
    setActiveTab('Overview');
  };

  const handleSplashComplete = useCallback(() => {
    setShowSplash(false);
  }, []);

  // Show splash screen during startup
  if (showSplash || isInitializing) {
    return <SplashScreen onComplete={handleSplashComplete} />;
  }

  const tabs = ['Overview', 'Demo', 'Pipeline', 'About'];
  if (user) tabs.splice(2, 0, 'Results');

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <nav className="flex flex-wrap items-center justify-between px-6 py-4 bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="text-2xl font-bold text-blue-600">Promptee</div>

        {/* Mobile-responsive tab scrolling */}
        <div className="flex space-x-6 overflow-x-auto w-full md:w-auto order-3 md:order-2 mt-4 md:mt-0 pb-2 md:pb-0 scrollbar-hide">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              aria-current={activeTab === tab ? 'page' : undefined}
              className={`text-sm font-medium transition-colors whitespace-nowrap ${activeTab === tab ? 'text-blue-600 border-b-2 border-blue-600' : 'text-slate-500 hover:text-slate-900'
                }`}
            >
              {tab}
            </button>
          ))}
        </div>

        <div className="order-2 md:order-3">
          {user ? (
            <div className="flex items-center space-x-4 text-sm">
              <span className="flex items-center text-slate-600 bg-slate-100 px-3 py-1.5 rounded-md" aria-label="Logged in user">
                <span className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-pulse"></span> {user.email.split('@')[0]}
              </span>
              <button onClick={handleSignOut} className="border border-slate-300 px-4 py-1.5 rounded-md hover:bg-slate-50 focus:ring-2 focus:ring-blue-500">Sign Out</button>
            </div>
          ) : (
            <button onClick={() => setShowSignIn(true)} className="bg-blue-600 text-white px-5 py-2 rounded-md font-medium hover:bg-blue-700 transition-colors focus:ring-2 focus:ring-blue-500 focus:ring-offset-2">
              Sign In
            </button>
          )}
        </div>
      </nav>

      <main className="pb-20 focus:outline-none" tabIndex={-1}>
        {activeTab === 'Overview' && <OverviewTab onTryDemo={() => setActiveTab('Demo')} />}
        {activeTab === 'Demo' && <DemoTab user={user} onSignIn={() => setShowSignIn(true)} optimizedData={optimizedData} setOptimizedData={setOptimizedData} />}

        {activeTab === 'Results' && user && <ResultsTab />}
        {activeTab === 'Pipeline' && <PipelineTab />}
        {activeTab === 'About' && <AboutTab />}
      </main>

      {showSignIn && (
        <SignInModal
          onClose={() => setShowSignIn(false)}
          onSuccess={(userData) => {
            setUser(userData);
            localStorage.setItem('promptee_user', JSON.stringify(userData));
            setShowSignIn(false);
          }}
        />
      )}
    </div>
  );
}