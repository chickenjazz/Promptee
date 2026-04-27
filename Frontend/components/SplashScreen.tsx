'use client';

import React, { useState, useEffect, useCallback } from 'react';

const TARGET_WORD = 'Promptee';
const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';

interface OdometerLetterProps {
    targetChar: string;
    delay: number;
    cycleDuration: number;
    onSettled?: () => void;
}

function OdometerLetter({ targetChar, delay, cycleDuration, onSettled }: OdometerLetterProps) {
    const [displayChar, setDisplayChar] = useState(' ');
    const [phase, setPhase] = useState<'waiting' | 'cycling' | 'settling' | 'done'>('waiting');
    const [offset, setOffset] = useState(0);

    useEffect(() => {
        const waitTimer = setTimeout(() => {
            setPhase('cycling');
        }, delay);
        return () => clearTimeout(waitTimer);
    }, [delay]);

    useEffect(() => {
        if (phase !== 'cycling') return;

        const interval = setInterval(() => {
            const randomChar = CHARS[Math.floor(Math.random() * CHARS.length)];
            setDisplayChar(randomChar);
            setOffset(Math.random() > 0.5 ? 1 : -1);
        }, 30);

        const settleTimer = setTimeout(() => {
            clearInterval(interval);
            setPhase('settling');
        }, cycleDuration);

        return () => {
            clearInterval(interval);
            clearTimeout(settleTimer);
        };
    }, [phase, cycleDuration]);

    useEffect(() => {
        if (phase !== 'settling') return;

        const nearChars = [
            CHARS[Math.floor(Math.random() * CHARS.length)],
            CHARS[Math.floor(Math.random() * CHARS.length)],
            targetChar,
        ];
        let step = 0;
        const interval = setInterval(() => {
            if (step < nearChars.length) {
                setDisplayChar(nearChars[step]);
                setOffset(step < nearChars.length - 1 ? (Math.random() > 0.5 ? 1 : -1) : 0);
                step++;
            } else {
                clearInterval(interval);
                setPhase('done');
                onSettled?.();
            }
        }, 50);

        return () => clearInterval(interval);
    }, [phase, targetChar, onSettled]);

    const isResolved = phase === 'done';

    return (
        <span
            className="inline-block relative overflow-hidden"
            style={{
                width: '0.7em',
                height: '1.2em',
            }}
        >
            <span
                className="absolute inset-0 flex items-center justify-center transition-all"
                style={{
                    transform: `translateY(${isResolved ? 0 : offset * 2}px)`,
                    transitionDuration: isResolved ? '300ms' : '30ms',
                    color: isResolved ? '#111111' : 'rgba(0,0,0,0.25)',
                    filter: isResolved ? 'none' : 'blur(0.5px)',
                }}
            >
                {displayChar}
            </span>
        </span>
    );
}

interface SplashScreenProps {
    onComplete: () => void;
    duration?: number;
}

export default function SplashScreen({ onComplete, duration = 3500 }: SplashScreenProps) {
    const [settledCount, setSettledCount] = useState(0);
    const [allSettled, setAllSettled] = useState(false);
    const [fadeOut, setFadeOut] = useState(false);
    const [subtitleVisible, setSubtitleVisible] = useState(false);

    const handleLetterSettled = useCallback(() => {
        setSettledCount((prev) => prev + 1);
    }, []);

    useEffect(() => {
        if (settledCount >= TARGET_WORD.length && !allSettled) {
            setAllSettled(true);
            setTimeout(() => setSubtitleVisible(true), 300);
            setTimeout(() => setFadeOut(true), 1800);
            setTimeout(() => onComplete(), 2400);
        }
    }, [settledCount, allSettled, onComplete]);

    const letters = TARGET_WORD.split('');

    return (
        <div
            className={`fixed inset-0 z-[9999] flex flex-col items-center justify-center transition-opacity duration-700 ${fadeOut ? 'opacity-0' : 'opacity-100'
                }`}
            style={{
                background: '#ffffff',
            }}
        >
            {/* Odometer text */}
            <div className="relative z-10 flex items-center">
                <div
                    className="flex font-semibold tracking-tight"
                    style={{
                        fontSize: 'clamp(2.5rem, 8vw, 5rem)',
                        fontFamily: 'var(--font-geist-sans), system-ui, sans-serif',
                    }}
                >
                    {letters.map((char, index) => (
                        <OdometerLetter
                            key={index}
                            targetChar={char}
                            delay={100 + index * 120}
                            cycleDuration={200 + index * 60}
                            onSettled={handleLetterSettled}
                        />
                    ))}
                </div>
            </div>

            {/* Subtitle */}
            <p
                className="relative z-10 mt-4 text-xs tracking-[0.25em] uppercase transition-all duration-700"
                style={{
                    color: 'rgba(0, 0, 0, 0.35)',
                    fontFamily: 'var(--font-geist-mono), monospace',
                    letterSpacing: '0.25em',
                    opacity: subtitleVisible ? 1 : 0,
                    transform: subtitleVisible ? 'translateY(0)' : 'translateY(6px)',
                }}
            >
                Prompt Optimization Engine
            </p>

            {/* Minimal loading indicator */}
            <div
                className="absolute bottom-12 left-1/2 -translate-x-1/2 h-[1px] rounded-full overflow-hidden"
                style={{ width: '120px', background: 'rgba(0, 0, 0, 0.06)' }}
            >
                <div
                    className="h-full rounded-full"
                    style={{
                        background: 'rgba(0, 0, 0, 0.2)',
                        animation: allSettled
                            ? 'none'
                            : 'loading-bar 2s ease-in-out forwards',
                        width: allSettled ? '100%' : undefined,
                        transition: allSettled ? 'width 0.3s ease' : undefined,
                    }}
                />
            </div>

            <style jsx>{`
                @keyframes loading-bar {
                    0% { width: 0%; }
                    100% { width: 90%; }
                }
            `}</style>
        </div>
    );
}
