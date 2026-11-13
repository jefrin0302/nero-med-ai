import React from 'react';
import { useAuth } from '../../context/AuthContext';
import { ShieldAlert, ArrowRightLeft, X, Lock } from 'lucide-react';

export default function AccessDeniedModal() {
  const { accessDeniedMessage, clearAccessDenied, switchRole, currentUser } = useAuth();

  if (!accessDeniedMessage) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fadeIn">
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl max-w-md w-full shadow-2xl p-6 relative overflow-hidden">
        
        {/* Accent Top Bar */}
        <div className="absolute top-0 left-0 right-0 h-1.5 bg-gradient-to-r from-rose-500 via-amber-500 to-rose-600" />

        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-center justify-center shrink-0">
            <ShieldAlert className="w-6 h-6 text-rose-500" />
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
                <span>Access Boundary Enforced</span>
                <span className="text-[10px] font-mono bg-rose-100 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400 px-2 py-0.5 rounded border border-rose-300 dark:border-rose-800">
                  HTTP 403
                </span>
              </h3>
              <button 
                onClick={clearAccessDenied}
                className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors p-1"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="mt-2 text-xs font-mono text-slate-500 dark:text-slate-400 bg-slate-100 dark:bg-slate-800/80 px-2.5 py-1.5 rounded border border-slate-200 dark:border-slate-700">
              Attempted Target: <span className="text-rose-500 font-semibold">{accessDeniedMessage.attemptedPath}</span>
            </div>

            <p className="mt-3 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              {accessDeniedMessage.message}
            </p>

            <div className="mt-4 pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between gap-3">
              <button
                onClick={clearAccessDenied}
                className="px-3 py-1.5 text-xs font-semibold text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
              >
                Dismiss
              </button>
              
              <button
                onClick={() => {
                  switchRole(accessDeniedMessage.requiredRole);
                }}
                className="px-4 py-2 text-xs font-bold bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-all shadow-md shadow-indigo-600/20 flex items-center gap-1.5"
              >
                <ArrowRightLeft className="w-3.5 h-3.5" />
                <span>Switch to {accessDeniedMessage.requiredRole} Mode</span>
              </button>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

