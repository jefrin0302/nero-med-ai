import React from 'react';
import { useAuth } from '../../context/AuthContext';
import { 
  ShieldCheck, 
  Cpu, 
  LogOut, 
  ArrowRightLeft, 
  Database, 
  FileText, 
  Activity, 
  Layers, 
  Search,
  BookOpen
} from 'lucide-react';

export default function Navbar() {
  const { currentUser, currentPath, switchRole, logout, navigateTo } = useAuth();

  if (!currentUser) return null;

  return (
    <header className="sticky top-0 z-50 bg-slate-900/95 dark:bg-[#0f172a]/95 backdrop-blur-md border-b border-slate-800 text-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        
        {/* Brand Group */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-indigo-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Database className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-base tracking-tight text-white">NeuroMed <span className="text-indigo-400">RAG</span></span>
              <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Control Plane
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium">Enterprise Knowledge & Vector Ops</p>
          </div>
        </div>

        {/* Center Navigation Links */}
        <nav className="hidden md:flex items-center gap-1.5 bg-slate-800/60 p-1 rounded-lg border border-slate-700/60 text-xs font-semibold">
          {currentUser.role === 'ADMIN' ? (
            <>
              <button
                onClick={() => navigateTo('/admin')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                  currentPath === '/admin'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Knowledge Base</span>
              </button>
              <button
                onClick={() => navigateTo('/admin/queries')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                  currentPath === '/admin/queries'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <Search className="w-3.5 h-3.5" />
                <span>Query Audit Logs</span>
              </button>
            </>
          ) : (
            <>
              <button
                onClick={() => navigateTo('/technician')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                  currentPath === '/technician'
                    ? 'bg-cyan-600 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Chunking & Sandboxing</span>
              </button>
              <button
                onClick={() => navigateTo('/technician/embeddings')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                  currentPath === '/technician/embeddings'
                    ? 'bg-cyan-600 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Embedding Queue</span>
              </button>
              <button
                onClick={() => navigateTo('/technician/retrieval')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                  currentPath === '/technician/retrieval'
                    ? 'bg-cyan-600 text-white shadow-sm'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700/50'
                }`}
              >
                <Search className="w-3.5 h-3.5" />
                <span>Vector Similarity Debugger</span>
              </button>
            </>
          )}
        </nav>

        {/* Right Actions & Demo Role Switcher */}
        <div className="flex items-center gap-3">
          
          {/* Quick Demo Role Switcher Badge */}
          <div className="flex items-center gap-2 bg-slate-800/80 px-2.5 py-1 rounded-lg border border-slate-700">
            <span className="text-[11px] text-slate-400 font-medium hidden sm:inline">Active:</span>
            <span className={`text-xs font-bold px-2 py-0.5 rounded-full flex items-center gap-1.5 ${
              currentUser.role === 'ADMIN'
                ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
            }`}>
              {currentUser.role === 'ADMIN' ? (
                <>
                  <ShieldCheck className="w-3 h-3 text-indigo-400" />
                  <span>ADMIN</span>
                </>
              ) : (
                <>
                  <Cpu className="w-3 h-3 text-cyan-400" />
                  <span>TECHNICIAN</span>
                </>
              )}
            </span>

            {/* One-click Role Switch Button */}
            <button
              onClick={() => switchRole(currentUser.role === 'ADMIN' ? 'TECHNICIAN' : 'ADMIN')}
              title={`Switch role to ${currentUser.role === 'ADMIN' ? 'Technician' : 'Admin'}`}
              className="text-[11px] font-semibold bg-slate-700/80 hover:bg-slate-600 text-slate-200 px-2 py-1 rounded transition-colors flex items-center gap-1.5 shadow-sm ml-1"
            >
              <ArrowRightLeft className="w-3 h-3 text-amber-400" />
              <span className="hidden lg:inline">Switch to {currentUser.role === 'ADMIN' ? 'Technician' : 'Admin'}</span>
            </button>
          </div>

          {/* User Profile Pill */}
          <div className="hidden sm:flex items-center gap-2 pl-2 border-l border-slate-700">
            <div className="w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center font-bold text-xs text-indigo-300 border border-slate-600">
              {currentUser.avatar}
            </div>
            <div className="text-left hidden xl:block">
              <div className="text-xs font-bold text-slate-200 leading-tight">{currentUser.name}</div>
              <div className="text-[10px] text-slate-400 leading-tight truncate max-w-[140px]">{currentUser.title}</div>
            </div>
          </div>

          {/* Sign Out Button */}
          <button
            onClick={logout}
            title="Sign Out"
            className="text-xs text-rose-400 hover:text-rose-300 hover:bg-rose-500/10 p-2 rounded-lg transition-colors border border-rose-500/20 flex items-center gap-1.5"
          >
            <LogOut className="w-4 h-4" />
            <span className="hidden sm:inline font-semibold">Sign Out</span>
          </button>
        </div>

      </div>
    </header>
  );
}

