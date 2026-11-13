import React, { useState } from 'react';
import { useAuth, DEMO_USERS } from '../../context/AuthContext';
import { 
  ShieldCheck, 
  Cpu, 
  Database, 
  Lock, 
  User, 
  ArrowRight, 
  CheckCircle2, 
  FileText, 
  Layers, 
  Activity, 
  Info
} from 'lucide-react';

export default function LoginPage() {
  const { login } = useAuth();
  const [selectedRole, setSelectedRole] = useState('ADMIN');
  const [username, setUsername] = useState(DEMO_USERS.ADMIN.username);
  const [password, setPassword] = useState('••••••••••••');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleRoleChange = (role) => {
    setSelectedRole(role);
    setUsername(DEMO_USERS[role].username);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setTimeout(() => {
      login(selectedRole, { username, password });
      setIsSubmitting(false);
    }, 450);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-white flex flex-col justify-center items-center p-4 relative overflow-hidden font-sans">
      
      {/* Background Ambient Glows */}
      <div className="absolute top-1/4 left-1/4 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 translate-x-1/2 translate-y-1/2 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Main Container */}
      <div className="max-w-md w-full bg-slate-900/80 backdrop-blur-xl border border-slate-800 rounded-3xl shadow-2xl p-8 relative z-10">
        
        {/* Header Branding */}
        <div className="text-center mb-8">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-indigo-600 to-cyan-500 mx-auto flex items-center justify-center shadow-xl shadow-indigo-500/25 mb-4">
            <Database className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight text-white">
            NeuroMed <span className="text-indigo-400">RAG</span> Platform
          </h1>
          <p className="text-xs text-slate-400 mt-1.5 font-medium">
            Clinical Decision Support & Vector Retrieval Control Plane
          </p>
        </div>

        {/* Role Selector Tabs */}
        <div className="mb-6">
          <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
            Select Access Role
          </label>
          <div className="grid grid-cols-2 gap-2 bg-slate-950/70 p-1.5 rounded-xl border border-slate-800">
            <button
              type="button"
              onClick={() => handleRoleChange('ADMIN')}
              className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-xs font-bold transition-all ${
                selectedRole === 'ADMIN'
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <ShieldCheck className="w-4 h-4 text-indigo-300" />
              <span>Admin Portal</span>
            </button>

            <button
              type="button"
              onClick={() => handleRoleChange('TECHNICIAN')}
              className={`flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-xs font-bold transition-all ${
                selectedRole === 'TECHNICIAN'
                  ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
              }`}
            >
              <Cpu className="w-4 h-4 text-cyan-300" />
              <span>Technician Console</span>
            </button>
          </div>
        </div>

        {/* Dynamic Role Scope Explainer */}
        <div className={`p-3.5 rounded-xl border text-xs leading-relaxed mb-6 transition-colors ${
          selectedRole === 'ADMIN'
            ? 'bg-indigo-950/40 border-indigo-500/30 text-indigo-200'
            : 'bg-cyan-950/40 border-cyan-500/30 text-cyan-200'
        }`}>
          <div className="font-bold flex items-center gap-1.5 mb-1 text-slate-100">
            <Info className="w-3.5 h-3.5 shrink-0" />
            <span>Workspace Scope & Authorization:</span>
          </div>
          {selectedRole === 'ADMIN' ? (
            <ul className="space-y-1 text-[11.5px] text-slate-300 mt-1 pl-1">
              <li className="flex items-center gap-1.5">
                <CheckCircle2 className="w-3 h-3 text-indigo-400 shrink-0" />
                <span>Executive Knowledge Ingestion & Approval</span>
              </li>
              <li className="flex items-center gap-1.5">
                <CheckCircle2 className="w-3 h-3 text-indigo-400 shrink-0" />
                <span>Document Status Management & Query Auditing</span>
              </li>
            </ul>
          ) : (
            <ul className="space-y-1 text-[11.5px] text-slate-300 mt-1 pl-1">
              <li className="flex items-center gap-1.5">
                <CheckCircle2 className="w-3 h-3 text-cyan-400 shrink-0" />
                <span>Chunking Sandbox, Sliders & Text Overlap Inspector</span>
              </li>
              <li className="flex items-center gap-1.5">
                <CheckCircle2 className="w-3 h-3 text-cyan-400 shrink-0" />
                <span>Vector Embedding Queue, Latency & Cosine Debugger</span>
              </li>
            </ul>
          )}
        </div>

        {/* Login Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Authorized Identity
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-500">
                <User className="w-4 h-4" />
              </div>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                className="w-full pl-9 pr-3 py-2.5 bg-slate-950/80 border border-slate-700/80 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>
            <div className="text-[11px] text-slate-400 mt-1 font-mono">
              Identity: <span className="text-slate-200">{DEMO_USERS[selectedRole].name}</span>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1.5">
              Security Token / Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-500">
                <Lock className="w-4 h-4" />
              </div>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full pl-9 pr-3 py-2.5 bg-slate-950/80 border border-slate-700/80 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className={`w-full py-3 px-4 rounded-xl text-xs font-bold flex items-center justify-center gap-2 transition-all shadow-lg mt-6 ${
              selectedRole === 'ADMIN'
                ? 'bg-indigo-600 hover:bg-indigo-500 shadow-indigo-600/30'
                : 'bg-cyan-600 hover:bg-cyan-500 shadow-cyan-600/30'
            } text-white`}
          >
            {isSubmitting ? (
              <span>Authenticating Clinical Token...</span>
            ) : (
              <>
                <span>Enter {selectedRole === 'ADMIN' ? 'Admin Dashboard' : 'Technician Workspace'}</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        <div className="mt-6 pt-4 border-t border-slate-800 text-center">
          <p className="text-[11px] text-slate-500">
            NeuroMed CDSS Architecture • HIPAA/GDPR Calibrated Consensus System
          </p>
        </div>

      </div>
    </div>
  );
}

