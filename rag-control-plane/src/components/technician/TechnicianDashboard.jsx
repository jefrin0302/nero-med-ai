import React, { useState } from 'react';
import { 
  Cpu, 
  Layers, 
  Database, 
  Search, 
  Activity, 
  RefreshCw,
  Terminal,
  Server,
  Zap,
  CheckCircle2
} from 'lucide-react';
import ChunkingConfigurator from './ChunkingConfigurator';
import EmbeddingMonitor from './EmbeddingMonitor';
import RetrievalDebugger from './RetrievalDebugger';
import { VECTOR_STORE_CONFIG, SYSTEM_KPIS } from '../../services/ragPipelineService';

export default function TechnicianDashboard() {
  const [activeTab, setActiveTab] = useState('chunking'); // 'chunking' | 'embedding' | 'retrieval'
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefreshTelemetry = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 600);
  };

  return (
    <div className="space-y-6">
      {/* DEVELOPER STATUS BAR / TELEMETRY HEADER */}
      <div className="bg-tech-card border border-tech-border rounded-xl p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Cluster Live • Node #2 (Prod-RAG-East)
              </span>
              <span className="text-xs text-tech-muted font-mono">Telemetry Sync: {SYSTEM_KPIS.lastIndexSync}</span>
            </div>
            <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
              <Terminal className="w-5 h-5 text-cyan-400" />
              RAG Pipeline &amp; Vector Store Console
            </h1>
            <p className="text-xs text-tech-muted mt-0.5">
              Granular observability, chunking strategy calibration, and vector retrieval inspector for clinical AI pipelines.
            </p>
          </div>

          <div className="flex items-center gap-2 self-start md:self-auto">
            <button
              onClick={handleRefreshTelemetry}
              disabled={isRefreshing}
              className="px-3 py-1.5 bg-tech-surface hover:bg-slate-800 text-slate-300 hover:text-white rounded-lg text-xs font-mono font-medium border border-tech-border flex items-center gap-1.5 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 text-cyan-400 ${isRefreshing ? 'animate-spin' : ''}`} />
              <span>Refresh Metrics</span>
            </button>
          </div>
        </div>

        {/* QUICK STATS CHIPS */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-4 border-t border-tech-border">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-950/60 border border-cyan-800 flex items-center justify-center text-cyan-400">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] text-tech-muted font-mono uppercase block">Model Architecture</span>
              <span className="text-xs font-bold font-mono text-slate-200">
                {VECTOR_STORE_CONFIG.modelName}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-950/60 border border-emerald-800 flex items-center justify-center text-emerald-400">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] text-tech-muted font-mono uppercase block">Indexed Geometry</span>
              <span className="text-xs font-bold font-mono text-emerald-400">
                {VECTOR_STORE_CONFIG.dimensions} Dim • Cosine
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-purple-950/60 border border-purple-800 flex items-center justify-center text-purple-400">
              <Server className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] text-tech-muted font-mono uppercase block">Total Vectors</span>
              <span className="text-xs font-bold font-mono text-purple-300">
                {VECTOR_STORE_CONFIG.totalVectors} vectors
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-teal-950/60 border border-teal-800 flex items-center justify-center text-teal-400">
              <Zap className="w-4 h-4" />
            </div>
            <div>
              <span className="text-[10px] text-tech-muted font-mono uppercase block">Index Status</span>
              <span className="text-xs font-bold font-mono text-teal-300 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 text-teal-400" />
                {VECTOR_STORE_CONFIG.indexStatus}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* TAB NAVIGATION PILLS */}
      <div className="flex items-center gap-2 border-b border-tech-border pb-2 overflow-x-auto">
        <button
          onClick={() => setActiveTab('chunking')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all ${
            activeTab === 'chunking'
              ? 'bg-cyan-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-tech-card'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Chunking Strategy &amp; Sandbox</span>
        </button>

        <button
          onClick={() => setActiveTab('embedding')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all ${
            activeTab === 'embedding'
              ? 'bg-cyan-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-tech-card'
          }`}
        >
          <Activity className="w-4 h-4" />
          <span>Embedding &amp; Vector Store Monitor</span>
        </button>

        <button
          onClick={() => setActiveTab('retrieval')}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all ${
            activeTab === 'retrieval'
              ? 'bg-cyan-600 text-white shadow-sm'
              : 'text-slate-400 hover:text-slate-200 hover:bg-tech-card'
          }`}
        >
          <Search className="w-4 h-4" />
          <span>Retrieval Debugger &amp; Similarity Inspector</span>
        </button>
      </div>

      {/* TAB CONTENT PANELS */}
      <div>
        {activeTab === 'chunking' && <ChunkingConfigurator />}
        {activeTab === 'embedding' && <EmbeddingMonitor />}
        {activeTab === 'retrieval' && <RetrievalDebugger />}
      </div>
    </div>
  );
}

