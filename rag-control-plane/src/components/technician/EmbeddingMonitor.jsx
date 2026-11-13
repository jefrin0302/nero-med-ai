import React, { useState } from 'react';
import { 
  Database, 
  Cpu, 
  Activity, 
  Layers, 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  ChevronDown, 
  ChevronUp, 
  HardDrive, 
  RefreshCw,
  Zap,
  Terminal
} from 'lucide-react';
import { VECTOR_STORE_CONFIG, EMBEDDING_QUEUE_JOBS } from '../../services/ragPipelineService';

export default function EmbeddingMonitor() {
  const [config, setConfig] = useState(VECTOR_STORE_CONFIG);
  const [jobs, setJobs] = useState(EMBEDDING_QUEUE_JOBS);
  const [expandedJobId, setExpandedJobId] = useState('job-embed-898'); // Expand the failed job by default for error inspection

  const toggleExpand = (jobId) => {
    setExpandedJobId(prev => prev === jobId ? null : jobId);
  };

  return (
    <div className="space-y-6">
      
      {/* Vector Store & Model Telemetry Header Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* Model Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-white shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Active Embedding Model</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-lg font-extrabold text-cyan-400 font-mono tracking-tight">
            {config.modelName}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            Provider: {config.provider}
          </div>
        </div>

        {/* Vector Dimensions & Metric */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-white shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Vector Geometry</span>
            <Layers className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-xl font-extrabold text-white font-mono">
            {config.dimensions} <span className="text-xs font-normal text-slate-400">Dimensions</span>
          </div>
          <div className="text-[11px] text-indigo-400 font-semibold mt-1">
            Metric: {config.distanceMetric}
          </div>
        </div>

        {/* Index Structure & Capacity */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-white shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Vector Index (HNSW)</span>
            <Database className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-xl font-extrabold text-emerald-400 font-mono">
            {config.totalVectors} <span className="text-xs font-normal text-slate-400">Dense Vectors</span>
          </div>
          <div className="text-[11px] text-slate-400 mt-1 font-mono">
            M={config.hnswM} • efSearch={config.hnswEfSearch}
          </div>
        </div>

        {/* Memory & Hardware */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 text-white shadow-lg">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-bold uppercase tracking-wider">Memory & Quantization</span>
            <HardDrive className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-xl font-extrabold text-white font-mono">
            {config.memoryUsageMb} <span className="text-xs font-normal text-slate-400">MB RAM</span>
          </div>
          <div className="text-[11px] text-amber-400 font-semibold mt-1">
            {config.quantization}
          </div>
        </div>

      </div>

      {/* Embedding Pipeline Queue & Batch Engine */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl text-white">
        
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-5 border-b border-slate-800">
          <div>
            <h3 className="text-base font-bold flex items-center gap-2 text-white">
              <Activity className="w-5 h-5 text-cyan-400" />
              <span>Embedding Pipeline Worker Queue & Diagnostic Logs</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Real-time batch vectorization progress, throughput velocity (chunks/sec), and exception stack traces
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-950/70 text-emerald-400 border border-emerald-800">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>Worker Daemon: Active (4 Threads)</span>
            </span>
          </div>
        </div>

        {/* Queue Items */}
        <div className="mt-5 space-y-4">
          {jobs.map((job) => {
            const isExpanded = expandedJobId === job.id;
            const isProcessing = job.status === 'Processing';
            const isFailed = job.status === 'Failed';
            const isCompleted = job.status === 'Completed';

            return (
              <div 
                key={job.id}
                className={`rounded-2xl border transition-all ${
                  isFailed
                    ? 'bg-rose-950/15 border-rose-900/60'
                    : (isProcessing ? 'bg-cyan-950/20 border-cyan-800/80 shadow-md' : 'bg-slate-950/50 border-slate-800')
                }`}
              >
                <div className="p-4 sm:p-5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    
                    {/* Left Meta */}
                    <div className="flex items-start gap-3">
                      <div className={`p-2 rounded-xl mt-0.5 shrink-0 ${
                        isFailed 
                          ? 'bg-rose-950 text-rose-400 border border-rose-800' 
                          : (isProcessing ? 'bg-cyan-950 text-cyan-400 border border-cyan-800' : 'bg-emerald-950 text-emerald-400 border border-emerald-800')
                      }`}>
                        {isFailed && <AlertTriangle className="w-4 h-4" />}
                        {isProcessing && <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />}
                        {isCompleted && <CheckCircle2 className="w-4 h-4" />}
                      </div>

                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs text-slate-400">{job.id}</span>
                          <span className="text-slate-600">•</span>
                          <span className={`text-[11px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                            isFailed 
                              ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                              : (isProcessing ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30')
                          }`}>
                            {job.status}
                          </span>
                          <span className="text-slate-600">•</span>
                          <span className="text-xs text-slate-400">{job.startTime}</span>
                        </div>

                        <h4 className="text-sm font-bold text-white mt-1">
                          {job.documentTitle}
                        </h4>
                      </div>
                    </div>

                    {/* Right Metrics & Speed */}
                    <div className="flex items-center gap-4 text-xs font-mono">
                      <div>
                        <div className="text-[10px] text-slate-500 uppercase">Velocity</div>
                        <div className="font-bold text-cyan-400 flex items-center gap-1">
                          <Zap className="w-3 h-3 text-cyan-400" />
                          <span>{job.speedChunksPerSec} chk/s</span>
                        </div>
                      </div>

                      <div className="pl-3 border-l border-slate-800">
                        <div className="text-[10px] text-slate-500 uppercase">Tokens</div>
                        <div className="font-bold text-white">
                          {job.tokensConsumed.toLocaleString()}
                        </div>
                      </div>

                      <div className="pl-3 border-l border-slate-800">
                        <div className="text-[10px] text-slate-500 uppercase">Progress</div>
                        <div className="font-bold text-slate-200">
                          {job.chunksCompleted} / {job.chunksTotal} ({job.progress}%)
                        </div>
                      </div>

                      {/* Expand Button for Failures or Details */}
                      {isFailed && (
                        <button
                          onClick={() => toggleExpand(job.id)}
                          className="ml-2 p-1.5 rounded-lg bg-rose-900/40 hover:bg-rose-900/60 text-rose-300 border border-rose-800 transition-colors flex items-center gap-1 text-[11px] font-sans font-bold"
                        >
                          <Terminal className="w-3 h-3" />
                          <span>{isExpanded ? 'Hide Trace' : 'Inspect Stack Trace'}</span>
                          {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        </button>
                      )}
                    </div>

                  </div>

                  {/* Progress Bar */}
                  <div className="mt-4">
                    <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${
                          isFailed 
                            ? 'bg-rose-500' 
                            : (isProcessing ? 'bg-gradient-to-r from-cyan-500 to-indigo-500' : 'bg-emerald-500')
                        }`}
                        style={{ width: `${job.progress}%` }}
                      />
                    </div>
                  </div>

                  {/* Expandable Stack Trace Error Box */}
                  {isExpanded && isFailed && job.error && (
                    <div className="mt-4 p-4 rounded-xl bg-black/80 border border-rose-900/80 font-mono text-xs text-rose-300 leading-relaxed overflow-x-auto">
                      <div className="flex items-center gap-2 text-rose-400 font-bold mb-2 pb-2 border-b border-rose-950 text-[11px] uppercase tracking-wider">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>Worker Exception Stack Trace (WorkerThread #2)</span>
                      </div>
                      <pre className="whitespace-pre-wrap text-[11px] text-rose-200">
                        {job.error}
                      </pre>
                    </div>
                  )}

                </div>
              </div>
            );
          })}
        </div>

      </div>

    </div>
  );
}

