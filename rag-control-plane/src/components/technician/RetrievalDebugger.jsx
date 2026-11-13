import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Sliders, 
  Zap, 
  Database, 
  Clock, 
  CheckCircle2, 
  ChevronRight, 
  Copy, 
  Check, 
  FileText, 
  Tag, 
  Layers, 
  Sparkles,
  Info,
  Code
} from 'lucide-react';
import { simulateVectorSearch, INITIAL_DOCUMENTS } from '../../services/ragPipelineService';

const SAMPLE_PROMPTS = [
  "Penicillamine starting dose and titration schedule in symptomatic hepatic disease",
  "How to interpret Leipzig scoring when Kayser-Fleischer rings are absent",
  "Baseline 24-hour urinary copper cutoffs for pediatric Wilson disease",
  "Dietary copper restrictions regarding shellfish, nuts, and dark chocolate",
  "Trientine dihydrochloride dosing vs D-penicillamine chelation efficacy"
];

export default function RetrievalDebugger() {
  const [query, setQuery] = useState(SAMPLE_PROMPTS[0]);
  const [topK, setTopK] = useState(4);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.70);
  const [selectedDocId, setSelectedDocId] = useState('all');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResponse, setSearchResponse] = useState(null);
  const [expandedChunkId, setExpandedChunkId] = useState(null);
  const [copiedChunkId, setCopiedChunkId] = useState(null);
  const [viewMetadataChunk, setViewMetadataChunk] = useState(null);

  const handleRunSearch = () => {
    if (!query.trim()) return;
    setIsSearching(true);
    setTimeout(() => {
      const response = simulateVectorSearch(query, topK, similarityThreshold, selectedDocId);
      setSearchResponse(response);
      setIsSearching(false);
      if (response.results.length > 0) {
        setExpandedChunkId(response.results[0].chunk.id);
      }
    }, 180);
  };

  useEffect(() => {
    handleRunSearch();
  }, []);

  const handleCopyText = (chunkId, text) => {
    navigator.clipboard.writeText(text);
    setCopiedChunkId(chunkId);
    setTimeout(() => setCopiedChunkId(null), 1800);
  };

  return (
    <div className="space-y-6">
      {/* HEADER BANNER */}
      <div className="bg-tech-card border border-tech-border rounded-xl p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                <Database className="w-3.5 h-3.5" /> Vector Retrieval Engine
              </span>
              <span className="text-xs text-tech-muted font-mono">Qdrant HNSW • Cosine Metric</span>
            </div>
            <h2 className="text-lg font-bold text-white tracking-tight">
              Retrieval Debugger &amp; Similarity Inspector
            </h2>
            <p className="text-xs text-tech-muted mt-0.5">
              Probe raw embedding space, evaluate cosine distance boundaries, and audit payload citations in real-time.
            </p>
          </div>

          <div className="flex items-center gap-3 self-start md:self-auto">
            <div className="bg-tech-surface border border-tech-border px-3 py-1.5 rounded-lg text-right">
              <span className="text-[10px] text-tech-muted uppercase font-mono block">Target Index</span>
              <span className="text-xs font-bold font-mono text-cyan-400">text-embedding-3-large</span>
            </div>
            <div className="bg-tech-surface border border-tech-border px-3 py-1.5 rounded-lg text-right">
              <span className="text-[10px] text-tech-muted uppercase font-mono block">Dimensions</span>
              <span className="text-xs font-bold font-mono text-emerald-400">3072-D</span>
            </div>
          </div>
        </div>
      </div>

      {/* QUERY INPUT & TUNING PARAMETERS */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* LEFT 2 COLS: SEARCH INPUT & PRESETS */}
        <div className="lg:col-span-2 bg-tech-card border border-tech-border rounded-xl p-5 space-y-4">
          <label className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
            Test Vector Query Input
          </label>
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleRunSearch()}
              placeholder="Enter clinical query or diagnostic hypothesis to inspect retrieval..."
              className="w-full bg-tech-surface border border-tech-border rounded-lg pl-10 pr-24 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 font-sans"
            />
            <Search className="w-4 h-4 text-tech-muted absolute left-3.5 top-3.5" />
            <button
              onClick={handleRunSearch}
              disabled={isSearching || !query.trim()}
              className="absolute right-1.5 top-1.5 bottom-1.5 px-3.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-md text-xs font-bold flex items-center gap-1.5 shadow-sm transition-all disabled:opacity-50"
            >
              {isSearching ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <Zap className="w-3.5 h-3.5" />
                  <span>Execute</span>
                </>
              )}
            </button>
          </div>

          {/* CLINICAL PROMPT CHIPS */}
          <div>
            <span className="text-[11px] font-semibold text-tech-muted uppercase tracking-wider block mb-2">
              Clinical Benchmark Queries:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {SAMPLE_PROMPTS.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => { setQuery(p); }}
                  className={`text-xs px-2.5 py-1 rounded-md text-left transition-colors border ${
                    query === p
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 font-medium'
                      : 'bg-tech-surface hover:bg-slate-800 text-slate-400 hover:text-slate-200 border-tech-border'
                  }`}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT 1 COL: GRANULAR TUNING SLIDERS */}
        <div className="bg-tech-card border border-tech-border rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-tech-border pb-2.5">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-cyan-400" /> Retrieval Hyperparameters
            </span>
            <button
              onClick={() => { setTopK(4); setSimilarityThreshold(0.70); setSelectedDocId('all'); }}
              className="text-[10px] text-tech-muted hover:text-slate-300 underline font-mono"
            >
              Reset Defaults
            </button>
          </div>

          {/* TOP-K SLIDER */}
          <div>
            <div className="flex justify-between items-center text-xs mb-1">
              <span className="text-slate-400 font-medium">Top-K Results</span>
              <span className="font-mono font-bold text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800">
                {topK} chunks
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer h-1.5 bg-tech-surface rounded-lg"
            />
            <div className="flex justify-between text-[10px] text-tech-muted font-mono mt-0.5">
              <span>1</span>
              <span>5</span>
              <span>10</span>
            </div>
          </div>

          {/* SIMILARITY THRESHOLD */}
          <div>
            <div className="flex justify-between items-center text-xs mb-1">
              <span className="text-slate-400 font-medium">Similarity Cutoff (Cosine)</span>
              <span className="font-mono font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800">
                ≥ {similarityThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.05"
              value={similarityThreshold}
              onChange={(e) => setSimilarityThreshold(parseFloat(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer h-1.5 bg-tech-surface rounded-lg"
            />
            <div className="flex justify-between text-[10px] text-tech-muted font-mono mt-0.5">
              <span>0.50 (Loose)</span>
              <span>0.70</span>
              <span>0.95 (Strict)</span>
            </div>
          </div>

          {/* DOCUMENT RESTRICTION FILTER */}
          <div>
            <label className="text-xs text-slate-400 font-medium block mb-1">
              Corpus Scope Filter
            </label>
            <select
              value={selectedDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
              className="w-full bg-tech-surface border border-tech-border text-xs rounded-md p-2 text-slate-300 focus:outline-none focus:border-cyan-500 font-sans"
            >
              <option value="all">All Active Corpus Documents (Full RAG)</option>
              {INITIAL_DOCUMENTS.map(d => (
                <option key={d.id} value={d.id}>
                  {d.title.substring(0, 36)}...
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* LATENCY BREAKDOWN & DIAGNOSTIC METRICS */}
      {searchResponse && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-tech-card border border-tech-border rounded-lg p-3">
            <span className="text-[10px] text-tech-muted uppercase font-mono block">Query Embedding Time</span>
            <div className="flex items-center gap-1.5 mt-1">
              <Clock className="w-4 h-4 text-cyan-400" />
              <span className="text-base font-bold font-mono text-white">
                {searchResponse.latencyBreakdown.embeddingTimeMs}
              </span>
              <span className="text-xs text-tech-muted font-mono">ms</span>
            </div>
          </div>

          <div className="bg-tech-card border border-tech-border rounded-lg p-3">
            <span className="text-[10px] text-tech-muted uppercase font-mono block">HNSW Vector Scan Time</span>
            <div className="flex items-center gap-1.5 mt-1">
              <Zap className="w-4 h-4 text-emerald-400" />
              <span className="text-base font-bold font-mono text-white">
                {searchResponse.latencyBreakdown.searchTimeMs}
              </span>
              <span className="text-xs text-tech-muted font-mono">ms</span>
            </div>
          </div>

          <div className="bg-tech-card border border-tech-border rounded-lg p-3">
            <span className="text-[10px] text-tech-muted uppercase font-mono block">Total Round-Trip Latency</span>
            <div className="flex items-center gap-1.5 mt-1">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span className="text-base font-bold font-mono text-purple-300">
                {searchResponse.latencyBreakdown.totalLatencyMs}
              </span>
              <span className="text-xs text-tech-muted font-mono">ms</span>
            </div>
          </div>

          <div className="bg-tech-card border border-tech-border rounded-lg p-3">
            <span className="text-[10px] text-tech-muted uppercase font-mono block">Citations Qualifying</span>
            <div className="flex items-center gap-1.5 mt-1">
              <CheckCircle2 className="w-4 h-4 text-teal-400" />
              <span className="text-base font-bold font-mono text-white">
                {searchResponse.results.length} / {searchResponse.totalRetrieved}
              </span>
              <span className="text-xs text-tech-muted">yielded</span>
            </div>
          </div>
        </div>
      )}

      {/* RETRIEVED CHUNKS LIST & INSPECTOR */}
      <div className="bg-tech-card border border-tech-border rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-tech-border pb-3">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" /> Retrieved Vector Candidates
            </h3>
            <span className="text-xs text-tech-muted">
              Ranked in descending order by Cosine Similarity relative to query embedding
            </span>
          </div>
          {searchResponse && (
            <span className="text-xs font-mono text-tech-muted">
              Threshold: ≥ {similarityThreshold.toFixed(2)} • Limit: {topK}
            </span>
          )}
        </div>

        {searchResponse && searchResponse.results.length === 0 ? (
          <div className="text-center py-12 bg-tech-surface/40 rounded-lg border border-dashed border-tech-border">
            <Info className="w-8 h-8 text-amber-400 mx-auto mb-2" />
            <h4 className="text-sm font-bold text-slate-300">No Vector Chunks Qualified</h4>
            <p className="text-xs text-tech-muted max-w-md mx-auto mt-1">
              No chunks in the active index met the strict cosine similarity cutoff of ≥ {similarityThreshold.toFixed(2)}. 
              Try lowering the threshold slider or broadening your search terms.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {searchResponse && searchResponse.results.map((res, index) => {
              const isExpanded = expandedChunkId === res.chunk.id;
              const score = res.cosineScore;
              
              // Color badge based on score
              const scoreBadgeClass = score >= 0.88 
                ? 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30'
                : score >= 0.78
                ? 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30'
                : 'bg-amber-500/15 text-amber-400 border-amber-500/30';

              return (
                <div 
                  key={res.chunk.id}
                  className={`border rounded-lg transition-all ${
                    isExpanded 
                      ? 'border-cyan-500/60 bg-tech-surface shadow-md' 
                      : 'border-tech-border bg-tech-surface/50 hover:border-slate-700'
                  }`}
                >
                  {/* CARD SUMMARY HEADER */}
                  <div 
                    onClick={() => setExpandedChunkId(isExpanded ? null : res.chunk.id)}
                    className="p-3.5 flex flex-col md:flex-row md:items-center justify-between gap-3 cursor-pointer select-none"
                  >
                    <div className="flex items-center gap-3">
                      <span className="w-6 h-6 rounded-md bg-tech-border flex items-center justify-center text-xs font-mono font-bold text-slate-300">
                        #{index + 1}
                      </span>
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold font-mono text-cyan-300">
                            {res.chunk.id}
                          </span>
                          <span className="text-xs font-medium text-slate-200">
                            {res.docTitle}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className="text-[10px] text-tech-muted flex items-center gap-1">
                            <Tag className="w-3 h-3" /> {res.category}
                          </span>
                          <span className="text-tech-muted text-xs">•</span>
                          <span className="text-[10px] font-mono text-tech-muted">
                            {res.tokenCount} tokens
                          </span>
                          <span className="text-tech-muted text-xs">•</span>
                          <span className="text-[10px] text-tech-muted">
                            {res.chunk.metadata.pageRange}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-2 self-end md:self-auto">
                      <span className={`px-2.5 py-1 rounded text-xs font-mono font-bold border flex items-center gap-1.5 ${scoreBadgeClass}`}>
                        <Sparkles className="w-3 h-3" />
                        {score.toFixed(3)}
                        <span className="text-[10px] font-sans font-normal opacity-75">
                          ({res.relevanceTier})
                        </span>
                      </span>

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setViewMetadataChunk(res.chunk);
                        }}
                        title="Inspect Vector Payload Metadata"
                        className="p-1.5 rounded bg-tech-card hover:bg-slate-800 text-slate-400 hover:text-slate-200 border border-tech-border"
                      >
                        <Code className="w-3.5 h-3.5" />
                      </button>

                      <ChevronRight className={`w-4 h-4 text-tech-muted transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                    </div>
                  </div>

                  {/* EXPANDED CONTENT DRAWER */}
                  {isExpanded && (
                    <div className="px-4 pb-4 pt-1 border-t border-tech-border/80 space-y-3">
                      <div className="flex items-center justify-between text-xs text-tech-muted pt-2">
                        <span className="font-semibold uppercase tracking-wider text-[10px]">
                          Decoded Payload Text (Exact Retrieved Context)
                        </span>
                        <button
                          onClick={() => handleCopyText(res.chunk.id, res.chunk.text)}
                          className="flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 font-mono"
                        >
                          {copiedChunkId === res.chunk.id ? (
                            <>
                              <Check className="w-3.5 h-3.5 text-emerald-400" />
                              <span className="text-emerald-400">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" />
                              <span>Copy Chunk</span>
                            </>
                          )}
                        </button>
                      </div>

                      <div className="p-3 bg-slate-950/80 rounded border border-tech-border text-xs text-slate-300 font-mono leading-relaxed max-h-48 overflow-y-auto">
                        {res.chunk.text}
                      </div>

                      {/* STATS FOOTER */}
                      <div className="flex flex-wrap items-center justify-between text-[11px] font-mono text-tech-muted pt-1">
                        <div>
                          Word Bounds: [{res.chunk.metadata.startWord} → {res.chunk.metadata.endWord}]
                        </div>
                        <div className="flex items-center gap-3">
                          <span>Vector Dim: 3072</span>
                          <span>Norm: L2 Unit</span>
                          <span className="text-emerald-400">HNSW Verified</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* METADATA INSPECTION MODAL */}
      {viewMetadataChunk && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
          <div className="bg-tech-card border border-tech-border rounded-xl w-full max-w-xl p-5 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-tech-border pb-3">
              <div className="flex items-center gap-2">
                <Code className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">
                  Vector Store Payload: {viewMetadataChunk.id}
                </h3>
              </div>
              <button
                onClick={() => setViewMetadataChunk(null)}
                className="text-slate-400 hover:text-white text-xs font-mono px-2 py-1 bg-tech-surface rounded border border-tech-border"
              >
                Close
              </button>
            </div>

            <pre className="p-4 bg-slate-950 rounded-lg border border-tech-border text-xs font-mono text-emerald-400 max-h-96 overflow-y-auto leading-relaxed">
              {JSON.stringify(
                {
                  id: viewMetadataChunk.id,
                  document_id: viewMetadataChunk.docId,
                  document_title: viewMetadataChunk.metadata.docTitle,
                  category: viewMetadataChunk.metadata.category,
                  token_count: viewMetadataChunk.tokenCount,
                  page_range: viewMetadataChunk.metadata.pageRange,
                  word_span: {
                    start: viewMetadataChunk.metadata.startWord,
                    end: viewMetadataChunk.metadata.endWord
                  },
                  vector_geometry: {
                    model: "text-embedding-3-large",
                    dimensions: 3072,
                    metric: "cosine"
                  },
                  payload_preview: viewMetadataChunk.text.substring(0, 140) + "..."
                },
                null,
                2
              )}
            </pre>

            <div className="flex justify-end">
              <button
                onClick={() => setViewMetadataChunk(null)}
                className="px-4 py-2 bg-tech-surface hover:bg-slate-800 text-xs font-bold text-slate-200 rounded border border-tech-border"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

