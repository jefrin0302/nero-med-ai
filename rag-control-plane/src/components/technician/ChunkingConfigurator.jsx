import React, { useState, useMemo } from 'react';
import { 
  Sliders, 
  Layers, 
  RefreshCw, 
  FileText, 
  Cpu, 
  Sparkles, 
  Info, 
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Eye,
  Hash
} from 'lucide-react';
import { generateDocumentChunks } from '../../services/ragPipelineService';

export default function ChunkingConfigurator({ documents, onTriggerReindex }) {
  const [selectedDocId, setSelectedDocId] = useState(documents[0]?.id || '');
  const [method, setMethod] = useState('recursive'); // 'fixed' | 'recursive' | 'semantic' | 'markdown'
  const [chunkSize, setChunkSize] = useState(512);
  const [chunkOverlap, setChunkOverlap] = useState(64);
  const [selectedChunkIndex, setSelectedChunkIndex] = useState(0);
  const [isReindexing, setIsReindexing] = useState(false);
  const [reindexProgress, setReindexProgress] = useState(0);

  const activeDoc = useMemo(() => {
    return documents.find(d => d.id === selectedDocId) || documents[0];
  }, [documents, selectedDocId]);

  // Dynamically compute chunks with highlighted overlap
  const chunks = useMemo(() => {
    if (!activeDoc) return [];
    return generateDocumentChunks(activeDoc, method, chunkSize, chunkOverlap);
  }, [activeDoc, method, chunkSize, chunkOverlap]);

  const activeChunk = chunks[selectedChunkIndex] || chunks[0];

  const handleStartReindex = () => {
    setIsReindexing(true);
    setReindexProgress(10);
    const interval = setInterval(() => {
      setReindexProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            setIsReindexing(false);
            if (onTriggerReindex) onTriggerReindex(selectedDocId, { method, chunkSize, chunkOverlap });
          }, 400);
          return 100;
        }
        return prev + 18;
      });
    }, 250);
  };

  return (
    <div className="space-y-6">
      
      {/* Configuration Controls Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl text-white">
        
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-5 border-b border-slate-800">
          <div>
            <h3 className="text-base font-bold flex items-center gap-2 text-white">
              <Sliders className="w-5 h-5 text-cyan-400" />
              <span>Chunking Strategy & Sandbox Parameter Tuning</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Adjust token partition boundaries, semantic windowing, and sliding overlap in real time
            </p>
          </div>

          {/* Action Trigger */}
          <div className="flex items-center gap-3">
            <button
              onClick={handleStartReindex}
              disabled={isReindexing}
              className="px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-bold transition-all shadow-lg shadow-cyan-600/20 flex items-center gap-2"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isReindexing ? 'animate-spin' : ''}`} />
              <span>{isReindexing ? `Re-indexing (${reindexProgress}%)` : 'Trigger Corpus Re-index'}</span>
            </button>
          </div>
        </div>

        {/* Sliders & Parameters Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-5">
          
          {/* Method Picker */}
          <div>
            <label className="block text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-cyan-400" />
              <span>Chunking Method</span>
            </label>
            <div className="space-y-1.5">
              {[
                { id: 'recursive', name: 'Recursive Character', desc: 'Hierarchical split on \\n\\n, \\n, and sentence boundaries' },
                { id: 'fixed', name: 'Fixed Size (Sliding)', desc: 'Rigid token window with sliding step overlap' },
                { id: 'semantic', name: 'Semantic Splitter', desc: 'Threshold-based cosine distance embedding drift' },
                { id: 'markdown', name: 'Markdown / Header-Aware', desc: 'Preserves section headers and clinical tables' },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMethod(m.id)}
                  className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all ${
                    method === m.id
                      ? 'bg-cyan-950/60 border-cyan-500 text-cyan-200 shadow-sm'
                      : 'bg-slate-950/40 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                  }`}
                >
                  <div className="font-bold">{m.name}</div>
                  <div className="text-[10.5px] text-slate-400 mt-0.5">{m.desc}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Chunk Size Control */}
          <div className="bg-slate-950/50 p-4 rounded-2xl border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <Hash className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Chunk Size (Tokens)</span>
                </label>
                <span className="font-mono text-sm font-extrabold text-cyan-400 bg-cyan-950/80 px-2.5 py-0.5 rounded-lg border border-cyan-500/30">
                  {chunkSize} tokens
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mb-4">
                Recommended for Wilson CDSS: 512 tokens to balance symptom context and vector resolution.
              </p>
              
              <input
                type="range"
                min="128"
                max="2048"
                step="64"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />

              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
                <span>128</span>
                <span>512 (Optimal)</span>
                <span>1024</span>
                <span>2048</span>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-400 flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
              <span>Generates ~{chunks.length} total vectors for this document</span>
            </div>
          </div>

          {/* Chunk Overlap Control */}
          <div className="bg-slate-950/50 p-4 rounded-2xl border border-slate-800 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                  <RefreshCw className="w-3.5 h-3.5 text-amber-400" />
                  <span>Chunk Overlap (Tokens)</span>
                </label>
                <span className="font-mono text-sm font-extrabold text-amber-400 bg-amber-950/80 px-2.5 py-0.5 rounded-lg border border-amber-500/30">
                  {chunkOverlap} tokens
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mb-4">
                Ensures dosage instructions and criteria are not split across discrete chunk seams.
              </p>

              <input
                type="range"
                min="0"
                max="300"
                step="16"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
                className="w-full accent-amber-500 cursor-pointer"
              />

              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
                <span>0 tokens</span>
                <span>64 tokens</span>
                <span>128 tokens</span>
                <span>300 tokens</span>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-slate-800 text-[11px] text-slate-400 flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              <span>Overlap ratio: {((chunkOverlap / chunkSize) * 100).toFixed(1)}% of window</span>
            </div>
          </div>

        </div>

      </div>

      {/* Live Chunk Previewer & Overlap Inspector */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl text-white">
        
        {/* Document Selector & Stats Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-cyan-950/80 text-cyan-400 border border-cyan-800">
              <Eye className="w-4 h-4" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-white">Live Document Chunk Previewer</h4>
              <p className="text-[11px] text-slate-400">Inspecting exact partition indices and highlighted overlap boundaries</p>
            </div>
          </div>

          {/* Document Picker */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-semibold">Corpus Target:</span>
            <select
              value={selectedDocId}
              onChange={(e) => {
                setSelectedDocId(e.target.value);
                setSelectedChunkIndex(0);
              }}
              className="bg-slate-950 border border-slate-700 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:ring-1 focus:ring-cyan-500"
            >
              {documents.map(d => (
                <option key={d.id} value={d.id}>{d.title}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Chunk Inspection Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 pt-5">
          
          {/* Chunk Selector List (Left Column) */}
          <div className="lg:col-span-4 space-y-2 max-h-[460px] overflow-y-auto pr-2 custom-code-scroll">
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
              <span>Generated Chunks ({chunks.length})</span>
              <span>Tokens / Chk</span>
            </div>
            
            {chunks.map((chk, idx) => (
              <button
                key={chk.id}
                onClick={() => setSelectedChunkIndex(idx)}
                className={`w-full p-3 rounded-xl border text-left transition-all flex items-center justify-between ${
                  selectedChunkIndex === idx
                    ? 'bg-cyan-950/80 border-cyan-500 text-white shadow-md'
                    : 'bg-slate-950/40 border-slate-800/80 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                }`}
              >
                <div>
                  <div className="font-mono text-xs font-bold text-cyan-400">{chk.id}</div>
                  <div className="text-[11px] text-slate-400 truncate max-w-[180px] mt-0.5">
                    {chk.text.slice(0, 40)}...
                  </div>
                </div>
                <div className="text-right font-mono text-xs">
                  <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-bold">
                    {chk.tokenCount} tk
                  </span>
                </div>
              </button>
            ))}
          </div>

          {/* Chunk Detailed Content & Overlap Highlight (Right Column) */}
          <div className="lg:col-span-8 bg-slate-950 rounded-2xl border border-slate-800 p-5 flex flex-col justify-between">
            {activeChunk ? (
              <>
                <div>
                  {/* Active Chunk Header */}
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-xs font-extrabold text-cyan-400 bg-cyan-950 px-2.5 py-1 rounded border border-cyan-500/40">
                        {activeChunk.id}
                      </span>
                      <span className="text-xs text-slate-400 font-mono">
                        Tokens: <strong className="text-white">{activeChunk.tokenCount}</strong> • Chars: {activeChunk.charLength}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                        {chunkOverlap} Token Overlap Window
                      </span>
                    </div>
                  </div>

                  {/* Highlight Legend */}
                  <div className="mb-3 flex items-center gap-4 text-[11px] text-slate-400">
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded bg-amber-500/30 border border-amber-500" />
                      <span>Boundary Overlap (Shared with previous chunk)</span>
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded bg-slate-800 border border-slate-700" />
                      <span>Unique Chunk Body</span>
                    </span>
                  </div>

                  {/* Rendered Text with Highlighted Overlap */}
                  <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800/80 font-mono text-xs text-slate-200 leading-relaxed max-h-[300px] overflow-y-auto custom-code-scroll">
                    {activeChunk.hasOverlap && activeChunk.overlapText ? (
                      <div>
                        <span className="bg-amber-500/25 text-amber-200 border-b-2 border-amber-400 px-1 py-0.5 rounded font-bold">
                          {activeChunk.overlapText}
                        </span>
                        <span> {activeChunk.text.replace(activeChunk.overlapText, '').trim()}</span>
                      </div>
                    ) : (
                      <div>{activeChunk.text}</div>
                    )}
                  </div>
                </div>

                {/* Metadata Payload Card */}
                <div className="mt-4 pt-3 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono">
                  <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block text-[10px]">DOCUMENT ID</span>
                    <span className="text-slate-200 truncate block">{activeChunk.metadata.docId}</span>
                  </div>
                  <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block text-[10px]">CATEGORY</span>
                    <span className="text-cyan-400 block">{activeChunk.metadata.category}</span>
                  </div>
                  <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block text-[10px]">WORD RANGE</span>
                    <span className="text-slate-200 block">{activeChunk.metadata.startWord} – {activeChunk.metadata.endWord}</span>
                  </div>
                  <div className="p-2 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-slate-500 block text-[10px]">PAGE ESTIMATE</span>
                    <span className="text-slate-200 block">{activeChunk.metadata.pageRange}</span>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center py-12 text-slate-500 text-xs">
                Select a chunk on the left to preview boundary partitions.
              </div>
            )}
          </div>

        </div>

      </div>

    </div>
  );
}

