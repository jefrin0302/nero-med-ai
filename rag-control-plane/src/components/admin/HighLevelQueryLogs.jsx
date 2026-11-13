import React, { useState } from 'react';
import { 
  Search, 
  Flag, 
  CheckCircle2, 
  AlertCircle, 
  Clock, 
  BookOpen, 
  User, 
  ExternalLink,
  MessageSquare,
  Sparkles
} from 'lucide-react';

export default function HighLevelQueryLogs({ logs, onFlagLog }) {
  const [selectedLog, setSelectedLog] = useState(null);
  const [flagReason, setFlagReason] = useState('');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [filterQuery, setFilterQuery] = useState('');

  const filteredLogs = logs.filter(log => 
    log.question.toLowerCase().includes(filterQuery.toLowerCase()) ||
    log.clinician.toLowerCase().includes(filterQuery.toLowerCase()) ||
    log.citedDocuments.some(doc => doc.toLowerCase().includes(filterQuery.toLowerCase()))
  );

  const openFlagModal = (log) => {
    setSelectedLog(log);
    setFlagReason(log.flagReason || '');
    setIsModalOpen(true);
  };

  const handleSaveFlag = (e) => {
    e.preventDefault();
    if (selectedLog) {
      onFlagLog(selectedLog.id, flagReason);
      setIsModalOpen(false);
      setSelectedLog(null);
    }
  };

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl shadow-sm p-6 mb-6">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-5 border-b border-slate-100 dark:border-slate-800">
        <div>
          <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-indigo-500" />
            <span>High-Level Clinical Query Feed & Verification</span>
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Audit clinical citations, source documents retrieved by copilot, and flag outdated consensus guidance
          </p>
        </div>

        {/* Search */}
        <div className="relative w-full sm:w-64">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            placeholder="Filter questions or citations..."
            className="w-full pl-9 pr-3 py-2 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-xl text-xs text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
        </div>
      </div>

      {/* Query Feed Cards */}
      <div className="mt-5 space-y-3.5">
        {filteredLogs.length === 0 ? (
          <div className="text-center py-8 text-slate-400 text-xs">
            No query logs match your filter criteria.
          </div>
        ) : (
          filteredLogs.map((log) => {
            return (
              <div
                key={log.id}
                className={`p-4 rounded-2xl border transition-all ${
                  log.flagged
                    ? 'bg-amber-50/50 dark:bg-amber-950/20 border-amber-300 dark:border-amber-800/60'
                    : 'bg-slate-50/70 dark:bg-slate-800/40 border-slate-200/80 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
                }`}
              >
                <div className="flex flex-col md:flex-row md:items-start justify-between gap-3">
                  
                  {/* Left Question & Clinician Meta */}
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        <span>{log.timestamp}</span>
                      </span>
                      <span className="text-slate-300 dark:text-slate-700">•</span>
                      <span className="text-[11px] font-semibold text-indigo-600 dark:text-indigo-400 flex items-center gap-1">
                        <User className="w-3 h-3" />
                        <span>{log.clinician}</span>
                      </span>
                      {log.flagged && (
                        <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/20 text-amber-700 dark:text-amber-300 border border-amber-400/40 flex items-center gap-1">
                          <Flag className="w-2.5 h-2.5 fill-amber-500" />
                          <span>Flagged for Review</span>
                        </span>
                      )}
                    </div>

                    <h4 className="text-sm font-bold text-slate-900 dark:text-white leading-snug">
                      "{log.question}"
                    </h4>

                    {/* Cited Source Documents Pills */}
                    <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
                      <span className="text-[11px] text-slate-400 font-semibold mr-1">
                        Cited Sources:
                      </span>
                      {log.citedDocuments.map((docTitle, i) => (
                        <span
                          key={i}
                          className="px-2.5 py-1 rounded-lg text-[11px] font-medium bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 flex items-center gap-1 shadow-2xs"
                        >
                          <BookOpen className="w-3 h-3 text-indigo-500" />
                          <span className="truncate max-w-[260px]">{docTitle}</span>
                        </span>
                      ))}
                    </div>

                    {/* Flag Reason note if flagged */}
                    {log.flagged && log.flagReason && (
                      <div className="mt-2 text-xs bg-amber-100/70 dark:bg-amber-900/40 border border-amber-200 dark:border-amber-800 rounded-lg p-2 text-amber-900 dark:text-amber-200">
                        <span className="font-bold">Admin Flag Note:</span> {log.flagReason}
                      </div>
                    )}
                  </div>

                  {/* Right Metrics & Flag Button */}
                  <div className="flex md:flex-col items-center md:items-end justify-between gap-2 shrink-0 pt-2 md:pt-0 border-t md:border-t-0 border-slate-200 dark:border-slate-700">
                    <div className="flex items-center gap-2">
                      <div className="text-right">
                        <div className="text-[11px] text-slate-400">Similarity</div>
                        <div className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400">
                          {(log.similarityScore * 100).toFixed(0)}% Match
                        </div>
                      </div>
                      <div className="text-right pl-2 border-l border-slate-200 dark:border-slate-700">
                        <div className="text-[11px] text-slate-400">Latency</div>
                        <div className="text-xs font-mono font-bold text-slate-700 dark:text-slate-300">
                          {log.responseLatencyMs}ms
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => openFlagModal(log)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                        log.flagged
                          ? 'bg-amber-100 dark:bg-amber-900/50 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-700'
                          : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:text-amber-600 dark:hover:text-amber-400'
                      }`}
                    >
                      <Flag className={`w-3.5 h-3.5 ${log.flagged ? 'fill-amber-500' : ''}`} />
                      <span>{log.flagged ? 'Edit Flag' : 'Flag Outdated'}</span>
                    </button>
                  </div>

                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Flag Outdated Modal */}
      {isModalOpen && selectedLog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/75 backdrop-blur-sm animate-fadeIn">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl max-w-md w-full shadow-2xl p-6">
            <h3 className="text-base font-bold text-slate-900 dark:text-white mb-2 flex items-center gap-2">
              <Flag className="w-5 h-5 text-amber-500" />
              <span>Flag Clinical Document for Update</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-4 leading-relaxed">
              Tag this citation as outdated or in need of guideline renewal. Technical teams will be notified to re-evaluate the source document.
            </p>

            <form onSubmit={handleSaveFlag} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                  Reason / Medical Advisory Note
                </label>
                <textarea
                  value={flagReason}
                  onChange={(e) => setFlagReason(e.target.value)}
                  required
                  rows={3}
                  placeholder="e.g. 2024 AASLD consensus supersedes this penicillamine titration cutoff. Upload new draft."
                  className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-xl text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-amber-500"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2 border-t border-slate-100 dark:border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-3.5 py-1.5 text-xs font-semibold text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 text-xs font-bold bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-xl shadow-md transition-all flex items-center gap-1.5"
                >
                  <Flag className="w-3.5 h-3.5" />
                  <span>Save Clinical Flag</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </div>
  );
}

