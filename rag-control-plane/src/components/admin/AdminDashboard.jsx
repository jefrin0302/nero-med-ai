import React, { useState } from 'react';
import { useAuth } from '../../context/AuthContext';
import { 
  PlusCircle, 
  BookOpen, 
  ShieldCheck, 
  FileText, 
  MessageSquare, 
  Sparkles, 
  ExternalLink,
  X,
  Layers,
  Calendar,
  Lock,
  Tag
} from 'lucide-react';
import SystemHealthCards from './SystemHealthCards';
import KnowledgeInventoryTable from './KnowledgeInventoryTable';
import HighLevelQueryLogs from './HighLevelQueryLogs';
import DocumentIngestionModal from './DocumentIngestionModal';
import { INITIAL_DOCUMENTS, INITIAL_QUERY_LOGS, SYSTEM_KPIS } from '../../services/ragPipelineService';

export default function AdminDashboard() {
  const { currentUser, navigateTo } = useAuth();
  
  const [documents, setDocuments] = useState(INITIAL_DOCUMENTS);
  const [queryLogs, setQueryLogs] = useState(INITIAL_QUERY_LOGS);
  const [kpis, setKpis] = useState(SYSTEM_KPIS);
  const [activeTab, setActiveTab] = useState('inventory'); // 'inventory' | 'queries'
  const [isIngestModalOpen, setIsIngestModalOpen] = useState(false);
  const [inspectedDoc, setInspectedDoc] = useState(null);

  // Document management actions
  const handleAddDocument = (newDoc) => {
    setDocuments(prev => [newDoc, ...prev]);
  };

  const handleToggleStatus = (docId) => {
    setDocuments(prev => prev.map(doc => {
      if (doc.id === docId) {
        const nextStatus = doc.status === 'Active' ? 'Disabled' : 'Active';
        return { ...doc, status: nextStatus };
      }
      return doc;
    }));
  };

  const handleDeleteDocument = (docId) => {
    setDocuments(prev => prev.filter(doc => doc.id !== docId));
  };

  const handleFlagLog = (logId, reason) => {
    setQueryLogs(prev => prev.map(log => {
      if (log.id === logId) {
        return { ...log, flagged: true, flagReason: reason };
      }
      return log;
    }));
  };

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-slate-50 dark:bg-[#090d16] p-4 sm:p-6 lg:p-8 transition-colors">
      <div className="max-w-7xl mx-auto">
        
        {/* Executive Banner */}
        <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white rounded-3xl p-6 sm:p-8 shadow-xl border border-indigo-900/40 relative overflow-hidden mb-6">
          <div className="absolute top-0 right-0 -mt-10 -mr-10 w-80 h-80 bg-indigo-500/15 rounded-full blur-3xl pointer-events-none" />
          
          <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 mb-3">
                <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
                <span>CLINICAL KNOWLEDGE GOVERNANCE & CONSENSUS REPOSITORY</span>
              </div>
              <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white leading-tight">
                Knowledge Base Management
              </h1>
              <p className="text-xs sm:text-sm text-slate-300 mt-2 max-w-2xl leading-relaxed">
                Welcome back, {currentUser?.name}. Curate approved medical literature, monitor automated clinical citations, and inspect guideline health across the hospital decision support system.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => setIsIngestModalOpen(true)}
                className="px-5 py-3 rounded-2xl bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs sm:text-sm transition-all shadow-lg shadow-indigo-600/30 flex items-center gap-2 group"
              >
                <PlusCircle className="w-4 h-4 group-hover:rotate-90 transition-transform" />
                <span>Ingest New Document</span>
              </button>
            </div>
          </div>
        </div>

        {/* System Health KPI Overview */}
        <SystemHealthCards kpis={kpis} documents={documents} />

        {/* Section Tabs */}
        <div className="flex items-center gap-2 mb-6 border-b border-slate-200 dark:border-slate-800 pb-3">
          <button
            onClick={() => setActiveTab('inventory')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
              activeTab === 'inventory'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200/60 dark:hover:bg-slate-800'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>Knowledge Inventory ({documents.length})</span>
          </button>

          <button
            onClick={() => setActiveTab('queries')}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
              activeTab === 'queries'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-200/60 dark:hover:bg-slate-800'
            }`}
          >
            <MessageSquare className="w-4 h-4" />
            <span>Physician Query Auditing ({queryLogs.length})</span>
          </button>
        </div>

        {/* Active Tab View */}
        {activeTab === 'inventory' ? (
          <KnowledgeInventoryTable
            documents={documents}
            onToggleStatus={handleToggleStatus}
            onDeleteDocument={handleDeleteDocument}
            onViewDetails={(doc) => setInspectedDoc(doc)}
          />
        ) : (
          <HighLevelQueryLogs
            logs={queryLogs}
            onFlagLog={handleFlagLog}
          />
        )}

      </div>

      {/* Document Ingestion Modal */}
      <DocumentIngestionModal
        isOpen={isIngestModalOpen}
        onClose={() => setIsIngestModalOpen(false)}
        onAddDocument={handleAddDocument}
      />

      {/* Document Details Inspector Drawer/Modal */}
      {inspectedDoc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/75 backdrop-blur-sm animate-fadeIn">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl max-w-2xl w-full shadow-2xl p-6 relative max-h-[85vh] flex flex-col">
            <div className="flex items-start justify-between pb-4 border-b border-slate-200 dark:border-slate-800">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 flex items-center justify-center border border-indigo-200 dark:border-indigo-800">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900 dark:text-white leading-tight">
                    {inspectedDoc.title}
                  </h3>
                  <div className="flex items-center gap-2 mt-1 text-xs text-slate-400">
                    <span>{inspectedDoc.filename}</span>
                    <span>•</span>
                    <span className="font-semibold text-indigo-500">{inspectedDoc.category}</span>
                  </div>
                </div>
              </div>
              <button 
                onClick={() => setInspectedDoc(null)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="my-4 overflow-y-auto pr-2 space-y-4 text-xs">
              <div>
                <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-1">
                  Executive Clinical Summary
                </h4>
                <p className="text-slate-600 dark:text-slate-300 leading-relaxed bg-slate-50 dark:bg-slate-800/60 p-3 rounded-xl border border-slate-200 dark:border-slate-700">
                  {inspectedDoc.description}
                </p>
              </div>

              <div>
                <h4 className="font-bold text-slate-800 dark:text-slate-200 mb-1">
                  Extracted Representative Text
                </h4>
                <div className="p-3 bg-slate-100 dark:bg-slate-950 rounded-xl font-mono text-[11px] text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-800 leading-relaxed whitespace-pre-wrap">
                  {inspectedDoc.content}
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-slate-100 dark:border-slate-800">
                <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 uppercase font-bold">Total Chunks</div>
                  <div className="text-sm font-extrabold text-indigo-600 dark:text-indigo-400 font-mono mt-0.5">
                    {inspectedDoc.chunkCount}
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 uppercase font-bold">Token Estimate</div>
                  <div className="text-sm font-extrabold text-slate-800 dark:text-slate-200 font-mono mt-0.5">
                    {inspectedDoc.tokenCount.toLocaleString()}
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 uppercase font-bold">Usage Citations</div>
                  <div className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400 font-mono mt-0.5">
                    {inspectedDoc.usageCount}
                  </div>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 text-center">
                  <div className="text-[10px] text-slate-400 uppercase font-bold">Avg Latency</div>
                  <div className="text-sm font-extrabold text-slate-800 dark:text-slate-200 font-mono mt-0.5">
                    {inspectedDoc.avgLatencyMs} ms
                  </div>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-200 dark:border-slate-800 flex justify-end">
              <button
                onClick={() => setInspectedDoc(null)}
                className="px-4 py-2 text-xs font-bold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:bg-slate-200 rounded-xl"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

