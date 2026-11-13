import React, { useState } from 'react';
import { 
  Search, 
  Filter, 
  Trash2, 
  Power, 
  FileText, 
  CheckCircle2, 
  AlertTriangle, 
  Clock, 
  ExternalLink,
  ShieldCheck,
  Eye
} from 'lucide-react';

export default function KnowledgeInventoryTable({ 
  documents, 
  onToggleStatus, 
  onDeleteDocument, 
  onViewDetails 
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('ALL');

  const filteredDocs = documents.filter(doc => {
    const matchesSearch = doc.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          doc.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          doc.category.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCat = categoryFilter === 'ALL' || doc.category === categoryFilter;
    return matchesSearch && matchesCat;
  });

  const categories = ['ALL', 'Hepatology', 'Pharmacotherapy', 'Genetics', 'Pediatrics'];

  return (
    <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl shadow-sm overflow-hidden mb-6">
      
      {/* Table Header & Search Filter Controls */}
      <div className="p-5 border-b border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-base font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-indigo-500" />
            <span>Knowledge Base Inventory</span>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
              {filteredDocs.length} Documents
            </span>
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Active guidelines, therapeutic manuals, and consensus references available to the RAG copilot
          </p>
        </div>

        {/* Filter Controls */}
        <div className="flex items-center gap-2.5 w-full sm:w-auto">
          {/* Search Box */}
          <div className="relative flex-1 sm:w-64">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search title or category..."
              className="w-full pl-9 pr-3 py-2 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-xl text-xs text-slate-900 dark:text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Category Dropdown */}
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="px-3 py-2 bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 rounded-xl text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {categories.map(cat => (
              <option key={cat} value={cat}>
                {cat === 'ALL' ? 'All Categories' : cat}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Responsive Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-50 dark:bg-slate-800/50 text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider text-[11px] border-b border-slate-200 dark:border-slate-800">
            <tr>
              <th className="py-3.5 px-5">Document Title & Filename</th>
              <th className="py-3.5 px-4">Category</th>
              <th className="py-3.5 px-4">Upload Date</th>
              <th className="py-3.5 px-4">Status</th>
              <th className="py-3.5 px-4 text-center">Usage Count</th>
              <th className="py-3.5 px-4 text-center">Permissions</th>
              <th className="py-3.5 px-5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-slate-800 text-slate-700 dark:text-slate-300">
            {filteredDocs.length === 0 ? (
              <tr>
                <td colSpan="7" className="py-10 text-center text-slate-400 text-xs">
                  No documents found matching "{searchTerm}". Try clearing your filters.
                </td>
              </tr>
            ) : (
              filteredDocs.map((doc) => {
                const isActive = doc.status === 'Active';
                return (
                  <tr 
                    key={doc.id}
                    className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    {/* Title & Filename */}
                    <td className="py-3.5 px-5">
                      <div className="flex items-start gap-3">
                        <div className="p-2 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800/60 text-indigo-600 dark:text-indigo-400 shrink-0 mt-0.5">
                          <FileText className="w-4 h-4" />
                        </div>
                        <div>
                          <div className="font-bold text-slate-900 dark:text-white leading-snug">
                            {doc.title}
                          </div>
                          <div className="text-[11px] text-slate-400 font-mono flex items-center gap-2 mt-0.5">
                            <span>{doc.filename}</span>
                            <span>•</span>
                            <span>{doc.fileSize}</span>
                            <span>•</span>
                            <span className="text-indigo-500 font-semibold">{doc.chunkCount} Chunks</span>
                          </div>
                        </div>
                      </div>
                    </td>

                    {/* Category */}
                    <td className="py-3.5 px-4 whitespace-nowrap">
                      <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
                        {doc.category}
                      </span>
                    </td>

                    {/* Upload Date */}
                    <td className="py-3.5 px-4 whitespace-nowrap text-slate-500 font-mono text-[11px]">
                      {doc.uploadDate}
                    </td>

                    {/* Status Badge */}
                    <td className="py-3.5 px-4 whitespace-nowrap">
                      {doc.status === 'Active' && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                          <span>Active</span>
                        </span>
                      )}
                      {doc.status === 'Processing' && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-50 dark:bg-amber-950/50 text-amber-600 dark:text-amber-400 border border-amber-200 dark:border-amber-800">
                          <Clock className="w-3 h-3 text-amber-500" />
                          <span>Processing</span>
                        </span>
                      )}
                      {doc.status === 'Disabled' && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-slate-100 dark:bg-slate-800 text-slate-500 border border-slate-300 dark:border-slate-700">
                          <span>Disabled</span>
                        </span>
                      )}
                      {doc.status === 'Failed' && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-rose-50 dark:bg-rose-950/50 text-rose-600 dark:text-rose-400 border border-rose-200 dark:border-rose-800">
                          <AlertTriangle className="w-3 h-3 text-rose-500" />
                          <span>Failed</span>
                        </span>
                      )}
                    </td>

                    {/* Usage Count */}
                    <td className="py-3.5 px-4 text-center whitespace-nowrap">
                      <span className="font-extrabold text-slate-900 dark:text-white font-mono">
                        {doc.usageCount.toLocaleString()}
                      </span>
                      <span className="text-[10px] text-slate-400 block">queries</span>
                    </td>

                    {/* Permissions */}
                    <td className="py-3.5 px-4 text-center whitespace-nowrap">
                      <span className="text-[10.5px] font-semibold text-slate-600 dark:text-slate-400">
                        {doc.accessPermission || 'Public CDSS'}
                      </span>
                    </td>

                    {/* Actions */}
                    <td className="py-3.5 px-5 text-right whitespace-nowrap">
                      <div className="flex items-center justify-end gap-1.5">
                        
                        {/* Enable/Disable Toggle */}
                        <button
                          onClick={() => onToggleStatus(doc.id)}
                          title={isActive ? "Disable from live search" : "Enable for live search"}
                          className={`p-1.5 rounded-lg border transition-all ${
                            isActive
                              ? 'border-emerald-300 dark:border-emerald-800/80 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-950/50'
                              : 'border-slate-300 dark:border-slate-700 text-slate-400 hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800'
                          }`}
                        >
                          <Power className="w-3.5 h-3.5" />
                        </button>

                        {/* View Excerpt Details */}
                        {onViewDetails && (
                          <button
                            onClick={() => onViewDetails(doc)}
                            title="Inspect Document Scope"
                            className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-500 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                          >
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                        )}

                        {/* Delete Button */}
                        <button
                          onClick={() => {
                            if (window.confirm(`Are you sure you want to delete "${doc.title}" from the knowledge base?`)) {
                              onDeleteDocument(doc.id);
                            }
                          }}
                          title="Delete from Knowledge Base"
                          className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-700 text-slate-400 hover:text-rose-600 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-950/30 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

    </div>
  );
}

