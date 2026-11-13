import React, { useState } from 'react';
import { 
  UploadCloud, 
  FileText, 
  X, 
  CheckCircle2, 
  AlertCircle, 
  Lock, 
  Tag, 
  Sparkles,
  ArrowRight
} from 'lucide-react';

export default function DocumentIngestionModal({ isOpen, onClose, onAddDocument }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('Hepatology');
  const [accessPermission, setAccessPermission] = useState('Public CDSS');
  const [description, setDescription] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStep, setProcessStep] = useState('');

  if (!isOpen) return null;

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (file) => {
    setSelectedFile(file);
    if (!title) {
      // Auto-generate clean title from filename
      const cleanName = file.name
        .replace(/\.[^/.]+$/, "")
        .replace(/_/g, " ")
        .replace(/-/g, " ");
      setTitle(cleanName.charAt(0).toUpperCase() + cleanName.slice(1));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title) return;

    setIsProcessing(true);
    setProcessStep('Extracting clean clinical prose & citations...');

    setTimeout(() => {
      setProcessStep('Executing semantic verification & security scan...');
      setTimeout(() => {
        setProcessStep('Publishing to Active Knowledge Inventory...');
        setTimeout(() => {
          const newDoc = {
            id: `doc-${Date.now().toString(36)}`,
            title: title,
            filename: selectedFile ? selectedFile.name : `${title.replace(/\s+/g, '_')}.pdf`,
            category: category,
            format: selectedFile ? selectedFile.name.split('.').pop().toUpperCase() : 'PDF',
            fileSize: selectedFile ? `${(selectedFile.size / (1024 * 1024)).toFixed(1)} MB` : '2.4 MB',
            uploadDate: new Date().toISOString().split('T')[0],
            status: 'Active',
            chunkCount: Math.floor(40 + Math.random() * 80),
            tokenCount: Math.floor(20000 + Math.random() * 45000),
            usageCount: 0,
            avgLatencyMs: 32,
            similarityThreshold: 0.82,
            accessPermission: accessPermission,
            description: description || 'Clinical knowledge document ingested via Administrative Console.',
            content: `Clinical reference document: ${title}. Synthesized guidance for multi-organ Wilson disease diagnosis, copper biochemical bounds, and monitoring guidelines.`
          };

          onAddDocument(newDoc);
          setIsProcessing(false);
          onClose();
        }, 600);
      }, 700);
    }, 600);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/75 backdrop-blur-sm animate-fadeIn">
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl max-w-xl w-full shadow-2xl p-6 relative overflow-hidden">
        
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-100 dark:border-slate-800">
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2">
              <UploadCloud className="w-5 h-5 text-indigo-500" />
              <span>Ingest Clinical Knowledge Document</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Upload guidelines, consensus statements, or research protocols into the CDSS knowledge base
            </p>
          </div>
          <button 
            onClick={onClose}
            disabled={isProcessing}
            className="p-1 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Processing Overlay */}
        {isProcessing ? (
          <div className="py-12 flex flex-col items-center justify-center text-center">
            <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center mb-4">
              <Sparkles className="w-8 h-8 text-indigo-500 animate-spin" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-1">
              Processing & Ingesting Document
            </h3>
            <p className="text-xs text-indigo-600 dark:text-indigo-400 font-medium animate-pulse">
              {processStep}
            </p>
            <div className="w-64 h-2 bg-slate-100 dark:bg-slate-800 rounded-full mt-6 overflow-hidden">
              <div className="h-full bg-indigo-600 rounded-full animate-indeterminate" style={{ width: '80%' }} />
            </div>
            <p className="text-[11px] text-slate-400 mt-3">
              Automated chunking and embedding are executed securely in the background.
            </p>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="mt-4 space-y-4">
            
            {/* Drag & Drop Area */}
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-2xl p-6 text-center transition-all cursor-pointer ${
                dragActive
                  ? 'border-indigo-500 bg-indigo-500/5'
                  : 'border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600 bg-slate-50/50 dark:bg-slate-800/30'
              }`}
            >
              <input
                type="file"
                id="file-upload"
                accept=".pdf,.docx,.doc,.txt,.md"
                onChange={handleFileInput}
                className="hidden"
              />
              <label htmlFor="file-upload" className="cursor-pointer block">
                <div className="w-12 h-12 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800/60 flex items-center justify-center mx-auto mb-3">
                  <UploadCloud className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
                </div>
                {selectedFile ? (
                  <div>
                    <span className="text-xs font-bold text-slate-900 dark:text-white block">
                      {selectedFile.name}
                    </span>
                    <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold mt-1 inline-flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      Ready ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
                    </span>
                  </div>
                ) : (
                  <div>
                    <span className="text-xs font-bold text-slate-800 dark:text-slate-200 block">
                      Click to browse or drag & drop clinical file
                    </span>
                    <span className="text-[11px] text-slate-400 mt-1 block">
                      Supported formats: PDF, DOCX, TXT, Markdown (Max 25 MB)
                    </span>
                  </div>
                )}
              </label>
            </div>

            {/* Document Title */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Document Title & Consensus Name *
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. EASL Guidelines for Pediatric Wilson Disease (2026)"
                required
                className="w-full px-3.5 py-2.5 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-xl text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            {/* Category & Access Permission Grid */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1 flex items-center gap-1">
                  <Tag className="w-3 h-3 text-slate-400" />
                  <span>Clinical Category</span>
                </label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-xl text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="Hepatology">Hepatology & Liver</option>
                  <option value="Pharmacotherapy">Pharmacotherapy & Chelation</option>
                  <option value="Genetics">Genetics (ATP7B)</option>
                  <option value="Pediatrics">Pediatrics & Family Screening</option>
                  <option value="Neurology">Neurology & Neuroimaging</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1 flex items-center gap-1">
                  <Lock className="w-3 h-3 text-slate-400" />
                  <span>Access Permission</span>
                </label>
                <select
                  value={accessPermission}
                  onChange={(e) => setAccessPermission(e.target.value)}
                  className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-xl text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="Public CDSS">Public CDSS (All Clinicians)</option>
                  <option value="Restricted Research">Restricted Research Access</option>
                  <option value="Chief Clinicians Only">Chief Clinicians Only</option>
                </select>
              </div>
            </div>

            {/* Clinical Summary */}
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1">
                Clinical Context & Scope Summary
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                placeholder="Briefly describe key diagnostic cutoffs, protocols, or patient cohorts covered..."
                className="w-full px-3.5 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-xl text-xs text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            {/* Footer Actions */}
            <div className="pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
              <span className="text-[11px] text-slate-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                <span>Technical chunking handled automatically</span>
              </span>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 text-xs font-semibold text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!title}
                  className="px-5 py-2.5 text-xs font-bold bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl transition-all shadow-md shadow-indigo-600/20 flex items-center gap-2 disabled:opacity-50"
                >
                  <span>Process & Publish</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

          </form>
        )}

      </div>
    </div>
  );
}

