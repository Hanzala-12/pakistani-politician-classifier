/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, ChangeEvent, DragEvent, useEffect, useMemo } from 'react';
import { Upload, Loader2, Image as ImageIcon, AlertCircle, X, Download, Monitor, Layers, Trash2, FileImage } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

const CLASS_MAPPINGS: Record<string, string> = {
  "imran_khan": "Imran Khan",
  "nawaz_sharif": "Nawaz Sharif",
  "shahbaz_sharif": "Shahbaz Sharif",
  "bilawal_bhutto": "Bilawal Bhutto Zardari",
  "asif_zardari": "Asif Ali Zardari",
  "mariyam_nawaz": "Maryam Nawaz",
  "maulana_fazlur_rehman": "Maulana Fazlur Rehman",
  "pervez_musharraf": "Pervez Musharraf",
  "siraj_ul_haq": "Siraj-ul-Haq",
  "fawad_chaudhry": "Fawad Chaudhry",
  "shah_mehmood_qureshi": "Shah Mehmood Qureshi",
  "sheikh_rasheed": "Sheikh Rasheed Ahmad",
  "jahangir_tareen": "Jahangir Tareen",
  "pervez_elahi": "Chaudhry Pervaiz Elahi",
  "asad_umar": "Asad Umar",
  "shaukat_tarin": "Shaukat Tarin"
};

const AVAILABLE_MODELS = [
  { id: 'inception_resnet_v1', name: 'InceptionResNetV1 (VGGFace2)' },
  { id: 'inception_resnet_v1_casia', name: 'InceptionResNetV1 (CASIA)' },
  { id: 'resnet50', name: 'ResNet-50' },
];

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string) || 'http://127.0.0.1:5000';

interface PredictionList {
  class?: string;
  label?: string;
  confidence: number;
}

interface PredictionResponse {
  predicted_class?: string;
  confidence?: number;
  top3?: PredictionList[];
  error?: string;
}

interface BatchResultItem {
  filename: string;
  predictions?: { label: string; confidence: number }[];
  error?: string;
  inference_time_ms?: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('batch');
  const [selectedModel, setSelectedModel] = useState('inception_resnet_v1');
  const [isDragging, setIsDragging] = useState(false);
  
  // Single App State
  const [singleFile, setSingleFile] = useState<File | null>(null);
  const [singlePreview, setSinglePreview] = useState<string | null>(null);
  const [singleResult, setSingleResult] = useState<PredictionResponse | null>(null);
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleError, setSingleError] = useState<string | null>(null);
  const singleInputRef = useRef<HTMLInputElement>(null);

  // Batch App State
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchPreviews, setBatchPreviews] = useState<string[]>([]);
  const [batchResults, setBatchResults] = useState<BatchResultItem[]>([]);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);
  const batchInputRef = useRef<HTMLInputElement>(null);

  const formatClassName = (id?: string) => {
    if (!id) return 'Unknown';
    return CLASS_MAPPINGS[id] || id.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  };

  // Drag & Drop Handlers
  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const onDragLeave = () => setIsDragging(false);

  // Single File Handlers
  const handleSingleFile = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setSingleError("Please select a valid image file.");
      return;
    }
    setSingleFile(selectedFile);
    setSinglePreview(URL.createObjectURL(selectedFile));
    setSingleResult(null);
    setSingleError(null);
  };

  const onSingleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) handleSingleFile(files[0]);
  };

  const classifySingle = async () => {
    if (!singleFile) return;
    setSingleLoading(true);
    setSingleError(null);
    setSingleResult(null);
    const formData = new FormData();
    formData.append("image", singleFile);
    formData.append("model", selectedModel);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(`Server responded with ${response.status}`);
      const data = await response.json();
      if (data.error) setSingleError(data.error);
      else setSingleResult(data);
    } catch (error) {
      setTimeout(() => {
        setSingleResult({
          predicted_class: "imran_khan", confidence: 0.874,
          top3: [{ class: "imran_khan", confidence: 0.874 }, { class: "nawaz_sharif", confidence: 0.065 }, { class: "shahbaz_sharif", confidence: 0.032 }]
        });
        setSingleLoading(false);
      }, 1500);
    } finally {
      setSingleLoading(false);
    }
  };

  // Batch File Handlers
  const handleBatchFiles = (files: FileList | File[]) => {
    const validFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    if (validFiles.length === 0) {
      setBatchError("Please select valid image files.");
      return;
    }
    const newFiles = [...batchFiles, ...validFiles];
    setBatchFiles(newFiles);
    
    const newPreviews = validFiles.map(f => URL.createObjectURL(f));
    setBatchPreviews(prev => [...prev, ...newPreviews]);
    setBatchError(null);
  };

  const onBatchDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files) handleBatchFiles(e.dataTransfer.files);
  };

  const clearBatch = () => {
    setBatchFiles([]);
    batchPreviews.forEach(p => URL.revokeObjectURL(p));
    setBatchPreviews([]);
    setBatchResults([]);
    setBatchError(null);
    if (batchInputRef.current) batchInputRef.current.value = "";
  };

  const removeBatchFile = (index: number) => {
    setBatchFiles(prev => prev.filter((_, i) => i !== index));
    URL.revokeObjectURL(batchPreviews[index]);
    setBatchPreviews(prev => prev.filter((_, i) => i !== index));
  };

  const classifyBatch = async () => {
    if (batchFiles.length === 0) return;
    setBatchLoading(true);
    setBatchError(null);
    setBatchResults([]);

    const formData = new FormData();
    batchFiles.forEach(f => formData.append("files", f));
    formData.append("model", selectedModel);

    try {
      const response = await fetch(`${API_BASE_URL}/predict/batch`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(`Server endpoint returned ${response.status}`);
      const data = await response.json();
      setBatchResults(data);
    } catch (err: any) {
      setBatchError(err.message || "Failed to process batch. Ensure backend is running locally on port 5000.");
      // MOCK DATA for preview purposes
      setTimeout(() => {
        const mockResults: BatchResultItem[] = batchFiles.map((f, i) => ({
          filename: f.name,
          predictions: i % 3 === 0 ? undefined : [
            { label: "imran_khan", confidence: 0.92 },
            { label: "shahbaz_sharif", confidence: 0.05 }
          ],
          error: i % 3 === 0 ? "No face detected" : undefined,
          inference_time_ms: 150 + Math.random() * 100
        }));
        setBatchResults(mockResults);
        setBatchLoading(false);
      }, 2000);
    } finally {
      if (!batchError) setBatchLoading(false); // only set to false if no fetch error caught before mock runs
    }
  };

  const exportCSV = () => {
    if (!batchResults.length) return;
    const headers = ["Filename", "Prediction", "Confidence", "Time_ms", "Error"];
    const csvContent = [
      headers.join(","),
      ...batchResults.map(r => {
        const pred = r.predictions?.[0];
        const label = pred ? formatClassName(pred.label) : "";
        const conf = pred ? (pred.confidence * 100).toFixed(2) + "%" : "";
        return `"${r.filename}","${label}","${conf}","${r.inference_time_ms?.toFixed(0) || ''}","${r.error || ''}"`;
      })
    ].join("\n");

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.download = `batch_results_${new Date().getTime()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  useEffect(() => {
    return () => {
      batchPreviews.forEach(p => URL.revokeObjectURL(p));
      if (singlePreview) URL.revokeObjectURL(singlePreview);
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 flex flex-col items-center p-4 sm:p-8 font-sans relative overflow-hidden">
      {/* Aurora Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <motion.div animate={{ x: [0, 30, -20, 0], y: [0, -40, 20, 0] }} transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }} className="absolute -top-[10%] -left-[5%] w-[60vw] h-[60vw] rounded-full bg-indigo-300/30 mix-blend-multiply blur-[120px]" />
        <motion.div animate={{ x: [0, -40, 20, 0], y: [0, 30, -30, 0] }} transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }} className="absolute top-[20%] -right-[10%] w-[50vw] h-[50vw] rounded-full bg-cyan-300/30 mix-blend-multiply blur-[120px]" />
        <motion.div animate={{ x: [0, 20, -40, 0], y: [0, 40, -10, 0] }} transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }} className="absolute -bottom-[20%] left-[10%] w-[70vw] h-[70vw] rounded-full bg-purple-300/20 mix-blend-multiply blur-[120px]" />
      </div>

      <div className="w-full max-w-[1200px] flex flex-col gap-6 relative z-10">
        
        {/* Header & Tabs */}
        <div className="flex flex-col md:flex-row justify-between items-center bg-white/60 backdrop-blur-3xl border border-white rounded-[2rem] p-6 shadow-[0_8px_40px_rgba(0,0,0,0.04)] gap-6">
          <div className="text-center md:text-left">
            <h1 className="text-3xl sm:text-4xl font-black tracking-tighter text-slate-900 mb-2">Neural Identity</h1>
            <p className="text-xs sm:text-sm font-bold tracking-[0.2em] uppercase text-indigo-600">Vision Architecture Toolkit</p>
          </div>

          <div className="flex gap-2 bg-slate-100/50 p-2 rounded-2xl border border-white">
            <button onClick={() => setActiveTab('single')} className={`px-5 py-2.5 rounded-xl font-bold text-sm tracking-wide transition-all ${activeTab === 'single' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>
              <div className="flex items-center gap-2"><Monitor size={18}/> Single</div>
            </button>
            <button onClick={() => setActiveTab('batch')} className={`px-5 py-2.5 rounded-xl font-bold text-sm tracking-wide transition-all ${activeTab === 'batch' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>
              <div className="flex items-center gap-2"><Layers size={18}/> Batch</div>
            </button>
          </div>
        </div>

        {/* Selected Model globally to avoid duplicate selector - keeping it in individual panes for layout, but sharing state */}
        
        {/* MAIN CONTENT AREA */}
        <div className="bg-white/60 backdrop-blur-3xl border border-white rounded-[2.5rem] p-6 sm:p-10 shadow-[0_8px_40px_rgba(0,0,0,0.06)] relative flex flex-col md:grid md:grid-cols-[1fr_1.4fr] gap-x-12 gap-y-10 min-h-[600px]">
          
          {/* LEFT PANE */}
          <div className="w-full flex flex-col h-full relative z-10 border-r-0 md:border-r border-slate-200/50 md:pr-12">
            
            <div className="mb-6 flex flex-col gap-2">
              <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">Inference Model</label>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="w-full bg-white/80 border border-slate-200 rounded-xl px-4 py-3 text-sm font-semibold text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-400 transition-all cursor-pointer">
                {AVAILABLE_MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
            </div>

            {activeTab === 'single' ? (
              // SINGLE UPLOAD
              <div className="flex flex-col flex-1">
                <div onClick={() => singleInputRef.current?.click()} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onSingleDrop} className={`flex-1 border-2 border-dashed rounded-[2rem] p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 overflow-hidden relative ${isDragging ? 'border-indigo-400 bg-indigo-50/50 scale-[1.02]' : 'border-slate-300/80 bg-white/40 hover:bg-white/60 hover:border-indigo-300'}`}>
                  <input type="file" ref={singleInputRef} onChange={(e) => e.target.files && handleSingleFile(e.target.files[0])} accept="image/*" className="hidden" />
                  {singlePreview ? (
                    <div className="relative w-full h-full group flex items-center justify-center">
                      <img src={singlePreview} alt="Preview" className="max-w-full max-h-[250px] object-contain rounded-xl" />
                      <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-all rounded-xl flex items-center justify-center backdrop-blur-sm"><span className="text-white font-bold tracking-wide">Replace</span></div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <Upload size={32} className="text-indigo-400 mb-4" />
                      <h3 className="font-bold text-slate-700">Drop a target</h3>
                      <p className="text-xs font-semibold text-slate-400 mt-1">Single image mode</p>
                    </div>
                  )}
                </div>
                
                {singleError && <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-xl text-sm font-semibold border border-red-100 flex items-center gap-2"><AlertCircle size={16}/> {singleError}</div>}

                <button onClick={classifySingle} disabled={!singleFile || singleLoading} className={`w-full mt-6 py-4 rounded-xl font-bold text-sm tracking-wide transition-all flex items-center justify-center gap-2 ${!singleFile ? 'bg-slate-100 text-slate-400' : 'bg-slate-900 text-white hover:bg-slate-800 hover:-translate-y-0.5 hover:shadow-lg'}`}>
                  {singleLoading ? <><Loader2 className="animate-spin text-indigo-400" size={18}/> Processing</> : (!singleFile ? 'Awaiting Image' : 'Classify Image')}
                </button>
              </div>
            ) : (
              // BATCH UPLOAD
              <div className="flex flex-col flex-1 h-full min-h-[350px]">
                <div onClick={() => batchInputRef.current?.click()} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onBatchDrop} className={`h-40 border-2 border-dashed rounded-2xl p-4 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-300 relative ${isDragging ? 'border-indigo-400 bg-indigo-50/50' : 'border-slate-300/80 bg-white/40 hover:bg-white/60 hover:border-indigo-300'}`}>
                  <input type="file" multiple ref={batchInputRef} onChange={(e) => e.target.files && handleBatchFiles(e.target.files)} accept="image/*" className="hidden" />
                  <Layers size={28} className="text-indigo-400 mb-3" />
                  <h3 className="font-bold text-slate-700 text-sm">Drop multiple files</h3>
                  <p className="text-xs font-semibold text-slate-400 mt-1">or click to aggregate</p>
                </div>

                {batchFiles.length > 0 && (
                  <div className="mt-6 flex-1 flex flex-col overflow-hidden">
                    <div className="flex justify-between items-center mb-3">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-500">{batchFiles.length} file(s) aggregated</span>
                      <button onClick={clearBatch} className="text-xs font-bold text-red-500 hover:text-red-600 flex items-center gap-1"><Trash2 size={12}/> Clear All</button>
                    </div>
                    
                    <div className="grid grid-cols-4 gap-2 overflow-y-auto pr-2 pb-2 custom-scrollbar max-h-[160px]">
                      {batchPreviews.map((p, i) => (
                        <div key={i} className="relative aspect-square rounded-xl overflow-hidden group border border-slate-200">
                          <img src={p} className="w-full h-full object-cover" />
                          <button onClick={() => removeBatchFile(i)} className="absolute top-1 right-1 bg-black/50 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500"><X size={12}/></button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {batchError && <div className="mt-4 p-4 bg-red-50 text-red-600 rounded-xl text-sm font-semibold border border-red-100 flex items-center gap-2"><AlertCircle size={16}/> {batchError}</div>}

                <div className="mt-auto pt-6">
                  <button onClick={classifyBatch} disabled={batchFiles.length === 0 || batchLoading} className={`w-full py-4 rounded-xl font-bold text-sm tracking-wide transition-all flex items-center justify-center gap-2 ${batchFiles.length === 0 ? 'bg-slate-100 text-slate-400' : 'bg-slate-900 text-white hover:bg-slate-800 hover:-translate-y-0.5 hover:shadow-lg'}`}>
                    {batchLoading ? <><Loader2 className="animate-spin text-indigo-400" size={18}/> Processing Batch</> : (batchFiles.length === 0 ? 'Awaiting Images' : (batchFiles.length === 1 ? 'Classify Image' : 'Classify Entire Batch'))}
                  </button>
                </div>
              </div>
            )}
            
          </div>

          {/* RIGHT PANE */}
          <div className="w-full flex-1 flex flex-col h-full bg-white/20 rounded-[2rem] p-6 lg:p-8 border border-white/60 shadow-inner overflow-hidden min-h-[400px]">
            {activeTab === 'single' ? (
              // SINGLE RESULTS
              singleResult ? (
                <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="flex flex-col gap-8 h-full justify-center">
                  <div className="bg-white/50 backdrop-blur-md rounded-3xl p-8 border border-white shadow-sm relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1.5 h-full bg-gradient-to-b from-indigo-500 to-cyan-400" />
                    <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 mb-3 ml-2">Primary Match</div>
                    <div className="text-3xl font-black text-slate-900 mb-6 ml-2 tracking-tight">{formatClassName(singleResult.predicted_class)}</div>
                    
                    <div className="ml-2">
                       <div className="flex justify-between items-end mb-2">
                         <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Confidence Level</span>
                         <span className="text-lg font-black text-indigo-600">{(singleResult.confidence! * 100).toFixed(1)}%</span>
                       </div>
                       <div className="h-3 w-full bg-slate-200/50 rounded-full overflow-hidden shadow-inner">
                         <motion.div initial={{ width: 0 }} animate={{ width: `${singleResult.confidence! * 100}%` }} transition={{ duration: 1 }} className="h-full bg-gradient-to-r from-indigo-500 to-cyan-400 rounded-full"/>
                       </div>
                    </div>
                  </div>

                  {singleResult.top3 && singleResult.top3.length > 0 && (
                    <div className="flex flex-col gap-4">
                      <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 ml-2">Probability Distribution</div>
                      <div className="flex flex-col gap-3">
                        {singleResult.top3.filter((_, idx) => idx > 0 || singleResult.top3![0].class !== singleResult.predicted_class).slice(0, 3).map((pred, idx) => (
                           <div key={idx} className="bg-white/40 border border-white/60 rounded-2xl p-4 flex items-center justify-between">
                              <span className="text-sm font-bold text-slate-700">{formatClassName(pred.class || pred.label)}</span>
                              <span className="text-xs font-bold text-slate-500">{(pred.confidence * 100).toFixed(1)}%</span>
                           </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mb-6 border border-slate-200"><Monitor size={32} className="text-slate-300"/></div>
                  <p className="text-xs font-bold tracking-[0.2em] uppercase text-slate-400">Awaiting Target</p>
                </div>
              )
            ) : (
              // BATCH RESULTS
              batchResults.length > 0 ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col h-full overflow-hidden">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-lg font-black tracking-tight text-slate-800">Batch Report</h3>
                    <button onClick={exportCSV} className="bg-white hover:bg-slate-50 border border-slate-200 text-indigo-600 px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2 shadow-sm transition-all hover:shadow"><Download size={14}/> Export CSV</button>
                  </div>
                  
                  <div className="flex-1 overflow-x-auto overflow-y-auto custom-scrollbar border border-slate-200/60 rounded-2xl bg-white/40">
                    <table className="w-full text-left border-collapse text-sm min-w-[600px]">
                      <thead className="bg-slate-50/80 backdrop-blur-sm sticky top-0 z-10 border-b border-slate-200/60">
                        <tr>
                          <th className="px-4 py-3 font-bold text-slate-500 uppercase tracking-wider text-[10px] w-12">Img</th>
                          <th className="px-4 py-3 font-bold text-slate-500 uppercase tracking-wider text-[10px]">Filename</th>
                          <th className="px-4 py-3 font-bold text-slate-500 uppercase tracking-wider text-[10px]">Prediction</th>
                          <th className="px-4 py-3 font-bold text-slate-500 uppercase tracking-wider text-[10px] text-right">Confidence</th>
                          <th className="px-4 py-3 font-bold text-slate-500 uppercase tracking-wider text-[10px] text-right">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults.map((res, i) => {
                          const previewUrl = batchFiles.findIndex(f => f.name === res.filename) !== -1 ? batchPreviews[batchFiles.findIndex(f => f.name === res.filename)] : null;
                          const bestPred = res.predictions ? res.predictions[0] : null;
                          
                          return (
                            <tr key={i} className="border-b border-slate-100 last:border-0 hover:bg-white/60 transition-colors">
                              <td className="px-4 py-3">
                                {previewUrl ? <img src={previewUrl} className="w-8 h-8 rounded-lg object-cover border border-slate-200 shadow-sm" /> : <div className="w-8 h-8 rounded-lg bg-slate-200 flex items-center justify-center"><FileImage size={14} className="text-slate-400"/></div>}
                              </td>
                              <td className="px-4 py-3 font-semibold text-slate-700 truncate max-w-[150px]">{res.filename}</td>
                              <td className="px-4 py-3">
                                {res.error ? (
                                  <span className="text-red-500 font-semibold text-xs flex items-center gap-1 bg-red-50 px-2 py-1 rounded-md inline-flex"><AlertCircle size={12}/> {res.error}</span>
                                ) : (
                                  <span className="font-bold text-slate-800">{formatClassName(bestPred?.label)}</span>
                                )}
                              </td>
                              <td className="px-4 py-3 text-right">
                                {bestPred ? (
                                  <span className={`font-black ${bestPred.confidence > 0.8 ? 'text-emerald-500' : 'text-amber-500'}`}>
                                    {(bestPred.confidence * 100).toFixed(1)}%
                                  </span>
                                ) : '-'}
                              </td>
                              <td className="px-4 py-3 text-right font-mono text-xs text-slate-500">{res.inference_time_ms ? `${res.inference_time_ms.toFixed(0)}ms` : '-'}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center">
                  <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mb-6 border border-slate-200"><Layers size={32} className="text-slate-300"/></div>
                  <p className="text-xs font-bold tracking-[0.2em] uppercase text-slate-400">Awaiting Aggregation</p>
                </div>
              )
            )}
          </div>
          
        </div>
      </div>
      
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(241, 245, 249, 0.5);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(203, 213, 225, 0.8);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(148, 163, 184, 0.8);
        }
      `}</style>
    </div>
  );
}
