/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, ChangeEvent, DragEvent } from 'react';
import { Upload, Loader2, Image as ImageIcon, AlertCircle } from 'lucide-react';
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

interface RawPredictionItem {
  class?: string;
  label?: string;
  name?: string;
  confidence?: number;
}

interface RawPredictionResponse {
  predicted_label?: string;
  predicted_class?: string;
  confidence?: number;
  top_k?: RawPredictionItem[];
  top3?: RawPredictionItem[];
  error?: string;
}

interface PredictionItem {
  class: string;
  confidence: number;
}

interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  top3: PredictionItem[];
}

const DEFAULT_MODEL = 'inception_resnet_v1';

const normalizeTopItems = (items?: RawPredictionItem[]): PredictionItem[] => {
  return (items || [])
    .map((item) => {
      const label = item.class || item.label || item.name;
      if (!label) {
        return null;
      }
      return {
        class: label,
        confidence: typeof item.confidence === 'number' ? item.confidence : 0,
      };
    })
    .filter((item): item is PredictionItem => item !== null);
};

const normalizeResponse = (data: RawPredictionResponse): PredictionResponse | null => {
  const predictedClass = data.predicted_label || data.predicted_class;
  if (!predictedClass) {
    return null;
  }

  const confidence = typeof data.confidence === 'number' ? data.confidence : 0;
  const top3 = normalizeTopItems(data.top_k || data.top3);
  return {
    predicted_class: predictedClass,
    confidence,
    top3,
  };
};

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [model, setModel] = useState<string>(DEFAULT_MODEL);

  const apiParam = new URLSearchParams(window.location.search).get('api');
  const API_URL = apiParam || import.meta.env.VITE_API_URL || 'http://localhost:5000/predict';

  const formatClassName = (id: string) => {
    return CLASS_MAPPINGS[id] || id.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
  };

  const handleFileChange = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      setErrorMsg("Please select a valid image file.");
      return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setErrorMsg(null);
  };

  const onFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileChange(files[0]);
    }
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileChange(files[0]);
    }
  };

  const classifyImage = async () => {
    if (!file) return;

    setIsLoading(true);
    setErrorMsg(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);
    if (model) {
      formData.append("model", model);
    }

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });
      let data: RawPredictionResponse | null = null;
      try {
        data = await response.json();
      } catch (err) {
        data = null;
      }

      if (!response.ok) {
        const message = data && data.error ? data.error : `Server responded with ${response.status}`;
        throw new Error(message);
      }

      if (!data) {
        setErrorMsg("Invalid response from server.");
        return;
      }

      const normalized = data.error ? null : normalizeResponse(data);
      if (data.error) {
        setErrorMsg(data.error);
      } else if (!normalized) {
        setErrorMsg("Invalid response from server.");
      } else {
        setResult(normalized);
      }
    } catch (error) {
      const message = error instanceof Error
        ? error.message
        : `Failed to connect to backend server at ${API_URL}. Make sure it is running and CORS is enabled.`;
      setErrorMsg(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-900 flex items-center justify-center p-4 sm:p-8 font-sans relative overflow-hidden">
      
      {/* Aurora Background Orbs */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <motion.div 
          animate={{ x: [0, 30, -20, 0], y: [0, -40, 20, 0] }} 
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
          className="absolute -top-[10%] -left-[5%] w-[60vw] h-[60vw] rounded-full bg-indigo-300/30 mix-blend-multiply blur-[120px]" 
        />
        <motion.div 
          animate={{ x: [0, -40, 20, 0], y: [0, 30, -30, 0] }} 
          transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-[20%] -right-[10%] w-[50vw] h-[50vw] rounded-full bg-cyan-300/30 mix-blend-multiply blur-[120px]" 
        />
        <motion.div 
          animate={{ x: [0, 20, -40, 0], y: [0, 40, -10, 0] }} 
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
          className="absolute -bottom-[20%] left-[10%] w-[70vw] h-[70vw] rounded-full bg-purple-300/20 mix-blend-multiply blur-[120px]" 
        />
      </div>

      {/* Main Glass Panel */}
      <div className="w-full max-w-[1000px] bg-white/60 backdrop-blur-3xl border border-white rounded-[2.5rem] p-6 sm:p-12 shadow-[0_8px_40px_rgba(0,0,0,0.06)] relative z-10 flex flex-col md:grid md:grid-cols-[1fr_1.1fr] gap-x-16 gap-y-10 items-start">
        
        {/* Header Section */}
        <div className="md:col-span-2 text-center md:text-left">
          <h1 className="text-4xl sm:text-5xl font-black tracking-tighter text-slate-900 mb-3">
            Identity Architecture
          </h1>
          <p className="text-xs sm:text-sm font-bold tracking-[0.2em] uppercase text-indigo-600">
            Politician Classification • Neural Vision Model
          </p>
        </div>

        {/* Left Side: Upload & Action */}
        <div className="w-full flex flex-col h-full relative z-10">
          <div className="relative flex flex-col h-full">
            {/* Architectural Upload Zone */}
            <div 
              onClick={() => fileInputRef.current?.click()}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              className={`flex-1 border-2 border-dashed rounded-[2rem] p-8 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-500 overflow-hidden min-h-[280px] relative ${
                isDragging 
                  ? 'border-indigo-400 bg-indigo-50/50 scale-[1.02]' 
                  : 'border-slate-300/80 bg-white/40 hover:bg-white/60 hover:border-indigo-300 hover:shadow-[0_8px_30px_rgba(99,102,241,0.05)]'
              } ${preview ? 'border-transparent p-0' : ''}`}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={onFileSelect} 
                accept="image/*" 
                className="hidden" 
              />
              
              {preview ? (
                <div className="relative w-full h-full group">
                  <img src={preview} alt="Preview" className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" />
                  <div className="absolute inset-0 bg-slate-900/40 opacity-0 group-hover:opacity-100 transition-all duration-300 backdrop-blur-sm flex flex-col items-center justify-center">
                    <div className="bg-white/20 p-4 rounded-full backdrop-blur-md mb-3 text-white border border-white/30 transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
                      <ImageIcon size={24} />
                    </div>
                    <p className="text-white font-medium tracking-wide">Replace Image</p>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center transform transition-transform duration-300 hover:-translate-y-1">
                  <div className="p-5 bg-gradient-to-br from-white to-slate-50 border border-white shadow-sm rounded-2xl mb-6 relative group text-indigo-500">
                    <div className="absolute inset-0 bg-indigo-400/20 rounded-2xl blur-xl group-hover:bg-indigo-400/30 transition-colors" />
                    <Upload size={32} strokeWidth={1.5} className="relative z-10" />
                  </div>
                  <h3 className="text-base font-bold text-slate-800">Drop analysis target</h3>
                  <p className="mt-2 text-sm text-slate-500 font-medium">Click to browse filesystem</p>
                </div>
              )}
            </div>

            <div className="mt-6">
              <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">
                Model
              </label>
              <select
                value={model}
                onChange={(event) => setModel(event.target.value)}
                className="mt-2 w-full rounded-2xl border border-slate-200 bg-white/60 px-4 py-3 text-sm font-semibold text-slate-800 shadow-sm backdrop-blur transition focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-200"
              >
                <option value="inception_resnet_v1">Inception ResNet V1 (VGGFace2)</option>
                <option value="inception_resnet_v1_casia">Inception ResNet V1 (CASIA-WebFace)</option>
                <option value="resnet50">ResNet50 (Rescaled)</option>
              </select>
              <p className="mt-3 text-[10px] text-slate-500 font-semibold tracking-wide">
                API: <span className="font-mono text-slate-400">{API_URL}</span>
              </p>
            </div>

            {/* Error Message */}
            <AnimatePresence>
              {errorMsg && (
                <motion.div 
                  initial={{ opacity: 0, y: -10, height: 0 }}
                  animate={{ opacity: 1, y: 0, height: 'auto' }}
                  exit={{ opacity: 0, y: -10, height: 0 }}
                  className="mt-6 bg-red-50 border border-red-100 text-red-600 p-4 rounded-2xl flex items-center gap-3 text-sm font-semibold shadow-sm overflow-hidden"
                >
                  <AlertCircle size={18} className="shrink-0" />
                  <p>{errorMsg}</p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Action Button */}
            <button
              onClick={classifyImage}
              disabled={!file || isLoading}
              className={`w-full mt-6 py-4 px-6 rounded-2xl font-bold text-base transition-all duration-300 flex items-center justify-center gap-3 tracking-wide ${
                !file 
                  ? 'bg-slate-100/50 text-slate-400 cursor-not-allowed shadow-inner' 
                  : 'bg-slate-900 hover:bg-slate-800 text-white shadow-[0_8px_20px_rgba(0,0,0,0.15)] hover:shadow-[0_12px_25px_rgba(0,0,0,0.2)] hover:-translate-y-0.5 active:scale-[0.98]'
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin text-indigo-400" size={20} />
                  <span>Processing Matrix...</span>
                </>
              ) : (
                'Execute Classification'
              )}
            </button>
          </div>
        </div>

        {/* Right Side: Results Panel */}
        <div className="w-full flex-1 md:min-h-[400px] flex flex-col justify-center">
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div 
                key="results"
                initial={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
                animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
                exit={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                className="flex flex-col gap-8 h-full"
              >
                {/* Primary Metric */}
                <div className="bg-white/40 backdrop-blur-xl rounded-[2rem] p-8 border border-white/60 shadow-[0_4px_24px_rgba(0,0,0,0.02)] relative overflow-hidden group">
                  <div className="absolute top-0 left-0 w-1.5 h-full bg-gradient-to-b from-indigo-500 to-cyan-400" />
                  <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 mb-3 ml-2">Primary Match</div>
                  <div className="text-3xl sm:text-4xl font-black text-slate-900 mb-6 ml-2 tracking-tight">
                    {formatClassName(result.predicted_class!)}
                  </div>
                  
                  <div className="ml-2">
                    <div className="flex justify-between items-end mb-2">
                      <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Confidence Level</span>
                      <span className="text-lg font-black text-indigo-600">
                        {(result.confidence! * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-3 w-full bg-slate-200/50 rounded-full overflow-hidden shadow-inner">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: `${result.confidence! * 100}%` }}
                        transition={{ duration: 1.5, ease: [0.22, 1, 0.36, 1] }}
                        className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-400 rounded-full"
                      />
                    </div>
                  </div>
                </div>

                {/* Sub-metrics */}
                {result.top3 && result.top3.length > 0 && (
                  <div className="flex flex-col gap-4">
                    <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400 ml-2">Probability Distribution</div>
                    <div className="flex flex-col gap-3">
                      {result.top3.map((pred, idx) => {
                        if (pred.class === result.predicted_class && idx === 0) return null;
                        
                        return (
                          <div key={idx} className="bg-white/40 border border-white/60 rounded-2xl p-4 flex items-center justify-between shadow-sm">
                            <span className="text-sm font-bold text-slate-700">
                              {formatClassName(pred.class)}
                            </span>
                            <div className="flex items-center gap-4 w-1/2">
                              <div className="h-2 w-full bg-slate-200/50 rounded-full overflow-hidden flex-1">
                                <motion.div 
                                  initial={{ width: 0 }}
                                  animate={{ width: `${pred.confidence * 100}%` }}
                                  transition={{ duration: 1.2, delay: 0.1 + (idx * 0.1), ease: "easeOut" }}
                                  className="h-full bg-slate-400 rounded-full"
                                />
                              </div>
                              <span className="text-xs font-bold text-slate-500 w-10 text-right">
                                {(pred.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
                
                <div className="mt-auto pt-4 flex items-center justify-between border-t border-slate-200/50">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-[10px] font-bold tracking-widest text-slate-500 uppercase">System Active</span>
                  </div>
                  <span className="text-[10px] font-mono text-slate-400">REST // Local Inference</span>
                </div>
              </motion.div>
            ) : (
               <motion.div 
                 key="empty"
                 initial={{ opacity: 0 }}
                 animate={{ opacity: 1 }}
                 exit={{ opacity: 0 }}
                 className="hidden md:flex flex-col items-center justify-center h-full min-h-[350px] rounded-[2.5rem] bg-white/20 border border-white/40 shadow-sm"
               >
                 <div className="p-6 bg-white/40 rounded-full mb-6 relative border border-white">
                   <ImageIcon size={40} className="text-slate-400" />
                 </div>
                 <p className="text-slate-400 text-sm font-bold tracking-wider uppercase">Standing By</p>
                 <p className="text-slate-400 text-xs mt-2 max-w-[200px] text-center leading-relaxed">System awaits visual input to begin identity extraction.</p>
               </motion.div>
            )}
          </AnimatePresence>
        </div>

      </div>
      
      {/* Footer Details */}
      <div className="absolute bottom-6 w-full text-center pointer-events-none z-0">
        <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-400">
          Neural Architecture &copy; 2024
        </p>
      </div>

    </div>
  );
}
