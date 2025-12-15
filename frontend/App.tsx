
import React, { useState, useEffect, useRef } from 'react';
import { SystemState, LogEntry } from './types';
import { ControlPanel } from './components/ControlPanel';
import { SystemVisualizer } from './components/SystemVisualizer';
import { DocumentationModal } from './components/DocumentationModal';
import { SimulationModal } from './components/SimulationModal';
import { NanoMicroscopeModal } from './components/NanoMicroscopeModal';
import { analyzeSystemState } from './services/geminiService';
import { BrainCircuit, Info, Sparkles, Cpu, Zap, Share2, BookOpen } from 'lucide-react';

const App: React.FC = () => {
  const [state, setState] = useState<SystemState>({
    v1: 65,
    v2: 20,
    v3: 85,
  });
  
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDocsOpen, setIsDocsOpen] = useState(false);
  const [isSimOpen, setIsSimOpen] = useState(false);
  const [isNanoOpen, setIsNanoOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  // Ref to track previous state for generating diff logs
  const prevStateRef = useRef<SystemState>(state);

  const addLog = (type: 'SYS' | 'ISA' | 'WARN' | 'CMD', message: string) => {
    const now = new Date();
    const timeString = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;
    const id = Math.random().toString(36).substring(7);
    setLogs(prev => [{ id, timestamp: timeString, type, message }, ...prev].slice(0, 50));
  };

  // Initial boot log
  useEffect(() => {
    addLog('SYS', 'LuminaCore BIOS v2.1 initialized');
    addLog('SYS', 'Calibrating AWG Grating Temperature...');
    addLog('SYS', 'System Ready. Waiting for Optical Bus...');
  }, []);

  // Effect to generate L-ISA logs on state change
  useEffect(() => {
    const prev = prevStateRef.current;
    if (prev.v1 !== state.v1) {
       addLog('ISA', `LOAD_DAC [ADDR:0x1A] VAL:${(state.v1/100).toFixed(2)}V // Eu-Red`);
    }
    if (prev.v2 !== state.v2) {
       addLog('ISA', `LOAD_DAC [ADDR:0x1B] VAL:${(state.v2/100).toFixed(2)}V // Tb-Grn`);
    }
    if (prev.v3 !== state.v3) {
       addLog('ISA', `LOAD_DAC [ADDR:0x1C] VAL:${(state.v3/100).toFixed(2)}V // Tm-Blu`);
    }
    prevStateRef.current = state;
  }, [state]);

  const handleStateChange = (key: keyof SystemState, value: number) => {
    setState(prev => ({ ...prev, [key]: value }));
  };

  const handleScenarioSelect = (scenario: 'GLASSES' | 'DRONE' | 'CLOUD') => {
      if (scenario === 'GLASSES') {
          addLog('SYS', 'Loading Mission Profile: SMART_GLASSES (Low Power)');
          addLog('CMD', 'CONFIG_MODE: STANDBY_MONITOR');
          setState({ v1: 0, v2: 0, v3: 45 }); // Blue light only (Wake word monitoring)
      } else if (scenario === 'DRONE') {
          addLog('SYS', 'Loading Mission Profile: DRONE_NAV (High Perf)');
          addLog('CMD', 'CONFIG_MODE: OBSTACLE_AVOIDANCE');
          setState({ v1: 85, v2: 80, v3: 10 }); // Red/Green high for Matrix/Conv
      } else if (scenario === 'CLOUD') {
          addLog('SYS', 'Loading Mission Profile: DATACENTER (Max Throughput)');
          addLog('CMD', 'CONFIG_MODE: LLM_INFERENCE');
          setState({ v1: 95, v2: 95, v3: 95 }); // All channels blazing
      }
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setAnalysis(null);
    addLog('SYS', 'Initiating Gemini 2.5 Analysis Link...');
    try {
      const result = await analyzeSystemState(state);
      setAnalysis(result);
      addLog('SYS', 'Analysis Data Received.');
    } catch (e) {
      console.error(e);
      setAnalysis("Failed to analyze system state.");
      addLog('WARN', 'Gemini API Connection Failed.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30">
      
      <DocumentationModal isOpen={isDocsOpen} onClose={() => setIsDocsOpen(false)} />
      <SimulationModal isOpen={isSimOpen} onClose={() => setIsSimOpen(false)} />
      <NanoMicroscopeModal isOpen={isNanoOpen} onClose={() => setIsNanoOpen(false)} />

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-slate-950/90 backdrop-blur-xl border-b border-slate-800">
        <div className="max-w-[1400px] mx-auto px-6 h-18 flex items-center justify-between py-3">
          <div className="flex items-center gap-4">
            <div className="p-2.5 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl shadow-lg shadow-indigo-500/20 text-white">
              <BrainCircuit size={24} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-white leading-none">Lumina<span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Core</span></h1>
              <div className="flex items-center gap-2 mt-1">
                 <span className="px-1.5 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20 text-[10px] font-mono text-indigo-300">v2025.2</span>
                 <p className="text-xs text-slate-500 font-medium">Edge AI Photonic Simulator</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-4 md:gap-6">
             <div className="hidden md:flex items-center gap-4 text-xs font-mono text-slate-500">
                <span className="flex items-center gap-1"><Zap size={12}/> 100 TOPS/W</span>
                <span className="flex items-center gap-1"><Share2 size={12}/> PASSIVE AWG</span>
                <span className="flex items-center gap-1"><Cpu size={12}/> 4nm NODE</span>
             </div>
             <div className="h-8 w-px bg-slate-800 mx-2 hidden md:block"></div>
             
             <button
               onClick={() => setIsDocsOpen(true)}
               className="text-xs font-medium text-slate-300 hover:text-white transition-colors flex items-center gap-1.5 px-3 py-1.5 rounded-lg hover:bg-slate-800"
             >
                <BookOpen size={14} className="text-slate-400" />
                White Paper
             </button>

             <a 
                href="https://github.com/google/generative-ai-js" 
                target="_blank" 
                rel="noreferrer"
                className="text-xs font-medium text-slate-400 hover:text-white transition-colors flex items-center gap-1.5 bg-slate-900 py-1.5 px-3 rounded-full border border-slate-800 hover:border-slate-600"
              >
                <Sparkles size={12} className="text-indigo-400" />
                Gemini Analysis
              </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="pt-28 pb-12 px-6 max-w-[1400px] mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 h-[calc(100vh-2rem)]">
        
        {/* Left Column: Controls (3 cols) */}
        <div className="lg:col-span-3 h-full flex flex-col gap-6 overflow-hidden">
          <ControlPanel 
            state={state} 
            onChange={handleStateChange}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            onScenarioSelect={handleScenarioSelect}
            logs={logs}
            onOpenSimulation={() => setIsSimOpen(true)}
            onOpenNanoScope={() => setIsNanoOpen(true)}
          />
        </div>

        {/* Center Column: Visualization (6 cols) */}
        <div className="lg:col-span-6 h-[500px] lg:h-full flex flex-col">
          <div className="flex-grow h-full">
            <SystemVisualizer state={state} />
          </div>
        </div>

        {/* Right Column: AI Analysis (3 cols) */}
        <div className="lg:col-span-3 h-full flex flex-col">
          <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-0 h-full flex flex-col relative overflow-hidden shadow-xl">
             
             {/* Header */}
             <div className="p-6 border-b border-slate-800 bg-slate-900/50">
                <div className="flex items-center gap-2 mb-1">
                    <Sparkles className="text-amber-400" size={18} />
                    <h2 className="text-sm font-bold text-slate-100 uppercase tracking-wide">Physicist Analysis</h2>
                </div>
                <p className="text-xs text-slate-500">Gemini 2.5 Flash • Optical Core</p>
             </div>

             <div className="flex-grow overflow-y-auto p-6 bg-gradient-to-b from-slate-900/50 to-slate-950/50">
                {analysis ? (
                  <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
                    <div className="p-5 bg-indigo-500/10 border border-indigo-500/20 rounded-xl shadow-inner shadow-black/20">
                      <p className="text-sm leading-7 text-indigo-100 font-light font-sans">
                        {analysis}
                      </p>
                    </div>
                    <div className="mt-4 flex gap-2">
                        <span className="px-2 py-1 rounded bg-slate-800 text-[10px] text-slate-400 border border-slate-700">Latency: 120ms</span>
                        <span className="px-2 py-1 rounded bg-slate-800 text-[10px] text-slate-400 border border-slate-700">Confidence: 99.8%</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center text-slate-500 gap-6">
                    <div className="relative">
                        <div className="absolute inset-0 bg-indigo-500/20 blur-xl rounded-full"></div>
                        <BrainCircuit size={64} className="relative text-slate-700" />
                    </div>
                    <div className="space-y-2">
                        <p className="text-sm font-medium text-slate-400">Awaiting Optical Data</p>
                        <p className="text-xs max-w-[220px] mx-auto text-slate-600">
                        Configure the nanocrystal voltages and run analysis to simulate a specific AI workload.
                        </p>
                    </div>
                  </div>
                )}
             </div>
             
             {/* Footer decorative */}
             <div className="h-1 w-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 opacity-50"></div>
          </div>
        </div>

      </main>
    </div>
  );
};

export default App;
