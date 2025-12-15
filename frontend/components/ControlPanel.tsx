
import React, { useRef, useEffect } from 'react';
import { SystemState, LogEntry } from '../types';
import { Cpu, Zap, Activity, Grid3x3, Waves, Layers, Glasses, Plane, Server, Terminal, PlayCircle, Microscope } from 'lucide-react';

interface ControlPanelProps {
  state: SystemState;
  onChange: (key: keyof SystemState, value: number) => void;
  onAnalyze: () => void;
  isAnalyzing: boolean;
  onScenarioSelect: (scenario: 'GLASSES' | 'DRONE' | 'CLOUD') => void;
  logs: LogEntry[];
  onOpenSimulation: () => void;
  onOpenNanoScope: () => void;
}

const Slider = ({ 
  label, 
  subLabel,
  value, 
  color, 
  onChange, 
  icon: Icon,
  description
}: { 
  label: string; 
  subLabel: string;
  value: number; 
  color: string; 
  onChange: (val: number) => void;
  icon: React.ElementType;
  description: string;
}) => (
  <div className="mb-4 group">
    <div className="flex justify-between items-center mb-1">
      <div className="flex items-center gap-2">
        <div className={`p-1.5 rounded-lg bg-${color}-500/10 text-${color}-400 border border-${color}-500/20`}>
          <Icon size={14} />
        </div>
        <div>
          <span className="text-xs font-semibold text-slate-200 block">{label}</span>
          <span className="text-[10px] font-mono text-${color}-400">{subLabel}</span>
        </div>
      </div>
      <span className={`font-mono text-xs text-${color}-400 font-bold`}>{value}%</span>
    </div>
    <div className="relative h-1.5 w-full">
        <div className="absolute top-0 left-0 bottom-0 right-0 bg-slate-800 rounded-lg"></div>
        <div 
            className={`absolute top-0 left-0 bottom-0 bg-${color}-500 rounded-lg opacity-30 transition-all duration-300`} 
            style={{ width: `${value}%` }}
        ></div>
        <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`absolute top-0 left-0 w-full h-2 opacity-0 cursor-pointer z-10`}
        />
        <div 
            className={`absolute top-0 h-1.5 w-1.5 bg-${color}-400 rounded-full shadow-[0_0_10px_rgba(0,0,0,0.5)] transition-all duration-75 pointer-events-none`}
            style={{ left: `calc(${value}% - 3px)` }}
        />
    </div>
  </div>
);

export const ControlPanel: React.FC<ControlPanelProps> = ({ state, onChange, onAnalyze, isAnalyzing, onScenarioSelect, logs, onOpenSimulation, onOpenNanoScope }) => {
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll terminal
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Simulated metrics
  const totalThroughput = ((state.v1 * 1.5 + state.v2 * 1.2 + state.v3) / 10).toFixed(1);
  const efficiency = (200 + (300 - (state.v1 + state.v2 + state.v3))).toFixed(0);
  const temp = (22 + (state.v1 + state.v2 + state.v3) * 0.05).toFixed(1);

  return (
    <div className="bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl p-5 h-full flex flex-col shadow-xl overflow-hidden">
      
      {/* Mission Profiles (Scenarios) */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-3 text-slate-400">
           <Zap size={14} className="text-yellow-400" />
           <span className="text-xs font-bold uppercase tracking-wider">Mission Profile</span>
        </div>
        <div className="grid grid-cols-3 gap-2">
            <button 
                onClick={() => onScenarioSelect('GLASSES')}
                className="flex flex-col items-center justify-center gap-1.5 p-3 rounded-xl bg-slate-800/50 hover:bg-cyan-500/10 border border-slate-700 hover:border-cyan-500/50 transition-all group"
            >
                <Glasses size={18} className="text-slate-400 group-hover:text-cyan-400" />
                <span className="text-[10px] text-slate-400 group-hover:text-cyan-300 font-medium">Glasses</span>
            </button>
            <button 
                onClick={() => onScenarioSelect('DRONE')}
                className="flex flex-col items-center justify-center gap-1.5 p-3 rounded-xl bg-slate-800/50 hover:bg-emerald-500/10 border border-slate-700 hover:border-emerald-500/50 transition-all group"
            >
                <Plane size={18} className="text-slate-400 group-hover:text-emerald-400" />
                <span className="text-[10px] text-slate-400 group-hover:text-emerald-300 font-medium">Drone</span>
            </button>
            <button 
                onClick={() => onScenarioSelect('CLOUD')}
                className="flex flex-col items-center justify-center gap-1.5 p-3 rounded-xl bg-slate-800/50 hover:bg-purple-500/10 border border-slate-700 hover:border-purple-500/50 transition-all group"
            >
                <Server size={18} className="text-slate-400 group-hover:text-purple-400" />
                <span className="text-[10px] text-slate-400 group-hover:text-purple-300 font-medium">Cloud</span>
            </button>
        </div>
      </div>

      <div className="w-full h-px bg-slate-800 mb-6" />

      {/* Manual Controls */}
      <div className="flex-grow space-y-4 overflow-y-auto pr-2 custom-scrollbar">
        <div className="flex items-center justify-between mb-2 text-slate-400">
           <div className="flex items-center gap-2">
                <Cpu size={14} className="text-indigo-400" />
                <span className="text-xs font-bold uppercase tracking-wider">Manual Override</span>
           </div>
           {/* Nano Scope Button */}
           <button 
                onClick={onOpenNanoScope}
                className="p-1.5 rounded-md bg-slate-800 hover:bg-purple-900/30 text-purple-400 border border-slate-700 hover:border-purple-500/50 transition-all"
                title="Open 3D Nanocrystal Lab"
           >
               <Microscope size={14} />
           </button>
        </div>
        
        <Slider
          label="Pixel R (Matrix)"
          subLabel="Eu³⁺ • λ1"
          description="Drivers for Matrix Multiplier"
          value={state.v1}
          color="red"
          icon={Grid3x3}
          onChange={(v) => onChange('v1', v)}
        />
        <Slider
          label="Pixel G (Conv)"
          subLabel="Tb³⁺ • λ2"
          description="Drivers for Convolution Core"
          value={state.v2}
          color="green"
          icon={Layers}
          onChange={(v) => onChange('v2', v)}
        />
        <Slider
          label="Pixel B (Activ)"
          subLabel="Tm³⁺ • λ3"
          description="Drivers for Non-linear Unit"
          value={state.v3}
          color="blue"
          icon={Waves}
          onChange={(v) => onChange('v3', v)}
        />
        
        {/* Telemetry */}
        <div className="mt-6 p-4 bg-slate-950/50 rounded-xl border border-slate-800/50 backdrop-blur-sm">
           <div className="grid grid-cols-1 gap-2 font-mono text-xs">
             <div className="flex justify-between items-center group">
                <span className="text-slate-500">OPS:</span>
                <span className="text-emerald-400 font-bold">{totalThroughput} POPS</span>
             </div>
             <div className="flex justify-between items-center group">
                <span className="text-slate-500">EFF:</span>
                <span className="text-cyan-400 font-bold">{efficiency} TOPS/W</span>
             </div>
              <div className="flex justify-between items-center group">
                <span className="text-slate-500">TMP:</span>
                <span className={`${Number(temp) > 40 ? 'text-red-400' : 'text-slate-300'} font-bold`}>
                  {temp}°C
                </span>
             </div>
           </div>
        </div>

        {/* Digital Twin Launcher */}
        <button
          onClick={onOpenSimulation}
          className="w-full mt-4 py-3 px-4 bg-slate-800 hover:bg-emerald-900/30 text-emerald-400 hover:text-emerald-300 border border-slate-700 hover:border-emerald-500/50 rounded-xl flex items-center justify-center gap-2 transition-all text-xs font-bold uppercase tracking-wide group"
        >
            <PlayCircle size={16} className="group-hover:scale-110 transition-transform" />
            Launch Digital Twin
        </button>
      </div>

      <div className="w-full h-px bg-slate-800 my-4" />

      {/* L-ISA Terminal */}
      <div className="flex flex-col h-32 bg-black/50 rounded-lg border border-slate-800 font-mono text-[10px] overflow-hidden relative">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900 border-b border-slate-800">
              <Terminal size={10} className="text-emerald-500" />
              <span className="text-slate-400 font-bold">L-ISA INSTRUCTION LOG</span>
          </div>
          <div className="flex-grow overflow-y-auto p-2 space-y-1 text-slate-300">
              {logs.map((log) => (
                  <div key={log.id} className="flex gap-2">
                      <span className="text-slate-600 shrink-0">[{log.timestamp}]</span>
                      <span className={`
                        shrink-0 font-bold
                        ${log.type === 'SYS' ? 'text-blue-400' : ''}
                        ${log.type === 'ISA' ? 'text-emerald-400' : ''}
                        ${log.type === 'WARN' ? 'text-red-400' : ''}
                        ${log.type === 'CMD' ? 'text-purple-400' : ''}
                      `}>{log.type}</span>
                      <span className="break-all opacity-80">{log.message}</span>
                  </div>
              ))}
              <div ref={logEndRef} />
          </div>
          {/* Scanline effect */}
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-emerald-500/5 to-transparent opacity-10 pointer-events-none animate-scanline"></div>
      </div>

      <button
        onClick={onAnalyze}
        disabled={isAnalyzing}
        className="mt-4 w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold rounded-xl flex items-center justify-center gap-2 transition-all shadow-lg shadow-indigo-900/20 text-sm"
      >
        {isAnalyzing ? (
          <>
            <Activity className="animate-spin" size={16} />
            <span className="animate-pulse">Processing...</span>
          </>
        ) : (
          <>
            <Activity size={16} />
            Analyze State
          </>
        )}
      </button>
    </div>
  );
};
