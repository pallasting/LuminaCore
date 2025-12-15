import React, { useMemo } from 'react';
import { SystemState } from '../types';
import { motion } from 'framer-motion';

interface SystemVisualizerProps {
  state: SystemState;
}

export const SystemVisualizer: React.FC<SystemVisualizerProps> = ({ state }) => {
  // Calculate mixed color for the main waveguide bus
  const waveguideColor = useMemo(() => {
    // Basic additive mixing logic
    const r = Math.min(255, Math.floor((state.v1 / 100) * 255));
    const g = Math.min(255, Math.floor((state.v2 / 100) * 255));
    const b = Math.min(255, Math.floor((state.v3 / 100) * 255));
    // Ensure visibility even at low values for the diagram
    const minVis = 40;
    return `rgb(${Math.max(minVis, r)}, ${Math.max(minVis, g)}, ${Math.max(minVis, b)})`;
  }, [state]);

  const beamOpacity = (val: number) => Math.max(0.1, val / 100);

  // Packet speed based on voltage (higher voltage = more packets/activity visual)
  const packetSpeed = (val: number) => Math.max(0.5, 2 - (val / 100) * 1.5); 

  // COORD CONSTANTS
  const LAYER_1_X = 50;
  const LAYER_2_X = 180;
  const BUS_START_X = 300;
  const BUS_END_X = 420;
  const AWG_X = 450;
  const LAYER_4_X = 600; // Cores start
  const CORE_WIDTH = 120;
  const LAYER_4_END_X = LAYER_4_X + CORE_WIDTH;
  const LAYER_5_X = 850; // Detectors

  const Y_TOP = 100;
  const Y_MID = 250;
  const Y_BOT = 400;

  // Signal Quality Color
  const getSignalColor = (val: number) => {
      if (val > 60) return '#4ade80'; // Green
      if (val > 20) return '#facc15'; // Yellow
      return '#ef4444'; // Red
  };

  return (
    <div className="relative w-full h-full bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden flex items-center justify-center p-4 shadow-2xl shadow-black/80">
      
      {/* Schematic Background Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(30,41,59,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(30,41,59,0.1)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-to-r from-slate-950 via-transparent to-slate-950 pointer-events-none" />

      <svg 
        viewBox="0 0 1000 500" 
        className="w-full h-full max-w-6xl z-10"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
          <filter id="glow-green" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
           <filter id="glow-blue" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
           <linearGradient id="busGradient" x1="0%" y1="0%" x2="100%" y2="0%">
             <stop offset="0%" stopColor={waveguideColor} stopOpacity="0.4" />
             <stop offset="50%" stopColor={waveguideColor} stopOpacity="1" />
             <stop offset="100%" stopColor={waveguideColor} stopOpacity="1" />
           </linearGradient>
           <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
             <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1e293b" strokeWidth="0.5"/>
           </pattern>
        </defs>

        {/* --- Layer 1: Electronic Control (Left) --- */}
        <g transform={`translate(${LAYER_1_X}, 0)`}>
           <rect x="0" y="50" width="60" height="400" rx="4" fill="#0f172a" stroke="#334155" strokeWidth="2" />
           <text x="30" y="35" textAnchor="middle" fill="#64748b" className="text-[10px] font-mono uppercase tracking-widest">System 1: Elec</text>
           
           {/* Signal Traces to Layer 2 */}
           <path d={`M 60 ${Y_TOP} L ${LAYER_2_X - 10} ${Y_TOP}`} stroke="#ef4444" strokeWidth="1" strokeDasharray="2 2" opacity="0.5" />
           <path d={`M 60 ${Y_MID} L ${LAYER_2_X - 10} ${Y_MID}`} stroke="#22c55e" strokeWidth="1" strokeDasharray="2 2" opacity="0.5" />
           <path d={`M 60 ${Y_BOT} L ${LAYER_2_X - 10} ${Y_BOT}`} stroke="#3b82f6" strokeWidth="1" strokeDasharray="2 2" opacity="0.5" />
        </g>

        {/* --- Layer 2: Nanocrystal Emitter Array (Sources) --- */}
        <g transform={`translate(${LAYER_2_X}, 0)`}>
           <rect x="-10" y="40" width="80" height="420" rx="8" fill="#1e293b" stroke="#475569" strokeWidth="1" />
           <text x="30" y="25" textAnchor="middle" fill="#94a3b8" className="text-[10px] font-mono uppercase tracking-widest">Source Array</text>
           
           {/* Pixel R */}
           <g transform={`translate(30, ${Y_TOP})`}>
              <rect x="-15" y="-15" width="30" height="30" fill="#450a0a" stroke="#ef4444" strokeWidth="2" rx="4"/>
              <motion.rect x="-10" y="-10" width="20" height="20" fill="#ef4444" rx="2" filter="url(#glow-red)" animate={{ opacity: beamOpacity(state.v1) }} />
              <text x="30" y="5" fill="#ef4444" className="text-[9px] font-mono">Eu³⁺</text>
           </g>
           
           {/* Pixel G */}
           <g transform={`translate(30, ${Y_MID})`}>
              <rect x="-15" y="-15" width="30" height="30" fill="#052e16" stroke="#22c55e" strokeWidth="2" rx="4"/>
               <motion.rect x="-10" y="-10" width="20" height="20" fill="#22c55e" rx="2" filter="url(#glow-green)" animate={{ opacity: beamOpacity(state.v2) }} />
              <text x="30" y="5" fill="#22c55e" className="text-[9px] font-mono">Tb³⁺</text>
           </g>
           
           {/* Pixel B */}
           <g transform={`translate(30, ${Y_BOT})`}>
              <rect x="-15" y="-15" width="30" height="30" fill="#172554" stroke="#3b82f6" strokeWidth="2" rx="4"/>
              <motion.rect x="-10" y="-10" width="20" height="20" fill="#3b82f6" rx="2" filter="url(#glow-blue)" animate={{ opacity: beamOpacity(state.v3) }} />
              <text x="30" y="5" fill="#3b82f6" className="text-[9px] font-mono">Tm³⁺</text>
           </g>
        </g>

        {/* --- Layer 3: Optical Router (Multiplexer -> Waveguide -> AWG) --- */}
        
        {/* Multiplexing Lines - Animated Packets */}
        <motion.path 
            d={`M ${LAYER_2_X + 50} ${Y_TOP} C ${BUS_START_X - 50} ${Y_TOP}, ${BUS_START_X - 50} ${Y_MID}, ${BUS_START_X} ${Y_MID}`} 
            stroke="#ef4444" strokeWidth="3" fill="none" 
            strokeDasharray="10 20"
            animate={{ strokeDashoffset: -300 }}
            transition={{ duration: packetSpeed(state.v1), repeat: Infinity, ease: "linear" }}
            opacity={beamOpacity(state.v1)} 
        />
        <motion.path 
            d={`M ${LAYER_2_X + 50} ${Y_MID} L ${BUS_START_X} ${Y_MID}`} 
            stroke="#22c55e" strokeWidth="3" fill="none" 
            strokeDasharray="10 20"
            animate={{ strokeDashoffset: -300 }}
            transition={{ duration: packetSpeed(state.v2), repeat: Infinity, ease: "linear" }}
            opacity={beamOpacity(state.v2)} 
        />
        <motion.path 
            d={`M ${LAYER_2_X + 50} ${Y_BOT} C ${BUS_START_X - 50} ${Y_BOT}, ${BUS_START_X - 50} ${Y_MID}, ${BUS_START_X} ${Y_MID}`} 
            stroke="#3b82f6" strokeWidth="3" fill="none" 
            strokeDasharray="10 20"
            animate={{ strokeDashoffset: -300 }}
            transition={{ duration: packetSpeed(state.v3), repeat: Infinity, ease: "linear" }}
            opacity={beamOpacity(state.v3)} 
        />

        {/* Main WDM Bus */}
        <g transform={`translate(${BUS_START_X}, ${Y_MID})`}>
           <path d={`M 0 0 L ${BUS_END_X - BUS_START_X} 0`} stroke="url(#busGradient)" strokeWidth="8" strokeOpacity="0.2" />
           {/* Light packets animation - Main Bus */}
           <motion.rect 
             width="30" height="4" fill="white" y="-2" rx="2"
             animate={{ x: [0, BUS_END_X - BUS_START_X], opacity: [0, 1, 0] }}
             transition={{ duration: 0.4, repeat: Infinity, ease: "linear" }}
           />
           <text x={(BUS_END_X - BUS_START_X)/2} y="-15" textAnchor="middle" fill="#e2e8f0" className="text-[10px] font-bold">WDM BUS</text>
        </g>

        {/* The AWG (Arrayed Waveguide Grating) Router */}
        <g transform={`translate(${AWG_X}, ${Y_MID - 50})`}>
           <polygon points="0,20 40,0 40,100 0,80" fill="#334155" stroke="#94a3b8" strokeWidth="1" opacity="0.8"/>
           <text x="20" y="115" textAnchor="middle" fill="#94a3b8" className="text-[10px] font-mono">AWG Router</text>
           
           {/* Internal Gratings */}
           <path d="M 10 30 L 30 20" stroke="#475569" />
           <path d="M 10 40 L 30 30" stroke="#475569" />
           <path d="M 10 50 L 30 40" stroke="#475569" />
           <path d="M 10 60 L 30 50" stroke="#475569" />
           <path d="M 10 70 L 30 60" stroke="#475569" />
        </g>

        {/* --- Layer 4: Parallel Compute Cores (Demuxed) --- */}
        {/* Fan Out from AWG - Animated Packets */}
        <motion.path 
            d={`M ${AWG_X + 40} ${Y_MID - 20} C ${AWG_X + 80} ${Y_MID - 20}, ${LAYER_4_X - 50} ${Y_TOP}, ${LAYER_4_X} ${Y_TOP}`} 
            stroke="#ef4444" strokeWidth="3" fill="none"
            strokeDasharray="5 15"
            animate={{ strokeDashoffset: -200 }}
            transition={{ duration: packetSpeed(state.v1), repeat: Infinity, ease: "linear" }} 
            opacity={beamOpacity(state.v1)} 
        />
        <motion.path 
            d={`M ${AWG_X + 40} ${Y_MID} L ${LAYER_4_X} ${Y_MID}`} 
            stroke="#22c55e" strokeWidth="3" fill="none" 
            strokeDasharray="5 15"
            animate={{ strokeDashoffset: -200 }}
            transition={{ duration: packetSpeed(state.v2), repeat: Infinity, ease: "linear" }}
            opacity={beamOpacity(state.v2)} 
        />
        <motion.path 
            d={`M ${AWG_X + 40} ${Y_MID + 20} C ${AWG_X + 80} ${Y_MID + 20}, ${LAYER_4_X - 50} ${Y_BOT}, ${LAYER_4_X} ${Y_BOT}`} 
            stroke="#3b82f6" strokeWidth="3" fill="none" 
            strokeDasharray="5 15"
            animate={{ strokeDashoffset: -200 }}
            transition={{ duration: packetSpeed(state.v3), repeat: Infinity, ease: "linear" }}
            opacity={beamOpacity(state.v3)} 
        />

        <g transform={`translate(${LAYER_4_X}, 0)`}>
           {/* Top Core (Matrix) */}
           <g transform={`translate(0, ${Y_TOP - 40})`}>
             <rect width={CORE_WIDTH} height="80" rx="8" fill="#1e293b" stroke="#ef4444" strokeWidth="1" fillOpacity="0.4" />
             <text x="60" y="30" textAnchor="middle" fill="#ef4444" className="text-xs font-bold">Matrix Core</text>
             <text x="60" y="45" textAnchor="middle" fill="#94a3b8" className="text-[8px] font-mono">MZI Mesh</text>
             <g opacity={beamOpacity(state.v1)}>
               <circle cx="30" cy="55" r="5" stroke="#ef4444" fill="none" />
               <circle cx="60" cy="55" r="5" stroke="#ef4444" fill="none" />
               <circle cx="90" cy="55" r="5" stroke="#ef4444" fill="none" />
               <path d="M 35 55 L 55 55 M 65 55 L 85 55" stroke="#ef4444" strokeWidth="1" />
             </g>
           </g>

           {/* Middle Core (Conv) */}
           <g transform={`translate(0, ${Y_MID - 40})`}>
             <rect width={CORE_WIDTH} height="80" rx="8" fill="#1e293b" stroke="#22c55e" strokeWidth="1" fillOpacity="0.4" />
             <text x="60" y="30" textAnchor="middle" fill="#22c55e" className="text-xs font-bold">Conv Core</text>
             <text x="60" y="45" textAnchor="middle" fill="#94a3b8" className="text-[8px] font-mono">Ring Resonator</text>
             <g opacity={beamOpacity(state.v2)}>
               <circle cx="60" cy="55" r="15" stroke="#22c55e" fill="none" opacity="0.6" />
               <circle cx="60" cy="55" r="10" stroke="#22c55e" fill="none" opacity="0.8" />
             </g>
           </g>

           {/* Bottom Core (Activ) */}
           <g transform={`translate(0, ${Y_BOT - 40})`}>
             <rect width={CORE_WIDTH} height="80" rx="8" fill="#1e293b" stroke="#3b82f6" strokeWidth="1" fillOpacity="0.4" />
             <text x="60" y="30" textAnchor="middle" fill="#3b82f6" className="text-xs font-bold">Activ Core</text>
             <text x="60" y="45" textAnchor="middle" fill="#94a3b8" className="text-[8px] font-mono">SOA Non-linear</text>
             <path d="M 30 65 L 60 65 L 90 55" stroke="#3b82f6" strokeWidth="2" fill="none" opacity={beamOpacity(state.v3)} />
           </g>
        </g>

        {/* --- Layer 5: Readout (Detectors) --- */}
        {/* Connections from Layer 4 End to Layer 5 Start */}
        <g>
          <path d={`M ${LAYER_4_END_X} ${Y_TOP} L ${LAYER_5_X} ${Y_TOP}`} stroke="#ef4444" strokeDasharray="4 2" strokeWidth="1" opacity={beamOpacity(state.v1)} />
          <path d={`M ${LAYER_4_END_X} ${Y_MID} L ${LAYER_5_X} ${Y_MID}`} stroke="#22c55e" strokeDasharray="4 2" strokeWidth="1" opacity={beamOpacity(state.v2)} />
          <path d={`M ${LAYER_4_END_X} ${Y_BOT} L ${LAYER_5_X} ${Y_BOT}`} stroke="#3b82f6" strokeDasharray="4 2" strokeWidth="1" opacity={beamOpacity(state.v3)} />
        </g>

        <g transform={`translate(${LAYER_5_X}, 0)`}>
           <text x="30" y="25" textAnchor="middle" fill="#64748b" className="text-[10px] font-mono uppercase tracking-widest">Readout & SNR</text>

           {/* Detector R */}
           <g transform={`translate(0, ${Y_TOP})`}>
              <polygon points="0,-10 20,0 0,10" fill="#ef4444" opacity="0.8" />
              {/* SNR Indicator */}
              <rect x="25" y="-6" width="40" height="12" rx="2" fill="#1e293b" stroke="#334155" />
              <rect x="27" y="-4" width={state.v1 * 0.36} height="8" rx="1" fill={getSignalColor(state.v1)} opacity="0.8" />
              <text x="75" y="4" fill="#94a3b8" className="text-[9px] font-mono">{(state.v1 * 0.98).toFixed(0)}</text>
           </g>

           {/* Detector G */}
           <g transform={`translate(0, ${Y_MID})`}>
              <polygon points="0,-10 20,0 0,10" fill="#22c55e" opacity="0.8" />
              <rect x="25" y="-6" width="40" height="12" rx="2" fill="#1e293b" stroke="#334155" />
              <rect x="27" y="-4" width={state.v2 * 0.36} height="8" rx="1" fill={getSignalColor(state.v2)} opacity="0.8" />
              <text x="75" y="4" fill="#94a3b8" className="text-[9px] font-mono">{(state.v2 * 0.92).toFixed(0)}</text>
           </g>

           {/* Detector B */}
           <g transform={`translate(0, ${Y_BOT})`}>
              <polygon points="0,-10 20,0 0,10" fill="#3b82f6" opacity="0.8" />
              <rect x="25" y="-6" width="40" height="12" rx="2" fill="#1e293b" stroke="#334155" />
              <rect x="27" y="-4" width={state.v3 * 0.36} height="8" rx="1" fill={getSignalColor(state.v3)} opacity="0.8" />
              <text x="75" y="4" fill="#94a3b8" className="text-[9px] font-mono">{(state.v3 * 0.89).toFixed(0)}</text>
           </g>
        </g>

      </svg>
      
      {/* Overlay Status */}
      <div className="absolute top-4 left-4 text-[10px] font-mono text-slate-500 bg-slate-900/80 p-2 rounded border border-slate-800 backdrop-blur-sm">
        <div className="flex gap-2"><span className="text-slate-600">LINK:</span> <span className="text-emerald-500">ESTABLISHED</span></div>
        <div className="flex gap-2"><span className="text-slate-600">BER:</span> <span className="text-slate-300">1e-12</span></div>
        <div className="flex gap-2"><span className="text-slate-600">TEMP:</span> <span className="text-slate-300">24.5°C</span></div>
      </div>
    </div>
  );
};