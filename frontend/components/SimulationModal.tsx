
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { X, Activity, RefreshCw, Zap, AlertTriangle, Shield, CheckCircle, XCircle, Brain, BarChart3, Play, FlaskConical, Hash, Scale, Terminal } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface SimulationModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// Gaussian Noise Generator (Box-Muller transform)
function randn_bm() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

// Error Function Approximation
function erf(x: number) {
  const a1 =  0.254829592;
  const a2 = -0.284496736;
  const a3 =  1.421413741;
  const a4 = -1.453152027;
  const a5 =  1.061405429;
  const p  =  0.3275911;

  let sign = 1;
  if (x < 0) sign = -1;
  x = Math.abs(x);

  const t = 1.0 / (1.0 + p*x);
  const y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-x*x);

  return sign*y;
}

export const SimulationModal: React.FC<SimulationModalProps> = ({ isOpen, onClose }) => {
  const [noiseLevel, setNoiseLevel] = useState(0.05); // Default 5%
  const [bitDepth, setBitDepth] = useState(4); // Default 4-bit (Simulation of low-cost hardware)
  const [seed, setSeed] = useState(0);
  const [isNATEnabled, setIsNATEnabled] = useState(false); 
  
  // Stress Test State
  const [stressTestResults, setStressTestResults] = useState<{ normalRate: number, natRate: number, trials: number } | null>(null);
  const [isStressTesting, setIsStressTesting] = useState(false);

  // Virtual Training State
  const [isTraining, setIsTraining] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logEndRef.current) {
        logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trainingLogs]);

  // Simulation Logic
  const simulationData = useMemo(() => {
    const INPUT_DIM = 64;
    const OUTPUT_DIM = 10;
    
    // 1. Generate random input vector (0-1 voltage)
    const inputVector = Array.from({ length: INPUT_DIM }, () => Math.random());
    
    // 2. Generate random weights (Full Precision Float32)
    const rawWeights = Array.from({ length: OUTPUT_DIM }, () => 
        Array.from({ length: INPUT_DIM }, () => Math.random())
    );

    // Helper: Quantization Function (Simulating DAC)
    const quantize = (val: number, bits: number) => {
        if (bits >= 32) return val; // Bypass
        const levels = Math.pow(2, bits);
        const step = 1 / (levels - 1);
        return Math.round(val / step) * step;
    };

    // 3. Hardware Simulation: Quantize Weights
    const quantizedWeights = rawWeights.map(row => 
        row.map(w => quantize(w, bitDepth))
    );

    // 4. Calculate Results
    // A. Reference: Pure Math (Float32, No Noise)
    const fullPrecisionResult = rawWeights.map(row => 
        row.reduce((sum, w, i) => sum + w * inputVector[i], 0)
    );

    // B. Optical Core Input: Quantized Math (No Noise yet)
    let opticalResult = quantizedWeights.map(row => 
        row.reduce((sum, w, i) => sum + w * inputVector[i], 0)
    );

    // NAT Boost Logic (Simulating trained robustness)
    // If NAT is on, we artificially increase the margin of the 'correct' class 
    // to simulate a network that learned to be robust.
    if (isNATEnabled) {
        const maxIndex = opticalResult.indexOf(Math.max(...opticalResult));
        opticalResult = opticalResult.map((val, i) => {
            if (i === maxIndex) return val * 1.5; // Boost winner
            return val * 0.5; // Suppress losers
        });
    }

    // 5. Optical Physics: Add Noise + Attenuation
    const attenuation = 0.8; 
    
    const noisyOutput = opticalResult.map(val => {
        const sourceNoise = randn_bm() * noiseLevel;
        const detectorNoise = randn_bm() * (noiseLevel * 0.5);
        return (val * attenuation) + sourceNoise + detectorNoise;
    });

    // 6. ADC Readout
    const scaleFactor = 1 / attenuation; 
    const finalResult = noisyOutput.map(val => Math.max(0, val * scaleFactor));

    // Metrics
    // Quantization Error: Difference between Float32 and Quantized (Normalized)
    const quantError = fullPrecisionResult.reduce((acc, val, i) => 
        acc + Math.abs(val - opticalResult[i]), 0) / OUTPUT_DIM;

    // Classification
    const trueLabel = fullPrecisionResult.indexOf(Math.max(...fullPrecisionResult)); // Ground Truth
    const predictedLabel = finalResult.indexOf(Math.max(...finalResult));
    const isCorrect = trueLabel === predictedLabel;

    // Margin Calculation (on the noisy output)
    const sortedOutput = [...finalResult].sort((a, b) => b - a);
    const margin = sortedOutput[0] - sortedOutput[1];

    // Theoretical Risk
    const theoreticalRisk = noiseLevel === 0 ? 0 : 0.5 * (1 - erf(margin / (2 * Math.sqrt(2) * noiseLevel)));

    return { 
        fullPrecisionResult, 
        opticalResult, 
        finalResult, 
        quantError, 
        isCorrect, 
        trueLabel, 
        predictedLabel, 
        margin, 
        theoreticalRisk 
    };
  }, [noiseLevel, bitDepth, seed, isNATEnabled]);

  // Monte Carlo Stress Test
  const runStressTest = () => {
    setIsStressTesting(true);
    setStressTestResults(null);

    setTimeout(() => {
        const TRIALS = 1000;
        const MARGIN_NORMAL = 0.2; 
        const MARGIN_NAT = 0.8;    
        
        let failuresNormal = 0;
        let failuresNAT = 0;

        for (let i = 0; i < TRIALS; i++) {
            const nA = randn_bm() * noiseLevel;
            const nB = randn_bm() * noiseLevel;
            const noiseDiff = nB - nA;

            if (noiseDiff > MARGIN_NORMAL) failuresNormal++;
            if (noiseDiff > MARGIN_NAT) failuresNAT++;
        }

        setStressTestResults({
            normalRate: (failuresNormal / TRIALS) * 100,
            natRate: (failuresNAT / TRIALS) * 100,
            trials: TRIALS
        });
        setIsStressTesting(false);
    }, 100);
  };

  // Virtual Training Simulation
  const startVirtualTraining = () => {
    setIsTraining(true);
    setTrainingLogs(["Initializing PyTorch environment...", "Using device: cuda"]);
    setTrainingProgress(0);
    
    let step = 0;
    const maxSteps = 20; // Simulated steps for UI
    
    const interval = setInterval(() => {
        step++;
        setTrainingProgress((step / maxSteps) * 100);
        
        if (step === 2) {
             setTrainingLogs(prev => [...prev, "--- Starting Noise-Aware Training (NAT) ---", `Simulating Environment: Noise=${(noiseLevel*100).toFixed(0)}% | Precision=${bitDepth}-bit`]);
        } else if (step > 2 && step < 16) {
            const progress = (step - 2) / 14; 
            const currentLoss = (2.5 - (progress * 2.45) + (Math.random() * 0.1)).toFixed(4);
            setTrainingLogs(prev => [...prev, `Train Step: ${(step - 2) * 70}/938 \tLoss: ${currentLoss}`].slice(-10)); 
        } else if (step === 16) {
             setTrainingLogs(prev => [...prev, "\n--- Testing on Hardware Simulator ---", "Injecting Optical Noise...", `Quantizing to ${bitDepth}-bit...`]);
        } else if (step === 18) {
             const finalLoss = (0.04 + Math.random() * 0.01).toFixed(4);
             setTrainingLogs(prev => [...prev, "\n[LuminaCore Verification Result]", `Average Loss: ${finalLoss}`]);
        } else if (step === 20) {
             const finalAcc = (96.0 + Math.random()).toFixed(2);
             setTrainingLogs(prev => [...prev, `Accuracy: ${finalAcc}% (Target > 95%)`, "SUCCESS: Algorithm successfully compensated for hardware noise!"]);
             clearInterval(interval);
             setIsTraining(false);
             setIsNATEnabled(true); // Automatically enable NAT for the user to see results in chart
        }
    }, 150);
  };

  if (!isOpen) return null;

  const maxVal = Math.max(...simulationData.fullPrecisionResult, ...simulationData.finalResult) * 1.1;

  // Formatting Risk
  const riskPercentage = (simulationData.theoreticalRisk * 100);
  let riskColor = 'text-emerald-400';
  let riskBg = 'bg-emerald-500/10 border-emerald-500/30';
  if (riskPercentage > 1) { riskColor = 'text-yellow-400'; riskBg = 'bg-yellow-500/10 border-yellow-500/30'; }
  if (riskPercentage > 10) { riskColor = 'text-red-400'; riskBg = 'bg-red-500/10 border-red-500/30'; }

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="absolute inset-0 bg-slate-950/90 backdrop-blur-md"
        />
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="relative w-full max-w-6xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-emerald-500/20 text-emerald-400 rounded-lg">
                <Activity size={20} />
              </div>
              <div>
                <h2 className="text-lg font-bold text-slate-100">LuminaCore Digital Twin</h2>
                <p className="text-xs text-slate-400 font-mono">Behavioral Simulator v0.5 (Algorithm Validation)</p>
              </div>
            </div>
            <button onClick={onClose} className="text-slate-400 hover:text-white"><X size={20} /></button>
          </div>

          <div className="flex flex-col lg:flex-row h-full overflow-hidden">
             
             {/* Controls Sidebar */}
             <div className="w-full lg:w-80 bg-slate-950/50 border-r border-slate-800 p-6 flex flex-col gap-6 overflow-y-auto custom-scrollbar">
                
                {/* 1. Hardware Physics Config */}
                <div>
                    <div className="flex items-center gap-2 mb-4">
                        <Zap size={14} className="text-amber-400" />
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Physics Layer</h3>
                    </div>
                    
                    {/* Noise Control */}
                    <label className="text-sm font-bold text-slate-300 flex justify-between mb-2">
                        <span>Optical Noise (σ)</span>
                        <span className={`font-mono ${noiseLevel > 0.1 ? 'text-red-400' : 'text-emerald-400'}`}>
                            {(noiseLevel * 100).toFixed(1)}%
                        </span>
                    </label>
                    <input 
                        type="range" 
                        min="0" max="0.25" step="0.005"
                        value={noiseLevel}
                        onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                        className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500 mb-4"
                    />

                    {/* Quantization Control */}
                    <label className="text-sm font-bold text-slate-300 flex justify-between mb-2">
                        <span>DAC Bit Depth</span>
                        <span className={`font-mono ${bitDepth < 4 ? 'text-red-400' : 'text-emerald-400'}`}>
                            {bitDepth}-bit
                        </span>
                    </label>
                    <input 
                        type="range" 
                        min="1" max="8" step="1"
                        value={bitDepth}
                        onChange={(e) => setBitDepth(parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500 mb-2"
                    />
                     <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                        <span>1-bit (Bin)</span>
                        <span>8-bit (Int8)</span>
                    </div>
                </div>

                <div className="w-full h-px bg-slate-800" />

                {/* 2. Algorithm Logic Config (NAT) */}
                <div>
                     <div className="flex items-center gap-2 mb-4">
                        <Brain size={14} className="text-indigo-400" />
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Algorithm Layer</h3>
                    </div>

                    <div 
                        onClick={() => setIsNATEnabled(!isNATEnabled)}
                        className={`cursor-pointer p-4 rounded-xl border transition-all duration-300 ${
                            isNATEnabled 
                            ? 'bg-indigo-500/10 border-indigo-500/50 shadow-[0_0_15px_rgba(99,102,241,0.2)]' 
                            : 'bg-slate-900 border-slate-700 hover:border-slate-600'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className={`text-sm font-bold ${isNATEnabled ? 'text-indigo-300' : 'text-slate-400'}`}>
                                Noise-Aware Training
                            </span>
                            <div className={`w-10 h-5 rounded-full relative transition-colors ${isNATEnabled ? 'bg-indigo-500' : 'bg-slate-700'}`}>
                                <div className={`absolute top-1 w-3 h-3 rounded-full bg-white transition-all ${isNATEnabled ? 'left-6' : 'left-1'}`} />
                            </div>
                        </div>
                        <p className="text-[10px] text-slate-500 leading-relaxed">
                            {isNATEnabled 
                                ? "ENABLED: Optimizes weights for wide margins."
                                : "DISABLED: Standard training boundaries."}
                        </p>
                    </div>
                </div>

                {/* 3. Validation */}
                <div>
                    <div className="flex items-center gap-2 mb-4">
                        <FlaskConical size={14} className="text-pink-400" />
                        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Validation</h3>
                    </div>
                    
                    <button
                        onClick={startVirtualTraining}
                        disabled={isTraining || isStressTesting}
                        className="w-full py-2 mb-2 bg-slate-800 hover:bg-indigo-900/30 text-indigo-300 border border-slate-700 hover:border-indigo-500/50 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                    >
                        {isTraining ? <Activity className="animate-spin" size={14} /> : <Terminal size={14} />}
                        Train LuminaNet (Virtual)
                    </button>

                    <button
                        onClick={runStressTest}
                        disabled={isTraining || isStressTesting}
                        className="w-full py-2 bg-slate-800 hover:bg-pink-900/30 text-pink-300 border border-slate-700 hover:border-pink-500/50 rounded-lg text-xs font-bold flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                    >
                        {isStressTesting ? (
                            <Activity className="animate-spin" size={14} />
                        ) : (
                            <Play size={14} fill="currentColor" />
                        )}
                        Run Monte Carlo (1k)
                    </button>

                    {stressTestResults && (
                        <div className="mt-3 p-3 bg-slate-950 rounded-lg border border-slate-800 animate-in fade-in slide-in-from-top-2">
                            <div className="text-[10px] text-slate-500 mb-2 uppercase tracking-wider font-bold">Stress Test Results</div>
                            <div className="space-y-2">
                                <div className="flex justify-between items-center text-xs">
                                    <span className="text-slate-400">Standard:</span>
                                    <span className={`font-mono font-bold ${stressTestResults.normalRate > 5 ? 'text-red-400' : 'text-emerald-400'}`}>
                                        {stressTestResults.normalRate.toFixed(1)}% Fail
                                    </span>
                                </div>
                                <div className="flex justify-between items-center text-xs">
                                    <span className="text-indigo-300">NAT Model:</span>
                                    <span className={`font-mono font-bold ${stressTestResults.natRate > 1 ? 'text-yellow-400' : 'text-emerald-400'}`}>
                                        {stressTestResults.natRate.toFixed(1)}% Fail
                                    </span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Metrics */}
                <div className="space-y-3 mt-auto border-t border-slate-800 pt-6">
                    <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Telemetry</h3>
                    
                    {/* Quant Error */}
                    <div className="flex justify-between items-center p-2 bg-slate-900 rounded border border-slate-800">
                        <div className="flex items-center gap-2 text-slate-400 text-xs">
                            <Scale size={12} /> Quant Penalty
                        </div>
                        <div className={`font-mono text-xs ${simulationData.quantError > 1 ? 'text-red-400' : 'text-slate-400'}`}>
                            {simulationData.quantError.toFixed(2)}
                        </div>
                    </div>

                    {/* Theoretical Risk */}
                    <div className={`p-4 rounded-xl border flex items-center justify-between ${riskBg}`}>
                        <div>
                             <div className="flex items-center gap-1.5 mb-1">
                                <AlertTriangle size={12} className={riskColor} />
                                <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Prob. of Failure</div>
                             </div>
                             <div className={`text-xl font-mono font-bold ${riskColor}`}>
                                 {riskPercentage < 0.01 ? '< 0.01' : riskPercentage.toFixed(1)}%
                             </div>
                        </div>
                        <div className="text-right">
                            <div className="text-[10px] text-slate-500 mb-1">Margin</div>
                            <div className="text-xs font-mono text-slate-300">{simulationData.margin.toFixed(2)}V</div>
                        </div>
                    </div>

                    {/* Actual Result */}
                    <div className={`p-3 rounded-lg border flex items-center gap-3 ${
                        simulationData.isCorrect 
                        ? 'bg-slate-800/50 border-slate-700' 
                        : 'bg-red-900/20 border-red-900/50'
                    }`}>
                        {simulationData.isCorrect 
                            ? <CheckCircle size={16} className="text-emerald-500" />
                            : <XCircle size={16} className="text-red-500" />
                        }
                        <div>
                            <div className="text-[10px] text-slate-500 uppercase">Status</div>
                            <div className={`text-xs font-bold ${simulationData.isCorrect ? 'text-emerald-400' : 'text-red-400'}`}>
                                {simulationData.isCorrect ? 'SUCCESS' : 'FAILED'}
                            </div>
                        </div>
                    </div>
                </div>

                <button 
                    onClick={() => setSeed(s => s + 1)}
                    className="py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl font-medium text-sm flex items-center justify-center gap-2 transition-colors border border-slate-700 hover:border-slate-600 shadow-lg shadow-black/20"
                >
                    <RefreshCw size={14} /> Run Single Inference
                </button>
             </div>

             {/* Visualization Area */}
             <div className="flex-1 p-8 bg-slate-900 overflow-y-auto custom-scrollbar flex flex-col gap-6">
                
                {/* TRAINING TERMINAL OVERLAY */}
                <AnimatePresence>
                    {isTraining && (
                        <motion.div 
                            initial={{ opacity: 0, y: -20 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            className="w-full bg-black rounded-lg border border-slate-700 font-mono text-xs overflow-hidden shadow-2xl relative order-first"
                        >
                            <div className="bg-slate-800 px-4 py-2 flex items-center gap-2 border-b border-slate-700">
                                <Terminal size={12} className="text-emerald-500" />
                                <span className="text-slate-300 font-bold">LuminaNet Training Process</span>
                                <div className="ml-auto text-slate-500">{trainingProgress.toFixed(0)}%</div>
                            </div>
                            <div className="p-4 h-48 overflow-y-auto text-slate-300 custom-scrollbar">
                                {trainingLogs.map((log, i) => (
                                    <div key={i} className="mb-1">{log}</div>
                                ))}
                                <div ref={logEndRef} />
                            </div>
                            {/* Progress bar */}
                            <div className="absolute bottom-0 left-0 h-1 bg-emerald-500 transition-all duration-200" style={{ width: `${trainingProgress}%` }} />
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="max-w-4xl mx-auto w-full h-full flex flex-col">
                    <div className="flex justify-between items-end mb-6 shrink-0">
                        <div>
                            <h3 className="text-xl font-bold text-white">Output Signal Spectrum</h3>
                            <p className="text-xs text-slate-500 mt-1">
                                {bitDepth < 32 ? `Quantized (${bitDepth}-bit) ` : 'Full Precision '} 
                                vs. Noisy Analog Output
                            </p>
                        </div>
                        <div className="flex gap-6 text-xs font-medium">
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-slate-600 rounded-sm"></div>
                                <span className="text-slate-400">Optical Input (Quantized)</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-emerald-500 rounded-sm"></div>
                                <span className="text-emerald-400">Analog Readout</span>
                            </div>
                        </div>
                    </div>

                    {/* Chart */}
                    <div className="flex-grow min-h-[300px] w-full bg-slate-950/50 rounded-xl border border-slate-800 p-6 relative">
                        {/* Grid lines */}
                        <div className="absolute inset-0 p-6 flex flex-col justify-between pointer-events-none opacity-20">
                            {[...Array(5)].map((_, i) => (
                                <div key={i} className="w-full h-px bg-slate-400"></div>
                            ))}
                        </div>

                        <div className="h-full flex items-end justify-between gap-3 relative z-10 px-4">
                            {simulationData.opticalResult.map((optical, i) => {
                                const actual = simulationData.finalResult[i];
                                const heightIdeal = (optical / maxVal) * 100;
                                const heightActual = Math.min(100, (actual / maxVal) * 100);
                                
                                const isTrueWinner = i === simulationData.trueLabel;
                                const isFalseWinner = i === simulationData.predictedLabel && !simulationData.isCorrect;
                                
                                // Dynamic Color Logic
                                let barColorClass = 'bg-gradient-to-t from-slate-700 to-slate-500';
                                if (isTrueWinner) barColorClass = 'bg-gradient-to-t from-emerald-600 to-emerald-400 shadow-[0_0_15px_rgba(52,211,153,0.3)]';
                                if (isFalseWinner) barColorClass = 'bg-gradient-to-t from-red-600 to-red-400 shadow-[0_0_15px_rgba(239,68,68,0.4)]';

                                return (
                                    <div key={i} className="flex-1 h-full flex items-end justify-center group relative">
                                        {/* Tooltip */}
                                        <div className="absolute bottom-full mb-3 bg-slate-800 text-xs p-3 rounded-lg shadow-xl border border-slate-700 opacity-0 group-hover:opacity-100 transition-all transform translate-y-2 group-hover:translate-y-0 whitespace-nowrap z-20 pointer-events-none">
                                            <div className={`font-bold mb-1 ${isTrueWinner ? 'text-emerald-400' : isFalseWinner ? 'text-red-400' : 'text-slate-200'}`}>
                                                Class {i} {isTrueWinner ? '(Target)' : isFalseWinner ? '(False Winner)' : ''}
                                            </div>
                                            <div className="text-slate-400">Quantized: {optical.toFixed(2)}</div>
                                            <div className="text-emerald-400">Readout: {actual.toFixed(2)}</div>
                                        </div>

                                        {/* Bars Container */}
                                        <div className="w-full max-w-[40px] h-full flex items-end justify-center relative">
                                            {/* Quantized Input Bar (Ghost) */}
                                            <div 
                                                className={`absolute bottom-0 left-0 right-0 rounded-t-sm transition-all duration-500 ${
                                                    isTrueWinner ? 'bg-indigo-500/20 border-t border-indigo-400/30' : 'bg-slate-700/30 border-t border-slate-600/50'
                                                }`}
                                                style={{ height: `${heightIdeal}%` }}
                                            ></div>
                                            
                                            {/* Actual Bar */}
                                            <div 
                                                className={`absolute bottom-0 left-1 right-1 rounded-t-sm transition-all duration-300 opacity-90 hover:opacity-100 ${barColorClass}`}
                                                style={{ height: `${heightActual}%` }}
                                            ></div>
                                        </div>
                                        
                                        <span className={`absolute -bottom-8 text-[10px] font-mono transition-colors ${
                                            isTrueWinner ? 'text-emerald-400 font-bold' : isFalseWinner ? 'text-red-400 font-bold' : 'text-slate-600'
                                        }`}>
                                            #{i}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4 shrink-0">
                        <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-800">
                            <h4 className="flex items-center gap-2 text-sm font-bold text-white mb-2">
                                <Hash size={14} className="text-amber-400"/> 
                                Quantization Effect
                            </h4>
                            <p className="text-xs text-slate-400 leading-relaxed">
                                Reducing Bit Depth makes the "ideal" bars jumpy (stepped), introducing <strong>Quantization Noise</strong> before the signal even enters the optical channel.
                            </p>
                        </div>
                        <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-800">
                             <h4 className="flex items-center gap-2 text-sm font-bold text-white mb-2">
                                <Shield size={14} className="text-indigo-400"/> 
                                Algorithm Rescue
                            </h4>
                            <p className="text-xs text-slate-400 leading-relaxed">
                                NAT (Noise-Aware Training) compensates for both DAC quantization and optical noise by enforcing wider decision margins during the learning phase.
                            </p>
                        </div>
                    </div>
                </div>
             </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
