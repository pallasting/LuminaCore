
export interface SystemState {
  v1: number; // Voltage for Red (Europium)
  v2: number; // Voltage for Green (Terbium)
  v3: number; // Voltage for Blue (Thulium/Neodymium)
}

export interface SimulationMetrics {
  totalPower: number;
  computeThroughput: number; // Theoretical FLOPS proxy
  temperature: number; // Simulated temp in Celsius
}

export enum ComputeType {
  MATRIX = 'Matrix Multiplication',
  CONVOLUTION = 'Convolution',
  ACTIVATION = 'Non-linear Activation'
}

export interface AnalysisResponse {
  summary: string;
  physicalImplications: string;
  computeContext: string;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  type: 'SYS' | 'ISA' | 'WARN' | 'CMD';
  message: string;
}
