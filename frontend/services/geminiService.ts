import { GoogleGenAI } from "@google/genai";
import { SystemState } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export const analyzeSystemState = async (state: SystemState): Promise<string> => {
  try {
    const prompt = `
      Act as a senior optical physicist and computer architect specializing in Post-Moore Photonic Computing.
      Analyze the current configuration of the "LuminaCore" processor.
      
      Architecture Context:
      - Technology: Electroluminescent Rare Earth Nanocrystals (4nm) + Passive AWG Routing.
      - Mechanism: Wavelength Division Multiplexing (WDM) allows parallel processing in a single waveguide.
      
      Current State Configuration:
      - Channel λ1 (Europium/Red, Matrix MAC Unit): ${state.v1}% Intensity
      - Channel λ2 (Terbium/Green, Convolution Unit): ${state.v2}% Intensity
      - Channel λ3 (Thulium/Blue, Non-linear Activation): ${state.v3}% Intensity

      Provide a concise, scientific analysis (max 150 words) covering:
      1. **Dominant Compute Mode**: Which mathematical operation is prioritizing the optical budget?
      2. **Physical Implications**: Mention the excitation state of the specific ions (Eu³⁺, Tb³⁺, or Tm³⁺) and the WDM bus load.
      3. **AI Workload Mapping**: What kind of AI task does this mix represent (e.g., CNN inference, LLM token generation, Sparse activation)?

      Format the output as a clean, readable paragraph. Do not use Markdown formatting like bold or headers.
    `;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });

    return response.text || "Analysis unavailable.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Unable to connect to the optical analysis core. Please check your connection or API key.";
  }
};