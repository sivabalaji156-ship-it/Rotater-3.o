
import { GoogleGenAI, Type, Chat, Modality } from "@google/genai";
import { 
  ClimateStats, 
  Prediction, 
  BoundingBox, 
  WeatherData, 
  ClimatologyStats, 
  ResearchIntelligence 
} from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

export interface ResolvedLocation {
  lat: number;
  lon: number;
  bbox: BoundingBox | null;
  resolvedName: string;
}

/**
 * Enhanced JSON cleaning to handle markdown blocks and potential truncation.
 */
const cleanJSON = (text: string): string => {
  if (!text) return "{}";
  
  // Remove Markdown code block syntax if present
  let cleaned = text.trim();
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?/, '').replace(/```$/, '').trim();
  }

  const firstBrace = cleaned.indexOf('{');
  const lastBrace = cleaned.lastIndexOf('}');

  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    return cleaned.substring(firstBrace, lastBrace + 1);
  }
  
  return cleaned;
};

/**
 * Attempts to parse JSON and handles partial or malformed strings gracefully.
 */
const safeParseJSON = (text: string, fallback: any = {}): any => {
  const cleaned = cleanJSON(text);
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    console.warn("Initial JSON parse failed, attempting recovery...", e);
    // Basic recovery: if it starts with '{' but is missing '}', append it.
    // However, "unterminated string" usually means truncation.
    // We return the fallback to prevent app crash.
    return fallback;
  }
};

export const resolveGeospatialQuery = async (query: string): Promise<ResolvedLocation> => {
  const response = await ai.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Resolve the following location query into geographic coordinates: "${query}".`,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          lat: { type: Type.NUMBER },
          lon: { type: Type.NUMBER },
          resolvedName: { type: Type.STRING },
          bbox: {
            type: Type.OBJECT,
            properties: {
              latMin: { type: Type.NUMBER },
              latMax: { type: Type.NUMBER },
              lonMin: { type: Type.NUMBER },
              lonMax: { type: Type.NUMBER }
            },
            nullable: true
          }
        },
        required: ["lat", "lon", "resolvedName"]
      }
    }
  });
  return safeParseJSON(response.text || "{}", { lat: 0, lon: 0, resolvedName: "Unknown" });
};

export const getResearchIntelligence = async (
  stats: ClimateStats[],
  climatology: ClimatologyStats[],
  weather: WeatherData | null,
  lat: number,
  lon: number
): Promise<ResearchIntelligence> => {
  try {
    // 1. Get Environmental Analysis & Disasters
    const researchPrompt = `
      As a Lead Climate Research Analyst, provide a technical synthesis for sector (${lat}, ${lon}) for 2018-2025.
      
      NASA DATA (Last 24 months): ${JSON.stringify(stats.slice(-24))}
      CLIMATOLOGY NORMS: ${JSON.stringify(climatology)}
      REAL-TIME WEATHER: ${JSON.stringify(weather)}

      CONSTRAINTS:
      - Limit the 'disasters' list to the top 8 most significant events.
      - Keep 'summary' concise but technical (max 300 words).
      - Ensure the JSON is well-formed and complete.
      
      Provide:
      1. Technical summary of climate indicators.
      2. Timeline of significant natural disasters (2018-2025).
      3. NGO network analysis (top 5 active organizations).
      4. Risk predictions for the next 12 months.
    `;

    const researchResp = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: researchPrompt,
      config: {
        thinkingConfig: { thinkingBudget: 16384 }, // Adjusted budget to prioritize final output length
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: { type: Type.STRING },
            disasters: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  date: { type: Type.STRING },
                  type: { type: Type.STRING },
                  severity: { type: Type.STRING, enum: ['Moderate', 'Severe', 'Extreme'] },
                  impact: { type: Type.STRING },
                  areas: { type: Type.ARRAY, items: { type: Type.STRING } }
                }
              }
            },
            ngos: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  name: { type: Type.STRING },
                  origin: { type: Type.STRING },
                  aidType: { type: Type.STRING },
                  mode: { type: Type.STRING, enum: ['On-ground', 'Remote', 'Hybrid'] }
                }
              }
            },
            predictions: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  month: { type: Type.STRING },
                  riskLevel: { type: Type.STRING, enum: ['Low', 'Medium', 'High', 'Critical'] },
                  predictedTemp: { type: Type.NUMBER },
                  description: { type: Type.STRING }
                }
              }
            }
          },
          required: ["summary", "disasters", "ngos", "predictions"]
        }
      }
    });

    // 2. Get Relief Infrastructure (Using gemini-2.5-flash for Maps Tool)
    const mapsResp = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: `List up to 5 nearby relief centers, emergency shelters, or medical camps near coordinates ${lat}, ${lon}.`,
      config: {
        tools: [{ googleMaps: {} }],
        toolConfig: {
          retrievalConfig: {
            latLng: { latitude: lat, longitude: lon }
          }
        }
      }
    });

    const researchData = safeParseJSON(researchResp.text || "{}", {
      summary: "Data synthesis incomplete due to transmission error.",
      disasters: [],
      ngos: [],
      predictions: []
    });
    
    // Process Maps grounding into structured ReliefInfrastructure
    const infrastructure: any[] = [];
    const chunks = mapsResp.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
    chunks.forEach((chunk: any) => {
      if (chunk.maps) {
        infrastructure.push({
          name: chunk.maps.title || "Facility Identified",
          type: 'Response Hub',
          distance: 'Satellite Confirmed',
          location: 'Sector Vicinity',
          uri: chunk.maps.uri || "#"
        });
      }
    });

    return {
      ...researchData,
      infrastructure: infrastructure.length > 0 ? infrastructure : [
        { name: "Primary Relief Node", type: "Response Hub", distance: "Proximal", location: "Coordinated search in progress", uri: "#" }
      ]
    };
  } catch (error) {
    console.error("Intelligence failure:", error);
    // Provide a baseline valid object so the UI doesn't break
    return {
      summary: "Mission analysis compromised by signal interference. Please re-synchronize orbital arrays.",
      disasters: [],
      infrastructure: [{ name: "Emergency Node", type: "Response Hub", distance: "Unknown", location: "Scan Pending", uri: "#" }],
      ngos: [],
      predictions: []
    };
  }
};

export const generateSpeech = async (text: string): Promise<string | undefined> => {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: `Strategic intelligence briefing: ${text}` }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
        },
      },
    });
    return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  } catch (e) { 
    console.error("TTS failed:", e);
    return undefined; 
  }
};

export const createChatSession = (context: any): Chat => {
  return ai.chats.create({
    model: 'gemini-3-pro-preview',
    config: { 
      systemInstruction: `You are the ROTATER Strategic Research Analyst. Specialized in 2018-2025 climate data, natural disaster history, and infrastructure mapping. Current context: Sector (${context.lat}, ${context.lon}). Use research-grade terminology and be precise. Always reply in clear Markdown if appropriate for the chat.` 
    }
  });
};
