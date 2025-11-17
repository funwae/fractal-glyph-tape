/**
 * TypeScript types for Fractal Glyph Memory Service API
 */

export interface MemoryWriteRequest {
  actor_id: string;
  text: string;
  tags?: string[];
  region?: string;
  source?: "user" | "assistant" | "system";
}

export interface MemoryWriteResponse {
  status: "ok" | "error";
  world: string;
  region: string;
  addresses: string[];
  glyph_density: number;
  error?: string;
}

export interface MemoryReadFocus {
  region?: string;
  max_depth?: number;
}

export interface MemoryReadRequest {
  actor_id: string;
  query: string;
  focus?: MemoryReadFocus;
  token_budget?: number;
  mode?: "glyph" | "text" | "mixed";
}

export interface MemoryContextItem {
  address: string;
  glyphs: string[];
  summary?: string;
  excerpt?: string;
  score: number;
}

export interface MemoryReadResponse {
  status: "ok" | "error";
  world: string;
  region: string;
  mode: string;
  context: MemoryContextItem[];
  token_estimate: number;
  error?: string;
}

export interface RegionInfo {
  region: string;
  record_count: number;
  span_count: number;
  first_timestamp?: string;
  last_timestamp?: string;
}

export interface RegionsListResponse {
  status: "ok" | "error";
  actor_id: string;
  regions: RegionInfo[];
  error?: string;
}

export interface AddressInfo {
  address: string;
  span_count: number;
  created_at: string;
}

export interface AddressesListResponse {
  status: "ok" | "error";
  actor_id: string;
  region: string;
  addresses: AddressInfo[];
  error?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}
