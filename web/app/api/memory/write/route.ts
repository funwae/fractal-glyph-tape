import { NextRequest, NextResponse } from "next/server";

const MEMORY_API_URL = process.env.MEMORY_API_URL || "http://localhost:8001";

/**
 * Proxy endpoint for memory write operations.
 * Forwards requests to the Python FastAPI backend.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${MEMORY_API_URL}/api/memory/write`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Memory write error:", error);
    return NextResponse.json(
      {
        status: "error",
        error: "Failed to write to memory",
        world: "",
        region: "",
        addresses: [],
        glyph_density: 0,
      },
      { status: 500 }
    );
  }
}
