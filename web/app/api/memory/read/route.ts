import { NextRequest, NextResponse } from "next/server";

const MEMORY_API_URL = process.env.MEMORY_API_URL || "http://localhost:8001";

/**
 * Proxy endpoint for memory read operations.
 * Forwards requests to the Python FastAPI backend.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const response = await fetch(`${MEMORY_API_URL}/api/memory/read`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Memory read error:", error);
    return NextResponse.json(
      {
        status: "error",
        error: "Failed to read from memory",
        world: "",
        region: "",
        mode: "mixed",
        context: [],
        token_estimate: 0,
      },
      { status: 500 }
    );
  }
}
