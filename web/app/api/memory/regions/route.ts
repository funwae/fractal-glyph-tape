import { NextRequest, NextResponse } from "next/server";

// Force dynamic rendering for API routes
export const dynamic = 'force-dynamic';

const MEMORY_API_URL = process.env.MEMORY_API_URL || "http://localhost:8001";

/**
 * Proxy endpoint for listing memory regions.
 * Forwards requests to the Python FastAPI backend.
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const actorId = searchParams.get("actor_id");

    if (!actorId) {
      return NextResponse.json(
        { status: "error", error: "actor_id is required" },
        { status: 400 }
      );
    }

    const response = await fetch(
      `${MEMORY_API_URL}/api/memory/regions?actor_id=${encodeURIComponent(actorId)}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Memory regions error:", error);
    return NextResponse.json(
      {
        status: "error",
        error: "Failed to fetch regions",
        actor_id: "",
        regions: [],
      },
      { status: 500 }
    );
  }
}
