import { NextRequest, NextResponse } from "next/server";

// Force dynamic rendering for API routes
export const dynamic = 'force-dynamic';

const MEMORY_API_URL = process.env.MEMORY_API_URL || "http://localhost:8001";

/**
 * Proxy endpoint for listing memory addresses.
 * Forwards requests to the Python FastAPI backend.
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const actorId = searchParams.get("actor_id");
    const region = searchParams.get("region");
    const limit = searchParams.get("limit") || "100";

    if (!actorId || !region) {
      return NextResponse.json(
        { status: "error", error: "actor_id and region are required" },
        { status: 400 }
      );
    }

    const response = await fetch(
      `${MEMORY_API_URL}/api/memory/addresses?actor_id=${encodeURIComponent(
        actorId
      )}&region=${encodeURIComponent(region)}&limit=${limit}`,
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
    console.error("Memory addresses error:", error);
    return NextResponse.json(
      {
        status: "error",
        error: "Failed to fetch addresses",
        actor_id: "",
        region: "",
        addresses: [],
      },
      { status: 500 }
    );
  }
}
