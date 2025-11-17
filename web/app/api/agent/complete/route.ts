import { NextRequest, NextResponse } from "next/server";

/**
 * Simple agent completion endpoint (placeholder).
 *
 * In a full implementation, this would:
 * 1. Take the chat messages and memory context
 * 2. Build a prompt for an LLM
 * 3. Call the LLM API
 * 4. Return the response
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { messages, memory } = body;

    // Get the last user message
    const lastUserMessage = messages
      .filter((m: any) => m.role === "user")
      .pop();

    // Simple placeholder response based on memory context
    let reply = "I understand. ";

    if (memory && memory.context && memory.context.length > 0) {
      const glyphCount = memory.context.reduce(
        (sum: number, item: any) => sum + item.glyphs.length,
        0
      );

      reply += `I found ${memory.context.length} relevant memory items with ${glyphCount} glyphs. `;

      // Include a snippet from the first context item
      const firstItem = memory.context[0];
      if (firstItem.summary) {
        reply += `The most relevant memory is: "${firstItem.summary}". `;
      }
    }

    reply += "How can I help you further?";

    return NextResponse.json({ reply });
  } catch (error) {
    console.error("Agent completion error:", error);
    return NextResponse.json(
      { error: "Failed to generate response" },
      { status: 500 }
    );
  }
}
