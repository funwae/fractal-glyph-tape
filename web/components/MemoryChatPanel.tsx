"use client";

import { useState, useRef, useEffect } from "react";
import { Send } from "lucide-react";
import { ChatMessage, MemoryReadResponse } from "@/types/memory";

interface MemoryChatPanelProps {
  actorId: string;
  tokenBudget: number;
  mode: "glyph" | "text" | "mixed";
  onNewMemoryContext: (context: MemoryReadResponse) => void;
}

export default function MemoryChatPanel({
  actorId,
  tokenBudget,
  mode,
  onNewMemoryContext,
}: MemoryChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: input.trim(),
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      // 1. Write user message to memory
      await fetch("/api/memory/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          actor_id: actorId,
          text: userMessage.content,
          tags: ["chat"],
          source: "user",
        }),
      });

      // 2. Read memory context
      const readResponse = await fetch("/api/memory/read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          actor_id: actorId,
          query: userMessage.content,
          token_budget: tokenBudget,
          mode: mode,
        }),
      });

      const memoryContext: MemoryReadResponse = await readResponse.json();
      onNewMemoryContext(memoryContext);

      // 3. Call agent LLM backend
      const agentResponse = await fetch("/api/agent/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: messages.concat(userMessage),
          memory: memoryContext,
        }),
      });

      const agentData = await agentResponse.json();
      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: agentData.reply || "I received your message.",
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // 4. Write assistant reply to memory
      await fetch("/api/memory/write", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          actor_id: actorId,
          text: assistantMessage.content,
          tags: ["chat", "assistant"],
          source: "assistant",
        }),
      });
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage: ChatMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error processing your message.",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <p className="text-lg mb-2">👋 Welcome to Memory Console</p>
            <p className="text-sm">Start chatting to see the memory system in action</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                msg.role === "user"
                  ? "bg-purple-600 text-white"
                  : "bg-white/10 text-gray-100 border border-white/20"
              }`}
            >
              <div className="text-xs opacity-70 mb-1">
                {msg.role === "user" ? "You" : "Assistant"}
              </div>
              <div className="whitespace-pre-wrap">{msg.content}</div>
              <div className="text-xs opacity-50 mt-1">
                {new Date(msg.timestamp).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white/10 text-gray-100 border border-white/20 rounded-lg px-4 py-2">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-white/10 p-4">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            className="flex-1 px-4 py-2 rounded-lg bg-white/10 border border-white/20 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 resize-none"
            rows={2}
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-6 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors flex items-center gap-2"
          >
            <Send size={18} />
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
