"use client";

import { useState } from "react";
import MemoryChatPanel from "@/components/MemoryChatPanel";
import MemoryContextPanel from "@/components/MemoryContextPanel";
import MemoryTimeline from "@/components/MemoryTimeline";
import GlyphClusterList from "@/components/GlyphClusterList";
import AddressInspector from "@/components/AddressInspector";
import { MemoryReadResponse } from "@/types/memory";

export default function MemoryConsolePage() {
  const [actorId, setActorId] = useState("demo-user");
  const [tokenBudget, setTokenBudget] = useState(2048);
  const [mode, setMode] = useState<"glyph" | "text" | "mixed">("mixed");
  const [currentContext, setCurrentContext] = useState<MemoryReadResponse | null>(null);
  const [selectedAddress, setSelectedAddress] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"context" | "timeline" | "glyphs" | "address">(
    "context"
  );

  const handleNewMemoryContext = (context: MemoryReadResponse) => {
    setCurrentContext(context);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="border-b border-white/10 bg-black/20 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">Memory Console</h1>
              <p className="text-sm text-gray-400 mt-1">
                Fractal Glyph Memory Service - Agent Memory OS
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Actor ID:</label>
                <input
                  type="text"
                  value={actorId}
                  onChange={(e) => setActorId(e.target.value)}
                  className="px-3 py-1 rounded bg-white/10 border border-white/20 text-white text-sm focus:outline-none focus:border-purple-500"
                  placeholder="Enter actor ID"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="container mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[calc(100vh-12rem)]">
          {/* Left Column - Chat & Controls */}
          <div className="flex flex-col gap-4">
            {/* Controls */}
            <div className="bg-white/5 backdrop-blur-sm rounded-lg border border-white/10 p-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-2">
                    Token Budget: {tokenBudget}
                  </label>
                  <input
                    type="range"
                    min="512"
                    max="8192"
                    step="512"
                    value={tokenBudget}
                    onChange={(e) => setTokenBudget(Number(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-2">Mode</label>
                  <select
                    value={mode}
                    onChange={(e) => setMode(e.target.value as typeof mode)}
                    className="w-full px-3 py-2 rounded bg-white/10 border border-white/20 text-white text-sm focus:outline-none focus:border-purple-500"
                  >
                    <option value="glyph">Glyph</option>
                    <option value="text">Text</option>
                    <option value="mixed">Mixed</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Chat Panel */}
            <div className="flex-1 bg-white/5 backdrop-blur-sm rounded-lg border border-white/10 overflow-hidden">
              <MemoryChatPanel
                actorId={actorId}
                tokenBudget={tokenBudget}
                mode={mode}
                onNewMemoryContext={handleNewMemoryContext}
              />
            </div>
          </div>

          {/* Right Column - Memory Visualization */}
          <div className="flex flex-col gap-4">
            {/* Tab Navigation */}
            <div className="bg-white/5 backdrop-blur-sm rounded-lg border border-white/10">
              <div className="flex border-b border-white/10">
                <button
                  onClick={() => setActiveTab("context")}
                  className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                    activeTab === "context"
                      ? "bg-purple-500/20 text-purple-300 border-b-2 border-purple-500"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  Context
                </button>
                <button
                  onClick={() => setActiveTab("timeline")}
                  className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                    activeTab === "timeline"
                      ? "bg-purple-500/20 text-purple-300 border-b-2 border-purple-500"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  Timeline
                </button>
                <button
                  onClick={() => setActiveTab("glyphs")}
                  className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                    activeTab === "glyphs"
                      ? "bg-purple-500/20 text-purple-300 border-b-2 border-purple-500"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  Glyphs
                </button>
                <button
                  onClick={() => setActiveTab("address")}
                  className={`flex-1 px-4 py-3 text-sm font-medium transition-colors ${
                    activeTab === "address"
                      ? "bg-purple-500/20 text-purple-300 border-b-2 border-purple-500"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  Address
                </button>
              </div>

              {/* Tab Content */}
              <div className="p-4 h-[calc(100vh-20rem)] overflow-y-auto">
                {activeTab === "context" && (
                  <MemoryContextPanel
                    context={currentContext}
                    onAddressClick={setSelectedAddress}
                  />
                )}
                {activeTab === "timeline" && (
                  <MemoryTimeline actorId={actorId} />
                )}
                {activeTab === "glyphs" && (
                  <GlyphClusterList actorId={actorId} />
                )}
                {activeTab === "address" && (
                  <AddressInspector
                    actorId={actorId}
                    address={selectedAddress}
                  />
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
