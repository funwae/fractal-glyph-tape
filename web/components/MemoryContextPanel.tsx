"use client";

import { MemoryReadResponse } from "@/types/memory";
import { Database, Hash } from "lucide-react";

interface MemoryContextPanelProps {
  context: MemoryReadResponse | null;
  onAddressClick: (address: string) => void;
}

export default function MemoryContextPanel({
  context,
  onAddressClick,
}: MemoryContextPanelProps) {
  if (!context) {
    return (
      <div className="text-center text-gray-500 mt-8">
        <Database size={48} className="mx-auto mb-4 opacity-50" />
        <p>No memory context loaded yet</p>
        <p className="text-sm mt-2">Send a message to retrieve context</p>
      </div>
    );
  }

  if (context.status === "error") {
    return (
      <div className="text-center text-red-400 mt-8">
        <p>Error loading context: {context.error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="bg-white/5 rounded-lg p-4 border border-white/10">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">World:</span>
            <span className="ml-2 text-white">{context.world}</span>
          </div>
          <div>
            <span className="text-gray-400">Region:</span>
            <span className="ml-2 text-white">{context.region}</span>
          </div>
          <div>
            <span className="text-gray-400">Mode:</span>
            <span className="ml-2 text-white">{context.mode}</span>
          </div>
          <div>
            <span className="text-gray-400">Token Estimate:</span>
            <span className="ml-2 text-white">{context.token_estimate}</span>
          </div>
        </div>
      </div>

      {/* Context Items */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2">
          <Hash size={16} />
          Context Items ({context.context.length})
        </h3>

        {context.context.length === 0 && (
          <p className="text-gray-500 text-sm text-center py-4">
            No context items found
          </p>
        )}

        {context.context.map((item, idx) => (
          <div
            key={idx}
            className="bg-white/5 rounded-lg p-4 border border-white/10 hover:border-purple-500/50 transition-colors cursor-pointer"
            onClick={() => onAddressClick(item.address)}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-2">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onAddressClick(item.address);
                }}
                className="text-xs font-mono text-purple-400 hover:text-purple-300 transition-colors"
              >
                {item.address}
              </button>
              <div className="text-xs bg-purple-500/20 text-purple-300 px-2 py-1 rounded">
                Score: {item.score.toFixed(2)}
              </div>
            </div>

            {/* Glyphs */}
            {item.glyphs.length > 0 && (
              <div className="mb-2">
                <div className="text-2xl text-yellow-400 font-serif">
                  {item.glyphs.join(" ")}
                </div>
              </div>
            )}

            {/* Summary/Excerpt */}
            {item.summary && (
              <div className="text-sm text-gray-300 mb-2">
                <span className="text-gray-500">Summary:</span> {item.summary}
              </div>
            )}

            {item.excerpt && (
              <div className="text-xs text-gray-400 bg-white/5 rounded p-2 font-mono">
                {item.excerpt}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
