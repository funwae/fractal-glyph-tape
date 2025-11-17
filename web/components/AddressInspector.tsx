"use client";

import { useState, useEffect } from "react";
import { MapPin, Copy } from "lucide-react";

interface AddressInspectorProps {
  actorId: string;
  address: string | null;
}

interface ParsedAddress {
  world: string;
  region: string;
  triPath: string;
  depth: number;
  timeSlice: number;
}

export default function AddressInspector({
  actorId,
  address,
}: AddressInspectorProps) {
  const [parsed, setParsed] = useState<ParsedAddress | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (address) {
      parseAddress(address);
    } else {
      setParsed(null);
    }
  }, [address]);

  const parseAddress = (addr: string) => {
    try {
      // Format: "world/region#tri_path@dDEPTHtTIME"
      const match = addr.match(
        /^([^/]+)\/([^#]+)#([^@]+)@d(\d+)t(\d+)$/
      );

      if (match) {
        setParsed({
          world: match[1],
          region: match[2],
          triPath: match[3],
          depth: parseInt(match[4]),
          timeSlice: parseInt(match[5]),
        });
      }
    } catch (error) {
      console.error("Error parsing address:", error);
      setParsed(null);
    }
  };

  const handleCopy = () => {
    if (address) {
      navigator.clipboard.writeText(address);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (!address) {
    return (
      <div className="text-center text-gray-500 mt-8">
        <MapPin size={48} className="mx-auto mb-4 opacity-50" />
        <p>No address selected</p>
        <p className="text-sm mt-2">Click on a context item to inspect its address</p>
      </div>
    );
  }

  if (!parsed) {
    return (
      <div className="text-center text-red-400 mt-8">
        <p>Invalid address format</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2">
          <MapPin size={16} />
          Address Inspector
        </h3>
        <button
          onClick={handleCopy}
          className="text-xs flex items-center gap-1 px-2 py-1 bg-white/10 hover:bg-white/20 rounded transition-colors text-gray-300"
        >
          <Copy size={12} />
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>

      {/* Full Address */}
      <div className="bg-white/5 rounded-lg p-4 border border-white/10">
        <div className="text-xs text-gray-400 mb-1">Full Address</div>
        <div className="font-mono text-sm text-purple-300 break-all">
          {address}
        </div>
      </div>

      {/* Parsed Components */}
      <div className="space-y-3">
        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-xs text-gray-400 mb-1">World</div>
          <div className="text-white">{parsed.world}</div>
        </div>

        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-xs text-gray-400 mb-1">Region</div>
          <div className="text-white">{parsed.region}</div>
        </div>

        <div className="bg-white/5 rounded-lg p-4 border border-white/10">
          <div className="text-xs text-gray-400 mb-1">Triangular Path</div>
          <div className="font-mono text-white">{parsed.triPath}</div>
          <div className="text-xs text-gray-500 mt-1">
            Fractal coordinates in glyph tape space
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="bg-white/5 rounded-lg p-4 border border-white/10">
            <div className="text-xs text-gray-400 mb-1">Depth</div>
            <div className="text-2xl text-white">{parsed.depth}</div>
            <div className="text-xs text-gray-500 mt-1">
              {parsed.depth === 0
                ? "Summary"
                : parsed.depth === 1
                ? "Key phrases"
                : "Full detail"}
            </div>
          </div>

          <div className="bg-white/5 rounded-lg p-4 border border-white/10">
            <div className="text-xs text-gray-400 mb-1">Time Slice</div>
            <div className="text-2xl text-white">{parsed.timeSlice}</div>
            <div className="text-xs text-gray-500 mt-1">Temporal index</div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="space-y-2">
        <div className="text-xs text-gray-400 mb-2">Actions</div>
        <button
          className="w-full px-4 py-2 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/50 text-purple-300 rounded-lg transition-colors text-sm"
          disabled
        >
          View in Fractal Map (Coming Soon)
        </button>
        <button
          className="w-full px-4 py-2 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/50 text-purple-300 rounded-lg transition-colors text-sm"
          disabled
        >
          Explore in 3D Viewer (Coming Soon)
        </button>
      </div>
    </div>
  );
}
