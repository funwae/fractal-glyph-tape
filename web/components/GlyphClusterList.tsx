"use client";

import { useEffect, useState } from "react";
import { Boxes } from "lucide-react";
import { AddressesListResponse, RegionsListResponse } from "@/types/memory";

interface GlyphClusterListProps {
  actorId: string;
}

interface GlyphInfo {
  glyph: string;
  address: string;
  spanCount: number;
}

export default function GlyphClusterList({ actorId }: GlyphClusterListProps) {
  const [glyphs, setGlyphs] = useState<GlyphInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRegion, setSelectedRegion] = useState<string>("");
  const [regions, setRegions] = useState<string[]>([]);

  useEffect(() => {
    loadRegions();
  }, [actorId]);

  useEffect(() => {
    if (selectedRegion) {
      loadGlyphs();
    }
  }, [selectedRegion, actorId]);

  const loadRegions = async () => {
    try {
      const response = await fetch(`/api/memory/regions?actor_id=${actorId}`);
      const data: RegionsListResponse = await response.json();

      if (data.status === "ok" && data.regions.length > 0) {
        const regionNames = data.regions.map((r) => r.region);
        setRegions(regionNames);
        setSelectedRegion(regionNames[0]);
      }
    } catch (error) {
      console.error("Error loading regions:", error);
    }
  };

  const loadGlyphs = async () => {
    try {
      setLoading(true);
      const response = await fetch(
        `/api/memory/addresses?actor_id=${actorId}&region=${selectedRegion}&limit=50`
      );
      const data: AddressesListResponse = await response.json();

      if (data.status === "ok") {
        // Group by glyph (simplified - in reality would need to fetch actual spans)
        const glyphMap = new Map<string, GlyphInfo>();

        data.addresses.forEach((addr) => {
          // Extract glyphs from address (placeholder logic)
          const glyph = "谷阜"; // Would extract from actual span data

          if (!glyphMap.has(glyph)) {
            glyphMap.set(glyph, {
              glyph,
              address: addr.address,
              spanCount: addr.span_count,
            });
          }
        });

        setGlyphs(Array.from(glyphMap.values()));
      }
    } catch (error) {
      console.error("Error loading glyphs:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center text-gray-500 mt-8">
        <Boxes size={48} className="mx-auto mb-4 opacity-50 animate-pulse" />
        <p>Loading glyphs...</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Region Selector */}
      {regions.length > 0 && (
        <div>
          <label className="block text-sm text-gray-400 mb-2">Region</label>
          <select
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="w-full px-3 py-2 rounded bg-white/10 border border-white/20 text-white text-sm focus:outline-none focus:border-purple-500"
          >
            {regions.map((region) => (
              <option key={region} value={region}>
                {region}
              </option>
            ))}
          </select>
        </div>
      )}

      <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2">
        <Boxes size={16} />
        Glyphs & Clusters ({glyphs.length})
      </h3>

      {glyphs.length === 0 ? (
        <div className="text-center text-gray-500 mt-8">
          <Boxes size={48} className="mx-auto mb-4 opacity-50" />
          <p>No glyphs found</p>
          <p className="text-sm mt-2">
            Glyphs will appear as you add memories
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          {glyphs.map((info, idx) => (
            <div
              key={idx}
              className="bg-white/5 rounded-lg p-4 border border-white/10 hover:border-purple-500/50 transition-colors cursor-pointer"
            >
              <div className="text-3xl text-yellow-400 font-serif mb-2 text-center">
                {info.glyph}
              </div>
              <div className="text-xs text-gray-400 text-center">
                {info.spanCount} span{info.spanCount !== 1 ? "s" : ""}
              </div>
              <div className="text-xs text-purple-400 mt-1 text-center truncate font-mono">
                {info.address.split("#")[1]?.split("@")[0] || "..."}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
