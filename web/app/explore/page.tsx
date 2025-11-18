"use client";

import { useState, useEffect } from "react";
import ClusterMap from "@/components/visualizer/ClusterMap";
import ClusterDetails from "@/components/visualizer/ClusterDetails";
import FilterPanel from "@/components/visualizer/FilterPanel";

interface LayoutPoint {
  cluster_id: string;
  x: number;
  y: number;
  glyph: string;
  language: string;
  frequency: number;
}

interface ClusterInfo {
  cluster_id: string;
  glyph: string;
  size: number;
  language: string;
  frequency: number;
  representative_phrase: string;
  example_phrases: string[];
  embedding_centroid: number[];
  coherence_score: number;
}

interface Filters {
  language: string;
  minFrequency: number;
}

export default function ExplorePage() {
  const [layout, setLayout] = useState<LayoutPoint[]>([]);
  const [filteredLayout, setFilteredLayout] = useState<LayoutPoint[]>([]);
  const [selectedCluster, setSelectedCluster] = useState<ClusterInfo | null>(null);
  const [hoveredCluster, setHoveredCluster] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>({
    language: "all",
    minFrequency: 0,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // API base URL - adjust based on environment
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Load layout data
  useEffect(() => {
    const fetchLayout = async () => {
      try {
        setLoading(true);
        const response = await fetch(`${API_BASE}/layout`);

        if (!response.ok) {
          throw new Error(`Failed to fetch layout: ${response.statusText}`);
        }

        const data = await response.json();
        setLayout(data);
        setFilteredLayout(data);
        setError(null);
      } catch (err) {
        console.error("Error fetching layout:", err);
        setError(err instanceof Error ? err.message : "Failed to load visualization data");
      } finally {
        setLoading(false);
      }
    };

    fetchLayout();
  }, [API_BASE]);

  // Apply filters
  useEffect(() => {
    let filtered = layout;

    if (filters.language !== "all") {
      filtered = filtered.filter((point) => point.language === filters.language);
    }

    if (filters.minFrequency > 0) {
      filtered = filtered.filter((point) => point.frequency >= filters.minFrequency);
    }

    setFilteredLayout(filtered);
  }, [layout, filters]);

  // Fetch cluster details
  const handleClusterClick = async (clusterId: string) => {
    try {
      const response = await fetch(`${API_BASE}/cluster/${clusterId}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch cluster details: ${response.statusText}`);
      }

      const data = await response.json();
      setSelectedCluster(data);
    } catch (err) {
      console.error("Error fetching cluster details:", err);
    }
  };

  const handleClusterHover = (clusterId: string | null) => {
    setHoveredCluster(clusterId);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading visualization...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="max-w-md p-8 bg-gray-800 rounded-lg border border-red-500/30">
          <h2 className="text-xl font-bold text-red-400 mb-4">Error Loading Visualization</h2>
          <p className="text-gray-300 mb-4">{error}</p>
          <p className="text-sm text-gray-400">
            Make sure the API server is running at {API_BASE}
          </p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Fractal Glyph Tape Explorer
              </h1>
              <p className="text-sm text-gray-400 mt-1">
                Explore {layout.length > 0 ? layout.length.toLocaleString() : 'phrase'} clusters
              </p>
            </div>
            <a
              href="/"
              className="text-gray-400 hover:text-white transition-colors"
            >
              ← Back to Home
            </a>
          </div>
          <div className="p-3 bg-amber-900/20 border border-amber-700/50 rounded-lg">
            <p className="text-sm text-amber-200">
              <strong>Note:</strong> Interactive map is currently available via local viz server; see README for run_viz_server.py.
            </p>
          </div>
        </div>
      </header>

      <div className="flex h-[calc(100vh-80px)]">
        {/* Filter Panel */}
        <FilterPanel
          filters={filters}
          onFiltersChange={setFilters}
          totalClusters={layout.length}
          filteredClusters={filteredLayout.length}
        />

        {/* Main Visualization */}
        <div className="flex-1 p-6">
          <ClusterMap
            layout={filteredLayout}
            selectedClusterId={selectedCluster?.cluster_id || null}
            hoveredClusterId={hoveredCluster}
            onClusterClick={handleClusterClick}
            onClusterHover={handleClusterHover}
          />
        </div>

        {/* Details Panel */}
        {selectedCluster && (
          <ClusterDetails
            cluster={selectedCluster}
            onClose={() => setSelectedCluster(null)}
          />
        )}
      </div>
    </div>
  );
}
