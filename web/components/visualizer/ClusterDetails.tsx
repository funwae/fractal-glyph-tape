"use client";

import { X } from "lucide-react";

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

interface ClusterDetailsProps {
  cluster: ClusterInfo;
  onClose: () => void;
}

export default function ClusterDetails({ cluster, onClose }: ClusterDetailsProps) {
  return (
    <div className="w-96 bg-gray-800 border-l border-gray-700 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-700 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="text-4xl">{cluster.glyph}</div>
            <div>
              <h2 className="text-lg font-semibold text-white">
                Cluster {cluster.cluster_id}
              </h2>
              <p className="text-sm text-gray-400">
                {cluster.language.toUpperCase()} • {cluster.size.toLocaleString()} phrases
              </p>
            </div>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-700 rounded transition-colors"
          aria-label="Close details"
        >
          <X className="w-5 h-5 text-gray-400" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Representative Phrase */}
        <div>
          <h3 className="text-sm font-semibold text-gray-300 mb-2">
            Representative Phrase
          </h3>
          <p className="text-white bg-gray-900/50 rounded p-3 border border-gray-700">
            {cluster.representative_phrase}
          </p>
        </div>

        {/* Metrics */}
        <div>
          <h3 className="text-sm font-semibold text-gray-300 mb-2">Metrics</h3>
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-gray-900/50 rounded p-3 border border-gray-700">
              <div className="text-2xl font-bold text-blue-400">
                {cluster.size.toLocaleString()}
              </div>
              <div className="text-xs text-gray-400 mt-1">Phrases</div>
            </div>
            <div className="bg-gray-900/50 rounded p-3 border border-gray-700">
              <div className="text-2xl font-bold text-green-400">
                {cluster.frequency.toLocaleString()}
              </div>
              <div className="text-xs text-gray-400 mt-1">Frequency</div>
            </div>
            <div className="bg-gray-900/50 rounded p-3 border border-gray-700 col-span-2">
              <div className="text-2xl font-bold text-purple-400">
                {cluster.coherence_score.toFixed(3)}
              </div>
              <div className="text-xs text-gray-400 mt-1">Coherence Score</div>
            </div>
          </div>
        </div>

        {/* Example Phrases */}
        {cluster.example_phrases && cluster.example_phrases.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">
              Example Phrases
            </h3>
            <div className="space-y-2">
              {cluster.example_phrases.slice(0, 10).map((phrase, idx) => (
                <div
                  key={idx}
                  className="text-sm text-gray-300 bg-gray-900/50 rounded p-2 border border-gray-700"
                >
                  {phrase}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Embedding Info */}
        {cluster.embedding_centroid && cluster.embedding_centroid.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-gray-300 mb-2">
              Embedding
            </h3>
            <div className="text-xs text-gray-400 bg-gray-900/50 rounded p-3 border border-gray-700">
              {cluster.embedding_centroid.length}-dimensional centroid vector
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
