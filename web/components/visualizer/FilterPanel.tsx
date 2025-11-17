"use client";

import { Filter } from "lucide-react";

interface Filters {
  language: string;
  minFrequency: number;
}

interface FilterPanelProps {
  filters: Filters;
  onFiltersChange: (filters: Filters) => void;
  totalClusters: number;
  filteredClusters: number;
}

export default function FilterPanel({
  filters,
  onFiltersChange,
  totalClusters,
  filteredClusters,
}: FilterPanelProps) {
  const handleLanguageChange = (language: string) => {
    onFiltersChange({ ...filters, language });
  };

  const handleFrequencyChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFiltersChange({ ...filters, minFrequency: parseInt(e.target.value) || 0 });
  };

  return (
    <div className="w-64 bg-gray-800 border-r border-gray-700 p-4 flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 mb-6">
        <Filter className="w-5 h-5 text-gray-400" />
        <h2 className="text-lg font-semibold text-white">Filters</h2>
      </div>

      {/* Stats */}
      <div className="mb-6 p-3 bg-gray-900/50 rounded border border-gray-700">
        <div className="text-2xl font-bold text-blue-400">
          {filteredClusters.toLocaleString()}
        </div>
        <div className="text-xs text-gray-400 mt-1">
          of {totalClusters.toLocaleString()} clusters
        </div>
      </div>

      {/* Language Filter */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Language
        </label>
        <select
          value={filters.language}
          onChange={(e) => handleLanguageChange(e.target.value)}
          className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="all">All Languages</option>
          <option value="en">English</option>
          <option value="zh">Chinese</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="ja">Japanese</option>
        </select>
      </div>

      {/* Frequency Filter */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Minimum Frequency: {filters.minFrequency}
        </label>
        <input
          type="range"
          min="0"
          max="1000"
          step="10"
          value={filters.minFrequency}
          onChange={handleFrequencyChange}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0</span>
          <span>1000+</span>
        </div>
      </div>

      {/* Reset Button */}
      <button
        onClick={() => onFiltersChange({ language: "all", minFrequency: 0 })}
        className="w-full py-2 px-4 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded transition-colors"
      >
        Reset Filters
      </button>

      {/* Info */}
      <div className="mt-auto pt-6 border-t border-gray-700">
        <p className="text-xs text-gray-500">
          Filter clusters by language and frequency to explore different regions of the
          fractal tape.
        </p>
      </div>
    </div>
  );
}
