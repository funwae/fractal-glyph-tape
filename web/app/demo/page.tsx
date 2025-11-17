"use client";

import { useState } from "react";
import { Languages, Sparkles, ArrowRight } from "lucide-react";
import EncoderDemo from "@/components/demo/EncoderDemo";
import MultilingualShowcase from "@/components/demo/MultilingualShowcase";

export default function DemoPage() {
  const [activeTab, setActiveTab] = useState<"encoder" | "multilingual">("encoder");

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              FGT Interactive Demo
            </h1>
            <p className="text-sm text-gray-400 mt-1">
              Experience cross-lingual phrase compression in real-time
            </p>
          </div>
          <a
            href="/"
            className="text-gray-400 hover:text-white transition-colors"
          >
            ← Back to Home
          </a>
        </div>
      </header>

      {/* Tabs */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-4">
            <button
              onClick={() => setActiveTab("encoder")}
              className={`flex items-center gap-2 px-6 py-4 border-b-2 transition-colors ${
                activeTab === "encoder"
                  ? "border-blue-500 text-white"
                  : "border-transparent text-gray-400 hover:text-white"
              }`}
            >
              <Sparkles className="w-5 h-5" />
              <span className="font-medium">Encoder Demo</span>
            </button>
            <button
              onClick={() => setActiveTab("multilingual")}
              className={`flex items-center gap-2 px-6 py-4 border-b-2 transition-colors ${
                activeTab === "multilingual"
                  ? "border-purple-500 text-white"
                  : "border-transparent text-gray-400 hover:text-white"
              }`}
            >
              <Languages className="w-5 h-5" />
              <span className="font-medium">Multilingual Examples</span>
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        {activeTab === "encoder" ? <EncoderDemo /> : <MultilingualShowcase />}
      </div>

      {/* Info Banner */}
      <div className="fixed bottom-6 right-6 max-w-sm bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-lg p-4 text-white">
        <div className="flex items-start gap-3">
          <ArrowRight className="w-6 h-6 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold mb-1">Try the Live Demo</h3>
            <p className="text-sm opacity-90">
              Paste text in any language to see glyph compression in action
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
