"use client";

import { Globe, ArrowRight } from "lucide-react";

interface Example {
  glyph: string;
  cluster_id: string;
  semantic_meaning: string;
  examples: Array<{
    language: string;
    lang_code: string;
    phrase: string;
  }>;
}

const MULTILINGUAL_EXAMPLES: Example[] = [
  {
    glyph: "谷",
    cluster_id: "cluster_42",
    semantic_meaning: "File Sharing Request",
    examples: [
      { language: "English", lang_code: "en", phrase: "Can you send me that file?" },
      { language: "Chinese", lang_code: "zh", phrase: "你能发给我那个文件吗？" },
      { language: "Spanish", lang_code: "es", phrase: "¿Puedes enviarme ese archivo?" },
      { language: "French", lang_code: "fr", phrase: "Peux-tu m'envoyer ce fichier?" },
      { language: "German", lang_code: "de", phrase: "Kannst du mir diese Datei schicken?" },
    ],
  },
  {
    glyph: "阜",
    cluster_id: "cluster_127",
    semantic_meaning: "Gratitude Expression",
    examples: [
      { language: "English", lang_code: "en", phrase: "Thank you very much" },
      { language: "Chinese", lang_code: "zh", phrase: "非常感谢" },
      { language: "Spanish", lang_code: "es", phrase: "Muchas gracias" },
      { language: "French", lang_code: "fr", phrase: "Merci beaucoup" },
      { language: "Japanese", lang_code: "ja", phrase: "どうもありがとうございます" },
    ],
  },
  {
    glyph: "霞",
    cluster_id: "cluster_893",
    semantic_meaning: "Help Request",
    examples: [
      { language: "English", lang_code: "en", phrase: "Can you help me?" },
      { language: "Chinese", lang_code: "zh", phrase: "你能帮帮我吗？" },
      { language: "Spanish", lang_code: "es", phrase: "¿Puedes ayudarme?" },
      { language: "French", lang_code: "fr", phrase: "Peux-tu m'aider?" },
      { language: "Portuguese", lang_code: "pt", phrase: "Você pode me ajudar?" },
    ],
  },
];

export default function MultilingualShowcase() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto">
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-purple-500/10 border border-purple-500/30 rounded-full text-purple-400 text-sm font-medium mb-4">
          <Globe className="w-4 h-4" />
          <span>Cross-Lingual Phrase Families</span>
        </div>
        <h2 className="text-3xl font-bold mb-4">
          One Glyph, Many Languages
        </h2>
        <p className="text-gray-400 text-lg">
          FGT clusters semantically similar phrases across languages.
          Each glyph represents a phrase family that transcends linguistic boundaries.
        </p>
      </div>

      {/* Examples */}
      <div className="space-y-6">
        {MULTILINGUAL_EXAMPLES.map((example, idx) => (
          <div
            key={idx}
            className="bg-gradient-to-br from-gray-800 to-gray-850 border border-gray-700 rounded-xl p-6 hover:border-purple-500/50 transition-all"
          >
            {/* Header */}
            <div className="flex items-center gap-4 mb-6">
              <div className="text-6xl">{example.glyph}</div>
              <div className="flex-1">
                <h3 className="text-xl font-bold text-white mb-1">
                  {example.semantic_meaning}
                </h3>
                <p className="text-sm text-gray-400">
                  Cluster: {example.cluster_id} • {example.examples.length} languages
                </p>
              </div>
            </div>

            {/* Examples Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {example.examples.map((ex, exIdx) => (
                <div
                  key={exIdx}
                  className="p-4 bg-gray-900/50 rounded-lg border border-gray-700/50"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded font-medium">
                      {ex.lang_code.toUpperCase()}
                    </span>
                    <span className="text-sm text-gray-400">{ex.language}</span>
                  </div>
                  <p className="text-white">{ex.phrase}</p>
                </div>
              ))}
            </div>

            {/* Info */}
            <div className="mt-4 pt-4 border-t border-gray-700/50">
              <div className="flex items-center gap-2 text-sm text-gray-400">
                <ArrowRight className="w-4 h-4 text-purple-400" />
                <span>
                  All these phrases share the same glyph <span className="text-white font-mono">{example.glyph}</span>,
                  enabling cross-lingual search and compression
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Benefits */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
        <div className="p-6 bg-blue-500/10 border border-blue-500/30 rounded-lg">
          <h3 className="text-lg font-semibold text-blue-400 mb-2">
            Cross-Lingual Search
          </h3>
          <p className="text-sm text-gray-400">
            Query in one language, retrieve results in any language within the same phrase family
          </p>
        </div>
        <div className="p-6 bg-purple-500/10 border border-purple-500/30 rounded-lg">
          <h3 className="text-lg font-semibold text-purple-400 mb-2">
            Language-Agnostic Compression
          </h3>
          <p className="text-sm text-gray-400">
            Compress text efficiently regardless of source language using universal glyphs
          </p>
        </div>
        <div className="p-6 bg-green-500/10 border border-green-500/30 rounded-lg">
          <h3 className="text-lg font-semibold text-green-400 mb-2">
            Translation via Phrase Families
          </h3>
          <p className="text-sm text-gray-400">
            Translate by mapping through shared semantic spaces instead of direct text-to-text
          </p>
        </div>
      </div>
    </div>
  );
}
