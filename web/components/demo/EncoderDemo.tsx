"use client";

import { useState } from "react";
import { Copy, Check, Loader2 } from "lucide-react";

interface EncodedResult {
  original: string;
  encoded: string;
  glyphs: Array<{
    glyph: string;
    cluster_id: string;
    original_phrase: string;
    language: string;
  }>;
  compression_ratio: number;
  original_length: number;
  encoded_length: number;
}

const EXAMPLE_TEXTS = [
  {
    title: "File Sharing Request",
    text: "Can you send me that file? I need it for the meeting tomorrow.",
    languages: ["en"],
  },
  {
    title: "Multilingual Greeting",
    text: "Hello! 你好！Bonjour! How are you doing today?",
    languages: ["en", "zh", "fr"],
  },
  {
    title: "Technical Support",
    text: "I'm having trouble with my computer. Can you help me fix this issue?",
    languages: ["en"],
  },
];

export default function EncoderDemo() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState<EncodedResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleEncode = async () => {
    if (!inputText.trim()) {
      setError("Please enter some text");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Note: In production, this would call the actual API
      // For demo purposes, we'll simulate the encoding
      await new Promise((resolve) => setTimeout(resolve, 1000));

      // Mock result
      const mockResult: EncodedResult = {
        original: inputText,
        encoded: "谷阜 " + inputText.substring(0, 20) + "... 霞",
        glyphs: [
          {
            glyph: "谷",
            cluster_id: "cluster_42",
            original_phrase: "Can you send me that file",
            language: "en",
          },
          {
            glyph: "阜",
            cluster_id: "cluster_127",
            original_phrase: "I need it for",
            language: "en",
          },
          {
            glyph: "霞",
            cluster_id: "cluster_893",
            original_phrase: "tomorrow",
            language: "en",
          },
        ],
        compression_ratio: 1.5,
        original_length: inputText.length,
        encoded_length: Math.floor(inputText.length / 1.5),
      };

      setResult(mockResult);
    } catch (err) {
      setError("Failed to encode text. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const loadExample = (text: string) => {
    setInputText(text);
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-8">
      {/* Examples */}
      <div>
        <h2 className="text-lg font-semibold mb-4 text-gray-200">
          Try These Examples
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {EXAMPLE_TEXTS.map((example, idx) => (
            <button
              key={idx}
              onClick={() => loadExample(example.text)}
              className="text-left p-4 bg-gray-800 hover:bg-gray-750 rounded-lg border border-gray-700 transition-colors"
            >
              <h3 className="font-medium text-white mb-2">{example.title}</h3>
              <p className="text-sm text-gray-400 line-clamp-2 mb-2">
                {example.text}
              </p>
              <div className="flex gap-2">
                {example.languages.map((lang) => (
                  <span
                    key={lang}
                    className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded"
                  >
                    {lang}
                  </span>
                ))}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Enter Text to Encode
        </label>
        <textarea
          value={inputText}
          onChange={(e) => {
            setInputText(e.target.value);
            setError(null);
          }}
          className="w-full h-32 px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          placeholder="Type or paste text in any language..."
        />
        {error && <p className="text-red-400 text-sm mt-2">{error}</p>}
      </div>

      {/* Encode Button */}
      <button
        onClick={handleEncode}
        disabled={loading || !inputText.trim()}
        className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-all flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <Loader2 className="w-5 h-5 animate-spin" />
            <span>Encoding...</span>
          </>
        ) : (
          <span>Encode with FGT</span>
        )}
      </button>

      {/* Results */}
      {result && (
        <div className="space-y-6 animate-fadeIn">
          {/* Encoded Output */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">
                Encoded Output
              </label>
              <button
                onClick={() => handleCopy(result.encoded)}
                className="text-sm text-gray-400 hover:text-white flex items-center gap-1 transition-colors"
              >
                {copied ? (
                  <>
                    <Check className="w-4 h-4" />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <div className="p-4 bg-gray-800 border border-gray-700 rounded-lg">
              <p className="text-white font-mono text-lg">{result.encoded}</p>
            </div>
          </div>

          {/* Compression Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-gradient-to-br from-blue-500/10 to-blue-600/10 rounded-lg border border-blue-500/30">
              <div className="text-3xl font-bold text-blue-400">
                {result.compression_ratio.toFixed(2)}x
              </div>
              <div className="text-sm text-gray-400 mt-1">Compression Ratio</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-green-500/10 to-green-600/10 rounded-lg border border-green-500/30">
              <div className="text-3xl font-bold text-green-400">
                {result.original_length}
              </div>
              <div className="text-sm text-gray-400 mt-1">Original Length</div>
            </div>
            <div className="p-4 bg-gradient-to-br from-purple-500/10 to-purple-600/10 rounded-lg border border-purple-500/30">
              <div className="text-3xl font-bold text-purple-400">
                {result.encoded_length}
              </div>
              <div className="text-sm text-gray-400 mt-1">Encoded Length</div>
            </div>
          </div>

          {/* Detected Glyphs */}
          <div>
            <h3 className="text-lg font-semibold mb-4 text-gray-200">
              Detected Phrase Families
            </h3>
            <div className="space-y-3">
              {result.glyphs.map((glyph, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-gray-800 border border-gray-700 rounded-lg"
                >
                  <div className="flex items-start gap-4">
                    <div className="text-4xl">{glyph.glyph}</div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-white font-medium">
                          {glyph.original_phrase}
                        </span>
                        <span className="text-xs px-2 py-1 bg-blue-500/20 text-blue-400 rounded">
                          {glyph.language}
                        </span>
                      </div>
                      <p className="text-sm text-gray-400">
                        Cluster: {glyph.cluster_id}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
