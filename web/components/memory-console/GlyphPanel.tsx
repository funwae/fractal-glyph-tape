'use client'

interface Glyph {
  glyph_id: number
  glyph_str: string
  cluster_id: number
  frequency: number
  semantic_summary?: string
}

interface GlyphPanelProps {
  glyphs: Glyph[]
}

export default function GlyphPanel({ glyphs }: GlyphPanelProps) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <h3 className="text-lg font-semibold mb-3">Active Glyphs</h3>

      {glyphs.length === 0 ? (
        <p className="text-sm text-slate-400">No glyphs in current context</p>
      ) : (
        <div className="space-y-2">
          {glyphs.slice(0, 10).map((glyph, index) => (
            <div
              key={index}
              className="bg-slate-900 rounded p-2 border border-slate-700 hover:border-purple-500/50 transition-colors"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-2xl font-bold text-purple-400">
                  {glyph.glyph_str}
                </span>
                <span className="text-xs text-slate-400">
                  #{glyph.glyph_id}
                </span>
              </div>
              {glyph.semantic_summary && (
                <p className="text-xs text-slate-300 line-clamp-2">
                  {glyph.semantic_summary}
                </p>
              )}
              <div className="flex items-center justify-between mt-1">
                <span className="text-xs text-slate-400">
                  Cluster: {glyph.cluster_id}
                </span>
                <span className="text-xs text-slate-400">
                  Freq: {glyph.frequency}
                </span>
              </div>
            </div>
          ))}
          {glyphs.length > 10 && (
            <p className="text-xs text-slate-400 text-center italic">
              +{glyphs.length - 10} more glyphs
            </p>
          )}
        </div>
      )}
    </div>
  )
}
