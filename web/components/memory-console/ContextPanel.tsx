'use client'

interface MemoryContext {
  memories: any[]
  addresses: string[]
  glyphs: any[]
  token_estimate: number
  policy: string
  memories_selected: number
}

interface ContextPanelProps {
  memoryContext: MemoryContext | null
}

export default function ContextPanel({ memoryContext }: ContextPanelProps) {
  if (!memoryContext) {
    return (
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <h3 className="text-lg font-semibold mb-2">Memory Context</h3>
        <p className="text-sm text-slate-400">No memory context loaded yet</p>
      </div>
    )
  }

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <h3 className="text-lg font-semibold mb-3">Memory Context</h3>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        <div className="bg-slate-900 rounded p-2">
          <p className="text-xs text-slate-400">Memories</p>
          <p className="text-lg font-bold text-blue-400">{memoryContext.memories_selected}</p>
        </div>
        <div className="bg-slate-900 rounded p-2">
          <p className="text-xs text-slate-400">Tokens</p>
          <p className="text-lg font-bold text-purple-400">{memoryContext.token_estimate}</p>
        </div>
      </div>

      {/* Policy */}
      <div className="mb-4">
        <p className="text-xs text-slate-400 mb-1">Policy Mode</p>
        <span className="inline-block px-2 py-1 bg-blue-600/20 border border-blue-500/50 rounded text-sm">
          {memoryContext.policy}
        </span>
      </div>

      {/* Addresses */}
      <div>
        <p className="text-sm font-medium mb-2">Fractal Addresses</p>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {memoryContext.addresses.length === 0 ? (
            <p className="text-xs text-slate-400">No addresses</p>
          ) : (
            memoryContext.addresses.slice(0, 5).map((address, index) => (
              <div key={index} className="bg-slate-900 rounded px-2 py-1 text-xs font-mono">
                {address}
              </div>
            ))
          )}
          {memoryContext.addresses.length > 5 && (
            <p className="text-xs text-slate-400 italic">
              +{memoryContext.addresses.length - 5} more
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
