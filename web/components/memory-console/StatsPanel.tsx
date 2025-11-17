'use client'

interface StatsPanelProps {
  stats: any
}

export default function StatsPanel({ stats }: StatsPanelProps) {
  if (!stats) {
    return (
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <h3 className="text-lg font-semibold mb-2">Storage Stats</h3>
        <p className="text-sm text-slate-400">Loading...</p>
      </div>
    )
  }

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <h3 className="text-lg font-semibold mb-3">Storage Stats</h3>

      <div className="grid grid-cols-2 gap-2">
        {stats.total_entries !== undefined && (
          <div className="bg-slate-900 rounded p-2">
            <p className="text-xs text-slate-400">Total Entries</p>
            <p className="text-xl font-bold text-green-400">{stats.total_entries}</p>
          </div>
        )}

        {stats.unique_worlds !== undefined && (
          <div className="bg-slate-900 rounded p-2">
            <p className="text-xs text-slate-400">Worlds</p>
            <p className="text-xl font-bold text-blue-400">{stats.unique_worlds}</p>
          </div>
        )}

        {stats.unique_regions !== undefined && (
          <div className="bg-slate-900 rounded p-2">
            <p className="text-xs text-slate-400">Regions</p>
            <p className="text-xl font-bold text-purple-400">{stats.unique_regions}</p>
          </div>
        )}

        {stats.total_tokens !== undefined && (
          <div className="bg-slate-900 rounded p-2">
            <p className="text-xs text-slate-400">Total Tokens</p>
            <p className="text-xl font-bold text-orange-400">
              {stats.total_tokens.toLocaleString()}
            </p>
          </div>
        )}

        {stats.unique_actors !== undefined && (
          <div className="bg-slate-900 rounded p-2 col-span-2">
            <p className="text-xs text-slate-400">Active Actors</p>
            <p className="text-xl font-bold text-cyan-400">{stats.unique_actors}</p>
          </div>
        )}
      </div>
    </div>
  )
}
