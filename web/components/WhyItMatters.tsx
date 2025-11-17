import { Zap, ArrowUpRight, Languages } from 'lucide-react'

const benefits = [
  {
    icon: Zap,
    title: 'Semantic Compression',
    points: [
      'Replace repeated patterns with short glyph codes',
      'Store one shared phrase-family table instead of millions of near-duplicates',
      'Keep meaning; drop redundancy',
    ],
  },
  {
    icon: ArrowUpRight,
    title: 'Bigger-Feeling Context',
    points: [
      'A single glyph token can stand in for an entire motif the model already knows',
      'Under fixed token budgets, prompts carry more semantic content and longer histories',
      '2.5-4x more effective context per token budget',
    ],
  },
  {
    icon: Languages,
    title: 'Cross-Lingual by Design',
    points: [
      'English, Mandarin, and other languages sharing the same intent land in the same phrase family',
      'Glyph IDs act as language-agnostic anchors for retrieval and analysis',
      '13-7 percentage point gains in cross-lingual retrieval',
    ],
  },
]

export default function WhyItMatters() {
  return (
    <section className="py-24 px-6 lg:px-8 bg-gradient-to-b from-slate-950 to-slate-900">
      <div className="mx-auto max-w-7xl">
        <div className="mx-auto max-w-2xl text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl mb-4">
            Why It Matters
          </h2>
          <p className="text-lg text-gray-400">
            Three core capabilities that transform how LLMs handle language
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3">
          {benefits.map((benefit) => (
            <div
              key={benefit.title}
              className="rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/80 to-slate-800/50 p-8 backdrop-blur"
            >
              <div className="rounded-lg bg-primary/10 w-12 h-12 flex items-center justify-center mb-6">
                <benefit.icon className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-4">{benefit.title}</h3>
              <ul className="space-y-3">
                {benefit.points.map((point, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
                    <span className="text-secondary mt-1">•</span>
                    <span>{point}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Stats */}
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
          <div className="rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur">
            <div className="text-4xl font-bold text-primary mb-2">50-70%</div>
            <div className="text-gray-400">Compression Ratio</div>
          </div>
          <div className="rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur">
            <div className="text-4xl font-bold text-secondary mb-2">2.5-4x</div>
            <div className="text-gray-400">Context Extension</div>
          </div>
          <div className="rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur">
            <div className="text-4xl font-bold text-accent mb-2">+13pp</div>
            <div className="text-gray-400">Cross-Lingual Retrieval</div>
          </div>
        </div>
      </div>
    </section>
  )
}
