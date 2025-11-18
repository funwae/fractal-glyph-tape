import { Zap, ArrowUpRight, Languages, Brain } from 'lucide-react'

const benefits = [
  {
    icon: Brain,
    title: 'Intelligent Memory Retrieval',
    points: [
      'Foveated allocation strategy: 30% early context, 30% relevant, 40% recent',
      'Delivers the right memories at the right time for agent decision-making',
      '+46.7pp accuracy improvement over naive truncation under tight budgets',
    ],
  },
  {
    icon: Zap,
    title: 'Semantic Compression',
    points: [
      'Replace repeated patterns with short glyph codes',
      'Store one shared phrase-family table instead of millions of near-duplicates',
      '50-70% compression while preserving semantic content',
    ],
  },
  {
    icon: Languages,
    title: 'Cross-Lingual by Design',
    points: [
      'English, Spanish, Chinese, and other languages sharing the same intent cluster together',
      'Glyph IDs act as language-agnostic anchors for retrieval and analysis',
      '90-95% precision across language pairs',
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
            <div className="text-4xl font-bold text-primary mb-2">+46.7pp</div>
            <div className="text-gray-400">Accuracy Gain (256 tokens)</div>
          </div>
          <div className="rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur">
            <div className="text-4xl font-bold text-secondary mb-2">50-70%</div>
            <div className="text-gray-400">Compression Ratio</div>
          </div>
          <div className="rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur">
            <div className="text-4xl font-bold text-accent mb-2">90-95%</div>
            <div className="text-gray-400">Cross-Lingual Precision</div>
          </div>
        </div>
      </div>
    </section>
  )
}
