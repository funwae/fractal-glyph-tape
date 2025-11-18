import { Package, Brain, Globe, Search, Cpu } from 'lucide-react'

const features = [
  {
    icon: Cpu,
    title: 'Agent Memory Service',
    description: 'Production-ready REST API for intelligent memory retrieval. Foveated allocation delivers +46.7pp accuracy gain at a 256-token budget on synthetic multi-turn dialogs.',
  },
  {
    icon: Package,
    title: 'Semantic Compression',
    description: 'Smaller corpora and logs with reconstructable meaning. 50-70% compression on our test corpora while preserving semantic content.',
  },
  {
    icon: Brain,
    title: 'Effective Context Extension',
    description: 'More usable signal per token under fixed context windows. Fit 2.5-4x more semantic content in the same token budget on our internal benchmarks.',
  },
  {
    icon: Globe,
    title: 'Cross-Lingual Bridging',
    description: 'Shared glyph IDs for phrase families spanning multiple languages. 90-95% cross-lingual precision on EN↔ES↔ZH retrieval experiments.',
  },
]

export default function Features() {
  return (
    <section className="py-24 px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <div className="mx-auto max-w-2xl text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl mb-4">
            What&apos;s in this repo?
          </h2>
          <p className="text-lg text-gray-400">
            A complete research prototype for phrase-level semantic compression and cross-lingual LLMs
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="relative rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 hover:border-primary/50 transition-all backdrop-blur"
            >
              <feature.icon className="h-8 w-8 text-primary mb-4" />
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-sm text-gray-400">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Metrics note */}
        <div className="mt-12 text-center">
          <p className="text-sm text-gray-500 italic">
            All metrics are from internal experiments; see README and docs/PHASE-5-RESULTS.md for setup and limitations.
          </p>
        </div>

        {/* Implementation checklist */}
        <div className="mt-16 rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur">
          <h3 className="text-xl font-bold mb-6">Implementation Includes:</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            {[
              'Memory Service API – REST endpoints for agent memory read/write',
              'Foveated retrieval – 3-zone allocation (early/relevant/recent)',
              'Memory Console UI – interactive chat with context visualization',
              'Multilingual embeddings & clustering – phrase families with metadata',
              'Glyph encoding system – integer glyph IDs → Mandarin glyph strings',
              'Fractal tape builder – 2D projection + recursive triangular addressing',
              'Hybrid tokenizer – wraps base tokenizer with glyph-aware spans',
              'Benchmark suite – Phase 5 validation with +46.7pp accuracy gains',
            ].map((item, i) => (
              <div key={i} className="flex items-start gap-2">
                <span className="text-green-400 mt-1">✓</span>
                <span className="text-gray-300">{item}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
