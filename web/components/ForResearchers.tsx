import { Book, Github, FileText, MessageSquare } from 'lucide-react'

const topics = [
  'Tokenization and representation learning',
  'Semantic compression and storage',
  'Cross-lingual alignment',
  'Long-context LLMs',
]

const resources = [
  {
    icon: Github,
    title: 'Open-source Python implementation',
    description: 'Full codebase available on GitHub under MIT license',
    href: 'https://github.com/funwae/fractal-glyph-tape',
  },
  {
    icon: Book,
    title: '45+ docs with specs, math, and experiment protocols',
    description: 'Complete technical documentation, from vision to implementation',
    href: 'https://github.com/funwae/fractal-glyph-tape/tree/main/docs',
  },
  {
    icon: FileText,
    title: 'Ready-made scripts for experiments',
    description: 'Reproducible evaluation suite for compression, context, and retrieval',
    href: 'https://github.com/funwae/fractal-glyph-tape/blob/main/docs/EXPERIMENT_EXECUTION_PLAN.md',
  },
]

export default function ForResearchers() {
  return (
    <section className="py-24 px-6 lg:px-8 bg-gradient-to-b from-slate-900 to-slate-950">
      <div className="mx-auto max-w-7xl">
        <div className="mx-auto max-w-2xl text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl mb-4">
            For Researchers and Builders
          </h2>
          <p className="text-lg text-gray-400 mb-8">
            If you care about:
          </p>
          <ul className="text-left inline-block space-y-2 text-gray-300">
            {topics.map((topic) => (
              <li key={topic} className="flex items-center gap-2">
                <span className="text-primary">•</span>
                <span>{topic}</span>
              </li>
            ))}
          </ul>
          <p className="text-lg text-gray-400 mt-8">
            …then FGT is designed to be <span className="text-primary font-semibold">picked apart, extended, and argued with</span>.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3 mb-12">
          {resources.map((resource) => (
            <a
              key={resource.title}
              href={resource.href}
              target="_blank"
              rel="noopener noreferrer"
              className="group rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/80 to-slate-800/50 p-8 backdrop-blur hover:border-primary/50 transition-all"
            >
              <resource.icon className="h-8 w-8 text-primary mb-4 group-hover:scale-110 transition-transform" />
              <h3 className="text-lg font-semibold mb-2">{resource.title}</h3>
              <p className="text-sm text-gray-400">{resource.description}</p>
            </a>
          ))}
        </div>

        {/* CTAs */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <a
            href="https://github.com/funwae/fractal-glyph-tape"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full sm:w-auto rounded-lg bg-primary px-8 py-4 text-base font-semibold text-white shadow-lg hover:bg-primary/90 transition-all flex items-center justify-center gap-2"
          >
            <Github className="w-5 h-5" />
            Get the Code
          </a>
          <a
            href="https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full sm:w-auto rounded-lg border border-gray-600 px-8 py-4 text-base font-semibold text-white hover:bg-white/10 transition-all flex items-center justify-center gap-2"
          >
            <FileText className="w-5 h-5" />
            Read the Paper Draft
          </a>
          <a
            href="https://github.com/funwae/fractal-glyph-tape/discussions"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full sm:w-auto rounded-lg border border-gray-600 px-8 py-4 text-base font-semibold text-white hover:bg-white/10 transition-all flex items-center justify-center gap-2"
          >
            <MessageSquare className="w-5 h-5" />
            Join Discussions
          </a>
        </div>

        {/* Research status */}
        <div className="mt-16 rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur">
          <div className="text-center">
            <p className="text-lg text-gray-300 mb-2">
              FGT is <span className="text-accent font-semibold">research software</span>
            </p>
            <p className="text-gray-400">
              We invite feedback, experiments, and extensions. If you&apos;re working on tokenization, compression, or cross-lingual LLMs, this is for you.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
