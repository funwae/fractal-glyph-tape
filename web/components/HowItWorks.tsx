import { Database, Sparkles, Map } from 'lucide-react'

const steps = [
  {
    icon: Database,
    title: 'Cluster',
    description: 'We embed and cluster phrases into phrase families, keeping examples, statistics, and language labels.',
  },
  {
    icon: Sparkles,
    title: 'Glyph & Fractal',
    description: 'Each family gets a glyph code and a coordinate on a fractal tape—a recursive triangular map of phrase space.',
  },
  {
    icon: Map,
    title: 'Integrate',
    description: 'A hybrid tokenizer and LLM adapter let existing models consume glyph-coded text and learn to expand glyphs into natural language.',
  },
]

export default function HowItWorks() {
  return (
    <section className="py-24 px-6 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <div className="mx-auto max-w-2xl text-center mb-16">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl mb-4">
            How It Works
          </h2>
          <p className="text-lg text-gray-400">
            Three steps to a navigable phrase memory
          </p>
        </div>

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-3 mb-12">
          {steps.map((step, i) => (
            <div key={step.title} className="relative">
              {i < steps.length - 1 && (
                <div className="hidden lg:block absolute top-12 left-full w-full h-0.5 bg-gradient-to-r from-primary/50 to-transparent -ml-4" />
              )}
              <div className="rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/80 to-slate-800/50 p-8 backdrop-blur h-full">
                <div className="rounded-lg bg-primary/10 w-12 h-12 flex items-center justify-center mb-6">
                  <step.icon className="h-6 w-6 text-primary" />
                </div>
                <div className="text-sm text-primary font-semibold mb-2">Step {i + 1}</div>
                <h3 className="text-xl font-bold mb-3">{step.title}</h3>
                <p className="text-sm text-gray-300">{step.description}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Pipeline diagram */}
        <div className="rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/30 p-8 backdrop-blur">
          <h3 className="text-lg font-semibold mb-6 text-center">Full Pipeline</h3>
          <div className="flex flex-wrap items-center justify-center gap-3 text-sm">
            {['Corpus', 'Embeddings', 'Clusters', 'Glyph IDs', 'Fractal Tape', 'Tokenizer', 'LLM'].map((step, i, arr) => (
              <div key={step} className="flex items-center gap-3">
                <div className="rounded-lg border border-gray-700 bg-slate-900/50 px-4 py-2 font-mono">
                  {step}
                </div>
                {i < arr.length - 1 && (
                  <div className="text-primary">→</div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Quickstart */}
        <div className="mt-12 rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/80 to-slate-800/50 p-8 backdrop-blur">
          <h3 className="text-xl font-bold mb-4">Quickstart</h3>
          <pre className="bg-slate-950/50 rounded-lg p-4 overflow-x-auto text-sm">
            <code className="text-gray-300">
              {`# 1) Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Build a demo tape
python scripts/run_full_build.py --config configs/demo.yaml

# 3) Try the CLI
echo "Can you send me that file?" | fgt encode
echo "谷阜" | fgt decode

# 4) Launch the visualizer
uvicorn fgt.viz.app:app --reload`}
            </code>
          </pre>
        </div>
      </div>
    </section>
  )
}
