import { Github, FileText, Map, Brain } from 'lucide-react'

export default function Hero() {
  return (
    <section className="relative overflow-hidden px-6 pt-20 pb-32 lg:px-8">
      {/* Background gradient effects */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute left-1/2 top-0 -translate-x-1/2 blur-3xl opacity-20">
          <div className="aspect-[1155/678] w-[72.1875rem] bg-gradient-to-tr from-primary to-secondary" />
        </div>
      </div>

      <div className="mx-auto max-w-4xl text-center">
        {/* Main headline */}
        <h1 className="text-5xl font-bold tracking-tight sm:text-7xl mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-gray-400">
          Fractal Glyph Tape
        </h1>

        <p className="text-xl sm:text-2xl text-gray-400 mb-4 font-medium">
          Agent Memory OS: Dense, fractal, cross-lingual phrase memory.
        </p>

        {/* Subheadline */}
        <p className="mt-6 text-lg leading-8 text-gray-300 max-w-3xl mx-auto">
          Intelligent memory retrieval for AI agents. Fractal Glyph Tape (FGT) clusters phrases, assigns them <span className="text-primary font-semibold">glyph codes</span>, and uses{' '}
          <span className="text-accent font-semibold">foveated memory</span> to deliver the right context at the right time—achieving{' '}
          <span className="text-secondary font-semibold">+46.7pp accuracy gain at a 256-token budget</span> on synthetic multi-turn dialogs.
        </p>

        {/* CTAs */}
        <div className="mt-10 flex items-center justify-center gap-x-6 gap-y-4 flex-wrap">
          <a
            href="/memory-console"
            className="rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-3 text-base font-semibold text-white shadow-lg hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 transition-all flex items-center gap-2"
          >
            <Brain className="w-5 h-5" />
            Memory Console
          </a>
          <a
            href="/demo"
            className="rounded-lg bg-gradient-to-r from-primary to-secondary px-6 py-3 text-base font-semibold text-white shadow-lg hover:opacity-90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary transition-all flex items-center gap-2"
          >
            <FileText className="w-5 h-5" />
            Try the Demo
          </a>
          <a
            href="/explore"
            className="rounded-lg bg-primary px-6 py-3 text-base font-semibold text-white shadow-lg hover:bg-primary/90 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary transition-all flex items-center gap-2"
          >
            <Map className="w-5 h-5" />
            Explore the Map
          </a>
          <a
            href="https://github.com/funwae/fractal-glyph-tape"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg border border-gray-600 px-6 py-3 text-base font-semibold text-white hover:bg-white/10 transition-all flex items-center gap-2"
          >
            <Github className="w-5 h-5" />
            Get the Code
          </a>
          <a
            href="https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md"
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg border border-gray-600 px-6 py-3 text-base font-semibold text-white hover:bg-white/10 transition-all flex items-center gap-2"
          >
            <FileText className="w-5 h-5" />
            Read the Research
          </a>
        </div>

        {/* Visual demo placeholder */}
        <div className="mt-16 rounded-2xl border border-gray-800 bg-gradient-to-br from-slate-900/50 to-slate-800/50 p-8 backdrop-blur">
          <div className="font-mono text-sm text-left">
            <div className="text-gray-400 mb-2"># Encode text to glyph representation</div>
            <div className="text-green-400">$ echo "Can you send me that file?" | fgt encode</div>
            <div className="text-secondary mt-2 text-2xl">谷阜</div>

            <div className="text-gray-400 mt-6 mb-2"># Decode glyph back to phrase family</div>
            <div className="text-green-400">$ echo "谷阜" | fgt decode</div>
            <div className="text-gray-300 mt-2">
              Phrase family #1247: <span className="text-accent">File-sharing request</span>
              <br />
              <span className="text-gray-500 text-xs">
                • "Can you send me that file?" (en)<br />
                • "Mind emailing the document?" (en)<br />
                • "你能发给我那个文件吗？" (zh)<br />
                • "¿Puedes enviarme ese archivo?" (es)
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
