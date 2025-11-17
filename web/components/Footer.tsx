import { Github, FileText, Mail } from 'lucide-react'

const links = {
  documentation: [
    { name: 'Research Abstract', href: 'https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md' },
    { name: 'Paper Outline', href: 'https://github.com/funwae/fractal-glyph-tape/blob/main/docs/PAPER_OUTLINE.md' },
    { name: 'Technical Docs', href: 'https://github.com/funwae/fractal-glyph-tape/tree/main/docs' },
    { name: 'API Documentation', href: 'https://github.com/funwae/fractal-glyph-tape/tree/main/docs' },
  ],
  community: [
    { name: 'GitHub Repository', href: 'https://github.com/funwae/fractal-glyph-tape' },
    { name: 'Discussions', href: 'https://github.com/funwae/fractal-glyph-tape/discussions' },
    { name: 'Issues', href: 'https://github.com/funwae/fractal-glyph-tape/issues' },
    { name: 'Contributing', href: 'https://github.com/funwae/fractal-glyph-tape/blob/main/CONTRIBUTING.md' },
  ],
}

export default function Footer() {
  return (
    <footer className="border-t border-gray-800 px-6 py-12 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-12">
          {/* About */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Fractal Glyph Tape</h3>
            <p className="text-sm text-gray-400 mb-4">
              A fractal-addressable phrase memory for semantic compression and cross-lingual LLMs.
            </p>
            <p className="text-sm text-gray-500">
              Built by <span className="text-primary font-semibold">Glyphd Labs</span>
            </p>
            <p className="text-xs text-gray-500 mt-2 italic">
              Turning the space of &quot;things we say&quot; into a structured, navigable map.
            </p>
          </div>

          {/* Documentation */}
          <div>
            <h3 className="text-sm font-semibold mb-4 text-gray-400">Documentation</h3>
            <ul className="space-y-2">
              {links.documentation.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-gray-300 hover:text-primary transition-colors"
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>

          {/* Community */}
          <div>
            <h3 className="text-sm font-semibold mb-4 text-gray-400">Community</h3>
            <ul className="space-y-2">
              {links.community.map((link) => (
                <li key={link.name}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-gray-300 hover:text-primary transition-colors"
                  >
                    {link.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="border-t border-gray-800 pt-8 flex flex-col sm:flex-row justify-between items-center gap-4">
          <p className="text-sm text-gray-500">
            © 2025 Glyphd Labs. Released under MIT License.
          </p>

          <div className="flex items-center gap-6">
            <a
              href="https://github.com/funwae/fractal-glyph-tape"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-primary transition-colors"
              aria-label="GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="https://github.com/funwae/fractal-glyph-tape/blob/main/docs/RESEARCH_ABSTRACT.md"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-primary transition-colors"
              aria-label="Research Paper"
            >
              <FileText className="w-5 h-5" />
            </a>
            <a
              href="mailto:contact@glyphd.com"
              className="text-gray-400 hover:text-primary transition-colors"
              aria-label="Contact"
            >
              <Mail className="w-5 h-5" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
