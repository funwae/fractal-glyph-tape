import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fractal Glyph Tape | Dense, Cross-Lingual Phrase Memory for LLMs",
  description: "Fractal Glyph Tape (FGT) is a fractal-addressable phrase memory that compresses language into semantic glyph codes, extends LLM context, and bridges languages. Open-source research from Glyphd Labs.",
  keywords: [
    "fractal glyph tape",
    "semantic compression",
    "cross-lingual LLMs",
    "phrase memory",
    "hybrid tokenization",
    "glyph encoding",
    "LLM context extension",
    "multilingual embeddings",
    "fractal addressing",
    "glyphd labs"
  ],
  authors: [{ name: "Glyphd Labs" }],
  openGraph: {
    title: "Fractal Glyph Tape | Dense, Cross-Lingual Phrase Memory for LLMs",
    description: "A fractal-addressable phrase memory that makes language denser, more cross-lingual, and more explorable for LLMs.",
    type: "website",
    url: "https://glyphd.com/fgt",
  },
  twitter: {
    card: "summary_large_image",
    title: "Fractal Glyph Tape",
    description: "A fractal-addressable phrase memory that makes language denser, more cross-lingual, and more explorable for LLMs.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
