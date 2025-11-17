import Hero from '@/components/Hero'
import Features from '@/components/Features'
import HowItWorks from '@/components/HowItWorks'
import WhyItMatters from '@/components/WhyItMatters'
import ForResearchers from '@/components/ForResearchers'
import Footer from '@/components/Footer'

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-white">
      <Hero />
      <Features />
      <WhyItMatters />
      <HowItWorks />
      <ForResearchers />
      <Footer />
    </main>
  )
}
