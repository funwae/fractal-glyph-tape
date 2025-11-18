'use client'

import { useState, useEffect, useRef } from 'react'
import ChatInterface from '@/components/memory-console/ChatInterface'
import ContextPanel from '@/components/memory-console/ContextPanel'
import GlyphPanel from '@/components/memory-console/GlyphPanel'
import StatsPanel from '@/components/memory-console/StatsPanel'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
}

interface MemoryContext {
  memories: any[]
  addresses: string[]
  glyphs: any[]
  token_estimate: number
  policy: string
  memories_selected: number
}

export default function MemoryConsolePage() {
  const [actorId, setActorId] = useState('hayden')
  const [messages, setMessages] = useState<Message[]>([])
  const [memoryContext, setMemoryContext] = useState<MemoryContext | null>(null)
  const [tokenBudget, setTokenBudget] = useState(2048)
  const [policyMode, setPolicyMode] = useState<'recent' | 'relevant' | 'mixed'>('mixed')
  const [isLoading, setIsLoading] = useState(false)
  const [apiUrl, setApiUrl] = useState('http://localhost:8001')
  const [stats, setStats] = useState<any>(null)

  // Load stats on mount and actor change
  useEffect(() => {
    loadStats()
  }, [actorId, apiUrl])

  const loadStats = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/memory/stats?actor_id=${actorId}`)
      if (response.ok) {
        const data = await response.json()
        setStats(data)
      }
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return

    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Call agent chat endpoint
      const response = await fetch(`${apiUrl}/api/agent/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          actor_id: actorId,
          messages: [...messages, userMessage].map(m => ({
            role: m.role,
            content: m.content
          })),
          token_budget: tokenBudget,
          mode: policyMode,
          llm_provider: 'mock'
        })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()

      // Add assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString()
      }

      setMessages(prev => [...prev, assistantMessage])
      setMemoryContext(data.memory_context)

      // Reload stats
      await loadStats()

    } catch (error) {
      console.error('Failed to send message:', error)
      const errorMessage: Message = {
        role: 'system',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleTestWrite = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`${apiUrl}/api/memory/write`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          actor_id: actorId,
          text: 'Today I started using a fractal glyph memory service for my AI projects.',
          tags: ['devlog', 'ai'],
          source: 'user'
        })
      })

      if (response.ok) {
        const data = await response.json()
        setMessages(prev => [...prev, {
          role: 'system',
          content: `✓ Test memory written: ${data.entry_id}\nAddress: ${data.address}`,
          timestamp: new Date().toISOString()
        }])
        await loadStats()
      }
    } catch (error) {
      console.error('Test write failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleTestRead = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(`${apiUrl}/api/memory/read`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          actor_id: actorId,
          query: 'fractal glyph memory AI projects',
          token_budget: tokenBudget,
          mode: policyMode
        })
      })

      if (response.ok) {
        const data = await response.json()
        setMemoryContext(data)
        setMessages(prev => [...prev, {
          role: 'system',
          content: `✓ Found ${data.memories_selected} memories (${data.token_estimate} tokens)`,
          timestamp: new Date().toISOString()
        }])
      }
    } catch (error) {
      console.error('Test read failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      <div className="container mx-auto px-4 py-6 max-w-7xl">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Fractal Glyph Memory Console
          </h1>
          <p className="text-slate-400">
            Interactive demo of fractal-addressable memory with policy-based foveation
          </p>
          <div className="mt-3 p-3 bg-amber-900/20 border border-amber-700/50 rounded-lg">
            <p className="text-sm text-amber-200">
              <strong>Note:</strong> Hosted demo is UI-only for now; run the full console locally via QUICKSTART to see live memory behavior.
            </p>
          </div>
        </div>

        {/* Config Bar */}
        <div className="bg-slate-800 rounded-lg p-4 mb-4 border border-slate-700">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Actor ID</label>
              <input
                type="text"
                value={actorId}
                onChange={(e) => setActorId(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Policy Mode</label>
              <select
                value={policyMode}
                onChange={(e) => setPolicyMode(e.target.value as any)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-white text-sm"
              >
                <option value="mixed">Mixed</option>
                <option value="recent">Recent</option>
                <option value="relevant">Relevant</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Token Budget</label>
              <input
                type="number"
                value={tokenBudget}
                onChange={(e) => setTokenBudget(parseInt(e.target.value))}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded text-white text-sm"
                min="128"
                max="16000"
              />
            </div>
            <div className="flex items-end">
              <button
                onClick={handleTestWrite}
                disabled={isLoading}
                className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 rounded text-sm font-medium transition-colors"
              >
                Test Write
              </button>
            </div>
            <div className="flex items-end">
              <button
                onClick={handleTestRead}
                disabled={isLoading}
                className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 rounded text-sm font-medium transition-colors"
              >
                Test Read
              </button>
            </div>
          </div>
        </div>

        {/* Main Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Left Column: Chat */}
          <div className="lg:col-span-2">
            <ChatInterface
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
            />
          </div>

          {/* Right Column: Context */}
          <div className="space-y-4">
            <StatsPanel stats={stats} />
            <ContextPanel memoryContext={memoryContext} />
            <GlyphPanel glyphs={memoryContext?.glyphs || []} />
          </div>
        </div>
      </div>
    </div>
  )
}
