'use client'

import { useState, useRef, useEffect } from 'react'

interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
}

interface ChatInterfaceProps {
  messages: Message[]
  onSendMessage: (content: string) => void
  isLoading: boolean
}

export default function ChatInterface({ messages, onSendMessage, isLoading }: ChatInterfaceProps) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim())
      setInput('')
    }
  }

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'user':
        return 'bg-blue-600/20 border-blue-500/50'
      case 'assistant':
        return 'bg-purple-600/20 border-purple-500/50'
      case 'system':
        return 'bg-slate-700/50 border-slate-600/50'
      default:
        return 'bg-slate-800 border-slate-700'
    }
  }

  const getRoleLabel = (role: string) => {
    switch (role) {
      case 'user':
        return 'You'
      case 'assistant':
        return 'Assistant'
      case 'system':
        return 'System'
      default:
        return role
    }
  }

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 flex flex-col h-[600px]">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700">
        <h2 className="text-lg font-semibold">Chat Interface</h2>
        <p className="text-sm text-slate-400">Messages are automatically stored as memories</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 ? (
          <div className="text-center text-slate-400 py-12">
            <p className="mb-2">No messages yet</p>
            <p className="text-sm">Start chatting to see memory-augmented responses</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`p-3 rounded-lg border ${getRoleColor(message.role)}`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium">{getRoleLabel(message.role)}</span>
                <span className="text-xs text-slate-400">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </span>
              </div>
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex items-center space-x-2 text-slate-400">
            <div className="animate-spin h-4 w-4 border-2 border-slate-400 border-t-transparent rounded-full"></div>
            <span className="text-sm">Processing...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-slate-700">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1 px-4 py-2 bg-slate-900 border border-slate-600 rounded text-white placeholder-slate-400 focus:outline-none focus:border-blue-500"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded font-medium transition-colors"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}
