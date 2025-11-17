"""
FastAPI application for FGMS

Endpoints:
- POST /api/memory/write - Write a memory
- POST /api/memory/read - Read memories with foveation
- GET /api/memory/stats - Get storage statistics
- POST /api/agent/chat - Chat with agent using memory
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import hashlib

from ..models import MemoryEntry, FractalAddress, Glyph
from ..storage import SQLiteMemoryStore
from ..foveation import FoveationEngine


# Pydantic models for API
class WriteMemoryRequest(BaseModel):
    actor_id: str = Field(..., description="Actor ID")
    text: str = Field(..., description="Memory text content")
    tags: List[str] = Field(default_factory=list, description="Optional tags")
    source: str = Field(default="user", description="Source of memory")


class ReadMemoryRequest(BaseModel):
    actor_id: str = Field(..., description="Actor ID")
    query: Optional[str] = Field(None, description="Optional query for relevance")
    token_budget: int = Field(2048, description="Token budget", ge=128, le=32000)
    mode: str = Field("mixed", description="Policy mode: recent, relevant, or mixed")
    world: Optional[str] = Field(None, description="Optional world filter")
    region: Optional[str] = Field(None, description="Optional region filter")
    tags: Optional[List[str]] = Field(None, description="Optional tag filters")


class AgentChatRequest(BaseModel):
    actor_id: str = Field(..., description="Actor ID")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    token_budget: int = Field(2048, description="Memory token budget", ge=128, le=16000)
    mode: str = Field("mixed", description="Memory policy mode")
    llm_provider: str = Field("mock", description="LLM provider: mock, openai, anthropic")
    model: Optional[str] = Field(None, description="Model name")


class WriteMemoryResponse(BaseModel):
    entry_id: str
    address: str
    token_estimate: int


class ReadMemoryResponse(BaseModel):
    memories: List[Dict[str, Any]]
    addresses: List[str]
    glyphs: List[Dict[str, Any]]
    token_estimate: int
    policy: str
    candidates_considered: int
    memories_selected: int


class AgentChatResponse(BaseModel):
    response: str
    memory_context: Dict[str, Any]
    memories_used: int
    tokens_used: int


def create_address_from_text(actor_id: str, text: str, timestamp: datetime) -> FractalAddress:
    """
    Generate a fractal address for a memory entry.

    Uses content hashing to determine location in fractal space.
    """
    # Hash the content to generate tri_path
    content_hash = hashlib.md5(f"{actor_id}:{text}".encode()).hexdigest()

    # Extract tri_path from hash (convert hex to base-3)
    hex_value = int(content_hash[:8], 16)
    tri_path = ""
    for _ in range(6):
        tri_path = str(hex_value % 3) + tri_path
        hex_value //= 3

    # Determine world and region from content
    # Simple heuristic: use first words as region classifier
    words = text.lower().split()
    if any(w in words for w in ["project", "code", "software", "tech"]):
        region = "tech"
    elif any(w in words for w in ["personal", "me", "i", "my"]):
        region = "personal"
    else:
        region = "general"

    return FractalAddress(
        world="earth",  # Default world
        region=region,
        tri_path=tri_path,
        depth=len(tri_path),
        time_slice=timestamp
    )


def create_mock_glyphs(text: str) -> List[Glyph]:
    """
    Create mock glyphs for demo purposes.

    In production, this would use the actual glyph encoder.
    """
    # Simple mock: create one glyph per sentence
    sentences = text.split('.')
    glyphs = []

    for i, sentence in enumerate(sentences[:3]):  # Max 3 glyphs
        if not sentence.strip():
            continue

        glyph_id = hash(sentence.strip()) % 10000
        glyphs.append(Glyph(
            glyph_id=glyph_id,
            glyph_str=f"谷{chr(ord('阜') + i % 10)}",  # Mock glyph
            cluster_id=glyph_id // 10,
            frequency=1,
            semantic_summary=sentence.strip()[:50]
        ))

    return glyphs


# Initialize store and engine (singleton)
store = SQLiteMemoryStore("data/fgms_memory.db")
engine = FoveationEngine(store)


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Fractal Glyph Memory System API",
        description="REST API for fractal-addressable memory with foveation",
        version="0.1.0"
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/memory/write", response_model=WriteMemoryResponse)
    async def write_memory(request: WriteMemoryRequest) -> WriteMemoryResponse:
        """
        Write a new memory entry.

        The system automatically:
        - Generates a fractal address based on content
        - Creates glyph representations
        - Stores with full-text indexing
        """
        try:
            # Generate address
            timestamp = datetime.now()
            address = create_address_from_text(
                actor_id=request.actor_id,
                text=request.text,
                timestamp=timestamp
            )

            # Create glyphs
            glyphs = create_mock_glyphs(request.text)

            # Create memory entry
            entry = MemoryEntry(
                entry_id=str(uuid.uuid4()),
                actor_id=request.actor_id,
                address=address,
                text=request.text,
                glyphs=glyphs,
                tags=request.tags,
                source=request.source,
                created_at=timestamp,
                token_estimate=int(len(request.text.split()) * 1.3)
            )

            # Write to store
            entry_id = store.write(entry)

            return WriteMemoryResponse(
                entry_id=entry_id,
                address=address.to_string(),
                token_estimate=entry.token_estimate
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memory/read", response_model=ReadMemoryResponse)
    async def read_memory(request: ReadMemoryRequest) -> ReadMemoryResponse:
        """
        Read memories using foveation policy.

        Retrieves memories within token budget using specified policy:
        - recent: Most recent memories first
        - relevant: Most relevant to query
        - mixed: Blend of recent and relevant
        """
        try:
            result = engine.retrieve(
                actor_id=request.actor_id,
                query=request.query,
                token_budget=request.token_budget,
                mode=request.mode,
                world=request.world,
                region=request.region,
                tags=request.tags
            )

            return ReadMemoryResponse(**result)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/memory/stats")
    async def get_stats(actor_id: Optional[str] = None) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = store.get_stats(actor_id=actor_id)
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agent/chat", response_model=AgentChatResponse)
    async def agent_chat(request: AgentChatRequest) -> AgentChatResponse:
        """
        Chat with agent using memory-augmented context.

        Flow:
        1. Extract user's message
        2. Retrieve relevant memories within budget
        3. Build prompt with memory context
        4. Call LLM (mock for demo)
        5. Store assistant response as memory
        6. Return response with metadata
        """
        try:
            if not request.messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            # Get last user message
            user_message = None
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

            if not user_message:
                raise HTTPException(status_code=400, detail="No user message found")

            # Store user message as memory
            user_entry_id = await write_memory(WriteMemoryRequest(
                actor_id=request.actor_id,
                text=user_message,
                tags=["chat", "user"],
                source="user"
            ))

            # Retrieve memory context
            memory_context = engine.retrieve(
                actor_id=request.actor_id,
                query=user_message,
                token_budget=request.token_budget,
                mode=request.mode
            )

            # Build prompt with memory
            memory_text = "\n\n".join([
                f"[{m['created_at']}] {m['text']}"
                for m in memory_context['memories'][-10:]  # Last 10 for context
            ])

            prompt = f"""You are an AI assistant with access to the user's memory.

Context from memory:
{memory_text}

Current conversation:
{chr(10).join([f"{m['role']}: {m['content']}" for m in request.messages[-3:]])}

Respond helpfully using the context provided."""

            # Mock LLM response (in production, call actual LLM)
            if request.llm_provider == "mock":
                response_text = f"I understand you said: '{user_message}'. "
                if memory_context['memories_selected'] > 0:
                    response_text += f"Based on {memory_context['memories_selected']} relevant memories, I can help you with that."
                else:
                    response_text += "I don't have much context yet, but I'm learning!"
            else:
                # TODO: Implement actual LLM calls
                response_text = "LLM integration not implemented yet. Use llm_provider='mock'."

            # Store assistant response as memory
            await write_memory(WriteMemoryRequest(
                actor_id=request.actor_id,
                text=response_text,
                tags=["chat", "assistant"],
                source="agent"
            ))

            return AgentChatResponse(
                response=response_text,
                memory_context=memory_context,
                memories_used=memory_context['memories_selected'],
                tokens_used=memory_context['token_estimate']
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        """API root endpoint."""
        return {
            "name": "Fractal Glyph Memory System API",
            "version": "0.1.0",
            "endpoints": {
                "write": "/api/memory/write",
                "read": "/api/memory/read",
                "stats": "/api/memory/stats",
                "chat": "/api/agent/chat"
            }
        }

    return app


# Create app instance
app = create_app()
