"""FastAPI backend for Fractal Glyph Tape visualization."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from loguru import logger


# Pydantic models for API responses
class TapeOverview(BaseModel):
    """Overview statistics for the tape."""
    n_clusters: int
    n_phrases: int
    embedding_dim: int
    projection_method: str
    fractal_depth: int


class ClusterInfo(BaseModel):
    """Information about a cluster."""
    cluster_id: int
    glyph_string: str
    size: int
    fractal_address: str
    coords: tuple[float, float]
    examples: List[Dict[str, str]]


class MapPoint(BaseModel):
    """A point on the fractal map."""
    cluster_id: int
    glyph_string: str
    x: float
    y: float
    size: int
    fractal_address: str


def create_app(tape_db_path: str = "tape/v1/tape_index.db") -> FastAPI:
    """
    Create FastAPI application for tape visualization.

    Args:
        tape_db_path: Path to tape SQLite database

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Fractal Glyph Tape Visualizer",
        description="API for exploring the semantic fractal tape",
        version="0.1.0"
    )

    # Enable CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store tape path in app state
    app.state.tape_db_path = tape_db_path

    def get_db_connection():
        """Get database connection."""
        db_path = Path(app.state.tape_db_path)
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Tape database not found: {db_path}")
        return sqlite3.connect(str(db_path))

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the main visualization page."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fractal Glyph Tape Visualizer</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .info { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Fractal Glyph Tape Visualizer</h1>
            <div class="info">
                <h2>API Endpoints</h2>
                <ul>
                    <li><a href="/api/tape/overview">/api/tape/overview</a> - Tape statistics</li>
                    <li><a href="/api/tape/map">/api/tape/map</a> - Get all map points</li>
                    <li><a href="/api/tape/clusters/0">/api/tape/clusters/{cluster_id}</a> - Cluster details</li>
                    <li><a href="/api/tape/search?query=test">/api/tape/search</a> - Search glyphs</li>
                    <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                    <li><a href="/viz">/viz</a> - Interactive fractal map</li>
                </ul>
            </div>
        </body>
        </html>
        """

    @app.get("/api/tape/overview", response_model=Dict[str, Any])
    async def get_tape_overview():
        """Get overview statistics for the tape."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Count clusters
            cursor.execute("SELECT COUNT(*) FROM clusters")
            n_clusters = cursor.fetchone()[0]

            # Get total size (sum of all cluster sizes)
            cursor.execute("SELECT SUM(size) FROM clusters")
            n_phrases = cursor.fetchone()[0] or 0

            conn.close()

            return {
                "n_clusters": n_clusters,
                "n_phrases": n_phrases,
                "embedding_dim": 384,  # From config
                "projection_method": "umap",
                "fractal_depth": 10,
            }

        except Exception as e:
            logger.error(f"Error getting overview: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tape/clusters/{cluster_id}", response_model=Dict[str, Any])
    async def get_cluster_details(cluster_id: int):
        """Get detailed information about a specific cluster."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Get cluster info with glyph and address
            cursor.execute("""
                SELECT c.cluster_id, c.size, c.metadata,
                       g.glyph_string,
                       a.fractal_address, a.x_coord, a.y_coord
                FROM clusters c
                JOIN glyphs g ON c.cluster_id = g.cluster_id
                JOIN addresses a ON c.cluster_id = a.cluster_id
                WHERE c.cluster_id = ?
            """, (cluster_id,))

            row = cursor.fetchone()
            conn.close()

            if not row:
                raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")

            cluster_id, size, metadata_str, glyph_string, fractal_address, x, y = row

            # Parse metadata
            try:
                metadata = eval(metadata_str) if metadata_str else {}
            except:
                metadata = {}

            examples = metadata.get("examples", [])

            return {
                "cluster_id": cluster_id,
                "glyph_string": glyph_string,
                "size": size,
                "fractal_address": fractal_address,
                "coords": (float(x), float(y)),
                "examples": examples[:10],  # Limit to 10 examples
                "metadata": metadata,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting cluster details: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tape/map", response_model=List[Dict[str, Any]])
    async def get_fractal_map(limit: Optional[int] = Query(None, description="Limit number of points")):
        """Get all points for the fractal map visualization."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            query = """
                SELECT a.cluster_id, a.x_coord, a.y_coord, a.fractal_address,
                       g.glyph_string, c.size
                FROM addresses a
                JOIN glyphs g ON a.cluster_id = g.cluster_id
                JOIN clusters c ON a.cluster_id = c.cluster_id
                ORDER BY a.cluster_id
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            points = []
            for cluster_id, x, y, fractal_address, glyph_string, size in rows:
                points.append({
                    "cluster_id": cluster_id,
                    "glyph_string": glyph_string,
                    "x": float(x),
                    "y": float(y),
                    "size": size,
                    "fractal_address": fractal_address,
                })

            return points

        except Exception as e:
            logger.error(f"Error getting map data: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/tape/search")
    async def search_glyphs(
        query: str = Query(..., description="Search query"),
        limit: int = Query(10, description="Maximum results")
    ):
        """Search for glyphs by partial match."""
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Search by glyph string or fractal address
            cursor.execute("""
                SELECT DISTINCT g.cluster_id, g.glyph_string,
                       a.fractal_address, c.size
                FROM glyphs g
                JOIN addresses a ON g.cluster_id = a.cluster_id
                JOIN clusters c ON g.cluster_id = c.cluster_id
                WHERE g.glyph_string LIKE ? OR a.fractal_address LIKE ?
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit))

            rows = cursor.fetchall()
            conn.close()

            results = []
            for cluster_id, glyph_string, fractal_address, size in rows:
                results.append({
                    "cluster_id": cluster_id,
                    "glyph_string": glyph_string,
                    "fractal_address": fractal_address,
                    "size": size,
                })

            return results

        except Exception as e:
            logger.error(f"Error searching: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/viz", response_class=HTMLResponse)
    async def get_visualization():
        """Serve interactive fractal map visualization."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fractal Glyph Tape - Interactive Map</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                }
                #controls {
                    padding: 10px;
                    background: #f0f0f0;
                    border-bottom: 1px solid #ccc;
                }
                #map {
                    flex: 1;
                    width: 100%;
                }
                #info {
                    position: absolute;
                    top: 60px;
                    right: 10px;
                    background: white;
                    padding: 15px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    max-width: 300px;
                    display: none;
                }
                button {
                    padding: 8px 16px;
                    margin: 0 5px;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <div id="controls">
                <h2 style="margin: 0 0 10px 0;">Fractal Glyph Tape - Interactive Map</h2>
                <button onclick="loadMap()">Load Map</button>
                <button onclick="resetView()">Reset View</button>
                <span id="status" style="margin-left: 10px;"></span>
            </div>
            <div id="map"></div>
            <div id="info">
                <h3>Cluster Info</h3>
                <div id="cluster-details"></div>
            </div>

            <script>
                let mapData = null;

                async function loadMap() {
                    const status = document.getElementById('status');
                    status.textContent = 'Loading map data...';

                    try {
                        const response = await fetch('/api/tape/map');
                        mapData = await response.json();
                        status.textContent = `Loaded ${mapData.length} clusters`;
                        renderMap();
                    } catch (error) {
                        status.textContent = 'Error loading map: ' + error.message;
                        console.error(error);
                    }
                }

                function renderMap() {
                    if (!mapData) return;

                    const trace = {
                        x: mapData.map(p => p.x),
                        y: mapData.map(p => p.y),
                        text: mapData.map(p => `${p.glyph_string} (${p.cluster_id})<br>Size: ${p.size}<br>${p.fractal_address}`),
                        mode: 'markers',
                        type: 'scatter',
                        marker: {
                            size: mapData.map(p => Math.sqrt(p.size) + 3),
                            color: mapData.map(p => p.cluster_id),
                            colorscale: 'Viridis',
                            showscale: true,
                            colorbar: { title: 'Cluster ID' }
                        },
                        hovertemplate: '%{text}<extra></extra>'
                    };

                    const layout = {
                        title: 'Semantic Phrase Space (Fractal Map)',
                        xaxis: { title: 'Dimension 1' },
                        yaxis: { title: 'Dimension 2' },
                        hovermode: 'closest'
                    };

                    Plotly.newPlot('map', [trace], layout);

                    // Add click handler
                    document.getElementById('map').on('plotly_click', async function(data) {
                        const point = data.points[0];
                        const clusterId = mapData[point.pointIndex].cluster_id;
                        await showClusterInfo(clusterId);
                    });
                }

                async function showClusterInfo(clusterId) {
                    try {
                        const response = await fetch(`/api/tape/clusters/${clusterId}`);
                        const cluster = await response.json();

                        const info = document.getElementById('info');
                        const details = document.getElementById('cluster-details');

                        let html = `
                            <p><strong>Glyph:</strong> ${cluster.glyph_string}</p>
                            <p><strong>Cluster ID:</strong> ${cluster.cluster_id}</p>
                            <p><strong>Size:</strong> ${cluster.size} phrases</p>
                            <p><strong>Address:</strong> ${cluster.fractal_address}</p>
                            <p><strong>Coords:</strong> (${cluster.coords[0].toFixed(3)}, ${cluster.coords[1].toFixed(3)})</p>
                        `;

                        if (cluster.examples && cluster.examples.length > 0) {
                            html += '<p><strong>Examples:</strong></p><ul>';
                            cluster.examples.slice(0, 5).forEach(ex => {
                                html += `<li>${ex.text || 'N/A'}</li>`;
                            });
                            html += '</ul>';
                        }

                        details.innerHTML = html;
                        info.style.display = 'block';
                    } catch (error) {
                        console.error('Error loading cluster info:', error);
                    }
                }

                function resetView() {
                    Plotly.relayout('map', {
                        'xaxis.autorange': true,
                        'yaxis.autorange': true
                    });
                }

                // Auto-load on page load
                window.onload = loadMap;
            </script>
        </body>
        </html>
        """

    return app


# For running with uvicorn directly
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
