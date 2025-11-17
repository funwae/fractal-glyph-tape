"use client";

import { useEffect, useRef } from "react";

interface LayoutPoint {
  cluster_id: string;
  x: number;
  y: number;
  glyph: string;
  language: string;
  frequency: number;
}

interface ClusterMapProps {
  layout: LayoutPoint[];
  selectedClusterId: string | null;
  hoveredClusterId: string | null;
  onClusterClick: (clusterId: string) => void;
  onClusterHover: (clusterId: string | null) => void;
}

export default function ClusterMap({
  layout,
  selectedClusterId,
  hoveredClusterId,
  onClusterClick,
  onClusterHover,
}: ClusterMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Language color map
  const getLanguageColor = (language: string): string => {
    const colors: Record<string, string> = {
      en: "#3b82f6", // blue
      zh: "#ef4444", // red
      es: "#10b981", // green
      fr: "#f59e0b", // amber
      de: "#8b5cf6", // purple
      ja: "#ec4899", // pink
      unknown: "#6b7280", // gray
    };
    return colors[language] || colors.unknown;
  };

  // Frequency to size mapping
  const getPointSize = (frequency: number): number => {
    const minSize = 3;
    const maxSize = 12;
    const logFreq = Math.log(frequency + 1);
    const maxLogFreq = Math.log(
      Math.max(...layout.map((p) => p.frequency)) + 1
    );
    return minSize + (maxSize - minSize) * (logFreq / maxLogFreq);
  };

  // Find cluster at position
  const findClusterAt = (
    x: number,
    y: number,
    canvas: HTMLCanvasElement
  ): string | null => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const canvasX = (x - rect.left) * scaleX;
    const canvasY = (y - rect.top) * scaleY;

    for (const point of layout) {
      const px = point.x * canvas.width;
      const py = point.y * canvas.height;
      const size = getPointSize(point.frequency);

      const dist = Math.sqrt((canvasX - px) ** 2 + (canvasY - py) ** 2);
      if (dist <= size) {
        return point.cluster_id;
      }
    }

    return null;
  };

  // Draw visualization
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;

    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    // Clear canvas
    ctx.fillStyle = "#111827"; // gray-900
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Draw points
    for (const point of layout) {
      const x = point.x * rect.width;
      const y = point.y * rect.height;
      const size = getPointSize(point.frequency);
      const color = getLanguageColor(point.language);

      // Highlight selected or hovered
      const isSelected = point.cluster_id === selectedClusterId;
      const isHovered = point.cluster_id === hoveredClusterId;

      if (isSelected || isHovered) {
        // Draw glow
        ctx.shadowBlur = 15;
        ctx.shadowColor = color;
      } else {
        ctx.shadowBlur = 0;
      }

      // Draw point
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.globalAlpha = isSelected || isHovered ? 1.0 : 0.7;
      ctx.fill();
      ctx.globalAlpha = 1.0;

      // Draw border for selected
      if (isSelected) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw glyph for hovered or selected
      if ((isHovered || isSelected) && point.glyph) {
        ctx.shadowBlur = 0;
        ctx.fillStyle = "#ffffff";
        ctx.font = `bold ${size * 2}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(point.glyph, x, y - size - 10);
      }
    }

    ctx.shadowBlur = 0;
  }, [layout, selectedClusterId, hoveredClusterId]);

  // Handle mouse events
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const clusterId = findClusterAt(e.clientX, e.clientY, canvas);
    onClusterHover(clusterId);
  };

  const handleMouseLeave = () => {
    onClusterHover(null);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const clusterId = findClusterAt(e.clientX, e.clientY, canvas);
    if (clusterId) {
      onClusterClick(clusterId);
    }
  };

  return (
    <div ref={containerRef} className="w-full h-full relative bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        className="cursor-pointer"
      />

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-semibold mb-2">Languages</h3>
        <div className="space-y-1 text-xs">
          {["en", "zh", "es", "fr", "de", "ja"].map((lang) => (
            <div key={lang} className="flex items-center gap-2">
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: getLanguageColor(lang) }}
              />
              <span className="text-gray-300">{lang.toUpperCase()}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Instructions */}
      <div className="absolute top-4 left-4 bg-gray-800/90 backdrop-blur-sm rounded-lg p-3 border border-gray-700">
        <p className="text-xs text-gray-300">
          <span className="font-semibold">Hover</span> to preview •{" "}
          <span className="font-semibold">Click</span> for details
        </p>
      </div>
    </div>
  );
}
