"use client";

import { useEffect, useState } from "react";
import { Clock } from "lucide-react";
import { RegionsListResponse } from "@/types/memory";

interface MemoryTimelineProps {
  actorId: string;
}

interface TimelineEvent {
  type: "write" | "read";
  region: string;
  timestamp: string;
  count: number;
}

export default function MemoryTimeline({ actorId }: MemoryTimelineProps) {
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadTimeline();
  }, [actorId]);

  const loadTimeline = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/memory/regions?actor_id=${actorId}`);
      const data: RegionsListResponse = await response.json();

      if (data.status === "ok") {
        // Convert regions to timeline events
        const timelineEvents: TimelineEvent[] = data.regions.flatMap((region) => {
          const events: TimelineEvent[] = [];

          if (region.first_timestamp) {
            events.push({
              type: "write",
              region: region.region,
              timestamp: region.first_timestamp,
              count: region.record_count,
            });
          }

          return events;
        });

        // Sort by timestamp
        timelineEvents.sort(
          (a, b) =>
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );

        setEvents(timelineEvents);
      }
    } catch (error) {
      console.error("Error loading timeline:", error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center text-gray-500 mt-8">
        <Clock size={48} className="mx-auto mb-4 opacity-50 animate-spin" />
        <p>Loading timeline...</p>
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="text-center text-gray-500 mt-8">
        <Clock size={48} className="mx-auto mb-4 opacity-50" />
        <p>No memory events yet</p>
        <p className="text-sm mt-2">Timeline will appear as you use the memory system</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2">
        <Clock size={16} />
        Memory Timeline ({events.length} events)
      </h3>

      <div className="space-y-3">
        {events.map((event, idx) => (
          <div
            key={idx}
            className="bg-white/5 rounded-lg p-4 border border-white/10 hover:border-purple-500/50 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      event.type === "write" ? "bg-green-500" : "bg-blue-500"
                    }`}
                  />
                  <span className="text-sm font-medium text-white capitalize">
                    {event.type}
                  </span>
                </div>
                <div className="text-sm text-gray-400">
                  Region: <span className="text-purple-400">{event.region}</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {event.count} record{event.count !== 1 ? "s" : ""}
                </div>
              </div>
              <div className="text-xs text-gray-400">
                {new Date(event.timestamp).toLocaleString()}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
