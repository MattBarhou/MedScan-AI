"use client";

import { Card } from "@heroui/react";

export default function HeatmapViewer({ heatmapBase64, caption }) {
  if (!heatmapBase64) {
    return null;
  }

  return (
    <Card className="border border-slate-200 bg-white shadow-sm">
      <Card.Header className="pb-2">
        <Card.Title className="text-lg text-slate-900">Model Attention Map</Card.Title>
        <Card.Description>
          {caption ||
            "Highlighted areas show where the model focused when making this prediction."}
        </Card.Description>
      </Card.Header>

      <Card.Content>
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50 p-3">
          <img
            src={heatmapBase64}
            alt="Grad-CAM heatmap overlay"
            className="mx-auto max-h-80 w-full rounded-lg object-contain"
          />
        </div>
      </Card.Content>
    </Card>
  );
}
