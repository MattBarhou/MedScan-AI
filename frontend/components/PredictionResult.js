"use client";

import { Card, Chip, ProgressBar } from "@heroui/react";

import HeatmapViewer from "@/components/HeatmapViewer";

export default function PredictionResult({ result }) {
  if (!result) {
    return null;
  }

  const confidencePercent = Math.round(result.confidence * 100);

  return (
    <div className="space-y-6">
      <Card className="border border-slate-200 bg-white shadow-sm">
        <Card.Header className="pb-2">
          <Card.Title className="text-lg text-slate-900">Prediction Result</Card.Title>
          <Card.Description>Filename: {result.filename}</Card.Description>
        </Card.Header>

        <Card.Content className="space-y-6">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm text-slate-500">Predicted class</span>
            <Chip
              color={result.predicted_class === "Disease" ? "danger" : "success"}
              variant="soft"
            >
              {result.predicted_class}
            </Chip>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-500">Confidence</span>
              <span className="font-medium text-slate-900">{confidencePercent}%</span>
            </div>
            <ProgressBar aria-label="Confidence" value={confidencePercent}>
              <ProgressBar.Track>
                <ProgressBar.Fill />
              </ProgressBar.Track>
            </ProgressBar>
          </div>

          <div className="space-y-4">
            <p className="text-sm font-medium text-slate-900">Class probabilities</p>
            {Object.entries(result.probabilities).map(([label, value]) => {
              const percent = Math.round(value * 100);

              return (
                <div key={label} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-slate-700">{label}</span>
                    <span className="text-slate-500">{percent}%</span>
                  </div>
                  <ProgressBar aria-label={`${label} probability`} value={percent}>
                    <ProgressBar.Track>
                      <ProgressBar.Fill />
                    </ProgressBar.Track>
                  </ProgressBar>
                </div>
              );
            })}
          </div>
        </Card.Content>
      </Card>

      <HeatmapViewer
        heatmapBase64={result.heatmap_base64}
        caption={result.heatmap_caption}
      />
    </div>
  );
}
