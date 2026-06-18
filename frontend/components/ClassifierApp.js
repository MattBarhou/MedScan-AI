"use client";

import { useEffect, useState } from "react";
import { Alert, Button, Spinner } from "@heroui/react";

import DisclaimerCard from "@/components/DisclaimerCard";
import ImageUploader from "@/components/ImageUploader";
import PredictionResult from "@/components/PredictionResult";
import { predictImage, validateImageFile } from "@/lib/api";

const DEFAULT_DISCLAIMER =
  "This prediction is for educational purposes only and is not a medical diagnosis.";

export default function ClassifierApp() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  function handleFileSelect(file) {
    setError("");
    setResult(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
  }

  function handleClear() {
    setSelectedFile(null);
    setResult(null);
    setError("");
  }

  async function handlePredict() {
    if (!selectedFile) {
      setError("Please select an image before predicting.");
      return;
    }

    setIsLoading(true);
    setError("");
    setResult(null);

    try {
      const prediction = await predictImage(selectedFile);
      setResult(prediction);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <ImageUploader
        previewUrl={previewUrl}
        selectedFile={selectedFile}
        onFileSelect={handleFileSelect}
        onClear={handleClear}
        disabled={isLoading}
      />

      <div className="flex justify-center">
        <Button
          type="button"
          size="lg"
          onPress={handlePredict}
          isDisabled={!selectedFile || isLoading}
          className="min-w-40"
        >
          {isLoading ? (
            <>
              <Spinner size="sm" />
              Predicting...
            </>
          ) : (
            "Predict"
          )}
        </Button>
      </div>

      {error && (
        <Alert status="danger" className="border border-red-200 bg-red-50">
          <Alert.Indicator />
          <Alert.Content>
            <Alert.Title className="text-red-900">Something went wrong</Alert.Title>
            <Alert.Description className="text-red-800">{error}</Alert.Description>
          </Alert.Content>
        </Alert>
      )}

      <PredictionResult result={result} />

      <DisclaimerCard text={result?.disclaimer || DEFAULT_DISCLAIMER} />
    </div>
  );
}
