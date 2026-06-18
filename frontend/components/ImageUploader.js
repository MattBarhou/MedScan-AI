"use client";

import { useRef } from "react";
import { Button, Card } from "@heroui/react";

export default function ImageUploader({
  previewUrl,
  selectedFile,
  onFileSelect,
  onClear,
  disabled,
}) {
  const inputRef = useRef(null);

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    onFileSelect(file || null);
  }

  return (
    <Card className="border border-slate-200 bg-white shadow-sm">
      <Card.Header className="pb-2">
        <Card.Title className="text-lg text-slate-900">Upload Image</Card.Title>
        <Card.Description>
          Choose one medical image in JPG, JPEG, or PNG format.
        </Card.Description>
      </Card.Header>

      <Card.Content className="space-y-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/jpg,.jpg,.jpeg,.png"
            onChange={handleFileChange}
            disabled={disabled}
            className="block w-full cursor-pointer rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700 file:mr-4 file:rounded-md file:border-0 file:bg-slate-200 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-slate-700 hover:file:bg-slate-300 disabled:cursor-not-allowed disabled:opacity-60"
          />

          {selectedFile && (
            <Button
              type="button"
              variant="secondary"
              onPress={onClear}
              isDisabled={disabled}
            >
              Clear
            </Button>
          )}
        </div>

        <div className="flex min-h-[240px] items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4">
          {previewUrl ? (
            <img
              src={previewUrl}
              alt="Selected preview"
              className="max-h-72 w-full rounded-lg object-contain"
            />
          ) : (
            <p className="text-sm text-slate-500">Image preview will appear here</p>
          )}
        </div>
      </Card.Content>
    </Card>
  );
}
