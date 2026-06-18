"use client";

import { Alert } from "@heroui/react";

export default function DisclaimerCard({ text }) {
  return (
    <Alert status="warning" className="border border-amber-200 bg-amber-50">
      <Alert.Indicator />
      <Alert.Content>
        <Alert.Title className="text-amber-900">Medical disclaimer</Alert.Title>
        <Alert.Description className="text-amber-800">{text}</Alert.Description>
      </Alert.Content>
    </Alert>
  );
}
