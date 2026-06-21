import { Suspense, use, useState } from "react";

import { PredictionResult } from "../components/PredictionResult";
import { ProspectForm } from "../components/ProspectForm";
import { getFeatureSchema, predictPremium } from "../services/api";
import type { FeatureSchemaResponse, PredictRequest, PredictResponse } from "../types/api";


const featureSchemaPromise: Promise<FeatureSchemaResponse> = getFeatureSchema().catch(() => ({
  features: [],
}));


export function QuotePage() {
  return (
    <Suspense
      fallback={
        <main className="quote-page">
          <h1>Car Insurance Premium Quote</h1>
          <div className="schema-loading-card">
            <span className="loading-spinner" aria-hidden="true" />
            <p>Loading feature schema...</p>
          </div>
        </main>
      }
    >
      <QuotePageContent />
    </Suspense>
  );
}


function QuotePageContent() {
  const featureSchema = use(featureSchemaPromise);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handlePredict(payload: PredictRequest): Promise<PredictResponse> {
    try {
      const response = await predictPremium(payload);
      setResult(response);
      setError(null);
      return response;
    } catch (caughtError) {
      const message = caughtError instanceof Error ? caughtError.message : "Prediction failed.";
      setError(message);
      setResult(null);
      throw caughtError;
    }
  }

  return (
    <main className="quote-page">
      <h1>Car Insurance Premium Quote</h1>
      <p className="quote-page-subtitle">
        Get an instant demo estimate based on your driver and vehicle profile.
      </p>
      <div className="quote-grid">
        <ProspectForm onAction={handlePredict} featureSchema={featureSchema} />
        <PredictionResult result={result} error={error} />
      </div>
    </main>
  );
}
