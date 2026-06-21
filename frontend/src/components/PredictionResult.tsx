import type { PredictResponse } from "../types/api";

interface PredictionResultProps {
  result: PredictResponse | null;
  error: string | null;
}

export function PredictionResult({ result, error }: PredictionResultProps) {
  const formatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
  });

  if (error !== null) {
    return (
      <section className="result-card result-card-error">
        <h2 className="card-title">Estimated Annual Premium (USD)</h2>
        <p role="alert" className="error-message">
          {error}
        </p>
      </section>
    );
  }

  if (result !== null) {
    return (
      <section className="result-card result-card-success">
        <h2 className="card-title">Estimated Annual Premium (USD)</h2>
        <p className="premium-value">{formatter.format(result.predicted_premium)}</p>
        <p className="result-meta">
          Model: <strong>{result.model_name}</strong>
        </p>
        <p className="result-meta">
          Version: <strong>{result.model_version}</strong>
        </p>
        <p className="result-disclaimer">{result.disclaimer}</p>
      </section>
    );
  }

  return (
    <section className="result-card">
      <h2 className="card-title">Estimated Annual Premium (USD)</h2>
      <p className="result-placeholder">Submit the form to see your predicted premium.</p>
    </section>
  );
}
