import { useFormStatus } from "react-dom";

import type { FeatureSchemaResponse, PredictRequest, PredictResponse } from "../types/api";

interface ProspectFormProps {
  onAction: (payload: PredictRequest) => Promise<PredictResponse>;
  featureSchema: FeatureSchemaResponse;
}

export function ProspectForm({ onAction, featureSchema }: ProspectFormProps) {
  async function submitAction(formData: FormData): Promise<void> {
    const payload: PredictRequest = {
      driver_age: Number(formData.get("driver_age")),
      driver_experience: Number(formData.get("driver_experience")),
      previous_accidents: Number(formData.get("previous_accidents")),
      annual_mileage_x1000_km: Number(formData.get("annual_mileage_x1000_km")),
      car_manufacturing_year: Number(formData.get("car_manufacturing_year")),
      car_age: Number(formData.get("car_age")),
    };

    try {
      await onAction(payload);
    } catch {
      // The page-level state already captures and renders user-friendly errors.
    }
  }

  return (
    <form action={submitAction} className="quote-form-card">
      <h2 className="card-title">Prospect Details</h2>
      <p className="card-subtitle">
        {featureSchema.features.length > 0
          ? `Loaded ${featureSchema.features.length} fields from backend schema.`
          : "Using default form fields."}
      </p>

      <div className="form-grid">
        <label htmlFor="driver_age">Driver Age</label>
        <input id="driver_age" name="driver_age" type="number" min={16} max={100} step={1} required />

        <label htmlFor="driver_experience">Driver Experience (years)</label>
        <input id="driver_experience" name="driver_experience" type="number" min={0} max={84} step={1} required />

        <label htmlFor="previous_accidents">Previous Accidents</label>
        <input id="previous_accidents" name="previous_accidents" type="number" min={0} max={50} step={1} required />

        <label htmlFor="annual_mileage_x1000_km">Annual Mileage (x1000 km)</label>
        <input
          id="annual_mileage_x1000_km"
          name="annual_mileage_x1000_km"
          type="number"
          min={1}
          max={300}
          step={1}
          required
        />

        <label htmlFor="car_manufacturing_year">Car Manufacturing Year</label>
        <input
          id="car_manufacturing_year"
          name="car_manufacturing_year"
          type="number"
          min={1980}
          max={2100}
          step={1}
          required
        />

        <label htmlFor="car_age">Car Age</label>
        <input id="car_age" name="car_age" type="number" min={0} max={60} step={1} required />
      </div>

      <SubmitButton />
    </form>
  );
}

function SubmitButton() {
  const { pending } = useFormStatus();

  return (
    <button type="submit" className="quote-submit-button" disabled={pending}>
      {pending ? (
        <span className="inline-spinner-wrap">
          <span className="inline-spinner" aria-hidden="true" />
          Calculating...
        </span>
      ) : (
        "Get Quote"
      )}
    </button>
  );
}
