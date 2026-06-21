export interface PredictRequest {
  driver_age: number;
  driver_experience: number;
  previous_accidents: number;
  annual_mileage_x1000_km: number;
  car_manufacturing_year: number;
  car_age: number;
}

export interface PredictResponse {
  predicted_premium: number;
  model_name: string;
  model_version: string;
  disclaimer: string;
}

export interface FeatureField {
  name: string;
  type: string;
  description: string;
  example: string | number | boolean | null;
}

export interface FeatureSchemaResponse {
  features: FeatureField[];
}
