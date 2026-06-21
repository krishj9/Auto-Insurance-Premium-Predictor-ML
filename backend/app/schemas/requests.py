from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictRequest(BaseModel):
    """Typed request payload for premium prediction."""

    model_config = ConfigDict(extra="forbid")

    driver_age: int = Field(..., ge=16, le=100, description="Driver age in years.")
    driver_experience: int = Field(..., ge=0, le=84, description="Driving experience in years.")
    previous_accidents: int = Field(..., ge=0, le=50, description="Number of prior accidents.")
    annual_mileage_x1000_km: int = Field(..., ge=1, le=300, description="Annual mileage in thousands of km.")
    car_manufacturing_year: int = Field(..., ge=1980, description="Vehicle manufacturing year.")
    car_age: int = Field(..., ge=0, le=60, description="Vehicle age in years.")

    @model_validator(mode="after")
    def validate_consistency(self) -> "PredictRequest":
        current_year = datetime.now(timezone.utc).year
        if self.car_manufacturing_year > current_year + 1:
            raise ValueError("car_manufacturing_year cannot be in the far future.")

        max_experience = max(self.driver_age - 16, 0)
        if self.driver_experience > max_experience:
            raise ValueError("driver_experience cannot exceed plausible years since driving age.")

        return self
