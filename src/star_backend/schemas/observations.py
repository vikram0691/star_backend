from pydantic import BaseModel, Field, field_validator
from datetime import date

class ObservationBase(BaseModel):
    """Defines the physical reality of a single environmental observation. This is the gatekeeper schema for all observations. This is the 'Gatekeeper' for all incoming Excel/CSV data."""

    rain_fall: float = Field(ge = 0, description="Total monthly precipitation measured in millimeters (mm).")
    evaporation: float = Field(ge = 0, description="The total monthly evaporation measured in millimeters (mm).")

    minimum_humidity: float = Field(ge = 0, le = 100, description="Monthly average of daily minimum relative humidity (%).")
    maximum_humidity: float = Field(ge = 0, le = 100, description="Monthly average of daily maximum relative humidity (%).")

    atm_min_temp: float = Field(ge = -10, le = 55, description="Monthly average of daily minimum atmospheric temperature (°C).")
    atm_max_temp: float = Field(ge = -10, le = 55, description="Monthly average of daily maximum atmospheric temperature (°C).")

    soil_min_temp: float = Field(ge = -10, le = 55, description="Monthly average of daily minimum soil temperature (°C).")
    soil_max_temp: float = Field(ge = -10, le = 55, description="Monthly average of daily maximum soil temperature (°C).")
    
    wind_speed_kmph: float = Field(ge = 0, description="Monthly average wind speed measured in kilometers per hour (km/h).")

    is_outbreak_2019: bool = Field(description="Binary flag (True/False) indicating if the observation falls within the documented 2019 scrub typhus outbreak period.")
    is_covid_period: bool = Field(description="Binary flag (True/False) identifying data points during the COVID-19 pandemic, accounting for potential healthcare reporting bias.")

    @field_validator('maximum_humidity', "atm_max_temp", "soil_max_temp")
    @classmethod
    def validate_min_max(cls, v: float, info):
        """Greneric validator to ensure logical consistency (Max >= Min)."""
        field_map = {
            "maximum_humidity": "minimum_humidity",
            "atm_max_temp": "atm_min_temp",
            "soil_max_temp": "soil_min_temp"
        }
        min_filed = field_map.get(info.field_name)

        if min_filed in info.data and v < info.data[min_filed]:
            raise ValueError(f"{info.field_name} must be greater than or equal to {min_filed}.")
        return v

class TrainingObservation(ObservationBase):
    """Schema for historical (gold) training data."""
    time_stamp: date
    case_count: float = Field(ge=0)


class InferenceObservation(ObservationBase):
    """Schema for future exogenous inputs."""
    time_stamp: date