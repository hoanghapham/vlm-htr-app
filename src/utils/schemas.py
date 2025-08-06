from enum import Enum
from pydantic import BaseModel


class Pipeline(Enum):
    FlorencePipeline = "FlorencePipeline"
    TraditionalPipeline = "TraditionalPipeline"


class PredictionInput(BaseModel):
    pipeline: Pipeline
    image_path: str

    class Config:
        use_enum_values = True


class PredictionOutput(BaseModel):
    output: dict | None