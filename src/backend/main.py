import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import ValidationError

from vlm.htr.pipelines.florence import FlorencePipeline
# from vlm.htr.pipelines.traditional import TraditionalPipeline
from utils.schemas import PredictionInput, PredictionOutput

app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = FlorencePipeline(
    pipeline_type       = "line_od__ocr",
    line_od_model_path  = "nazounoryuu/florence_base__mixed__page__line_od",
    ocr_model_path      = "nazounoryuu/florence_base__mixed__line_bbox__ocr",
    batch_size          = 2,
    device              = DEVICE,
)


@app.get("/")
def get_root():
    return {"app_name": "HTR with VLM"}


@app.post("/predict", response_model=PredictionOutput)
def predict(content: dict):
    try:
        pred_input = PredictionInput(**content)
        result_page = pipeline.run(Image.open(pred_input.image_path))
        return PredictionOutput(output=result_page.dict)
    except ValidationError as e:
        print(e)
        return PredictionOutput(output=None)
    

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000)