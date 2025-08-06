import requests
import json
import time
import spaces
import os

import gradio as gr
from pathlib import Path
from urllib.parse import urljoin
from dotenv import load_dotenv

from PIL import Image
from PIL.Image import Image as PILImage
from vlm.utils.file_tools import list_files
from vlm.data_types import Page
from vlm.utils.logger import CustomLogger
from utils.schemas import PredictionInput, Pipeline

PROJECT_DIR         = Path(__file__).parent.parent.parent.parent
load_dotenv(PROJECT_DIR / ".env")

GRADIO_CACHE_DIR    = PROJECT_DIR / ".gradio_cache"
OUTPUT_CACHE_DIR    = GRADIO_CACHE_DIR / "outputs"

EXAMPLES_DIR        = Path(__file__).parent.parent / "assets/examples"
BACKEND_APP_URL     = os.environ.get("BACKEND_APP_URL")
PREDICT_ENDPOINT    = urljoin(BACKEND_APP_URL, "/predict")

os.environ["GRADIO_CACHE_DIR"]  = str(GRADIO_CACHE_DIR)


def get_examples() -> list[[tuple[PILImage, str]]]:
    """Get examples from assets folder"""
    img_paths = list_files(EXAMPLES_DIR, extensions=[".jpg", ".png", ".tif"])
    return [(Image.open(path), path.name) for path in img_paths]


def get_selected_example(event: gr.SelectData) -> list[str]:
    """Get path to the selected example image."""
    return [(event.value["image"]["path"], event.value["caption"])]


logger = CustomLogger("vlm-htr-app")


@spaces.GPU()
def run_htr_pipeline(
    # pipeline: FlorencePipeline | None, 
    images: list[tuple], 
    outputs: list, 
    progress=gr.Progress(track_tqdm=True),
    use_cache: bool = True
):
    """Run HTR Pipeline, with progress bar"""

    # Currently support only one image
    # TODO: iterate through images and run the pipeline
    assert images is not None, "Please select at least one image"
    image_path = images[-1][0]

    # Get image name
    if images[-1][1] is None:
        image_name = Path(image_path).name
    else:
        image_name = images[-1][1]

    progress(0.0, desc="Starting up...")
    time.sleep(1)

    # Cache result from previous run
    cache_path = Path(OUTPUT_CACHE_DIR) / Path(image_name).with_suffix(".json")

    if use_cache and cache_path.exists():
        progress(0.5, desc="Cache found, loading cache...")
        time.sleep(1)
        page = Page.from_json(str(cache_path))
    else:
        progress(0.5, desc="Transcribing...")
        time.sleep(1)
        logger.info(f"Processing {image_name}")
        content = PredictionInput(pipeline=Pipeline.FlorencePipeline, image_path=image_path)
        response = requests.post(PREDICT_ENDPOINT, json=content.model_dump())
        
        if response.status_code == 200:
            page = Page.from_dict(json.loads(response.text)["output"])
        else: 
            response.raise_for_status()
        
        # Need to update image path to path of the currently cached image to display later
        page.path = image_path
        page.to_json(cache_path)
        logger.info(f"Done, saved result to {cache_path}")
        
        progress(1.0, desc="Done")

        gr.Info("Transcribing done, redirecting to output tab...")
        time.sleep(1)

    new_outputs = outputs + [(image_path, page)]
    
    return new_outputs


with gr.Blocks(title="submit") as submit_block:
    with gr.Row():

        input_images = gr.Gallery(
            label="Input images",
            file_types=["image"],
            interactive=True,
            object_fit="scale-down"
        )
        
        with gr.Column():
            examples = gr.Gallery(
                label="Examples",
                value=get_examples(),
                show_label=False,
                interactive=False,
                allow_preview=False,
                object_fit="scale-down",
                min_width=250,
                height="100%",
                columns=4,
                container=False,
            )
            # device = gr.Dropdown(choices=["cpu", "cuda"], label="Device", value="cpu", interactive=True)
            use_cache = gr.Radio([True, False], label="Use cached result", value=True, interactive=True)
            run_btn = gr.Button("Transcribe")