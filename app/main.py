#%%
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

# Need this to be able to write cache on HF Space
HF_HOME                         = ".cache/huggingface"
HF_MODULES_CACHE                = HF_HOME + "/modules"
os.environ["HF_HOME"]           = HF_HOME
os.environ["HF_MODULES_CACHE"]  = HF_MODULES_CACHE

import time
import gradio as gr
import spaces
import torch
from PIL import Image
from jinja2 import Environment, FileSystemLoader
from src.file_tools import suppress_stdout_stderr


from src.file_tools import list_files
from src.htr.pipelines.florence import FlorencePipeline
from src.data_types import Page
from src.logger import CustomLogger
from app.configs import css, theme

#%%
_ENV = Environment(loader=FileSystemLoader(PROJECT_DIR / "app/assets/jinja_templates"))
_IMAGE_TEMPLATE = _ENV.get_template("image")
_TRANSCRIPTION_TEMPLATE = _ENV.get_template("transcription")

GRADIO_CACHE_DIR    = ".gradio_cache"
EXAMPLES_DIR        = Path(__file__).parent / "assets/examples"
OUTPUT_CACHE_DIR    = GRADIO_CACHE_DIR + "/outputs"
BATCH_SIZE = 2

os.environ["GRADIO_CACHE_DIR"]  = GRADIO_CACHE_DIR

if not Path(OUTPUT_CACHE_DIR).exists():
    Path(OUTPUT_CACHE_DIR).mkdir(parents=True)


logger = CustomLogger(__name__)

# Helper functions
def render_image(image, image_path, lines):
    return _IMAGE_TEMPLATE.render(
        image=image,
        image_path=image_path,
        lines=lines,
    )


def render_transcription(page: Page):
    regions = page.regions
    return _TRANSCRIPTION_TEMPLATE.render(regions=regions)


def render_result(inputs: list[tuple[str, Page]]):
    """Use image and page data to render HTML"""
    # Currently only support displaying the last image processed
    image: str  = inputs[-1][0]
    page: Page  = inputs[-1][1]

    image_out = render_image(Image.open(image), page.path, page.lines)
    text_out = render_transcription(page)

    return image_out, text_out


def change_tab():
    """Navigate to output tab"""
    return gr.Tabs(selected=1)


def get_examples() -> list[[tuple[Image.Image, str]]]:
    """Get examples from assets folder"""
    img_paths = list_files(EXAMPLES_DIR, extensions=[".jpg", ".png", ".tif"])
    return [(Image.open(path), path.name) for path in img_paths]


def get_selected_example(event: gr.SelectData) -> list[str]:
    """Get path to the selected example image."""
    return [(event.value["image"]["path"], event.value["caption"])]


# Pipeline functions
@spaces.GPU()
def init_pipeline(device="cpu") -> FlorencePipeline:
    """Initiate the pipeline"""
    pipeline = FlorencePipeline(
        pipeline_type       = "line_od__ocr",
        line_od_model_path  = "nazounoryuu/florence_base__mixed__page__line_od",
        ocr_model_path      = "nazounoryuu/florence_base__mixed__line_bbox__ocr",
        batch_size          = BATCH_SIZE,
        device              = device,
        logger              = logger,
    )
    return pipeline


@spaces.GPU()
def run_htr_pipeline(
    pipeline: FlorencePipeline | None, 
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

    if images[-1][1] is None:
        image_name = Path(image_path).name
    else:
        image_name = images[-1][1]

    progress(0.0, desc="Starting up...")
    time.sleep(1)

    # progress(0.3, desc="Loading pipeline...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if pipeline is None is None:
        progress(0.3, desc="Initiating pipeline...")
        time.sleep(1)
        with suppress_stdout_stderr():
            pipeline = init_pipeline(device=DEVICE)

    # Cache result from previous run
    cache_path = Path(OUTPUT_CACHE_DIR) / Path(image_name).with_suffix(".json")

    print(use_cache)
    
    if use_cache and cache_path.exists():
        progress(0.5, desc="Cache found, loading cache...")
        time.sleep(1)
        page = Page.from_json(cache_path)
    else:
        progress(0.5, desc="Transcribing...")
        time.sleep(1)
        logger.info(f"Processing {image_name}")
        page = pipeline.run(Image.open(image_path).convert("RGB"))
    
    # Save cache
    # Need to update image path to path of the currently cached image to display later
    page.path = image_path
    page.to_json(cache_path)
    logger.info(f"Done, saved result to {cache_path}")
    
    progress(1.0, desc="Done")

    gr.Info("Transcribing done, redirecting to output tab...")
    time.sleep(1)

    new_outputs = outputs + [(image_path, page)]
    return new_outputs, pipeline


# Interfaces
# Submit tab
with gr.Blocks(title="submit") as submit:
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

# Output tab
with gr.Blocks(title="output") as output:
    with gr.Row():
        with gr.Column(scale=2):
            output_img = gr.HTML(
                label="Annotated image",
                padding=False,
                elem_classes="svg-image",
                container=True,
                max_height="80vh",
                min_height="80vh",
                show_label=True,
            )

        with gr.Column(scale=1):
            output_text = gr.HTML(
                label="Transcription",
                padding=False,
                elem_classes="transcription",
                container=True,
                max_height="65vh",
                min_height="65vh",
                show_label=True
            )

# Main
with gr.Blocks(
    title="HTR with VLM",
    css=css,
    theme=theme
) as demo:
    gr.Markdown("<h1>HTR with VLM</h1>", elem_classes="title-h1")
    gr.Markdown("""
    This handwritten text recognition pipeline uses Florence-2 fine-tuned for text line detection and OCR tasks. Steps in the pipeline:
        
    1. Detect text lines from the page image
    2. Perform text recognition on detected lines
                
    This space does not have access to GPU.
    Inference on CPU will be extremely slow, so I cached example results to disk. Some notes:
    
    - To view example outputs, select one image from the examples, and choose  **Used cached result: True**.
        To transcribe an example from scratch, choose **False**.
    - New images uploaded will be transcribed from scratch.
    """)

    # Setup output collection
    outputs = gr.State([])

    # Setup state object to store pipeline
    pipeline = gr.State(None)
    
    # Tabs
    with gr.Tabs() as tabs:

        with gr.Tab(label="Input", id=0) as input_tab:
            submit.render()

        with gr.Tab("Output", id=1) as output_tab:
            output.render()

    # Events
    # If selected an example, push it to input_images gallery
    examples.select(get_selected_example, None, input_images)

    # If click run, run the pipeline
    run_btn.click(
        fn=run_htr_pipeline,
        inputs=[pipeline, input_images, outputs, use_cache],
        outputs=[outputs, pipeline],
        show_progress="full",
        show_progress_on=[input_images]
    )

    # When new result arrive, auto-navigate to output tab, and render result
    outputs.change(change_tab, [], tabs)
    outputs.change(render_result, inputs=outputs, outputs=[output_img, output_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, enable_monitoring=False, show_api=False)