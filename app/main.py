#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

import os
import time
import gradio as gr
import spaces
from PIL import Image
from jinja2 import Environment, FileSystemLoader


from src.file_tools import list_files
from src.htr.pipelines.florence import FlorencePipeline
from src.data_types import Page
from src.logger import CustomLogger
from app.configs import css, theme

#%%
_ENV = Environment(loader=FileSystemLoader(PROJECT_DIR / "app/assets/jinja_templates"))
_IMAGE_TEMPLATE = _ENV.get_template("image")
_TRANSCRIPTION_TEMPLATE = _ENV.get_template("transcription")

HF_HOME             = "/home/user/huggingface"
HF_MODULES_CACHE    = HF_HOME + "/modules"
GRADIO_CACHE        = ".gradio"
OUTPUT_CACHE_DIR    = GRADIO_CACHE + "/outputs"
EXAMPLES_DIR        = Path(__file__).parent / "assets/examples"
BATCH_SIZE = 2

if not Path(OUTPUT_CACHE_DIR).exists():
    Path(OUTPUT_CACHE_DIR).mkdir(parents=True)

os.environ["HF_HOME"] = "/home/user/huggingface"
os.environ["GRADIO_CACHE_DIR"] = str(GRADIO_CACHE)

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


def get_examples() -> list[Image.Image]:
    """Get examples from assets folder"""
    img_paths = list_files(EXAMPLES_DIR, extensions=[".jpg", ".png", ".tif"])
    return [Image.open(path) for path in img_paths]


def get_selected_example(event: gr.SelectData) -> list[str]:
    """Get path to the selected example image."""
    return [event.value["image"]["path"]]


# Pipeline functions
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
    device: str, 
    progress=gr.Progress(track_tqdm=True)
):
    """Run HTR Pipeline, with progress bar"""

    # Currently support only one image
    # TODO: iterate through images and run the pipeline
    image_path = images[-1][0]

    progress(0.0, desc="Starting up...")
    time.sleep(1)

    if pipeline is None:
        progress(0.1, desc="Initiate pipeline...")
        pipeline = init_pipeline(device=device)
    
    progress(0.3, desc="Running pipeline...")

    # Cache result from previous run
    use_cache = True
    cache_path = Path(OUTPUT_CACHE_DIR) / Path(image_path).name
    
    if use_cache and cache_path.exists():
        page_data = Page.from_json(cache_path)
    else:
        page_data = pipeline.run(Image.open(image_path).convert("RGB"))
        page_data.path = image_path  # Need to update file path to display the image later

    # Save to cache
    page_data.to_json(cache_path)

    progress(1.0, desc="Done")

    gr.Info("Transcribing done, redirecting to output tab...")
    time.sleep(1)

    new_outputs = outputs + [(image_path, page_data)]
    return new_outputs


# Interfaces
# Submit tab
with gr.Blocks(title="submit") as submit:
    with gr.Row():

        input_images = gr.Gallery(
            file_types=["image"],
            label="Input images",
            interactive=True,
            object_fit="scale-down"
        )
        
        examples = gr.Gallery(
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

    with gr.Row():
        device = gr.Dropdown(choices=["cpu", "cuda"], label="Device", value="cpu", interactive=True)
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

    # States
    pipeline = gr.State(None)
    outputs = gr.State([])

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
        inputs=[pipeline, input_images, outputs, device],
        outputs=[outputs],
        show_progress="full",
        show_progress_on=[input_images]
    )

    # When new result arrive, auto-navigate to output tab, and render result
    outputs.change(change_tab, [], tabs)
    outputs.change(render_result, inputs=outputs, outputs=[output_img, output_text])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, enable_monitoring=False, show_api=False)