#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

import io

import gradio as gr
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Any

from src.visualization import draw_page_line_segments, draw_bboxes_xyxy
from src.data_processing.utils import XMLParser
from src.file_tools import list_files

from src.data_processing.visual_tasks import IMAGE_EXTENSIONS
from src.htr.pipelines.florence import FlorencePipeline
from src.data_types import Page, Region, Line
from src.logger import CustomLogger

logger = CustomLogger(__name__)

# if args.device == "cuda":
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
#     DEVICE = args.device

DEVICE = "cpu"
BATCH_SIZE = 2

def init_pipeline():

    logger.info("Initiate HTR pipeline...")

    pipeline = FlorencePipeline(
        pipeline_type       = "line_od__ocr",
        line_od_model_path  = PROJECT_DIR / "models/florence_based__mixed__page__line_od",
        ocr_model_path      = PROJECT_DIR / "models/florence_based__mixed__line_bbox__ocr",
        batch_size          = BATCH_SIZE,
        device              = DEVICE,
        logger              = logger
    )
    return pipeline

#%%

def draw_result(image: PILImage, page_data: Page):

    # Draw chart, save image to buffer, then return from buffer
    line_bboxes = [line.bbox for line in page_data.lines]
    fig, ax = draw_bboxes_xyxy(image, line_bboxes)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    return Image.open(buffer)


def transcribe(image: PILImage):
    pipeline = init_pipeline()
    image = Image.fromarray(image)
    page_data = pipeline.run(image)
    output_image = draw_result(image, page_data)
    output_text = page_data.text

    return output_image, output_text


def display_annotation(image: PILImage, xml_file: Any):
    parser = XMLParser()
    content = parser._parse_xml(xml_file)

    # Display image with bbox
    regions = parser.get_regions(content)
    fig, ax = draw_page_line_segments(Image.fromarray(image), regions)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Text
    lines_lst = []
    for region in regions:
        for line in region["lines"]:
            lines_lst.append(line["transcription"])
    
    text = "\n".join(lines_lst)

    return Image.open(buffer), text


def get_examples():
    img_paths = list_files(PROJECT_DIR / "examples", extensions=[".jpg", ".png", ".tif"])
    xml_paths = list_files(PROJECT_DIR / "examples", extensions=[".xml"])

    results = []

    for img_path, xml_path in zip(img_paths, xml_paths):
        image = Image.open(img_path)
        results.append([image, str(xml_path)])

    return results



def change_tab():
    return gr.Tabs(selected=1)

with gr.Blocks() as demo:
    gr.Markdown("# HTR with VLM - Demo")

    pipeline_container = gr.State([])

    with gr.Tabs() as tabs:

        # Input
        with gr.Tab(label="Input", id=0) as input_tab:
            with gr.Row():
                # input_image = gr.Image(label="Input image")
                with gr.Column():
                    input_image = gr.Image(label="Input image")
                    input_xml = gr.File(label="XML File")

                examples = gr.Examples(
                    examples=get_examples(),
                    fn=display_annotation,
                    inputs=[input_image, input_xml],
                    cache_examples=False,
                    # cache_mode="lazy",
                    run_on_click=False,
                )

            with gr.Row():
                device = gr.Dropdown(choices=["cpu", "cuda"], label="Device", value="cpu", interactive=True)
                run_btn = gr.Button("Transcribe")

        # Output
        with gr.Tab("Output", id=1) as output_tab:
            with gr.Row():
                output_img = gr.Image(label="Output image")
                output_text = gr.Textbox(label="Transcription")

    # Main flow
    run_btn.click(
        fn=display_annotation,
        inputs=[input_image, input_xml],
        outputs=[output_img, output_text],
    )
    run_btn.click(change_tab, [], tabs)

demo.launch(allowed_paths=[str(PROJECT_DIR / "examples")])