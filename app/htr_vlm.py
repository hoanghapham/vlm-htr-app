#%%
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.append(str(PROJECT_DIR))

import io
import os
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
from configs.gradio_configs import css, theme

from jinja2 import Environment, FileSystemLoader

_ENV = Environment(loader=FileSystemLoader(PROJECT_DIR / "app/assets/jinja_templates"))
_IMAGE_TEMPLATE = _ENV.get_template("image")
_TRANSCRIPTION_TEMPLATE = _ENV.get_template("transcription")

GRADIO_CACHE = ".gradio_cache"
EXAMPLES_DIRECTORY = os.path.join(GRADIO_CACHE, "examples")

logger = CustomLogger(__name__)

if os.environ.get("GRADIO_CACHE_DIR", GRADIO_CACHE) != GRADIO_CACHE:
    os.environ["GRADIO_CACHE_DIR"] = GRADIO_CACHE
    logger.warning("Setting GRADIO_CACHE_DIR to '%s' (overriding a previous value).")


# if args.device == "cuda":
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# else:
#     DEVICE = args.device

DEVICE = "cpu"
BATCH_SIZE = 2

def init_pipeline(progress=gr.Progress(track_tqdm=True)):

    # logger.info("Initiate HTR pipeline...")
    progress(0.1, desc="Initiate FlorencePipeline...")

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

def render_image(image, image_path, lines):
    html = _IMAGE_TEMPLATE.render(
        image=image,
        image_path=image_path,
        lines=lines,
    )
    with open("temp/image_display.html", "w") as f:
        f.write(html)

    return html


def render_transcription(page: Page):
    regions = page.regions
    return _TRANSCRIPTION_TEMPLATE.render(regions=regions)


def draw_result(image: PILImage, page_data: Page):

    # Draw chart, save image to buffer, then return from buffer
    line_bboxes = [line.bbox for line in page_data.lines]
    fig, ax = draw_bboxes_xyxy(image, line_bboxes)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    return Image.open(buffer)



def xml_to_page(xml_file: Any):
    parser = XMLParser()
    root = parser._parse_xml(xml_file)
    xml_regions = parser.get_regions(root)
    xml_lines = parser.get_lines(root)

    page_lines = [Line(bbox=data["bbox"], polygon=data["polygon"], text=data["transcription"]) for data in xml_lines]

    regions = []
    for xml_region in xml_regions:
        region_line = xml_region["lines"]
        lines = [Line(bbox=data["bbox"], polygon=data["polygon"], text=data["transcription"]) for data in region_line]
        regions.append(Region(bbox=xml_region["bbox"], polygon=xml_region["polygon"], lines=lines))
    
    img_path = Path(__file__).parent / "assets" / "examples" / (Path(xml_file.name).stem + ".jpg")
    page_path = str(img_path.relative_to(Path(__file__).parent.parent))
    page = Page(regions=regions, lines=page_lines, path=page_path)
    print("page path: ", page.path)

    return page


# def display_annotation(image: gr.Image, xml_file: gr.File):
#     page = xml_to_page(xml_file)
#     # print("Page path:", page.path)

#     # fig, ax = draw_page_line_segments(Image.fromarray(image), page.regions)
#     # buffer = io.BytesIO()
#     # fig.savefig(buffer, format='png')
#     # buffer.seek(0)
#     # image = Image.open(buffer)
    
#     # Text
#     # text_out = "\n".join([line.text for line in page.lines])
#     image_out = render_image(Image.fromarray(image), page.path, page.lines)
#     text_out = render_transcription(page)

    # return image_out, text_out

def display_result(inputs: list[(Path | str, Page)]):
    image = inputs[0][0]
    page = inputs[0][1]

    image_out = render_image(Image.open(image), page.path, page.lines)
    text_out = render_transcription(page)

    with open("temp/image_display.html", "w") as f:
        f.write(image_out)


    return image_out, text_out


def run_htr_pipeline(pipeline: FlorencePipeline | None, images, outputs: list, progress=gr.Progress(track_tqdm=True)):
    if pipeline is None:
        pipeline = init_pipeline()
    
    progress(0.5, desc="Transcribing...")
    
    use_cache = True
    cache_path = Path("temp/page.json")
    image = images[0][0]
    print(image)

    
    if use_cache and cache_path.exists():
        page_data = Page.from_json("temp/page.json")
    else:
    # For now only support 1 image
        page_data = pipeline.run(Image.open(image))
    
    page_data.path = image

    page_data.to_json("temp/page.json")

    progress(1.0, desc="Done")

    outputs = outputs + [(image, page_data)]
    return outputs



def change_tab():
    return gr.Tabs(selected=1)


def get_examples():
    img_paths = list_files(Path(__file__).parent / "assets" / "examples", extensions=[".jpg", ".png", ".tif"])

    return [Image.open(path) for path in img_paths]
    # return [(Image.open(path), str(path)) for path in img_paths]


def get_selected_example_image(event: gr.SelectData) -> str:
    """
    Get path to the selected example image.
    """
    return [event.value["image"]["path"]]

with gr.Blocks(
    title="HTR with VLM",
    css=css,
    theme=theme
) as demo:

    gr.Markdown("<h1>HTR with VLM</h1>", elem_classes="title-h1")

    pipeline = gr.State(None)
    outputs = gr.State([])

    with gr.Tabs() as tabs:

        # Input
        with gr.Tab(label="Input", id=0) as input_tab:
            with gr.Row():
                # input_image = gr.Image(label="Input image")
                # Example, need to input xml along with image
                # with gr.Column():
                    # input_image = gr.Image(label="Input image")
                    # input_xml = gr.File(label="XML File")

                # input_images = gr.File(label="Input image", file_types=["image"])

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

                # Can support multiple files using gr.Gallery()
                

                # examples = gr.Examples(
                #     examples=lambda tup: [tup[1] for tup in get_examples()],
                #     # fn=display_annotation,
                #     inputs=[input_images],  # Pass data to this input component
                #     cache_examples=False,
                #     # cache_mode="lazy",
                #     run_on_click=False,
                # )

            with gr.Row():
                device = gr.Dropdown(choices=["cpu", "cuda"], label="Device", value="cpu", interactive=True)
                run_btn = gr.Button("Transcribe")

        # Output
        with gr.Tab("Output", id=1) as output_tab:
            with gr.Row():
                # output_img = gr.Image(label="Output image")
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
    
    # gr.HTML("<img src='/gradio_api/file=app/assets/examples/Bergskollegium__Relationer_och_skrivelser__E3_5__1691-1692__40004026_00008.jpg'>")
    # Event listeners

    # If selected an example, push it to input_images gallery
    examples.select(get_selected_example_image, None, input_images)

    # If click run, run the pipeline
    run_btn.click(
        fn=run_htr_pipeline,
        inputs=[pipeline, input_images, outputs],
        outputs=[outputs],
    )
    # outputs.change(change_tab, [], tabs)
    run_btn.click(change_tab, [], tabs)
    outputs.change(display_result, inputs=outputs, outputs=[output_img, output_text])


allowed_path = Path(__file__).parent / "assets/examples"
demo.launch()