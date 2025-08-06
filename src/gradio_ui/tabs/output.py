
import gradio as gr
from PIL import Image
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from vlm.data_types import Page


PROJECT_DIR            = Path(__file__).parent.parent.parent.parent
TEMPLATES_DIR          = PROJECT_DIR / "src/gradio_ui/assets/jinja_templates"
ENV                    = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
IMAGE_TEMPLATE         = ENV.get_template("image")
TRANSCRIPTION_TEMPLATE = ENV.get_template("transcription")


# Helper functions
def render_image(image, image_path, lines):
    return IMAGE_TEMPLATE.render(
        image=image,
        image_path=image_path,
        lines=lines,
    )


def render_transcription(page: Page):
    regions = page.regions
    return TRANSCRIPTION_TEMPLATE.render(regions=regions)


def render_result(inputs: list[tuple[str, Page]]):
    """Use image and page data to render HTML"""
    # Currently only support displaying the last image processed
    image: str  = inputs[-1][0]
    page: Page  = inputs[-1][1]

    image_out = render_image(Image.open(image), page.path, page.lines)
    text_out = render_transcription(page)

    with open(PROJECT_DIR / "temp/image_out.html", "w") as f:
        f.write(image_out)

    return image_out, text_out


with gr.Blocks(title="output") as output_block:
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