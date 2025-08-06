#%%
import os
import uvicorn
import gradio as gr
from pathlib import Path
from urllib.parse import urljoin

from gradio_ui.configs import css, theme
from gradio_ui.tabs.submit import (
    submit_block, 
    input_images, 
    examples, 
    use_cache, 
    run_btn, 
    get_selected_example,
    run_htr_pipeline
)
from gradio_ui.tabs.output import (
    output_block, 
    output_img, 
    output_text, 
    render_result
)
from backend.main import app
from dotenv import load_dotenv


# Set paths
PROJECT_DIR                     = Path(__file__).parent.parent.parent
GRADIO_CACHE_DIR                = PROJECT_DIR / ".gradio_cache"
HF_HOME                         = ".cache/huggingface"
HF_MODULES_CACHE                = HF_HOME + "/modules"

EXAMPLES_DIR                    = Path(__file__).parent / "assets/examples"
OUTPUT_CACHE_DIR                = GRADIO_CACHE_DIR / "outputs"

os.environ["GRADIO_CACHE_DIR"]  = str(GRADIO_CACHE_DIR)
os.environ["HF_HOME"]           = HF_HOME
os.environ["HF_MODULES_CACHE"]  = HF_MODULES_CACHE

load_dotenv(PROJECT_DIR / ".env")

if not Path(OUTPUT_CACHE_DIR).exists():
    Path(OUTPUT_CACHE_DIR).mkdir(parents=True)


def change_tab():
    """Navigate to output tab"""
    return gr.Tabs(selected=1)


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
    outputs_collection = gr.State([])
    
    # Tabs
    with gr.Tabs() as tabs:

        with gr.Tab(label="Input", id=0) as input_tab:
            submit_block.render()

        with gr.Tab("Output", id=1) as output_tab:
            output_block.render()

    # Events
    # If selected an example, push it to input_images gallery
    examples.select(get_selected_example, None, input_images)

    # If click run, run the pipeline
    run_btn.click(
        fn=run_htr_pipeline,
        inputs=[input_images, outputs_collection, use_cache],
        outputs=[outputs_collection],
        show_progress="full",
        show_progress_on=[input_images]
    )

    # When new result arrive, auto-navigate to output tab, and render result
    outputs_collection.change(change_tab, [], tabs)
    outputs_collection.change(render_result, inputs=outputs_collection, outputs=[output_img, output_text])


# Setup app
BACKEND_APP_URL                 = os.environ.get("BACKEND_APP_URL")
GRADIO_APP_PATH                 = "/gradio"
GRADIO_APP_URL                  = urljoin(BACKEND_APP_URL, GRADIO_APP_PATH)

# Mount gradio app on top of the fastapi app
app = gr.mount_gradio_app(app, demo, path=GRADIO_APP_PATH, root_path=GRADIO_APP_PATH)


if __name__ == "__main__":
    # demo.launch(server_name="0.0.0.0", server_port=7860, enable_monitoring=False, show_api=False)
    print(f"\nTo view the app, navigate to {GRADIO_APP_URL}\n")
    uvicorn.run("gradio_ui.main:app", host="0.0.0.0", port=8000)