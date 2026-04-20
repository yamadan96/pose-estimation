"""Gradio web application for ViTPose human pose estimation."""

import logging
import os

import gradio as gr
from PIL import Image

from src.predictor import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

predictor = Predictor()


def estimate_pose(image: Image.Image | None) -> Image.Image | None:
    """Run pose estimation on the input image.

    Args:
        image: Input PIL image from Gradio upload.

    Returns:
        PIL image with skeleton overlay, or None if no image provided.
    """
    if image is None:
        return None

    if not predictor._initialized:
        predictor.initialize(device="cuda")

    return predictor.predict(image)


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks application.

    Returns:
        Configured Gradio Blocks instance.
    """
    with gr.Blocks(
        title="ViTPose - Human Pose Estimation",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # ViTPose - Human Pose Estimation
            Upload an image to detect human poses and visualize the skeleton overlay.

            **Model**: [ViTPose-base-simple](https://huggingface.co/usyd-community/vitpose-base-simple)
            (Xu et al., 2022) with COCO 17 keypoints.
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                submit_button = gr.Button("Estimate Pose", variant="primary")

            with gr.Column():
                output_image = gr.Image(
                    label="Pose Estimation Result",
                    type="pil",
                )

        submit_button.click(
            fn=estimate_pose,
            inputs=[input_image],
            outputs=[output_image],
        )

        input_image.change(
            fn=estimate_pose,
            inputs=[input_image],
            outputs=[output_image],
        )

        gr.Markdown(
            """
            ---
            **Note**: This demo uses the full image as a bounding box for
            single-person pose estimation. For multi-person scenarios,
            an object detector (e.g., RT-DETR) would be used upstream.
            """
        )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app = build_app()
    app.launch(server_port=port)
