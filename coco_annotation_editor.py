"""
To open the editor window execute the following command in the terminal with the python venv
venv setup: install gradio with pip install gradio activate venv with source venv_folder_name/bin/activate
Now you can execute the following terminal command (on Linux equivilent OS).

python3 coco_annotation_editor.py \
--images data/NHRA_Dataset/val \
--annotations data/NHRA_Dataset/val_annotations.json \
--port 7870

Once executed copy and past URL into web browser to access UI
"""
import argparse
import colorsys
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Building Dataset Cleaner"
    )
    parser.add_argument("--images",required=True, type=Path, help="Directory containing the dataset images",)
    return parser.parse_args()
""" Data loading functions """
def get_first_image(image_dir: Path) -> Image.Image:
    """Load the first JPEG image in the dataset directory."""

    image_files = sorted(
        list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.jpg"))
    )

    if not image_files:
        raise FileNotFoundError(
            f"No JPEG images found in {image_dir}"
        )

    return Image.open(image_files[0]).convert("RGB")
""" UI setup functions"""
def create_header():
    """Create the application header."""
    gr.Markdown(
        """
        # Building Dataset Cleaner
        Review and clean images and their COCO-style annotations.
        """
    )


def create_dataset_panel():
    """Create the dataset selection/control panel."""

    with gr.Column():
        gr.Markdown("### Dataset")

        dataset_path = gr.Textbox(
            label="Dataset directory",
            placeholder="/path/to/dataset",
        )

        load_button = gr.Button(
            "Load Dataset",
            variant="primary"
        )

        image_counter = gr.Markdown(
            "Image: 0 / 0"
        )

        with gr.Row():
            previous_button = gr.Button("← Previous")
            next_button = gr.Button("Next →")

        gr.Markdown("### Display")

        show_annotations = gr.Checkbox(
            label="Show bounding boxes",
            value=True
        )

        show_labels = gr.Checkbox(
            label="Show labels",
            value=True
        )

        return {
            "dataset_path": dataset_path,
            "load_button": load_button,
            "image_counter": image_counter,
            "previous_button": previous_button,
            "next_button": next_button,
            "show_annotations": show_annotations,
            "show_labels": show_labels,
        }


def create_image_panel():
    """Create the image display panel."""

    with gr.Column():
        gr.Markdown("### Image")

        image_display = gr.Image(
            label="Current Image",
            type="pil",
            height=600,
        )

        return {
            "image_display": image_display,
        }


def create_annotation_panel():
    """Create the annotation editing panel."""

    with gr.Column():
        gr.Markdown("### Annotation Actions")

        with gr.Row():
            accept_button = gr.Button(
                "✓ Accept",
                variant="primary"
            )

            reject_button = gr.Button(
                "✗ Reject",
                variant="stop"
            )

        with gr.Row():
            delete_button = gr.Button(
                "Delete Annotation"
            )

            reset_button = gr.Button(
                "Reset Changes"
            )

        annotation_info = gr.JSON(
            label="Current Annotation",
            value={}
        )

        return {
            "accept_button": accept_button,
            "reject_button": reject_button,
            "delete_button": delete_button,
            "reset_button": reset_button,
            "annotation_info": annotation_info,
        }


def create_status_bar():
    """Create the status/output area."""

    status = gr.Markdown(
        "Status: Ready"
    )

    return status


def create_app():
    """Construct the complete Gradio application."""

    args = parse_args()

    image_dir = args.images.expanduser().resolve()

    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"Image directory does not exist: {image_dir}"
        )

    first_image = get_first_image(image_dir)

    with gr.Blocks(
        title="Building Dataset Cleaner"
    ) as app:

        create_header()

        with gr.Row():

            # Left control panel
            with gr.Column(scale=1, min_width=250):
                dataset_panel = create_dataset_panel()

            # Main image viewer
            with gr.Column(scale=3):
                image_panel = create_image_panel()

            # Right annotation panel
            with gr.Column(scale=1, min_width=250):
                annotation_panel = create_annotation_panel()

        gr.Markdown("---")

        create_status_bar()

        # Load the first image
        image_panel["image_display"].value = first_image
        app.load(
            fn=lambda: get_first_image(image_dir),
            outputs=image_panel["image_display"],
        )
    app.launch()


if __name__ == "__main__":
    create_app()