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
    parser.add_argument("--images",required=True, type=Path, help="Directory containing the dataset images")
    parser.add_argument("--annotations",required=True,type=Path,help="COCO annotation JSON file")
    return parser.parse_args()

def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()

def colour_for_category(category_id: int) -> tuple[int, int, int]:
    # Golden-ratio hue spacing gives stable, visually distinct colours.
    hue = (category_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)

""" Data loading functions """
def get_image_files(image_dir: Path) -> list[Path]:
    """Get all JPEG images in the dataset directory."""
    image_files = sorted(
        list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.jpg"))
    )
    if not image_files:
        raise FileNotFoundError(
            f"No JPEG images found in {image_dir}"
        )
    return image_files

def load_coco_annotations(annotation_path: Path) -> tuple[dict[int,dict[str,Any]], dict[int,list[str,Any]], dict[int, str]]:
    with annotation_path.open("r",encoding="utf-8") as file:
        coco: dict[str, Any] = json.load(file)
    images = {
        int(image["id"]): image
        for image in coco.get("images", [])
    }
    annotations_by_image = defaultdict(list)
    for annotation in coco.get("annotations", []):

        image_id = annotation.get("image_id")

        if image_id is not None:
            annotations_by_image[int(image_id)].append(
                annotation
            )
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
    }
    return images, annotations_by_image, categories

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
            "image_counter": image_counter,
            "previous_button": previous_button,
            "next_button": next_button,
            "show_annotations": show_annotations,
            "show_labels": show_labels,
        }


def create_image_panel():
    """Create the image display panel."""
    with gr.Column():
        gr.Markdown("### COCO Annotated Image")
        image_display = gr.Image(show_label=False,type="pil",height=640)
        return {"image_display": image_display}


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
    annotation_path = args.annotations.expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(
            f"Image directory does not exist: {image_dir}"
        )
    if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")

    image_files = get_image_files(image_dir)
    (coco_images,annotations_by_image,categories) = load_coco_annotations(annotation_path)
    coco_by_filename = {str(image["file_name"]): image
                        for image in coco_images.values()}
    def load_image(index: int):
        image_path = image_files[index]
        image = Image.open(image_path).convert("RGB")
        image_record = coco_by_filename.get(image_path.name)

        if image_record is None:
            return image

        image_id = int(image_record["id"])

        annotations = annotations_by_image.get(
            image_id,
            []
        )

        draw = ImageDraw.Draw(image)
        font = load_font(max(16, round(min(image.width, image.height) / 45)))
        for annotation in annotations:
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x, y, width, height = map(float,bbox[:4])
            category_id = int(annotation["category_id"])
            colour = colour_for_category(category_id)
            draw.rectangle((x,y,x + width,y + height,),outline=colour,width=3)
            label = categories.get(category_id,f"Unknown ({category_id})")
            text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            padding = 3
            draw.rectangle((x,y,x + text_width + 2 * padding,y + text_height + 2 * padding,),fill=colour)
            draw.text((x+padding, y+padding),label,fill="white", font=font, stroke_width=1, stroke_fill='black')
        return image
      
    """ Image cycling functions without wrap around to avoid confusion """
    def next_image(index: int):
        index = min(index + 1, len(image_files) - 1)
        return load_image(index), index


    def previous_image(index: int):
        index = max(index - 1, 0)
        return load_image(index), index

    with gr.Blocks(
        title="Building Dataset Cleaner"
    ) as app:
        image_index = gr.State(0) 
        create_header()

        with gr.Row():

            # Left control panel
            with gr.Column(scale=1, min_width=150):
                dataset_panel = create_dataset_panel()

            # Main image viewer
            with gr.Column(scale=5):
                image_panel = create_image_panel()

            # Right annotation panel
            with gr.Column(scale=1, min_width=150):
                annotation_panel = create_annotation_panel()

        gr.Markdown("---")

        create_status_bar()
        app.load(
            fn=lambda: load_image(0),
            outputs=image_panel["image_display"],
        )
        dataset_panel["next_button"].click(
            fn=next_image, inputs=image_index,
            outputs = [image_panel["image_display"],image_index])
        dataset_panel["previous_button"].click(
            fn=previous_image, inputs=image_index,
            outputs = [image_panel["image_display"],image_index])
    app.launch()


if __name__ == "__main__":
    create_app()