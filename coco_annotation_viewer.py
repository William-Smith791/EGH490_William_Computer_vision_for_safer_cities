#!/usr/bin/env python3
# python3 coco_annotation_viewer.py \
#  --images data/NHRA_Dataset/val \
# --annotations data/NHRA_Dataset/val_annotations.json \
# --port 7870
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
        description="View COCO bounding-box annotations in a Gradio app."
    )
    parser.add_argument("--images", required=True, type=Path, help="Image root directory")
    parser.add_argument(
        "--annotations", required=True, type=Path, help="COCO annotation JSON"
    )
    parser.add_argument("--port", type=int, default=7870, help="Gradio server port")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host, e.g. 0.0.0.0 for remote access"
    )
    parser.add_argument("--share", action="store_true", help="Create a Gradio share link")
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


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    supplied = Path(file_name).expanduser()
    candidates = []

    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.append(image_root / supplied)

    # Useful for JSON files that contain an obsolete directory prefix.
    candidates.append(image_root / supplied.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find '{file_name}' under image root '{image_root}'."
    )


def main() -> None:
    args = parse_args()
    image_root = args.images.expanduser().resolve()
    annotation_path = args.annotations.expanduser().resolve()

    if not image_root.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_root}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        coco: dict[str, Any] = json.load(file)

    images = coco.get("images", [])
    if not images:
        raise ValueError("The JSON contains no entries in its 'images' array.")

    category_names = {
        int(category["id"]): str(category.get("name", category["id"]))
        for category in coco.get("categories", [])
    }

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id is not None:
            annotations_by_image[int(image_id)].append(annotation)

    def render(index: int):
        index = max(0, min(int(index), len(images) - 1))
        image_record = images[index]
        image_id = int(image_record["id"])
        file_name = str(image_record["file_name"])
        annotations = annotations_by_image.get(image_id, [])

        try:
            image_path = resolve_image_path(image_root, file_name)
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            placeholder = Image.new("RGB", (1200, 700), "white")
            draw = ImageDraw.Draw(placeholder)
            font = load_font(28)
            draw.multiline_text(
                (30, 30),
                f"Unable to open image:\n{file_name}\n\n{error}",
                fill="black",
                font=font,
                spacing=10,
            )
            info = (
                f"### Image {index + 1} of {len(images)}\n"
                f"**File:** `{file_name}`  \n"
                f"**Error:** {error}"
            )
            return placeholder, info, index

        draw = ImageDraw.Draw(image)
        width, height = image.size
        line_width = max(3, round(min(width, height) / 300))
        font = load_font(max(16, round(min(width, height) / 45)))

        class_counts: dict[str, int] = defaultdict(int)

        for annotation in annotations:
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue

            x, y, box_width, box_height = map(float, bbox[:4])
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            x2 = max(0, min(width - 1, x + box_width))
            y2 = max(0, min(height - 1, y + box_height))

            category_id = int(annotation.get("category_id", -1))
            category_name = category_names.get(category_id, f"class {category_id}")
            class_counts[category_name] += 1
            colour = colour_for_category(category_id)

            draw.rectangle((x1, y1, x2, y2), outline=colour, width=line_width)

            annotation_id = annotation.get("id")
            label = category_name
            if annotation_id is not None:
                label += f"  [ann {annotation_id}]"

            text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            padding = max(3, line_width)

            label_x = x1
            label_y = y1 - text_height - 2 * padding
            if label_y < 0:
                label_y = y1

            label_right = min(width, label_x + text_width + 2 * padding)
            if label_right == width:
                label_x = max(0, width - text_width - 2 * padding)

            draw.rectangle(
                (
                    label_x,
                    label_y,
                    label_x + text_width + 2 * padding,
                    label_y + text_height + 2 * padding,
                ),
                fill=colour,
            )
            draw.text(
                (label_x + padding, label_y + padding),
                label,
                fill="white",
                font=font,
                stroke_width=1,
                stroke_fill="black",
            )

        counts_text = ", ".join(
            f"{name}: {count}" for name, count in sorted(class_counts.items())
        )
        if not counts_text:
            counts_text = "No bounding-box annotations"

        info = (
            f"### Image {index + 1} of {len(images)}\n"
            f"**File:** `{file_name}`  \n"
            f"**Image ID:** `{image_id}`  \n"
            f"**Annotations:** {len(annotations)}  \n"
            f"**Classes:** {counts_text}"
        )
        return image, info, index

    def previous(index: int):
        return render(int(index) - 1)

    def next_image(index: int):
        return render(int(index) + 1)

    css = """
    #viewer-image img { object-fit: contain !important; }
    #navigation-row { position: sticky; bottom: 0; z-index: 10; background: var(--body-background-fill); padding: 8px 0; }
    """

    with gr.Blocks(title="COCO Annotation Viewer", css=css) as demo:
        gr.Markdown("# COCO Annotation Viewer")
        info = gr.Markdown()
        viewer = gr.Image(
            label="Annotated image",
            type="pil",
            interactive=False,
            height=760,
            elem_id="viewer-image",
        )
        index_state = gr.State(0)

        with gr.Row(elem_id="navigation-row"):
            back_button = gr.Button("← Back", variant="secondary")
            next_button = gr.Button("Next →", variant="primary")

        back_button.click(
            fn=previous,
            inputs=index_state,
            outputs=[viewer, info, index_state],
        )
        next_button.click(
            fn=next_image,
            inputs=index_state,
            outputs=[viewer, info, index_state],
        )
        demo.load(
            fn=lambda: render(0),
            outputs=[viewer, info, index_state],
        )

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
