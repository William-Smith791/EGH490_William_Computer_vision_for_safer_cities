"""
To open the editor window execute the following command in the terminal with the python venv
venv setup: install gradio with pip install gradio activate venv with source venv_folder_name/bin/activate
Now you can execute the following terminal command (on Linux equivilent OS).

python3 coco_annotation_editor.py \
--images data/NHRA_Dataset/val \
--annotations data/NHRA_Dataset/val_annotations.json \
--change_log change_log.json
--port 7870

testing command
python3 coco_annotation_editor.py \
--images data/NHRA_Dataset/val \
--annotations data/NHRA_Dataset/val_annotations_test.json \
--change_log change_log.json
Once executed copy and past URL into web browser to access UI
To alter labels click the dropdown menu located at the bottom of the right panel then select the desired annotation
make sure to click apply to save the changes to the JSON this is NOT done automatically
"""
import argparse
import colorsys
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import base64
from io import BytesIO
import inspect
import html

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
"""Indexing constants"""
ADDED_ANNOTATIONS = 0
REMOVED_ANNOTATIONS = 1
MOVED_ANNOTATIONS = 2
CHANGED_LABELS = 3
""" Miscellaneous functions"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Building Dataset Cleaner"
    )
    parser.add_argument("--images",required=True, type=Path, help="Directory containing the dataset images")
    parser.add_argument("--annotations",required=True,type=Path,help="COCO annotation JSON file")
    parser.add_argument("--change_log",required=True,type=Path,help="automatic change log")
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
        image_display = gr.HTML(height=640,elem_id="interactive-image",js_on_load="""
    element.addEventListener("click", function(event) {
        const group = event.target.closest(".annotation-group");
        if (!group) {
            return;
        }
        const annotationId = group.getAttribute("data-annotation-id");
        if (!annotationId) {
            return;
        }
        trigger(
            "annotation_select",{annotation_id: parseInt(annotationId)});
    });
    function setupDragging() {
        const svg = element.querySelector("svg");
        console.log("SETTING UP SVG:",svg);
        let drawStart = null;
        let previewBox = null;
        if (!svg) {
            console.log("SVG NOT FOUND");
            return;
        }
        console.log("SVG FOUND");
        svg.addEventListener("mousedown", function(event) {
            if (svg.getAttribute("data-add-mode") === "true")
            {
                event.preventDefault();
                const point = svg.createSVGPoint();
                point.x = event.clientX;
                point.y = event.clientY;
                const start = point.matrixTransform(svg.getScreenCTM().inverse());
                drawing = true;
                drawStart = start;
                previewBox = document.createElementNS("http://www.w3.org/2000/svg","rect");
                previewBox.setAttribute("class","annotation-preview");
                previewBox.setAttribute("x",start.x);
                previewBox.setAttribute("y",start.y);
                previewBox.setAttribute("width",0);
                previewBox.setAttribute("height",0);
                previewBox.setAttribute("fill","transparent");
                previewBox.setAttribute("stroke","white");
                previewBox.setAttribute("stroke-width",3);
                previewBox.setAttribute("stroke-dasharray","8 4");
                svg.appendChild(previewBox);
                function draw(event) {
                    if (!drawing) {
                        return;
                    }
                    const point = svg.createSVGPoint();
                    point.x = event.clientX;
                    point.y = event.clientY;
                    const position =point.matrixTransform(svg.getScreenCTM().inverse());
                    const x =Math.min(drawStart.x,position.x);
                    const y =Math.min(drawStart.y,position.y);
                    const width =Math.abs(position.x -drawStart.x);
                    const height =Math.abs(position.y -drawStart.y);
                    previewBox.setAttribute("x", x);
                    previewBox.setAttribute("y", y);
                    previewBox.setAttribute("width", width);
                    previewBox.setAttribute("height", height);
                }
                function stopDraw(event) {
                    if (!drawing) {
                        return;
                    }
                    drawing = false;
                    document.removeEventListener("mousemove",draw);
                    document.removeEventListener("mouseup",stopDraw);
                    const finalX = parseFloat(previewBox.getAttribute("x"));
                    const finalY = parseFloat(previewBox.getAttribute("y"));
                    const finalWidth = parseFloat(previewBox.getAttribute("width"));
                    const finalHeight = parseFloat(previewBox.getAttribute("height"));
                    trigger("annotation_add", {bbox: [finalX,finalY,finalWidth,finalHeight]});
                }

                document.addEventListener("mousemove",draw);

                document.addEventListener("mouseup",stopDraw);
                return;
            }
            const group = event.target.closest(".annotation-group");
            if (!group) {
                return;
            }
            // RESIZING
            if (event.target.classList.contains("resize-handle")) {
                const handle = event.target;
                // Determine which corner was grabbed
                const isTL = handle.classList.contains("handle-tl");
                const isTR = handle.classList.contains("handle-tr");
                const isBL = handle.classList.contains("handle-bl");
                const isBR = handle.classList.contains("handle-br");
                if (!isTL &&!isTR &&!isBL &&!isBR) {
                    return;
                }
                event.preventDefault();
                let resizing = true;
                const startMouse = svg.createSVGPoint();
                startMouse.x = event.clientX;
                startMouse.y = event.clientY;
                const startPosition = startMouse.matrixTransform(svg.getScreenCTM().inverse());
                const box = group.querySelector(".annotation-box");
                const labelBackground = group.querySelector(".annotation-label-bg");
                const labelText = group.querySelector(".annotation-label");
                const startX = parseFloat(box.getAttribute("x"));
                const startY = parseFloat(box.getAttribute("y"));
                const startWidth = parseFloat(box.getAttribute("width"));
                const startHeight = parseFloat(box.getAttribute("height"));
                function resize(event) {
                    if (!resizing) {
                        return;
                    }
                    const mouse = svg.createSVGPoint();
                    mouse.x = event.clientX;
                    mouse.y = event.clientY;
                    const position = mouse.matrixTransform(svg.getScreenCTM().inverse());
                    const dx = position.x -startPosition.x;
                    const dy = position.y -startPosition.y;
                    let newX = startX;
                    let newY = startY;
                    let newWidth = startWidth;
                    let newHeight = startHeight;
                    // TOP-LEFT
                    if (isTL) {
                        newX = Math.min(startX + dx,startX + startWidth - 10);
                        newY = Math.min(startY + dy,startY + startHeight - 10);
                        newWidth = startWidth -(newX - startX);
                        newHeight = startHeight -(newY - startY);
                    }
                    // TOP-RIGHT
                    if (isTR) {
                        newY = Math.min(startY + dy,startY + startHeight - 10);
                        newWidth = Math.max(10,startWidth + dx);
                        newHeight = startHeight -(newY - startY);
                    }
                    // BOTTOM-LEFT
                    if (isBL) {
                        newX = Math.min(startX + dx,startX + startWidth - 10 );
                        newWidth = startWidth -(newX - startX);
                        newHeight = Math.max(10,startHeight + dy);
                    }
                    // BOTTOM-RIGHT
                    if (isBR) {
                        newWidth = Math.max(10,startWidth + dx);
                        newHeight = Math.max(10,startHeight + dy);
                    }
                    // Update bounding box
                    box.setAttribute("x",newX);
                    box.setAttribute( "y",newY);
                    box.setAttribute("width",newWidth);
                    box.setAttribute("height",newHeight);
                    // Move label with top-left corner
                    labelBackground.setAttribute("x",newX);
                    labelBackground.setAttribute("y",newY);
                    labelText.setAttribute("x",newX + 3);
                    labelText.setAttribute("y",newY + parseFloat(labelText.getAttribute("font-size")));
                    // Get all handles
                    const handleTL = group.querySelector(".handle-tl");
                    const handleTR = group.querySelector(".handle-tr");
                    const handleBL = group.querySelector(".handle-bl");
                    const handleBR = group.querySelector(".handle-br");
                    // Update handle positions
                    handleTL.setAttribute("x",newX - 6);
                    handleTL.setAttribute( "y",newY - 6);
                    handleTR.setAttribute("x",newX + newWidth - 6);
                    handleTR.setAttribute("y",newY - 6);
                    handleBL.setAttribute("x", newX - 6);
                    handleBL.setAttribute( "y",newY + newHeight - 6);
                    handleBR.setAttribute("x",newX + newWidth - 6);
                    handleBR.setAttribute( "y",newY + newHeight - 6);
                }
                function stopResize() {
                    resizing = false;
                    const finalX = parseFloat(box.getAttribute("x"));
                    const finalY = parseFloat(box.getAttribute("y"));
                    const finalWidth = parseFloat(box.getAttribute("width"));
                    const finalHeight = parseFloat(box.getAttribute("height"));
                    trigger("annotation_edit",{annotation_id:parseInt(group.getAttribute("data-annotation-id")),bbox:[finalX,finalY,finalWidth,finalHeight]});
                    document.removeEventListener("mousemove",resize);
                    document.removeEventListener("mouseup",stopResize);
                }
                document.addEventListener("mousemove",resize);
                document.addEventListener( "mouseup",stopResize);
                return;
            }
            // dragging code
            event.preventDefault();
            let dragging = true;
            const startMouse = svg.createSVGPoint();
            startMouse.x = event.clientX;
            startMouse.y = event.clientY;
            const startPosition = startMouse.matrixTransform(svg.getScreenCTM().inverse());
            const startGroupX = 0;
            const startGroupY = 0;
            group.style.cursor = "grabbing";
            let finalDX = 0;
            let finalDY = 0;
            function move(event) {
                if (!dragging) {
                    return;
                }
                const mouse = svg.createSVGPoint();
                mouse.x = event.clientX;
                mouse.y = event.clientY;
                const position = mouse.matrixTransform(svg.getScreenCTM().inverse());
                const dx = position.x -startPosition.x;
                const dy = position.y -startPosition.y;
                finalDX = dx;
                finalDY = dy;
                group.setAttribute("transform",`translate(${startGroupX + dx}, ${startGroupY + dy})`);
            }
            function stop() {
                dragging = false;
                const box = group.querySelector(".annotation-box");
                const startX = parseFloat(box.getAttribute("x"));
                const startY = parseFloat(box.getAttribute("y"));
                const Width = parseFloat(box.getAttribute("width"));
                const Height = parseFloat(box.getAttribute("height"));
                const finalX = startX+finalDX;
                const finalY = startY+finalDY;
                trigger("annotation_move",{annotation_id:parseInt(group.getAttribute("data-annotation-id")),bbox:[finalX,finalY,Width,Height]});
                group.style.cursor = "move";
                document.removeEventListener("mousemove",move);
                document.removeEventListener("mouseup",stop);
            }
            document.addEventListener("mousemove",move);
            document.addEventListener("mouseup",stop);
        });
    }
    setupDragging();
    watch("value", () => {
        setupDragging();
    });
""")
        return {"image_display": image_display}


def create_annotation_panel(categories):
    """Create the annotation editing panel."""
    with gr.Column():
        gr.Markdown("### Annotation Actions")
        with gr.Row():
            apply_button = gr.Button("✓ Apply",variant="primary")
        with gr.Row():
            Add_annotation = gr.Button("Add Annotation")
            delete_button = gr.Button("Delete Annotation")
            reset_button = gr.Button("Reset Changes")
        # annotation_info = gr.JSON(label="Current Annotation",value={})
        label_dropdown = gr.Dropdown(choices=list(categories.values()),label="Annotation Label",interactive=True)
        return {
            "apply_button": apply_button,
            "Add_annotation": Add_annotation,
            "delete_button": delete_button,
            "reset_button": reset_button,
            "label_dropdown":label_dropdown
        }


def create_status_bar():
    """Create the status/output area."""
    status = gr.Markdown(
        "Status: Ready"
    )
    return status

"""Construct the complete Gradio application."""
def create_app():
    """Load image and annotation data."""
    args = parse_args()
    image_dir = args.images.expanduser().resolve()
    annotation_path = args.annotations.expanduser().resolve()
    change_log_path = args.change_log.expanduser().resolve()
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
    """ Rendering function"""
    def render_interactive_image(index: int, selected_annotation_id=None, edited_annotations=None, add_mode=False):
        image_path = image_files[index]
        image = Image.open(image_path).convert("RGB")
        image_record = coco_by_filename.get(image_path.name)
        if image_record is None:
            return "<p>No COCO annotation found for this image.</p>"
        image_id = int(image_record["id"])
        annotations = annotations_by_image.get(image_id,[])
        # Convert the PIL image into a base64 encoded JPEG
        image_buffer = BytesIO()
        image.save(image_buffer, format="JPEG")
        image_data = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
        # Start the SVG
        svg_parts = [f"""
            <div style="width: 100%; overflow:auto;">
            <svg
            width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}" data-add-mode="{str(add_mode).lower()}"
            >
            <image
            href="data:image/jpeg;base64,{image_data}" x="0" y="0" width="{image.width}" height="{image.height}"
            />"""]
        # Draw annotations
        all_annotations = list(annotations)
        # Add locally created annotations
        for annotation_id, edited in (edited_annotations or {}).items():
            if edited['new'] and edited['image_id'] == image_id:
                all_annotations.append({"id": edited['id'],"image_id": edited['image_id'],
                                        "category_id": edited["category_id"],"bbox": edited["bbox"]})
        for annotation in all_annotations:
            (annotation, is_deleted) = get_current_annotation(annotation,edited_annotations or {})
            if is_deleted:
                continue
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x, y, width, height = map(float, bbox[:4])
            annotation_id = int(annotation["id"])
            category_id = int(annotation["category_id"])
            red, green, blue = colour_for_category(category_id)
            colour = f"rgb({red},{green},{blue})"
            # Highlight selected annotation
            label = categories.get(category_id,f"Unknown ({category_id})")
            font_size = max(16,round(min(image.width, image.height) / 45))
            padding = 3
            label_width = len(label) * font_size * 0.6
            label_height = font_size + 2 * padding
            # Normal annotation boundary
            annotation_svg = f""" 
            <g
                class="annotation-group"
                data-annotation-id="{annotation_id}"
                transform="translate(0, 0)"
                style="cursor: move;"
            >
                <rect
                class="annotation-box"
                x="{x}" y="{y}" width="{width}" height="{height}" 
                fill="transparent" stroke="{colour}" stroke-width="3"
                />
                <rect
                    class="annotation-label-bg"
                    x="{x}" y="{y}" width="{label_width + 2 * padding}" height="{label_height}" fill="{colour}"
                    style="pointer-events: none;"
                />
                <text
                    class="annotation-label"
                    x="{x + padding}" y="{y + font_size}" font-size="{font_size}px" font-family="Arial, sans-serif"
                    font-weight="bold" fill="white" stroke="black" stroke-width="1" paint-order="stroke"
                    style="pointer-events: none;"
                >
                {label}</text>
                """
            if annotation_id == selected_annotation_id:
                annotation_svg += f"""
                <!-- Resize handles -->
                    <rect
                        class="resize-handle handle-tl"
                        x="{x - 6}"
                        y="{y - 6}"
                        width="12"
                        height="12"
                        fill="white"
                        stroke="black"
                        stroke-width="2"
                        style="cursor: nwse-resize;"
                    />
                    <rect
                        class="resize-handle handle-tr"
                        x="{x + width - 6}"
                        y="{y - 6}"
                        width="12"
                        height="12"
                        fill="white"
                        stroke="black"
                        stroke-width="2"
                        style="cursor: nesw-resize;"
                    />
                    <rect
                        class="resize-handle handle-bl"
                        x="{x - 6}"
                        y="{y + height - 6}"
                        width="12"
                        height="12"
                        fill="white"
                        stroke="black"
                        stroke-width="2"
                        style="cursor: nesw-resize;"
                    />
                    <rect
                        class="resize-handle handle-br"
                        x="{x + width - 6}"
                        y="{y + height - 6}"
                        width="12"
                        height="12"
                        fill="white"
                        stroke="black"
                        stroke-width="2"
                        style="cursor: nwse-resize;"
                        />"""
            annotation_svg+= f"""</g>"""
            svg_parts.append(annotation_svg)
        # Close SVG
        svg_parts.append("""</svg></div>""")
        return "".join(svg_parts)
    """ Json saving function """
    def save_current_annotations(index: int,edited_annotations: dict):
        nonlocal annotations_by_image
        changes = [0,0,0,0]
        # Load the current test JSON
        with open(annotation_path,"r") as file:
            annotation_JSON: dict[str,any] = json.load(file)
        # load change log
        with open(change_log_path,"r") as log:
            change_log: dict = json.load(log)
        # Apply local edits
        updated_annotations = []
        for annotation in annotation_JSON['annotations']:
            annotation_id = int(annotation["id"])
            if annotation_id in edited_annotations:
                edits = edited_annotations[annotation_id]
                if edits['deleted']:
                    continue
                new_annotation = {"id":edits["id"], "image_id":edits['image_id'],"category_id":edits["category_id"],
                                  "bbox":edits['bbox'],"area":edits["area"],"is_crowd":edits["is_crowd"],
                                  "segmentation":edits["segmentation"]}
                updated_annotations.append(new_annotation)
                if edits['bbox'] != annotation['bbox']: 
                    changes[MOVED_ANNOTATIONS] = changes[MOVED_ANNOTATIONS]+1
                if edits['category_id'] != annotation['category_id']:
                    changes[CHANGED_LABELS] = changes[CHANGED_LABELS]+1
            else:
                updated_annotations.append(annotation)

        # Replace this image's annotations in the test JSON
        for annotation in annotation_JSON['annotations']:
            for new_annotation in updated_annotations:
                if int(new_annotation['id']) == int(annotation['id']):
                    annotation.update(new_annotation)
                        
        # Add locally created annotations
        for annotation_id, edits in edited_annotations.items():
            if edits['new']:
                annotation_JSON['annotations'].append({"id": edits['id'],"image_id": edits['image_id'],
                                        "category_id": int(edits["category_id"]),"bbox": edits["bbox"],
                                        "area:":edits['area'], "is_crowd":0, "segmentation":[]})
                changes[ADDED_ANNOTATIONS] = changes[ADDED_ANNOTATIONS]+1
        # Remove deleted annotations
        for annotation_id, edited in edited_annotations.items():
            if edited['deleted'] and not edited['new']:
                for annotation in annotation_JSON['annotations']:
                    if annotation['id'] == edited['id']:
                        annotation_JSON['annotations'].remove(annotation)
                        changes[REMOVED_ANNOTATIONS] = changes[REMOVED_ANNOTATIONS]+1
        # document changes
        change_log = {'false_positive': change_log['false_positive']+changes[REMOVED_ANNOTATIONS],
                      'false_negative':change_log['false_negative']+changes[ADDED_ANNOTATIONS],
                      'localisation':change_log['localisation']+changes[MOVED_ANNOTATIONS],
                      'classification':change_log["classification"]+changes[CHANGED_LABELS]}
        # Write the updated test JSON
        with open(annotation_path, "w") as f:
            json.dump(annotation_JSON, f, indent=4)
        # write the new change log
        with open(change_log_path, "w") as log_write:
            json.dump(change_log, log_write, indent=4)
        print(f"changes saved")
        edited_annotations.clear()
        annotations_by_image.clear()
        annotations_by_image = defaultdict(list)
        for annotation in annotation_JSON["annotations"]:
            image_id = int(annotation["image_id"])
            annotations_by_image[image_id].append(annotation)
        return(edited_annotations,render_interactive_image(index))

    """Annotation utility functions/updating functions"""
    def select_svg_annotation(index: int, edited_annotations: dict, evt: gr.EventData):
        annotation_id = evt.annotation_id
        if annotation_id is None:
            raise ValueError(f"parsed annotation ID is of None type: expected integer")
        image_path = image_files[index]
        image_record = coco_by_filename.get(image_path.name)
        if image_record is None:
            raise ValueError(f"parsed image path of type None: expected String")
        image_id = int(image_record["id"])
        annotations = annotations_by_image.get(image_id, [])
        selected_annotation = None
        for annotation in annotations:
            if int(annotation["id"]) == int(annotation_id):
                selected_annotation = annotation
                break
        if (annotation_id in edited_annotations):
            if edited_annotations[annotation_id]['new']:
                selected_annotation = edited_annotations[annotation_id]
        if selected_annotation is None:
            return (render_interactive_image(index,None,edited_annotations),None,None)
        category_id = int(selected_annotation["category_id"])
        return (render_interactive_image(index, int(annotation_id), edited_annotations),int(annotation_id),
                categories.get(category_id,f"Unknown ({category_id})"))
    
    def get_current_annotation(annotation,edited_annotations):
                annotation_id = int(annotation["id"])
                if annotation_id in edited_annotations:
                    edited = annotation.copy()
                    edited.update(edited_annotations[annotation_id])
                    is_deleted = edited_annotations[annotation_id]['deleted']
                    return edited, is_deleted
                return annotation, False
    
    def update_edited_annotation(index: int,edited_annotations: dict,evt: gr.EventData):
        image_path = image_files[index]
        image_record = coco_by_filename.get(image_path.name)
        image_id = int(image_record['id'])
        annotations = annotations_by_image.get(image_id,[])
        annotation_id = int(evt._data['annotation_id'])
        for annotation in annotations:
                    if annotation_id == annotation['id']:
                        category = annotation['category_id']
        new_bbox = [float(value) for value in evt._data['bbox']]
        area = new_bbox[2] * new_bbox[3]
        if annotation_id not in edited_annotations:
            edited_annotations.update({annotation_id:{'id':annotation_id,'bbox':new_bbox, 'deleted':False, "new":False,
                                                       "image_id":image_id, "category_id":category,
                                                       "area":area, "is_crowd":0,"segmentation":[]}})
        else:
            edited_annotations.update({annotation_id:{'id':annotation_id,'bbox':new_bbox, 'deleted':edited_annotations[annotation_id]['deleted'],
                                                      "new":edited_annotations[annotation_id]['new'],
                                                      "image_id":image_id, "category_id":edited_annotations[annotation_id]['category_id'],
                                                      "area":area, "is_crowd":0,"segmentation":[]}})
        return (render_interactive_image(index,annotation_id,edited_annotations), edited_annotations)
    
    def delete_annotation(index: int, selected_annotation_id: int, edited_annotation: dict):
        if selected_annotation_id is not None:
            edited_annotation.update({selected_annotation_id:{'id':selected_annotation_id,
                                                              'bbox': edited_annotation[selected_annotation_id]['bbox'],
                                                              'deleted': True, 'new':edited_annotation[selected_annotation_id]['new'],
                                                              "image_id":edited_annotation[selected_annotation_id]['image_id'],
                                                              "category_id":edited_annotation[selected_annotation_id]['category_id'],
                                                              "area":edited_annotation[selected_annotation_id]['area'],
                                                              "is_crowd":edited_annotation[selected_annotation_id]["is_crowd"],
                                                              "segmentation":edited_annotation[selected_annotation_id]["segmentation"]}})
        return (render_interactive_image(index,selected_annotation_id,edited_annotation),edited_annotation)
    
    def add_annotation(index: int,edited_annotations: dict, evt: gr.EventData):
        image_path = image_files[index]
        image_record = coco_by_filename.get(image_path.name)
        image_id = int(image_record['id'])
        bbox = [float(value)for value in evt._data["bbox"]]
        area = bbox[2]*bbox[3]
        existing_ids = set(int(annotation["id"])for annotation in annotations_by_image.get(int(coco_by_filename[image_files[index].name]["id"]),[]))
        existing_ids.update(int(annotation_id)for annotation_id in edited_annotations)
        new_annotation_id = (max(existing_ids, default=0) + 1)
        edited_annotations[new_annotation_id] = {'id': new_annotation_id,'bbox':bbox, 'deleted':False, 'new':True, 
                                                 'category_id':0, 'image_id':image_id,
                                                 "area":area,"is_crowd":0,"segmentation":[]}
        return(render_interactive_image(index,new_annotation_id,edited_annotations,False), edited_annotations,False)

    def update_annotation_label(index: int,annotation_id: int,edited_annotations: dict,label: str):
        if annotation_id is None or label is None:
            return (edited_annotations or {},render_interactive_image(index,annotation_id,edited_annotations or {}))
        # Find the category ID corresponding to the selected label
        category_id = None
        for cat_id, category_name in categories.items():
            if category_name == label:
                category_id = int(cat_id)
                break
        if category_id is None:
            raise ValueError(f"Unknown annotation label: {label}")
        edited_annotations[annotation_id]["category_id"] = category_id
        return (edited_annotations,render_interactive_image(index,int(annotation_id),edited_annotations))
    
    def enable_add_mode() -> bool:
        return(True)
    
    """ Image cycling functions without wrap around to avoid confusion """
    def next_image(index: int, edited_annotations: dict):
        index = min(index + 1, len(image_files) - 1)
        return render_interactive_image(index, edited_annotations=edited_annotations), index, None, None

    def previous_image(index: int, edited_annotations: dict):
        index = max(index - 1, 0)
        return render_interactive_image(index, edited_annotations=edited_annotations), index, None, None
    
    """App building and button handling"""
    with gr.Blocks(
        title="Building Dataset Cleaner",
    ) as app:
        image_index = gr.State(0)
        selected_annotation_id = gr.State(None)
        edited_annotations = gr.State({})
        add_mode = gr.State(False)
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
                annotation_panel = create_annotation_panel(categories)
        gr.Markdown("---")
        create_status_bar()
        """Event control and button interactions"""
        image_panel["image_display"].annotation_select(
            fn=select_svg_annotation,
            inputs=[image_index, edited_annotations],
            outputs=[image_panel["image_display"],selected_annotation_id,annotation_panel['label_dropdown']])
        
        image_panel["image_display"].annotation_edit(
            fn=update_edited_annotation,
            inputs=[image_index, edited_annotations],
            outputs=[image_panel["image_display"],edited_annotations])
        
        image_panel["image_display"].annotation_move(
                    fn=update_edited_annotation,
                    inputs=[image_index, edited_annotations],
                    outputs=[image_panel["image_display"],edited_annotations])
        annotation_panel["apply_button"].click(fn=save_current_annotations,
                                               inputs=[image_index,edited_annotations],
                                               outputs=[edited_annotations,image_panel["image_display"]])
        annotation_panel["delete_button"].click(
            fn=delete_annotation,
            inputs=[image_index,selected_annotation_id, edited_annotations],
            outputs=[image_panel["image_display"], edited_annotations])
        
        annotation_panel["Add_annotation"].click(
            fn=enable_add_mode,
            inputs=None,
            outputs=add_mode)
        
        add_mode.change(
            fn=lambda mode, index, selected, edits:render_interactive_image(index,selected,edits,mode),
            inputs=[add_mode,image_index,selected_annotation_id,edited_annotations],
            outputs=image_panel["image_display"])
        
        image_panel["image_display"].annotation_add(
            fn=add_annotation,
            inputs=[image_index, edited_annotations],
            outputs=[image_panel["image_display"],edited_annotations,add_mode])

        app.load(fn=lambda: render_interactive_image(0),outputs=image_panel["image_display"])
        dataset_panel["next_button"].click(
            fn=next_image,
            inputs=[image_index, edited_annotations],
            outputs=[image_panel["image_display"],image_index,selected_annotation_id,annotation_panel['label_dropdown']]
        )
        dataset_panel["previous_button"].click(
                    fn=previous_image,
                    inputs=[image_index, edited_annotations],
                    outputs=[image_panel["image_display"],image_index,selected_annotation_id,annotation_panel['label_dropdown']]
                )
        annotation_panel["label_dropdown"].change(
        fn=update_annotation_label,
        inputs=[image_index,selected_annotation_id,edited_annotations,annotation_panel["label_dropdown"]],
        outputs=[edited_annotations,image_panel["image_display"]])
    app.launch()


if __name__ == "__main__":
    create_app()