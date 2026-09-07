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
