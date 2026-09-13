import csv
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw


COLORS = {
    "red": (220, 35, 35),
    "green": (35, 175, 70),
    "blue": (40, 90, 220),
    "yellow": (235, 195, 35),
}
SHAPES = ("circle", "square", "triangle")
POSITIONS = ("left", "center", "right")


def create_toy_grounding_dataset(
    output_root, samples_per_combination=30, image_size=64, seed=42
):
    """Create colored-shape images whose captions require visual attributes."""
    output_root = Path(output_root)
    image_root = output_root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    caption_file = output_root / "captions.csv"
    rng = random.Random(seed)
    rows = []
    sample_index = 0

    x_centers = {
        "left": image_size * 0.25,
        "center": image_size * 0.50,
        "right": image_size * 0.75,
    }
    for color_name, color_value in COLORS.items():
        for shape in SHAPES:
            for position in POSITIONS:
                caption = f"a {color_name} {shape} on the {position}"
                for _ in range(samples_per_combination):
                    background = rng.randint(225, 255)
                    image = Image.new(
                        "RGB", (image_size, image_size), (background,) * 3
                    )
                    draw = ImageDraw.Draw(image)
                    radius = rng.randint(image_size // 8, image_size // 6)
                    center_x = round(x_centers[position] + rng.randint(-3, 3))
                    center_y = image_size // 2 + rng.randint(-8, 8)
                    bounds = (
                        center_x - radius,
                        center_y - radius,
                        center_x + radius,
                        center_y + radius,
                    )

                    if shape == "circle":
                        draw.ellipse(bounds, fill=color_value)
                    elif shape == "square":
                        draw.rectangle(bounds, fill=color_value)
                    else:
                        draw.polygon(
                            [
                                (center_x, center_y - radius),
                                (center_x - radius, center_y + radius),
                                (center_x + radius, center_y + radius),
                            ],
                            fill=color_value,
                        )

                    filename = f"toy_{sample_index:05d}.png"
                    image.save(image_root / filename)
                    rows.append((filename, caption))
                    sample_index += 1

    rng.shuffle(rows)
    with caption_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "caption"))
        writer.writerows(rows)
    return image_root, caption_file, rows


def extract_visual_attributes(caption):
    words = set(re.findall(r"[a-z]+", caption.lower()))
    color = next((value for value in COLORS if value in words), None)
    shape = next((value for value in SHAPES if value in words), None)
    position = next((value for value in POSITIONS if value in words), None)
    return color, shape, position


def normalize_caption(caption):
    return " ".join(re.findall(r"[a-z]+", caption.lower()))
