from collections import Counter

from PIL import Image

from tinyvlm.dataset import ImageCaptionDataset, split_by_image
from tinyvlm.toy_grounding import (
    COLORS,
    POSITIONS,
    SHAPES,
    create_toy_grounding_dataset,
    extract_visual_attributes,
    normalize_caption,
)
from tokenizer import CharTokenizer


def test_create_toy_grounding_dataset(tmp_path):
    image_root, caption_file, rows = create_toy_grounding_dataset(
        tmp_path, samples_per_combination=2, image_size=32, seed=7
    )
    dataset = ImageCaptionDataset(
        image_root, caption_file, CharTokenizer(), image_size=32, max_text_len=32
    )

    assert len(rows) == 72
    assert len(dataset) == 72
    assert dataset[0]["image"].shape == (3, 32, 32)
    assert len({filename for filename, _ in rows}) == 72

    combinations = Counter(extract_visual_attributes(caption) for _, caption in rows)
    expected = {
        (color, shape, position)
        for color in COLORS
        for shape in SHAPES
        for position in POSITIONS
    }
    assert set(combinations) == expected
    assert set(combinations.values()) == {2}

    train_indices, val_indices, train_images, val_images = split_by_image(
        dataset.samples, val_ratio=0.25, seed=11
    )
    assert train_images.isdisjoint(val_images)
    assert len(train_indices) + len(val_indices) == len(dataset)


def test_generated_pixels_match_caption_attributes(tmp_path):
    image_root, _, rows = create_toy_grounding_dataset(
        tmp_path, samples_per_combination=1, image_size=64, seed=19
    )

    for filename, caption in rows:
        color, shape, position = extract_visual_attributes(caption)
        expected_rgb = COLORS[color]
        with Image.open(image_root / filename) as image:
            pixels = image.convert("RGB")
            coordinates = [
                (x, y)
                for y in range(pixels.height)
                for x in range(pixels.width)
                if pixels.getpixel((x, y)) == expected_rgb
            ]

        assert coordinates, f"no {color} foreground pixels in {filename}"
        center_x = sum(x for x, _ in coordinates) / len(coordinates)
        if position == "left":
            assert center_x < 0.4 * pixels.width
        elif position == "center":
            assert 0.4 * pixels.width < center_x < 0.6 * pixels.width
        else:
            assert center_x > 0.6 * pixels.width

        xs = [x for x, _ in coordinates]
        ys = [y for _, y in coordinates]
        bounding_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        fill_ratio = len(coordinates) / bounding_area
        if shape == "square":
            assert fill_ratio == 1.0
        elif shape == "circle":
            assert 0.70 < fill_ratio < 0.90
        else:
            assert 0.45 < fill_ratio < 0.65


def test_toy_grounding_generation_is_seed_reproducible(tmp_path):
    first_root, first_csv, first_rows = create_toy_grounding_dataset(
        tmp_path / "first", samples_per_combination=1, image_size=32, seed=23
    )
    second_root, second_csv, second_rows = create_toy_grounding_dataset(
        tmp_path / "second", samples_per_combination=1, image_size=32, seed=23
    )
    third_root, _, third_rows = create_toy_grounding_dataset(
        tmp_path / "third", samples_per_combination=1, image_size=32, seed=24
    )

    assert first_rows == second_rows
    assert first_csv.read_text(encoding="utf-8") == second_csv.read_text(
        encoding="utf-8"
    )
    assert all(
        (first_root / filename).read_bytes()
        == (second_root / filename).read_bytes()
        for filename, _ in first_rows
    )
    assert any(
        (first_root / filename).read_bytes()
        != (third_root / filename).read_bytes()
        for filename, _ in third_rows
    )


def test_visual_attribute_extraction_and_caption_normalization():
    assert extract_visual_attributes("A BLUE triangle on the RIGHT!") == (
        "blue",
        "triangle",
        "right",
    )
    assert extract_visual_attributes("an object with no known attributes") == (
        None,
        None,
        None,
    )
    assert normalize_caption("  A red, CIRCLE on the left! ") == (
        "a red circle on the left"
    )
