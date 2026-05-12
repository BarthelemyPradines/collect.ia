from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

from src.llm.structured_output import BoundingBox


def export_with_bounding_boxes(
    image_path: Path,
    bounding_boxes: list[BoundingBox],
    output_dir: Path | None = None,
) -> Path:
    """Draw bounding boxes on an image and save to output/temp.

    Coordinates in each BoundingBox are normalized [y_min, x_min, y_max, x_max]
    on a 0-1000 scale (Gemini convention).

    Returns the path to the saved image.
    """
    if output_dir is None:
        output_dir = Path("outputs/temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = list(ImageColor.colormap.keys())

    with Image.open(image_path) as im:
        width, height = im.size
        draw = ImageDraw.Draw(im)

        for i, bbox in enumerate(bounding_boxes):
            abs_y_min = int(bbox.box_2d[0] / 1000 * height)
            abs_x_min = int(bbox.box_2d[1] / 1000 * width)
            abs_y_max = int(bbox.box_2d[2] / 1000 * height)
            abs_x_max = int(bbox.box_2d[3] / 1000 * width)

            color = colors[i % len(colors)]

            draw.rectangle(
                ((abs_x_min, abs_y_min), (abs_x_max, abs_y_max)),
                outline=color,
                width=4,
            )
            if bbox.label:
                draw.text((abs_x_min + 8, abs_y_min + 6), bbox.label, fill=color)

        out_path = output_dir / f"bbox_{image_path.name}"
        im.save(out_path)

    return out_path
