from typing import Dict, List

import cairosvg

from vectorify.svg import Svg
from PIL import Image, ImageDraw, ImageFont
import os

EXAMPLE_CONFIG = {
    "n_colors": [2, 4, 8, 16],
    "median_size": [0, 3, 5, 7],
    "min_region": [0, 50, 100, 200],
    "smooth": [0, 1, 2, 3],
    "simplify": [0, 0.5, 1, 2],
    "buffer": [0, 1, 2, 5],
}


svgs = [
    f"docs/svgs/{k}_{v}.svg" for k, values in EXAMPLE_CONFIG.items() for v in values
]

pngs = {
    k: [f"docs/pngs/{k}_{v}.png" for v in values]
    for k, values in EXAMPLE_CONFIG.items()
}


def generate_example_svgs() -> None:
    for k, v in EXAMPLE_CONFIG.items():
        for value in v:
            output_path = f"docs/examples/{k}_{value}.svg"
            Svg.generate_svg(
                filepath="docs/strawberries.png",
                output_path=output_path,
                **{k: value},
            )


def convert_svg_to_png(svg_path: str, png_path: str) -> None:
    cairosvg.svg2png(url=svg_path, write_to=png_path)


def convert_svgs_to_pngs(svg_paths: list[str], png_paths: list[str]) -> None:
    for svg_path in svg_paths:
        png_path = svg_path.replace("svgs", "pngs").replace(".svg", ".png")
        key = svg_path.split("/")[-1].split(".")[0]
        arr = key.split("_")
        key = "_".join(arr[0:-1])
        value = arr[-1]
        convert_svg_to_png(svg_path, png_path)
        img = Image.open(png_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=40)
        draw.text(
            (50, 50),
            f"{key}: {value}",
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 255),
        )
        new_size = (img.width // 2, img.height // 2)
        img = img.resize(new_size)
        img.save(png_path)


def combine_pngs_4wide(
    png_groups: Dict[str, List[str]] = pngs,
    out_dir: str = "docs/img",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for key, paths in png_groups.items():
        images = [Image.open(p).convert("RGBA") for p in paths]

        rows = []

        for row_idx in range(0, len(images), 4):
            row = images[row_idx : row_idx + 4]
            if not row:
                continue

            target_h = min(im.height for im in row)
            row = [
                im.resize(
                    (int(im.width * target_h / im.height), target_h),
                    Image.LANCZOS,
                )
                for im in row
            ]

            total_w = sum(im.width for im in row)
            max_h = max(im.height for im in row)

            combined = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))

            x = 0
            for im in row:
                combined.paste(im, (x, 0))
                x += im.width

            rows.append(combined)

        if len(rows) == 1:
            final = rows[0]
        else:
            total_h = sum(r.height for r in rows)
            max_w = max(r.width for r in rows)
            final = Image.new("RGBA", (max_w, total_h), (0, 0, 0, 0))

            y = 0
            for r in rows:
                final.paste(r, (0, y))
                y += r.height

        final.save(f"{out_dir}/{key}.png")
