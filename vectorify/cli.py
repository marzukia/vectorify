import argparse

from vectorify.svg import Svg


def main():
    parser = argparse.ArgumentParser(description="Vectorify - raster to SVG converter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = subparsers.add_parser("generate", help="Convert a raster image to SVG")
    gen.add_argument("filepath", type=str, help="Path to the input image")
    gen.add_argument("output", type=str, help="Output SVG path")
    gen.add_argument("--n_colors", type=int)
    gen.add_argument("--median_size", type=float)
    gen.add_argument("--min_region", type=float)
    gen.add_argument("--smooth", type=float)
    gen.add_argument("--simplify", type=float)
    gen.add_argument("--buffer", type=float)

    # --- update-readme ---
    subparsers.add_parser(
        "update-readme", help="Regenerate example images for the README"
    )

    args = parser.parse_args()

    if args.command == "generate":
        Svg.generate_svg(
            filepath=args.filepath,
            output_path=f"{args.output}",
            n_colors=args.n_colors or 8,
            median_size=args.median_size or 5,
            min_region=args.min_region or 50,
            smooth=args.smooth or 2,
            simplify=args.simplify or 0.5,
            buffer=args.buffer or 1,
        )

    elif args.command == "update-readme":
        from vectorify.docs import (
            generate_example_svgs,
            convert_svgs_to_pngs,
            combine_pngs_4wide,
            svgs,
            pngs,
        )

        print("Generating example SVGs…")
        generate_example_svgs()
        print("Converting SVGs → PNGs…")
        convert_svgs_to_pngs(svgs, list(pngs.values()))
        print("Combining PNGs into grid images…")
        combine_pngs_4wide()
        print("Done – README images updated.")
