from typing import Dict, List

import numpy as np

from vectorify.utils import (
    extract_color_contours,
    load_image,
    quantize_image,
    rgba_to_rgb,
)


class Element:
    def __init__(
        self,
        tag: str,
        attributes: dict[str, str],
        content: str = "",
    ):
        self.tag = tag
        self.attributes = attributes
        self.content = content

    @property
    def attrs(self) -> str:
        return " ".join(f'{k}="{v}"' for k, v in self.attributes.items())

    def to_html(self) -> str:
        return f"<{self.tag} {self.attrs}>{self.content}</{self.tag}>"


class Path(Element):
    def __init__(self, contour: np.ndarray, fill: str = "none"):
        self.contour = contour
        r, g, b = rgba_to_rgb(fill)
        self.fill = f"rgb({r},{g},{b})"
        self.d = self.contour
        super().__init__(
            tag="path",
            attributes={
                "d": self.d,
                "fill": self.fill,
            },
        )

    @property
    def d(self) -> str:
        return self._d

    @d.setter
    def d(self, value: str):
        subpaths = []
        keep = np.append(True, np.any(np.diff(value, axis=0) != 0, axis=1))
        contour = value[keep]
        moves = " ".join(f"L{round(y, 2)},{round(x, 2)}" for x, y in contour)
        moves = "M" + moves[1:] + "Z"
        subpaths.append(moves)
        self._d = "".join(subpaths)


class Svg(Element):
    def __init__(self, width: int, height: int, contours: List[np.ndarray]):
        self.paths = contours
        super().__init__(
            tag="svg",
            attributes={
                "width": str(width),
                "height": str(height),
                "xmlns": "http://www.w3.org/2000/svg",
            },
            content=self.paths,
        )

    @property
    def paths(self) -> str:
        return self._paths

    @paths.setter
    def paths(self, contours: Dict) -> str:
        parts = []
        for color, color_contours in contours.items():
            a = float(color[3] / 255 if len(color) > 3 else 1)
            if a <= 0.5:
                continue
            r, g, b = rgba_to_rgb(color)
            fill = f"rgb({r},{g},{b})"
            # Merge all contour d-strings for the same fill into one <path>
            d_parts = []
            for contour in color_contours:
                keep = np.append(
                    True, np.any(np.diff(contour, axis=0) != 0, axis=1)
                )
                c = contour[keep]
                moves = " ".join(
                    f"L{round(y, 2)},{round(x, 2)}" for x, y in c
                )
                d_parts.append("M" + moves[1:] + "Z")
            d = "".join(d_parts)
            parts.append(f'<path d="{d}" fill="{fill}"/>')
        self._paths = "".join(parts)

    @classmethod
    def generate_svg(
        cls,
        filepath: str,
        output_path: str,
        n_colors: int = 8,
        median_size: int = 5,
        min_region: int = 50,
        smooth: float = 2,
        simplify: float = 0.5,
        buffer: float = 1,
    ):
        import time

        t0 = time.perf_counter()
        img = load_image(filepath)
        print(f"[vectorify] load_image: {time.perf_counter() - t0:.2f}s")

        t1 = time.perf_counter()
        labels, centroids = quantize_image(
            img,
            n_colors=n_colors,
            median_size=median_size,
            min_region=min_region,
        )
        print(f"[vectorify] quantize_image: {time.perf_counter() - t1:.2f}s")

        t2 = time.perf_counter()
        color_contours = extract_color_contours(
            labels,
            centroids,
            smooth=smooth,
            simplify=simplify,
            buffer=buffer,
        )
        print(f"[vectorify] extract_contours: {time.perf_counter() - t2:.2f}s")

        t3 = time.perf_counter()
        svg = Svg(width=img.width, height=img.height, contours=color_contours)
        with open(output_path, "w") as f:
            f.write(svg.to_html())
        print(f"[vectorify] write_svg: {time.perf_counter() - t3:.2f}s")
        print(f"[vectorify] total: {time.perf_counter() - t0:.2f}s")
