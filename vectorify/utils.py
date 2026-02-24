from PIL import Image
from rdp import rdp
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.ndimage import median_filter, label as ndlabel
from skimage.measure import find_contours
from scipy.ndimage import binary_dilation, gaussian_filter1d


def rgba_to_rgb(rgba: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r, g, b, a = rgba
    a /= 255
    r = int(r * a + 255 * (1 - a))
    g = int(g * a + 255 * (1 - a))
    b = int(b * a + 255 * (1 - a))
    return r, g, b


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return img


def clean_labels(
    labels: np.ndarray,
    median_size: int = 0,
    min_region: int = 0,
) -> np.ndarray:
    if median_size > 0:
        labels = median_filter(labels, size=median_size)
    if min_region > 0:
        structure = np.ones((3, 3), dtype=int)
        for lbl in np.unique(labels):
            mask = labels == lbl
            components, n = ndlabel(mask, structure=structure)
            for comp_id in range(1, n + 1):
                comp_mask = components == comp_id
                if comp_mask.sum() < min_region:
                    border = binary_dilation(comp_mask, structure) & ~comp_mask
                    if border.any():
                        neighbour_labels = labels[border]
                        replacement = np.bincount(neighbour_labels).argmax()
                    else:
                        replacement = lbl
                    labels[comp_mask] = replacement
    return labels


def quantize_image(
    img: Image.Image,
    n_colors: int,
    median_size: int = 0,
    min_region: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(img)
    h, w, c = arr.shape
    pixels = arr.reshape(-1, c).astype(np.float64)
    centroids, labels = kmeans2(pixels, n_colors + 1, minit="++")
    labels = labels.reshape(h, w)
    labels = clean_labels(
        labels,
        median_size=median_size,
        min_region=min_region,
    )
    return labels, centroids


def extract_color_contours(
    labels: np.ndarray,
    centroids: np.ndarray,
    smooth: float = 0,
    simplify: float = 0,
    no_data: list[tuple[int, ...]] = [],
    buffer: float = 0.5,
) -> dict[tuple[int, ...], list[np.ndarray]]:
    color_contours: dict[tuple[int, ...], list[np.ndarray]] = {}
    for lbl in np.unique(labels):
        color = tuple(centroids[lbl].astype(np.uint8).tolist())
        if color[:3] in no_data or color in no_data:
            color = (color[0], color[1], color[2], 0)
        mask = (labels == lbl).astype(np.float64)
        contours = find_contours(mask, level=0.5)
        if buffer > 0:
            buffered = []
            for c in contours:
                edges = np.diff(c, axis=0, append=c[:1])
                normals = np.column_stack([edges[:, 1], -edges[:, 0]])
                lengths = np.linalg.norm(normals, axis=1, keepdims=True)
                lengths[lengths == 0] = 1
                normals = normals / lengths
                buffered.append(c - buffer * normals)
            contours = buffered
        if smooth > 0:
            contours = [
                np.column_stack(
                    [
                        gaussian_filter1d(c[:, 0], sigma=smooth, mode="wrap"),
                        gaussian_filter1d(c[:, 1], sigma=smooth, mode="wrap"),
                    ]
                )
                for c in contours
            ]
        if simplify > 0:
            contours = [rdp(c, epsilon=simplify) for c in contours]
        if contours:
            color_contours[color] = contours
    return color_contours
