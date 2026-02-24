from PIL import Image
from rdp import rdp
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist
from skimage.measure import find_contours, label as sklabel
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
        component_map = sklabel(labels, connectivity=2)
        sizes = np.bincount(component_map.ravel())
        small_mask = sizes < min_region
        small_ids = np.nonzero(small_mask)[0]
        if len(small_ids) > 0:
            from scipy.ndimage import find_objects

            slices = find_objects(component_map)
            h, w = labels.shape
            for comp_id in small_ids:
                if comp_id == 0:
                    continue
                sl = slices[comp_id - 1]
                if sl is None:
                    continue
                # Expand bounding box by 1 pixel for dilation
                r0 = max(0, sl[0].start - 1)
                r1 = min(h, sl[0].stop + 1)
                c0 = max(0, sl[1].start - 1)
                c1 = min(w, sl[1].stop + 1)
                local_comp = component_map[r0:r1, c0:c1]
                local_labels = labels[r0:r1, c0:c1]
                comp_mask = local_comp == comp_id
                border = binary_dilation(comp_mask, structure) & ~comp_mask
                if border.any():
                    neighbour_labels = local_labels[border]
                    replacement = np.bincount(neighbour_labels).argmax()
                else:
                    replacement = local_labels[comp_mask][0]
                labels[r0:r1, c0:c1][comp_mask] = replacement
    return labels


def quantize_image(
    img: Image.Image,
    n_colors: int,
    median_size: int = 0,
    min_region: int = 0,
    max_sample_pixels: int = 262_144,
) -> tuple[np.ndarray, np.ndarray]:
    import time

    arr = np.array(img)
    h, w, c = arr.shape

    # Downsample for k-means centroid computation
    total_pixels = h * w
    if total_pixels > max_sample_pixels:
        scale = (max_sample_pixels / total_pixels) ** 0.5
        sample_img = img.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.LANCZOS,
        )
        sample_arr = np.array(sample_img).reshape(-1, c).astype(np.float64)
    else:
        sample_arr = arr.reshape(-1, c).astype(np.float64)

    t = time.perf_counter()
    from sklearn.cluster import MiniBatchKMeans

    mbk = MiniBatchKMeans(
        n_clusters=n_colors + 1,
        batch_size=min(4096, len(sample_arr)),
        n_init=3,
        max_iter=100,
    )
    mbk.fit(sample_arr)
    centroids = mbk.cluster_centers_
    print(f"  [quantize] kmeans: {time.perf_counter() - t:.2f}s")

    # Assign every full-res pixel to the nearest centroid (chunked cdist, float32)
    t = time.perf_counter()
    centroids = centroids.astype(np.float32)
    pixels = arr.reshape(-1, c).astype(np.float32)
    n_pixels = pixels.shape[0]
    chunk_size = 500_000
    labels = np.empty(n_pixels, dtype=np.int32)
    for i in range(0, n_pixels, chunk_size):
        chunk = pixels[i : i + chunk_size]
        labels[i : i + chunk_size] = cdist(
            chunk, centroids, metric="sqeuclidean"
        ).argmin(axis=1)
    labels = labels.reshape(h, w)
    print(f"  [quantize] assign labels: {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    labels = clean_labels(
        labels,
        median_size=median_size,
        min_region=min_region,
    )
    print(f"  [quantize] clean_labels: {time.perf_counter() - t:.2f}s")
    return labels, centroids


def extract_color_contours(
    labels: np.ndarray,
    centroids: np.ndarray,
    smooth: float = 0,
    simplify: float = 0,
    no_data: list[tuple[int, ...]] = [],
    buffer: float = 0.5,
    max_contour_pixels: int = 1_000_000,
) -> dict[tuple[int, ...], list[np.ndarray]]:
    h, w = labels.shape
    total_pixels = h * w

    if total_pixels > max_contour_pixels:
        scale = (max_contour_pixels / total_pixels) ** 0.5
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        row_idx = (np.arange(new_h) * h / new_h).astype(int)
        col_idx = (np.arange(new_w) * w / new_w).astype(int)
        work_labels = labels[np.ix_(row_idx, col_idx)]
        scale_y = h / new_h
        scale_x = w / new_w
    else:
        work_labels = labels
        scale_y = 1.0
        scale_x = 1.0

    unique_labels = np.unique(work_labels)
    color_contours: dict[tuple[int, ...], list[np.ndarray]] = {}
    for lbl in unique_labels:
        color = tuple(centroids[lbl].astype(np.uint8).tolist())
        if color[:3] in no_data or color in no_data:
            color = (color[0], color[1], color[2], 0)
        mask = (work_labels == lbl).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        # Scale contours back to original image coordinates
        if scale_y != 1.0 or scale_x != 1.0:
            contours = [c * np.array([[scale_y, scale_x]]) for c in contours]
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
