![logo for vectorify](https://github.com/marzukia/vectorify/raw/main/docs/logo.png)

Convert raster images (PNG, JPG, etc.) into simplified SVG vector graphics using k-means colour quantization and contour tracing.

## Installation

Requires Python 3.11+.

```bash
uv sync
```

## Usage

```bash
uv run vectorify generate <filepath> <output> [options]
```

### Example

```bash
uv run vectorify generate photo.png ./output.svg --n_colors 8 --smooth 2 --simplify 0.5
```

### Update README images

Regenerate all example SVGs, PNGs, and grid images used in this README:

```bash
uv run vectorify update-readme
```

### Arguments

| Argument   | Type   | Description                  |
| ---------- | ------ | ---------------------------- |
| `filepath` | string | Path to the input image      |
| `out_dir`  | string | Directory for the output SVG |

### Options

| Option          | Type  | Default | Description                                                  |
| --------------- | ----- | ------- | ------------------------------------------------------------ |
| `--n_colors`    | int   | 8       | Number of colours to quantize the image into                 |
| `--median_size` | float | 5       | Median filter kernel size for label smoothing (0 = disabled) |
| `--min_region`  | float | 50      | Minimum connected region size in pixels (0 = disabled)       |
| `--smooth`      | float | 2       | Gaussian smoothing sigma applied to contours (0 = disabled)  |
| `--simplify`    | float | 0.5     | RDP simplification epsilon in pixels (0 = disabled)          |
| `--buffer`      | float | 1       | Outward normal offset to expand shapes and fill gaps         |

## Configuration

### `buffer`

Expands each shape outward along its edge normals by the given pixel amount. Eliminates sub-pixel gaps between adjacent colour regions caused by smoothing or simplification.

![buffer example](https://github.com/marzukia/vectorify/raw/main/docs/img/buffer.png)

---

### `median_size`

Applies a median filter of this kernel size to the label map. Fills small holes, removes straggler pixels, and smooths region boundaries. Larger values produce more aggressive cleanup.

![median_size example](https://github.com/marzukia/vectorify/raw/main/docs/img/median_size.png)

---

### `min_region`

Removes connected components smaller than this many pixels, replacing them with their most common neighbouring colour. Catches isolated specks that the median filter misses.

![min_region example](https://github.com/marzukia/vectorify/raw/main/docs/img/min_region.png)

---

### `n_colors`

Number of distinct colours the image is quantized into via k-means clustering. Lower values produce a more simplified, poster-like result.

![n_colors example](https://github.com/marzukia/vectorify/raw/main/docs/img/n_colors.png)

---

### `simplify`

Epsilon value for the Ramer-Douglas-Peucker algorithm. Reduces the number of vertices in each contour path. Higher values = fewer vertices = simpler shapes.

![simplify example](https://github.com/marzukia/vectorify/raw/main/docs/img/simplify.png)

---

### `smooth`

Gaussian smoothing sigma applied to contour vertices. Rounds off jagged pixel-aligned edges into flowing curves. Applied before simplification.

![smooth example](https://github.com/marzukia/vectorify/raw/main/docs/img/smooth.png)
