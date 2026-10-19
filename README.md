# 3DGS Measurement Tool

Browser viewer for 3D Gaussian Splat `.ply` files. Click two points to measure the distance between them.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python viewer.py --scene-path path/to/splat_30000.ply
```

Open http://localhost:8080 in a browser.

## Use

- **Click** a point in the scene → red marker drops on it (Point A)
- **Click** a second point → Point B, plus a line and the **Distance** shown in the side panel
- A third click starts a new pair; **Clear points** resets
- **View** folder: roll the camera ±20°
- **Settings** folder: marker size, pick sensitivity (raise it if clicks miss)

Distances are in the scene's units (whatever the reconstruction used, not necessarily meters).

## Options

| Flag | Default | What it does |
|---|---|---|
| `--scene-path` | required | `.ply` file to load |
| `--port` | `8080` | Server port |
| `--pick-angle-deg` | `0.5` | Click tolerance; bigger = easier to hit, less precise |
| `--marker-radius` | auto | Marker size in scene units |
