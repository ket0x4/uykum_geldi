# Uykum Geldi

> *OpenCV & YOLO Human Detection Alarm — Sleep in peace at work*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/model-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/models/yolo11/)
[![Accelerated by MPS](https://img.shields.io/badge/Apple%20Silicon-MPS%20Ready-orange.svg)]()

`uykum-geldi` watches your webcam (or video feed) in real time using YOLO object detection. When a person enters the camera's view, it immediately plays an alarm audio alert and desktop notifications.

---

## Quickstart

### Using `uv`

```bash
# Clone the repository
git clone https://github.com/ket0x4/uykum_geldi.git
cd uykum_geldi

# Create environment and install dependencies
uv venv --python 3.12
uv pip install -e .

# Run monitoring with default webcam (0)
uv run uykum-geldi
```

> **macOS Note**: Ensure Terminal (or your IDE) has camera permissions under **System Settings → Privacy & Security → Camera**.

---

## CLI Usage & Options

Run the CLI help command anytime:

```bash
uykum-geldi --help
```

### Options & Flags

| Flag | Short | Default | Description |
|---|---|---|---|
| `--source` | `-s` | `0` | Camera device index (e.g. `0`, `1`) or video file path / RTSP URL |
| `--model` | `-m` | `yolo11n.pt` | YOLO model name or path (`yolo11n.pt`, `yolo11s.pt`, etc.) |
| `--conf` | `-c` | `0.40` | Minimum confidence threshold for human detection (0.0 to 1.0) |
| `--device` | `-d` | `auto` | Device acceleration: `auto`, `mps` (Apple Silicon), `cuda`, or `cpu` |
| `--cooldown` | | `3.0` | Cooldown period in seconds between alarm triggers |
| `--sound` | | `alarm.mp3` | Path to alarm audio file |
| `--headless` | | `False` | Run in terminal without opening an OpenCV GUI window |
| `--no-notify` | | `False` | Disable OS desktop notifications |
| `--fps-limit` | | `None` | Cap maximum FPS to conserve battery (e.g. `--fps-limit 15`) |
| `--skip-frames` | | `0` | Skip N frames between YOLO inferences (e.g. `--skip-frames 2`) |
| `--config` | | `None` | Path to a custom TOML configuration file |

---

## Configuration File (`config.toml`)

You can create an optional `config.toml` file in the current directory:

```toml
source = 0
model = "yolo11n.pt"
conf = 0.40
device = "auto"
cooldown = 3.0
sound = "alarm.mp3"
headless = false
fps_limit = 15.0
```

Any CLI flag passed on the command line automatically overrides the values in `config.toml`.

---

## GUI Keyboard Shortcuts

When running in GUI mode:
- Press **`q`** or **`ESC`** to cleanly stop monitoring and close all windows.

---

## License

MIT License. Feel free to use and adapt!
