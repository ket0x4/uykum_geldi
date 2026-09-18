import argparse
import contextlib
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def auto_device(requested: str = "auto") -> str:
    if requested in ("mps", "cuda", "cpu"):
        return requested
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def play_alarm(sound_path: str):
    p = Path(sound_path)
    if not p.is_file():
        p = Path(__file__).resolve().parent.parent.parent / sound_path
    if not p.is_file():
        return
    cmd = (
        ["afplay", str(p)]
        if sys.platform == "darwin"
        else ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(p)]
    )
    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        print("\a", end="", flush=True)


def send_notification(msg: str):
    if sys.platform == "darwin":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    f'display notification "{msg}" with title "Uykum Geldi"',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def load_config(path: str | None) -> dict:
    target = Path(path) if path else Path("config.toml")
    if target.is_file():
        with contextlib.suppress(OSError, tomllib.TOMLDecodeError):
            with open(target, "rb") as f:
                data = tomllib.load(f)
            flat = {}
            for v in data.values():
                if isinstance(v, dict):
                    flat.update(v)
            flat.update({k: v for k, v in data.items() if not isinstance(v, dict)})
            return flat
    return {}


def main():
    parser = argparse.ArgumentParser(description="Uykum Geldi - Human detection alarm")
    parser.add_argument(
        "-s", "--source", default="0", help="Camera index or video file"
    )
    parser.add_argument("-m", "--model", default="yolo11n.pt", help="YOLO model path")
    parser.add_argument(
        "-c", "--conf", type=float, default=0.40, help="Confidence threshold"
    )
    parser.add_argument(
        "-d", "--device", default="auto", help="Compute device: auto, mps, cuda, cpu"
    )
    parser.add_argument(
        "--cooldown", type=float, default=3.0, help="Alarm cooldown in seconds"
    )
    parser.add_argument("--sound", default="alarm.mp3", help="Path to alarm audio file")
    parser.add_argument(
        "--headless", action="store_true", help="Run without GUI window"
    )
    parser.add_argument(
        "--no-notify", action="store_true", help="Disable desktop notifications"
    )
    parser.add_argument("--fps-limit", type=float, default=None, help="Cap maximum FPS")
    parser.add_argument(
        "--skip-frames", type=int, default=0, help="Skip N frames between inferences"
    )
    parser.add_argument("--config", default=None, help="Path to TOML config file")

    args = parser.parse_args()
    cfg = load_config(args.config)
    for k, v in cfg.items():
        if getattr(args, k, None) == parser.get_default(k):
            setattr(args, k, v)

    source = int(args.source) if str(args.source).isdigit() else args.source
    device = auto_device(args.device)

    print(f"Loading {args.model} on {device.upper()}...")
    model = YOLO(args.model)
    try:
        model.to(device)
    except (RuntimeError, ValueError):
        device = "cpu"
        model.to(device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Error: Cannot open video source '{source}'.")

    print(
        f"Monitoring on {device.upper()} ({'Headless' if args.headless else 'GUI: press q to exit'})..."
    )

    last_alert = 0.0
    frame_idx = 0
    has_person = False
    annotated = None

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.skip_frames == 0 or (frame_idx % (args.skip_frames + 1) == 0):
                res = model.predict(
                    frame, conf=args.conf, device=device, verbose=False
                )[0]
                has_person = (
                    bool((res.boxes.cls == 0).any())
                    if res.boxes is not None and len(res.boxes) > 0
                    else False
                )
                if not args.headless:
                    annotated = res.plot()

            now = time.time()
            if has_person and (now - last_alert >= args.cooldown):
                last_alert = now
                print(f"[{time.strftime('%H:%M:%S')}] ALERT: Person detected!")
                play_alarm(args.sound)
                if not args.no_notify:
                    send_notification("Person detected approaching!")

            if not args.headless:
                display = annotated if annotated is not None else frame
                cv2.imshow("Uykum Geldi", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.fps_limit and args.fps_limit > 0:
                elapsed = time.time() - t0
                wait = (1.0 / args.fps_limit) - elapsed
                if wait > 0:
                    time.sleep(wait)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("\nStopped.")


if __name__ == "__main__":
    main()
