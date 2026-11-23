"""Create a small offline synthetic video for testing the upload pipeline."""
from pathlib import Path
import argparse
import cv2
import numpy as np


def create_sample(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    if not writer.isOpened():
        raise OSError("MP4 encoder is unavailable.")
    try:
        for frame in range(200):
            image = np.full((360, 640, 3), (42, 49, 52), dtype=np.uint8)
            cv2.putText(image, "SYNTHETIC DEMO", (15, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (230, 230, 230), 2)
            cv2.line(image, (320, 45), (320, 340), (100, 155, 155), 2)
            for offset, colour, y in ((0, (40, 60, 220), 140), (130, (210, 100, 35), 225)):
                x = 40 + ((frame * 3 + offset) % 550)
                cv2.circle(image, (x, y - 45), 12, (180, 190, 200), -1)
                cv2.rectangle(image, (x - 14, y - 30), (x + 14, y + 25), colour, -1)
                cv2.line(image, (x - 7, y + 25), (x - 10, y + 55), (210, 210, 210), 5)
                cv2.line(image, (x + 7, y + 25), (x + 10, y + 55), (210, 210, 210), 5)
            writer.write(image)
    finally:
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="data/synthetic-sample.mp4")
    args = parser.parse_args()
    create_sample(Path(args.output))
