"""Print local video metadata without exporting decrypted evidence."""
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
import cv2
from fedsight.service import FedSight


def main(case_title):
    service = FedSight(Path("data/workbench"))
    case = next(c for c in service.repo.list(None, "case") if c["title"] == case_title)
    for source in service.repo.list(case["id"], "source"):
        path = None
        try:
            with NamedTemporaryFile(dir=service.work, suffix=source["suffix"], delete=False) as temp:
                temp.write(service.read_asset(source["asset"], "metadata-check"))
                path = Path(temp.name)
            cap = cv2.VideoCapture(str(path))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                print(source["name"], "fps=", fps, "frames=", frames,
                      "duration_s=", frames / fps if fps else "unknown",
                      "size=", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                cap.release()
        finally:
            if path:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1])
