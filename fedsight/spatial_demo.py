"""Synthetic floor plan and homography calibration fixtures for FedSight."""
from __future__ import annotations

import io
from PIL import Image, ImageDraw, ImageFont

def create_demo_floor_plan(service, case_id: str, actor: str = "demo-reviewer") -> dict:
    """Generate a clean architectural schematic floor plan and store it in the vault.

    Returns the asset dict.
    """
    w, h = 800, 450
    img = Image.new("RGB", (w, h), "#0d1b26")
    draw = ImageDraw.Draw(img)

    # Grid lines (light background)
    for x in range(0, w, 40):
        draw.line([(x, 0), (x, h)], fill="#142634", width=1)
    for y in range(0, h, 40):
        draw.line([(0, y), (w, y)], fill="#142634", width=1)

    # Outer wall border
    draw.rectangle([30, 30, w - 30, h - 30], outline="#29465a", width=2)

    # Room 1: Atrium (west)
    draw.rectangle([50, 50, 260, 390], outline="#3b6982", fill="#102535", width=2)
    draw.text((70, 65), "ZONE A · ATRIUM", fill="#58ddd5")
    draw.text((70, 85), "Camera 0 Coverage", fill="#88a0b0")

    # Room 2: Corridor (centre hallway)
    draw.rectangle([260, 160, 540, 280], outline="#3b6982", fill="#132a3d", width=2)
    draw.text((320, 175), "ZONE B · MAIN CORRIDOR", fill="#58ddd5")
    draw.text((320, 195), "Camera 1 Coverage", fill="#88a0b0")

    # Room 3: Exit Lobby (east)
    draw.rectangle([540, 50, 750, 390], outline="#3b6982", fill="#102535", width=2)
    draw.text((560, 65), "ZONE C · SECURE EXIT", fill="#58ddd5")
    draw.text((560, 85), "Camera 2 Coverage", fill="#88a0b0")

    # Doorways (openings)
    draw.line([(260, 180), (260, 260)], fill="#58ddd5", width=3)  # Atrium to Corridor
    draw.line([(540, 180), (540, 260)], fill="#58ddd5", width=3)  # Corridor to Exit

    # Compass / Scale marker
    draw.text((w - 180, h - 50), "SCALE: 1m = 20px (40m × 22.5m)", fill="#7890a0")
    draw.line([(w - 180, h - 35), (w - 80, h - 35)], fill="#58ddd5", width=2)
    draw.text((w - 135, h - 30), "5m", fill="#58ddd5")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    return service.save_asset(png_bytes, case_id, actor, role="floor_plan", parent_id=None,
                              filename="synthetic_facility_floor_plan.png", width=w, height=h)


def setup_demo_spatial(service, case_id: str, cameras: list[dict],
                       sources: list[dict], actor: str = "demo-reviewer"):
    """Calibrate synthetic cameras against the demo floor plan."""
    floor_asset = create_demo_floor_plan(service, case_id, actor)

    # 4 video corners in 640x360 coordinates
    video_pts = [[60, 60], [580, 60], [580, 300], [60, 300]]

    # Corresponding floor coordinates (top-left, top-right, bottom-right, bottom-left)
    camera_floor_boxes = [
        # Atrium
        [[70, 70], [240, 70], [240, 370], [70, 370]],
        # Corridor
        [[280, 175], [520, 175], [520, 265], [280, 265]],
        # Exit
        [[560, 70], [730, 70], [730, 370], [560, 370]],
    ]

    for camera, source, floor_pts in zip(cameras, sources, camera_floor_boxes):
        service.add_floor_calibration(
            case_id,
            actor,
            source_id=source["id"],
            camera_id=camera["id"],
            video_points=video_pts,
            floor_points=floor_pts,
            floor_plan_asset=floor_asset,
            pixels_per_metre=20.0,
            provenance="Synthetic facility layout calibration (20 px/m)",
        )
