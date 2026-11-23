import numpy as np
from fedsight.attributes import dominant_colour
from fedsight.quality import appearance_attributes


def test_hsv_colours_on_synthetic_clothing_regions():
    crop = np.zeros((100, 50, 3), dtype=np.uint8)
    crop[:50] = [255, 0, 0]
    crop[50:] = [0, 0, 255]
    assert dominant_colour(crop[:40]) == "red"
    assert dominant_colour(crop[60:]) == "blue"
    attributes = appearance_attributes(crop)
    assert attributes["upper_color"] == "red"
    assert attributes["lower_color"] == "blue"
    assert dominant_colour(np.empty((0, 0, 3), dtype=np.uint8)) == "unknown"
