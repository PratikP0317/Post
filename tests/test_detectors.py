from open_vocab_track.detectors.florence2 import Florence2Detector
from open_vocab_track.detectors.locate_anything import LocateAnythingDetector
from open_vocab_track.detectors.moondream import MoondreamDetector


def test_moondream_normalized_boxes_are_scaled():
    detections = MoondreamDetector.parse_objects(
        [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.8}], 1000, 500, "green shirt"
    )
    assert detections[0].xyxy == (100.0, 100.0, 400.0, 400.0)


def test_florence_phrase_grounding_result():
    result = {"<CAPTION_TO_PHRASE_GROUNDING>": {"bboxes": [[1, 2, 30, 40]], "labels": ["person"]}}
    detections = Florence2Detector.parse_result(result, "green shirt")
    assert detections[0].xyxy == (1.0, 2.0, 30.0, 40.0)
    assert detections[0].label == "person"


def test_locate_anything_special_tokens_are_scaled_and_invalid_removed():
    answer = "<ref>person</ref><box><100><200><400><800></box><box>none</box>"
    detections = LocateAnythingDetector.parse_answer(answer, 1000, 500, "green shirt")
    assert detections[0].xyxy == (100.0, 100.0, 400.0, 400.0)
