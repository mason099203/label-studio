import json
import pytest

from training.datasets import (
    _default_export_format_for_task,
    _get_first_input_data_key,
    _get_pose_kpt_shape,
    _rectangle_labels_use_obb,
    detect_training_interface,
    resolve_deploy_task_context,
)


class StubProject:
    def __init__(self, parsed_config, label_config=""):
        self.parsed_config = parsed_config
        self.label_config = label_config

    def get_parsed_config(self):
        return self.parsed_config


def test_get_first_input_data_key_accepts_dollar_prefixed_value():
    info = {"inputs": [{"type": "Image", "value": "$image"}]}
    assert _get_first_input_data_key(info) == "image"


def test_get_first_input_data_key_accepts_plain_value():
    info = {"inputs": [{"type": "Image", "value": "image"}]}
    assert _get_first_input_data_key(info) == "image"


def test_detect_training_interface_accepts_plain_image_value_for_choices():
    project = StubProject(
        {
            "choice": {
                "type": "Choices",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "image"}],
                "labels": ["Adult content", "Weapons", "Violence"],
            }
        }
    )

    assert detect_training_interface(project) == {
        "task_type": "classification",
        "training_model": "yolo_classify",
        "control_name": "choice",
        "data_key": "image",
        "labels": ["Adult content", "Weapons", "Violence"],
    }


def test_detect_training_interface_detects_rectangle_detect():
    project = StubProject(
        {
            "label": {
                "type": "RectangleLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["cat"],
            }
        }
    )

    spec = detect_training_interface(project)
    assert spec["task_type"] == "detect"
    assert spec["training_model"] == "yolo_detect"


def test_detect_training_interface_detects_polygon_segmentation():
    project = StubProject(
        {
            "label": {
                "type": "PolygonLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["cat"],
            }
        }
    )

    spec = detect_training_interface(project)
    assert spec["task_type"] == "segmentation"
    assert spec["training_model"] == "yolo_segment"


def test_detect_training_interface_detects_pose():
    project = StubProject(
        {
            "bbox": {
                "type": "RectangleLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["person"],
            },
            "kp": {
                "type": "KeyPointLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["nose", "eye"],
                "labels_attrs": {
                    "nose": {"model_index": "0"},
                    "eye": {"model_index": "1"},
                },
            },
        }
    )

    spec = detect_training_interface(project)
    assert spec["task_type"] == "pose"
    assert spec["training_model"] == "yolo_pose"
    assert spec["keypoint_control_name"] == "kp"


def test_detect_training_interface_detects_obb_from_model_obb_attr():
    label_config = """<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image" model_obb="true">
    <Label value="ship"/>
  </RectangleLabels>
</View>"""
    project = StubProject(
        {
            "label": {
                "type": "RectangleLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["ship"],
            }
        },
        label_config=label_config,
    )

    spec = detect_training_interface(project)
    assert spec["task_type"] == "obb"
    assert spec["training_model"] == "yolo_obb"


def test_detect_training_interface_rejects_pose_without_rectangle():
    project = StubProject(
        {
            "kp": {
                "type": "KeyPointLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["nose"],
            }
        }
    )

    with pytest.raises(ValueError, match="RectangleLabels"):
        detect_training_interface(project)


def test_detect_training_interface_detects_brush_semantic_segmentation():
    project = StubProject(
        {
            "label": {
                "type": "BrushLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["cat", "dog"],
            }
        }
    )

    spec = detect_training_interface(project)
    assert spec["task_type"] == "semantic_segmentation"
    assert spec["training_model"] == "yolo_semantic"
    assert spec["labels"] == ["cat", "dog"]


def test_get_pose_kpt_shape_uses_model_index():
    parsed = {
        "kp": {
            "type": "KeyPointLabels",
            "labels": ["a", "b", "c"],
            "labels_attrs": {
                "a": {"model_index": "0"},
                "b": {"model_index": "2"},
            },
        }
    }

    assert _get_pose_kpt_shape(parsed) == (3, 3)


def test_default_export_format_for_task():
    assert _default_export_format_for_task("detect") == "YOLO_WITH_IMAGES"
    assert _default_export_format_for_task("segmentation") == "YOLO_WITH_IMAGES"
    assert _default_export_format_for_task("semantic_segmentation") == "JSON_MIN"
    assert _default_export_format_for_task("pose") == "YOLO_WITH_IMAGES"
    assert _default_export_format_for_task("obb") == "YOLO_OBB_WITH_IMAGES"
    assert _default_export_format_for_task("classification") == "JSON_MIN"


def test_rectangle_labels_use_obb():
    project = StubProject({}, label_config='<RectangleLabels model_obb="true"></RectangleLabels>')
    assert _rectangle_labels_use_obb(project) is True
    project = StubProject({}, label_config='<RectangleLabels></RectangleLabels>')
    assert _rectangle_labels_use_obb(project) is False


def test_resolve_deploy_task_context_from_dataset_meta(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dataset_config = run_dir / "dataset_config.json"
    dataset_config.write_text(
        json.dumps(
            {
                "training_model": "yolo_segment",
                "task_type": "segmentation",
            }
        ),
        encoding="utf-8",
    )
    run_meta = {
        "params": {
            "dataset_config": str(dataset_config),
        }
    }
    project = StubProject(
        {
            "label": {
                "type": "PolygonLabels",
                "to_name": ["image"],
                "inputs": [{"type": "Image", "value": "$image"}],
                "labels": ["cat"],
            }
        }
    )
    ctx = resolve_deploy_task_context(project, run_dir, run_meta)
    assert ctx["task_type"] == "segmentation"
    assert ctx["training_model"] == "yolo_segment"
    assert ctx["default_imgsz"] == 640
    assert ctx["is_cnn"] is False


def test_validate_yolo_training_rejects_cls_weights_with_yaml_dataset():
    from training.yolo_catalog import validate_yolo_training_request

    meta = {
        "task_type": "detect",
        "training_model": "yolo_detect",
        "dataset_root": "/tmp/ds",
        "data_yaml": "/tmp/ds/data.yaml",
    }
    err = validate_yolo_training_request(base_weights="yolo11n-cls.pt", dataset_meta=meta)
    assert err is not None
    assert "data.yaml" in err


def test_validate_yolo_training_accepts_detect_weights_with_yaml_dataset(tmp_path):
    from training.yolo_catalog import validate_yolo_training_request

    ds = tmp_path / "ds"
    ds.mkdir()
    (ds / "data.yaml").write_text("path: .\n", encoding="utf-8")
    meta = {
        "task_type": "detect",
        "training_model": "yolo_detect",
        "dataset_root": str(ds),
        "data_yaml": str(ds / "data.yaml"),
    }
    assert validate_yolo_training_request(base_weights="yolo11n.pt", dataset_meta=meta) is None


def test_validate_yolo_training_accepts_cls_weights_with_classify_layout(tmp_path):
    from training.yolo_catalog import validate_yolo_training_request

    ds = tmp_path / "cls"
    train = ds / "train" / "cat"
    val = ds / "val" / "cat"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    (train / "1.jpg").write_bytes(b"x")
    (val / "2.jpg").write_bytes(b"x")
    meta = {
        "task_type": "classification",
        "training_model": "yolo_classify",
        "dataset_root": str(ds),
        "train_dir": str(ds / "train"),
        "val_dir": str(ds / "val"),
    }
    assert validate_yolo_training_request(base_weights="yolo11n-cls.pt", dataset_meta=meta) is None
