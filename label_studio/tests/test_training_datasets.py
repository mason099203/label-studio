from training.datasets import _get_first_input_data_key, detect_training_interface


class StubProject:
    def __init__(self, parsed_config):
        self.parsed_config = parsed_config

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
