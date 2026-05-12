from enum import Enum

from pydantic import BaseModel, create_model


class BoundingBox(BaseModel):
    """Represents a bounding box with its 2D coordinates and associated label.

    Attributes:
        box_2d: Bounding box coordinates in the format [y_min, x_min, y_max, x_max].
        label: Label associated with the detected object.
    """

    box_2d: list[int]
    label: str


def parse_structured_output_spec(
    spec: str, ask_bounding_box: bool = False,
) -> type[BaseModel]:
    """Parse a structured output spec string and return a Pydantic model.

    Supported formats:
        - "bool"              -> model with a single `answer: bool` field
        - "enum=A,B,C"        -> model with a single `answer: Enum("A","B","C")` field
        - "enums=A,B,C"       -> model with `answer: list[Enum("A","B","C")]`

    When ask_bounding_box is True, a bounding_box (single enum) or
    bounding_boxes (list/bool) field is added to the model.
    """
    spec = spec.strip()

    if spec.lower() == "bool":
        fields: dict = {
            "answer": (bool, ...),
            "source": (str, ...),
        }
        if ask_bounding_box:
            fields["bounding_boxes"] = (list[BoundingBox], ...)
        return create_model("BoolAnswer", **fields)

    if spec.startswith("enum=") or spec.startswith("enums="):
        multi = spec.startswith("enums=")
        prefix_len = 6 if multi else 5
        raw_values = spec[prefix_len:]
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not values:
            raise ValueError(f"No enum values found in spec: {spec!r}")

        enum_cls = Enum("AnswerEnum", {v: v for v in values})
        answer_type = list[enum_cls] if multi else enum_cls
        fields = {
            "answer": (answer_type, ...),
            "source": (str, ...),
        }
        if ask_bounding_box:
            fields["bounding_box"] = (BoundingBox, ...)
        model_name = "EnumsAnswer" if multi else "EnumAnswer"
        return create_model(model_name, **fields)

    raise ValueError(f"Unknown structured output spec: {spec!r}")
