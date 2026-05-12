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

    if spec.startswith("enum="):
        raw_values = spec[5:]  # everything after "enum="
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not values:
            raise ValueError(f"No enum values found in spec: {spec!r}")

        enum_cls = Enum("AnswerEnum", {v: v for v in values})
        fields = {
            "answer": (enum_cls, ...),
            "source": (str, ...),
        }
        if ask_bounding_box:
            fields["bounding_box"] = (BoundingBox, ...)
        return create_model("EnumAnswer", **fields)

    raise ValueError(f"Unknown structured output spec: {spec!r}")
