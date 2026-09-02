"""What ``to_strict_json_schema`` must keep doing to a Pydantic model.

This transform had no direct test, which mattered more than it looks: the
ingestion tests stub ``LLMClient.extract`` outright, so the schema built at the
call site was never once exercised in CI. A change that broke it would ship
green and then fail on every real posting with a 400.

The two transforms are not interchangeable with anything a framework offers.
Closing objects and filling ``required`` is standard strict-mode hygiene and
several libraries do it. Inlining ``$ref`` is not standard — it exists because
Groq's validator does not resolve refs, so ``SomeModel | None`` arrives as
``anyOf: [{$ref}, {type: null}]`` and is rejected as ambiguous. Nothing in
LangChain does this, so the transform has to survive any move onto it.
"""

from typing import Any

import pytest
from pydantic import BaseModel, Field

from app.schemas.extraction import ExtractedJob, to_strict_json_schema
from app.schemas.matching import RubricJudgment

SCHEMAS = [ExtractedJob, RubricJudgment]


def walk(node: Any) -> list[Any]:
    """Every dict and list in the tree, including the root."""
    found = [node]
    if isinstance(node, dict):
        for value in node.values():
            found.extend(walk(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(walk(item))
    return found


def objects(schema: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in walk(schema)
        if isinstance(node, dict) and node.get("type") == "object" and "properties" in node
    ]


@pytest.mark.parametrize("model", SCHEMAS, ids=lambda m: m.__name__)
def test_no_ref_or_defs_survives_anywhere(model: type[BaseModel]) -> None:
    """The whole point of the transform, asserted at every depth.

    Not just ``"$defs" not in schema``: a ref nested inside an array's items or
    an anyOf branch is exactly the case Groq rejects, and exactly the case a
    shallow check misses.
    """
    schema = to_strict_json_schema(model)

    assert "$defs" not in schema
    for node in walk(schema):
        if isinstance(node, dict):
            assert "$ref" not in node, f"unresolved ref: {node}"


@pytest.mark.parametrize("model", SCHEMAS, ids=lambda m: m.__name__)
def test_every_object_is_closed_and_fully_required(model: type[BaseModel]) -> None:
    schema = to_strict_json_schema(model)
    found = objects(schema)

    assert found, "expected at least the root object"
    for node in found:
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])


def test_the_nested_salary_model_is_inlined_not_referenced() -> None:
    """``ExtractedJob.salary`` is a model, so Pydantic emits it as a ``$ref``.

    Naming the field explicitly rather than trusting the generic walk above:
    this is the concrete case that breaks if the transform is ever dropped in
    favour of a framework's own schema conversion.
    """
    salary = to_strict_json_schema(ExtractedJob)["properties"]["salary"]

    assert salary["type"] == "object"
    assert "raw_text" in salary["properties"]
    assert salary["additionalProperties"] is False
    assert salary["required"] == list(salary["properties"])


def test_a_model_inside_a_list_is_inlined_too() -> None:
    """``requirements`` is ``list[ExtractedRequirement]`` — the ref is in `items`."""
    requirements = to_strict_json_schema(ExtractedJob)["properties"]["requirements"]

    assert requirements["type"] == "array"
    assert requirements["items"]["properties"]["kind"]["enum"] == ["must", "nice"]
    assert requirements["items"]["additionalProperties"] is False


def test_keys_alongside_a_ref_win_over_the_definition() -> None:
    """Pydantic puts a field's description next to the ``$ref``, not inside it.

    Merging the other way round would silently drop every description on a
    nested model field — no error, just a worse extraction.
    """

    class Inner(BaseModel):
        value: str = Field(description="from the definition")

    class Outer(BaseModel):
        inner: Inner = Field(description="from the field")

    inner = to_strict_json_schema(Outer)["properties"]["inner"]

    assert inner["description"] == "from the field"
    assert inner["properties"]["value"]["description"] == "from the definition"


def test_a_self_referencing_model_fails_loudly() -> None:
    """The depth bound, which is what stands in for cycle detection.

    A recursive schema would otherwise inline forever and hang the request
    rather than fail it.
    """

    class Node(BaseModel):
        child: "Node | None"

    with pytest.raises(ValueError, match="self-referencing"):
        to_strict_json_schema(Node)
