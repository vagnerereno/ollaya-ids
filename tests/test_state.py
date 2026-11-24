import pandas as pd

from ollaya_ids.state import labeled_request, row_to_state


def test_state_rounds_floats_and_keeps_identifiers() -> None:
    row = pd.Series({"row_id": "csv:1", "x": 1.23456789, "kind": " GOOSE ", "class": "normal"})

    state = row_to_state(row, ["x", "kind"], precision=4)
    request = labeled_request(row, ["x", "kind"], "class", precision=4)

    assert state == {"x": 1.2346, "kind": "GOOSE"}
    assert request["questions"]["attack_type"]["label"] == "normal"
    assert request["_meta"]["id"] == "csv:1"
