from pathlib import Path

from ollaya_ids.data import load_ereno


def test_mixed_csv_and_arff_sections_are_loaded_explicitly(tmp_path: Path) -> None:
    dataset = tmp_path / "mixed.csv"
    dataset.write_text(
        "x,kind,class\n"
        "1,a,normal\n"
        "2,b,attack\n"
        "2,b,attack\n"
        "3,c,normal\n"
        "3,c,attack\n"
        "@relation sample\n"
        "@attribute x numeric\n"
        "@attribute kind {a,b}\n"
        "@attribute @class@ {normal,attack}\n"
        "@data\n"
        "4,a,normal\n"
        "5,b,attack\n",
        encoding="utf-8",
    )

    csv_frame, csv_profile = load_ereno(dataset, section="csv")
    arff_frame, arff_profile = load_ereno(dataset, section="arff")
    all_frame, all_profile = load_ereno(dataset, section="all")

    assert len(csv_frame) == 2
    assert csv_profile.duplicate_rows_removed == 1
    assert csv_profile.conflicting_vectors_removed == 2
    assert len(arff_frame) == 2
    assert arff_profile.class_counts == {"attack": 1, "normal": 1}
    assert len(all_frame) == 4
    assert all_profile.rows_read == 7


def test_numeric_columns_are_converted(tmp_path: Path) -> None:
    dataset = tmp_path / "numeric.csv"
    dataset.write_text("x,kind,class\n1,a,normal\n2,b,attack\n", encoding="utf-8")

    frame, _ = load_ereno(dataset)

    assert frame["x"].dtype.kind in "ifu"
    assert frame["kind"].dtype == object
