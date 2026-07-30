from pathlib import Path

from zenreg import iter_bids_like_image_files


def test_iter_bids_like_image_files_discovers_supported_images(tmp_path):
    image_path = tmp_path / "ID000001" / "TP000" / "image_01.ome.tif"
    image_path.parent.mkdir(parents=True)
    image_path.write_text("dummy")
    ignored = tmp_path / "ID000001" / "TP000" / "notes.txt"
    ignored.write_text("not an image")

    records = iter_bids_like_image_files(tmp_path)

    assert len(records) == 1
    assert records[0].subject_id == "ID000001"
    assert records[0].experiment_tag == "TP000"
    assert records[0].image_path == image_path


def test_iter_bids_like_image_files_respects_requested_subjects_and_experiments(tmp_path):
    for subject in ("ID000001", "ID000002"):
        for experiment in ("TP000", "TP001"):
            folder = tmp_path / subject / experiment
            folder.mkdir(parents=True)
            (folder / "image_01.czi").write_text("dummy")

    records = iter_bids_like_image_files(
        tmp_path,
        subject_ids=("ID000002",),
        experiment_tags=("TP001",),
    )

    assert [record.image_path for record in records] == [
        Path(tmp_path / "ID000002" / "TP001" / "image_01.czi")
    ]
