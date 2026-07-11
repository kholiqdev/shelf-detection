import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_NOTEBOOK = ROOT / "notebooks/train_shelf_detection_colab.ipynb"
POC_NOTEBOOK = ROOT / "notebooks/poc_shelf_detection_colab.ipynb"


def load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_source(path: Path) -> str:
    notebook = load_notebook(path)
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_notebooks_use_shelf_detection_identity() -> None:
    training_source = notebook_source(TRAIN_NOTEBOOK)
    poc_source = notebook_source(POC_NOTEBOOK)
    all_sources = training_source + "\n" + poc_source

    for stale_value in ("ShelfScan", "ShelfScanAI", "VahantSharma", "shelfscan"):
        assert stale_value not in all_sources

    assert "Shelf Detection" in training_source
    assert "Shelf Detection" in poc_source
    assert "https://github.com/kholiqdev/shelf-detection.git" in training_source
    assert 'REPO_DIR = Path("/content/shelf-detection")' in training_source
    assert '/content/drive/MyDrive/shelf-detection' in all_sources
    assert 'RUN_NAME = "shelf-detection-colab-v1"' in training_source


def test_training_uses_official_sku110k_dataset() -> None:
    source = notebook_source(TRAIN_NOTEBOOK)

    assert 'DATASET_NAME = "SKU-110K.yaml"' in source
    assert "check_det_dataset(DATASET_NAME)" in source
    assert 'YOLO("yolov8m.pt").train(' in source
    assert "data=DATASET_NAME" in source
    assert "DATASET_ZIP" not in source
    assert "SKU110K.zip" not in source
    assert "src.data.prepare" not in source
    assert "dataset_target.symlink_to" not in source
    assert "training/train.py" not in source
    assert "configs/training.yaml" not in source


def test_poc_evaluates_official_sku110k_dataset() -> None:
    source = notebook_source(POC_NOTEBOOK)

    assert 'DATASET_NAME = "SKU-110K.yaml"' in source
    assert "data=DATASET_NAME" in source
    assert '"dataset": DATASET_NAME' in source
    assert "DATASET_ZIP" not in source
    assert "SKU110K.zip" not in source
    assert "evaluation_yaml.write_text" not in source


def test_all_code_cells_compile() -> None:
    for notebook_path in (TRAIN_NOTEBOOK, POC_NOTEBOOK):
        notebook = load_notebook(notebook_path)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, f"{notebook_path.name}:cell-{index}", "exec")
