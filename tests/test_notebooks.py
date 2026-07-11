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

    for stale_value in ("ShelfScan", "ShelfScanAI", "VahantSharma"):
        assert stale_value not in all_sources

    assert "Shelf Detection" in training_source
    assert "Shelf Detection" in poc_source
    assert "https://github.com/kholiqdev/shelf-detection.git" in training_source
    assert "def detect_colab() -> bool:" in all_sources
    assert "IS_COLAB = detect_colab()" in all_sources
    assert 'COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive/shelf-detection")' in all_sources
    assert "LOCAL_PROJECT_ROOT = Path.cwd().resolve()" in all_sources
    assert "DRIVE_ROOT = COLAB_DRIVE_ROOT if IS_COLAB else LOCAL_PROJECT_ROOT" in all_sources
    assert "if IS_COLAB:" in all_sources
    assert 'RUN_NAME = "shelf-detection-colab-v1"' in training_source


def test_training_uses_official_sku110k_dataset() -> None:
    source = notebook_source(TRAIN_NOTEBOOK)

    assert 'DATASET_NAME = "SKU-110K.yaml"' in source
    assert 'MODEL_NAME = "yolov8m.pt"' in source
    assert "DRIVE_SKU110K_DIR = DRIVE_DATASETS_DIR / \"SKU-110K\"" in source
    assert "DRIVE_DATASET_YAML = DRIVE_DATASETS_DIR / \"SKU-110K.drive.yaml\"" in source
    assert 'COLAB_RUNTIME_ROOT = Path("/content")' in source
    assert "RUNTIME_ROOT = COLAB_RUNTIME_ROOT if IS_COLAB else LOCAL_PROJECT_ROOT" in source
    assert 'LOCAL_SKU110K_DIR = RUNTIME_ROOT / "SKU-110K"' in source
    assert "SAVE_SKU110K_TO_DRIVE = True" in source
    assert "RESTORE_SKU110K_FROM_DRIVE = True" in source
    assert "DATASET_FOR_TRAINING" in source
    assert "write_sku110k_yaml(DRIVE_DATASET_YAML, DRIVE_SKU110K_DIR)" in source
    assert '"rsync", "-ah", "--info=progress2"' in source
    assert "yolo26" not in source.lower()
    assert '"ultralytics>=8.4.70"' in source
    assert "check_det_dataset(DATASET_FOR_TRAINING)" in source
    assert "YOLO(MODEL_NAME).train(" in source
    assert "data=DATASET_FOR_TRAINING" in source
    assert "EPOCHS = 100" in source
    assert "BATCH = 8" in source
    assert "WORKERS = 8" in source
    assert "LR0 = 0.01" in source
    assert "LRF = 0.01" in source
    assert "MOMENTUM = 0.937" in source
    assert "WEIGHT_DECAY = 0.0005" in source
    assert "WARMUP_EPOCHS = 3" in source
    assert "PATIENCE = 20" in source
    assert "AMP = True" in source
    assert "CACHE = False" in source
    assert 'PYTORCH_INSTALL_ARGS = ["torch", "torchvision", "torchaudio"]' in source
    assert 'WANDB_PROJECT = "shelfscan"' in source
    assert 'os.environ.get("WANDB_API_KEY")' in source
    assert "from google.colab import userdata" in source
    assert "DATASET_ZIP" not in source
    assert "SKU110K.zip" not in source
    assert "src.data.prepare" not in source
    assert "dataset_target.symlink_to" not in source
    assert "training/train.py" not in source
    assert "configs/training.yaml" not in source


def test_poc_evaluates_sku110k_and_shelfscan_field_sources() -> None:
    source = notebook_source(POC_NOTEBOOK)

    assert 'DATASET_NAME = "SKU-110K.yaml"' in source
    assert '"ultralytics>=8.4.70"' in source
    assert 'PYTORCH_INSTALL_ARGS = ["torch", "torchvision", "torchaudio"]' in source
    assert "data=DATASET_NAME" in source
    assert '"dataset": DATASET_NAME' in source
    assert "RUN_SKU110K_TEST_EVALUATION = False" in source
    assert "RUN_FIELD_SOURCE_EVALUATION = False" in source
    assert 'LOCAL_INPUT_IMAGE_PATH = LOCAL_PROJECT_ROOT / "poc_images/shelf.jpg"' in source
    assert "google.colab import files" in source
    assert "Mode upload hanya tersedia di Google Colab" in source
    assert "Grocery Planogram Control Dataset" in source
    assert "https://github.com/meyucel/Grocery-Planogram-Control-Dataset.git" in source
    assert "https://github.com/gulvarol/grocerydataset.git" in source
    assert "GroceryDataset_part1.tar.gz" in source
    assert "GroceryDataset_part2.tar.gz" in source
    assert "research only, not commercial use" in source
    assert "Wikimedia Commons" in source
    assert "Supermarket_in_South_Korea.JPG" in source
    assert "Grocery_store%2C_type_A%2C_Volkovo" in source
    assert "Supermarket_shelf_in_Koog_aan_de_Zaan.jpg" in source
    assert "field_source_results.json" in source
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
