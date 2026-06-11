from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "bottle"


def main():
    # Use the local MVTec bottle dataset.
    datamodule = Folder(
        name="my_bottle",
        root=str(DATA_DIR),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defect",
        train_batch_size=32,
        eval_batch_size=32,
        val_split_mode="none",
        num_workers=0,
    )

    model = Patchcore(backbone="resnet18")

    # Disable validation because this datamodule is configured without a val split.
    engine = Engine(
        accelerator="auto",
        max_epochs=1,
        limit_val_batches=0,
    )

    print("[INFO] Starting training with validation disabled...")
    engine.fit(model=model, datamodule=datamodule)

    print("[INFO] Running test...")
    metrics = engine.test(model=model, datamodule=datamodule)
    print(metrics)


if __name__ == "__main__":
    main()
