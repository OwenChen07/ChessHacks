import modal

# --- CONFIG -------------------------------------------------------------------
GPU_TYPE = "A10G"  # or: "T4", "L4", "A100", etc.

# Build an image with dependencies AND mount your local files
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "tqdm",
        "pandas",
        "numpy",
        "python-chess"  # Added chess library
    )
    .add_local_file("train.py", "/root/train.py")
    .add_local_file("chess_model.py", "/root/chess_model.py")
    .add_local_file("chess_dataset.py", "/root/chess_dataset.py")
    .add_local_file("fen_to_tensor.py", "/root/fen_to_tensor.py")
    .add_local_file("move_encoder.py", "/root/move_encoder.py")
    # Add your data directory
    .add_local_dir("data", "/root/data")
)

# Create or load a persisted volume
model_volume = modal.Volume.from_name(
    "chess-model-vol",
    create_if_missing=True
)

app = modal.App("chess-training")


# --- GPU TRAINING FUNCTION ----------------------------------------------------
@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/root/model": model_volume},
    timeout=60 * 60 * 5,
)
def train_remote():
    # Import from the files added to the image
    import sys
    sys.path.insert(0, "/root")
    
    from train import train_model

    print("Starting training on Modal GPU...")

    model = train_model(
        csv_path="/root/data/data2.csv",
        num_epochs=5,
        batch_size=64,
        learning_rate=0.001,
        num_filters=128,
        num_residual_blocks=5,
        max_rows=1000,
        save_path="/root/model/chess_model.pth",
    )

    print("Training finished. Model saved to /root/model/chess_model.pth")
    
    # Commit the volume to persist the model
    model_volume.commit()


# --- LOCAL ENTRYPOINT ----------------------------------------------------------
@app.local_entrypoint()
def main():
    # Use .remote() to run and wait for results
    train_remote.remote()

# You can also call the function directly after deploying
# This allows you to trigger training without keeping your laptop connected