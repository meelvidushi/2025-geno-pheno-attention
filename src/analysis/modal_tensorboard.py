from pathlib import Path

import modal

app = modal.App("geno-pheno-attention-tensorboard-server")
MOUNT = Path("/data")
volume = modal.Volume.from_name("geno-pheno-attention-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("tensorboard==2.19.0")


@app.function(
    image=image,
    volumes={str(MOUNT): volume},
    memory=1024,
)
@modal.web_server(6006)
def tensorboard_server(logdir: str = "checkpoints"):
    import subprocess

    full_logdir = str(MOUNT / logdir)
    subprocess.Popen(f"tensorboard --logdir={full_logdir} --bind_all --port 6006", shell=True)


if __name__ == "__main__":
    # To deploy: modal deploy metl/trainers/modal_tensorboard.py
    # For development: modal serve metl/trainers/modal_tensorboard.py
    pass
