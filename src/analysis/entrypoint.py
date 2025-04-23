import modal
import tyro

from analysis.modal_infrastructure import app, train_model_with_modal
from analysis.train import RunParams, train_model

if __name__ == "__main__":
    config = tyro.cli(RunParams)

    if config.use_modal:
        with modal.enable_output(), app.run(detach=True):
            train_model_with_modal(config)
    else:
        train_model(config)
