import os
from os.path import abspath, dirname, join

import click
import yaml
from pytorch_lightning import Trainer, seed_everything

import datasets.confid_dataset as datasets
import models.confid_network as confid_network


@click.command()
### Add your options here
@click.option(
    "--checkpoint",
    "-ckpt",
    type=str,
    help="path to checkpoint file (.ckpt)",
    required=True,
)
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config/configKeypoints.yaml"),
)
@click.option(
    "--paths",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config/paths.yaml"),
)
@click.option("--embeddings", is_flag=True)
@click.option("--confid", is_flag=True)
def main(checkpoint, config, paths, embeddings, confid):
    seed_everything(42, workers=True)
    cfg = yaml.safe_load(open(config))
    cfg["paths"] = yaml.safe_load(open(paths))["confid_paths"]
    cfg["paths"]["data_path"] = os.path.join(
        cfg["paths"]["log_path"], cfg["experiment"]["id"], "confid_data"
    )
    # enforce batch_size=1
    cfg["data"]["batch_size_confid"] = 1
    cfg["data"]["eval_n_samples"] = cfg["data"]["sample_size"]
    cfg["confid_model"]["eval_n_samples"] = cfg["data"]["eval_n_samples"]

    log_name = cfg["experiment"]["id"]
    
    log_type = "confid"
    # define log folder
    log_dir = join(cfg["paths"]["log_path"], log_name)
    cfg["paths"]["log_dir"] = log_dir
    preds_dir = join(log_dir, "preds")
    if not os.path.exists(preds_dir):
        os.mkdir(preds_dir)
    cfg["paths"]["preds_dir"] = preds_dir

    checkpoint_dir = os.path.join(log_dir, "models")

    # Load data and model
    data = datasets.ConfidDataModule(cfg)
    data.setup()

    computed_min_kernel_influence = data.train_data.min_kernel_size.item()
    # factor out kernel influence radius
    cfg["confid_model"][
        "min_kernel_r"
    ] = computed_min_kernel_influence / 1.5

    model = confid_network.ConfidNet.load_from_checkpoint(
        join(checkpoint_dir, checkpoint), hparams=cfg
    )

    # Setup trainer
    trainer = Trainer(accelerator="gpu", devices=1, logger=False)

    # Test!
    trainer.test(model, data)


if __name__ == "__main__":
    main(auto_envvar_prefix="PLS")
