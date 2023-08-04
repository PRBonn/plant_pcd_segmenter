import os
from os.path import abspath, dirname, join

import click
import yaml
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import datasets.sbub_dataset as datasets
import models.segmentation_network as networks


@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config/configFull.yaml"),
)
@click.option(
    "--paths",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=join(dirname(abspath(__file__)), "config/paths.yaml"),
)
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.",
    default=None,
)
@click.option(
    "--checkpoint",
    "-ckpt",
    type=str,
    help="path to checkpoint file (.ckpt) to resume training.",
    default=None,
)
@click.option("--train_embeddings", is_flag=True)
@click.option("--train_confid", is_flag=True)
@click.option("--generate_confid_dataset", is_flag=True)
def main(
    config,
    paths,
    weights,
    checkpoint,
    train_embeddings,
    train_confid,
    generate_confid_dataset,
):
    cfg = yaml.safe_load(open(config))
    if "dataset" in cfg["data"].keys():
        if cfg["data"]["dataset"] == "patches":
            dataset_name = "patches"
            cfg["paths"] = yaml.safe_load(open(paths))["patch_paths"]
        elif cfg["data"]["dataset"] == "plants":
            dataset_name = "plants"
            cfg["paths"] = yaml.safe_load(open(paths))["plant_paths"]
    else:
        dataset_name = "patches"
        cfg["paths"] = yaml.safe_load(open(paths))["patch_paths"]

    if checkpoint == "None":
        checkpoint = None
    # copy over parameters
    cfg["offset_model"]["loss_n_samples"] = cfg["data"]["loss_n_samples"]
    cfg["offset_model"]["eval_n_samples"] = cfg["data"]["eval_n_samples"]
    cfg["offset_model"]["sample_size"] = cfg["data"]["sample_size"]
    cfg["offset_model"]["batch_size"] = cfg["data"]["batch_size_embed"]

    cfg["offset_model"]["enabled"] = True
    cfg["data"]["generate_confid_dataset"] = generate_confid_dataset

    if generate_confid_dataset:
        cfg["train"]["check_val_every_n_epoch"] = 1
        
    seed_everything(42, workers=True)
    # Load data and model
    if dataset_name == "forest":
        data = digiforest_dataset.DigiforestDataModule(cfg)
    else:
        data = datasets.EmbedDataModule(cfg)

    log_name = cfg["experiment"]["id"]
    # if train_embeddings:
    log_type = "emb"
    tb_logger = pl_loggers.TensorBoardLogger(
        os.path.join(cfg["paths"]["log_path"], log_name, log_type),
        default_hp_metric=False,
    )
    # define log folder
    log_dir = os.path.join(cfg["paths"]["log_path"], log_name)
    cfg["paths"]["log_dir"] = log_dir
    checkpoint_dir = os.path.join(log_dir, "models")

    computed_min_kernel_influence = data.train_data.min_kernel_size.item()
    # factor out kernel influence radius
    cfg["offset_model"]["min_kernel_r"] = computed_min_kernel_influence / 1.5

    if weights:
        model = networks.SegNetwork.load_from_checkpoint(
            os.path.join(checkpoint_dir, weights), hparams=cfg
        )
    else:
        model = networks.SegNetwork(cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/" + cfg["train"]["val_metric"],
        filename=log_type + "best_" + cfg["train"]["val_metric"],
        mode=cfg["train"]["val_metric_min_max"],
        save_last=True,
    )

    if checkpoint:
        checkpoint = os.path.join(checkpoint_dir, checkpoint)

    # Setup trainer
    trainer = Trainer(
        # gpus=cfg["train"]["n_gpus"],
        accelerator="gpu",
        devices=cfg["train"]["n_gpus"],
        logger=tb_logger,
        max_epochs=cfg["train"]["max_epoch"],
        max_steps=(
            cfg["data"]["confid_data_samples"]
            if generate_confid_dataset
            else cfg["train"]["max_embed_steps"]
        ),
        callbacks=[lr_monitor, checkpoint_saver],
        deterministic=False,
        log_every_n_steps=5,
        accumulate_grad_batches=cfg["data"]["accumulate_grad"],
        check_val_every_n_epoch=cfg["train"]["check_val_every_n_epoch"],
        profiler="advanced",
    )

    # Train!
    trainer.fit(model, data, ckpt_path=checkpoint)


if __name__ == "__main__":
    main(auto_envvar_prefix="PLS")
