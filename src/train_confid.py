import os
from os.path import abspath, dirname, join

import click
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import datasets.confid_dataset as confid_dataset
import models.confid_network as confid_network


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
def main(config, paths, weights, checkpoint):
    cfg = yaml.safe_load(open(config))    
    cfg["paths"] = yaml.safe_load(open(paths))["confid_paths"]
    cfg['paths']['data_path'] = os.path.join(cfg["paths"]["log_path"], cfg["experiment"]["id"], 'confid_data')
    
    cfg["confid_model"]["eval_n_samples"] = cfg["data"]["eval_n_samples"]
    
    if checkpoint == 'None':
        checkpoint = None

    seed_everything(42, workers=True)
    # Load data and model
    data = confid_dataset.ConfidDataModule(cfg)
    computed_min_kernel_influence = data.train_data.min_kernel_size.item()
    # factor out kernel influence radius
    cfg['confid_model']['min_kernel_r'] = computed_min_kernel_influence / 1.5
    model = confid_network.ConfidNet(cfg)
    
    # insert computed min kernel size
    
    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    tb_logger = pl_loggers.TensorBoardLogger(
        os.path.join(cfg["paths"]["log_path"], cfg["experiment"]["id"]),
        default_hp_metric=False,
    )
    
    # define log folder
    log_name = cfg["experiment"]["id"]
    log_dir = os.path.join(cfg["paths"]["log_path"], log_name)
    cfg["paths"]["log_dir"] = log_dir
    checkpoint_dir = os.path.join(log_dir, "models")
    checkpoint_saver = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val/confid_error",
        filename="best_confid_error",
        mode="min",
        save_last=True,
    )
    
    if checkpoint:
        checkpoint = os.path.join(checkpoint_dir, checkpoint)

    # Setup trainer
    trainer = Trainer(
        gpus=cfg["train"]["n_gpus"],
        accelerator='gpu',
        devices=cfg["train"]["n_gpus"],
        logger=tb_logger,
        resume_from_checkpoint=checkpoint,
        max_epochs=cfg["train"]["max_epoch"],
        max_steps=cfg['train']['max_confid_steps'],
        callbacks=[lr_monitor, checkpoint_saver],
        deterministic=False,
        log_every_n_steps=2,
        check_val_every_n_epoch=cfg["train"]["check_val_every_n_epoch"],
        accumulate_grad_batches=cfg["data"]["accumulate_grad"],
        # val_check_interval=0.3,
    )

    # Train!
    trainer.fit(model, data)


if __name__ == "__main__":
    main(auto_envvar_prefix="PLS")
