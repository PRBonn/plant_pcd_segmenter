import os
from os.path import abspath, dirname, join

import click
import yaml
from pytorch_lightning import Trainer, seed_everything

import datasets.digiforest_dataset as digiforest_dataset
import datasets.sbub_dataset as datasets
import models.segmentation_network as networks


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
    if 'dataset' in cfg['data'].keys():
        if cfg['data']['dataset'] == 'patches':
            dataset_name = 'patches'
            cfg["paths"] = yaml.safe_load(open(paths))["patch_paths"]
        elif cfg['data']['dataset'] == 'forest':
            dataset_name = 'forest'
            cfg["paths"] = yaml.safe_load(open(paths))["forest_paths"]
        elif cfg['data']['dataset'] == 'plants':
            dataset_name = 'plants'
            cfg["paths"] = yaml.safe_load(open(paths))["plant_paths"]
    else:
        dataset_name = 'patches'
        cfg["paths"] = yaml.safe_load(open(paths))["patch_paths"]
    # enforce batch_size=1
    cfg["data"]["batch_size_embed"] = 1
    cfg["data"]["eval_n_samples"] = cfg["data"]["sample_size"]
    cfg["offset_model"]["eval_n_samples"] = cfg["data"]["eval_n_samples"]
    

    
    log_name = cfg["experiment"]["id"]
    log_type= "emb"
    # define log folder
    log_dir = join(cfg["paths"]["log_path"], log_name)
    cfg["paths"]["log_dir"] = log_dir
    preds_dir = join(log_dir, "preds")
    if not os.path.exists(preds_dir):
            os.mkdir(preds_dir)
    cfg["paths"]["preds_dir"] = preds_dir
    
    checkpoint_dir = os.path.join(log_dir, "models")
    

    # Load data and model
    if dataset_name == 'forest':
        data = digiforest_dataset.DigiforestDataModule(cfg)
    else:
        data = datasets.EmbedDataModule(cfg)
    # data.setup()
    
    computed_min_kernel_influence = data.train_data.min_kernel_size.item()
    # factor out kernel influence radius
    cfg["offset_model"]["min_kernel_r"] = computed_min_kernel_influence / 1.5
    
    model = networks.SegNetwork.load_from_checkpoint(join(checkpoint_dir,checkpoint), hparams=cfg)

    # Setup trainer
    trainer = Trainer(accelerator="cpu", devices=1, logger=False)

    # Test!
    trainer.test(model, data)


if __name__ == "__main__":
    main(auto_envvar_prefix="PLS")
