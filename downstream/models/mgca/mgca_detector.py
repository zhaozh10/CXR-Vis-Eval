import datetime
import os
from argparse import ArgumentParser

import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger

from downstream.datasets.data_module import DataModule
from downstream.datasets.detection_dataset import (OBJCXRDetectionDataset,
                                             RSNADetectionDataset)
from downstream.datasets.transforms import DetectionDataTransforms
from downstream.models.backbones.detector_backbone import ResNetDetector
from downstream.models.ssl_detector import SSLDetector
from downstream.utils.misc import extract_backbone
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# torch.use_deterministic_algorithms(True)
def cli_main():
    parser = ArgumentParser("Finetuning of object detection task for MGCA")
    parser.add_argument("--base_model", type=str, default="resnet_50")
    parser.add_argument("--ckpt_path", type=str,
                        default=None)
    parser.add_argument("--dataset", type=str,
                        default="rsna", help="rsna or object_cxr")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if args.ckpt_path==None:
        method_prefix="ImageNet"
    elif "dino" in args.ckpt_path:
        method_prefix=args.ckpt_path.split('/')[-2]
        method_prefix+="-"+args.ckpt_path.split('/')[-1][:-4]
    else:
        method_prefix=args.ckpt_path.split('/')[-2].split('_')[0]
        method_prefix+="-"+args.ckpt_path.split('/')[-1][:-4]
    args.deterministic = False
    args.max_epochs = 50
    args.accelerator = "gpu"

    seed_everything(args.seed)

    if args.dataset == "rsna":
        datamodule = DataModule(RSNADetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    elif args.dataset == "object_cxr":
        datamodule = DataModule(OBJCXRDetectionDataset, None, DetectionDataTransforms,
                                args.data_pct, args.batch_size, args.num_workers)
    else:
        raise RuntimeError(f"{args.dataset} does not exist!")

    args.img_encoder = ResNetDetector("resnet_50")
    # if args.ckpt_path:
    #     ckpt = torch.load(args.ckpt_path)
    #     if "mmpretrain" in args.ckpt_path:
    #         ckpt_dict = extract_backbone(ckpt['state_dict'],prefix='backbone')
    #     else:
    #         ckpt_dict = extract_backbone(ckpt['state_dict'],prefix='img_encoder_q.model')
    #     args.img_encoder.model.load_state_dict(ckpt_dict)

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        if "mmpretrain" in args.ckpt_path:
            ckpt_dict = extract_backbone(ckpt['state_dict'],prefix='backbone')
        elif "saliency" in args.ckpt_path:
            ckpt_dict = extract_backbone(ckpt,prefix='encoder')
        elif "dino" in args.ckpt_path:
            ckpt_dict=extract_backbone(ckpt['student'],prefix='module.backbone')
        else:
            ckpt_dict = extract_backbone(ckpt['state_dict'],prefix='img_encoder_q.model')
        
        print("load weight!")
    else:
        ckpt = torch.load('../preTrain/resnet50-19c8e357.pth')
        ckpt_dict = extract_backbone(ckpt,prefix=None)
    args.img_encoder.model.load_state_dict(ckpt_dict)
    # Freeze encoder
    for param in args.img_encoder.parameters():
        param.requires_grad = False

    model = SSLDetector(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    # ckpt_dir = os.path.join(
    #     BASE_DIR, f"./ckpts/detection/{extension}")
    ckpt_dir =  f"./data/ckpts/detection/{extension}"
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_mAP", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_mAP", min_delta=0.,
                      patience=10, verbose=False, mode="max")
    ]
    # logger_dir = os.path.join(
    #     BASE_DIR, f"/home/sdb/MGCA")
    logger_dir = './data/detection'
    os.makedirs(logger_dir, exist_ok=True)
    
    wandb_logger = WandbLogger(
        project="detection", save_dir=logger_dir,
        name=f"{method_prefix}_{args.dataset}_pct{args.data_pct}_seed{args.seed}_lr{args.learning_rate}_{extension}")
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger
    )
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
