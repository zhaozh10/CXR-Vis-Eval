import datetime
import os
from argparse import ArgumentParser
# from downstream.models.backbones.encoder import ImageEncoder
from downstream.models.backbones import cnn_backbones,vits
from downstream.utils.misc import extract_backbone,extract_vit_backbone
import torch
from dateutil import tz
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torchvision.models import resnet50
from downstream.datasets.classification_dataset import (CheXpertImageDataset,
                                                  NIHImageDataset,
                                                  COVIDXImageDataset,
                                                  RSNAImageDataset,
                                                  SIIMClsDataset,
                                                  ChestStructImageDataset
                                                  )
from downstream.datasets.data_module import DataModule
from downstream.datasets.transforms import DataTransforms, Moco2Transform
from downstream.models.ssl_finetuner import SSLFineTuner

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["https_proxy"]="http://127.0.0.1:7890"
# os.environ['WANDB_MODE']="offline"

def cli_main():
    parser = ArgumentParser()
    
    parser.add_argument("--base_model", type=str, default="vit_base_p16")
    parser.add_argument("--ckpt_path", type=str,
                        default=None)
    parser.add_argument("--dataset", type=str, default="nih")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--data_pct", type=float, default=0.01)
    parser.add_argument("--eval_type", type=str, default="linear")
    # add trainer args
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.model_name=args.base_model
    if args.ckpt_path==None:
        method_prefix="ImageNet"
    elif "dino" in args.ckpt_path:
        method_prefix=args.ckpt_path.split('/')[-2]
        method_prefix+="-"+args.ckpt_path.split('/')[-1][:-4]
    elif "mgca" in args.ckpt_path:
        method_prefix=args.ckpt_path.split('/')[-1].replace('.ckpt','')
    else:
        method_prefix=args.ckpt_path.split('/')[-2].split('_')[0]
        method_prefix+="-"+args.ckpt_path.split('/')[-1][:-4]
    # set max epochs
    args.max_epochs = 50

    seed_everything(args.seed)

    if args.dataset == "chexpert":
        # define datamodule
        # check transform here
        datamodule = DataModule(CheXpertImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 5
        multilabel = True
    elif args.dataset == "nih":
        # define datamodule
        # check transform here
        datamodule = DataModule(NIHImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 14
        multilabel = True
    elif args.dataset == "cheststruct":
        # define datamodule
        # check transform here
        datamodule = DataModule(ChestStructImageDataset, None,
                                Moco2Transform, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = True
    elif args.dataset == "rsna":
        datamodule = DataModule(RSNAImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 1
        multilabel = True
    elif args.dataset == "siim":
        datamodule = DataModule(SIIMClsDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 1
        multilabel = True
    elif args.dataset == "covidx":
        datamodule = DataModule(COVIDXImageDataset, None,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
        num_classes = 3
        multilabel = False
    else:
        raise RuntimeError(f"no dataset called {args.dataset}")

    # if args.path:
    #     model = MGCA.load_from_checkpoint(args.path, strict=False)
    # else:
    #     model = MGCA()

    # args.model_name = model.hparams.img_encoder
    # args.backbone = model.img_encoder_q
    # args.in_features = args.backbone.feature_dim
    # args.num_classes = num_classes
    # args.multilabel = multilabel
    # args.backbone=resnet50()
    
    
    if 'resnet' in  args.base_model:
        model_function = getattr(cnn_backbones, args.base_model)
        args.backbone, args.in_features,_ = model_function()
    # model_name='vit_base_p16'
    elif 'vit' in args.base_model:
        model_name=args.base_model
        vit_name = model_name.split('_')[1]
        patch_size=eval(model_name.split('_')[2][1:])
        args.backbone, args.in_features = vits.create_vit(vit_name, image_size=224, patch_size=patch_size)


    args.num_classes=num_classes
    args.multilabel=multilabel

    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path)
        if 'resnet' in args.base_model:
            if "mmpretrain" in args.ckpt_path:
                ckpt_dict = extract_backbone(ckpt['state_dict'],prefix='backbone')
            elif "saliency" in args.ckpt_path:
                ckpt_dict = extract_backbone(ckpt,prefix='unet.encoder')
            elif "dino" in args.ckpt_path:
                ckpt_dict=extract_backbone(ckpt['student'],prefix='module.backbone')
            elif "clip" in args.ckpt_path:
                ckpt_dict=extract_backbone(ckpt['state_dict'],prefix="visual.backbone")
            else:
                ckpt_dict = extract_backbone(ckpt['state_dict'],prefix='img_encoder_q.model')
        elif 'vit' in args.base_model:
            # model_weight_mae=torch.load("../preTrain/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.pth")
            # model_weight_mae=torch.load(ckpt)
            if "MAE" in args.ckpt_path or 'LocalMIM' in args.ckpt_path or 'GazeMIM' in args.ckpt_path or 'maskalign' in args.ckpt_path or 'dmae' in args.ckpt_path or "DeepMIM" in args.ckpt_path:
                ckpt_dict=extract_vit_backbone(ckpt['model'],source=None,prefix=None)
            elif "attmask" in args.ckpt_path:
                ckpt_dict=extract_vit_backbone(ckpt['student'],source=None,prefix='module.backbone')
            elif "mae" in args.ckpt_path:
                ckpt_dict=extract_vit_backbone(ckpt['state_dict'],source='mae',prefix='backbone')
            elif "simmim" in args.ckpt_path:
                ckpt_dict = extract_vit_backbone(ckpt['model'],source=None,prefix='encoder')
            elif "hpm" in args.ckpt_path or "droppos" in args.ckpt_path:
                ckpt_dict = extract_vit_backbone(ckpt['state_dict'],source=None,prefix='module')
            else:
                ckpt_dict=extract_vit_backbone(ckpt['state_dict'],source=None, prefix='img_encoder_q.model')
            
            # model.load_state_dict(state_dict)
        
        print("load weight!")
    else:
        if 'resnet' in args.base_model:
            ckpt = torch.load('../preTrain/resnet50-19c8e357.pth')
            ckpt_dict = extract_backbone(ckpt,prefix=None)
        elif 'vit' in args.base_model:
            ckpt=torch.load('../preTrain/vit_base_p16_224_timm.pth')
            ckpt_dict=extract_vit_backbone(ckpt,source=None)
        # ckpt_dict=ckpt['state_dict']
    args.backbone.load_state_dict(ckpt_dict,strict=False)
    # finetune
    tuner = SSLFineTuner(**args.__dict__)

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    # ckpt_dir = os.path.join(
    #     BASE_DIR, f"../../../data/ckpts/mgca_finetune/{extension}")
    if args.eval_type=="linear":
        ckpt_dir =  f"./data/ckpts/linear_probing/{extension}"
    else:
        ckpt_dir =  f"./data/ckpts/fine_tuning/{extension}"
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=0.,
                      patience=10, verbose=False, mode="min")
    ]

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    if args.eval_type == "linear":
        logger_dir = './data/linear_probing'
    else:
        logger_dir='./data/fine_tuning'
    # logger_dir = os.path.join(
    #     BASE_DIR, f"../../../data/wandb")
    os.makedirs(logger_dir, exist_ok=True)
    if args.eval_type == "linear":
        wandb_logger = WandbLogger(
            project="mgca_linear_prob",
            save_dir=logger_dir,
            name=f"{method_prefix}_{args.dataset}_{args.data_pct}__seed{args.seed}_lr{args.learning_rate}_{extension}")
    else:
        wandb_logger = WandbLogger(
            project="mgca_fine_tuning",
            save_dir=logger_dir,
            name=f"{method_prefix}_{args.dataset}_{args.data_pct}__seed{args.seed}_lr{args.learning_rate}_{extension}")
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=wandb_logger)

    tuner.training_steps = tuner.num_training_steps(trainer, datamodule)

    # train
    trainer.fit(tuner, datamodule)
    # test
    trainer.test(tuner, datamodule, ckpt_path="best")


if __name__ == "__main__":
    cli_main()
