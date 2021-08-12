from comet_ml import Experiment
from logging import disable
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import BYOL

from utils import DiseaseDataset, make_transform, make_train_transform_SWAV, make_eval_transform_SWAV
from models import ResNet50, MobileNetv2, Inceptionv3, EfficientNet1, BYOLNet, SWAVNet

from pytorch_lightning.metrics.classification import Accuracy
from pytorch_lightning.loggers import TensorBoardLogger


from pytorch_lightning.loggers import CometLogger

from pytorch_lightning.callbacks import ModelCheckpoint

import argparse

def main(cmd_args):

    args = {
        'batch_size':8,
        'epochs':10,
        'max_steps':100000,
        'num_workers':4,
        'freeze_fe':True,
        'model':cmd_args.model,
        'dataset':'isicbinary',
        'model_dir':"cv-skin-disease-2021"
    }

    num_classes = 2
    img_dir = "/IMG_DIR/"
    label_dir = "/LABEL/DIR/ISIC_binary_Metadata.csv"
        
    if args['model'] == 'resnet50':
        model = ResNet50(num_classes=num_classes,model_dir=args['model_dir'],freeze_fe=args['freeze_fe'])
        num_img_copies = 1
        train_make_transform = make_transform
    elif args['model'] == 'mobilenetv2':
        model = MobileNetv2(num_classes=num_classes,model_dir=args['model_dir'],freeze_fe=args['freeze_fe'])
        num_img_copies = 1
        train_make_transform = make_transform
    elif args['model'] == 'inceptionv3':
        model = Inceptionv3(num_classes=num_classes,model_dir=args['model_dir'],freeze_fe=args['freeze_fe'])
        num_img_copies = 1
        train_make_transform = make_transform
    elif args['model'] == 'efficientnet':
        model = EfficientNet1(num_classes=num_classes,model_dir=args['model_dir'],freeze_fe=args['freeze_fe'])
        num_img_copies = 1
        train_make_transform = make_transform
    elif args['model'] == 'BYOL':
        model = BYOLNet(num_classes=num_classes,model_dir=args['model_dir'],freeze_fe=args['freeze_fe'],max_epochs=args['epochs'])
        model.input_size = 224
        model.name = "BYOL"
        num_img_copies = 2
        train_make_transform = make_transform
    elif args['model'] == 'SWAV':
        model = SWAVNet(num_classes=num_classes,model_dir=args['model_dir'],freeze_fe=args['freeze_fe'])
        model.name = "SWAV"
        num_img_copies = 1
        train_make_transform = make_train_transform_SWAV

    """
    Rebalancing the dataset
    """
    import pandas as pd
    metadata=pd.read_csv(label_dir)  
    metadata=metadata[metadata['class'].isin(['malignant','benign'])]      
    from collections import Counter
    class_freqs = Counter(metadata['class'])
    smallest_class = min(class_freqs.values())
    rebalanced_metadata = []
    for class_ in class_freqs.keys():
        rebalanced_metadata.append(metadata[metadata['class']==class_].sample(smallest_class))
    rebalanced_metadata = pd.concat(rebalanced_metadata)

    dataset_train = DiseaseDataset(image_dir=img_dir,
                            metadata=rebalanced_metadata,
                            img_shape=model.input_size,
                            make_transform=train_make_transform,
                            num_img_copies=num_img_copies)

    dataset_val = DiseaseDataset(image_dir=img_dir,
                            metadata=rebalanced_metadata,
                            img_shape=model.input_size,
                            make_transform=train_make_transform,
                            num_img_copies=num_img_copies)

    train_test_split = 0.8
    train_len = int(train_test_split * len(dataset_train) )
    test_len = len(dataset_train) - train_len
    train, val = random_split(dataset_train, [train_len, test_len])
    val.dataset = dataset_val

    if 'SWAV' in args['model']:
        val.dataset.transform = make_eval_transform_SWAV(None)
    
    comet_logger = CometLogger(
        api_key="your_API_key",
        workspace="your_workspace", # Optional
        project_name=args['model_dir'], # Optional
        experiment_name=args['model']+'_'+args['dataset'], # Optional,
        save_dir=args['model'],
        disabled=False
    )

    comet_logger.log_hyperparams(args)

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('..','models'),filename=args['model']+'-{epoch:02d}', save_last=False ,save_top_k=-1)

    trainer = pl.Trainer(max_epochs=args['epochs'], 
                         max_steps=args['max_steps'],
                         gpus=1 if torch.cuda.is_available() else 0, 
                         default_root_dir=os.path.join(args['model']),
                         logger=comet_logger,
                         accelerator='dp',
                         callbacks=[checkpoint_callback])

    trainer.fit(model, 
            DataLoader(train,num_workers=args['num_workers'],batch_size=args['batch_size'],drop_last=True),#,  sampler = sampler),
            DataLoader(val,num_workers=args['num_workers'],batch_size=args['batch_size'],drop_last=True))#,  sampler = sampler))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher for ML jobs')
    parser.add_argument('--model', help='NN to train',type=str,default='mobilenetv2',choices=['mobilenetv2','resnet50','inceptionv3','efficientnet',"BYOL",'SWAV'])
    args = parser.parse_args()
    main(args)
