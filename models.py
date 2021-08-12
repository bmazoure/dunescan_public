import pytorch_lightning as pl
from torch.nn.modules.loss import MSELoss
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.metrics.classification import Accuracy
from sklearn.metrics import confusion_matrix
import numpy as np
from utils import set_parameter_requires_grad
from pl_bolts.models.self_supervised import BYOL
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder
from pl_bolts.models.self_supervised.byol.models import MLP

from torch.optim import Adam

from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
#from pl_bolts.callbacks.self_supervised import BYOLMAWeightUpdate
from pl_bolts.models.self_supervised.byol.models import SiameseArm
# from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from pl_bolts.models.self_supervised.swav.swav_resnet import resnet50

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_model_params as get_model_params_efficientnet


import glob, os

from copy import deepcopy

from collections import namedtuple
InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])

class DiseaseTransferLearning(pl.LightningModule):
    def __init__(self,num_classes,model_dir,freeze_fe=True):
        # init a pretrained resnet
        super().__init__()
        self.feature_extractor = None
        self.name = None
        self.model_dir = model_dir

        self.criteria = nn.CrossEntropyLoss()
        self.metric = Accuracy()

    def features(self, x):
        return self.feature_extractor(x)

    def top_fc(self, features):
        raise NotImplementedError('Method not implemented')

    def forward(self, x):
        features = self.features(x)
        x = self.top_fc(features)
        return x

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y = y.squeeze(-1)
        x = x.squeeze(1)
        y_hat = self.forward(x)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        acc = self.metric(y_hat.max(1)[1],y)
        return {'loss': loss, 
                'acc':acc}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y = y.squeeze(-1)
        x = x[0]
        y_hat = self.forward(x)
        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)
        val_acc = self.metric(y_hat.max(1)[1],y)
        val_preds = [np.eye(2)[y.detach().cpu().numpy()],y_hat.detach().cpu().numpy()]
        return {'val_loss': val_loss,
                'val_acc':val_acc,
                'val_preds':val_preds}

    def training_epoch_end(self, outputs):
        # OPTIONAL
        outputs = outputs[0]
        d = {}
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().detach().item()
        self.logger.experiment.log_metric('Loss/Train',avg_loss,self.current_epoch)
        d['train_loss'] = avg_loss
        if 'acc' in outputs[0].keys():
            avg_acc = torch.stack([x['acc'] for x in outputs]).mean().cpu().detach().item()
            self.logger.experiment.log_metric('Acc/Train',avg_acc,self.current_epoch)
            d['train_acc'] = avg_acc
        # self.current_epoch += 1
        # return d

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        d= {}
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().cpu().detach().item()
        self.logger.experiment.log_metric('Loss/Val',avg_loss,self.current_epoch)
        d['val_loss'] = avg_loss
        
        if 'val_preds' in outputs[0].keys():
            y_true = np.concatenate([x['val_preds'][0] for x in outputs])
            y_hat = np.concatenate([x['val_preds'][1] for x in outputs])
            self.logger.experiment.log_confusion_matrix(y_true, y_hat)
        if 'val_acc' in outputs[0].keys():
            avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean().cpu().detach().item()
            self.logger.experiment.log_metric('Acc/Val',avg_acc,self.current_epoch)
            d['val_acc'] = avg_acc
        # Save model to Comet
        model_file = glob.glob(os.path.join(self.name,'cv-skin-disease','**','**','*.ckpt'))
        if len(model_file): #and int(model_file[0].split('epoch')[1][1:].split('.')[0]) == self.current_epoch:
            self.logger.experiment.log_model(self.name+'_'+str(self.current_epoch), model_file[0])
        return d

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=0.001, mode="max")

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'val_checkpoint_on'
        }
        return [optimizer], []

""""
========================
Model definitions (classifiers)
"""

class ResNet50(DiseaseTransferLearning):
    def __init__(self,num_classes,model_dir,freeze_fe=True,load_pretrained=True):
        super().__init__(num_classes,model_dir,freeze_fe)

        self.feature_extractor = models.resnet50(
                                    pretrained=load_pretrained,
                                    num_classes=1000)
        set_parameter_requires_grad(self.feature_extractor,not freeze_fe)

        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, num_classes) # 2048

        self.input_size = 224
        self.name = 'resnet50'

    def features(self, x):
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)

        x = self.feature_extractor.layer1(x)
        x = self.feature_extractor.layer2(x)
        x = self.feature_extractor.layer3(x)
        x = self.feature_extractor.layer4(x)

        x = self.feature_extractor.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def top_fc(self, features):
        x = self.feature_extractor.fc(features)
        return x

class MobileNetv2(DiseaseTransferLearning):
    def __init__(self,num_classes,model_dir,freeze_fe=True,load_pretrained=True):
        super().__init__(num_classes,model_dir,freeze_fe)

        self.feature_extractor = models.mobilenet_v2(
                                    pretrained=load_pretrained,
                                    num_classes=1000)
                                    
        set_parameter_requires_grad(self.feature_extractor,not freeze_fe)

        num_ftrs = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier[1] = nn.Linear(num_ftrs, num_classes) # 1280

        self.input_size = 224
        self.name = "mobilenetv2"

    def features(self, x):
        x = self.feature_extractor.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return x

    def top_fc(self, features):
        x = self.feature_extractor.classifier(features)
        return x

class Inceptionv3(DiseaseTransferLearning):
    def __init__(self,num_classes,model_dir,freeze_fe=True,load_pretrained=True):
        super().__init__(num_classes,model_dir,freeze_fe)

        self.feature_extractor = models.inception_v3(
                                    pretrained=load_pretrained,
                                    num_classes=1000)

        set_parameter_requires_grad(self.feature_extractor,freeze_fe)

        num_ftrs = self.feature_extractor.AuxLogits.fc.in_features
        self.feature_extractor.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs,num_classes)
        self.input_size = 299

        self.name = "inceptionv3"

    def features(self, x):
        # N x 3 x 299 x 299
        x = self.feature_extractor.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.feature_extractor.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.feature_extractor.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.feature_extractor.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.feature_extractor.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.feature_extractor.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.feature_extractor.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.feature_extractor.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.feature_extractor.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.feature_extractor.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.feature_extractor.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.feature_extractor.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.feature_extractor.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.feature_extractor.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.feature_extractor.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.feature_extractor.training and self.feature_extractor.aux_logits
        if aux_defined:
            aux = self.feature_extractor.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.feature_extractor.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.feature_extractor.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.feature_extractor.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.feature_extractor.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.feature_extractor.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        return x

    def top_fc(self, features):
        x = self.feature_extractor.fc(features)
        return x

    def forward(self, x):
        features = self.features(x)
        x = self.top_fc(features)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

class EfficientNet1(DiseaseTransferLearning):
    def __init__(self,num_classes,model_dir,freeze_fe=True,load_pretrained=True):
        super().__init__(num_classes,model_dir,freeze_fe)

        if load_pretrained:
            self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            blocks_args, global_params = get_model_params_efficientnet('efficientnet-b0',{})
            self.feature_extractor = EfficientNet(blocks_args,global_params)

        set_parameter_requires_grad(self.feature_extractor,not freeze_fe)

        num_ftrs = self.feature_extractor._fc.in_features
        self.feature_extractor._fc = nn.Linear(num_ftrs,num_classes)

        self.input_size = 224
        self.name = "efficientnet"
    
    def features(self, x):
        x = self.feature_extractor.extract_features(x)
        # Pooling and final linear layer
        x = self.feature_extractor._avg_pooling(x)
        if self.feature_extractor._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self.feature_extractor._dropout(x)
        return x

    def top_fc(self, features):
        x = self.feature_extractor._fc(features)
        return x

class BYOLNet(DiseaseTransferLearning):
    def __init__(self,num_classes,model_dir,freeze_fe=True,max_epochs=10,load_pretrained=True):
        super().__init__(num_classes,model_dir,freeze_fe)

        self.online_network = SiameseArm()
        if load_pretrained:
            self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

        self.classifier = nn.Linear(2048,num_classes)

        self.name = "BYOL"
        self.input_size = 224

    def on_train_batch_end(self, epoch_end_outputs, batch, batch_idx, dataloader_idx):
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, epoch_end_outputs, batch, batch_idx, dataloader_idx)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def cosine_similarity(self, a, b):
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        sim = (a * b).sum(-1).mean()
        return sim

    def shared_step(self, batch, batch_idx):
        (img_1, img_2), y = batch
        
        if type(img_1) == list:
            img_1 = img_1[0]
            img_2 = img_2[0]

        img_1 = img_1.squeeze(1)
        img_2 = img_2.squeeze(1)
        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = - 2 * self.cosine_similarity(h1, z2)

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        loss_b = - 2 * self.cosine_similarity(h1, z2)

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss, y1

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss_a, loss_b, total_loss, features = self.shared_step(batch, batch_idx)

        x, y = batch
        y = y.squeeze(-1)
        y_hat = self.classifier(features)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        acc = self.metric(y_hat.max(1)[1],y)

        return {'loss':total_loss+loss,
                'acc':acc}

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss, features = self.shared_step(batch, batch_idx)

        x, y = batch
        y = y.squeeze(-1)
        y_hat = self.classifier(features)

        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)
        val_acc = self.metric(y_hat.max(1)[1],y)
        val_preds = [np.eye(2)[y.detach().cpu().numpy()],y_hat.detach().cpu().numpy()]
        return {'val_loss': total_loss+val_loss,
                'val_acc':val_acc,
                'val_preds':val_preds}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4, weight_decay=1.5e-6)
        #optimizer = LARSWrapper(optimizer)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=20
        )
        optimizer_fc = Adam(self.classifier.parameters(), lr=1e-4, weight_decay=1.5e-6)
        return [optimizer,optimizer_fc], [scheduler]
    
    def features(self, x):
        y = self.encoder(x)[0]
        y = self.pooler(y)
        y = y.view(y.size(0), -1)
        return y

    def top_fc(self, features):
        h = self.predictor(features)
        return h

class SiameseArm(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()

        if encoder is None:
            encoder = torchvision_ssl_encoder('resnet50')
        # Encoder
        self.encoder = encoder
        # Pooler
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Projector
        self.projector = MLP(input_dim=2048, hidden_size=4096, output_dim=256)
        # Predictor
        self.predictor = MLP(input_dim=256)

    def forward(self, x):
        #if type(x) == list:
        #    x = x[0]
        y = self.encoder(x)[0]
        #y = self.pooler(y)
        #y = y.view(y.size(0), -1)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h

class SWAVNet(DiseaseTransferLearning):
    def __init__(self,num_classes,model_dir,freeze_fe=True,max_epochs=10,load_pretrained=True):
        super().__init__(num_classes,model_dir,freeze_fe)

        self.model = resnet50(
                        normalize=True,
                        hidden_mlp=2048,
                        output_dim=128,
                        nmb_prototypes=3072,
                        first_conv=True,
                        maxpool1=True
                    )
        self.sinkhorn_iterations = 3
        self.crops_for_assign = [0, 1]
        self.epsilon = 0.05
        self.nmb_crops = [2, 6]
        self.temperature = 0.1

        self.epoch_queue_starts = 15
        self.queue_length = 0
        self.queue = None

        self.gpus = 1

        self.get_assignments = self.sinkhorn

        self.classifier = nn.Linear(3072,num_classes)

        self.name = "BYOL"
        self.input_size = 224

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.model.forward_backbone(x)

    def shared_step(self, batch, batch_idx):
        inputs_orig, y = batch
        inputs = inputs_orig[:-1]  # remove online train/eval transforms at this point

        # 1. normalize the prototypes
        with torch.no_grad():
            w = self.model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.model.prototypes.weight.copy_(w)

        # 2. multi-res forward passes
        embedding, output = self.model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)
        
        # 3. swav loss computation
        loss = 0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]

                # 4. time to use the queue
                if self.queue is not None:
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.model.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # 5. get assignments
                q = torch.exp(out / self.epsilon).t()
                q = self.get_assignments(q, self.sinkhorn_iterations)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_crops)-2), crop_id):
                p = self.softmax(output[bs * v: bs * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.nmb_crops) - 1)
        loss /= len(self.crops_for_assign)

        # output for linear layer
        embedding_val, output_val = self.model(inputs_orig[-1])

        return loss, output_val

    def on_train_epoch_start(self):
        if self.queue_length > 0:
            if self.trainer.current_epoch >= self.epoch_queue_starts and self.queue is None:
                self.queue = torch.zeros(
                    len(self.crops_for_assign),
                    self.queue_length ,  # change to nodes * gpus once multi-node
                    self.feat_dim,
                )

                if self.gpus > 0:
                    self.queue = self.queue.cuda()

        self.use_the_queue = False

    def training_step(self, batch, batch_idx, optimizer_idx):
        total_loss, features = self.shared_step(batch, batch_idx)

        x, y = batch
        y = y.squeeze(-1)
        y_hat = self.classifier(features)
        loss = self.criteria(y_hat, y)
        loss = loss.unsqueeze(dim=-1)
        acc = self.metric(y_hat.max(1)[1],y)

        return {'loss':total_loss+loss,
                'acc':acc}

    def validation_step(self, batch, batch_idx):
        total_loss, features = self.shared_step(batch, batch_idx)

        x, y = batch
        y = y.squeeze(-1)
        y_hat = self.classifier(features)

        val_loss = self.criteria(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)
        val_acc = self.metric(y_hat.max(1)[1],y)
        val_preds = [np.eye(2)[y.detach().cpu().numpy()],y_hat.detach().cpu().numpy()]
        return {'val_loss': total_loss+val_loss,
                'val_acc':val_acc,
                'val_preds':val_preds}

    def sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)

                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4, weight_decay=1.5e-6)
        # optimizer = LARSWrapper(optimizer)
        optimizer_fc = Adam(self.classifier.parameters(), lr=1e-4, weight_decay=0)
        return [optimizer,optimizer_fc]
    
    def features(self, x):
        y = self.encoder(x)[0]
        y = self.pooler(y)
        y = y.view(y.size(0), -1)
        return y

    def top_fc(self, features):
        h = self.predictor(features)
        return h


"""
Other helpful models (non-classification)
"""

class AutoEncoder(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,2)
        )
        self.dec = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2,256),
            nn.ReLU(),
            nn.Linear(256,input_dim)
        )

    def forward(self,x):
        return self.dec(self.enc(x))

class VAE(nn.Module):
    def __init__(self,alpha = 1):
        #Autoencoder only requires 1 dimensional argument since input and output-size is the same
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1280,512),nn.ReLU(),
                                     nn.Linear(512,128),nn.ReLU(),
                                     nn.Linear(128,32),nn.LeakyReLU())
        self.hidden2mu = nn.Linear(32,2)
        self.hidden2log_var = nn.Linear(32,2)
        self.alpha = alpha
        self.decoder = nn.Sequential(nn.Linear(2,128),nn.ReLU(),
                                     nn.Linear(128,512),nn.ReLU(),
                                     nn.Linear(512,1280))

    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn(size = (mu.size(0),mu.size(1)))
        z= z.type_as(mu) # Setting z to be .cuda when using GPU training 
        return mu + sigma*z

    def encode(self,x):
       hidden = self.encoder(x)
       mu = self.hidden2mu(hidden)
       log_var = self.hidden2log_var(hidden)
       return mu,log_var

    def forward(self,x):
       batch_size = x.size(0)
       x = x.view(batch_size,-1)
       mu,log_var = self.encode(x)
       hidden = self.reparametrize(mu,log_var)
       return self.decoder(hidden)