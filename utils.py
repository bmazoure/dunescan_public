import ipdb
from kornia.geometry import transform
import pandas as pd
import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import LabelEncoder

import pytorch_lightning as pl

from kornia.color import *
from kornia import augmentation as K
from torchvision.transforms import functional as tvF
from torchvision.transforms import transforms

import cv2, glob

import shutil  

def allowed_file(filename,ALLOWED_EXTENSIONS):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

"""
========================
PyTorch models
"""

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def make_transform(img_shape):
    def transform(img):
        return transforms.Compose([
                transforms.Resize([img_shape,img_shape]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                K.ColorJitter(0.1, 0.1, 0.1, 0.1),
                K.RandomHorizontalFlip(),
                K.RandomVerticalFlip(),
                # K.RandomMotionBlur(3, 35., 0.5),
                # K.RandomRotation(degrees=45.0),
                # K.RandomResizedCrop((224,244))
                ])(img)
    return transform

class SwAVTrainDataTransform(object):
    def __init__(
        self,
        normalize=None,
        size_crops = [96, 36],
        nmb_crops = [2, 4],
        min_scale_crops = [0.33, 0.10],
        max_scale_crops = [1, 0.33],
        gaussian_blur = True,
        jitter_strength = 1.
    ):
        self.jitter_strength = jitter_strength
        self.gaussian_blur = gaussian_blur

        self.size_crops = size_crops
        self.nmb_crops = nmb_crops
        self.min_scale_crops = min_scale_crops
        self.max_scale_crops = max_scale_crops

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        transform = []
        color_transform = [
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        if self.gaussian_blur:
            color_transform.append(
                GaussianBlur(kernel_size=int(0.1 * self.size_crops[0]), p=0.5)
            )

        self.color_transform = transforms.Compose(color_transform)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        for i in range(len(self.size_crops)):
            random_resized_crop = transforms.RandomResizedCrop(
                self.size_crops[i],
                scale=(self.min_scale_crops[i], self.max_scale_crops[i]),
            )

            transform.extend([transforms.Compose([
                random_resized_crop,
                transforms.RandomHorizontalFlip(p=0.5),
                self.color_transform,
                self.final_transform])
            ] * self.nmb_crops[i])

        self.transform = transform

        # add online train transform of the size of global view
        online_train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.size_crops[0]),
            transforms.RandomHorizontalFlip(),
            self.final_transform
        ])
        self.transform.append(online_train_transform)
        
    def __call__(self, sample):
        multi_crops = list(
            map(lambda transform: transform(sample), self.transform)
        )

        return multi_crops

class SwAVEvalDataTransform(SwAVTrainDataTransform):
    def __init__(
        self,
        normalize=None,
        size_crops = [96, 36],
        nmb_crops = [2, 4],
        min_scale_crops = [0.33, 0.10],
        max_scale_crops = [1, 0.33],
        gaussian_blur = True,
        jitter_strength = 1.
    ):
        super().__init__(
            normalize=normalize,
            size_crops=size_crops,
            nmb_crops=nmb_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength
        )

        input_height = self.size_crops[0]  # get global view crop
        test_transform = transforms.Compose([
            transforms.Resize(int(input_height + 0.1 * input_height)),
            transforms.CenterCrop(input_height),
            self.final_transform,
        ])

        # replace last transform to eval transform in self.transform list
        # self.transform[-1] = test_transform
        self.transform = test_transform

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

def make_train_transform_SWAV(img_shape):
    transform = SwAVTrainDataTransform()
    return transform

def make_eval_transform_SWAV(img_shape):
    transform = SwAVEvalDataTransform()
    return transform

class DiseaseDataset(Dataset):
    """Tabular and Image dataset."""

    def __init__(self, image_dir, metadata,img_shape, make_transform, num_img_copies=1):
        self.image_dir = image_dir
        # self.transform = transforms.Compose([transforms.Resize([300,300]),transforms.ToTensor(),K.RandomCrop(size=(224,224)),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) #,K.RandomSharpness(1.)])
        self.transform = make_transform(img_shape)
        self.num_img_copies = num_img_copies
        self.metadata = metadata
        self.enc_y = LabelEncoder() #(handle_unknown='ignore',sparse=False)
        
        if self.metadata['class'].dtype==int:
            self.y = self.metadata['class'].to_numpy().reshape(-1,1)
        else:
            self.y = self.enc_y.fit_transform(self.metadata['class'].to_numpy().reshape(-1,1)).reshape(-1,1)
    
        self.default_image = None
        self.default_y = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.metadata.iloc[idx]['image']            

        try:
            image = Image.open(os.path.join(self.image_dir,img_name+'.jpg'))
            if self.num_img_copies == 1:
                images = self.transform( image )
            else:
                images = []
                for _ in range(self.num_img_copies):
                    aug_img = self.transform( image )
                    images.append( aug_img )
                    
            y = torch.LongTensor(self.y[idx])

            if self.default_image is None:
                self.default_image = images
                self.default_y = y
        except Exception as e:
            images = self.default_image
            y = self.default_y
            

        return images, y

def img_transform(img_path,transform):
    image = Image.open(img_path)
    image = transform(image)
    return image

import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        # target_activations = []
        # for name, module in self.model._modules.items():
        #     if 'criteria' in name or 'metric' in name:
        #         continue
        #     if name == 'feature_extractor':
        #         target_activations, x = self.feature_extractor(x)
        #     elif "avgpool" in name.lower():
        #         x = module(x)
        #         x = x.view(x.size(0),-1)
        #     else:
        #         x = module(x)
        target_activations, x = self.feature_extractor(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.model.feature_extractor.classifier(x)
        
        return target_activations, x


def show_cam_on_image(img, mask, img_name):
    alpha = 0.2
    # import ipdb; ipdb.set_trace()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) 
    cam = (1-alpha)* heatmap / 255. + alpha * np.float32(img.detach().cpu().numpy()[0].transpose(1,2,0))
    cam = cam / np.max(cam)
    cv2.imwrite(img_name+".jpg", np.uint8(cam*255))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output



def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

