
![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)
![GitHub](https://img.shields.io/badge/Release-PROD-yellow.svg)
![GitHub](https://img.shields.io/badge/Languages-MULTI-blue.svg)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)

# Computer Vision Pretrained Models

![CV logo](https://github.com/balavenkatesh3322/CV-pretrained-model/blob/master/logo.jpg)

## What is pre-trained Model?
A pre-trained model is a model created by some one else to solve a similar problem. Instead of building a model from scratch to solve a similar problem, we can use the model trained on other problem as a starting point. A pre-trained model may not be 100% accurate in your application.

For example, if you want to build a self learning car. You can spend years to build a decent image recognition algorithm from scratch or you can take inception model (a pre-trained model) from Google which was built on [ImageNet](http://www.image-net.org/) data to identify images in those pictures.

## Other Pre-trained Models
* [NLP Pre-trained Models](https://github.com/balavenkatesh3322/NLP-pretrained-model).
* [Audio and Speech Pre-trained Models](https://github.com/balavenkatesh3322/audio-pretrained-model)

## Model Deployment library
* [Model Serving](https://github.com/balavenkatesh3322/model_deployment)

### Framework

* [Tensorflow](#tensorflow)
* [Keras](#keras)
* [PyTorch](#pytorch)
* [Caffe](#caffe)
* [MXNet](#mxnet)

### Model visualization
You can see visualizations of each model's network architecture by using [Netron](https://github.com/lutzroeder/Netron).

![CV logo](https://github.com/balavenkatesh3322/CV-pretrained-model/blob/master/netron.png)

### Tensorflow <a name="tensorflow"/>

| Model Name | Description | Framework | License |
|   :---:      |     :---:      |     :---:     |     :---:     |
| [ObjectDetection]( https://github.com/tensorflow/models/tree/master/research/object_detection)  | Localizing and identifying multiple objects in a single image.| `Tensorflow`| [Apache License]( https://raw.githubusercontent.com/tensorflow/models/master/LICENSE )
| [Mask R-CNN]( https://github.com/matterport/Mask_RCNN)  | The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.     | `Tensorflow`| [The MIT License (MIT)]( https://raw.githubusercontent.com/matterport/Mask_RCNN/master/LICENSE )
| [Faster-RCNN]( https://github.com/smallcorgi/Faster-RCNN_TF)  | This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.     | `Tensorflow`| [MIT License]( https://raw.githubusercontent.com/smallcorgi/Faster-RCNN_TF/master/LICENSE )
| [YOLO TensorFlow]( https://github.com/gliese581gg/YOLO_tensorflow)  | This is tensorflow implementation of the YOLO:Real-Time Object Detection.     | `Tensorflow`| [Custom]( https://raw.githubusercontent.com/gliese581gg/YOLO_tensorflow/master/LICENSE )
| [YOLO TensorFlow ++]( https://github.com/thtrieu/darkflow)  | TensorFlow implementation of 'YOLO: Real-Time Object Detection', with training and an actual support for real-time running on mobile devices.     | `Tensorflow`| [GNU GENERAL PUBLIC LICENSE]( https://raw.githubusercontent.com/thtrieu/darkflow/master/LICENSE )
| [MobileNet]( https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)  | MobileNets trade off between latency, size and accuracy while comparing favorably with popular models from the literature.     | `Tensorflow`| [The MIT License (MIT)]( https://raw.githubusercontent.com/tensorflow/models/master/LICENSE )
| [DeepLab]( https://github.com/tensorflow/models/tree/master/research/deeplab)  | Deep labeling for semantic image segmentation.     | `Tensorflow`| [Apache License]( https://raw.githubusercontent.com/tensorflow/models/master/LICENSE )
| [Colornet]( https://github.com/pavelgonchar/colornet)  | Neural Network to colorize grayscale images.     | `Tensorflow`| Not Found
| [SRGAN]( https://github.com/tensorlayer/srgan)  | Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.    | `Tensorflow`| Not Found
| [DeepOSM]( https://github.com/trailbehind/DeepOSM)  | Train TensorFlow neural nets with OpenStreetMap features and satellite imagery.     | `Tensorflow`| [The MIT License (MIT)]( https://raw.githubusercontent.com/trailbehind/DeepOSM/master/LICENSE )
| [Domain Transfer Network]( https://github.com/yunjey/domain-transfer-network)  | Implementation of Unsupervised Cross-Domain Image Generation.  | `Tensorflow`| [MIT License]( https://raw.githubusercontent.com/yunjey/domain-transfer-network/master/LICENSE )
| [Show, Attend and Tell]( https://github.com/yunjey/show-attend-and-tell)  | Attention Based Image Caption Generator.     | `Tensorflow`| [MIT License]( https://raw.githubusercontent.com/yunjey/show-attend-and-tell/master/LICENSE )
| [android-yolo]( https://github.com/natanielruiz/android-yolo)  | Real-time object detection on Android using the YOLO network, powered by TensorFlow.    | `Tensorflow`| [Apache License]( https://raw.githubusercontent.com/natanielruiz/android-yolo/master/LICENSE )
| [DCSCN Super Resolution]( https://github.com/jiny2001/dcscn-super-resolutiont)  | This is a tensorflow implementation of "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network", a deep learning based Single-Image Super-Resolution (SISR) model.     | `Tensorflow`| Not Found
| [GAN-CLS]( https://github.com/zsdonghao/text-to-image)  | This is an experimental tensorflow implementation of synthesizing images.     | `Tensorflow`| Not Found
| [U-Net]( https://github.com/zsdonghao/u-net-brain-tumor)  | For Brain Tumor Segmentation.     | `Tensorflow`| Not Found
| [Improved CycleGAN]( https://github.com/luoxier/CycleGAN_Tensorlayer)  |Unpaired Image to Image Translation.     | `Tensorflow`| [MIT License]( https://raw.githubusercontent.com/luoxier/CycleGAN_Tensorlayer/master/LICENSE )
| [Im2txt]( https://github.com/tensorflow/models/tree/master/research/im2txt)  | Image-to-text neural network for image captioning.     | `Tensorflow`| [Apache License]( https://raw.githubusercontent.com/tensorflow/models/master/LICENSE )
| [SLIM]( https://github.com/tensorflow/models/tree/master/research/slim)  | Image classification models in TF-Slim.     | `Tensorflow`| [Apache License]( https://raw.githubusercontent.com/tensorflow/models/master/LICENSE )
| [DELF]( https://github.com/tensorflow/models/tree/master/research/delf)  | Deep local features for image matching and retrieval.     | `Tensorflow`| [Apache License]( https://raw.githubusercontent.com/tensorflow/models/master/LICENSE )
| [Compression]( https://github.com/tensorflow/models/tree/master/research/compression)  | Compressing and decompressing images using a pre-trained Residual GRU network.     | `Tensorflow`| [Apache License]( https://raw.githubusercontent.com/tensorflow/models/master/LICENSE )
| [AttentionOCR]( https://github.com/tensorflow/models/tree/master/research/attention_ocr)  | A model for real-world image text extraction.     | `Tensorflow`| [Apache License]( https://raw.githubusercontent.com/tensorflow/models/master/LICENSE )

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### Keras <a name="keras"/>

| Model Name | Description | Framework | License |
|   :---:      |     :---:      |     :---:     |     :---:     |
| [Mask R-CNN]( https://github.com/matterport/Mask_RCNN)  | The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.| `Keras`| [The MIT License (MIT)]( https://raw.githubusercontent.com/matterport/Mask_RCNN/master/LICENSE )
| [VGG16]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)  | Very Deep Convolutional Networks for Large-Scale Image Recognition.     | `Keras`| [The MIT License (MIT)]( https://raw.githubusercontent.com/keras-team/keras-applications/master/LICENSE )
| [VGG19]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)  | Very Deep Convolutional Networks for Large-Scale Image Recognition.     | `Keras`| [The MIT License (MIT)]( https://raw.githubusercontent.com/keras-team/keras-applications/master/LICENSE )
| [ResNet]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)  | Deep Residual Learning for Image Recognition.     | `Keras`| [The MIT License (MIT)]( https://raw.githubusercontent.com/keras-team/keras-applications/master/LICENSE )
| [ResNet50](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py)  | Deep Residual Learning for Image Recognition.     | `Keras`| [The MIT License (MIT)]( https://raw.githubusercontent.com/keras-team/keras-applications/master/LICENSE )
| [MobileNet]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py)  | MobileNet v1 models for Keras.  | `Keras`| [The MIT License (MIT)]( https://raw.githubusercontent.com/keras-team/keras-applications/master/LICENSE )
| [MobileNet V2]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py)  | MobileNet v2 models for Keras.  | `Keras`| [The MIT License (MIT)]( https://raw.githubusercontent.com/keras-team/keras-applications/master/LICENSE )
| [Image analogies]( https://github.com/awentzonline/image-analogies)  | Generate image analogies using neural matching and blending.     | `Keras`| [The MIT License (MIT)]( https://raw.githubusercontent.com/awentzonline/image-analogies/master/LICENSE.txt )
| [Popular Image Segmentation Models]( https://github.com/divamgupta/image-segmentation-keras)  | Implementation of Segnet, FCN, UNet and other models in Keras.     | `Keras`| [MIT License]( https://raw.githubusercontent.com/divamgupta/image-segmentation-keras/master/LICENSE )
| [Ultrasound nerve segmentation]( https://github.com/jocicmarko/ultrasound-nerve-segmentation)  | This tutorial shows how to use Keras library to build deep neural network for ultrasound image nerve segmentation.     | `Keras`| [MIT License]( https://raw.githubusercontent.com/jocicmarko/ultrasound-nerve-segmentation/master/LICENSE.md )
| [DeepMask object segmentation]( https://github.com/abbypa/NNProject_DeepMask)  | This is a Keras-based Python implementation of DeepMask- a complex deep neural network for learning object segmentation masks.     | `Keras`| Not Found
| [Monolingual and Multilingual Image Captioning]( https://github.com/elliottd/GroundedTranslation)  | This is the source code that accompanies Multilingual Image Description with Neural Sequence Models.     | `Keras`| [BSD-3-Clause License]( https://raw.githubusercontent.com/elliottd/GroundedTranslation/master/LICENSE )
| [pix2pix]( https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)  | Keras implementation of Image-to-Image Translation with Conditional Adversarial Networks by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A.    | `Keras`| Not Found
| [Colorful Image colorization]( https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Colorful)  | B&W to color.   | `Keras`| Not Found
| [CycleGAN]( https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py)  | Implementation of _Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_.    | `Keras`| [MIT License]( https://raw.githubusercontent.com/eriklindernoren/Keras-GAN/master/LICENSE )
| [DualGAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/dualgan/dualgan.py)  | Implementation of _DualGAN: Unsupervised Dual Learning for Image-to-Image Translation_.   | `Keras`| [MIT License]( https://raw.githubusercontent.com/eriklindernoren/Keras-GAN/master/LICENSE )
| [Super-Resolution GAN]( https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py)  | Implementation of _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_.   | `Keras`| [MIT License]( https://raw.githubusercontent.com/eriklindernoren/Keras-GAN/master/LICENSE )

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### PyTorch <a name="pytorch"/>

| Model Name | Description | Framework | License |
|   :---:      |     :---:      |     :---:     |     :---:     |
|[detectron2](https://github.com/facebookresearch/detectron2) | Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms | `PyTorch` | [Apache License 2.0](https://raw.githubusercontent.com/facebookresearch/detectron2/master/LICENSE) 
| [FastPhotoStyle]( https://github.com/NVIDIA/FastPhotoStyle)  | A Closed-form Solution to Photorealistic Image Stylization.   | `PyTorch`| [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public Licens]( https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md )
| [pytorch-CycleGAN-and-pix2pix]( https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  | A Closed-form Solution to Photorealistic Image Stylization.   | `PyTorch`| [BSD License]( https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/master/LICENSE )
| [maskrcnn-benchmark]( https://github.com/facebookresearch/maskrcnn-benchmark)  | Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/facebookresearch/maskrcnn-benchmark/master/LICENSE )
| [deep-image-prior]( https://github.com/DmitryUlyanov/deep-image-prior)  | Image restoration with neural networks but without learning.   | `PyTorch`| [Apache License 2.0]( https://raw.githubusercontent.com/DmitryUlyanov/deep-image-prior/master/LICENSE )
| [StarGAN]( https://github.com/yunjey/StarGAN)  | StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/yunjey/StarGAN/master/LICENSE )
| [faster-rcnn.pytorch]( https://github.com/jwyang/faster-rcnn.pytorch)  | This project is a faster faster R-CNN implementation, aimed to accelerating the training of faster R-CNN object detection models.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/jwyang/faster-rcnn.pytorch/master/LICENSE )
| [pix2pixHD]( https://github.com/NVIDIA/pix2pixHD)  | Synthesizing and manipulating 2048x1024 images with conditional GANs.  | `PyTorch`| [BSD License]( https://raw.githubusercontent.com/NVIDIA/pix2pixHD/master/LICENSE.txt )
| [Augmentor]( https://github.com/mdbloice/Augmentor)  | Image augmentation library in Python for machine learning.  | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/mdbloice/Augmentor/master/LICENSE.md )
| [albumentations]( https://github.com/albumentations-team/albumentations)  | Fast image augmentation library.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/albumentations-team/albumentations/master/LICENSE )
| [Deep Video Analytics]( https://github.com/AKSHAYUBHAT/DeepVideoAnalytics)  | Deep Video Analytics is a platform for indexing and extracting information from videos and images   | `PyTorch`| [Custom]( https://raw.githubusercontent.com/AKSHAYUBHAT/DeepVideoAnalytics/master/LICENSE )
| [semantic-segmentation-pytorch]( https://github.com/CSAILVision/semantic-segmentation-pytorch)  | Pytorch implementation for Semantic Segmentation/Scene Parsing on MIT ADE20K dataset.   | `PyTorch`| [BSD 3-Clause License]( https://raw.githubusercontent.com/CSAILVision/semantic-segmentation-pytorch/master/LICENSE )
| [An End-to-End Trainable Neural Network for Image-based Sequence Recognition]( https://github.com/bgshih/crnn)  | This software implements the Convolutional Recurrent Neural Network (CRNN), a combination of CNN, RNN and CTC loss for image-based sequence recognition tasks, such as scene text recognition and OCR.   | `PyTorch`| [The MIT License (MIT)]( https://raw.githubusercontent.com/bgshih/crnn/master/LICENSE )
| [UNIT]( https://github.com/mingyuliutw/UNIT)  | PyTorch Implementation of our Coupled VAE-GAN algorithm for Unsupervised Image-to-Image Translation.   | `PyTorch`| [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License]( https://raw.githubusercontent.com/mingyuliutw/UNIT/master/LICENSE.md )
| [Neural Sequence labeling model]( https://github.com/jiesutd/NCRFpp)  | Sequence labeling models are quite popular in many NLP tasks, such as Named Entity Recognition (NER), part-of-speech (POS) tagging and word segmentation.   | `PyTorch`| [Apache License]( https://raw.githubusercontent.com/jiesutd/NCRFpp/master/LICENCE )
| [faster rcnn]( https://github.com/longcw/faster_rcnn_pytorch)  | This is a PyTorch implementation of Faster RCNN. This project is mainly based on py-faster-rcnn and TFFRCNN. For details about R-CNN please refer to the paper Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/longcw/faster_rcnn_pytorch/master/LICENSE )
| [pytorch-semantic-segmentation]( https://github.com/ZijunDeng/pytorch-semantic-segmentation)  | PyTorch for Semantic Segmentation.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/ZijunDeng/pytorch-semantic-segmentation/master/LICENSE )
| [EDSR-PyTorch]( https://github.com/thstkdgus35/EDSR-PyTorch)  | PyTorch version of the paper 'Enhanced Deep Residual Networks for Single Image Super-Resolution'.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/thstkdgus35/EDSR-PyTorch/master/LICENSE )
| [image-classification-mobile]( https://github.com/osmr/imgclsmob)  | Collection of classification models pretrained on the ImageNet-1K.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/osmr/imgclsmob/master/LICENSE )
| [FaderNetworks]( https://github.com/facebookresearch/FaderNetworks)  | Fader Networks: Manipulating Images by Sliding Attributes - NIPS 2017.   | `PyTorch`| [Creative Commons Attribution-NonCommercial 4.0 International Public License]( https://raw.githubusercontent.com/facebookresearch/FaderNetworks/master/LICENSE )
| [neuraltalk2-pytorch]( https://github.com/ruotianluo/ImageCaptioning.pytorch)  | Image captioning model in pytorch (finetunable cnn in branch with_finetune).   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/ruotianluo/ImageCaptioning.pytorch/master/LICENSE )
| [RandWireNN]( https://github.com/seungwonpark/RandWireNN)  | Implementation of: "Exploring Randomly Wired Neural Networks for Image Recognition".   | `PyTorch`| Not Found
| [stackGAN-v2]( https://github.com/hanzhanggit/StackGAN-v2)  |Pytorch implementation for reproducing StackGAN_v2 results in the paper StackGAN++.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/hanzhanggit/StackGAN-v2/master/LICENSE )
| [Detectron models for Object Detection]( https://github.com/ignacio-rocco/detectorch)  | This code allows to use some of the Detectron models for object detection from Facebook AI Research with PyTorch.   | `PyTorch`| [Apache License]( https://raw.githubusercontent.com/ignacio-rocco/detectorch/master/LICENSE )
| [DEXTR-PyTorch]( https://github.com/scaelles/DEXTR-PyTorch)  | This paper explores the use of extreme points in an object (left-most, right-most, top, bottom pixels) as input to obtain precise object segmentation for images and videos.   | `PyTorch`| [GNU GENERAL PUBLIC LICENSE]( https://raw.githubusercontent.com/scaelles/DEXTR-PyTorch/master/LICENSE )
| [pointnet.pytorch]( https://github.com/fxia22/pointnet.pytorch)  | Pytorch implementation for "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/fxia22/pointnet.pytorch/master/LICENSE )
| [self-critical.pytorch]( https://github.com/ruotianluo/self-critical.pytorch) | This repository includes the unofficial implementation Self-critical Sequence Training for Image Captioning and Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/ruotianluo/self-critical.pytorch/master/LICENSE )
| [vnet.pytorch]( https://github.com/mattmacy/vnet.pytorch)  | A Pytorch implementation for V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.   | `PyTorch`| [BSD 3-Clause License]( https://raw.githubusercontent.com/mattmacy/vnet.pytorch/master/LICENSE )
| [piwise]( https://github.com/bodokaiser/piwise)  | Pixel-wise segmentation on VOC2012 dataset using pytorch.   | `PyTorch`| [BSD 3-Clause License]( https://raw.githubusercontent.com/bodokaiser/piwise/master/LICENSE.md )
| [pspnet-pytorch]( https://github.com/Lextal/pspnet-pytorch)  | PyTorch implementation of PSPNet segmentation network.   | `PyTorch`| Not Found
| [pytorch-SRResNet]( https://github.com/twtygqyy/pytorch-SRResNet)  | Pytorch implementation for Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.   | `PyTorch`| [The MIT License (MIT)]( https://raw.githubusercontent.com/twtygqyy/pytorch-SRResNet/master/LICENSE )
| [PNASNet.pytorch]( https://github.com/chenxi116/PNASNet.pytorch)  | PyTorch implementation of PNASNet-5 on ImageNet.   | `PyTorch`| [Apache License]( https://raw.githubusercontent.com/chenxi116/PNASNet.pytorch/master/LICENSE )
| [img_classification_pk_pytorch]( https://github.com/felixgwu/img_classification_pk_pytorch)  | Quickly comparing your image classification models with the state-of-the-art models.   | `PyTorch`| Not Found
| [Deep Neural Networks are Easily Fooled]( https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks)  | High Confidence Predictions for Unrecognizable Images.   | `PyTorch`| [MIT License]( https://raw.githubusercontent.com/utkuozbulak/pytorch-cnn-adversarial-attacks/master/LICENSE )
| [pix2pix-pytorch]( https://github.com/mrzhu-cool/pix2pix-pytorch)  | PyTorch implementation of "Image-to-Image Translation Using Conditional Adversarial Networks".   | `PyTorch`| Not Found
| [NVIDIA/semantic-segmentation]( https://github.com/NVIDIA/semantic-segmentation)  | A PyTorch Implementation of Improving Semantic Segmentation via Video Propagation and Label Relaxation, In CVPR2019.   | `PyTorch`| [CC BY-NC-SA 4.0 license]( https://raw.githubusercontent.com/NVIDIA/semantic-segmentation/master/LICENSE )
| [Neural-IMage-Assessment]( https://github.com/kentsyx/Neural-IMage-Assessment)  | A PyTorch Implementation of Neural IMage Assessment.   | `PyTorch`| Not Found
| [torchxrayvision](https://github.com/mlmed/torchxrayvision) | Pretrained models for chest X-ray (CXR) pathology predictions. Medical, Healthcare, Radiology  | `PyTorch` | [Apache License]( https://raw.githubusercontent.com/mlmed/torchxrayvision/master/LICENSE ) |
| [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) | PyTorch image models, scripts, pretrained weights -- (SE)ResNet/ResNeXT, DPN, EfficientNet, MixNet, MobileNet-V3/V2, MNASNet, Single-Path NAS, FBNet, and more  | `PyTorch` | [Apache License 2.0]( https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE ) |

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### Caffe <a name="caffe"/>

| Model Name | Description | Framework | License |
|   :---:      |     :---:      |     :---:     |     :---:     |
| [OpenPose]( https://github.com/CMU-Perceptual-Computing-Lab/openpose)  | OpenPose represents the first real-time multi-person system to jointly detect human body, hand, and facial keypoints (in total 130 keypoints) on single images.   | `Caffe`| [Custom]( https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/LICENSE )
| [Fully Convolutional Networks for Semantic Segmentation]( https://github.com/shelhamer/fcn.berkeleyvision.org)  | Fully Convolutional Models for Semantic Segmentation.   | `Caffe`| Not Found
| [Colorful Image Colorization]( https://github.com/richzhang/colorization)  | Colorful Image Colorization.   | `Caffe`| [BSD-2-Clause License]( https://raw.githubusercontent.com/richzhang/colorization/master/LICENSE )
| [R-FCN]( https://github.com/YuwenXiong/py-R-FCN)  | R-FCN: Object Detection via Region-based Fully Convolutional Networks.   | `Caffe`| [MIT License]( https://raw.githubusercontent.com/YuwenXiong/py-R-FCN/master/LICENSE )
| [cnn-vis]( https://github.com/jcjohnson/cnn-vis)  |Inspired by Google's recent Inceptionism blog post, cnn-vis is an open-source tool that lets you use convolutional neural networks to generate images.   | `Caffe`| [The MIT License (MIT)]( https://raw.githubusercontent.com/jcjohnson/cnn-vis/master/LICENSE )
| [DeconvNet]( https://github.com/HyeonwooNoh/DeconvNet)  | Learning Deconvolution Network for Semantic Segmentation.   | `Caffe`| [Custom]( https://raw.githubusercontent.com/HyeonwooNoh/DeconvNet/master/LICENSE )

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### MXNet <a name="mxnet"/>

| Model Name | Description | Framework | License |
|   :---:      |     :---:      |     :---:     |     :---:     |
| [Faster RCNN]( https://github.com/ijkguo/mx-rcnn)  | Region Proposal Network solves object detection as a regression problem.   | `MXNet`| [Apache License, Version 2.0]( https://raw.githubusercontent.com/ijkguo/mx-rcnn/master/LICENSE )
| [SSD]( https://github.com/zhreshold/mxnet-ssd)  | SSD is an unified framework for object detection with a single network.   | `MXNet`| [MIT License]( https://raw.githubusercontent.com/zhreshold/mxnet-ssd/master/LICENSE )
| [Faster RCNN+Focal Loss]( https://github.com/unsky/focal-loss)  | The code is unofficial version for focal loss for Dense Object Detection.   | `MXNet`| Not Found
| [CNN-LSTM-CTC]( https://github.com/oyxhust/CNN-LSTM-CTC-text-recognition)  |I realize three different models for text recognition, and all of them consist of CTC loss layer to realize no segmentation for text images.   | `MXNet`| Not Found
| [Faster_RCNN_for_DOTA]( https://github.com/jessemelpolio/Faster_RCNN_for_DOTA)  | This is the official repo of paper _DOTA: A Large-scale Dataset for Object Detection in Aerial Images_.  | `MXNet`| [Apache License]( https://raw.githubusercontent.com/jessemelpolio/Faster_RCNN_for_DOTA/master/LICENSE )
| [RetinaNet]( https://github.com/unsky/RetinaNet)  | Focal loss for Dense Object Detection.   | `MXNet`| Not Found
| [MobileNetV2]( https://github.com/liangfu/mxnet-mobilenet-v2)  | This is a MXNet implementation of MobileNetV2 architecture as described in the paper _Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation_.   | `MXNet`| [Apache License]( https://raw.githubusercontent.com/liangfu/mxnet-mobilenet-v2/master/LICENSE )
| [neuron-selectivity-transfer]( https://github.com/TuSimple/neuron-selectivity-transfer)  | This code is a re-implementation of the imagenet classification experiments in the paper _Like What You Like: Knowledge Distill via Neuron Selectivity Transfer_.   | `MXNet`| [Apache License]( https://raw.githubusercontent.com/TuSimple/neuron-selectivity-transfer/master/LICENSE )
| [MobileNetV2]( https://github.com/chinakook/MobileNetV2.mxnet)  | This is a Gluon implementation of MobileNetV2 architecture as described in the paper _Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation_.   | `MXNet`| [Apache License]( https://raw.githubusercontent.com/chinakook/MobileNetV2.mxnet/master/LICENSE )
| [sparse-structure-selection]( https://github.com/TuSimple/sparse-structure-selection)  | This code is a re-implementation of the imagenet classification experiments in the paper _Data-Driven Sparse Structure Selection for Deep Neural Networks_.   | `MXNet`| [Apache License]( https://raw.githubusercontent.com/TuSimple/sparse-structure-selection/master/LICENSE )
| [FastPhotoStyle]( https://github.com/NVIDIA/FastPhotoStyle)  | A Closed-form Solution to Photorealistic Image Stylization.   | `MXNet`| [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License]( https://raw.githubusercontent.com/NVIDIA/FastPhotoStyle/master/LICENSE.md )

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

## Contributions
Your contributions are always welcome!!
Please have a look at `contributing.md`

## License

[MIT License](LICENSE)
