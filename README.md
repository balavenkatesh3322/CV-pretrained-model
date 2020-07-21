
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

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [ObjectDetection]( https://github.com/tensorflow/models/tree/master/research/object_detection)  | Localizing and identifying multiple objects in a single image.| `Tensorflow`
| [Mask R-CNN]( https://github.com/matterport/Mask_RCNN)  | The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.     | `Tensorflow`
| [Faster-RCNN]( https://github.com/smallcorgi/Faster-RCNN_TF)  | This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.     | `Tensorflow`
| [YOLO TensorFlow]( https://github.com/gliese581gg/YOLO_tensorflow)  | This is tensorflow implementation of the YOLO:Real-Time Object Detection.     | `Tensorflow`
| [YOLO TensorFlow ++]( https://github.com/thtrieu/darkflow)  | TensorFlow implementation of 'YOLO: Real-Time Object Detection', with training and an actual support for real-time running on mobile devices.     | `Tensorflow`
| [MobileNet]( https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)  | MobileNets trade off between latency, size and accuracy while comparing favorably with popular models from the literature.     | `Tensorflow`
| [DeepLab]( https://github.com/tensorflow/models/tree/master/research/deeplab)  | Deep labeling for semantic image segmentation.     | `Tensorflow`
| [Colornet]( https://github.com/pavelgonchar/colornet)  | Neural Network to colorize grayscale images.     | `Tensorflow`
| [SRGAN]( https://github.com/tensorlayer/srgan)  | Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.    | `Tensorflow`
| [DeepOSM]( https://github.com/trailbehind/DeepOSM)  | Train TensorFlow neural nets with OpenStreetMap features and satellite imagery.     | `Tensorflow`
| [Domain Transfer Network]( https://github.com/yunjey/domain-transfer-network)  | Implementation of Unsupervised Cross-Domain Image Generation.  | `Tensorflow`
| [Show, Attend and Tell]( https://github.com/yunjey/show-attend-and-tell)  | Attention Based Image Caption Generator.     | `Tensorflow`
| [android-yolo]( https://github.com/natanielruiz/android-yolo)  | Real-time object detection on Android using the YOLO network, powered by TensorFlow.    | `Tensorflow`
| [DCSCN Super Resolution]( https://github.com/jiny2001/dcscn-super-resolutiont)  | This is a tensorflow implementation of "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network", a deep learning based Single-Image Super-Resolution (SISR) model.     | `Tensorflow`
| [GAN-CLS]( https://github.com/zsdonghao/text-to-image)  | This is an experimental tensorflow implementation of synthesizing images.     | `Tensorflow`
| [U-Net]( https://github.com/zsdonghao/u-net-brain-tumor)  | For Brain Tumor Segmentation.     | `Tensorflow`
| [Improved CycleGAN]( https://github.com/luoxier/CycleGAN_Tensorlayer)  |Unpaired Image to Image Translation.     | `Tensorflow`
| [Im2txt]( https://github.com/tensorflow/models/tree/master/research/im2txt)  | Image-to-text neural network for image captioning.     | `Tensorflow`
| [Street]( https://github.com/tensorflow/models/tree/master/research/street)  | Identify the name of a street (in France) from an image using a Deep RNN. | `Tensorflow`
| [SLIM]( https://github.com/tensorflow/models/tree/master/research/slim)  | Image classification models in TF-Slim.     | `Tensorflow`
| [DELF]( https://github.com/tensorflow/models/tree/master/research/delf)  | Deep local features for image matching and retrieval.     | `Tensorflow`
| [Compression]( https://github.com/tensorflow/models/tree/master/research/compression)  | Compressing and decompressing images using a pre-trained Residual GRU network.     | `Tensorflow`
| [AttentionOCR]( https://github.com/tensorflow/models/tree/master/research/attention_ocr)  | A model for real-world image text extraction.     | `Tensorflow`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### Keras <a name="keras"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [Mask R-CNN]( https://github.com/matterport/Mask_RCNN)  | The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.| `Keras`
| [VGG16]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)  | Very Deep Convolutional Networks for Large-Scale Image Recognition.     | `Keras`
| [VGG19]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)  | Very Deep Convolutional Networks for Large-Scale Image Recognition.     | `Keras`
| [ResNet]( https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)  | Deep Residual Learning for Image Recognition.     | `Keras`
| [Image analogies]( https://github.com/awentzonline/image-analogies)  | Generate image analogies using neural matching and blending.     | `Keras`
| [Popular Image Segmentation Models]( https://github.com/divamgupta/image-segmentation-keras)  | Implementation of Segnet, FCN, UNet and other models in Keras.     | `Keras`
| [Ultrasound nerve segmentation]( https://github.com/jocicmarko/ultrasound-nerve-segmentation)  | This tutorial shows how to use Keras library to build deep neural network for ultrasound image nerve segmentation.     | `Keras`
| [DeepMask object segmentation]( https://github.com/abbypa/NNProject_DeepMask)  | This is a Keras-based Python implementation of DeepMask- a complex deep neural network for learning object segmentation masks.     | `Keras`
| [Monolingual and Multilingual Image Captioning]( https://github.com/elliottd/GroundedTranslation)  | This is the source code that accompanies Multilingual Image Description with Neural Sequence Models.     | `Keras`
| [pix2pix]( https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/pix2pix)  | Keras implementation of Image-to-Image Translation with Conditional Adversarial Networks by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A.    | `Keras`
| [Colorful Image colorization]( https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Colorful)  | B&W to color.   | `Keras`
| [CycleGAN]( https://github.com/eriklindernoren/Keras-GAN/blob/master/cyclegan/cyclegan.py)  | Implementation of _Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_.    | `Keras`
| [DualGAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/dualgan/dualgan.py)  | Implementation of _DualGAN: Unsupervised Dual Learning for Image-to-Image Translation_.   | `Keras`
| [Super-Resolution GAN]( https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py)  | Implementation of _Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network_.   | `Keras`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### PyTorch <a name="pytorch"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [FastPhotoStyle]( https://github.com/NVIDIA/FastPhotoStyle)  | A Closed-form Solution to Photorealistic Image Stylization.   | `PyTorch`
| [pytorch-CycleGAN-and-pix2pix]( https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  | A Closed-form Solution to Photorealistic Image Stylization.   | `PyTorch`
| [maskrcnn-benchmark]( https://github.com/facebookresearch/maskrcnn-benchmark)  | Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch.   | `PyTorch`
| [deep-image-prior]( https://github.com/DmitryUlyanov/deep-image-prior)  | Image restoration with neural networks but without learning.   | `PyTorch`
| [StarGAN]( https://github.com/yunjey/StarGAN)  | StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation.   | `PyTorch`
| [faster-rcnn.pytorch]( https://github.com/jwyang/faster-rcnn.pytorch)  | This project is a faster faster R-CNN implementation, aimed to accelerating the training of faster R-CNN object detection models.   | `PyTorch`
| [pix2pixHD]( https://github.com/NVIDIA/pix2pixHD)  | Synthesizing and manipulating 2048x1024 images with conditional GANs.  | `PyTorch`
| [Augmentor]( https://github.com/mdbloice/Augmentor)  | Image augmentation library in Python for machine learning.  | `PyTorch`
| [albumentations]( https://github.com/albumentations-team/albumentations)  | Fast image augmentation library.   | `PyTorch`
| [Deep Video Analytics]( https://github.com/AKSHAYUBHAT/DeepVideoAnalytics)  | Deep Video Analytics is a platform for indexing and extracting information from videos and images   | `PyTorch`
| [semantic-segmentation-pytorch]( https://github.com/CSAILVision/semantic-segmentation-pytorch)  | Pytorch implementation for Semantic Segmentation/Scene Parsing on MIT ADE20K dataset.   | `PyTorch`
| [An End-to-End Trainable Neural Network for Image-based Sequence Recognition]( https://github.com/bgshih/crnn)  | This software implements the Convolutional Recurrent Neural Network (CRNN), a combination of CNN, RNN and CTC loss for image-based sequence recognition tasks, such as scene text recognition and OCR.   | `PyTorch`
| [UNIT]( https://github.com/mingyuliutw/UNIT)  | PyTorch Implementation of our Coupled VAE-GAN algorithm for Unsupervised Image-to-Image Translation.   | `PyTorch`
| [Neural Sequence labeling model]( https://github.com/jiesutd/NCRFpp)  | Sequence labeling models are quite popular in many NLP tasks, such as Named Entity Recognition (NER), part-of-speech (POS) tagging and word segmentation.   | `PyTorch`
| [faster rcnn]( https://github.com/longcw/faster_rcnn_pytorch)  | This is a PyTorch implementation of Faster RCNN. This project is mainly based on py-faster-rcnn and TFFRCNN. For details about R-CNN please refer to the paper Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.   | `PyTorch`
| [pytorch-semantic-segmentation]( https://github.com/ZijunDeng/pytorch-semantic-segmentation)  | PyTorch for Semantic Segmentation.   | `PyTorch`
| [EDSR-PyTorch]( https://github.com/thstkdgus35/EDSR-PyTorch)  | PyTorch version of the paper 'Enhanced Deep Residual Networks for Single Image Super-Resolution'.   | `PyTorch`
| [image-classification-mobile]( https://github.com/osmr/imgclsmob)  | Collection of classification models pretrained on the ImageNet-1K.   | `PyTorch`
| [FaderNetworks]( https://github.com/facebookresearch/FaderNetworks)  | Fader Networks: Manipulating Images by Sliding Attributes - NIPS 2017.   | `PyTorch`
| [neuraltalk2-pytorch]( https://github.com/ruotianluo/ImageCaptioning.pytorch)  | Image captioning model in pytorch (finetunable cnn in branch with_finetune).   | `PyTorch`
| [RandWireNN]( https://github.com/seungwonpark/RandWireNN)  | Implementation of: "Exploring Randomly Wired Neural Networks for Image Recognition".   | `PyTorch`
| [stackGAN-v2]( https://github.com/hanzhanggit/StackGAN-v2)  |Pytorch implementation for reproducing StackGAN_v2 results in the paper StackGAN++.   | `PyTorch`
| [Detectron models for Object Detection]( https://github.com/ignacio-rocco/detectorch)  | This code allows to use some of the Detectron models for object detection from Facebook AI Research with PyTorch.   | `PyTorch`
| [DEXTR-PyTorch]( https://github.com/scaelles/DEXTR-PyTorch)  | This paper explores the use of extreme points in an object (left-most, right-most, top, bottom pixels) as input to obtain precise object segmentation for images and videos.   | `PyTorch`
| [pointnet.pytorch]( https://github.com/fxia22/pointnet.pytorch)  | Pytorch implementation for "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.   | `PyTorch`
| [self-critical.pytorch]( https://github.com/ruotianluo/self-critical.pytorch) | This repository includes the unofficial implementation Self-critical Sequence Training for Image Captioning and Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.   | `PyTorch`
| [vnet.pytorch]( https://github.com/mattmacy/vnet.pytorch)  | A Pytorch implementation for V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.   | `PyTorch`
| [piwise]( https://github.com/bodokaiser/piwise)  | Pixel-wise segmentation on VOC2012 dataset using pytorch.   | `PyTorch`
| [pspnet-pytorch]( https://github.com/Lextal/pspnet-pytorch)  | PyTorch implementation of PSPNet segmentation network.   | `PyTorch`
| [pytorch-SRResNet]( https://github.com/twtygqyy/pytorch-SRResNet)  | Pytorch implementation for Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.   | `PyTorch`
| [PNASNet.pytorch]( https://github.com/chenxi116/PNASNet.pytorch)  | PyTorch implementation of PNASNet-5 on ImageNet.   | `PyTorch`
| [img_classification_pk_pytorch]( https://github.com/felixgwu/img_classification_pk_pytorch)  | Quickly comparing your image classification models with the state-of-the-art models.   | `PyTorch`
| [Deep Neural Networks are Easily Fooled]( https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks)  | High Confidence Predictions for Unrecognizable Images.   | `PyTorch`
| [pix2pix-pytorch]( https://github.com/mrzhu-cool/pix2pix-pytorch)  | PyTorch implementation of "Image-to-Image Translation Using Conditional Adversarial Networks".   | `PyTorch`
| [NVIDIA/semantic-segmentation]( https://github.com/NVIDIA/semantic-segmentation)  | A PyTorch Implementation of Improving Semantic Segmentation via Video Propagation and Label Relaxation, In CVPR2019.   | `PyTorch`
| [Neural-IMage-Assessment]( https://github.com/kentsyx/Neural-IMage-Assessment)  | A PyTorch Implementation of Neural IMage Assessment.   | `PyTorch`
| [torchxrayvision](https://github.com/mlmed/torchxrayvision) | Pretrained models for chest X-ray (CXR) pathology predictions. Medical, Healthcare, Radiology  | `PyTorch` | 

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### Caffe <a name="caffe"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [OpenPose]( https://github.com/CMU-Perceptual-Computing-Lab/openpose)  | OpenPose represents the first real-time multi-person system to jointly detect human body, hand, and facial keypoints (in total 130 keypoints) on single images.   | `Caffe`
| [Fully Convolutional Networks for Semantic Segmentation]( https://github.com/shelhamer/fcn.berkeleyvision.org)  | Fully Convolutional Models for Semantic Segmentation.   | `Caffe`
| [Colorful Image Colorization]( https://github.com/richzhang/colorization)  | Colorful Image Colorization.   | `Caffe`
| [R-FCN]( https://github.com/YuwenXiong/py-R-FCN)  | R-FCN: Object Detection via Region-based Fully Convolutional Networks.   | `Caffe`
| [cnn-vis]( https://github.com/jcjohnson/cnn-vis)  |Inspired by Google's recent Inceptionism blog post, cnn-vis is an open-source tool that lets you use convolutional neural networks to generate images.   | `Caffe`
| [DeconvNet]( https://github.com/HyeonwooNoh/DeconvNet)  | Learning Deconvolution Network for Semantic Segmentation.   | `Caffe`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

### MXNet <a name="mxnet"/>

| Model Name | Description | Framework |
|   :---:      |     :---:      |     :---:     |
| [Faster RCNN]( https://github.com/ijkguo/mx-rcnn)  | Region Proposal Network solves object detection as a regression problem.   | `MXNet`
| [SSD]( https://github.com/zhreshold/mxnet-ssd)  | SSD is an unified framework for object detection with a single network.   | `MXNet`
| [Faster RCNN+Focal Loss]( https://github.com/unsky/focal-loss)  | The code is unofficial version for focal loss for Dense Object Detection.   | `MXNet`
| [CNN-LSTM-CTC]( https://github.com/oyxhust/CNN-LSTM-CTC-text-recognition)  |I realize three different models for text recognition, and all of them consist of CTC loss layer to realize no segmentation for text images.   | `MXNet`
| [Faster_RCNN_for_DOTA]( https://github.com/jessemelpolio/Faster_RCNN_for_DOTA)  | This is the official repo of paper _DOTA: A Large-scale Dataset for Object Detection in Aerial Images_.  | `MXNet`
| [RetinaNet]( https://github.com/unsky/RetinaNet)  | Focal loss for Dense Object Detection.   | `MXNet`
| [MobileNetV2]( https://github.com/liangfu/mxnet-mobilenet-v2)  | This is a MXNet implementation of MobileNetV2 architecture as described in the paper _Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation_.   | `MXNet`
| [neuron-selectivity-transfer]( https://github.com/TuSimple/neuron-selectivity-transfer)  | This code is a re-implementation of the imagenet classification experiments in the paper _Like What You Like: Knowledge Distill via Neuron Selectivity Transfer_.   | `MXNet`
| [MobileNetV2]( https://github.com/chinakook/MobileNetV2.mxnet)  | This is a Gluon implementation of MobileNetV2 architecture as described in the paper _Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation_.   | `MXNet`
| [sparse-structure-selection]( https://github.com/TuSimple/sparse-structure-selection)  | This code is a re-implementation of the imagenet classification experiments in the paper _Data-Driven Sparse Structure Selection for Deep Neural Networks_.   | `MXNet`
| [FastPhotoStyle]( https://github.com/NVIDIA/FastPhotoStyle)  | A Closed-form Solution to Photorealistic Image Stylization.   | `MXNet`

<div align="right">
    <b><a href="#framework">↥ Back To Top</a></b>
</div>

***

## Contributions
Your contributions are always welcome!!
Please have a look at `contributing.md`

## License

[MIT License](LICENSE)
