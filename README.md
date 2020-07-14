# Pretrained-model

## What is pre-trained Model?
A pre-trained model is a model created by some one else to solve a similar problem. Instead of building a model from scratch to solve a similar problem, we can use the model trained on other problem as a starting point. A pre-trained model may not be 100% accurate in your application.

For example, if you want to build a self learning car. You can spend years to build a decent image recognition algorithm from scratch or you can take inception model (a pre-trained model) from Google which was built on [ImageNet](http://www.image-net.org/) data to identify images in those pictures.

#### Computer Vision
* [Image Classification](#image_classification)
* [Object Detection & Image Segmentation](#object_detection)
* [Image Manipulation](#image_manipulation)

#### Natural Language Processing
* [Machine Comprehension](#machine_comprehension)
* [Machine Translation](#machine_translation)
* [Language Modelling](#language)



### Image Classification <a name="image_classification"/>
This collection of models take images as input, then classifies the major objects in the images into 1000 object categories such as keyboard, mouse, pencil, and many animals.

|Model Class |Reference |Description |
|-|-|-|
|<b>[MobileNet](vision/classification/mobilenet)</b>|[Sandler et al.](https://arxiv.org/abs/1801.04381)|Light-weight deep neural network best suited for mobile and embedded vision applications. <br>Top-5 error from paper - ~10%|
|<b>[ResNet](vision/classification/resnet)</b>|[He et al.](https://arxiv.org/abs/1512.03385)|A CNN model (up to 152 layers). Uses shortcut connections to achieve higher accuracy when classifying images. <br> Top-5 error from paper - ~3.6%|
|<b>[SqueezeNet](vision/classification/squeezenet)</b>|[Iandola et al.](https://arxiv.org/abs/1602.07360)|A light-weight CNN model providing AlexNet level accuracy with 50x fewer parameters. <br>Top-5 error from paper - ~20%|
|<b>[VGG](vision/classification/vgg)</b>|[Simonyan et al.](https://arxiv.org/abs/1409.1556)|Deep CNN model(up to 19 layers). Similar to AlexNet but uses multiple smaller kernel-sized filters that provides more accuracy when classifying images. <br>Top-5 error from paper - ~8%|
|<b>[AlexNet](vision/classification/alexnet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|A Deep CNN model (up to 8 layers) where the input is an image and the output is a vector of 1000 numbers. <br> Top-5 error from paper - ~15%|
|<b>[GoogleNet](vision/classification/inception_and_googlenet/googlenet)</b>|[Szegedy et al.](https://arxiv.org/pdf/1409.4842.pdf)|Deep CNN model(up to 22 layers). Comparatively smaller and faster than VGG and more accurate in detailing than AlexNet. <br> Top-5 error from paper - ~6.7%|
|<b>[CaffeNet](vision/classification/caffenet)</b>|[Krizhevsky et al.]( https://ucb-icsi-vision-group.github.io/caffe-paper/caffe.pdf)|Deep CNN variation of AlexNet for Image Classification in Caffe where the max pooling precedes the local response normalization (LRN) so that the LRN takes less compute and memory.|
|<b>[RCNN_ILSVRC13](vision/classification/rcnn_ilsvrc13)</b>|[Girshick et al.](https://arxiv.org/abs/1311.2524)|Pure Caffe implementation of R-CNN for image classification. This model uses localization of regions to classify and extract features from images.|
|<b>[DenseNet-121](vision/classification/densenet-121)</b>|[Huang et al.](https://arxiv.org/abs/1608.06993)|Model that has every layer connected to every other layer and passes on its own feature providing strong gradient flow and more diversified features.|
|<b>[Inception_V1](vision/classification/inception_and_googlenet/inception_v1)</b>|[Szegedy et al.](https://arxiv.org/abs/1409.4842)|This model is same as GoogLeNet, implemented through Caffe2 that has improved utilization of the computing resources inside the network and helps with the vanishing gradient problem. <br> Top-5 error from paper - ~6.7%|
|<b>[Inception_V2](vision/classification/inception_and_googlenet/inception_v2)</b>|[Szegedy et al.](https://arxiv.org/abs/1512.00567)|Deep CNN model for Image Classification as an adaptation to Inception v1 with batch normalization. This model has reduced computational cost and improved image resolution compared to Inception v1. <br> Top-5 error from paper ~4.82%|
|<b>[ShuffleNet_V1](vision/classification/shufflenet)</b>|[Zhang et al.](https://arxiv.org/abs/1707.01083)|Extremely computation efficient CNN model that is designed specifically for mobile devices. This model greatly reduces the computational cost and provides a ~13x speedup over AlexNet on ARM-based mobile devices. Compared to MobileNet, ShuffleNet achieves superior performance by a significant margin due to it's efficient structure. <br> Top-1 error from paper - ~32.6%|
|<b>[ShuffleNet_V2](vision/classification/shufflenet)</b>|[Zhang et al.](https://arxiv.org/abs/1807.11164)|Extremely computation efficient CNN model that is designed specifically for mobile devices. This network architecture design considers direct metric such as speed, instead of indirect metric like FLOP. <br> Top-1 error from paper - ~30.6%|
|<b>[ZFNet-512](vision/classification/zfnet-512)</b>|[Zeiler et al.](https://arxiv.org/abs/1311.2901)|Deep CNN model (up to 8 layers) that increased the number of features that the network is capable of detecting that helps to pick image features at a finer level of resolution. <br> Top-5 error from paper - ~14.3%|
<hr>



### Object Detection & Image Segmentation <a name="object_detection"/>
Object detection models detect the presence of multiple objects in an image and segment out areas of the image where the objects are detected. Semantic segmentation models partition an input image by labeling each pixel into a set of pre-defined categories.

|Model Class |Reference |Description |
|-|-|-|
|<b>[Tiny YOLOv2](vision/object_detection_segmentation/tiny-yolov2)</b>|[Redmon et al.](https://arxiv.org/pdf/1612.08242.pdf)|A real-time CNN for object detection that detects 20 different classes. A smaller version of the more complex full YOLOv2 network.|
|<b>[SSD](vision/object_detection_segmentation/ssd)</b>|[Liu et al.](https://arxiv.org/abs/1512.02325)|Single Stage Detector: real-time CNN for object detection that detects 80 different classes.|
|<b>[Faster-RCNN](vision/object_detection_segmentation/faster-rcnn)</b>|[Ren et al.](https://arxiv.org/abs/1506.01497)|Increases efficiency from R-CNN by connecting a RPN with a CNN to create a single, unified network for object detection that detects 80 different classes.|
|<b>[Mask-RCNN](vision/object_detection_segmentation/mask-rcnn)</b>|[He et al.](https://arxiv.org/abs/1703.06870)|A real-time neural network for object instance segmentation that detects 80 different classes. Extends Faster R-CNN as each of the 300 elected ROIs go through 3 parallel branches of the network: label prediction, bounding box prediction and mask prediction.|
|<b>[RetinaNet](vision/object_detection_segmentation/retinanet)</b>|[Lin et al.](https://arxiv.org/abs/1708.02002)|A real-time dense detector network for object detection that addresses class imbalance through Focal Loss. RetinaNet is able to match the speed of previous one-stage detectors and defines the state-of-the-art in two-stage detectors (surpassing R-CNN).|
|<b>[YOLO v2](vision/object_detection_segmentation/yolov2)</b>|[Redmon et al.](https://arxiv.org/abs/1612.08242)|A CNN model for real-time object detection system that can detect over 9000 object categories. It uses a single network evaluation, enabling it to be more than 1000x faster than R-CNN and 100x faster than Faster R-CNN.
|<b>[YOLO v2-coco](vision/object_detection_segmentation/yolov2-coco)</b>|[Redmon et al.](https://arxiv.org/abs/1612.08242)|A CNN model for real-time object detection system that can detect over 9000 object categories. It uses a single network evaluation, enabling it to be more than 1000x faster than R-CNN and 100x faster than Faster R-CNN. This model is trained with COCO dataset and contains 80 classes.
|<b>[YOLO v3](vision/object_detection_segmentation/yolov3)</b>|[Redmon et al.](https://arxiv.org/pdf/1804.02767.pdf)|A deep CNN model for real-time object detection that detects 80 different classes. A little bigger than YOLOv2 but still very fast. As accurate as SSD but 3 times faster.|
|<b>[Tiny YOLOv3](vision/object_detection_segmentation/tiny-yolov3)</b>|[Redmon et al.](https://arxiv.org/pdf/1804.02767.pdf)| A smaller version of YOLOv3 model. |
|<b>[YOLOv4](vision/object_detection_segmentation/yolov4)</b>|[Bochkovskiy et al.](https://arxiv.org/abs/2004.10934)|Optimizes the speed and accuracy of object detection. Two times faster than EfficientDet. It improves YOLOv3's AP and FPS by 10% and 12%, respectively, with mAP50 of 52.32 on the COCO 2017 dataset and FPS of 41.7 on Tesla 100.|
|<b>[DUC](vision/object_detection_segmentation/duc)</b>|[Wang et al.](https://arxiv.org/abs/1702.08502)|Deep CNN based pixel-wise semantic segmentation model with >80% [mIOU](/models/semantic_segmentation/DUC/README.md/#metric) (mean Intersection Over Union). Trained on cityscapes dataset, which can be effectively implemented in self driving vehicle systems.|
|FCN|[Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)|Deep CNN based segmentation model trained end-to-end, pixel-to-pixel that produces efficient inference and learning. Built off of AlexNet, VGG net, GoogLeNet classification methods. <br>[contribute](contribute.md)|
<hr>



### Image Manipulation <a name="image_manipulation"/>
Image manipulation models use neural networks to transform input images to modified output images. Some popular models in this category involve style transfer or enhancing images by increasing resolution.

|Model Class |Reference |Description |
|-|-|-|
|Unpaired Image to Image Translation using Cycle consistent Adversarial Network|[Zhu et al.](https://arxiv.org/abs/1703.10593)|The model uses learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. <br>[contribute](contribute.md)|
|<b>[Super Resolution with sub-pixel CNN](vision/super_resolution/sub_pixel_cnn_2016)</b> |	[Shi et al.](https://arxiv.org/abs/1609.05158)	|A deep CNN that uses sub-pixel convolution layers to upscale the input image. |
|<b>[Fast Neural Style Transfer](vision/style_transfer/fast_neural_style)</b> |	[Johnson et al.](https://arxiv.org/abs/1603.08155)	|This method uses a loss network pretrained for image classification to define perceptual loss functions that measure perceptual differences in content and style between images. The loss network remains fixed during the training process.|
<hr>



### Language Modelling <a name="language"/>
This subset of natural language processing models learns representations of language from large corpuses of text.

|Model Class |Reference |Description |
|-|-|-|
|Deep Neural Network Language Models | [Arisoy et al.](https://pdfs.semanticscholar.org/a177/45f1d7045636577bcd5d513620df5860e9e5.pdf)|A DNN acoustic model. Used in many natural language technologies. Represents a probability distribution over all possible word strings in a language. <br> [contribute](contribute.md)|
<hr>

### Machine Comprehension <a name="machine_comprehension"/>
This subset of natural language processing models that answer questions about a given context paragraph.

|Model Class |Reference |Description |
|-|-|-|
|<b>[Bidirectional Attention Flow](text/machine_comprehension/bidirectional_attention_flow)</b>|[Seo et al.](https://arxiv.org/pdf/1611.01603)|A model that answers a query about a given context paragraph.|
|<b>[BERT-Squad](text/machine_comprehension/bert-squad)</b>|[Devlin et al.](https://arxiv.org/pdf/1810.04805.pdf)|This model answers questions based on the context of the given input paragraph. |
|<b>[GPT-2](text/machine_comprehension/gpt-2)</b>|[Radford et al.](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)|A large transformer-based language model that given a sequence of words within some text, predicts the next word. |
<hr>

### Machine Translation <a name="machine_translation"/>
This class of natural language processing models learns how to translate input text to another language.

|Model Class |Reference |Description |
|-|-|-|
|Neural Machine Translation by jointly learning to align and translate|	[Bahdanau et al.](https://arxiv.org/abs/1409.0473)|Aims to build a single neural network that can be jointly tuned to maximize the translation performance. <br>[contribute](contribute.md)|
|Google's Neural Machine Translation System|	[Wu et al.](https://arxiv.org/abs/1609.08144)|This model helps to improve issues faced by the Neural Machine Translation (NMT) systems like parallelism that helps accelerate the final translation speed.<br>[contribute](contribute.md)|
<hr>


