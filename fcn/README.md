# Fully Convolutional Networks (FCNs)
Semantic segmentation is the task of labelling each pixel in an image with a class. Typical classification tasks with CNNs utilise fully-connected layers to produce class probabilities for an entire image.

In FCNs, the fully-connected layers are replaced with transposed convolutions to yield an output with the same height and width as the original image. This allows the model to then produce pixel-level classifications.

![Semantic segmentation](https://cdn-images-1.medium.com/max/1600/1*Tp5J-s8dhAaFWuQAIlhmeg.png)

## Architecture
For feature extraction, we use a VGG-16 network with pre-trained weights. The three fully-connected layers are discarded. **Note that when using pre-trained weights for VGG-16, it is highly advisable to subtract all input images by the VGG-16 mean.**

Since the feature extraction is based on the VGG-16 network, images are downsampled by `32x`, and thus image dimensions should be a multiple of 32.

![FCN architecture overview](https://devblogs.nvidia.com/wp-content/uploads/2016/11/figure15.png)

In the original paper for FCNs, the author executed two 2x-upsampling layers, followed by an 8x-upsampling. I later found that extending the network to four 2x-upsampling layers, followed by a final 2x-upsampling allows the model to converge faster, as well as produce higher accuracy in a binary mask generation task. Accuracy here is measured by [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient).

However, in Feature Pyramid Networks (FPNs), the first convolutional layer is skipped due to its `large memory footprint`. In addition, another `3x3` convolution is added to the upsampled results to `reduce the aliasing effect of upsampling`. These are additions to the model that can be considered.

    
## Resources
* [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
* [Original implementation in Caffe](https://github.com/shelhamer/fcn.berkeleyvision.org)
* [Referenced TensorFlow implementation of FCN](https://github.com/warmspringwinds/tf-image-segmentation)
* [U-net (more commonly used FCN)](https://arxiv.org/abs/1505.04597)
* [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
* [Ultrasound Nerve Segmentation (Kaggle competition for semantic segmentation)](https://www.kaggle.com/c/ultrasound-nerve-segmentation)