# Multi-task learning
Multi-task learning is widely used when there are multiple tasks given a single object. For example, in [Mask R-CNN](https://arxiv.org/abs/1703.06870), a feature map produced by a CNN is used for both object classification, and bounding box regression. Both the classification and regression tasks utilise the same CNN, differing only in the fully-connected layers.

![Example of multi-task learning architecture](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/more_images/LocalizationRegression2.png)

In this project, we explore multi-task learning with two classification tasks: digit type and colour. Our input data comes from the MNIST dataset, but each digit is also assigned a random colour from the set `{red, orange, yellow, green, blue, indigo, violet}`. The CNN is trained simultaneously on both classification tasks by taking the sum of cross entropy losses from both tasks.

## Getting started
The code to generate sample images can be found in `data_loader.py`. See `test_data_loader.ipynb` for a visual understanding of the generated images.

Model code can be found in `model.py`, note how there is only one CNN, but two fully-connected networks, as well as two corresponding cross entropy losses.

Training and testing can be found in `train.py` and `evaluate_model.ipynb` respectively.

## Resources
[An Overview of Multi-Task Learning in Deep Neural Networks: Sebastian Ruder](http://ruder.io/multi-task/)
