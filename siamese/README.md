# MNIST classification with Siamese networks
Siamese CNNs work by comparing the outputs of different images from the same CNN, and determining whether or not the two images are similar based on the distances between the outputs. For more information on Siamese CNNs, check out my [other project on Siamese CNNs](https://github.com/gabrielwong159/siamese).

## Model architecture
![Model architecture](https://camo.githubusercontent.com/b27757e11d8687dc846b016e0fac80a544e7b645/68747470733a2f2f736f72656e626f756d612e6769746875622e696f2f696d616765732f5369616d6573655f6469616772616d5f322e706e67)

Here we see an example model architecture for a Siamese CNN. While the filter and kernel sizes differ, the idea is essentially the same:
1. Convolutional layers
2. Fully-connected layers
3. L1-distance (or sometimes L2)
4. Fully-connected layers
5. L2-distance

The loss function reduces the distances produced by the CNN between two similar images, and increases the distances for dissimilar images.

## Resources
Check out the [`References`](https://github.com/gabrielwong159/siamese#references) section of my other project.
