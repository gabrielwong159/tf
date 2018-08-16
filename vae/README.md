# Variational autoencoders
Variational autoencoders (VAEs) are used as **generative models**, much like _generative adversarial networks_ (GANs). A [blog post by `kvfrans`](http://kvfrans.com/variational-autoencoders-explained/) describes the advantages that VAEs have over GANs.

## Architecture
VAEs have two primary components: an **encoder** and a **decoder**. The encoder functions is very much like a typical CNN, in which we obtain a vector representation of an image. This is done via feature extraction with convolutional layers, followed by fully-connected layers to obtain the vector. The decoder takes this particular vector representation, and attempts to re-construct the original image through upsampling.

At this point we have built an autoencoder. However, this is less useful in which the model is only trained to reproduce the input image. In order to solve this, we constrain the encoded vector representation of each image, the _latent vectors_, to conform to a _unit Gaussian distribution_. Subsequently, by sampling vectors from the Gaussian distribution we can generate new images.

However, this produces a trade-off between the accuracy of the re-construction, and how closely the latent vectors match the target Gaussian distribution. As such, we train the model simultaneously on both tasks. _Binary cross-entropy_ between the generated image and the input image is used as the loss function for the images, while [_Kullback-Leibler divergence_](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) is used as the loss function for the latent vectors. By minimizing the KL divergence between the latent vectors and samples from a true Gaussian distribution, we are able to allow the latent vectors to approximate our target Gaussian distribution.

## Evaluation
To observe the distribution of the resultant latent vectors, as well as a range of generated images, see the [Jupyter notebook](./evaluate_model.ipynb).


## Resources
* [Auto-Encoding Variational Bayes (_arXiv, 2013_)](https://arxiv.org/abs/1312.6114)
* [`transcranial` VAE demo](https://transcranial.github.io/keras-js/#/mnist-vae)
* [`transcranial` Keras implementation](https://github.com/transcranial/keras-js/blob/master/notebooks/demos/mnist_vae.ipynb)
* [Great blog post by `kvfrans` describing VAEs](http://kvfrans.com/variational-autoencoders-explained/)
* [`kvfrans` TensorFlow implementation with stddev as encoder output](https://github.com/kvfrans/variational-autoencoder)
* [VAE implementation with MSE for image loss](https://towardsdatascience.com/teaching-a-variational-autoencoder-vae-to-draw-mnist-characters-978675c95776)
* [VAE implementation in Kaggle kernel](https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder)