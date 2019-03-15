# ROI pooling
Perform ROI pooling by modifying the kernel and stride sizes on TensorFlow's max-pooling operation.

## How it works
Given an ordinary max-pooling operation, the output size of the image can be given by the following equation
<p align="center">
	<img src="https://latex.codecogs.com/gif.latex?O&space;=&space;\frac{I&space;-&space;K&space;&plus;&space;2P}{S}&space;&plus;&space;1" title="O = \frac{I - K + 2P}{S} + 1" />
</p>

where O is the output size, I is the input size, K is the kernel size, P is the amount of padding, and S is the stride size.

We first assume that there is no padding, i.e. `P = 0`. Rearranging the remaining terms, we obtain
<p align="center">
	<img src="https://latex.codecogs.com/gif.latex?I&space;=&space;K&space;&plus;&space;S&space;\cdot&space;(O&space;-&space;1)" title="I = K + S \cdot (O - 1)" />
</p>

Given that `I` and `O` are known variables, this leaves us with two unknowns: `K` and `S`.

I attempted to determine the kernel size `K` first, in order to ensure that the kernel size was large enough to cover the entire area of the image. However, remaining calculations on the stride caused images with particular sizes to have differently sized outputs. This was solved by fixing the stride, _then_ calculating the kernel size.

For the stride to be as even as possible, we take
<p align="center">
	<img src="https://latex.codecogs.com/gif.latex?S&space;=&space;\left&space;\lfloor&space;\frac{I}{O}&space;\right&space;\rfloor" title="S = \left \lfloor \frac{I}{O} \right \rfloor" />
</p>

Consequently, we compute
<p align="center">
	<img src="https://latex.codecogs.com/gif.latex?K&space;=&space;W&space;-&space;S&space;\cdot&space;(O&space;-&space;1)" title="K = W - S \cdot (O - 1)" />
</p>

## Implementation
The implementation of the max-pooling above is found in the model in [`maxpool.py`](models/maxpool.py#L44).

In this example, we train a regression model for object localization on fixed size images. This is found in [`train_nopool.ipynb`](./train_nopool.ipynb).

We then increase the size of the images, and use ROI pooling to ensure that the fully-connected layer is the same size as in the original. The results are shown in [`test_maxpool.ipynb`](./test_maxpool.ipynb).