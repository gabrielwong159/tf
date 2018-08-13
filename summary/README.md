# Summaries and TensorBoard
TensorBoard helps to visualize TensorFlow graphs, as well as monitor various metrics that your graph might have. 

This project is an example of how to add summaries to a piece of TensorFlow code, before writing the results via `tf.summary.FileWriter`. This allows us to visualize the weight and bias variables across training, as well as the loss and accuracy via TensorBoard.

<p align="center">
    <img alt="Visualizing training loss using TensorBoard"
         src="https://www.tensorflow.org/images/mnist_tensorboard.png" />
</p>

In addition, this allows you to visualize your graph using TensorBoard.

<p align="center">
    <img alt="Visualizing TensorFlow graphs using TensorBoard"
         src="https://www.tensorflow.org/images/colorby_structure.png" />
</p>

In this example we use the same MNIST model in the `mnist` project as our base architecture for training. We then log the model's training over 5000 iterations, as well as the testing on every 10th step.

## Resources
* [TensorBoard: Visualizing Learning](https://www.tensorflow.org/guide/summaries_and_tensorboard)
* [TensorBoard: Graph Visualization](https://www.tensorflow.org/guide/graph_viz)
* [Predicting Movie Review Sentiment with TensorFlow and TensorBoard](https://medium.com/@Currie32/predicting-movie-review-sentiment-with-tensorflow-and-tensorboard-53bf16af0acf)
