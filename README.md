# CS6910 Assignment 1

### Contents
- `Train.py`: Containing main function for training the neural network
  
- `Assignment_1/`
  
  - `Requirements.txt`: txt file containing all the Python libraries needed to train NN.
    
  - `hyperparameter_config.py`: the file containing all the hyperparameters and their default values.
    
  - `neural_network.py`: contains the class NeuralNetwork and all the functions that define the NN architectures used.

<br>

### Instructions for running the Neural Network code
To train the neural network, please follow the steps given below:

- Import the required libraries:
   ```
   pip install -r Assignment_1/Requirements.txt

- Please put your Wandb API key in `Train.py` before running the file to track the runs.

   
- Run the below code to run on [default parameters](https://github.com/GauravKanwat/DL-CS6910-Assignment-1?tab=readme-ov-file#hyperparameters-and-their-default-values).
   ```
   Python Train.py
   
- Use your parameters:
    - Example: `Python Train.py --batch_size 64` to run the NN with a batch size 64.

<br>

Link to the wandb report: [Link](https://api.wandb.ai/links/cs23m024-gaurav/uqtf06z1)

<br>

### Dataset

I have used the `fashion_mnist` and `mnist` datasets provided by Tensorflow keras.

Training has been performed on the `fashion_mnist` dataset.

It consist of 60,000 grayscale images of clothing items from 10 different categories, with each image being 28x28 pixels in size. I have split the dataset into a training set, a validation set, and a test set.

- Training Set: This portion of the dataset contains 50,000 images. These images are used to train machine learning models, allowing them to learn patterns and features present in the data. Each image in the training set is associated with a label indicating the category of clothing it represents.

- Validation Set: The validation set consists of 10,000 images. It is used to fine-tune model hyperparameters and monitor the model's performance during training. By evaluating the model on the validation set at regular intervals, adjustments can be made to improve the model's generalization ability and prevent overfitting.

- Test Set: The test set also contains 10,000 images and serves as an independent dataset to evaluate the final performance of the trained model. The model has not seen these images during training or validation, allowing for an unbiased assessment of its accuracy and effectiveness in classifying unseen data.

<br>

### Model Architecture

In a typical feedforward neural network architecture, there are three main types of layers:

1. Input Layer: This layer consists of neurons that receives the input data. The number of neurons in the input layer corresponds to the number of dimensions in the input data.

2. Hidden Layers: These layers are positioned between the input and output layers and are responsible for learning complex patterns and representations from the input data. Each hidden layer consists of multiple neurons, and the number of hidden layers and neurons per layer is determined based on the complexity of the problem and the amount of available data.

3. Output Layer: This layer produces the final output of the neural network. The number of neurons in the output layer depends on the input. For example, in a binary classification task, there may be one neuron representing each class, while in a multi-class classification task, there may be multiple neurons, each corresponding to a different class.

Additionally, each neuron in the network is associated with an activation function, which introduces non-linearity into the model and allows it to learn complex relationships in the data. Activation functions that I have used are sigmoid, tanh, ReLU (Rectified Linear Unit), and identity.

<br>

### Training Procedure

1. **Initialization:** Initialize the neural network model architecture with the specified number of input units, hidden layers, hidden neurons, and output units. Also, initialize the weights and biases of the network, either randomly or using a specific initialization method such as Xavier.

2. **Forward Propagation:** Perform forward propagation to compute the predicted outputs of the neural network for a given input batch. This involves passing the input data through each layer of the network, applying activation functions, and generating the final output predictions.

3. **Loss Computation:** Calculate the loss function value, which measures the difference between the predicted outputs and the actual target labels. For our assignment, we used both cross-entropy and mean squared error (MSE) loss functions.

4. **Backward Propagation:** Conduct backward propagation to compute the gradients of the loss function with respect to the weights and biases of the network. This involves applying the chain rule to propagate the error backward through the network, computing the gradients at each layer.

5. **Parameter Updates:** Update the weights and biases of the network using an optimization algorithm such as gradient descent, Momentum gradient, Nesterov accelerated gradient, RMSProp, Adam, or Nadam. This step adjusts the model parameters to minimize the loss function and improve performance.

6. **Validation and Testing:** Evaluate the trained model on a validation dataset to monitor its performance and prevent overfitting. Compute metrics such as accuracy, loss, and other relevant measures. Finally, assess the model's generalization capability on a separate test dataset.

<br>

### Hyperparameters and their default values
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | CS6910_Assignment_1 | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity` | CS23M024  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `--epochs` | 10 |  Number of epochs to train neural network.|
| `--batch_size` | 64 | Batch size used to train neural network. | 
| `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `--optimizer` | nadam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `--beta2` | 0.999 | Beta2 used by adam and nadam optimizers. |
| `--epsilon` | 1e-8 | Epsilon used by optimizers. |
| `--weight_decay` | 0.0 | Weight decay used by optimizers. |
| `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. | 
| `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |
<br>
