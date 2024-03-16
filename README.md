# CS6910 Assignment 1

### Contents
- `Train.py`: Main file for training the neural network
- `Assignment_1/`
  - `Requirements.txt`: txt file containing all the python libraries needed to train NN.
  - `hyperparameter_config.py`: the file containing all the hyperparameters and their default values.
  - `neural_network.py`: contains the class NeuralNetwork and all the functions that define the NN architectures used.

### Instructions for running the Neural Network code
To train the neural network, please follow the steps as given below:
1. Import the required libraries by typing `pip install -r Assignment_1/Requirements.txt`.
2. Please put your Wandb API key in `Train.py` before running the file to track the runs.
3. Run `Python Train.py`, to run on default parameters.
4. Run `Python Train.py --parameters`, to use your parameters.
   - Example: `Python Train.py --batch_size 64` to run the NN with batch size of 64.
5. Track the progress of runs and sweeps on [wandb.ai](https://wandb.ai/home).

Link to the wandb report: [Link](https://wandb.ai/cs23m024-gaurav/CS6910_Assignment_1/reports/Copy-of-oikantik-s-CS6910-Assignment-1--Vmlldzo3MTYxODMy)

### Hyperparameters and their default values
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | CS6910_Assignment_1 | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity` | Gaurav  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
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
