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
