import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import wandb
import hyperparameter_config
wandb.login(key="Your-API-Key")

from neural_network import NeuralNetwork, train_neural_network

def printImages(x_train, y_train):
  classes = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle Boot"}
  index = [list(y_train).index(i) for i in range(len(classes))]

  # image --> An image in a class; labels --> label
  images = []
  labels = []
  for i in index:
    images.append(x_train[i])
    labels.append(classes[y_train[i]])
  wandb.log({"Images": [wandb.Image(image, caption=caption) for image, caption in zip(images, labels)]}, step=i)

def main(args):
  
  # Taking dataset according to parameters passed by user
  if(args.dataset == "fashion_mnist"):
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  elif(args.dataset == "mnist"):  
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # Train test split using sklearn library
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=10000, random_state=42)
  
  #Labels of the dataset
  class_names = []
  if(args.dataset == "fashion_mnist"):
    class_names=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
  elif(args.dataset == "mnist"):
    class_names = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

  # Normalizing the pixels to avoid overflow
  x_train_norm = x_train / 255
  x_test_norm = x_test / 255
  x_val_norm = x_val / 255

  # 28 * 28 = 784 pixels
  num_of_pixels = 28 * 28

  output_neurons = 10
  
  #Taking pixels as inputs
  x_train_input = x_train_norm.reshape(len(x_train_norm), num_of_pixels)                      #flattening the image into 1d array
  x_test_input = x_test_norm.reshape(len(x_test_norm), num_of_pixels)                         #same thing
  x_val_reshape = x_val_norm.reshape(len(x_val_norm), num_of_pixels)
  
  # Taking transpose of the dataset, so it becomes 784 x 50000 meaning each column represents an image
  x_train_input = x_train_input.T
  x_test_input = x_test_input.T
  x_val = x_val_reshape.T

  
  # Define hyperparameters
  sweep_config = {
     'method' : 'random',
     'wandb_entity' : args.wandb_entity,
     'name' : args.wandb_project,
     'metric' : {
        'name' : 'val_accuracy',
        'goal' : 'maximize',
     },
     'parameters' : {
        'eta' : {
           'values' : [args.learning_rate]
        },
        'epochs' : {
           'values' : [args.epochs]
        },
        'num_hidden_layers' : {
           'values' : [args.num_layers]
        },
        'num_hidden_neurons' : {
           'values' : [args.hidden_size]
        },
        'activation_function' : {
           'values' : [args.activation]
        },
        'initialization' : {
           'values' : [args.weight_init]
        },
        'optimizer' : {
           'values' : [args.optimizer]
        },
        'batch_size' : {
           'values' : [args.batch_size]
        },
        'momentum' : {
           'values' : [args.momentum]
        },
        'beta' : {
           'values' : [args.beta]
        },
        'beta1' : {
           'values' : [args.beta1]
        },
        'beta2' : {
           'values' : [args.beta2]
        },
        'eps' : {
           'values' : [args.epsilon]
        },
        'weight_decay' : {
           'values' : [args.weight_decay]
        },
        'loss' : {
           'values' : [args.loss]
        }
     }
  }


  run_name = ""

  def train():
    with wandb.init() as run:
      
      # Creates names of runs based on parameters. Example => hl_4_bs_64_ac_reLU
      config = wandb.config
      run_name = "hl_" + str(config.num_hidden_layers) + "_bs_" + str(config.batch_size) + "_ac_" + config.activation_function
      wandb.run.name = run_name

    #   printImages(x_train, y_train)           ---> run when want to print the images 
      

      # creating the list of hidden_neurons
      hidden_neurons_list = []
      for i in range(config.num_hidden_layers):
        hidden_neurons_list.append(config.num_hidden_neurons)

      # Creates an object of class NeuralNetwork and Initializes the parameters   
      nn = NeuralNetwork(num_of_pixels, hidden_neurons_list, config.num_hidden_layers, output_neurons)
      weights, biases, prev_weights, prev_biases = nn.initialize_parameters(num_of_pixels, hidden_neurons_list, config.num_hidden_layers, output_neurons, config.initialization)
      
      # Train the network
      weights, biases = train_neural_network(nn, x_train_input, y_train, x_test_input, y_test, x_val, y_val, weights, biases, prev_weights, prev_biases, config.num_hidden_layers, config.activation_function,
                                             config.optimizer, config.epochs, config.batch_size, config.eta, config.momentum, config.beta, config.beta1, config.beta2, config.eps, config.weight_decay, config.loss)

      
      # Print the confusion matrix
      _, _, y_test_pred = nn.feedforward_propagation(x_test_input, weights, biases, config.num_hidden_layers, config.activation_function)
      y_test_pred = np.argmax(y_test_pred, axis=0)
      conf_matrix = wandb.plot.confusion_matrix(y_true = y_test, preds = y_test_pred, class_names = class_names)
      wandb.sklearn.plot_confusion_matrix(y_test, y_test_pred, class_names)
  
  
  sweep_id = wandb.sweep(sweep=sweep_config, project='Testing')
  wandb.agent(sweep_id, function=train,count=1)
  wandb.finish()

if __name__ == "__main__":
    args = hyperparameter_config.configParse()
    main(args)
