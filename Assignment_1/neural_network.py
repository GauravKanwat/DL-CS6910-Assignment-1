import numpy as np
from tqdm import tqdm
import wandb

class NeuralNetwork:
    def __init__(self, num_of_pixels, hidden_neurons_list, num_hidden_layers, output_neurons):

      self.num_of_pixels = num_of_pixels
      self.hidden_neurons_list = hidden_neurons_list
      self.num_hidden_layers = num_hidden_layers
      self.output_neurons = output_neurons


    def initialize_parameters(self, num_of_pixels, hidden_neurons_list, num_hidden_layers, output_neurons, initialization):
      
      weights = {}
      biases = {}
      prev_weights = {}
      prev_biases = {}

      # Xavier initialization
      if initialization == "Xavier":
        weights[0] = np.random.randn(hidden_neurons_list[0], num_of_pixels) * np.sqrt(1 / num_of_pixels)
        biases[0] = np.zeros((hidden_neurons_list[0], 1))

        # Initialize weights and biases for hidden layers
        for l in range(1, len(hidden_neurons_list)):
          weights[l] = np.random.randn(hidden_neurons_list[l], hidden_neurons_list[l-1]) * np.sqrt(1 / hidden_neurons_list[l-1])
          biases[l] = np.zeros((hidden_neurons_list[l], 1))

        # Initialize weights for last hidden layer to output layer
        weights[len(hidden_neurons_list)] = np.random.randn(output_neurons, hidden_neurons_list[-1]) * np.sqrt(1 / hidden_neurons_list[-1])
        biases[len(hidden_neurons_list)] = np.zeros((output_neurons, 1))

        # Initialize previous weights and biases
        for l in range(num_hidden_layers + 1):
          prev_weights[l] = np.zeros_like(weights[l])
          prev_biases[l] = np.zeros_like(biases[l])

        return weights, biases, prev_weights, prev_biases

      # Random initialization
      elif initialization == "random":
        weights[0] = np.random.rand(hidden_neurons_list[0], num_of_pixels) - 0.5
        biases[0] = np.random.rand(hidden_neurons_list[0], 1) - 0.5
        for l in range(num_hidden_layers):
          weights[l] = np.random.rand(hidden_neurons_list[l], num_of_pixels if l == 0 else hidden_neurons_list[l-1]) - 0.5
          biases[l] = np.random.rand(hidden_neurons_list[l], 1) - 0.5
        weights[num_hidden_layers] = np.random.rand(output_neurons, hidden_neurons_list[-1]) - 0.5
        biases[num_hidden_layers] = np.random.rand(output_neurons, 1) - 0.5

        for l in range(num_hidden_layers + 1):
          prev_weights[l] = np.zeros_like(weights[l])
          prev_biases[l] = np.zeros_like(biases[l])
      return weights, biases, prev_weights, prev_biases
      '''
          Initializing the weights and biases, both are dictionary which are storing random values generated by rand between (0 to 1) and subtracting 0.5 from it makes it between
          -0.5 to 0.5
      '''

    def sigmoid(self, x):
      sigmoid_x = np.where(x < -30, 1, 1 / (1 + np.exp(-x)))
      return sigmoid_x

    def reLU(self, Z):
        return np.maximum(0, Z)

    def tanh(self, x):
      return np.tanh(x)

    def identity(self, x):
      return x

    
    def softmax(self, x):
        max_x = np.max(x, axis=0)

        # avoiding overflow
        exp_x = np.exp(x - max_x)
        return exp_x / np.sum(exp_x, axis=0)
        
    
    def diff_mse(self, pred_output, Y):
      one_hot_Y = self.one_hot(Y)
      return np.multiply((pred_output - one_hot_Y), np.multiply(pred_output, (1 - pred_output)))
    


    # Feedforward Propagation algorithm
    def feedforward_propagation(self, X, weights, biases, num_hidden_layers, activation_function):
      a = []
      h = []

      for k in range(num_hidden_layers):
        
        if k == 0:
          
          a.append(np.dot(weights[k], X) + biases[k])
          if(activation_function == "ReLU"):
            h.append(self.reLU(a[k]))
          elif(activation_function == "sigmoid"):
            h.append(self.sigmoid(a[k]))
          elif(activation_function == "tanh"):
            h.append(self.tanh(a[k]))
          elif(activation_function == "identity"):
            h.append(self.identity(a[k]))
        
        else:
          
          a.append(np.dot(weights[k], h[k-1]) + biases[k])
          if(activation_function == "ReLU"):
            h.append(self.reLU(a[k]))
          elif(activation_function == "sigmoid"):
            h.append(self.sigmoid(a[k]))
          elif(activation_function == "tanh"):
            h.append(self.tanh(a[k]))
          elif(activation_function == "identity"):
            h.append(self.identity(a[k]))

      
      a.append(np.dot(weights[num_hidden_layers], h[num_hidden_layers - 1]) + biases[num_hidden_layers])
      y_hat = self.softmax(a[-1])
      return a, h, y_hat

    
    def one_hot(self, Y):
      if Y.max() != 9:
        one_hot_Y = np.zeros((Y.size, 10))
      else:
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
      one_hot_Y[np.arange(Y.size), Y] = 1
      one_hot_Y = one_hot_Y.T
      return one_hot_Y

    
    def deriv_sigmoid(self, Z):
      func = self.sigmoid(Z)
      return func * (1 - func)

    
    def deriv_reLU(self, Z):
      return Z > 0
        

    def deriv_tanh(self, x):
      return 1 - np.tanh(x)**2

    
    def deriv_identity(self, x):
      return 1


    # Back propagation algorithm
    def back_propagation(self, Y, fwd_A, fwd_H, weights, biases, pred_output, num_hidden_layers, activation_function, loss):
      one_hot_Y = self.one_hot(Y)
      dA = {}
      dH = {}
      dW = {}
      dB = {}

      if loss == 'cross_entropy':
        dA[num_hidden_layers] = pred_output - one_hot_Y
      elif loss == 'mean_squared_error' or loss == 'mse':
        dA[num_hidden_layers] = self.diff_mse(pred_output, Y)

      for k in range(num_hidden_layers, 0, -1):
        dW[k] = np.dot(dA[k], fwd_H[k-1].T)
        dB[k] = np.mean(dA[k], axis=1, keepdims=True)

        dH[k-1] = np.dot(weights[k].T, dA[k])
        if(activation_function == "ReLU"):
          dA[k-1] = np.multiply(dH[k-1], self.deriv_reLU(fwd_A[k-1]))
        elif(activation_function == "sigmoid"):
          dA[k-1] = np.multiply(dH[k-1], self.deriv_sigmoid(fwd_A[k-1]))
        elif(activation_function == "tanh"):
          dA[k-1] = np.multiply(dH[k-1], self.deriv_tanh(fwd_A[k-1]))
        elif(activation_function == "identity"):
          dA[k-1] = np.multiply(dH[k-1], self.deriv_identity(fwd_A[k-1]))
      return dW, dB

    def get_predictions(self, pred_output):
      return np.argmax(pred_output, axis = 0)

    def get_accuracy(self, y_pred, y_true):
      return np.sum(y_pred == y_true) / y_true.size

    
    def loss_function(self, y_pred, y_true, loss, weights, weight_decay):
      
      # Cross Entropy
      if(loss == 'cross_entropy'):
        epsilon = 1e-30
        cross_entropy_loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=0))

        # L2 Regularisation
        reg_loss = 0.5 * weight_decay * sum(np.sum(w ** 2) for w in weights.values())
        total_loss = cross_entropy_loss + reg_loss

      # Mean Squared Error  
      elif loss == 'mean_squared_error' or loss == 'mse':
        mse_loss = np.mean(np.sum((y_pred - y_true) ** 2))

        # L2 Regularisation
        reg_loss = 0.5 * weight_decay * sum(np.sum(w ** 2) for w in weights.values())
        total_loss = mse_loss + reg_loss
      return total_loss

    
    
    
    # Gradient descent and Optimizers
    def gradient_descent(self, weights, biases, dW, dB, eta):

      # Update weights and biases
      for l in range(1, self.num_hidden_layers + 1):
        weights[l] -= eta * dW[l]
        biases[l] -= eta * dB[l]

      return weights, biases

    
    def momentum_based_gradient_descent(self, weights, biases, prev_weights, prev_biases, dW, dB, eta, momentum):

      for l in range(1, self.num_hidden_layers + 1):
        uw = momentum * prev_weights[l] + eta * dW[l]
        ub = momentum * prev_biases[l] + eta * dB[l]

        # Update current and prev weights and biases
        weights[l] -= uw
        biases[l] -= ub
        prev_weights[l] = uw
        prev_biases[l] = ub
      return weights, biases, prev_weights, prev_biases


    
    def rmsProp_gradient_descent(self, weights, biases, dW, dB, eta, eps, beta):
      
      v_w = {}
      v_b = {}

      for l in range(1, self.num_hidden_layers + 1):
        v_w[l] = 0
        v_b[l] = 0

      for l in range(1, self.num_hidden_layers + 1):
        v_w[l] = (beta * v_w[l]) + ((1-beta) * dW[l] ** 2)
        v_b[l] = (beta * v_b[l]) + ((1-beta) * dB[l] ** 2)

        # Update weights and biases
        weights[l] -= eta * dW[l] / (np.sqrt(v_w[l]) + eps)
        biases[l] -= eta * dB[l] / (np.sqrt(v_b[l]) + eps)

      return weights, biases

    
    def adam_gradient_descent(self, weights, biases, ts, v_w, v_b, m_w, m_b, dW, dB, eta, eps, beta1, beta2):

      for l in range(1, self.num_hidden_layers + 1):
        mdW = (beta1 * m_w[l]) + (1-beta1) * dW[l]
        mdB = (beta1 * m_b[l]) + (1-beta1) * dB[l]

        vdW = beta2 * v_w[l] + (1 - beta2) * (dW[l] ** 2)
        vdB = beta2 * v_b[l] + (1 - beta2) * (dB[l] ** 2)

        m_w_hat = mdW/(1-np.power(beta1, ts))
        v_w_hat = vdW/(1-np.power(beta2, ts))
        m_b_hat = mdB/(1-np.power(beta1, ts))
        v_b_hat = vdB/(1-np.power(beta2, ts))

        #update weights and biases
        weights[l] -= eta*m_w_hat/(np.sqrt(v_w_hat+eps))
        biases[l] -= eta*m_b_hat/(np.sqrt(v_b_hat+eps))

        v_w[l] = vdW
        v_b[l] = vdB
        m_w[l] = mdW
        m_b[l] = mdB
      
      ts += 1

      return weights, biases, v_w, v_b, m_w, m_b, ts
    
    # <---------------------------------------------START--------------------------------------------------->
    

    ''' Add new optimizers here '''





    # <---------------------------------------------END----------------------------------------------------->


    
    def compute_accuracy(self, X_test, y_test, weights, biases, num_hidden_layers, activation_function):

      _, _, pred_output = self.feedforward_propagation(X_test, weights, biases, num_hidden_layers, activation_function)
      pred_labels = np.argmax(pred_output, axis=0)
      accuracy = np.mean(pred_labels == y_test)
      return accuracy
    




def train_neural_network(nn, x_train_input, y_train, x_test_input, y_test, x_val, y_val, weights, biases, prev_weights, prev_biases, num_hidden_layers, activation_function, optimizer, epochs, batch_size, eta, momentum, beta, beta1, beta2, eps, weight_decay, loss):
  
  data_size = len(x_train_input[0])

  if optimizer == "sgd":
    batch_size = 1

  lookahead_w = weights
  lookahead_b = biases
  ts = 1
  v_w = prev_weights.copy()
  v_b = prev_biases.copy()
  m_w = prev_weights.copy()
  m_b = prev_biases.copy()

  for iter in tqdm(range(epochs)):
    total_train_loss = 0
    for i in range(0, data_size, batch_size):
      if i<= data_size - batch_size:
        X_batch = x_train_input[:, i:i+batch_size]
        Y_batch = y_train[i:i+batch_size]

        if optimizer == "sgd":
          fwd_a, fwd_h, pred_output = nn.feedforward_propagation(X_batch, weights, biases, num_hidden_layers, activation_function)

          one_hot_Y = nn.one_hot(Y_batch)
          train_loss = nn.loss_function(pred_output, one_hot_Y, loss, weights, weight_decay)
          total_train_loss += train_loss

          dW, dB = nn.back_propagation(Y_batch, fwd_a, fwd_h, weights, biases, pred_output, num_hidden_layers, activation_function, loss)
          weights, biases = nn.gradient_descent(weights, biases, dW, dB, eta)

        elif optimizer == "momentum":
          fwd_a, fwd_h, pred_output = nn.feedforward_propagation(X_batch, weights, biases, num_hidden_layers, activation_function)

          one_hot_Y = nn.one_hot(Y_batch)
          train_loss = nn.loss_function(pred_output, one_hot_Y, loss, weights, weight_decay)
          total_train_loss += train_loss

          dW, dB = nn.back_propagation(Y_batch, fwd_a, fwd_h, weights, biases, pred_output, num_hidden_layers, activation_function, loss)
          weights, biases, _, _ = nn.momentum_based_gradient_descent(weights, biases, prev_weights, prev_biases, dW, dB, eta, momentum)

        elif optimizer == "nesterov" or optimizer == "nag":
          
          beta = momentum

          # Partial updates
          for l in range(1, num_hidden_layers+1):
            lookahead_w[l] = weights[l] - beta * prev_weights[l]
            lookahead_b[l] = biases[l] - beta * prev_biases[l]

          fwd_a, fwd_h, pred_output = nn.feedforward_propagation(X_batch, lookahead_w, lookahead_b, num_hidden_layers, activation_function)

          one_hot_Y = nn.one_hot(Y_batch)
          train_loss = nn.loss_function(pred_output, one_hot_Y, loss, weights, weight_decay)
          total_train_loss += train_loss

          dW, dB = nn.back_propagation(Y_batch, fwd_a, fwd_h, lookahead_w, lookahead_b, pred_output, num_hidden_layers, activation_function, loss)
          weights, biases, prev_weights, prev_biases = nn.momentum_based_gradient_descent(weights, biases, prev_weights, prev_biases, dW, dB, eta, beta)

        elif optimizer == "rmsprop":
          fwd_a, fwd_h, pred_output = nn.feedforward_propagation(X_batch, weights, biases, num_hidden_layers, activation_function)

          one_hot_Y = nn.one_hot(Y_batch)
          train_loss = nn.loss_function(pred_output, one_hot_Y, loss, weights, weight_decay)
          total_train_loss += train_loss

          dW, dB = nn.back_propagation(Y_batch, fwd_a, fwd_h, weights, biases, pred_output, num_hidden_layers, activation_function, loss)
          weights, biases = nn.rmsProp_gradient_descent(weights, biases, dW, dB, eta, eps, beta)

        elif optimizer == "adam":
          fwd_a, fwd_h, pred_output = nn.feedforward_propagation(X_batch, weights, biases, num_hidden_layers, activation_function)

          one_hot_Y = nn.one_hot(Y_batch)
          train_loss = nn.loss_function(pred_output, one_hot_Y, loss, weights, weight_decay)
          total_train_loss += train_loss

          dW, dB = nn.back_propagation(Y_batch, fwd_a, fwd_h, weights, biases, pred_output, num_hidden_layers, activation_function, loss)
          weights, biases, v_w, v_b, m_w, m_b, ts = nn.adam_gradient_descent(weights, biases, ts, v_w, v_b, m_w, m_b, dW, dB, eta, eps, beta1, beta2)

        elif optimizer == "nadam":

          # Partial updates
          for l in range(1, num_hidden_layers+1):
            lookahead_w[l] = weights[l] - beta * prev_weights[l]
            lookahead_b[l] = biases[l] - beta * prev_biases[l]

          fwd_a, fwd_h, pred_output = nn.feedforward_propagation(X_batch, lookahead_w, lookahead_b, num_hidden_layers, activation_function)

          one_hot_Y = nn.one_hot(Y_batch)
          train_loss = nn.loss_function(pred_output, one_hot_Y, loss, weights, weight_decay)
          total_train_loss += train_loss

          dW, dB = nn.back_propagation(Y_batch, fwd_a, fwd_h, lookahead_w, lookahead_b, pred_output, num_hidden_layers, activation_function, loss)
          weights, biases, v_w, v_b, m_w, m_b, ts = nn.adam_gradient_descent(weights, biases, ts, v_w, v_b, m_w, m_b, dW, dB, eta, eps, beta1, beta2)

    # <---------------------------------------START---------------------------------------------->
    
    
    ''' Call new optimizer here '''



    
    
    
    # <---------------------------------------END------------------------------------------------>
    
    avg_train_loss = total_train_loss / (data_size / batch_size)

    _, _, val_pred = nn.feedforward_propagation(x_val, weights, biases, num_hidden_layers, activation_function)
    val_one_hot = nn.one_hot(y_val)
    val_loss = nn.loss_function(val_pred, val_one_hot, loss, weights, weight_decay)
    if loss == 'mean_squared_error' or loss == 'mse':
      val_loss = val_loss / (data_size / batch_size)

    _, _, test_pred = nn.feedforward_propagation(x_test_input, weights, biases, num_hidden_layers, activation_function)
    test_one_hot = nn.one_hot(y_test)
    test_loss = nn.loss_function(test_pred, test_one_hot, loss, weights, weight_decay)
    if loss == 'mean_squared_error' or loss == 'mse':
      test_loss = test_loss / (data_size / batch_size)
    
    val_accuracy = nn.compute_accuracy(x_val, y_val, weights, biases, num_hidden_layers, activation_function)
    train_accuracy = nn.compute_accuracy(x_train_input, y_train, weights, biases, num_hidden_layers, activation_function)
    test_accuracy = nn.compute_accuracy(x_test_input, y_test, weights, biases, num_hidden_layers, activation_function)

    print(f"val accuracy: {val_accuracy * 100:.2f}%, Test accuracy : {test_accuracy * 100:.2f}, Val loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
    wandb.log({'val_accuracy' : val_accuracy * 100, 'accuracy' : train_accuracy * 100, 'test_accuracy' : test_accuracy * 100,
               'loss' : avg_train_loss, 'val loss' : val_loss, 'test_loss' : test_loss, 'epoch' : iter}, step=iter)

  return weights, biases
