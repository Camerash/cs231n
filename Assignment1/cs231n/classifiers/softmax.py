import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    prob = np.matmul(X[i], W) # Scores in the scope of SVM, here we describe this as unnormalized probabilities
    prob += -np.max(prob) # Stablize data, check out: https://deepnotes.io/softmax-crossentropy
    exp_prob = np.exp(prob) # Get e^(prob) of the respective data
    loss += -prob[y[i]] + np.log(np.sum(exp_prob))
    # Following equation: check out: https://deepnotes.io/softmax-crossentropy
    for j in range(num_class):
      dW[:, j] += (np.exp(prob[j]) * X[i] / np.sum(exp_prob)) # -pj*pi
      if j == y[i]: # i == j
        dW[:, j] -= X[i] # pi - pj*pi
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  prob = np.matmul(X, W) # Scores in the scope of SVM, here we describe this as unnormalized probabilities
  prob += -np.max(prob) # Stablize data, check out: https://deepnotes.io/softmax-crossentropy
  sum_of_prob_row = np.sum(np.exp(prob), axis=1) # Get sum of e^(prob) of the respective data
  loss = -np.sum(prob[np.arange(num_train), y]) + np.sum(np.log(sum_of_prob_row), axis=0)
  dW = (np.exp(prob) / sum_of_prob_row[:,np.newaxis]) # Convert sum of prob row to a column vector, divide  matrix of e^(prob) by that
 
  pi_factor = np.zeros_like(prob)
  pi_factor[np.arange(num_train), y] = 1
  dW -= pi_factor # Subtract the necessary pi factor where position i = j

  dW = np.matmul(X.T, dW) # Multiply the dW score matrix by input X
  
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

