import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    correct_label = y[i] # Get the class label of the corresponding data
    scores = np.matmul(X[i], W) # Score for the respective data
    correct_class_score = scores[correct_label] # Score for the correct class
    for j in xrange(num_classes): # Looping through all the incorrect classes
      if j == correct_label:
        # Skipping the correct class
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      # If the current class has a better score then the correct one, normalize
      if margin > 0:
        # Add margin to total loss
        loss += margin

        # Add [Xi1, Xi2, ..., XiD], that is the data array of the i-th data
        # into
        # the array of weight of dimension (gradient) of the respective class
        dW[:,j] += X[i,:]

        # Substract [Xi1, Xi2, ..., XiD], that is the data array of the i-th data
        # from
        # the array of weight of dimension (gradient) of the trained class
        dW[:,correct_label] -= X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train # Same with the gradient array

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W  # Same with the gradient array, but the regularization strength is doubled

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0] # Ref num of training data
  score_matrix = np.matmul(X, W) # Data multiplied by weight, i.e. scores for all data for all classes
  correct_class_scores = score_matrix[np.arange(num_train),y] # Get all the correct classes score using np index trick
  margins = np.maximum(0, score_matrix - np.matrix(correct_class_scores).T + 1) # Calculate margins
  margins[np.arange(num_train), y] = 0 # Set scores for the correct classes be 0
  losses = np.sum(margins, axis=1) # Sum all scores along the rows (Same data, different classes)
  loss = np.mean(losses, axis=0) # Mean all scores along column (different data)
  loss += reg * np.sum(W * W) # Add regularization

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
