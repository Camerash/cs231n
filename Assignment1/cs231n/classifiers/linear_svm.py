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
  losses = np.sum(margins) # Sum all scores along the rows (Same data, different classes)
  loss = losses / num_train # Mean all scores along column (different data)
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
  
  # Consider gradient calculation's precondition is the margin larger than 1
  # We can reuse the margin calculated in the above loss calculation
  gradient_factors = margins # Ref

  # dLi/dWj = 1(xiwj - xiwyi + delta > 0)xi
  # Focusing on this part: 1(xiwj - xiwyi + delta > 0)
  # According to the eqt, THE FACTOR OF ELEMENTS WHERE margins > 0 is 1
  gradient_factors[margins > 0] = 1 # Setting elements be 1 where the margins is larger than 0

  # dLi/dWyi = - sum_(j!=yi)(1(xiwj - xiwyi + delta > 0)xi)
  # Since xi can be taken out of the summation:
  # Focusing on this part: - sum_(j!=yi)(1(xiwj - xiwyi + delta > 0)
  # According to the eqt, THE FACTOR OF ELEMENTS WHERE j = yi should be THE NEGATIVE OF THE SUM OF ALL FACTORS (calculated just above) WHERE j =/= yi
  # Therefore:
  row_count_sum = np.sum(gradient_factors, axis=1) # Squashing (summing) the margins count along the rows (Same data, different classes)
  # The gradient factors of THE CORRECT CLASS is therefore the NEGATIVE OF SUM OF INCORRECT CLASS SCORE
  # That is:
  gradient_factors[np.arange(num_train), y] = -row_count_sum.T

  # Finally matrix multiplication for the missing term xi in the above steps
  # X has size has size (N, D), Gradient_factor has size (N, C), we want a gradient matrix with size (D, C)
  # Therefore transpose X
  dW = np.matmul(X.T, gradient_factors)

  # why you so MEAN (/jk here we divide gradients by number of data to get mean)
  dW /= num_train

  # Regularazation
  # If you use half regularize strength in calculating loss, here you don't have to double it up
  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
