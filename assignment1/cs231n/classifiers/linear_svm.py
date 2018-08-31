import numpy as np
from random import shuffle

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
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note margin delta = 1
      if margin > 0:
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

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
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = np.dot(X, W)
  margins = np.maximum(0, scores + 1 - np.choose(y, scores.T)[:, np.newaxis])
  loss = np.mean(np.sum(margins, axis=1)) - 1
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

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
  margins[margins > 0] = 1
  num_margins_gtzero = np.sum(margins, axis=1)
  # Need to know how many -sy (out of si - sy + 1 > 0) survived. Only those
  # will contribute to the gradient on that column for that example.
  margins[np.arange(num_train), y] -= num_margins_gtzero
  # Note that this sums across the num_train dimension, so we will have to divide
  dW = np.dot(X.T, margins)
  dW /= num_train
  dW += 2 * reg * W

  # For comparison and clarity, this is what we will have to do if we did not
  # do the matrix multiplication with X.T and margins, and decided to use
  # np.multiply instead
  # dw = np.tile(X.reshape(num_train, -1, 1), num_classes)
  # margins[margins > 0] = 1
  # nonzero_dw = np.multiply(margins.reshape(num_train, 1, num_classes), dw)
  # num_margins_gtzero = np.sum(margins, axis=1).reshape(num_train, 1, 1)
  # mask = np.ones_like(dw, bool)
  # mask[np.arange(num_train), :, y] = False
  # dw[mask] = 0
  # dW = nonzero_dw - np.multiply(num_margins_gtzero, dw)
  # dW = np.mean(dW, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
