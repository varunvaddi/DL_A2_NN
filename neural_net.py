import numpy as np
 
 
class NeuralNetwork:
    """
    A multi-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices.
 
    The network uses a nonlinearity after each fully connected layer except for the
    last. You will implement two different non-linearities and try them out: Relu
    and sigmoid.
 
    The outputs of the second fully-connected layer are the scores for each class.
    """
 
    def __init__(self, input_size, hidden_sizes, output_size, num_layers, nonlinearity='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
 
        W1: First layer weights; has shape (D, H_1)
        b1: First layer biases; has shape (H_1,)
        .
        .
        Wk: k-th layer weights; has shape (H_{k-1}, C)
        bk: k-th layer biases; has shape (C,)
 
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: List [H1,..., Hk] with the number of neurons Hi in the hidden layer i.
        - output_size: The number of classes C.
        - num_layers: Number of fully connected layers in the neural network.
        - nonlinearity: Either relu or sigmoid
        """
        self.num_layers = num_layers
 
        assert(len(hidden_sizes)==(num_layers-1))
        sizes = [input_size] + hidden_sizes + [output_size]
 
        self.params = {}
        for i in range(1, num_layers + 1):
          # weights Initialization to small random values
          self.params['W' + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) * np.sqrt(2 / sizes[i - 1])
          # biases Initialization to zeros
          self.params['b' + str(i)] = np.zeros(sizes[i])

        # sigmoid non-linearlity
        if nonlinearity == 'sigmoid':
            self.nonlinear = sigmoid
            self.nonlinear_grad = sigmoid_grad
        # relu non-linearity
        elif nonlinearity == 'relu':
            self.nonlinear = relu
            self.nonlinear_grad = relu_grad
 
 
    def forward(self, X):
        """
        Compute the scores for each class for all of the data samples.
 
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
 
        Returns:
        - scores: Matrix of shape (N, C) where scores[i, c] is the score for class
            c on input X[i] outputted from the last layer of your network.
        - layer_output: Dictionary containing output of each layer BEFORE
            nonlinear activation. You will need these outputs for the backprop
            algorithm. You should set layer_output[i] to be the output of layer i.
 
        """
        scores = X
        layer_output = {0: X}
        #############################################################################
        # TODO: Write the forward pass, computing the class scores for the input.   #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C). Store the output of each layer BEFORE nonlinear activation  #
        # in the layer_output dictionary                                            #
        #############################################################################
        for i in range(1, self.num_layers + 1):
            # retreive weights & biases from Params dictionary
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]
            # wx+b
            scores = scores.dot(W) + b
            #storing output of each layer before activation
            layer_output[i] = scores

            # Apply nonlinearity except for the last layer
            if i < self.num_layers:  
                scores = self.nonlinear(scores)
 
        return scores, layer_output
 
 
    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
 
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.
 
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
 
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Compute the forward pass
        # Store the result in the scores variable, which should be an array of shape (N, C).
        scores, layer_output = self.forward(X)
 
        # If the targets are not given, exit early
        if y is None:
            return scores
 
        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss using the scores      #
        # output from the forward function. The loss include both the data loss and #
        # L2 regularization for weights W1,...,Wk. Store the result in the variable #
        # loss, which should be a scalar. Use the Softmax classifier loss.          #
        #############################################################################

        # Scores are adjusted by subtracting maximum score in each row to attain Numerical stability for softmax
        scores = scores - np.max(scores, axis=1, keepdims=True)  
        #Softmax function to convert scores into probabilities
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        #Negative log-likelihood for correct classes
        # X.shape[0] gives 'N' --> which is no. of samples
        neg_loglikelihood_probs = -np.log(probs[np.arange(X.shape[0]), y])
        #Average data loss
        data_loss = np.sum(neg_loglikelihood_probs) / X.shape[0]
 
        # Compute the L2 regularization loss
        L2_reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
            L2_reg_loss += 0.5 * reg * np.sum(self.params['W' + str(i)] ** 2)

        # Total loss is sum of Data loss and L2 regularization loss
        loss = data_loss + L2_reg_loss
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
 
        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        
        dscores = probs
        #Subtracting 1 from the probability of correct class
        dscores[np.arange(X.shape[0]), y] -= 1
        #Normalizing by no. of samples
        dscores /= X.shape[0]
 
        for i in reversed(range(1, self.num_layers + 1)):
            #Grad of weight is product of prev layer output & dscores
            grads['W' + str(i)] = layer_output[i - 1].T.dot(dscores) + reg * self.params['W' + str(i)]
            #Grad of bias is sum of dscores across batch
            grads['b' + str(i)] = np.sum(dscores, axis=0)
            
            #Skip last layer, to propogate the gradients backward
            if i > 1:
                dscores = dscores.dot(self.params['W' + str(i)].T)
                dscores = self.nonlinear_grad(layer_output[i - 1]) * dscores
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
 
        return loss, grads
 
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.
 
        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """

        #No of samples
        num_train = X.shape[0]
        # using // to return integer instead of float value.
        iterations_per_epoch = max(num_train // batch_size, 1)
 
        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################

        for it in range(num_iters):
            #creating a mini-batch from training data
            indices = np.random.choice(num_train, batch_size, replace=True)
            #data of mini-batch created by indexing X & y with randomly selected indices
            X_batch = X[indices]
            y_batch = y[indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
 
            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
 
            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            #each parameter in self.params is updated by subtracting (learning rate * resp gradient)
            for param in self.params:
                self.params[param] -= learning_rate * grads[param]

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
 
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
 
            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
 
                # Decay learning rate
                learning_rate *= learning_rate_decay
 
        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }
 
    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.
 
        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.
 
        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """

        #computes scores matrix for each class for all input samples
        #layers_output returned from forward() is ignored here, as its not needed for prediction
        scores, _ = self.forward(X)
        #calculates the index(predicted class) of maximum score in each row
        y_pred = np.argmax(scores, axis=1)
 
        ###########################################################################
        # TODO: Implement classification prediction. You can use the forward      #
        # function you implemented                                                #
        ###########################################################################
 
        return y_pred
 
 
def sigmoid(X):
    #############################################################################
    # TODO: Write the sigmoid function                                          #
    #############################################################################
    return 1 / (1 + np.exp(-X))
 
def sigmoid_grad(X):
    #############################################################################
    # TODO: Write the sigmoid gradient function                                 #
    #############################################################################
    return sigmoid(X) * (1 - sigmoid(X))
 
def relu(X):
    #############################################################################
    #  TODO: Write the relu function                                            #
    #############################################################################
    return np.maximum(0, X)
 
def relu_grad(X):
    #############################################################################
    # TODO: Write the relu gradient function                                    #
    #############################################################################
    return (X > 0).astype(float)