import numpy as np


class FullLayer(object):
    def __init__(self, n_i, n_o):
        """
        Fully connected layer

        Parameters
        ----------
        n_i : integer
            The number of inputs
        n_o : integer
            The number of outputs
        """
        self.x = None
        self.W_grad = None
        self.b_grad = None

        # need to initialize self.W and self.b
        self.W = np.random.normal(loc=0, scale=np.sqrt(2/(n_i+n_o)), size=(n_o, n_i))
        self.b = np.zeros(shape=(1, n_o))

    def forward(self, x):
        """
        Compute "forward" computation of fully connected layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features

        Returns
        -------
        np.array
            The output of the layer

        Stores
        -------
        self.x : np.array
             The input data (need to store for backwards pass)
        """
        self.x = x
        
        return np.dot(x, self.W.T) + self.b


    def backward(self, y_grad):
        """
        Compute "backward" computation of fully connected layer

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        Stores
        -------
        self.b_grad : np.array
             The gradient with respect to b (same dimensions as self.b)
        self.W_grad : np.array
             The gradient with respect to W (same dimensions as self.W)
        """

        self.b_grad = np.sum(y_grad, axis=0)[np.newaxis,:]
        self.W_grad = np.matmul(y_grad.T, self.x)
        return np.matmul(y_grad, self.W)


    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate

        Stores
        -------
        self.W : np.array
             The updated value for self.W
        self.b : np.array
             The updated value for self.b
        """
        print(self.W[:10, :])
        self.W = self.W - lr * self.W_grad
        print(self.W[:10, :])

        self.b = self.b -  lr * self.b_grad