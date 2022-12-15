import numpy as np


class SoftMaxLayer(object):
    def __init__(self):
        """
        Constructor
        """
        self.y = None

    def forward(self, x):
        """
        Implement forward pass of softmax

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
        self.y : np.array
             The output of the layer (needed for backpropagation)
        """

        x2 = np.exp(x - np.max(x))
        out = x2 / np.sum(x2, axis=1)[:, np.newaxis]
        self.y = out
        return out

    def backward(self, y_grad):
        """
        Compute "backward" computation of softmax

        Parameters
        ----------
        y_grad : np.array
            The gradient at the output

        Returns
        -------
        np.array
            The gradient at the input

        """
        
        
        """
        z = self.forward(y_grad)
        z = np.reshape(z, (1, -1))
        diag = np.matmul(z, np.identity(z.size))
        jacobian = diag - np.matmul(z.T, z)
        jacobian = jacobian[0:y_grad.shape[1], 0:y_grad.shape[1]]
        x_grad = np.matmul(y_grad, jacobian)
        
        return x_grad
        """
        x_grad = np.zeros(np.shape(y_grad))
        for i in range(len(self.y)):
            diag = np.diag(self.y[i])
            reshaped_y = self.y[i].reshape(len(self.y[0]), 1)
            mul = np.matmul(reshaped_y, reshaped_y.T)
            jacobian = diag - mul
            x_grad[i] = np.matmul(jacobian, y_grad[i])
        x_grad = np.nan_to_num(x_grad)
        return x_grad



    def update_param(self, lr):
        pass  # no learning for softmax layer
