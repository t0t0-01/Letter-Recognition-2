from __future__ import print_function
import numpy as np
import math


class Sequential(object):
    def __init__(self, layers, loss):
        """
        Sequential model

        Implements a sequence of layers

        Parameters
        ----------
        layers : list of layer objects
        loss : loss object
        """
        self.layers = layers
        self.loss = loss

    def forward(self, x, target=None):
        """
        Forward pass through all layers
        
        if target is not none, then also do loss layer

        Parameters
        ----------
        x : np.array
            The input data of size number of training samples x number of features
        target : np.array
            The target data of size number of training samples x number of features (one-hot)

        Returns
        -------
        np.array
            The output of the model
        """
        
        data = np.copy(x)
        for l in self.layers:
            out = l.forward(data)
            data = out
            
        if target is not None:
            out = self.loss.forward(data, target)
        
        return out
        
        
    def backward(self):
        """
        Compute "backward" computation of fully connected layer

        Returns
        -------
        np.array
            The gradient at the input

        """
        prev_grad = self.loss.backward()
        
        for l in self.layers[::-1]:
            grad = l.backward(prev_grad)
            prev_grad = grad

        return grad

    def update_param(self, lr):
        """
        Update the parameters with learning rate lr

        Parameters
        ----------
        lr : floating point
            The learning rate
        """
        for l in self.layers:
            l.update_param(lr)
        
        
    def fit(self, x, y, epochs=10, lr=0.1, batch_size=128):
        """
        Fit parameters of all layers using batches

        Parameters
        ----------
        x : numpy matrix
            Training data (number of samples x number of features)
        y : numpy matrix
            Training labels (number of samples x number of features) (one-hot)
        epochs: integer
            Number of epochs to run (1 epoch = 1 pass through entire data)
        lr: float
            Learning rate
        batch_size: integer
            Number of data samples per batch of gradient descent
        """
        
        nb_of_batches = math.ceil(len(x) / batch_size)

        loss_per_epoch = []
        
        for epoch in range(0, epochs+1):
            start = 0
            loss_per_batch = []
            for i in range(1, nb_of_batches + 1):
                
                cutoff = i * batch_size if i != nb_of_batches else len(x)
                batch = x[start:cutoff, :]
                batch_targets = y[start:cutoff, :]
                start = cutoff
                loss = self.forward(batch, batch_targets)
                loss_per_batch.append(loss)
                
                if epoch != 0:                  #to add the loss at epoch 0, before update
                    self.backward()
                    self.update_param(lr)
                
            loss_per_epoch.append(np.mean(loss_per_batch))
        return loss_per_epoch
        
                
                
                
            
        

    def predict(self, x):
        """
        Return class prediction with input x

        Parameters
        ----------
        x : numpy matrix
            Testing data data (number of samples x number of features)

        Returns
        -------
        np.array
            The output of the model (integer class predictions)
        """
        y_hat = self.forward(x)
        m = np.argmax(y_hat, axis=1)
        return m
        
        