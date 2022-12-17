from tinygrad.tensor import Tensor, Function
from tqdm import tqdm
import numpy as np
import math
from functions.io_handling import *

args = argparser()
initial_lr = args.lr

# The class used to create the four model objects
class Polynomial_Model(Function):
    def __init__(self, coeffs):
        # Initialize the model's coefficients
        self.coeffs = coeffs

    def parameters(self):
        # Get the coefficients of the polynomial model
        return self.coeffs

    def forward(self, x):
        # Compute the output of the polynomial model
        y = 0
        for i, coeff in enumerate(self.coeffs):
            y += coeff * x ** i
        return y

# This is for batching. shuffling, and splitting of a dataset
class Dataset:
    def __init__(self, x, y, batch_size=1):
        self.x = x
        self.y = y
        self.data = np.zeros([len(self.x),]).tolist()
        for i in range(len(self.x)):
            self.data[i] = [self.x[i], self.y[i]]
        self.batch_size = batch_size
        self.num_samples = len(self.data)
        self.num_batches = self.num_samples // self.batch_size
        self.current_batch = 0
        self.shuffle()
        
    def shuffle(self):
        np.random.shuffle(self.data)

    def next_batch(self):
        if self.current_batch >= self.num_batches:
            self.current_batch = 0
            self.shuffle()

        start = self.current_batch * self.batch_size
        end = start + self.batch_size

        batch = self.data[start:end]
        self.current_batch += 1

        for i in range(len(batch)):
            batch[i] = [Tensor([[batch[i][0]]],requires_grad=True), Tensor([[batch[i][1]]], requires_grad=True)]

        return batch

    def split(self, train_fraction):
        train_split = int(round(self.num_samples * train_fraction))
        
        x = []
        y = []
        for data in self.data:
            x.append(data[0])
            y.append(data[1])

        x_train_data = x[:train_split]
        y_train_data = y[:train_split]
        x_validation_data = x[train_split:]
        y_validation_data = y[train_split:]

        return x_train_data, y_train_data, x_validation_data, y_validation_data


# Generate a list of n tensors with value 0 
def coeffs_gen(n):
    coeffs = np.zeros([n,1]).tolist()
    for i in range(n):
        coeffs[i] = Tensor.glorot_uniform(1,1)
    return coeffs

# Compute the loss of the model with respect to the ground truth (Loss Function)
def loss_func(y_pred, y_gt):
    squared_diff = (y_pred-y_gt)*(y_pred-y_gt)
    return squared_diff.mean()

# Modified loss function for testing, where the points are just normalize with respect to the y values of the training dataset
def loss_mod(y_pred, y_gt):
    squared_diff = ((y_pred-y_gt)/1000)*((y_pred-y_gt)/1000) # Since testing is in terms of 10^7 while training is in terms of 10^4
    return squared_diff.mean()

# Compute for the correlation of the ground truth and the predicted values (Accuracy Function)
def r2_score(x_gt, y_gt, model):
    y_pred = np.array(model.forward(x_gt).data)
    y_gt_mean = sum(y_gt)/len(y_gt)
    sum_of_squared_errors = np.sum((y_pred - y_gt) ** 2)
    sum_of_squared_mean_errors = np.sum((y_gt - y_gt_mean) ** 2)
    r2 = 1 - (sum_of_squared_errors / sum_of_squared_mean_errors)
    return r2

# Learning Rate Scheduler
def schedule(epoch):
    global initial_lr
    drop = 0.001
    epochs_drop = 250.0
    lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

# Perform the training of the model given the optimizer, dataset, and number of iterations
def train(model, optimizer, dataset, max_epochs, model_number=0):
    losses = []
    for epoch in tqdm(range(max_epochs)):
        epoch_loss = []
        tl = dataset.next_batch()
        for data in tl:
            y_pred = model.forward(data[0])
            loss = loss_func(y_pred, data[1])
            epoch_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        optimizer.lr = schedule(epoch)
        loss = (sum(epoch_loss)/len(epoch_loss)).sqrt()
        losses.append(loss.data[0])
        '''
        if (epoch % 50 == 0) | (epoch % max_epochs == 0):
            print(f" Epoch {epoch}, Loss: {loss.data[0]}")
        '''
    print(f"Model {model_number} [Training] RMSE Loss: {loss.data[0]}")
    return model, loss.data[0]

# Perform testing and validation of the model given a dataset
def test(model, dataset, mode='Testing', model_number=0):
    test_loss = []
    for data in dataset.data:
        y_pred = model.forward(data[0])
        if mode == 'Testing':
            loss = loss_mod(y_pred, data[1])
        else:
            loss = loss_func(y_pred, data[1])
        test_loss.append(loss)
    overall_loss = (sum(test_loss)/len(test_loss)).sqrt()
    r2score = r2_score(np.array(dataset.x), np.array(dataset.y), model)
    if mode == 'Validation':
        print(f"Model {model_number} [{mode}] RMSE Loss: {overall_loss.data[0]}, R2 Score: {r2score*100}%")
        return overall_loss.data[0]
    else:
        print(f"[{mode}] RMSE Loss: {overall_loss.data[0]},  R2 Score: {r2score*100}%")
