from functions.plotter import *
from functions.io_handling import *
from functions.train_test import *
from tinygrad.nn.optim import SGD
import pandas as pd

# Insert all inputs from the user  (Use the default values if the not specified)
args = argparser()
train_file = args.train_dataset
test_file = args.test_dataset
batch_size = args.batch_size
lr = args.lr
max_epochs = args.max_epoch

# Insert the train and test dataset using pandas library
train_data = pd.read_csv(train_file)
train_data = pd.DataFrame(train_data)
x_train_presplit = np.array(train_data['x'])
y_train_presplit = np.array(train_data['y'])
test_data = pd.read_csv(test_file)
test_data = pd.DataFrame(test_data)
x_test = np.array(test_data['x'])
y_test = np.array(test_data['y'])

# Organize all dataset that will be use into a Dataset object (from train_test.py Dataset Class)
split_train = Dataset(x_train_presplit, y_train_presplit)
x_train, y_train, x_valid, y_valid = split_train.split(train_fraction=0.8) # Split the training dataset to training and validation set
trainloader = Dataset(x_train, y_train, batch_size)
validloader = Dataset(x_valid,y_valid)
testloader = Dataset(x_test, y_test)

# Initialize a container for all models and their respective losses
models = []
losses = []

# Train and validate each model from Model 1 to Model 4
for i in range(2,6):
    coeffs = coeffs_gen(i)
    model = Polynomial_Model(coeffs)
    optimizer = SGD(model.parameters(), lr)
    print(f"Training Model {i-1}")
    model, loss = train(model, optimizer, trainloader, max_epochs, i-1)
    models.append(model)
    loss = test(model, validloader, mode="Validation", model_number=i-1)
    losses.append(loss)

# Pick the best model with the lowest loss
min_index = losses.index(min(losses))
final_model = models[min_index]
print(f"Choose Model {min_index+1}")

# Test the chosen model with the testing dataset
test(final_model, testloader, mode="Testing")

# Print the Degree and Coefficient of the Chosen Model
print("Degree:", min_index+1)
print("Coefficients:")
for x in final_model.coeffs:
    print(np.array(x.data)[0][0])

# Plot the predicted values against the ground truth
y_pred = final_model.forward(x_test)
plotter([[x_test, y_test],[x_test, np.array(y_pred.data)]], label=['GT', 'Pred'])