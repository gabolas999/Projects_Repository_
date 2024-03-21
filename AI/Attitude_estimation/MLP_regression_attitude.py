# Importing and setting up


import numpy as np
import torch
from matplotlib import pyplot as plt
import load_data
from math import pi
from scipy import signal 

# Set the PyTorch and numpy random seeds for reproducibility:
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)



# Parameters to tune the MLP
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
#Filtering
polynomialdegree = 1
smooth_window = 230 #160

#Model
n_hidden_neurons = 40
learning_rate = 4e-2 #5e-2
n_epochs = 3400  #2900      3600


# If sensor_data == True, real sensor data is used, otherwise a sine wave is used:
sensor_data = True
#-----------------------------------------------------------------------
#-----------------------------------------------------------------------

# Creating the training function for MLP
def train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y):
    print("Started training neural network")
    # Define the model:
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
        torch.nn.Sigmoid(), 
        torch.nn.Linear(n_hidden_neurons, 1)
    )
    # MSE loss function (mean squared error):
    loss_fn = torch.nn.MSELoss()

    # Various optimizer options
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # →  SGD optimiser = stochastic gradient descent 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the network:
    for t in range(n_epochs):
        # Forward pass
        y_pred = model(X)
        # Compute and print loss. We pass Tensors containing the predicted and
        # true values of y, and the loss function returns a Tensor containing
        # the loss.
        loss = loss_fn(y_pred, Y)
        if t % 300 == 0:
            print("Epoch: ", t, " Current loss: ", loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Return the trained model
    print("Finished training")
    return model



# Loading and filtering the data
if (sensor_data):
    # Load the data:
    [accel_x, accel_y, accel_z, gyro_p, gyro_q, gyro_r, att_phi, att_theta, att_psi,
        cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw] = load_data.load_sensor_data()

    # Data:
    # Whether to include the gyros:
    include_gyros = False
    if (include_gyros):
        X = np.hstack([accel_x, accel_y, accel_z, gyro_p, gyro_q, gyro_r])
    else:
        X = np.hstack([accel_x, accel_y, accel_z])
    n_features = X.shape[1]
    Y = att_phi
    print("Data loaded successfully")
else:
    # Load the sine data:
    [X, Y] = load_data.load_sine_data()
    n_features = X.shape[1]

# Total number of samples:
N = X.shape[0]

# Convert to torch tensors:     (tensor is the data structure pytorch operates in, so need to translate it into it.)
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()


# Filtering 
X_st = X / 1024
Y_st = Y * 180/pi

X_st_f = signal.savgol_filter(X_st, smooth_window , polynomialdegree, axis=0)
X_st_f = torch.from_numpy(X_st_f).float()



# Training the model
if sensor_data:
    # Flight Data
    model = train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X_st_f, Y_st)
    y_pred = model(X_st_f)
    y_pred = y_pred.detach().numpy()
    wind, poly = 40, 1

    # Smooth the final signal slightly
    y_pred = signal.savgol_filter(y_pred, wind, poly, axis=0)

    y_plot = Y_st.detach().numpy()
    y_plot = y_plot.reshape(N, 1)
else:
    # Sine function
    model = train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y)
    y_pred = model(X)
    y_pred = y_pred.detach().numpy()
    y_plot = Y.detach().numpy()
    y_plot = y_plot.reshape(N, 1)


# Plot the ground truth vs Network output.
truncate = 6405 # 6405
plt.figure()
plt.plot(y_plot[:truncate], 'ko', label='Ground Truth', markersize=1.5)
plt.plot(y_pred[:truncate], label='Network Output', linewidth=3.5)  # 1.5 when used without signal_filter
plt.legend()
plt.savefig('data/output_vs_ground_truth.png')
plt.show()


# Check the filtering and smoothing of the function
check_filter = False
if check_filter:
    plt.plot(X_st[:400,2], label ="X_st")
    plt.plot(X_st_f[:400,2], label ="X_st_f")
    plt.plot(Y[:400,], label ="Y")
    plt.plot(Y_st[:400,], label ="Y_st")
    plt.legend()
    plt.show()
