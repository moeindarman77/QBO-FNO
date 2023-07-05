import netCDF4 as nc
from utils import *

u_250 = nc.Dataset('u_f_flux_true_250m_100yrs.nc')
u_500 = nc.Dataset('u_f_flux_true_500m_100yrs.nc')

# ------------------- Prepare Data ------------------- #
# Set U_250 as the data
data = u_250 # Change the mode to 36 when you set the data to u_500
# data = u_500 # Change the mode to 18 when you set the data to u_500
# Remove the last row and column from u
u = data['u'][:-1, 1:-1]
# Remove the first row and column from f
f = data['f'][1:, 1:-1]

# ------------------- Plot the data ------------------- #
def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

# ------------------- Create Torch Dataset ------------------- #
import torch

# Set the random seed for reproducibility
torch.manual_seed(0)

# Set the device to be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create a torch dataset
class FNOData(torch.utils.data.Dataset):
    def __init__(self, u, f):
        super(FNOData, self).__init__()
        self.u = u
        self.f = f
        self.len = u.shape[0]

    def __getitem__(self, index):
        return self.u[index], self.f[index]

    def __len__(self):
        return self.len

dataset = FNOData(u, f)

# Split the dataset into train and validation sets (not randomly)
# Calculate sizes
validation_ratio = 0.1  # 10% validation
train_ratio = 0.8  # 80% training

train_size = int(train_ratio * len(dataset))
validation_size = int(validation_ratio * len(dataset))
test_size = len(dataset) - train_size - validation_size

# Create subsets
train_dataset = FNOData(u[:train_size], f[:train_size])
validation_dataset = FNOData(u[train_size:train_size + validation_size], f[train_size:train_size + validation_size])
test_dataset = FNOData(u[train_size + validation_size:], f[train_size + validation_size:])

# ----------------------------- Normalize the data --------------------------------- #
Normalize_input = True
Normalize_output = True
# Calculate mean and standard deviation of the training set
u_train_mean = train_dataset.u.mean()
u_train_std = train_dataset.u.std()
f_train_mean = train_dataset.f.mean()
f_train_std = train_dataset.f.std()

# Data maximum and minimum
u_max = train_dataset.u.max()
u_min = train_dataset.u.min()
f_max = train_dataset.f.max()
f_min = train_dataset.f.min()

print('u_train mean: {:10.3e}, std: {:10.3e}'.format(u_train_mean, u_train_std))
print('f_train mean: {:10.3e}, std: {:10.3e}'.format(f_train_mean, f_train_std))

print('u max: {:10.3e}, min: {:10.3e}'.format(u_max, u_min))
print('f max: {:10.3e}, min: {:10.3e}'.format(f_max, f_min))


if Normalize_input:
    train_dataset.u = (train_dataset.u - u_train_mean) / u_train_std
    validation_dataset.u = (validation_dataset.u - u_train_mean) / u_train_std
    test_dataset.u = (test_dataset.u - u_train_mean) / u_train_std

if Normalize_output:
    train_dataset.f = (train_dataset.f - f_train_mean) / f_train_std
    # # We don't normalize the validation set
    # validation_dataset.f = (validation_dataset.f - f_train_mean) / f_train_std 
    # test_dataset.f = (test_dataset.f - f_train_mean) / f_train_std

# ----------------------------- Create a dataloader -----------------------------
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=50, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=50, shuffle=False)

# -------------------------------------------- Create the Model -------------------------------------------- #
# Import Libraries
import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
#from prettytable import PrettyTable
from count_parameters import count_parameters
import hdf5storage
import pickle

# Note to remember from training
input_size = 512
output_size = 512
hidden_layer_size = 1000


# -------------------------------------------- Integrator -------------------------------------------- #
def RK4step(net,input_batch):
 output_1 = net(input_batch.cuda())
 output_2 = net(input_batch.cuda()+0.5*output_1)
 output_3 = net(input_batch.cuda()+0.5*output_2)
 output_4 = net(input_batch.cuda()+output_3)

 return input_batch.cuda() + time_step*(output_1+2*output_2+2*output_3+output_4)/6


def Eulerstep(net,input_batch):
 output_1 = net(input_batch.cuda())
 return input_batch.cuda() + time_step*(output_1) 
  

def directstep(net,input_batch):
  output_1 = net(input_batch.cuda())
  return output_1

# -------------------------------------------- FNO architecture -------------------------------------------- #

################################################################
#  1d Fourier Integral Operator
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(SpectralConv1d, self).__init__()
        """
        Initializes the 1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Args:
            in_channels (int): input channels to the FNO layer
            out_channels (int): output channels of the FNO layer
            modes (int): number of Fourier modes to multiply, at most floor(N/2) + 1
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        """
        Complex multiplication of the Fourier modes.
        [batch, in_channels, x], [in_channel, out_channels, x] -> [batch, out_channels, x]
            Args:
                input (torch.Tensor): input tensor of size [batch, in_channels, x]
                weights (torch.Tensor): weight tensor of size [in_channels, out_channels, x]
            Returns:
                torch.Tensor: output tensor with shape [batch, out_channels, x]
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fourier transformation, multiplication of relevant Fourier modes, backtransformation
        Args:
            x (torch.Tensor): input to forward pass os shape [batch, in_channels, x]
        Returns:
            torch.Tensor: output of size [batch, out_channels, x]
        """
        batchsize = x.shape[0]
        # Fourier transformation
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width, time_future, time_history):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: a driving function observed at T timesteps + 1 locations (u(1, x), ..., u(T, x),  x).
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        self.modes = modes
        self.width = width
        self.time_future = time_future
        self.time_history = time_history
        self.fc0 = nn.Linear(self.time_history+1, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.time_future)

    def forward(self, u):
        grid = self.get_grid(u.shape, u.device)
        x = torch.cat((u, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

# -------------------------------------------- Model Parameters --------------------------------------------
time_history = 1 #time steps to be considered as input to the solver
time_future = 1 #time steps to be considered as output of the solver
device = 'cuda'  #change to cpu of no cuda available

#Model parameters
modes = 36 # number of Fourier modes to multiply
width = 64 # input and output channels to the FNO layer

num_epochs = 50 #set to one so faster computation, in principle 20 is best
# learning_rate = 0.0001
learning_rate = 1e-4
lr_decay = 0.4
num_workers = 0

mynet = FNO1d(modes, width, time_future, time_history).cuda()
count_parameters(mynet)
mynet.cuda()

loss = nn.MSELoss()
#use two optimizers.  learing rates seem to work.

optimizer = optim.AdamW(mynet.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0, 5, 10, 15], gamma=lr_decay)

batch_size=100

best_loss = 1e6
train_loss_list = []
test_loss_list = []

################################################################################################################################
# -------------------------------------------- Training Loop --------------------------------------------
################################################################################################################################
for epoch in range(num_epochs):
    
    train_loss_total = 0
    train_r2_total = 0
    # Training Data Loop
    for i, data_train in enumerate(train_dataloader):
        inputs_train, labels_train = data_train[0].unsqueeze(2), data_train[1].unsqueeze(2)
        
        # Move tensors to the configured device
        inputs_train = inputs_train.to(device=device, dtype = torch.float32)
        labels_train = labels_train.to(device=device, dtype = torch.float32)

        # Forward pass train
        # model_output = directstep(mynet,inputs_train)
        model_output = mynet(inputs_train)
        train_loss = loss(model_output,labels_train)
        train_r2 = r2_score(labels_train,model_output)
        
        # Sum the loss
        train_loss_total += train_loss.item()
        train_r2_total += train_r2.item()

        # Backward and optimize
        optimizer.zero_grad()
        # train_loss.backward(retrain_graph=True)
        train_loss.backward()
        optimizer.step()
    # Average the loss
    train_loss_avg = train_loss_total/len(train_dataloader)
    train_r2_avg = train_r2_total/len(train_dataloader)
    del inputs_train, labels_train, model_output
    
    # Test Data Loop
    test_loss_total = 0
    test_r2_total = 0
    with torch.no_grad():
        for i, data_test in enumerate(test_dataloader):
            inputs_test, labels_test = data_test[0].unsqueeze(2), data_test[1].unsqueeze(2)
        
            # Move tensors to the configured device
            inputs_test = inputs_test.to(device=device, dtype = torch.float32)
            labels_test = labels_test.to(device=device, dtype = torch.float32)

            # Forward pass test
            model_output_test = directstep(mynet, inputs_test)
            if Normalize_output:
                model_output_test = model_output_test * f_train_std + f_train_mean
            test_loss = loss(model_output_test, labels_test)
            test_r2 = r2_score(labels_test,model_output_test)

            
            # Sum the loss
            test_loss_total += test_loss.item()
            test_r2_total += test_r2.item()
        # Average the loss
        test_loss_avg = test_loss_total/len(test_dataloader)
        test_r2_avg = test_r2_total/len(test_dataloader)
            
        # Check if the model is improving
        if test_loss_avg < best_loss:
            best_loss = test_loss_avg
            torch.save(mynet.state_dict(), 'best_model.pt')
        del inputs_test, labels_test, model_output_test

    # Print the progress in files
    # if (epoch+1) % 10 == 0:
    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:10.3e}, Test Loss: {:10.3e}, Train R2: {:10.3e}, Test R2: {:10.3e}'
            .format(epoch+1, num_epochs, i+1, len(train_dataloader), train_loss_avg, test_loss_avg, train_r2_avg, test_r2_avg))
    
    # Append the loss in to the list
    train_loss_list.append(train_loss_avg)
    test_loss_list.append(test_loss_avg)

# -------------------------------------------- Save the model and plot the loss history --------------------------------------------  
# Save the model checkpoint
torch.save(mynet.state_dict(), 'model_after_training.pt')

# Plot the loss history
plot_loss(train_loss_list, test_loss_list, save_fig=True, fig_name='loss_history_plot.pdf')


# -------------------------------------------- Inferenece on the validation data --------------------------------------------

val_loss_total = 0
val_r2_total = 0
pred_val = []
true_val = []
with torch.no_grad():
    for i, data_val in enumerate(validation_dataloader):
        inputs_val, labels_val = data_val[0].unsqueeze(2), data_val[1].unsqueeze(2)
        
        # Move tensors to the configured device
        inputs_val = inputs_val.to(device=device, dtype = torch.float32)
        labels_val = labels_val.to(device=device, dtype = torch.float32)

        # Forward pass test
        model_output_val = directstep(mynet, inputs_val)
        if Normalize_output:
            # Denormalize the output
            model_output_val = model_output_val * f_train_std + f_train_mean
        val_loss = loss(model_output_val, labels_val)
        val_r2 = r2_score(labels_val,model_output_val)


        # Append the prediction and the true labels
        pred_val.append(model_output_val.cpu().numpy())
        true_val.append(labels_val.cpu().numpy())

        # Sum the loss
        val_loss_total += val_loss.item()
        val_r2_total += val_r2.item()

    # Average the loss
    val_loss_avg = val_loss_total/len(validation_dataloader)
    val_r2_avg = val_r2_total/len(validation_dataloader)
    print('Validation Loss: {:10.3e}'.format(val_loss_avg))
    del inputs_val, labels_val, model_output_val


# -------------------------------------------- Saving OUTPUTS --------------------------------------------

## Save the predictions and the true labels in .nc format
from netCDF4 import Dataset

# Save the predictions and the true labels
pred_val = np.concatenate(pred_val, axis=0)
true_val = np.concatenate(true_val, axis=0)

print(pred_val.shape)
print(true_val.shape)

# Save in .nc format
rootgrp = Dataset("pred_true_val.nc", "w", format="NETCDF4")

# Create dimensions
sample_dim = rootgrp.createDimension("sample", None)  # for unlimited dimension
other_dim = rootgrp.createDimension("other", pred_val.shape[1])

# Create variables
pred_var = rootgrp.createVariable("pred_val","f4",("sample","other"))
true_var = rootgrp.createVariable("true_val","f4",("sample","other"))

# Assign data to variables
pred_var[:] = pred_val
true_var[:] = true_val

rootgrp.close()

