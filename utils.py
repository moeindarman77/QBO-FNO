from prettytable import PrettyTable
from torch.utils.data import Dataset
import h5py
import matplotlib.pyplot as plot
import numpy as np
import torch


# ---------------------- Correlation Coefficient -------------------------------
import torch

def corr2(x, y, batch_mode=False):
    """
    Calculate the mean correlation coefficient between single or batches of input data.
    
    Parameters
    ----------
    x, y : torch.Tensor
        Input tensors with the same number of matrices and the same batch size if batch_mode is True 
        (dimensions should be (N, H, W) or (N, 1, H, W) for single and batch input).
    batch_mode : bool, optional
        Set True to calculate the correlation coefficient for batches of input data, by default False.
        
    Returns
    -------
    mean_corr_coeffs : float or torch.Tensor
        Mean correlation coefficient between the matrices in x and y for single input (float) 
        or mean correlation coefficients for each batch if batch_mode is True (torch.Tensor with shape: (B,)).
    """
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise ValueError("Both input tensors must be torch.Tensor objects.")
    
    if x.shape != y.shape:
        raise ValueError("Both input tensors must have the same dimensions.")
    
    # Remove singleton dimensions, if present
    x = torch.squeeze(x, dim=-3) if len(x.shape) == 4 else x
    y = torch.squeeze(y, dim=-3) if len(y.shape) == 4 else y

    if batch_mode:
        batch_size = x.shape[0]
        mean_corr_coeffs = torch.empty(batch_size, dtype=torch.float32)

        for batch_idx in range(batch_size):
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]
            mean_corr_coeffs[batch_idx] = corr2(x_batch, y_batch)

        return mean_corr_coeffs

    else:
        x_mean_centered = x - x.mean(dim=(-2, -1)).reshape(-1, 1, 1)
        y_mean_centered = y - y.mean(dim=(-2, -1)).reshape(-1, 1, 1)

        r = (x_mean_centered * y_mean_centered).sum(dim=(-2, -1)) / torch.sqrt((x_mean_centered**2).sum(dim=(-2, -1)) * (y_mean_centered**2).sum(dim=(-2, -1)))
        mean_corr_coeff = r.mean().item()

        return mean_corr_coeff

# ---------------------- Spectrum Plot -------------------------------
def spectrum_plot(A):

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            u_fft[i,j,:] = abs(np.fft.fft(A[i,j,:]))

    u_fft_mean = np.mean(u_fft, axis = [0,1])

    # Display grid
    plot.grid(True, which ="both")
      
    # Linear X axis, Logarithmic Y axis
    plot.semilogy(range(100), u_fft_mean[0:15] )
      
    # Provide the title for the semilogy plot
    plot.title('Prediction Spectra')
      
    # Give x axis label for the semilogy plot
    plot.xlabel('k')
      
    # Give y axis label for the semilogy plot
    plot.ylabel('Spectra')
      
    # Display the semilogy plot
    plot.savefig('Spectra.png')

# ---------------------- MyDataset -------------------------------
class MyDataset(Dataset):
    def __init__(self, path, normalize_input=None, normalize_output=None):
        ## Path should be a list of path_x and path_y
        path_x = path[0]
        path_y = path[1]

        with h5py.File(path_x, 'r') as f:
            self.x1 = np.array(f['U']) #Loading U
            self.x2 = np.array(f['V']) #Loading V
        self.data = np.concatenate((np.expand_dims(self.x1, 1),np.expand_dims(self.x2, 1)), axis=1)
        self.data = np.transpose(self.data, (0, 1, 3, 2)) # This is to make the shape of data (N, C, Nx, Ny)
        print("The shape of input is: ", self.data.shape)
        self.data_mean = np.mean(self.data, axis = (0,2,3), keepdims=True)
        self.data_std = np.std(self.data, axis= (0,2,3), keepdims=True)

        with h5py.File(path_y, 'r') as f:
            self.label1 = np.array(f['S1'])
            self.label2 = np.array(f['S2'])
            self.label3 = np.array(f['S3'])
        self.label = np.concatenate((np.expand_dims(self.label1, 1),np.expand_dims(self.label2, 1),np.expand_dims(self.label3, 1)), axis=1)
        # self.label = np.expand_dims(self.label, 1)
        self.label = np.transpose(self.label, (0, 1, 3, 2))
        print("The shape of output is: ", self.label.shape)
        self.label_mean = np.mean(self.label, axis = (0,2,3), keepdims=True)
        self.label_std = np.std(self.label, axis = (0,2,3), keepdims=True)

        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        if self.normalize_input:
            self.data = (self.data - self.data_mean)/self.data_std
        if self.normalize_output:
            self.label = (self.label - self.label_mean)/self.label_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.label[idx]

        return sample, label

    def get_normalization_coef(self):
        return (self.data_mean.squeeze(), self.data_std.squeeze()), (self.label_mean.squeeze(), self.label_std.squeeze())

# ---------------------- Count Parameters ------------------------------- Needs to be checked
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0 
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params:,}")
    return total_params

# ---------------------- Contour Comparison ------------------------------- Needs to be checked
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker

def comparison_contour_plot(data1, data2):
    """
    Creates two contour plots of 2D matrices side by side for comparison
    
    Args:
        data1 (numpy.ndarray): A 2D array of size M x N for the first plot
        data2 (numpy.ndarray): A 2D array of size M x N for the second plot
    """
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Set font properties
    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_size(14)

    # Create contour plots for the two data arrays
    contour_levels = 500
    contour_plot1 = axs[0].contourf(data1, cmap='bwr', levels=contour_levels)
    contour_plot2 = axs[1].contourf(data2, cmap='bwr', levels=contour_levels)
    plt.rcParams["text.usetex"] = True

    # Set nice tick locations for the colorbars
    ticks1 = ticker.MaxNLocator(nbins=5).tick_values(data1.min(), data1.max())
    ticks2 = ticker.MaxNLocator(nbins=5).tick_values(data2.min(), data2.max())

    # Add colorbars to the plots
    colorbar1 = plt.colorbar(contour_plot1, ax=axs[0], ticks=ticks1, shrink = 0.7)
    colorbar2 = plt.colorbar(contour_plot2, ax=axs[1], ticks=ticks2, shrink = 0.7)

    # Set titles and axis labels for the subplots
    axs[0].set_title("Contour Plot 1", fontproperties=font)
    axs[0].set_xlabel("X-axis", fontproperties=font)
    axs[0].set_ylabel("Y-axis", fontproperties=font)
    axs[1].set_title("Contour Plot 2", fontproperties=font)
    axs[1].set_xlabel("X-axis", fontproperties=font)
    axs[1].set_ylabel("Y-axis", fontproperties=font)

    # Set equal aspect ratio for the subplots
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')

    # Set alpha value of the face color to 0
    axs[0].set_facecolor((0, 0, 0, 0))
    axs[1].set_facecolor((0, 0, 0, 0))

    # Show the plot
    plt.show()

# ---------------------- Contour Plot -------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker

def create_contour_plot(data):
    """
    Creates a contour plot of a 2D matrix
    
    Args:
        data (numpy.ndarray): A 2D array of size M x N
    """
    # Create a contour plot with a blue-white-red colorbar
    fig, ax = plt.subplots()

    # Set font properties
    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_size(14)

    # Create a contour plot
    contour_levels = 500
    contour_plot = ax.contourf(data, cmap='bwr', levels=contour_levels)
    plt.rcParams["text.usetex"] = True

    # Set nice tick locations for the colorbar
    ticks = ticker.MaxNLocator(nbins=5).tick_values(data.min(), data.max())

    # Add a colorbar to the plot
    colorbar = plt.colorbar(contour_plot, ticks=ticks)

    # Set title and axis labels
    ax.set_title("Contour Plot", fontproperties=font)
    ax.set_xlabel("X-axis", fontproperties=font)
    ax.set_ylabel("Y-axis", fontproperties=font)
    ax.set_facecolor((0, 0, 0, 0))  # Set alpha value of the face color to 0

    # Show the plot
    plt.show()

# -------------------------- 2D Derivative ---------------------------------
def derivative_2d_fhit(T, order, Tname):
    """
    Calculate spatial derivatives for 2D_FHIT
    Derivatives are calculated in spectral space
    Boundary conditions are periodic in x and y spatial dimensions
    Length of domain: 2*pi
    
    Args:
        T (ndarray): A 2D array of size M x N
        order (tuple): A tuple (orderX, orderY) specifying the order of the x and y derivatives, respectively
        Tname (str): A string representing the name of the input variable T
    
    Returns:
        ndarray: A 2D array containing the derivatives of T with respect to x and y
        str: A string representing the name of the output variable
    """
    orderX, orderY = order

    # Validating user input
    if orderX < 0 or orderY < 0:
        raise ValueError("Order of derivatives must be 0 or positive")
    elif orderX == 0 and orderY == 0:
        raise ValueError("Both order of derivatives are 0, at least one of them should be positive")

    Ngrid = T.shape
    Lx = 2 * np.pi  # Length of domain

    # Create wave numbers manually
    kx_pos = np.arange(0, Ngrid[0] / 2 + 1)
    kx_neg = np.arange(-Ngrid[0] / 2 + 1, 0)
    kx = np.concatenate((kx_pos, kx_neg)) * (2 * np.pi / Lx)
    
    # Making meshgrid
    Ky, Kx = np.meshgrid(kx, kx)
    
    ## This part needs to be checked 
    # Ngrid = T.shape
    # Lx = 2 * np.pi  # Length of domain
    
    # # Calculating the wave numbers
    # kx = 2 * np.pi * np.fft.fftfreq(Ngrid[0])
    # ky = 2 * np.pi * np.fft.fftfreq(Ngrid[1])

    # # Making meshgrid
    # Ky, Kx = np.meshgrid(ky, kx)



    # Calculating derivatives in spectral space
    T_hat = np.fft.fft2(T)
    Tdash_hat = ((1j * Kx) ** orderX) * ((1j * Ky) ** orderY) * T_hat
    Tdash = np.real(np.fft.ifft2(Tdash_hat))

    # Naming the variable
    TnameOut = Tname + ''.join(['x' * orderX, 'y' * orderY])

    return Tdash, TnameOut

# -------------------------- Energy Transfer ---------------------------------
import numpy as np

def energy_transfer_2d_fhit(U, V, Tau11, Tau12, Tau22):
    """
    Energy transfer of 2D_FHIT using SGS stress.
    Input is single snapshot (N x N matrix).
    
    Input:
    U, V: Velocities
    Tau11, Tau12, Tau22: SGS stress
    
    Output:
    PTau: energy transfer
    """
    
    Ux, _ = derivative_2d_fhit(U, [1, 0], 'U')
    Uy, _ = derivative_2d_fhit(U, [0, 1], 'U')
    Vx, _ = derivative_2d_fhit(V, [1, 0], 'V')
    
    PTau = -(Tau11 - Tau22) * Ux - Tau12 * (Uy + Vx)
    
    return PTau

# -------------------------- CC Energy Transfer ---------------------------------
import numpy as np

def cc_energy_transfer_2d_fhit(U, V, Tau11FDNS, Tau12FDNS, Tau22FDNS, Tau11Model, Tau12Model, Tau22Model):
    """
    Calculate the correlation coefficient (CC) of energy transfer between the truth (or filtered DNS data) and the model.
    
    Parameters
    ----------
    U, V : numpy.ndarray
        Velocities (NxN matrices).
    Tau11FDNS, Tau12FDNS, Tau22FDNS : numpy.ndarray
        SGS stress of the truth or filtered DNS data (NxN matrices).
    Tau11Model, Tau12Model, Tau22Model : numpy.ndarray
        SGS stress of the model (NxN matrices).

    Returns
    -------
    CCPTau : float
        CC between energy transfer of truth and model.
    CCPTauPos : float
        CC between energy transfer of grid points where P_FDNS > 0 (Positive - diffusion/forward scatter).
    CCPTauNeg : float
        CC between energy transfer of grid points where P_FDNS < 0 (Negative - backscatter).
    """
    
    PTauFDNS = energy_transfer_2d_fhit(U, V, Tau11FDNS, Tau12FDNS, Tau22FDNS)
    PTauModel = energy_transfer_2d_fhit(U, V, Tau11Model, Tau12Model, Tau22Model)

    CCPTau = np.corrcoef(PTauFDNS.ravel(), PTauModel.ravel())[0, 1]
    CCPTauPos = np.corrcoef(PTauFDNS[PTauFDNS > 0], PTauModel[PTauFDNS > 0])[0, 1]
    CCPTauNeg = np.corrcoef(PTauFDNS[PTauFDNS < 0], PTauModel[PTauFDNS < 0])[0, 1]
    
    return CCPTau, CCPTauPos, CCPTauNeg


# -------------------------- Post Proccess ---------------------------------
def post_proccess(U, V, Tau11FDNS, Tau12FDNS, Tau22FDNS, Tau11Model, Tau12Model, Tau22Model):

    """
    Perform post-processing on input data and return calculated means and standard deviations for CC values.

    Args:
        U: A 3D array of U values.
        V: A 3D array of V values.
        Tau11FDNS, Tau12FDNS, Tau22FDNS: 3D arrays of FDNS tau values.
        Tau11Model, Tau12Model, Tau22Model: 3D arrays of model tau values.

    Returns:
        output: A dictionary containing the mean and standard deviation of calculated CC values.
    """
    data_size = U.shape[0]

    # Initialize variables
    CCTau11 = np.zeros(data_size)
    CCTau12 = np.zeros(data_size)
    CCTau22 = np.zeros(data_size)
    CCPTau = np.zeros(data_size)
    CCPTauPos = np.zeros(data_size)
    CCPTauNeg = np.zeros(data_size)

    for i in range(data_size):
        # Calculate the CC for tau11, tau12, tau22 (3 numbers)
        CCTau11[i] = corr2(Tau11Model[i], Tau11FDNS[i])
        CCTau12[i] = corr2(Tau12Model[i], Tau12FDNS[i])
        CCTau22[i] = corr2(Tau22Model[i], Tau22FDNS[i])

        # Calculate CC(PTau>0) & CC(PTau<0) (2 numbers)
        CCPTau[i], CCPTauPos[i], CCPTauNeg[i] = cc_energy_transfer_2d_fhit(U[i], V[i], 
                                                                           Tau11FDNS[i], Tau12FDNS[i],Tau22FDNS[i],
                                                                            Tau11Model[i], Tau12Model[i],Tau22Model[i])

    # Calculate the MEAN and STD for calculated CCs
    output = {
        'CCTau11': (np.mean(CCTau11), np.std(CCTau11)),
        'CCTau12': (np.mean(CCTau12), np.std(CCTau12)),
        'CCTau22': (np.mean(CCTau22), np.std(CCTau22)),
        'CCPTau': (np.mean(CCPTau), np.std(CCPTau)),
        'CCPTauPos': (np.mean(CCPTauPos), np.std(CCPTauPos)),
        'CCPTauNeg': (np.mean(CCPTauNeg), np.std(CCPTauNeg))
    }

    return output

## -------------------------- Plotting Loss ---------------------------------
from matplotlib.ticker import MaxNLocator
def plot_loss(train_loss, test_loss, save_fig=False, fig_name=None):
    
    epoch = np.arange(1, len(train_loss)+1)
    
    # Create a figure
    fig, axs = plt.subplots()

    # Create line plots for the two data arrays
    axs.plot(epoch, train_loss, label='Train Loss', linewidth=2, color='red')
    axs.plot(epoch, test_loss, label='Test Loss', linewidth=2, color='blue')    

    # Set y-axis to log scale
    axs.set_yscale('log')

    # Update the font size for axis labels and title
    font_size = 14
    font = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.titlesize'] = font_size
    plt.rcParams['font.family'] = font
    plt.rcParams["text.usetex"] = True


    # Create scatter plots for the two data arrays
    point_size = 5
    axs.scatter(epoch, train_loss, color='red', marker='o', s=point_size)
    axs.scatter(epoch, test_loss, color='blue', marker='o', s=point_size)

    # Set titles and axis labels for the subplots
    axs.set_title("Training Loss vs. Test Loss")
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")

    # Set the x and y axis limits
    xmin, xmax = 1, len(epoch)
    ymin, ymax = min(train_loss) or min(test_loss), max(train_loss) or max(test_loss)
    axs.set_xlim(xmin, xmax)
    axs.set_ylim(ymin, ymax)
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Set equal aspect ratio for the subplots
    axs.set_aspect('auto')

    # Set alpha value of the face color to 0
    axs.set_facecolor((0, 0, 0, 0))

    # Show the legend
    axs.legend(frameon=False)

    if save_fig:
        if fig_name is None:
            fig_name = "loss.pdf"
        plt.savefig(fig_name, format="pdf")
    # Show the plot
    plt.show()



## -------------------------- NC Dataset ---------------------------------
import os
import logging
from netCDF4 import Dataset
from torch.utils.data import Dataset as TorchDataset
import torch
    
class NCDataset(TorchDataset):
    """
    Dataset for loading NetCDF files.
    """
    def __init__(self, input_directory: str, output_directory: str, variable_names: dict, data_fraction: float = 1.0, normalize_input: bool = False, normalize_output: bool = False):
        """
        :param input_directory: Directory where the input .nc files are stored.
        :param output_directory: Directory where the output .nc files are stored.
        :param variable_names: Dictionary of variable names to load from the .nc files. Should have 'input' and 'output' keys.
        :param data_fraction: Fraction of data files to use (default is 1.0, which means use all files).
        """
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.variable_names = variable_names
        self.data_fraction = data_fraction
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.input_files = self._load_files(input_directory)
        self.output_files = self._load_files(output_directory)

        if normalize_input or normalize_output:
            self._calculate_normalization_parameters()

    def _calculate_normalization_parameters(self):
        """
        Calculate normalization parameters (mean, std) for the entire dataset.
        """
        all_input_data = [self.load_nc_files(self.input_directory, var, idx) for idx in range(len(self.input_files)) for var in self.variable_names['input']]
        all_output_data = [self.load_nc_files(self.output_directory, var, idx) for idx in range(len(self.output_files)) for var in self.variable_names['output']]

        if self.normalize_input:
            self.input_mean = np.mean(all_input_data)
            self.input_std = np.std(all_input_data)

        if self.normalize_output:
            self.output_mean = np.mean(all_output_data)
            self.output_std = np.std(all_output_data)

    @staticmethod
    def load_nc_files(directory: str, variable_name: str, file_index: int):
        """
        Load data from a .nc file.

        :param directory: Directory where the .nc files are stored.
        :param variable_name: Name of the variable to load from the .nc files.
        :param file_index: Index of the file to load.
        :return: Data loaded from the .nc file, or None if an error occurred.
        """
        files = os.listdir(directory)
        files = [f for f in files if f.endswith('.nc')]

        if not files:
            logging.error("No .nc files found in the directory.")
            return None

        files.sort(key=lambda x: int(x.split('.')[0]))

        if file_index < 0 or file_index >= len(files):
            logging.error("file_index is out of range.")
            return None

        file_path = os.path.join(directory, files[file_index])

        try:
            with Dataset(file_path, 'r') as nc_file:
                if variable_name not in nc_file.variables:
                    logging.error(f"Variable {variable_name} not found in the file.")
                    return None

                data = nc_file.variables[variable_name][:]
        except Exception as e:
            logging.error(f"Error reading the .nc file: {str(e)}")
            return None

        return data

    def _load_files(self, directory):
        """
        Load the list of .nc files from the directory.

        :return: Sorted list of .nc files in the directory.
        """
        files = os.listdir(directory)
        files = [f for f in files if f.endswith('.nc')]
        files.sort(key=lambda x: int(x.split('.')[0]))
        
        # Only use a fraction of files if specified
        num_files = int(len(files) * self.data_fraction)
        files = files[:num_files]
        
        return files
    
    def __len__(self):
        """
        Get the number of .nc files in the directory.

        :return: Number of .nc files in the directory.
        """
        return len(self.input_files)
    
    def __getitem__(self, idx: int):
            inputs = [self.load_nc_files(self.input_directory, var, idx) for var in self.variable_names['input']]
            outputs = [self.load_nc_files(self.output_directory, var, idx) for var in self.variable_names['output']]
            
            if any(v is None for v in inputs+outputs):
                return None

            inputs = [torch.from_numpy(data) for data in inputs]
            outputs = [torch.from_numpy(data) for data in outputs]

            # Normalize inputs if flag is set
            if self.normalize_input:
                inputs = [(x - torch.mean(x)) / torch.std(x) for x in inputs]
            
            # Normalize outputs if flag is set
            if self.normalize_output:
                outputs = [(x - torch.mean(x)) / torch.std(x) for x in outputs]

            inputs = torch.cat(inputs, dim=1)
            outputs = torch.cat(outputs, dim=1)

            return inputs, outputs
