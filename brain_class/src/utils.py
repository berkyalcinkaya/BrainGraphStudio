import os
import numpy as np
from numpy.core.numeric import full
from scipy.io import loadmat
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from scipy.sparse import coo_matrix
import glob

def generate_flist(dir):
    curr_dir = os.getcwd()
    _, fname = os.path.split(dir) 
    fname = fname + ".txt"
    files = [os.path.join(curr_dir, file) for file in glob.glob(os.path.join(dir, "*"))]
    files = "\n".join(files)
    with open(fname, 'a') as f:
        f.write(files)
    return fname

def copy_elements(input_list, x):
    if x <= 0:
        raise ValueError("x must be a positive integer")

    copied_list = []
    for item in input_list:
        for _ in range(x):
            copied_list.append(item)
    return copied_list

def expand_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def adjacency_matrix_to_coo(adjacency_matrix):
    coo = coo_matrix(adjacency_matrix)
    return coo.data, np.array([coo.row, coo.col], dtype = np.uint8)

def show_binary(im):
    plt.imshow(im, cmap = plt.cm.gray)
    plt.show()
    return

def is2D(np_array):
    return len(np_array.shape)==2

def display_images_in_grid(images, multiplier = (4,4), title = None, rows = None, **kwargs):
    num_images = len(images)
    if rows is None:
        rows = math.ceil(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)
    
    fig, axes = plt.subplots(rows, cols)
    
    for i, image in enumerate(images):
        ax = axes[i // cols, i % cols]
        ax.imshow(image, **kwargs)
        ax.axis('off')
    
    plt.tight_layout()
    # Dynamically adjust figure size for Jupyter Notebook
    # figsize = (cols * multiplier[0], rows * multiplier[1])  # Modify the size multiplier as needed
    # fig.set_size_inches(figsize[0], figsize[1])

    if title:
        plt.suptitle(title, fontsize = 8, fontweight = "bold")
    
    plt.show()

def density_threshold(networks, density_percentile = 10, v = False, positive_only = False):
    if is2D(networks):
        networks = np.expand_dims(networks, 0)
    new_networks = networks.copy()
    for i in tqdm(range(networks.shape[0])):
        if positive_only:
            thresh = np.percentile(networks[i][networks[i]>=0], density_percentile)
        else:
            thresh = np.percentile(networks[i], density_percentile)
        if v:
            print("Threshold for", i, ": ", thresh)
        new_networks[i,:,:][new_networks[i]<thresh] = 0
    
    if is2D(networks):
        return new_networks[0,:,:]
    return new_networks 
    


def load_mat_flist(flist_file:str, key = 'CorrMatTot', is_3D = True)->np.ndarray:
    '''loads list of N .mat arrays as a numpy array concatenated along axis 0 
    
    PARAMETERS
    flist_file (str)
        a text file containing the path to the files on seperate lines
    
    RETURNS
    data (np.ndarray)
        the arrays expanded along the first axis. Shape (N, r,c) where r x c is the shape of a singular .mat array'''
    
    def get_mat(file):
        return loadmat(file.strip())[key]

    
    with open(flist_file, "r") as f:
        files = [file.strip() for file in f.readlines()]
    
    if is_3D:
        ars = []
        for file in files:
            mat = get_mat(file)
            for i in range(mat.shape[-1]):
                ars.append(mat[:,:,i])
        return np.array(ars, dtype = np.float64)
    else:
        return np.array([get_mat(file) for file in tqdm(files)], dtype = np.float64)



def extract_subnetworks_keep_full(subnetwork_indices:np.ndarray, fullnetwork:np.ndarray)->np.ndarray:
    '''
    Keeps only the indices of the fullnetwork that are specified in the binary subnetwork array
    
    PARAMETERS
    subnetwork_indices: (nd.ndarray)
        shape - N x fullnetwork.rows. Binary, where 1 specifies an index to keep in fullnetwork, 0 to discard
    
    fullnetwork: (np.ndarray)
        the raw networks of shape m x n x n or n x n if single array.
    
    RETURNS
    subnetwork (np.ndarray)
        m x n x n if 3D input else n x n with only the rows specified kept, all others set to 0 
    '''

    mask = np.outer(subnetwork_indices, subnetwork_indices)
    subnetwork = fullnetwork.copy()
    if len(subnetwork.shape) == 2:  
        subnetwork[mask==0]=0
        return subnetwork
    else:
        subnetwork[:, mask==0]=0
        return subnetwork

def extract_subnetworks(subnetwork_indices:np.ndarray, fullnetwork:np.ndarray)->np.ndarray:
    mask = subnetwork_indices==1
    if len(fullnetwork.shape)==3:
        return fullnetwork[:,mask,:][:,:,mask].copy()
    else:
        return fullnetwork[mask][:,mask]

        
def extract_subnetworks_from_csv(csv_file:str, fullnetwork:np.ndarray, name = None)->dict:
    '''Extracts a set of subnetworks whose indicies are given in a csv file
    
    PARAMETERS
    csv_file: (str)
        path to the csv file that contains subnetwork specifications. A single column is taken as a subnetwork and is expected to have a label as row 0 
    
    fullnetwork: (np.ndarray)
        array(s) - 2D or 3D - from which to extract subnetworks
    
    name: (optional, default - None)
        name or index of subnetwork to use from csv_file
    
    RETURNS
    subnetwork_dict (dictionary)
        keys are name of subnetwork taken from row1 of the csv_file, values are numpy array of subnetwor'''
    
    df = pd.read_csv(csv_file)
    if name is not None:
        if type(name) is int:
            df = df.iloc[:, name]
        else:
            df = df[name]
    subnetwork_dict = {}
    for subnetwork_name in df.columns:
        idx = df[subnetwork_name].to_numpy()
        subnetwork_dict[subnetwork_name] = extract_subnetworks(idx, fullnetwork)
    return subnetwork_dict


def calculate_pearson_coefficient(adjacency_matrix):
    # Calculate the correlation coefficient between each node and all others
    pearson_matrix = np.zeros_like(adjacency_matrix, dtype=float)
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if i == j:
                pearson_matrix[i, j] = 1.0
            else:
                x = adjacency_matrix[i, :]
                y = adjacency_matrix[j, :]
                pearson_coefficient = np.corrcoef(x, y)[0, 1]
                pearson_matrix[i, j] = pearson_coefficient
    
    return pearson_matrix

def calculate_place_holder_features(adjacency_matrix):
    return np.identity(adjacency_matrix.shape[0])

