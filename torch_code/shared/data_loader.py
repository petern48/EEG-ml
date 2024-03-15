import numpy as np
def train_data_prep(X,y,sub_sample,average,noise):

    total_X = None
    total_y = None

    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:800]
    print('Shape of X after trimming:',X.shape)

    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)


    total_X = X_max
    total_y = y
    print('Shape of X after maxpooling:',total_X.shape)

    # Averaging + noise
    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)

    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))
    print('Shape of X after averaging+noise and concatenating:',total_X.shape)

    # Subsampling

    for i in range(sub_sample):

        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)

        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))


    print('Shape of X after subsampling and concatenating:',total_X.shape)
    print('Shape of Y:',total_y.shape)
    return total_X,total_y


def test_data_prep(X, sub_sample):

    total_X = None


    # Trimming the data (sample,22,1000) -> (sample,22,800)
    X = X[:,:,0:800]
    print('Shape of X after trimming:',X.shape)

    # Maxpooling the data (sample,22,800) -> (sample,22,800/sub_sample)
    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)


    total_X = X_max
    print('Shape of X after maxpooling:',total_X.shape)

    return total_X


import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd

class EEG_Data(Dataset):

    def __init__(self, root_dir, split, transform=None, label_dict=None):
        """
        Initialize the eeg dataset with the root directory for the images,
        the split (train/val/test), an optional data transformation,
        and an optional label dictionary.

        Args:
            root_dir (str): Root directory for the eeg images.
            split (str): Split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional data transformation to apply to the images.
            label_dict (dict, optional): Optional dictionary mapping integer labels to class names.
        """
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.datastorch = []
        self.labels = []
        self.label_dict = ["Cue Onset left", "Cue Onset right", "Cue onset foot", "Cue onset tongue"]

        ################# Your Implementations #################################
        if self.split == 'train':
            # First generating the training and validation indices using random splitting
            X_train_valid = np.load(self.root_dir+"X_train_valid.npy")
            y_train_valid = np.load(self.root_dir+"y_train_valid.npy")

            # X_train_valid_prep,y_train_valid_prep = train_data_prep(X_train_valid,y_train_valid,2,2,True)
            # ind_valid = np.random.choice(8460, 1000, replace=False)
            # ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))

            # # Creating the training and validation sets using the generated indices
            # (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid]
            # (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]

            self.datas = torch.from_numpy(X_train_valid)
            self.labels = [int(i-769) for i in torch.from_numpy(y_train_valid)]

        if self.split == 'val':
            # First generating the training and validation indices using random splitting
            X_train_valid = np.load(self.root_dir+"X_train_valid.npy")
            y_train_valid = np.load(self.root_dir+"y_train_valid.npy")
            # X_train_valid_prep,y_train_valid_prep = train_data_prep(X_train_valid,y_train_valid,2,2,True)
            # ind_valid = np.random.choice(8460, 1000, replace=False)
            # ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))

            # # Creating the training and validation sets using the generated indices
            # (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid]
            # (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]


            self.datas = torch.from_numpy(X_train_valid)
            self.labels = [int(i-769) for i in torch.from_numpy(y_train_valid)]

        if self.split == 'test':
            x_test_og = np.load(self.root_dir+"X_test.npy")
            # x_test = test_data_prep(x_test_og)  # (2115, 1)  vals from 0-8 for participant
            y_test = np.load(self.root_dir+"y_test.npy")  # (443, 1)
            self.datas = torch.from_numpy(x_test_og)
            self.labels = [int(i-769) for i in torch.from_numpy(y_test)]

        ################# End of your Implementations ##########################

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        dataset_len = 0
        ################# Your Implementations #################################
        # Return the number of images in the dataset
        dataset_len = len(self.datas)
        ################# End of your Implementations ##########################
        return dataset_len

    def __getitem__(self, idx):
        """
        R10140    idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        ################# Your Implementations #################################
        # Load and preprocess image using self.root_dir,
        # self.filenames[idx], and self.transform (if specified)

        data = self.datas[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)
        ################# End of your Implementations ##########################
        return data, label