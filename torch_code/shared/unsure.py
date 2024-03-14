def preprocess_wrapper_trainval(X_train_valid, y_train_valid, mode):
    X_train_valid_prep,y_train_valid_prep = train_data_prep(X_train_valid,y_train_valid,2,2,True)
    np.random.seed(0)
    ind_valid = np.random.choice(8460, 1000, replace=False)
    ind_train = np.array(list(set(range(8460)).difference(set(ind_valid))))

    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = X_train_valid_prep[ind_train], X_train_valid_prep[ind_valid]
    (y_train, y_valid) = y_train_valid_prep[ind_train], y_train_valid_prep[ind_valid]

    if (mode == "val"):
        print("validation set returned")
        return x_valid, y_valid
    else:
        return x_train, y_train

class EEG_Data(Dataset):

    def __init__(self, root_dir, split, preprocess_wrapper_trainval, preprocess_test, transform=None, label_dict=None):
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

            X_train, Y_train =  preprocess_wrapper_trainval(X_train_valid, y_train_valid, "train")

            self.datas = torch.from_numpy(X_train)
            self.labels = [int(i-769) for i in torch.from_numpy(Y_train)]

        if self.split == 'val':
            # First generating the training and validation indices using random splitting
            X_train_valid = np.load(self.root_dir+"X_train_valid.npy")
            y_train_valid = np.load(self.root_dir+"y_train_valid.npy")

            X_valid, Y_valid =  preprocess_wrapper_trainval(X_train_valid, y_train_valid, "val")
            
            self.datas = torch.from_numpy(X_valid)
            self.labels = [int(i-769) for i in torch.from_numpy(Y_valid)]

        if self.split == 'test':
            x_test_og = np.load(self.root_dir+"X_test.npy")
            x_test = test_data_prep(x_test_og)  # (2115, 1)  vals from 0-8 for participant
            y_test = np.load(self.root_dir+"y_test.npy")  # (443, 1)

            self.datas = torch.from_numpy(x_test)
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