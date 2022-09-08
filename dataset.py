import torch
import h5py
import numpy as np

class FireEventDataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_path, indice_key, augment=False):
        with h5py.File(hdf5_path, 'r') as f:
            dset_x = f['wave_segments'][:,:]
            dset_y = f['class_labels'][:,:]
            dset_stats = f['statistics']
            split_indices = f[indice_key]
            self.x = dset_x[split_indices]
            self.y = dset_y[split_indices]
            self.mean, self.variance, self.sample_rate = dset_stats[0]
            class_0 = np.sum(self.y == 0) / self.y.shape[0]
            class_1 = np.sum(self.y == 1) / self.y.shape[0]
            print(indice_key, " shape: {}, balance: (0 : {}, 1 : {})".format(self.x.shape, class_0, class_1))
           
        if augment:
            with h5py.File("esc50_sr_32000.hdf5", "r") as f:
                self.aug = f['augmentation_segments'][:,:]
        self.augment = augment
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        if self.augment:
            p = np.random.uniform(0.1, 0.5)
            idx = np.random.randint(0, len(self.aug))
            aug = self.aug[idx]
            x_aug = (x * (1-p)) + (aug * p)
            return x_aug, y
        else:
            return x, y

