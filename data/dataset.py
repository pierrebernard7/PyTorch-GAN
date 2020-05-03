import h5py
from pathlib import Path
import torch
from torch.utils import data
from tqdm import tqdm
import numpy as np
import scipy.misc as sc
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.

    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """

    def __init__(self, file_path: str, dataset_key: str, input_size: tuple):

        self.input_size = input_size
        with h5py.File(file_path, 'r') as f:
            patient_list = list(f[dataset_key].keys())
        # Go through each patient of the key dataset
        data_x = []
        data_y = []
        infos = []
        with h5py.File(file_path, 'r') as f:
            for patient_id in tqdm(patient_list, total=len(patient_list),
                                   desc="Loading {} group".format(dataset_key),
                                   unit="patient"):
                set_patient_view_keys_format = "{}/{}/{{}}".format(dataset_key, patient_id)
                for view in f[dataset_key][patient_id].keys():
                    set_patient_view = f[set_patient_view_keys_format.format(view)]
                    imgs, gts, info = (set_patient_view['im'][()],
                                       np.eye(4, dtype='float32')[set_patient_view['gt'][()]],
                                       # set_patient_view['gt'][()],
                                       set_patient_view.attrs['info'])
                    data_x.append(imgs[0, :, :,
                                  :])  # data_x(type:list) // imgs.shape (type:np.array) => (2, 256, 256, 1)=> (=0 : es =1 : ed, height, width, depth=n of feature maps = n of input channels)
                    data_x.append(imgs[1, :, :, :])
                    data_y.append(gts[0, :, :])
                    data_y.append(gts[1, :, :])
                    # data_y.append(np.expand_dims(gts[0, :, :], axis=-1))
                    # data_y.append(np.expand_dims(gts[1, :, :], axis=-1))
                    infos.append([patient_id, view, info[0], info[1], info[6], info[7]])
                    infos.append([patient_id, view, info[0], info[1], info[6], info[7]])

        self.imgs = np.array(data_x)
        self.gt = np.array(data_y)
        self.infos = infos

    def transform(self, image, mask, input_size):
        ## Option 1 - Resize images and masks directly (as Tensors) using torch.nn.functional.interpolate

        # image = F.interpolate(image.unsqueeze(0), size=input_size, mode='nearest').squeeze(0)
        # mask = F.interpolate(mask.unsqueeze(0), size=input_size, mode='nearest').squeeze(0)

        ## Option 2 - Resize images and masks as PIL Images
        # Convert image and mask to PIL image
        image = transforms.ToPILImage(mode='L')(image.squeeze_(0))
        mask0 = transforms.ToPILImage(mode='L')(mask[0])
        mask1 = transforms.ToPILImage(mode='L')(mask[1])
        mask2 = transforms.ToPILImage(mode='L')(mask[2])  # mode = L : 8-bits pixels (32-bits per pixel), see modes :
        mask3 = transforms.ToPILImage(mode='L')(mask[3])  # https://pillow.readthedocs.io/en/5.1.x/handbook/concepts.html
        image = image.resize(input_size, Image.NEAREST)
        mask0 = mask0.resize(input_size, Image.NEAREST)
        mask1 = mask1.resize(input_size, Image.NEAREST)
        mask2 = mask2.resize(input_size, Image.NEAREST)  # Nearest neighbour interpolation method
        mask3 = mask3.resize(input_size, Image.NEAREST)
        image = transforms.ToTensor()(image)
        mask0 = transforms.ToTensor()(mask0)
        mask1 = transforms.ToTensor()(mask1)
        mask2 = transforms.ToTensor()(mask2)
        mask3 = transforms.ToTensor()(mask3)
        mask = torch.cat((mask0, mask1, mask2, mask3), 0).unsqueeze_(0)
        return image, mask

    def __getitem__(self, index):
        x = torch.from_numpy(np.transpose(self.imgs[index], (2, 0, 1)))
        y = torch.from_numpy(np.transpose(self.gt[index], (2, 0, 1)))
        x, y = self.transform(image=x, mask=y, input_size=self.input_size)
        return x, y

    def __len__(self):
        return len(self.infos)
