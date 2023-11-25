import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import TensorDataset
from torchvision import transforms


class FaceDataset(TensorDataset):
    def __init__(self, data_root, same_prob=0.2):
        self.dataset = []
        self.count = []
        path_list = glob.glob('{}/*'.format(data_root))

        for path in tqdm(path_list, desc='loading dataset'):
            file_list = glob.glob('{}/*.*g'.format(path))
            self.dataset.append(file_list)
            self.count.append(len(file_list))
        
        self.remap = []
        for identity in range(len(self.dataset)):
            for serial in range(len(self.dataset[identity])):
                self.remap.append((identity, serial))
        
        self.same_prob = same_prob
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        identity, serial = self.remap[index]
        X_s = Image.open(self.dataset[identity][serial])

        if np.random.rand() > self.same_prob:
            index = np.random.randint(len(self.remap))
            identity, serial = self.remap[index]
            X_t = Image.open(self.dataset[identity][serial])
            same = False
        else:
            X_t = X_s.copy()
            same = True
        
        X_s = self.transforms(X_s)
        X_t = self.transforms(X_t)
        return X_s, X_t, same

    def __len__(self):
        return len(self.remap)
