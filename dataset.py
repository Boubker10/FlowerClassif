import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class SimpleTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, aug=None) -> None:
        self.dataset = []
        self.root_dir = root_dir
        
        # Ajouter les différentes classes de fleurs et leurs étiquettes
        self.__add_dataset__("sunflower", [1, 0, 0])
        self.__add_dataset__("daisy", [0, 1, 0])
        self.__add_dataset__("rose", [0, 0, 1])

        # Vérifier si `aug` est un `Compose`, sinon l'utiliser directement
        if aug is None:
            self.augmentation = transforms.Compose([
                transforms.Resize((150, 150)),
                transforms.CenterCrop((128, 128)),
                transforms.ToTensor()  # Convertir l'image en tensor
            ])
        else:
            self.augmentation = aug
    
    def __add_dataset__(self, dir_name: str, class_label: list[int]) -> None:
        full_path = os.path.join(self.root_dir, dir_name)
        label = np.array(class_label)
        for fname in os.listdir(full_path):
            fpath = os.path.join(full_path, fname)
            fpath = os.path.abspath(fpath)
            self.dataset.append((fpath, label))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]

        # Charger l'image sous forme de PIL.Image
        image = Image.open(fpath).convert('RGB')

        # Appliquer les transformations (l'image doit être un PIL.Image ici)
        image = self.augmentation(image)

        # Convertir le label en tensor
        label = torch.Tensor(label)

        return image, label
