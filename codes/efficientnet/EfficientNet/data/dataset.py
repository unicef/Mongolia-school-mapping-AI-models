from torchvision import datasets
from torch.utils.data import DataLoader
from .transform import get_train_transform, get_validation_transform


def create_dataloader(data_dir, batch_size, image_size, mean, std, num_workers = 2, train = True):
    
    if train:
        shuffle = True 
        transform = get_train_transform(image_size = image_size, 
                                       mean = mean, 
                                       std = std)
    else:
        shuffle = False
        transform = get_validation_transform(image_size = image_size,
                                             mean = mean, 
                                             std = std)
        
    dataset = datasets.ImageFolder(root = data_dir,
                                   transform = transform)
    
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            num_workers = num_workers)
    
    return dataloader  