from torchvision import transforms


def get_train_transform(image_size, mean, std):
    
    train_transform = transforms.Compose([
        
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(1, 179)),
        transforms.RandomAffine(degrees=0, scale=(1.01, 1.20)),
        transforms.ColorJitter(brightness=(0.8, 1.2)),   
        
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)
    ])
    
    return train_transform


def get_validation_transform(image_size, mean, std):
    
    valid_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)
    ])
    
    return valid_transform


def get_prediction_transform(image_size, mean, std):
    
    predict_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)
    ])
    
    return predict_transform
