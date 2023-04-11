import torch
import torch.nn.functional as F

from models.efficientnet import EfficientNet
from data.transform import get_prediction_transform

from PIL import Image
import numpy as np
import os
from tqdm import tqdm



def load_model(efficientnet_version, weight_filepath, num_classes):
    """
    Loads EfficientNet model
    
    Args:
        efficientnet_version (str): EfficientNet model architecture version. e.g. 'b3', 'b5', 'b7'
        weight_filepath (str): trained model weight file path
        num_classes (int): total number of classes. e.g. 2
        
    Returns:
        model: trained EfficientNet model
    """
    
    
    model = EfficientNet(version = efficientnet_version,
                         pretrained = True,
                         num_classes = num_classes)

    checkpoint = torch.load(weight_filepath)
    model.load_state_dict(checkpoint['model'])
    
    return model


def inference(test_image_folder, # (str): folder containing test images
              output_folder, # (str): folder where prediction text files will be generated
              image_size, # (tuple): image resolution size. e.g. (256, 256)
              weight_filepath, # (str): trained model weight file path
              mean, # (list): a list of mean values for red, green, blue color channels. e.g. [0.485, 0.456, 0.406]
              std,  # (list): a list of standard deviation values for red, green, blue color channels. e.g. [0.229, 0.224, 0.225]
              class_names,  # (list): a list of class names. e.g. ['not_school', 'school']
              efficientnet_version): # (str): EfficientNet model architecture version. e.g. 'b3', 'b5', 'b7'
    
    """
    Run model prediction on all images inside a given folder and save probability scores in text files.
    Format: "non_school_probability, school_probability"
             "0.0005840617 0.999416"
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = len(class_names)
    image_list = sorted(os.listdir(test_image_folder))
    failed_images = []

    # create folder for output files
    os.makedirs(output_folder, exist_ok = True)

    model = load_model(efficientnet_version = efficientnet_version, 
                       weight_filepath = weight_filepath, 
                       num_classes = num_classes)
    model.eval()
    model.to(device)

    for filename in tqdm(image_list):

        try:        
            # read image
            image_path = os.path.join(test_image_folder, filename)
            image = np.array(Image.open(image_path))[:,:,:3]

            transform = get_prediction_transform(image_size = image_size,
                                                 mean = mean,
                                                 std = std)
            image = transform(image)
            image = torch.unsqueeze(image, 0)
            image = image.to(device)

            # predict
            outputs = model(image)
            outputs_softmax = F.softmax(outputs[0], dim = 0).cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            # pred_class_name = class_names[np.argmax(outputs[0])]

            # save prediction results in text file
            preds_str = ' '.join([str(i) for i in outputs_softmax])
            txt_save_dir = os.path.join(output_folder, filename.replace('.png', '.txt'))
            with open(txt_save_dir, 'w') as file:
                file.write(preds_str)

        except:
            print(f'Error: {filename}')
            failed_images.append(filename)

    if len(failed_images) > 0:
        print("Failed to run model prediction on the following images:\n", failed_images)