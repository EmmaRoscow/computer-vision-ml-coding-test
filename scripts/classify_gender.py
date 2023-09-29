import torch
import numpy as np
from simple_cnn import simple_cnn

class classify_gender:
    
    def __init__(self, model_path='/models/simple_cnn', model_name='model'):
   
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(DEVICE)

        self.model = simple_cnn().to(self.device)

        model_path = model_path + '/' + model_name
        self.model.load_state_dict(torch.load(model_path))
    

    def classify(self, images):

        # If single image, transpose to PyTorch's requirements and unsqueeze to turn it into a batch
        if images.ndim == 3:
            images = np.transpose(images, (2, 0, 1))
            images = np.expand_dims(images, axis=0)
        elif images.ndim == 4:
            images = np.transpose(images, (0, 3, 1, 2))
        else:
            raise ValueError("Input must have three dimensions for a single image or four dimensions for a batch of images")

        # Convert to tensor
        images = torch.Tensor(images).to(self.device)

        # Predict gender
        output = self.model(images)
        _, pred = torch.max(output, 1)
        gender = pred.cpu().detach().numpy()

        return gender