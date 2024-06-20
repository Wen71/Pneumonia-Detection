import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor, pipeline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import normalize, to_tensor
import cv2
import scipy as sp
import scipy.ndimage

# Load the model and feature extractor
model_name = "chest-xray-classification"
pipe = pipeline('image-classification', model=model_name, device=0)
# Load the model and feature extractor
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load an image
loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

def load_image(img_path):
    image = Image.open(img_path)
    image = image.convert("RGB")
    return image

def preprocess_image(image):
    loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    img_tensor = loader(image)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return img_tensor

def get_last_hidden_states(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']  # shape: (1, 3, 224, 224)

    pixel_values = pixel_values.to(device)

    # Perform a forward pass
    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True)
    return outputs


def generate_gradcam(image_path):
    image = load_image(image_path)
    outputs = get_last_hidden_states(image)
    last_hidden_states = outputs.hidden_states[-1]  # shape: (1, 197, 768)
    last_hidden_states = last_hidden_states[:, 1:, :]  # Remove the [CLS] token (1, 196, 768)
    
    out_features = last_hidden_states.reshape(14,14,768) # Changes shape from 2048 x 7 x7 to 7 x 7 x 2048. Just performs a matrix transpose on the output features tensor.
    W = model.classifier.weight #We obtain all the weights connecting the Global Average Pooling layer to the final fully connected layer.
    logits = outputs.logits  # shape: (1, 1000)
    predicted_class_idx = logits.argmax(-1).item()
    w = W[predicted_class_idx,:]
    cam = np.dot(out_features.detach(),w.detach().cpu())

    class_activation = scipy.ndimage.zoom(cam, zoom=(16,16),order=1)

    img = loader(image).float()
    img = np.transpose(img,(1,2,0)) #matplotlib supports channel-last dimensions so we perform a transpose operation on our image which changes its shape to (224x224,3)
    #we plot both input image and class_activation below to get our desired output.
    plt.imshow(img, cmap='jet',alpha=1)  #jet indicates that color-scheme we are using and alpha indicates the intensity of the color-scheme
    plt.imshow(class_activation,cmap='jet',alpha=0.5)
    plt.colorbar()
    plt.show()
    
if __name__=="__main__": 
    img_path = 'chest_xray/train/NORMAL/NORMAL2-IM-1345-0001-0002 copy.jpeg'
    generate_gradcam(img_path)


