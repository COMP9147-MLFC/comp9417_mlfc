import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision.models import vgg19, VGG19_Weights

# load an image and transform it
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    in_transform = transforms.Compose([
        transforms.Resize((size, int(1.5*size))),  
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Transform the image
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

# show tensor as image
def show_tensor_image(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    plt.imshow(image)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# content and style images
content = load_image("target.png").to(device)
style = load_image("style.png", shape=content.shape[-2:]).to(device)

# VGG19 pre-trained model
weights = VGG19_Weights.IMAGENET1K_V1
vgg = vgg19(weights=weights).features.to(device).eval()

# extract features from images
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2', 
                  '28': 'conv5_1'}
                  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

# calculates gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram 

# get content and style features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate gram matrices for each layer of style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create target image
target = content.clone().requires_grad_(True).to(device)

# weights
style_weights = {'conv1_1': 0.2,
                 'conv2_1': 0.4,
                 'conv3_1': 0.6,
                 'conv4_1': 0.8,
                 'conv5_1': 1.0}

content_weight = 0.5  # Alpha
style_weight = 1e6  # Beta

# update target image
optimizer = optim.Adam([target], lr=0.003)
steps = 500  
    
for i in range(1, steps + 1):
    optimizer.zero_grad()
    target_features = get_features(target, vgg)
    content_loss = F.mse_loss(target_features['conv4_2'], content_features['conv4_2'])

    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * F.mse_loss(target_gram, style_gram)
        _, d, h, w = target_feature.shape
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward(retain_graph=True if i < steps else False)
    optimizer.step()

    if i % 50 == 0:
        print(f'Iteration {i}/{steps}, Total Loss: {total_loss.item()}')

def tensor_to_pil(tensor):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    image = Image.fromarray((image * 255).astype(np.uint8))
    return image

# Convert to a PIL image
final_image = tensor_to_pil(target)
final_image.save('stylized_image.jpg')
print('Image saved as stylized_image.jpg')