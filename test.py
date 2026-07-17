import torch
import torchvision.models as models
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 1. Setup Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# 2. Prepare Image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#use this to change the image name
img_path = 'Pomeranian.jpeg'
img = Image.open(img_path).convert('RGB')
transformed_img = transform(img)
input_img = transformed_img.unsqueeze(0)

# 3. Get Model Prediction
output = model(input_img)
pred_label_idx = torch.argmax(output, 1)

# 4. Calculate Attribution (Integrated Gradients)
ig = IntegratedGradients(model)
attributions = ig.attribute(input_img, target=pred_label_idx, n_steps=50)

# 5. Process into a "Sketch"
# Convert attribution tensor to numpy [H, W, 3]
attr_np = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))

# Sum values and then take the absolute value
importance_map = np.abs(attr_np.sum(axis=2))

# Normalize
importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min())
importance_map = (importance_map * 255).astype(np.uint8)

#create sketch
_, sketch = cv2.threshold(importance_map, 60, 255, cv2.THRESH_BINARY_INV)

# 6. Results
plt.figure(figsize=(12, 5))

#Original Image
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

# Heatmap
plt.subplot(1, 3, 2)
plt.title("Heatmap")
#use deep lift to show which parts of the image are hot
plt.imshow(importance_map, cmap="hot")
plt.axis('off')

# sketch based on 
plt.subplot(1, 3, 3)
plt.title("Sample Sketch")
plt.imshow(sketch, cmap='gray')
plt.axis('off')

plt.show()

# Save the sketch
cv2.imwrite("drawing_sketch.jpeg", sketch)
print("saved sketch")