import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import glob
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Load EfficientNet-B0 ---
def load_efficientnet_b0(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model

model_path = r"Model_Outputs/efficientnet_b0/model_epoch_8.pth"
model = load_efficientnet_b0(num_classes=38).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Model loaded successfully!")

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((244, 244)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust', 
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper_bell___Bacterial_spot',
    'Pepper_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Yellow_Leaf_Curl_Virus',
    'Tomato___mosaic_virus',
    'Tomato___healthy'
]

# --- Predict with Grad-CAM ---
def predict_with_gradcam(img_path, model, target_layers, save_dir, true_label=None):
    # Load image
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = np.float32(img_rgb) / 255.0

    # Preprocess
    input_tensor = preprocess_image(
        img_float,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ).to(device)

    # Prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if output.shape[1] == 1:  # binary
            prob = torch.sigmoid(output).item()
            pred_label = 1 if prob >= 0.5 else 0
        else:  # multi-class
            prob = torch.softmax(output, dim=1)
            pred_label = prob.argmax(dim=1).item()
            prob = prob.max().item()

    # Grad-CAM
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    grayscale_cam = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
    cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    # Save images
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(save_dir, f"{base}_orig.jpg"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, f"{base}_cam.jpg"), cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Actual: {true_label}")
    axes[0].axis("off")

    # Map predicted label to class name
    pred_text = class_names[pred_label] if pred_label < len(class_names) else "Unknown"

    axes[1].imshow(cam_image)
    axes[1].set_title(f"Pred: {pred_text} | Prob: {prob:.4f}")  # <-- replaced numeric label
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    return pred_label, prob, cam_image


if __name__ == "__main__":
    test_dir = r"dataset/test"
    save_dir = r"Model_Outputs/XAI_outputs"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing test dataset: {test_dir}")
    print(f"Grad-CAM outputs will be saved to: {save_dir}\n")

    # --- Images inside class subfolders ---
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        image_paths = glob.glob(os.path.join(class_path, "*.*"))
        print(f"\nProcessing class folder: {class_name} ({len(image_paths)} images)")
        for i, img_path in enumerate(image_paths[:5], 1):  # limit samples
            print(f"[{class_name}] Processing image {i}/{min(5, len(image_paths))}: {os.path.basename(img_path)}")
            predict_with_gradcam(
                img_path,
                model,
                target_layers=[model.features[-1]],
                save_dir=save_dir,
                true_label=class_name
            )

    print("\nGrad-CAM processing completed for all images!")


