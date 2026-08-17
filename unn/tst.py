import os
import numpy as np
import torch
from PIL import Image
from kajsrg import UNetSR

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def test_single_image():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load model architecture and weights
    model = UNetSR(in_channels=1, out_channels=1).to(DEVICE)
    
    weights_path = os.path.join(SCRIPT_DIR, "unet_sr_model_v3.pth")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(SCRIPT_DIR, "unet_sr_model_v3.pth")

    print(f"Loading model weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

# 2. Setup matching file paths inside train/train/
    filename = "000226.npy"  # Or any file that exists in both folders
    
    lr_dir = os.path.join(SCRIPT_DIR, "train", "train", "NoisyLR")
    gt_dir = os.path.join(SCRIPT_DIR, "train", "train", "GT")

    lr_path = os.path.join(lr_dir, filename)
    gt_path = os.path.join(gt_dir, filename)

    # Fallback to the first available file in the training set if 000463.npy doesn't exist
    if not os.path.exists(lr_path):
        available_files = [f for f in os.listdir(lr_dir) if f.endswith('.npy')]
        if not available_files:
            raise FileNotFoundError(f"No .npy files found in {lr_dir}")
        filename = available_files[0]
        lr_path = os.path.join(lr_dir, filename)
        gt_path = os.path.join(gt_dir, filename)

    print(f"Testing image: {lr_path}")

    # 3. Preprocess input image
    lr_np = np.load(lr_path).astype(np.float32)
    lr_np = np.clip(lr_np, 0.0, 1.0)

    lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 4. Inference
    with torch.no_grad():
        with torch.amp.autocast('cuda') if DEVICE == "cuda" else torch.no_grad():
            output_tensor = model(lr_tensor)

    # 5. Postprocessing
    output_np = output_tensor.squeeze().cpu().numpy()

    input_img_scaled = (lr_np * 255.0).astype(np.uint8)
    output_img_scaled = (output_np * 255.0).astype(np.uint8)

    print(f"Input Shape:  {lr_np.shape}   (128x128 expected)")
    print(f"Output Shape: {output_np.shape} (256x256 expected)")

    # 6. Save visualization PNGs
    Image.fromarray(input_img_scaled).save("test_input_noisy.png")
    Image.fromarray(output_img_scaled).save("test_output_denoised_2x.png")

    if os.path.exists(gt_path):
        gt_np = np.clip(np.load(gt_path).astype(np.float32), 0.0, 1.0)
        gt_img_scaled = (gt_np * 255.0).astype(np.uint8)
        Image.fromarray(gt_img_scaled).save("test_ground_truth.png")
        print("Ground truth saved to 'test_ground_truth.png'")

    print("\nInference complete! Check 'test_output_denoised_2x.png' to view results.")

if __name__ == "__main__":
    test_single_image()