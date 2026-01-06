import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTConfig, ViTModel
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Path to the input data file. 
# Ensure this matches your directory structure (e.g., 'data/mars_data.txt')
DATA_FILE = "data/mars_data.txt"  
IMG_SIZE = 64                
PATCH_SIZE = 16
EPOCHS = 200                 

print("🚀 STARTING FINAL MARS AI PIPELINE...")

# --- 1. REAL DATA LOADER ---
def load_real_mars_data(filename):
    """
    Parses ASCII format Mars Climate Database data, normalizes it, 
    and resizes it for the Vision Transformer.
    """
    # Check if file exists in the specified path
    if not os.path.exists(filename):
        # Fallback: Check in the current directory if not found in 'data/'
        if os.path.exists("mars_data.txt"):
            filename = "mars_data.txt"
        else:
            print(f"❌ Error: '{filename}' not found! Please check the file path.")
            exit()

    print(f"📂 Loading Real Data from {filename}...")
    temps = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # Parsing logic for ASCII format provided by MCD
            # Skips headers and parses lines containing data delimeters '||'
            if "||" in line and not line.startswith("#") and not line.startswith("----"):
                parts = line.split("||")
                temps.append([float(x) for x in parts[1].split()])
        
        # Convert list to numpy array
        data = np.array(temps)
        
        # Min-Max Normalization (Scale to 0-1 range)
        # Neural networks require normalized data for stable convergence
        d_min, d_max = np.min(data), np.max(data)
        data_norm = (data - d_min) / (d_max - d_min)
        
        # Convert to PyTorch Tensor and add Batch & Channel dimensions
        # Shape becomes (1, 1, Height, Width)
        tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Resize to 64x64 using Bilinear Interpolation
        # ViT models typically require square inputs for efficient patching
        tensor_resized = torch.nn.functional.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear')
        
        print(f"✅ Data Loaded & Resized: {tensor_resized.shape}")
        return tensor_resized
    
    except Exception as e:
        print(f"❌ Error parsing data: {e}")
        exit()

# --- 2. THE AI MODEL (ViT + MAE Head) ---
class MarsReconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize Vision Transformer Configuration
        self.config = ViTConfig(
            image_size=IMG_SIZE, 
            patch_size=PATCH_SIZE, 
            num_channels=1, 
            hidden_size=256, 
            num_hidden_layers=4, 
            num_attention_heads=4
        )
        self.encoder = ViTModel(self.config)
        
        # Decoder Head: Projects latent features back to pixel space
        # Output dim = patch_size * patch_size (16*16 = 256 pixels)
        self.decoder_head = nn.Linear(256, PATCH_SIZE*PATCH_SIZE) 

    def forward(self, x):
        # Forward pass through Transformer Encoder
        outputs = self.encoder(x)
        
        # Extract features from the last hidden state
        # We skip the [CLS] token (index 0) as we are doing reconstruction, not classification
        features = outputs.last_hidden_state[:, 1:, :] 
        
        # Project features to reconstruct patches
        reconstructed = self.decoder_head(features)
        return reconstructed

# --- 3. TRAINING SETUP ---
real_mars_image = load_real_mars_data(DATA_FILE)
model = MarsReconstructor()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss() # Mean Squared Error Loss for regression

# --- 4. TRAINING LOOP ---
print(f"⚡ Training on Real Mars Telemetry for {EPOCHS} epochs...")
history = []

for epoch in range(EPOCHS):
    # Forward Pass: Predict patches
    reconstructed_patches = model(real_mars_image)
    
    # Create Ground Truth: Unfold original image into patches
    # This aligns the target data structure with the model output
    target_patches = real_mars_image.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    target_patches = target_patches.contiguous().view(1, (IMG_SIZE//PATCH_SIZE)**2, -1)
    
    # Calculate Loss
    loss = criterion(reconstructed_patches, target_patches)
    
    # Backpropagation: Update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    history.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"   Epoch {epoch} | Loss: {loss.item():.5f}")

print("✅ Training Complete!")

# --- 5. VISUALIZATION & SAVE ---
# Create results directory if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Reconstruct image from patches for visualization
# Reshaping logic: (Batch, Patches_H, Patches_W, Patch_H, Patch_W) -> Image
rec_patches = reconstructed_patches.detach().view(1, 4, 4, 16, 16)
rec_image = rec_patches.permute(0, 1, 3, 2, 4).contiguous().view(1, 1, 64, 64)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Input Data
axs[0].imshow(real_mars_image[0, 0], cmap='inferno')
axs[0].set_title("1. Real Mars Data (Input)")
axs[0].axis('off')

# Plot 2: Learning Curve
axs[1].plot(history, color='red')
axs[1].set_title(f"2. Convergence (Final Loss: {history[-1]:.4f})")
axs[1].set_xlabel("Epochs")
axs[1].set_ylabel("MSE Loss")

# Plot 3: Model Output
axs[2].imshow(rec_image[0, 0], cmap='inferno')
axs[2].set_title("3. AI Reconstruction (Output)")
axs[2].axis('off')

# Save result
save_path = "results/final_result.png"
plt.savefig(save_path) 
print(f"💾 Result saved to {save_path}")
plt.show()