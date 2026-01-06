import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTConfig, ViTModel
import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
# Dhyan de: Agar tera data 'data' folder me hai to path change karna padega
# Agar sab ek hi folder me hai to bas "mars_data.txt" rakh
DATA_FILE = "data/mars_data.txt"  
IMG_SIZE = 64                
PATCH_SIZE = 16
EPOCHS = 200                 

print("🚀 STARTING FINAL MARS AI PIPELINE...")

# --- 1. REAL DATA LOADER ---
def load_real_mars_data(filename):
    # Check if file exists
    if not os.path.exists(filename):
        # Fallback: Agar data folder me nahi mila, to current folder me check karo
        if os.path.exists("mars_data.txt"):
            filename = "mars_data.txt"
        else:
            print(f"❌ Error: '{filename}' nahi mili! Please check path.")
            exit()

    print(f"📂 Loading Real Data from {filename}...")
    temps = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # Parsing logic for ASCII format
            if "||" in line and not line.startswith("#") and not line.startswith("----"):
                parts = line.split("||")
                temps.append([float(x) for x in parts[1].split()])
        
        # Convert to numpy
        data = np.array(temps)
        
        # Normalize (0 to 1) - Critical for Neural Networks
        d_min, d_max = np.min(data), np.max(data)
        data_norm = (data - d_min) / (d_max - d_min)
        
        # Convert to Tensor (Add Batch & Channel dims) -> (1, 1, H, W)
        tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Resize to 64x64 (Kyuki tera data 64x48 tha, aur ViT ko square pasand hai)
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
        self.config = ViTConfig(
            image_size=IMG_SIZE, patch_size=PATCH_SIZE, num_channels=1, 
            hidden_size=256, num_hidden_layers=4, num_attention_heads=4
        )
        self.encoder = ViTModel(self.config)
        # Decoder: Latent features (256) -> Pixel Patch (16*16 = 256)
        self.decoder_head = nn.Linear(256, PATCH_SIZE*PATCH_SIZE) 

    def forward(self, x):
        outputs = self.encoder(x)
        # Skip [CLS] token (pehla token classification ke liye hota hai)
        features = outputs.last_hidden_state[:, 1:, :] 
        reconstructed = self.decoder_head(features)
        return reconstructed

# --- 3. TRAINING SETUP ---
real_mars_image = load_real_mars_data(DATA_FILE)
model = MarsReconstructor()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# --- 4. TRAINING LOOP (Learning from REAL Data) ---
print(f"⚡ Training on Real Mars Telemetry for {EPOCHS} epochs...")
history = []

for epoch in range(EPOCHS):
    # Forward Pass
    reconstructed_patches = model(real_mars_image)
    
    # Create Ground Truth (Image -> Patches)
    target_patches = real_mars_image.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
    target_patches = target_patches.contiguous().view(1, (IMG_SIZE//PATCH_SIZE)**2, -1)
    
    # Loss Calculation
    loss = criterion(reconstructed_patches, target_patches)
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    history.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"   Epoch {epoch} | Loss: {loss.item():.5f}")

print("✅ Training Complete!")

# --- 5. FINAL VISUALIZATION & SAVE ---
# Create results folder if not exists
if not os.path.exists("results"):
    os.makedirs("results")

# Reconstruct image from patches for display
rec_patches = reconstructed_patches.detach().view(1, 4, 4, 16, 16)
rec_image = rec_patches.permute(0, 1, 3, 2, 4).contiguous().view(1, 1, 64, 64)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(real_mars_image[0, 0], cmap='inferno')
axs[0].set_title("1. Real Mars Data (Input)")

axs[1].plot(history, color='red')
axs[1].set_title(f"2. Convergence (Final Loss: {history[-1]:.4f})")

axs[2].imshow(rec_image[0, 0], cmap='inferno')
axs[2].set_title("3. AI Reconstruction (Output)")

save_path = "results/final_result.png"
plt.savefig(save_path) 
print(f"💾 Result saved to {save_path}")
plt.show()

