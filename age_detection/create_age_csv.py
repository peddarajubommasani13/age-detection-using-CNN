import os
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

# ---------------------------
# Setup images folder
# ---------------------------
img_folder = "images"
os.makedirs(img_folder, exist_ok=True)

# ---------------------------
# Create dummy images if folder is empty
# ---------------------------
if not os.listdir(img_folder):
    print("Images folder is empty. Creating dummy images...")
    dummy_ages = [18, 25, 32, 45, 60]
    for i, age in enumerate(dummy_ages):
        img = Image.new('L', (100, 100), color=128)
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), str(age), fill=255)
        img.save(os.path.join(img_folder, f"{age}_face{i+1}.jpg"))
    print("Dummy images created!")

# ---------------------------
# Generate CSV
# ---------------------------
data = []

for img_file in os.listdir(img_folder):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        age = int(img_file.split('_')[0])  # filename starts with age
        img_path = os.path.join(img_folder, img_file)
        img = Image.open(img_path).convert('L').resize((48, 48))
        pixels = " ".join(map(str, np.array(img).flatten()))
        data.append([age, pixels])

df = pd.DataFrame(data, columns=["age", "pixels"])
df.to_csv("age_data.csv", index=False)
print("CSV created successfully!")
