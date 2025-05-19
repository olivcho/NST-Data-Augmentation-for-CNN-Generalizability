import kagglehub
import os
import shutil

# Create source_images directory if it doesn't exist
os.makedirs("data/style_images", exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("ikarus777/best-artworks-of-all-time")

# Move downloaded files to source_images folder
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join("data/style_images", file)
    shutil.move(src, dst)

print("Files downloaded and moved to style_images folder")