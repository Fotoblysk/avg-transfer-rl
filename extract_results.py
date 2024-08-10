
import os
import tarfile
import shutil
from concurrent.futures import ThreadPoolExecutor

# Define the source and destination directories
source_dir = 'cloud_results/0-0-0'
destination_dir = 'extracted_results'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

def extract_tarball(file_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(destination_dir)
    print(f"Extracted {file_path}")

# Collect all tar.gz files
tar_files = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.tar.gz'):
            tar_files.append(os.path.join(root, file))

# Use ThreadPoolExecutor to extract files in parallel
with ThreadPoolExecutor() as executor:
    executor.map(extract_tarball, tar_files)

print(f"All files have been extracted to {destination_dir}")