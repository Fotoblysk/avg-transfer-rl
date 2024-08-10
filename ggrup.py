import os
import tarfile
import tempfile

import torch
import re


def extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_to)


def process_model(model_path, dest_path):
    # Load the model
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))

        # Keep only the state['dqn']
        state_dqn = model['dqn']

        # Save the modified model
        torch.save({'dqn': state_dqn}, dest_path)
    except Exception as e:
        print(f"{model_path}: extraction failed")
        print(e)
        print("Trying emergency extraction")
        try:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            state_dqn = model['dqn_target']
            torch.save({'dqn': state_dqn}, dest_path)
        except Exception as e:
            print(f"{model_path}: FINAL EXTRACTION FAILED")
            print(e)


def process_extracted_results(src_dir, dest_dir):
    # Ensure the destination directories exist
    lunar_lander_dir = os.path.join(dest_dir, 'lunar_lander')
    minigrid_dir = os.path.join(dest_dir, 'minigrid')
    frozen_lake_dir = os.path.join(dest_dir, 'frozen_lake')

    os.makedirs(lunar_lander_dir, exist_ok=True)
    os.makedirs(minigrid_dir, exist_ok=True)
    os.makedirs(frozen_lake_dir, exist_ok=True)

    # Regex patterns to identify directories
    lunar_lander_pattern = re.compile(r'lunar_lander')
    minigrid_pattern = re.compile(r'MiniGrid')
    frozen_lake_pattern = re.compile(r'FrozenLake')

    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.tar.gz'):
                tar_path = os.path.join(root, file)

                # Create a temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_extract_dir:
                    # Extract the tar.gz file
                    extract_tar(tar_path, temp_extract_dir)

                    # Find the extracted directory
                    extracted_dir = os.path.join(temp_extract_dir, list(sorted(os.listdir(temp_extract_dir)))[0])

                    # Process the model file
                    model_path = os.path.join(extracted_dir, 'model')
                    if os.path.exists(model_path):
                        # Determine the destination directory based on the model name
                        if lunar_lander_pattern.search(extracted_dir):
                            dest_subdir = lunar_lander_dir
                        elif minigrid_pattern.search(extracted_dir):
                            dest_subdir = minigrid_dir
                        elif frozen_lake_pattern.search(extracted_dir):
                            dest_subdir = frozen_lake_dir
                        else:
                            continue  # Skip if the directory doesn't match any pattern

                        # Extract the model name without timestamp
                        model_name = os.path.basename(extracted_dir)
                        model_name = re.sub(r'_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}', '', model_name)

                        # Destination path for the modified model
                        dest_model_path = os.path.join(dest_subdir, f"{model_name}.pt")

                        # Process and save the modified model
                        process_model(model_path, dest_model_path)


if __name__ == "__main__":
    src_dir = 'extracted_results'
    dest_dir = 'sorted_models'
    process_extracted_results(src_dir, dest_dir)
