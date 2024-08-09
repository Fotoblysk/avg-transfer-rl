
import os
import tarfile
import tempfile
import torch
import re
import shutil

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

def process_csv(extracted_dir, dest_dir):
    stats_dir = os.path.join(extracted_dir, 'stats')
    csv_file = os.path.join(stats_dir, 'train_ep_data.csv')
    if os.path.exists(csv_file):
        # Extract the model name without timestamp
        model_name = os.path.basename(extracted_dir)
        model_name = re.sub(r'_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}', '', model_name)

        # Destination path for the CSV file
        dest_csv_path = os.path.join(dest_dir, f"{model_name}.csv")

        # Copy the CSV file to the destination
        shutil.copy(csv_file, dest_csv_path)

def process_extracted_results(src_dir, ):
    # Ensure the destination directories exist
    csv_results_dir = os.path.join('csv_results')

    os.makedirs(csv_results_dir, exist_ok=True)

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
                    extracted_dir = os.path.join(temp_extract_dir, os.listdir(temp_extract_dir)[0])

                    # Process the model file
                    # Process the CSV file if it's a MiniGrid directory
                    if minigrid_pattern.search(extracted_dir):
                        process_csv(extracted_dir, csv_results_dir)
                    if lunar_lander_pattern.search(extracted_dir):
                        process_csv(extracted_dir, csv_results_dir)
                    if frozen_lake_pattern.search(extracted_dir):
                        process_csv(extracted_dir, csv_results_dir)

if __name__ == "__main__":
    src_dir = ['extracted_results',
    'extracted_results_back',
    'extracted_results_bak2']
    for i in src_dir:
        process_extracted_results(i)