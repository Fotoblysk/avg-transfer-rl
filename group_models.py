import os
import tarfile
import shutil
#from concurrent.futures import ProcessPoolExecutor
import tempfile


def extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_to)


def compress_dir(dir_path, compressed_path):
    with tarfile.open(compressed_path, 'w:gz') as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))
    shutil.rmtree(dir_path)


def process_tar_file(tar_path, dest_dir):
    # Create a unique temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_extract_dir:
        # Extract the tar.gz file
        try:
            extract_tar(tar_path, temp_extract_dir)

            # Compress each directory inside the extracted 'results' directory
            results_path = os.path.join(temp_extract_dir, 'results')
            for item in os.listdir(results_path):
                item_path = os.path.join(results_path, item)
                if os.path.isdir(item_path):
                    compressed_path = os.path.join(dest_dir, f"{item}.tar.gz")
                    compress_dir(item_path, compressed_path)
        except:
            print("Bad archive")


def extract_and_compress(src_dir, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Collect all tar.gz files
    tar_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file == 'results.tar.gz':
                tar_files.append(os.path.join(root, file))

    # Determine the number of processes to use
    num_processes = 1#os.cpu_count()//2 or 1  # Fallback to 1 if os.cpu_count() returns None

    [process_tar_file(tar_path, dest_dir) for tar_path in tar_files]
    # Process tar.gz files in parallel
    #with ProcessPoolExecutor(max_workers=num_processes) as executor:
    #    futures = [executor.submit(process_tar_file, tar_path, dest_dir) for tar_path in tar_files]
    #    for future in futures:
    #        future.result()  # Wait for all tasks to complete


if __name__ == "__main__":
    src_dir = 'cloud_results/0-0-2'
    dest_dir = 'extracted_results'
    extract_and_compress(src_dir, dest_dir)