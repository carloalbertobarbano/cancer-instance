import gdown
import os
import tarfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default=f'{os.path.expanduser("~")}/data/')
parser.add_argument('--dst', type=str, default=f'{os.path.expanduser("~")}/data/')
args = parser.parse_args()

dataroot = args.root
dataset_folder = args.dst

# Fetch data from Google Drive
# URL for the dataset
url = 'https://drive.google.com/uc?id=1a42HQ4g9XrmuJ5iwQKQgzqi8Fap-lq29'
# Path to download the dataset to
download_path = f'{dataroot}/cancer_instance.tar.xz'

# Create required directories 
if not os.path.exists(dataroot):
  os.makedirs(dataroot)

if not os.path.exists(dataset_folder):
  os.makedirs(dataset_folder)

# Download the dataset from google drive
gdown.download(url, download_path, quiet=False)
print('Downloaded!')

print('Unzipping...')
with tarfile.open(download_path) as f:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(f, dataset_folder)

print('Done!')