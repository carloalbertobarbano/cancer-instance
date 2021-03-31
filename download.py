import gdown
import os
import tarfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default=f'{os.path.expanduser("~")}/data/')
parser.add_argument('--dst', type=str, default=f'{os.path.expanduser("~")}/data/cancer-instance')
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
    f.extractall(dataset_folder)

print('Done!')