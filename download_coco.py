"""
COCO Dataset Downloader
Downloads COCO 2017 train and val datasets (images and annotations)
"""
import os
import requests
import zipfile
from tqdm import tqdm
import argparse


def download_file(url, destination):
    """Download a file with progress bar and resume support"""
    print(f"Downloading {url}...")
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Check if partial file exists
    resume_byte_pos = 0
    if os.path.exists(destination):
        resume_byte_pos = os.path.getsize(destination)
        print(f"Resuming download from byte {resume_byte_pos}")
    
    headers = {}
    if resume_byte_pos:
        headers['Range'] = f'bytes={resume_byte_pos}-'
    
    response = requests.get(url, stream=True, headers=headers, timeout=30)
    total_size = int(response.headers.get('content-length', 0)) + resume_byte_pos
    
    mode = 'ab' if resume_byte_pos else 'wb'
    
    with open(destination, mode) as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        initial=resume_byte_pos,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=8192):
            size = file.write(data)
            progress_bar.update(size)


def extract_zip(zip_path, extract_to):
    """Extract a zip file with validation"""
    print(f"Validating {zip_path}...")
    try:
        # Validate zip file first
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Test the zip file
            bad_file = zip_ref.testzip()
            if bad_file:
                raise zipfile.BadZipFile(f"Corrupted file in zip: {bad_file}")
            
            print(f"Extracting {zip_path}...")
            zip_ref.extractall(extract_to)
        print(f"Extraction complete!")
    except zipfile.BadZipFile as e:
        print(f"❌ Error: {e}")
        print(f"Removing corrupted file: {zip_path}")
        os.remove(zip_path)
        raise Exception("Zip file is corrupted. Please run the script again to re-download.")


def download_coco_dataset(data_dir='./data/coco', download_images=True, download_annotations=True):
    """
    Download COCO 2017 dataset
    
    Args:
        data_dir: Directory to save the dataset
        download_images: Whether to download images
        download_annotations: Whether to download annotations
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # COCO 2017 URLs
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    # Download annotations
    if download_annotations:
        annotations_zip = os.path.join(data_dir, 'annotations_trainval2017.zip')
        if not os.path.exists(annotations_zip):
            download_file(urls['annotations'], annotations_zip)
            extract_zip(annotations_zip, data_dir)
            os.remove(annotations_zip)  # Clean up zip file
        else:
            print(f"Annotations already exist at {annotations_zip}")
    
    # Download images
    if download_images:
        # Train images
        train_zip = os.path.join(data_dir, 'train2017.zip')
        if not os.path.exists(os.path.join(data_dir, 'train2017')):
            if not os.path.exists(train_zip):
                download_file(urls['train_images'], train_zip)
            extract_zip(train_zip, data_dir)
            os.remove(train_zip)  # Clean up zip file
        else:
            print(f"Train images already exist")
        
        # Val images
        val_zip = os.path.join(data_dir, 'val2017.zip')
        if not os.path.exists(os.path.join(data_dir, 'val2017')):
            if not os.path.exists(val_zip):
                download_file(urls['val_images'], val_zip)
            extract_zip(val_zip, data_dir)
            os.remove(val_zip)  # Clean up zip file
        else:
            print(f"Val images already exist")
    
    print("\n✅ COCO dataset download complete!")
    print(f"Dataset location: {os.path.abspath(data_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download COCO 2017 Dataset')
    parser.add_argument('--data-dir', type=str, default='./data/coco',
                        help='Directory to save the dataset (default: ./data/coco)')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip downloading images')
    parser.add_argument('--no-annotations', action='store_true',
                        help='Skip downloading annotations')
    
    args = parser.parse_args()
    
    download_coco_dataset(
        data_dir=args.data_dir,
        download_images=not args.no_images,
        download_annotations=not args.no_annotations
    )
