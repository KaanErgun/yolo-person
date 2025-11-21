"""
COCO to YOLO Converter - Person Class Only
Converts COCO format annotations to YOLO format, filtering only 'person' class
"""
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_coco_to_yolo(coco_json_path, images_dir, output_dir, class_filter='person'):
    """
    Convert COCO annotations to YOLO format for a specific class
    
    Args:
        coco_json_path: Path to COCO JSON annotation file
        images_dir: Directory containing COCO images
        output_dir: Output directory for YOLO format dataset
        class_filter: Class name to filter (default: 'person')
    """
    print(f"Loading COCO annotations from {coco_json_path}...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Find person category ID
    person_cat_id = None
    for cat in coco_data['categories']:
        if cat['name'] == class_filter:
            person_cat_id = cat['id']
            print(f"Found '{class_filter}' category with ID: {person_cat_id}")
            break
    
    if person_cat_id is None:
        raise ValueError(f"Category '{class_filter}' not found in COCO dataset")
    
    # Create output directories
    output_images_dir = Path(output_dir) / 'images'
    output_labels_dir = Path(output_dir) / 'labels'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Group annotations by image
    print("Processing annotations...")
    image_annotations = {}
    for ann in tqdm(coco_data['annotations'], desc="Grouping annotations"):
        if ann['category_id'] == person_cat_id:
            image_id = ann['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(ann)
    
    # Create image_id to image_info mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    print(f"Found {len(image_annotations)} images with '{class_filter}' annotations")
    
    # Convert and copy files
    copied_images = 0
    created_labels = 0
    
    for image_id, annotations in tqdm(image_annotations.items(), desc="Converting to YOLO format"):
        if image_id not in image_info:
            continue
        
        img_info = image_info[image_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Source image path
        src_image_path = Path(images_dir) / img_filename
        if not src_image_path.exists():
            continue
        
        # Destination image path
        dst_image_path = output_images_dir / img_filename
        
        # Copy image
        shutil.copy2(src_image_path, dst_image_path)
        copied_images += 1
        
        # Create YOLO label file
        label_filename = Path(img_filename).stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # COCO bbox format: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format: [class, x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height
                
                # Class ID is always 0 for person (since we have only one class)
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        created_labels += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"   - Images copied: {copied_images}")
    print(f"   - Labels created: {created_labels}")
    print(f"   - Output directory: {output_dir}")
    
    return copied_images, created_labels


def process_coco_dataset(coco_dir='./data/coco', output_dir='./datasets/coco_person'):
    """
    Process full COCO dataset (train and val) and convert to YOLO format
    
    Args:
        coco_dir: Directory containing COCO dataset
        output_dir: Output directory for YOLO format dataset
    """
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    
    # Process train set
    print("\n" + "="*60)
    print("Processing TRAIN set")
    print("="*60)
    train_json = coco_dir / 'annotations' / 'instances_train2017.json'
    train_images = coco_dir / 'train2017'
    train_output = output_dir / 'train_temp'
    
    if train_json.exists():
        convert_coco_to_yolo(
            str(train_json),
            str(train_images),
            str(train_output),
            class_filter='person'
        )
    else:
        print(f"⚠️  Train annotations not found at {train_json}")
    
    # Process val set
    print("\n" + "="*60)
    print("Processing VAL set")
    print("="*60)
    val_json = coco_dir / 'annotations' / 'instances_val2017.json'
    val_images = coco_dir / 'val2017'
    val_output = output_dir / 'val_temp'
    
    if val_json.exists():
        convert_coco_to_yolo(
            str(val_json),
            str(val_images),
            str(val_output),
            class_filter='person'
        )
    else:
        print(f"⚠️  Val annotations not found at {val_json}")
    
    # Move to correct YOLO structure
    print("\n" + "="*60)
    print("Organizing dataset structure...")
    print("="*60)
    
    # Create final directories
    final_train_img = output_dir / 'images' / 'train'
    final_train_lbl = output_dir / 'labels' / 'train'
    final_val_img = output_dir / 'images' / 'val'
    final_val_lbl = output_dir / 'labels' / 'val'
    
    final_train_img.mkdir(parents=True, exist_ok=True)
    final_train_lbl.mkdir(parents=True, exist_ok=True)
    final_val_img.mkdir(parents=True, exist_ok=True)
    final_val_lbl.mkdir(parents=True, exist_ok=True)
    
    # Move train files
    if train_output.exists():
        for img_file in (train_output / 'images').glob('*'):
            shutil.move(str(img_file), str(final_train_img / img_file.name))
        for lbl_file in (train_output / 'labels').glob('*'):
            shutil.move(str(lbl_file), str(final_train_lbl / lbl_file.name))
        shutil.rmtree(train_output)
        print(f"✓ Train files moved to {final_train_img.parent.parent}")
    
    # Move val files
    if val_output.exists():
        for img_file in (val_output / 'images').glob('*'):
            shutil.move(str(img_file), str(final_val_img / img_file.name))
        for lbl_file in (val_output / 'labels').glob('*'):
            shutil.move(str(lbl_file), str(final_val_lbl / lbl_file.name))
        shutil.rmtree(val_output)
        print(f"✓ Val files moved to {final_val_img.parent.parent}")
    
    print("\n" + "="*60)
    print("✅ Dataset preparation complete!")
    print("="*60)
    print(f"Dataset ready at: {output_dir.absolute()}")
    print("\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/")
    print(f"    │   └── val/")
    print(f"    └── labels/")
    print(f"        ├── train/")
    print(f"        └── val/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert COCO to YOLO format (Person class only)')
    parser.add_argument('--coco-dir', type=str, default='./data/coco',
                        help='Directory containing COCO dataset (default: ./data/coco)')
    parser.add_argument('--output-dir', type=str, default='./datasets/coco_person',
                        help='Output directory for YOLO dataset (default: ./datasets/coco_person)')
    
    args = parser.parse_args()
    
    process_coco_dataset(
        coco_dir=args.coco_dir,
        output_dir=args.output_dir
    )
