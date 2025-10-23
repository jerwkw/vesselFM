import os
from pathlib import Path

def rename_sample_folders_and_files(root_dir):
    """
    Rename sample folders to UID and standardize file names.
    
    Args:
        root_dir: Path to the root directory containing train/test/val folders
    """
    root_path = Path(root_dir)
    
    # Iterate through train, test, val folders
    for split_folder in ['train', 'test', 'val']:
        split_path = root_path / split_folder
        
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {split_folder} folder...")
        
        # Iterate through each sample folder
        for sample_folder in split_path.iterdir():
            if not sample_folder.is_dir():
                continue
            
            print(f"  Processing sample: {sample_folder.name}")
            
            # Find the base image file (without _cowseg suffix)
            img_file = None
            mask_file = None
            
            for file in sample_folder.iterdir():
                if file.is_file() and file.suffix == '.nii':
                    if '_cowseg' in file.name:
                        mask_file = file
                    else:
                        img_file = file
            
            if img_file is None:
                print(f"    Warning: No base image file found in {sample_folder.name}")
                continue
            
            if mask_file is None:
                print(f"    Warning: No mask file found in {sample_folder.name}")
                continue
            
            # Extract UID from the image filename (remove .nii extension)
            uid = img_file.stem
            
            # Rename files first (before renaming folder)
            new_img_path = sample_folder / "img.nii"
            new_mask_path = sample_folder / "mask.nii"
            
            img_file.rename(new_img_path)
            print(f"    Renamed {img_file.name} -> img.nii")
            
            mask_file.rename(new_mask_path)
            print(f"    Renamed {mask_file.name} -> mask.nii")
            
            # Rename the sample folder to UID
            new_folder_path = split_path / uid
            
            # Check if target folder already exists
            if new_folder_path.exists() and new_folder_path != sample_folder:
                print(f"    Warning: Folder {uid} already exists, skipping folder rename")
            else:
                sample_folder.rename(new_folder_path)
                print(f"    Renamed folder {sample_folder.name} -> {uid}")

if __name__ == "__main__":
    # Set your data splits root directory
    data_root = r"./data_splits"
    
    print(f"Starting renaming process for: {data_root}")
    print("=" * 60)
    
    rename_sample_folders_and_files(data_root)
    
    print("\n" + "=" * 60)
    print("Renaming process completed!")
