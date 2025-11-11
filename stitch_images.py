import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

def find_custom_folders(dataset_dir):
    """Find all folders with names like 'name_hazy' or 'results_*' (excluding base GT and hazy)"""
    custom_folders = []
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            if item.endswith('_hazy') or item.startswith('results_'):
                if item != 'hazy':
                    custom_folders.append(item)
    return sorted(custom_folders)

def find_gt_hazy_folders(dataset_dir):
    """Find GT and hazy base folders"""
    gt_folder = None
    hazy_folder = None
    
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path):
            if item == 'GT':
                gt_folder = item
            elif item == 'hazy':
                hazy_folder = item
    
    return gt_folder, hazy_folder

def load_image(image_path):
    """Load image and convert BGR to RGB"""
    img = cv2.imread(image_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None

def find_image_file(folder_path, filename, folder_name):
    """Find image file with folder-specific naming pattern"""
    extensions = ['.png', '.jpg', '.jpeg']
    
    # Extract the suffix from folder name (e.g., 'GT' from 'GT', 'hazy' from 'hazy', '1' from 'results_1')
    if folder_name == 'GT':
        suffix = '_GT'
    else:
        suffix = '_hazy'
    
    for ext in extensions:
        full_path = os.path.join(folder_path, filename + suffix + ext)
        if os.path.exists(full_path):
            print("Found image file:", full_path)
            return full_path
        else:
            print("Image file not found:", full_path)
    
    return None

def create_grid(dataset_dir, filenames, custom_folders=None, save_dir=None):
    """
    Create a grid of images from specified filenames with improved visualization
    """
    gt_folder, hazy_folder = find_gt_hazy_folders(dataset_dir)
    
    if not gt_folder or not hazy_folder:
        print("Error: Could not find GT or hazy folders")
        return
    
    if custom_folders is None:
        custom_folders = find_custom_folders(dataset_dir)
    
    folders = [hazy_folder] + custom_folders + [gt_folder]
    
    # Create column labels
    column_labels = ['Hazy Input']
    for i, folder in enumerate(custom_folders, 1):
        column_labels.append(f'Method {chr(64 + i)}')
    column_labels.append('Reference (GT)')
    
    # Create batches of 3 filenames
    for batch_idx in range(0, len(filenames), 3):
        batch_filenames = filenames[batch_idx:batch_idx + 3]
        num_cols = len(folders)
        num_rows = len(batch_filenames)
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        fig.patch.set_facecolor('white')
        
        # Add column labels at the top
        for col, label in enumerate(column_labels):
            if num_rows == 1:
                ax = axes[col]
            else:
                ax = axes[0, col]
            ax.text(0.5, 1.25, label, ha='center', va='bottom', fontsize=16, fontweight='bold', 
                   transform=ax.transAxes, color='#2c3e50')
        
        for row, filename in enumerate(batch_filenames):
            for col, folder in enumerate(folders):
                folder_path = os.path.join(dataset_dir, folder)
                image_path = find_image_file(folder_path, filename, folder)
                
                if num_rows == 1:
                    ax = axes[col]
                else:
                    ax = axes[row, col]
                
                if image_path:
                    img = load_image(image_path)
                    if img is not None:
                        ax.imshow(img)
                    else:
                        ax.text(0.5, 0.5, '✗ Image not found', ha='center', va='center', 
                               fontsize=12, color='#e74c3c', fontweight='bold')
                else:
                    ax.text(0.5, 0.5, '✗ Image not found', ha='center', va='center', 
                           fontsize=12, color='#e74c3c', fontweight='bold')
                
                # Remove axis ticks and labels
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            batch_num = (batch_idx // 3) + 1
            save_path = os.path.join(save_dir, f'grid_batch_{batch_num}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"Saved: {save_path}")
            plt.close(fig)
        else:
            plt.show()

if __name__ == "__main__":
    dataset_dir = "Dense_Haze_NTIRE19"
    save_dir = "output_grids"  # Set to None to display instead of saving
    
    filenames = [
        '01',
        '02',
        '04',
        '06',
        '07',
        '10',
        '11',
        '12',
        '13',
        '20',
        '22',
        '26',
        '28',
        '30',
        '32',
        '34',
        '35',
        '36',
        '37',
        '38',
        '39',
        '40',
        '41',
        '42',
        '45',
        '47',
        '49'
    ]
    
    if os.path.exists(dataset_dir):
        print(f"Dataset directory: {dataset_dir}")
        gt_folder, hazy_folder = find_gt_hazy_folders(dataset_dir)
        custom_folders = find_custom_folders(dataset_dir)
        print(f"GT folder: {gt_folder}")
        print(f"Hazy folder: {hazy_folder}")
        print(f"Found custom folders: {custom_folders}")
        
        if filenames:
            create_grid(dataset_dir, filenames, save_dir=save_dir)
    else:
        print(f"Dataset directory not found: {dataset_dir}")