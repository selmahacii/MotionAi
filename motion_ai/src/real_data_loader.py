"""
Real Dataset Loader for Human Motion Intelligence System.
Supports COCO Keypoints, Human3.6M, and other real datasets.
"""

import os
import json
import gzip
import urllib.request
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

import sys
import os, sys; sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import NUM_KEYPOINTS, MOVEMENT_CLASSES, NUM_CLASSES, DataConfig


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    url: str
    size_mb: int
    num_samples: int
    format: str
    keypoints: int


# Real dataset configurations
DATASETS = {
    "coco_2017": DatasetInfo(
        name="COCO 2017 Keypoints",
        url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        size_mb=241,
        num_samples=149813,
        format="json",
        keypoints=17
    ),
    "lsp": DatasetInfo(
        name="Leeds Sports Pose",
        url="http://sam.johnson.io/research/lsp_dataset.zip",
        size_mb=200,
        num_samples=2000,
        format="jpg",
        keypoints=14
    ),
    "mpii": DatasetInfo(
        name="MPII Human Pose",
        url="http://human-pose.mpi-inf.mpg.de/results/mpii_human_pose_v1.tar.gz",
        size_mb=12600,
        num_samples=24920,
        format="mat",
        keypoints=16
    ),
    "human36m_sample": DatasetInfo(
        name="Human3.6M Sample",
        url="",  # Local processing required
        size_mb=0,
        num_samples=10000,
        format="npz",
        keypoints=17
    ),
    "amass_sample": DatasetInfo(
        name="AMASS Motion Sample",
        url="",  # Local processing required
        size_mb=0,
        num_samples=50000,
        format="npz",
        keypoints=17
    )
}


class DatasetDownloader:
    """Download and manage real datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url: str, filename: str, desc: str = None) -> Path:
        """Download a file with progress bar."""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"✓ {filename} already exists")
            return filepath
        
        print(f"Downloading {desc or filename}...")
        
        class ProgressTracker:
            def __init__(self):
                self.pbar = None
                self.downloaded = 0
                
            def hook(self, count, block_size, total_size):
                if self.pbar is None:
                    self.pbar = tqdm(total=total_size // (1024*1024), unit='MB')
                new_downloaded = count * block_size // (1024*1024)
                if new_downloaded > self.downloaded:
                    self.pbar.update(new_downloaded - self.downloaded)
                    self.downloaded = new_downloaded
        
        tracker = ProgressTracker()
        urllib.request.urlretrieve(url, filepath, reporthook=tracker.hook)
        if tracker.pbar:
            tracker.pbar.close()
        
        print(f"✓ Downloaded to {filepath}")
        return filepath
    
    def extract_archive(self, archive_path: Path, target_dir: str = None) -> Path:
        """Extract tar/zip archive."""
        if target_dir is None:
            target_dir = archive_path.stem
            
        target_path = self.data_dir / target_dir
        
        if target_path.exists():
            print(f"✓ Already extracted to {target_path}")
            return target_path
        
        print(f"Extracting {archive_path}...")
        
        if archive_path.suffix == '.zip':
            import zipfile
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_path)
        elif archive_path.suffix in ['.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(target_path)
        
        print(f"✓ Extracted to {target_path}")
        return target_path


class COCOKeypointsLoader:
    """
    Load COCO 2017 Keypoint annotations.
    
    COCO provides:
    - 17 keypoints per person
    - Multiple people per image
    - 2D keypoint coordinates with visibility flags
    """
    
    KEYPOINT_NAMES = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    def __init__(self, annotation_path: str, images_dir: str = None):
        self.annotation_path = Path(annotation_path)
        self.images_dir = Path(images_dir) if images_dir else None
        
        self.annotations = None
        self.images = None
        self.categories = None
        
    def load(self):
        """Load COCO annotations from JSON file."""
        print(f"Loading COCO annotations from {self.annotation_path}...")
        
        with open(self.annotation_path, 'r') as f:
            data = json.load(f)
        
        self.annotations = data.get('annotations', [])
        self.images = {img['id']: img for img in data.get('images', [])}
        self.categories = data.get('categories', [])
        
        # Filter to only keypoint annotations
        self.annotations = [
            ann for ann in self.annotations 
            if 'keypoints' in ann and len(ann['keypoints']) == 51  # 17 * 3
        ]
        
        print(f"  Loaded {len(self.annotations)} keypoint annotations")
        print(f"  Images: {len(self.images)}")
        
        return self
    
    def get_samples(self, min_visible_keypoints: int = 5) -> List[Dict]:
        """
        Get all valid samples with sufficient visible keypoints.
        
        Returns:
            List of dicts with:
            - image_id: int
            - image_path: str
            - keypoints: np.ndarray (17, 2)
            - visibility: np.ndarray (17,)
            - bbox: np.ndarray (4,)
        """
        samples = []
        
        for ann in tqdm(self.annotations, desc="Processing annotations"):
            kps = np.array(ann['keypoints']).reshape(-1, 3)
            keypoints = kps[:, :2]
            visibility = kps[:, 2]
            
            # Filter: need enough visible keypoints
            if (visibility > 0).sum() < min_visible_keypoints:
                continue
            
            image_info = self.images.get(ann['image_id'])
            if image_info is None:
                continue
            
            samples.append({
                'image_id': ann['image_id'],
                'image_path': image_info.get('file_name', ''),
                'image_width': image_info.get('width', 640),
                'image_height': image_info.get('height', 480),
                'keypoints': keypoints,
                'visibility': visibility,
                'bbox': np.array(ann.get('bbox', [0, 0, 100, 100]))
            })
        
        print(f"  Valid samples: {len(samples)}")
        return samples
    
    def normalize_keypoints(
        self, 
        keypoints: np.ndarray, 
        bbox: np.ndarray
    ) -> np.ndarray:
        """Normalize keypoints to [0, 1] range based on bounding box."""
        x, y, w, h = bbox
        normalized = keypoints.copy()
        normalized[:, 0] = (keypoints[:, 0] - x) / (w + 1e-6)
        normalized[:, 1] = (keypoints[:, 1] - y) / (h + 1e-6)
        return normalized
    
    def create_training_data(
        self, 
        output_dir: str,
        target_size: int = 256,
        train_split: float = 0.8
    ) -> Dict[str, np.ndarray]:
        """
        Create training-ready numpy arrays.
        
        Saves:
        - images.npy: (N, 3, H, W) normalized images
        - keypoints.npy: (N, 17, 2) normalized coordinates
        - visibility.npy: (N, 17) visibility flags
        """
        samples = self.get_samples(min_visible_keypoints=5)
        
        all_images = []
        all_keypoints = []
        all_visibility = []
        
        if self.images_dir is None:
            print("No images directory provided, using keypoint-only mode")
            # Just save keypoints
            for sample in tqdm(samples, desc="Processing keypoints"):
                # Normalize keypoints to [0, 1]
                normalized_kps = self.normalize_keypoints(
                    sample['keypoints'], sample['bbox']
                )
                all_keypoints.append(normalized_kps)
                all_visibility.append(sample['visibility'])
            
            keypoints = np.array(all_keypoints)
            visibility = np.array(all_visibility)
            
            # Save
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            np.save(output_path / 'keypoints.npy', keypoints)
            np.save(output_path / 'visibility.npy', visibility)
            
            # Split
            n_train = int(len(keypoints) * train_split)
            
            return {
                'train_keypoints': keypoints[:n_train],
                'train_visibility': visibility[:n_train],
                'val_keypoints': keypoints[n_train:],
                'val_visibility': visibility[n_train:],
                'n_samples': len(keypoints)
            }
        
        # Full image processing (if images available)
        from PIL import Image
        
        for sample in tqdm(samples, desc="Processing images"):
            img_path = self.images_dir / sample['image_path']
            if not img_path.exists():
                continue
            
            try:
                # Load and crop to bbox
                img = Image.open(img_path).convert('RGB')
                x, y, w, h = sample['bbox']
                img = img.crop((x, y, x + w, y + h))
                img = img.resize((target_size, target_size))
                
                # Convert to array
                img_array = np.array(img).transpose(2, 0, 1) / 255.0
                
                # Normalize keypoints
                normalized_kps = sample['keypoints'].copy()
                normalized_kps[:, 0] = normalized_kps[:, 0] / (w + 1e-6)
                normalized_kps[:, 1] = normalized_kps[:, 1] / (h + 1e-6)
                
                all_images.append(img_array)
                all_keypoints.append(normalized_kps)
                all_visibility.append(sample['visibility'])
                
            except Exception as e:
                continue
        
        if len(all_images) == 0:
            raise ValueError("No valid images found!")
        
        images = np.array(all_images)
        keypoints = np.array(all_keypoints)
        visibility = np.array(all_visibility)
        
        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'images.npy', images)
        np.save(output_path / 'keypoints.npy', keypoints)
        np.save(output_path / 'visibility.npy', visibility)
        
        # Split
        n_train = int(len(images) * train_split)
        
        return {
            'train_images': images[:n_train],
            'train_keypoints': keypoints[:n_train],
            'train_visibility': visibility[:n_train],
            'val_images': images[n_train:],
            'val_keypoints': keypoints[n_train:],
            'val_visibility': visibility[n_train:],
            'n_samples': len(images)
        }


class Human36MLoader:
    """
    Load Human3.6M dataset for motion prediction.
    
    Human3.6M provides:
    - 3D joint positions
    - 17 joints
    - Multiple actors and actions
    - ~3.6 million frames
    """
    
    ACTION_LABELS = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
        'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
        'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether'
    ]
    
    JOINT_NAMES = [
        'Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
        'Spine', 'Thorax', 'Neck', 'Head', 'LShoulder', 'LElbow',
        'LWrist', 'RShoulder', 'RElbow', 'RWrist'
    ]
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
        
    def load_raw_data(self, subject: str) -> Dict[str, np.ndarray]:
        """Load raw 3D poses for a subject."""
        # Check for preprocessed data
        processed_path = self.data_dir / f"{subject}_poses.npz"
        
        if processed_path.exists():
            data = np.load(processed_path, allow_pickle=True)
            return dict(data)
        
        # Otherwise, look for original data
        # This requires the original Human3.6M dataset
        print(f"No preprocessed data found at {processed_path}")
        print("Please run preprocessing script on original Human3.6M data")
        
        return {}
    
    def create_motion_sequences(
        self,
        seq_length: int = 30,
        future_length: int = 10,
        train_subjects: List[str] = ['S1', 'S5', 'S6', 'S7', 'S8'],
        test_subjects: List[str] = ['S9', 'S11']
    ) -> Dict[str, np.ndarray]:
        """
        Create sequences for motion prediction.
        
        Returns:
            train_x: (N, seq_length, 17, 3) - past 3D poses
            train_y: (N, future_length, 17, 3) - future 3D poses
            test_x, test_y: same for test set
        """
        train_sequences_x = []
        train_sequences_y = []
        test_sequences_x = []
        test_sequences_y = []
        
        for subject in train_subjects:
            data = self.load_raw_data(subject)
            if not data:
                continue
                
            for action in self.ACTION_LABELS:
                key = f"{action}"
                if key not in data:
                    continue
                    
                poses = data[key]  # (T, 17, 3)
                
                # Create sequences
                for i in range(len(poses) - seq_length - future_length):
                    train_sequences_x.append(poses[i:i+seq_length])
                    train_sequences_y.append(poses[i+seq_length:i+seq_length+future_length])
        
        for subject in test_subjects:
            data = self.load_raw_data(subject)
            if not data:
                continue
                
            for action in self.ACTION_LABELS:
                key = f"{action}"
                if key not in data:
                    continue
                    
                poses = data[key]
                
                for i in range(len(poses) - seq_length - future_length):
                    test_sequences_x.append(poses[i:i+seq_length])
                    test_sequences_y.append(poses[i+seq_length:i+seq_length+future_length])
        
        if len(train_sequences_x) == 0:
            raise ValueError("No sequences created. Check data directory.")
        
        return {
            'train_x': np.array(train_sequences_x),
            'train_y': np.array(train_sequences_y),
            'test_x': np.array(test_sequences_x),
            'test_y': np.array(test_sequences_y)
        }


class AMASSLoader:
    """
    Load AMASS (Archive of Motion Capture as Surface Shapes) dataset.
    
    AMASS provides:
    - 15+ hours of motion data
    - SMPL body model parameters
    - Can be converted to 17 COCO keypoints
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_subset(self, datasets: List[str] = None) -> Dict[str, np.ndarray]:
        """Load a subset of AMASS data."""
        # AMASS is very large, so we load preprocessed subsets
        subset_path = self.data_dir / 'amass_subset.npz'
        
        if subset_path.exists():
            data = np.load(subset_path, allow_pickle=True)
            return dict(data)
        
        print(f"No AMASS data found at {subset_path}")
        print("Download from https://amass.is.tue.mpg.de/")
        
        return {}
    
    def create_classification_data(
        self,
        seq_length: int = 30,
        action_labels: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """Create sequences for action classification."""
        data = self.load_subset()
        
        if not data:
            return {}
        
        sequences = []
        labels = []
        
        # Process each sequence
        for key, poses in data.items():
            if len(poses) < seq_length:
                continue
            
            # Determine action label from key name
            label = 0  # Default
            for i, action in enumerate(action_labels or MOVEMENT_CLASSES):
                if action.lower() in key.lower():
                    label = i
                    break
            
            # Create overlapping sequences
            for i in range(0, len(poses) - seq_length, seq_length // 2):
                sequences.append(poses[i:i+seq_length])
                labels.append(label)
        
        return {
            'sequences': np.array(sequences),
            'labels': np.array(labels)
        }


def download_coco_annotations(data_dir: str = "data") -> Tuple[Path, Path]:
    """Download COCO 2017 keypoint annotations."""
    downloader = DatasetDownloader(data_dir)
    
    # Download annotations
    ann_file = downloader.download_file(
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations_trainval2017.zip",
        "COCO 2017 Annotations"
    )
    
    # Extract
    extracted = downloader.extract_archive(ann_file)
    
    train_ann = extracted / "annotations" / "person_keypoints_train2017.json"
    val_ann = extracted / "annotations" / "person_keypoints_val2017.json"
    
    return train_ann, val_ann


def prepare_real_data(
    data_dir: str = "data",
    output_dir: str = "data/processed",
    dataset: str = "coco"
) -> Dict[str, Any]:
    """
    Prepare real dataset for training.
    
    This is the main entry point for data preparation.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset == "coco":
        # Download annotations if needed
        train_ann, val_ann = download_coco_annotations(data_dir)
        
        # Load and process
        print("\nProcessing COCO training set...")
        train_loader = COCOKeypointsLoader(str(train_ann))
        train_loader.load()
        train_data = train_loader.create_training_data(
            output_dir / "coco_train",
            train_split=1.0  # Already split by COCO
        )
        
        print("\nProcessing COCO validation set...")
        val_loader = COCOKeypointsLoader(str(val_ann))
        val_loader.load()
        val_data = val_loader.create_training_data(
            output_dir / "coco_val",
            train_split=1.0
        )
        
        return {
            'train': train_data,
            'val': val_data,
            'num_keypoints': 17,
            'dataset': 'coco'
        }
    
    elif dataset == "human36m":
        loader = Human36MLoader(data_dir / "human36m")
        data = loader.create_motion_sequences()
        
        # Save processed data
        np.savez(
            output_dir / "human36m_sequences.npz",
            **data
        )
        
        return {
            **data,
            'num_keypoints': 17,
            'dataset': 'human36m'
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare real datasets")
    parser.add_argument("--dataset", type=str, default="coco", 
                        choices=["coco", "human36m", "amass"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    
    args = parser.parse_args()
    
    data = prepare_real_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset=args.dataset
    )
    
    print(f"\nData prepared successfully!")
    print(f"Dataset: {data.get('dataset')}")
    print(f"Keypoints: {data.get('num_keypoints')}")
