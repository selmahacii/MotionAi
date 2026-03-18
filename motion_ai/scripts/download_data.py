"""
Dataset Download and Preparation Scripts.
Download COCO Keypoints and MPII Human Pose datasets for training.
"""

import os
import sys
import json
import zipfile
import argparse
from pathlib import Path
from typing import Optional, Dict, List
from urllib.request import urlretrieve
import time


class DatasetDownloader:
    """Download and prepare pose estimation datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, dest: Path, desc: str = None):
        """Download file with progress bar."""
        if desc:
            print(f"Downloading {desc}...")
        
        def progress_hook(count, block_size, total_size):
            percent = min(100, int(count * block_size * 100 / total_size))
            sys.stdout.write(f"\r  Progress: {percent}%")
            sys.stdout.flush()
        
        urlretrieve(url, dest, progress_hook)
        print()  # Newline after progress

    def extract_zip(self, zip_path: Path, dest: Path):
        """Extract ZIP file."""
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest)
        print("  Done!")

    def download_coco_keypoints(self, subset: str = "val2017"):
        """
        Download COCO Keypoints dataset.
        
        For full training, you need:
        - train2017.zip (~18GB) - Training images
        - val2017.zip (~1GB) - Validation images
        - annotations_trainval2017.zip (~241MB) - Keypoint annotations
        
        For quick testing, val2017 is sufficient.
        """
        coco_dir = self.data_dir / "coco"
        coco_dir.mkdir(exist_ok=True)

        # COCO URLs
        base_url = "http://images.cocodataset.org"
        
        files = {
            "images": f"{base_url}/zips/{subset}.zip",
            "annotations": f"{base_url}/annotations/annotations_trainval2017.zip"
        }

        # Download images
        images_zip = coco_dir / f"{subset}.zip"
        if not (coco_dir / subset).exists():
            if not images_zip.exists():
                self.download_file(files["images"], images_zip, f"COCO {subset} images")
            self.extract_zip(images_zip, coco_dir)
        else:
            print(f"COCO {subset} images already exist")

        # Download annotations
        ann_zip = coco_dir / "annotations_trainval2017.zip"
        if not (coco_dir / "annotations").exists():
            if not ann_zip.exists():
                self.download_file(files["annotations"], ann_zip, "COCO annotations")
            self.extract_zip(ann_zip, coco_dir)
        else:
            print("COCO annotations already exist")

        # Verify and report
        ann_file = coco_dir / "annotations" / f"person_keypoints_{subset}.json"
        if ann_file.exists():
            with open(ann_file) as f:
                data = json.load(f)
            print(f"\nCOCO {subset} Dataset Statistics:")
            print(f"  Images: {len(data['images'])}")
            print(f"  Annotations: {len(data['annotations'])}")
            print(f"  Keypoints: {data['categories'][0]['keypoints']}")

        return coco_dir

    def download_mpii(self):
        """
        Download MPII Human Pose dataset.
        
        Note: MPII requires academic registration.
        This creates the structure and provides instructions.
        """
        mpii_dir = self.data_dir / "mpii"
        mpii_dir.mkdir(exist_ok=True)

        print("""
MPII Human Pose Dataset
========================
MPII requires registration at: http://human-pose.mpi-inf.mpg.de/

Steps to download:
1. Visit http://human-pose.mpi-inf.mpg.de/
2. Register with your academic email
3. Download:
   - MPII Human Pose Dataset (images) (~12GB)
   - MPII Human Pose Annotations

After downloading, place files in:
  {mpii_dir}/
    images/
    mpii_human_pose_v1_u12_2.mat (annotations)
        """.format(mpii_dir=mpii_dir))

        # Create expected structure
        (mpii_dir / "images").mkdir(exist_ok=True)

        return mpii_dir

    def create_synthetic_dataset(self, num_samples: int = 10000):
        """
        Create a synthetic pose dataset for testing/development.
        Generates realistic-looking poses with variations.
        """
        import numpy as np
        
        syn_dir = self.data_dir / "synthetic"
        syn_dir.mkdir(exist_ok=True)

        print(f"Generating synthetic dataset with {num_samples} samples...")

        # Generate synthetic keypoints
        keypoints = []
        labels = []

        movements = [
            "walking", "running", "jumping", "sitting", "standing",
            "waving", "clapping", "kicking", "punching", "squatting"
        ]

        for i in range(num_samples):
            # Base skeleton (normalized 0-1)
            base_pose = np.array([
                [0.5, 0.1],   # nose
                [0.48, 0.08], # left_eye
                [0.52, 0.08], # right_eye
                [0.45, 0.1],  # left_ear
                [0.55, 0.1],  # right_ear
                [0.4, 0.25],  # left_shoulder
                [0.6, 0.25],  # right_shoulder
                [0.35, 0.4],  # left_elbow
                [0.65, 0.4],  # right_elbow
                [0.3, 0.55],  # left_wrist
                [0.7, 0.55],  # right_wrist
                [0.45, 0.5],  # left_hip
                [0.55, 0.5],  # right_hip
                [0.43, 0.75], # left_knee
                [0.57, 0.75], # right_knee
                [0.42, 0.95], # left_ankle
                [0.58, 0.95], # right_ankle
            ])

            # Add movement-specific variation
            movement_idx = i % len(movements)
            movement = movements[movement_idx]

            # Apply movement variations
            if movement == "walking":
                phase = (i % 20) / 20.0 * 2 * np.pi
                base_pose[15, 1] += 0.05 * np.sin(phase)  # left ankle
                base_pose[16, 1] += 0.05 * np.sin(phase + np.pi)  # right ankle
            elif movement == "running":
                phase = (i % 10) / 10.0 * 2 * np.pi
                base_pose[15, 1] += 0.1 * np.sin(phase)
                base_pose[16, 1] += 0.1 * np.sin(phase + np.pi)
                base_pose[9, 1] += 0.05 * np.cos(phase)
                base_pose[10, 1] += 0.05 * np.cos(phase + np.pi)
            elif movement == "jumping":
                jump_height = 0.1 * np.sin((i % 15) / 15.0 * np.pi)
                base_pose[:, 1] -= jump_height
            elif movement == "waving":
                phase = (i % 12) / 12.0 * 2 * np.pi
                base_pose[9, 0] += 0.1 * np.sin(phase)
                base_pose[9, 1] -= 0.15
            elif movement == "squatting":
                base_pose[13:17, 1] -= 0.15
                base_pose[11:13, 1] += 0.05

            # Add random noise
            noise = np.random.randn(17, 2) * 0.02
            pose = base_pose + noise
            pose = np.clip(pose, 0, 1)

            keypoints.append(pose)
            labels.append(movement_idx)

        # Save dataset
        np.save(syn_dir / "keypoints.npy", np.array(keypoints))
        np.save(syn_dir / "labels.npy", np.array(labels))

        # Save metadata
        metadata = {
            "num_samples": num_samples,
            "num_keypoints": 17,
            "num_classes": len(movements),
            "classes": movements,
            "keypoint_names": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
        }
        with open(syn_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {syn_dir}/")
        print(f"  Samples: {num_samples}")
        print(f"  Classes: {movements}")

        return syn_dir

    def create_motion_sequences(self, num_sequences: int = 5000, seq_length: int = 30):
        """
        Create synthetic motion sequences for prediction training.
        """
        import numpy as np

        seq_dir = self.data_dir / "motion_sequences"
        seq_dir.mkdir(exist_ok=True)

        print(f"Generating {num_sequences} motion sequences...")

        sequences = []
        future_frames = []

        for i in range(num_sequences):
            # Generate smooth motion using sine waves
            t = np.linspace(0, 4 * np.pi, seq_length + 10)  # 30 past + 10 future

            # Base walking motion
            base_pose = np.zeros((17, 2))
            base_pose[:, 0] = 0.5  # Center x
            base_pose[:, 1] = np.linspace(0.1, 0.95, 17)  # Vertical spread

            sequence = []
            for frame_idx, t_val in enumerate(t):
                pose = base_pose.copy()

                # Walking motion
                leg_phase = t_val + frame_idx * 0.1
                pose[15, 1] += 0.05 * np.sin(leg_phase)  # left ankle
                pose[16, 1] += 0.05 * np.sin(leg_phase + np.pi)  # right ankle
                pose[13, 1] += 0.03 * np.sin(leg_phase)  # left knee
                pose[14, 1] += 0.03 * np.sin(leg_phase + np.pi)  # right knee

                # Arm swing
                pose[9, 0] += 0.05 * np.sin(leg_phase + np.pi)  # left wrist
                pose[10, 0] += 0.05 * np.sin(leg_phase)  # right wrist

                # Add noise
                pose += np.random.randn(17, 2) * 0.01
                pose = np.clip(pose, 0, 1)
                sequence.append(pose)

            sequence = np.array(sequence)
            sequences.append(sequence[:seq_length])  # Input
            future_frames.append(sequence[seq_length:])  # Target

        # Save
        np.save(seq_dir / "input_sequences.npy", np.array(sequences))
        np.save(seq_dir / "future_frames.npy", np.array(future_frames))

        metadata = {
            "num_sequences": num_sequences,
            "input_length": seq_length,
            "prediction_length": 10,
            "num_keypoints": 17
        }
        with open(seq_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved to {seq_dir}/")
        print(f"  Input shape: {np.array(sequences).shape}")
        print(f"  Target shape: {np.array(future_frames).shape}")

        return seq_dir


def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--coco", action="store_true", help="Download COCO val2017")
    parser.add_argument("--coco-train", action="store_true", help="Download COCO train2017 (18GB)")
    parser.add_argument("--mpii", action="store_true", help="Show MPII download instructions")
    parser.add_argument("--synthetic", type=int, default=10000, 
                        help="Generate synthetic dataset with N samples")
    parser.add_argument("--motion", type=int, default=5000,
                        help="Generate motion sequences with N sequences")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Data directory")
    parser.add_argument("--all-synthetic", action="store_true",
                        help="Generate all synthetic datasets")

    args = parser.parse_args()

    downloader = DatasetDownloader(args.data_dir)

    if args.coco:
        downloader.download_coco_keypoints("val2017")
    
    if args.coco_train:
        downloader.download_coco_keypoints("train2017")

    if args.mpii:
        downloader.download_mpii()

    if args.synthetic > 0 or args.all_synthetic:
        downloader.create_synthetic_dataset(args.synthetic)

    if args.motion > 0 or args.all_synthetic:
        downloader.create_motion_sequences(args.motion)

    if not any([args.coco, args.coco_train, args.mpii, args.synthetic, args.motion, args.all_synthetic]):
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python scripts/download_data.py --all-synthetic")


if __name__ == "__main__":
    main()
