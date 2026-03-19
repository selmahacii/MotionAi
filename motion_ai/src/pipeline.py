"""
Selma Motion Engine (SME) - Full-Body Holistic Analysis Pipeline.
Copyright (c) 2026 Selma Haci. Proprietary Analytic Engine.

Combines Pose (33 pts) + Hands (21 pts x2) + Face (478 pts) for comprehensive tracking.
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    RunningMode
)

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import MOVEMENT_CLASSES, SKELETON_CONNECTIONS, inference_config

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


class SME_DataPacket:
    """Full holistic output: body + hands + face."""
    def __init__(self, keypoints, keypoint_scores, predicted_class, class_name,
                 class_confidence, class_probabilities, predicted_motion,
                 frame_idx=0, inference_time_ms=0.0,
                 face_landmarks=None, left_hand=None, right_hand=None,
                 all_pose_landmarks=None):
        self.keypoints = keypoints
        self.keypoint_scores = keypoint_scores
        self.predicted_class = predicted_class
        self.class_name = class_name
        self.class_confidence = class_confidence
        self.class_probabilities = class_probabilities
        self.predicted_motion = predicted_motion
        self.frame_idx = frame_idx
        self.inference_time_ms = inference_time_ms
        self.face_landmarks = face_landmarks       # (478, 2) or None
        self.left_hand = left_hand                 # (21, 2) or None
        self.right_hand = right_hand               # (21, 2) or None
        self.all_pose_landmarks = all_pose_landmarks  # (33, 2) full pose


class BaseEngine(ABC):
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> SME_DataPacket: pass
    @abstractmethod
    def reset(self): pass
    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]: pass


class AnalyticEngine(BaseEngine):
    """
    SME Holistic Engine: Full-body + Face + Hands detection.
    Uses MediaPipe Tasks API for Python 3.14+ compatibility.
    """

    MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def __init__(self, **kwargs):
        # Pose Landmarker (Full model - 33 landmarks)
        pose_path = os.path.normpath(os.path.join(MODELS_DIR, 'pose_landmarker_full.task'))
        if not os.path.exists(pose_path):
            pose_path = os.path.normpath(os.path.join(MODELS_DIR, 'pose_landmarker.task'))

        self.pose_landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

        # Hand Landmarker (21 landmarks per hand)
        hand_path = os.path.normpath(os.path.join(MODELS_DIR, 'hand_landmarker.task'))
        self.hand_landmarker = None
        if os.path.exists(hand_path):
            self.hand_landmarker = HandLandmarker.create_from_options(
                HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=hand_path),
                    running_mode=RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.4,
                    min_tracking_confidence=0.4
                )
            )

        # Face Landmarker (478 landmarks)
        face_path = os.path.normpath(os.path.join(MODELS_DIR, 'face_landmarker.task'))
        self.face_landmarker = None
        if os.path.exists(face_path):
            self.face_landmarker = FaceLandmarker.create_from_options(
                FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=face_path),
                    running_mode=RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.4,
                    min_tracking_confidence=0.4
                )
            )

        self.coordinate_buffer = deque(maxlen=60)
        self.cycle_count = 0

    def process_frame(self, frame: np.ndarray) -> SME_DataPacket:
        start_time = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # === POSE ===
        pose_result = self.pose_landmarker.detect(mp_image)
        all_pose = None
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            lms = pose_result.pose_landmarks[0]
            coords = np.array([[lms[i].x, lms[i].y] for i in self.MP_TO_COCO])
            scores = np.array([lms[i].visibility for i in self.MP_TO_COCO])
            all_pose = np.array([[l.x, l.y] for l in lms])
        else:
            coords = np.zeros((17, 2))
            scores = np.zeros(17)

        # === HANDS ===
        left_hand, right_hand = None, None
        if self.hand_landmarker:
            hand_result = self.hand_landmarker.detect(mp_image)
            if hand_result.hand_landmarks:
                for i, hand_lms in enumerate(hand_result.hand_landmarks):
                    pts = np.array([[l.x, l.y] for l in hand_lms])
                    handedness = hand_result.handedness[i][0].category_name
                    if handedness == "Left":
                        left_hand = pts
                    else:
                        right_hand = pts

        # === FACE ===
        face_lms = None
        if self.face_landmarker:
            face_result = self.face_landmarker.detect(mp_image)
            if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                face_lms = np.array([[l.x, l.y] for l in face_result.face_landmarks[0]])

        self.coordinate_buffer.append(coords)

        # Classification
        class_idx, class_conf = self._classify_posture(coords, scores)
        probs = np.zeros(len(MOVEMENT_CLASSES))
        if class_idx < len(probs):
            probs[class_idx] = class_conf

        latency = (time.time() - start_time) * 1000

        return SME_DataPacket(
            keypoints=coords,
            keypoint_scores=scores,
            predicted_class=class_idx,
            class_name=MOVEMENT_CLASSES[class_idx] if class_idx < len(MOVEMENT_CLASSES) else "unknown",
            class_confidence=class_conf,
            class_probabilities=probs,
            predicted_motion=np.array([]),
            frame_idx=self.cycle_count,
            inference_time_ms=latency,
            face_landmarks=face_lms,
            left_hand=left_hand,
            right_hand=right_hand,
            all_pose_landmarks=all_pose
        )

    def _classify_posture(self, coords, scores):
        if scores.mean() < 0.3:
            return 0, 0.3

        shoulder_y = (coords[5, 1] + coords[6, 1]) / 2
        hip_y = (coords[11, 1] + coords[12, 1]) / 2
        knee_y = (coords[13, 1] + coords[14, 1]) / 2
        ankle_y = (coords[15, 1] + coords[16, 1]) / 2
        wrist_y = min(coords[9, 1], coords[10, 1])

        torso = abs(hip_y - shoulder_y)
        legs = abs(ankle_y - hip_y)

        if torso < 0.01:
            return 0, 0.4

        # Arms raised
        if wrist_y < shoulder_y - 0.05:
            return 12, 0.88

        # Sitting
        ratio = legs / torso if torso > 0 else 0
        if ratio < 0.8:
            return 1, 0.92

        # Standing
        if ratio > 1.0:
            return 0, 0.90

        # Crouching
        if hip_y > 0.6:
            return 4, 0.80

        # Movement
        if len(self.coordinate_buffer) > 5:
            recent = list(self.coordinate_buffer)
            dx = abs(recent[-1][11, 0] - recent[-5][11, 0])
            if dx > 0.05:
                return 5, 0.75

        self.cycle_count += 1
        return 0, 0.70

    def reset(self):
        self.coordinate_buffer.clear()
        self.cycle_count = 0

    def get_diagnostics(self):
        return {
            "engine": "SME Holistic v7",
            "pose": "33 landmarks (Full)",
            "hands": "21 landmarks x2",
            "face": "478 landmarks",
            "status": "Active"
        }


def create_engine(use_simulation=False, device="cpu", **kwargs):
    return AnalyticEngine(**kwargs)
