"""
Selma Motion Engine - Holistic Dashboard.
Full-body + Face + Hands real-time analysis.
Copyright (c) 2026 Selma Haci.
"""

import streamlit as st
import numpy as np
import cv2
import time
import sys, os
import pandas as pd

project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
motion_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, motion_root)

from src.pipeline import create_engine
from src.config import MOVEMENT_CLASSES, SKELETON_CONNECTIONS, KEYPOINT_NAMES

st.set_page_config(page_title="Selma Motion Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0d1321 100%);
        font-family: 'Inter', sans-serif; color: #e4e4e7;
    }
    .metric-card {
        background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 12px;
        padding: 14px; text-align: center; margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.6rem; font-weight: 700;
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.7rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px;
    }
    .status-live {
        background: linear-gradient(135deg, #059669, #10b981);
        padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
        font-weight: 600; color: white; display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.7;} }
    .stitle {
        font-size: 0.9rem; font-weight: 600; color: #d1d5db;
        border-bottom: 1px solid #374151; padding-bottom: 5px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# MediaPipe Pose skeleton (33 landmarks)
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32)
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (0,17),(13,17),(17,18),(18,19),(19,20)
]

FACE_CONTOUR = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,54,103,67,109
]


def draw_holistic(frame, result):
    """Draw full body + hands + face on frame."""
    h, w = frame.shape[:2]

    # 1. Draw full pose skeleton (33 landmarks)
    if result.all_pose_landmarks is not None:
        pose = result.all_pose_landmarks
        for i, j in POSE_CONNECTIONS:
            if i < len(pose) and j < len(pose):
                pt1 = (int(pose[i][0]*w), int(pose[i][1]*h))
                pt2 = (int(pose[j][0]*w), int(pose[j][1]*h))
                cv2.line(frame, pt1, pt2, (0, 255, 200), 2, cv2.LINE_AA)
        for pt in pose:
            center = (int(pt[0]*w), int(pt[1]*h))
            cv2.circle(frame, center, 4, (0, 200, 255), -1, cv2.LINE_AA)

    # 2. Draw hands with finger detail
    for hand, color in [(result.left_hand, (255, 150, 50)), (result.right_hand, (50, 150, 255))]:
        if hand is not None:
            for i, j in HAND_CONNECTIONS:
                if i < len(hand) and j < len(hand):
                    pt1 = (int(hand[i][0]*w), int(hand[i][1]*h))
                    pt2 = (int(hand[j][0]*w), int(hand[j][1]*h))
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
            for pt in hand:
                center = (int(pt[0]*w), int(pt[1]*h))
                cv2.circle(frame, center, 3, (255, 255, 255), -1, cv2.LINE_AA)

    # 3. Draw face mesh contour
    if result.face_landmarks is not None:
        face = result.face_landmarks
        # Draw face contour
        for i in range(len(FACE_CONTOUR) - 1):
            idx1, idx2 = FACE_CONTOUR[i], FACE_CONTOUR[i+1]
            if idx1 < len(face) and idx2 < len(face):
                pt1 = (int(face[idx1][0]*w), int(face[idx1][1]*h))
                pt2 = (int(face[idx2][0]*w), int(face[idx2][1]*h))
                cv2.line(frame, pt1, pt2, (180, 120, 255), 1, cv2.LINE_AA)

        # Draw key facial features (eyes, nose, lips)
        eye_indices = [33, 133, 362, 263, 159, 145, 386, 374]  # eye corners
        nose_indices = [1, 4, 5, 195]
        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

        for idx in eye_indices:
            if idx < len(face):
                pt = (int(face[idx][0]*w), int(face[idx][1]*h))
                cv2.circle(frame, pt, 2, (0, 255, 255), -1)

        for i in range(len(lip_indices) - 1):
            idx1, idx2 = lip_indices[i], lip_indices[i+1]
            if idx1 < len(face) and idx2 < len(face):
                pt1 = (int(face[idx1][0]*w), int(face[idx1][1]*h))
                pt2 = (int(face[idx2][0]*w), int(face[idx2][1]*h))
                cv2.line(frame, pt1, pt2, (0, 180, 255), 1, cv2.LINE_AA)

        for idx in nose_indices:
            if idx < len(face):
                pt = (int(face[idx][0]*w), int(face[idx][1]*h))
                cv2.circle(frame, pt, 2, (0, 200, 0), -1)

    return frame


def main():
    st.markdown("## Selma Motion Engine")
    st.markdown("Full-Body Holistic Analysis: Pose + Face + Hands")
    st.markdown("---")

    tab_rt, tab_info = st.tabs(["Real-time Analysis", "Engine Info"])

    with tab_rt:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            start_btn = st.button("Start Live Analysis", type="primary", use_container_width=True)
        with c2:
            cam_id = st.selectbox("Camera", [0, 1, 2], index=0)
        with c3:
            max_frames = st.number_input("Max Frames", 100, 5000, 500, step=100)

        col_cam, col_data = st.columns([2.5, 1])
        with col_cam:
            st.markdown('<div class="stitle">Live Camera Feed (Pose + Face + Hands)</div>', unsafe_allow_html=True)
            frame_ph = st.empty()
        with col_data:
            st.markdown('<div class="stitle">Detection Status</div>', unsafe_allow_html=True)
            class_ph = st.empty()
            conf_ph = st.empty()
            det_pose = st.empty()
            det_face = st.empty()
            det_hands = st.empty()
            lat_ph = st.empty()
            fps_ph = st.empty()
            frame_ph2 = st.empty()

        st.markdown("---")
        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<div class="stitle">Latency History (ms)</div>', unsafe_allow_html=True)
            lat_chart = st.empty()
        with g2:
            st.markdown('<div class="stitle">Confidence History</div>', unsafe_allow_html=True)
            conf_chart = st.empty()

        g3, g4 = st.columns(2)
        with g3:
            st.markdown('<div class="stitle">Keypoint Visibility (17 COCO)</div>', unsafe_allow_html=True)
            kp_chart = st.empty()
        with g4:
            st.markdown('<div class="stitle">Activity Distribution</div>', unsafe_allow_html=True)
            act_chart = st.empty()

        if start_btn:
            engine = create_engine()
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                st.error("Cannot access camera.")
                return

            st.markdown('<span class="status-live">LIVE</span>', unsafe_allow_html=True)
            lat_hist, conf_hist, act_hist = [], [], []

            try:
                for fn in range(int(max_frames)):
                    ret, frame = cap.read()
                    if not ret: break

                    result = engine.process_frame(frame)
                    annotated = draw_holistic(frame.copy(), result)

                    lat_hist.append(result.inference_time_ms)
                    conf_hist.append(result.class_confidence)
                    act_hist.append(result.class_name)
                    avg_lat = np.mean(lat_hist[-30:])
                    cur_fps = 1000.0 / avg_lat if avg_lat > 0 else 0

                    if fn % 3 == 0:
                        frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", width=640)

                        class_ph.markdown(f'<div class="metric-card"><div class="metric-value">{result.class_name}</div><div class="metric-label">Activity</div></div>', unsafe_allow_html=True)
                        conf_ph.markdown(f'<div class="metric-card"><div class="metric-value">{result.class_confidence*100:.0f}%</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)

                        pose_ok = result.all_pose_landmarks is not None
                        face_ok = result.face_landmarks is not None
                        hands_ok = (result.left_hand is not None) or (result.right_hand is not None)
                        n_hands = int(result.left_hand is not None) + int(result.right_hand is not None)

                        det_pose.markdown(f'<div class="metric-card"><div class="metric-value">{"33 pts" if pose_ok else "---"}</div><div class="metric-label">Pose</div></div>', unsafe_allow_html=True)
                        det_face.markdown(f'<div class="metric-card"><div class="metric-value">{"478 pts" if face_ok else "---"}</div><div class="metric-label">Face Mesh</div></div>', unsafe_allow_html=True)
                        det_hands.markdown(f'<div class="metric-card"><div class="metric-value">{f"{n_hands} hand(s)" if hands_ok else "---"}</div><div class="metric-label">Hands</div></div>', unsafe_allow_html=True)

                        lat_ph.markdown(f'<div class="metric-card"><div class="metric-value">{avg_lat:.0f}ms</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)
                        fps_ph.markdown(f'<div class="metric-card"><div class="metric-value">{cur_fps:.0f}</div><div class="metric-label">FPS</div></div>', unsafe_allow_html=True)
                        frame_ph2.markdown(f'<div class="metric-card"><div class="metric-value">{fn}</div><div class="metric-label">Frames</div></div>', unsafe_allow_html=True)

                    if fn % 10 == 0 and fn > 0:
                        lat_chart.line_chart(pd.DataFrame({'Latency': lat_hist[-100:]}), color='#06b6d4')
                        conf_chart.area_chart(pd.DataFrame({'Confidence': conf_hist[-100:]}), color='#10b981')

                        kp_names = ['Nose','LEye','REye','LEar','REar','LSho','RSho','LElb','RElb','LWri','RWri','LHip','RHip','LKne','RKne','LAnk','RAnk']
                        kp_chart.bar_chart(pd.DataFrame({'Visibility': result.keypoint_scores}, index=kp_names), color='#8b5cf6')

                        recent = act_hist[-50:]
                        counts = {a: recent.count(a) for a in set(recent)}
                        act_chart.bar_chart(pd.DataFrame({'Frames': counts.values()}, index=counts.keys()), color='#f59e0b')

                    time.sleep(0.01)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                cap.release()
                if lat_hist:
                    st.success(f"Session complete: {len(lat_hist)} frames.")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Total Frames", len(lat_hist))
                    s2.metric("Avg Latency", f"{np.mean(lat_hist):.0f}ms")
                    s3.metric("Avg FPS", f"{np.mean([1000/l for l in lat_hist if l > 0]):.0f}")
                    s4.metric("Avg Confidence", f"{np.mean(conf_hist)*100:.0f}%")

    with tab_info:
        st.markdown("##### Engine Diagnostics")
        diag = create_engine().get_diagnostics()
        for k, v in diag.items():
            st.markdown(f'<div class="metric-card"><div class="metric-label">{k}</div><div class="metric-value" style="font-size:1rem;">{v}</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("##### Detection Capabilities")
        st.markdown("- **Pose**: 33 full body landmarks (MediaPipe Full)")
        st.markdown("- **Face**: 478 mesh landmarks (eyes, nose, lips, contour)")
        st.markdown("- **Hands**: 21 landmarks per hand (all 5 fingers)")
        st.markdown("- **Classification**: Biomechanical posture analysis")


if __name__ == "__main__":
    main()
