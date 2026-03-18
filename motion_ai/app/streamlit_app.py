"""
Streamlit Dashboard for Human Motion Intelligence System.
Real-time visualization of pose estimation, movement classification, and motion prediction.

Models:
1. PoseNet (Stacked Hourglass) - 17 keypoint pose estimation
2. MoveClassifier (BiLSTM + Attention) - 15-class movement classification
3. MotionFormer (Transformer) - 20→10 frame motion prediction
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_KEYPOINTS, KEYPOINT_DIM, MOVEMENT_CLASSES, NUM_CLASSES,
    SKELETON_CONNECTIONS,
    posenet_config, classifier_config, predictor_config
)
from src.pipeline import create_engine, BaseEngine, AnalyticEngine, SimulatedEngine, SME_DataPacket
from src.visualization import draw_skeleton, create_multi_skeleton_view


# Page config
st.set_page_config(
    page_title="Human Motion Intelligence",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_engine(
    posenet_path: Optional[str] = None,
    classifier_path: Optional[str] = None,
    predictor_path: Optional[str] = None,
    use_simulation: bool = True
) -> BaseEngine:
    """Load the Selma Motion Analytic Engine (cached)."""
    return create_engine(
        posenet_path=posenet_path,
        classifier_path=classifier_path,
        predictor_path=predictor_path,
        device="cpu",
        use_simulation=use_simulation
    )


def create_skeleton_plotly(
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    title: str = "Current Pose",
    color: str = "#2ca02c",
    show_keypoints: bool = True
) -> go.Figure:
    """Create an interactive skeleton visualization using Plotly."""
    fig = go.Figure()
    
    # Draw skeleton connections with gradient
    for i, j in SKELETON_CONNECTIONS:
        if i < len(keypoints) and j < len(keypoints):
            fig.add_trace(go.Scatter(
                x=[keypoints[i, 0], keypoints[j, 0]],
                y=[keypoints[i, 1], keypoints[j, 1]],
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Draw keypoints
    if show_keypoints:
        colors = scores if scores is not None else [0.9] * len(keypoints)
        hover_text = [
            f"{['nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder','L_elbow','R_elbow','L_wrist','R_wrist','L_hip','R_hip','L_knee','R_knee','L_ankle','R_ankle'][i]}<br>Score: {s:.2f}"
            for i, s in enumerate(colors)
        ]
        
        fig.add_trace(go.Scatter(
            x=keypoints[:, 0],
            y=keypoints[:, 1],
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Confidence", x=1.02, len=0.8),
                line=dict(color='white', width=2)
            ),
            text=hover_text,
            hoverinfo='text',
            name='Keypoints'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(range=[0, 1], constrain='domain', visible=False),
        yaxis=dict(range=[0, 1], constrain='domain', autorange='reversed', visible=False),
        width=400,
        height=450,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_prediction_visualization(
    current_kp: np.ndarray,
    predicted_kp: np.ndarray,
    title: str = "Motion Prediction"
) -> go.Figure:
    """Create visualization of predicted future motion with fading effect."""
    fig = go.Figure()
    
    # Current skeleton (solid green)
    for i, j in SKELETON_CONNECTIONS:
        if i < len(current_kp) and j < len(current_kp):
            fig.add_trace(go.Scatter(
                x=[current_kp[i, 0], current_kp[j, 0]],
                y=[current_kp[i, 1], current_kp[j, 1]],
                mode='lines',
                line=dict(color='#2ca02c', width=4),
                showlegend=False
            ))
    
    # Predicted skeletons (fading orange)
    n_pred = len(predicted_kp)
    for idx, pred_kp in enumerate(predicted_kp):
        alpha = 0.8 * (1 - (idx + 1) / (n_pred + 1))
        color = f'rgba(255, 165, 0, {alpha:.2f})'
        
        for i, j in SKELETON_CONNECTIONS:
            if i < len(pred_kp) and j < len(pred_kp):
                fig.add_trace(go.Scatter(
                    x=[pred_kp[i, 0], pred_kp[j, 0]],
                    y=[pred_kp[i, 1], pred_kp[j, 1]],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False
                ))
    
    # Add legend annotation
    fig.add_annotation(
        x=0.5, y=1.05,
        text="Green: Current | Orange: Predicted (fading = further future)",
        showarrow=False,
        xref="paper", yref="paper",
        font=dict(size=10, color="gray")
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(range=[0, 1], visible=False),
        yaxis=dict(range=[0, 1], autorange='reversed', visible=False),
        width=400,
        height=450,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_class_probabilities_chart(probabilities: np.ndarray) -> go.Figure:
    """Create horizontal bar chart of class probabilities."""
    colors = ['#2ca02c' if p == max(probabilities) else '#1f77b4' for p in probabilities]
    
    fig = go.Figure(go.Bar(
        y=MOVEMENT_CLASSES,
        x=probabilities,
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1%}' for p in probabilities],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text="Classification Probabilities", font=dict(size=16)),
        xaxis=dict(range=[0, 1.1], title="Probability"),
        yaxis=dict(categoryorder='total ascending'),
        width=500,
        height=400,
        margin=dict(l=120, r=50, t=40, b=40)
    )
    
    return fig


def create_trajectory_plot(history: List[np.ndarray], keypoint_idx: int = 0) -> go.Figure:
    """Create trajectory plot for a specific keypoint over time."""
    if not history:
        return go.Figure()
    
    trajectories = np.array(history)
    keypoint_names = ['nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
                      'L_elbow','R_elbow','L_wrist','R_wrist','L_hip','R_hip',
                      'L_knee','R_knee','L_ankle','R_ankle']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('X Position', 'Y Position'))
    
    fig.add_trace(
        go.Scatter(y=trajectories[:, keypoint_idx, 0], name='X', 
                   line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=trajectories[:, keypoint_idx, 1], name='Y', 
                   line=dict(color='#d62728', width=2)),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Keypoint Trajectory ({keypoint_names[keypoint_idx]})",
        height=250,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_metrics_display(
    inference_time: float,
    class_name: str,
    confidence: float,
    fps: float
) -> None:
    """Display metrics in a nice card format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("⏱️ Inference", f"{inference_time:.1f} ms")
    
    with col2:
        st.metric("🎬 FPS", f"{fps:.1f}")
    
    with col3:
        st.metric("🎯 Predicted Class", class_name.title())
    
    with col4:
        st.metric("📊 Confidence", f"{confidence:.1%}")


def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🏃 Human Motion Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown("### Real-time Pose Estimation, Movement Classification & Motion Prediction")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model settings
    st.sidebar.subheader("Model Configuration")
    use_mock = st.sidebar.checkbox("Use Demo Mode (Mock Pipeline)", value=True, 
                                    help="Demo mode generates synthetic motion for visualization")
    
    if not use_mock:
        st.sidebar.info("Enter paths to trained model weights:")
        posenet_path = st.sidebar.text_input("PoseNet Weights", value="", 
                                              placeholder="models/posenet/weights/posenet_best.pth")
        classifier_path = st.sidebar.text_input("Classifier Weights", value="",
                                                 placeholder="models/classifier/weights/classifier_best.pth")
        predictor_path = st.sidebar.text_input("Predictor Weights", value="",
                                                placeholder="models/predictor/weights/predictor_best.pth")
    else:
        posenet_path = classifier_path = predictor_path = None
        st.sidebar.success("✓ Demo mode active - using synthetic motion")
    
    # Visualization settings
    st.sidebar.subheader("Visualization")
    show_prediction = st.sidebar.checkbox("Show Motion Prediction", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
    show_trajectory = st.sidebar.checkbox("Show Keypoint Trajectory", value=True)
    selected_keypoint = st.sidebar.selectbox(
        "Track Keypoint", 
        range(NUM_KEYPOINTS),
        format_func=lambda i: ['nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
                               'L_elbow','R_elbow','L_wrist','R_wrist','L_hip','R_hip',
                               'L_knee','R_knee','L_ankle','R_ankle'][i],
        index=0
    )
    
    # Demo mode specific settings
    if use_mock:
        st.sidebar.subheader("Demo Motion")
        demo_class = st.sidebar.selectbox(
            "Movement Type",
            range(len(MOVEMENT_CLASSES)),
            format_func=lambda i: MOVEMENT_CLASSES[i],
            index=1  # Default to walking
        )
    
    # Initialize Selma Engine
    engine = load_engine(
        posenet_path=posenet_path,
        classifier_path=classifier_path,
        predictor_path=predictor_path,
        use_simulation=use_mock
    )
    
    # Set demo profile if simulated
    if use_mock and hasattr(engine, 'current_profile'):
        engine.current_profile = demo_class
    
    # Session state for history
    if 'keypoint_history' not in st.session_state:
        st.session_state.keypoint_history = []
    if 'class_history' not in st.session_state:
        st.session_state.class_history = []
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Real-Time Analysis", 
        "📊 Training Metrics", 
        "🧠 Model Architecture",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header("Real-Time Motion Analysis")
        
        # Control buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            run_analysis = st.button("▶️ Start Analysis", type="primary")
        with col2:
            if st.button("🔄 Reset"):
                engine.reset()
                st.session_state.keypoint_history = []
                st.session_state.class_history = []
                st.session_state.frame_count = 0
                st.rerun()
        
        # Create placeholder for dynamic content
        if run_analysis or st.session_state.frame_count > 0:
            # Run analysis loop
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            charts_placeholder = st.empty()
            
            # Generate fake frame
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Process frames
            n_frames = st.slider("Number of frames", 10, 60, 30)
            progress_bar = st.progress(0)
            
            for frame_idx in range(n_frames):
                # Execute SME Analysis cycle
                if use_mock and hasattr(engine, 'current_profile'):
                    engine.current_profile = demo_class
                
                # Fetch DataPacket
                result = engine.process_frame(fake_frame)
                
                # Update trajectory history
                st.session_state.keypoint_history.append(result.keypoints)
                st.session_state.class_history.append({
                    'class': result.predicted_class,
                    'confidence': result.class_confidence
                })
                st.session_state.frame_count += result.frame_idx
                
                # Limit history size
                if len(st.session_state.keypoint_history) > 100:
                    st.session_state.keypoint_history = st.session_state.keypoint_history[-100:]
                
                # Update progress
                progress_bar.progress((frame_idx + 1) / n_frames)
                
                # Create visualizations
                with frame_placeholder.container():
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        # Current skeleton
                        fig = create_skeleton_plotly(
                            result.keypoints,
                            result.keypoint_scores if show_confidence else None,
                            title=f"Current Pose (Frame {frame_idx + 1})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_b:
                        # Motion prediction
                        if show_prediction and len(result.predicted_motion) > 0:
                            fig = create_prediction_visualization(
                                result.keypoints,
                                result.predicted_motion
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Motion prediction requires more frames...")
                    
                    with col_c:
                        # Keypoint trajectory
                        if show_trajectory and len(st.session_state.keypoint_history) > 5:
                            fig = create_trajectory_plot(
                                st.session_state.keypoint_history[-30:],
                                keypoint_idx=selected_keypoint
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Collecting trajectory data...")
                
                # Update metrics
                with metrics_placeholder:
                    fps = 1000 / result.inference_time_ms if result.inference_time_ms > 0 else 0
                    create_metrics_display(
                        result.inference_time_ms,
                        result.class_name,
                        result.class_confidence,
                        fps
                    )
                
                # Real-time delay
                time.sleep(0.05)
            
            progress_bar.empty()
            st.success(f"✅ Analysis complete! Processed {n_frames} frames.")
            
            # Final summary
            st.subheader("📈 Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Class distribution pie chart
                class_counts = {}
                for entry in st.session_state.class_history:
                    cls = MOVEMENT_CLASSES[entry['class']]
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(class_counts.keys()),
                    values=list(class_counts.values()),
                    hole=.4,
                    marker_colors=['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd']
                )])
                fig.update_layout(title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence over time
                confidences = [entry['confidence'] for entry in st.session_state.class_history]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=confidences,
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='#2ca02c', width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    title="Classification Confidence Over Time",
                    xaxis_title="Frame",
                    yaxis_title="Confidence",
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Click 'Start Analysis' to begin real-time motion analysis")
            
            # Show sample visualization
            st.subheader("Sample Output Preview")
            
            sample_kp = np.array([
                [0.5, 0.1], [0.47, 0.08], [0.53, 0.08], [0.44, 0.1], [0.56, 0.1],
                [0.4, 0.25], [0.6, 0.25], [0.32, 0.35], [0.68, 0.35],
                [0.25, 0.45], [0.75, 0.45], [0.45, 0.5], [0.55, 0.5],
                [0.43, 0.7], [0.57, 0.7], [0.42, 0.9], [0.58, 0.9]
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_skeleton_plotly(sample_kp, title="Sample Skeleton (Standing)")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sample prediction
                rng = np.random.RandomState(42)
                predicted = sample_kp + rng.randn(10, 17, 2) * 0.02
                fig = create_prediction_visualization(sample_kp, predicted)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 Training Metrics")
        st.markdown("View training progress for each model")
        
        history_dir = Path(__file__).parent.parent / "models"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 PoseNet (Stacked Hourglass)")
            posenet_history = history_dir / "posenet" / "weights" / "training_history.json"
            if posenet_history.exists():
                with open(posenet_history) as f:
                    history = json.load(f)
                st.json({"epochs_trained": len(history.get("train_loss", []))})
                st.success("✓ Trained")
            else:
                st.info("Not trained yet. Run `python models/posenet/train.py`")
        
        with col2:
            st.markdown("### 🏃 MoveClassifier (BiLSTM)")
            classifier_history = history_dir / "classifier" / "weights" / "training_history.json"
            if classifier_history.exists():
                with open(classifier_history) as f:
                    history = json.load(f)
                st.json({"epochs_trained": len(history.get("train_loss", []))})
                st.success("✓ Trained")
            else:
                st.info("Not trained yet. Run `python models/classifier/train.py`")
        
        with col3:
            st.markdown("### 🔮 MotionFormer (Transformer)")
            predictor_history = history_dir / "predictor" / "weights" / "training_history.json"
            if predictor_history.exists():
                with open(predictor_history) as f:
                    history = json.load(f)
                st.json({"epochs_trained": len(history.get("train_loss", []))})
                st.success("✓ Trained")
            else:
                st.info("Not trained yet. Run `python models/predictor/train.py`")
    
    with tab3:
        st.header("⚙️ System Architecture")
        
        st.markdown("""
        The Selma Motion Engine consists of three optimized analytic components:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="model-card">
            <h3>🎯 PoseNet</h3>
            <h4>Stacked Hourglass Network</h4>
            <p><b>Purpose:</b> Pose Estimation</p>
            <p><b>Input:</b> 256×256 RGB Image</p>
            <p><b>Output:</b> 17 Keypoint Heatmaps (64×64)</p>
            <hr>
            <p><b>Architecture:</b></p>
            <ul>
            <li>2 Stacked Hourglass Modules</li>
            <li>256 Feature Channels</li>
            <li>Intermediate Supervision</li>
            <li>OHKM Loss for Hard Keypoints</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="model-card">
            <h3>🏃 MoveClassifier</h3>
            <h4>BiLSTM + Self-Attention</h4>
            <p><b>Purpose:</b> Movement Classification</p>
            <p><b>Input:</b> 30-Frame Keypoint Sequence</p>
            <p><b>Output:</b> 15 Movement Classes</p>
            <hr>
            <p><b>Architecture:</b></p>
            <ul>
            <li>Bidirectional LSTM (2 layers)</li>
            <li>Self-Attention (4 heads)</li>
            <li>Attention Pooling</li>
            <li>Torso-based Normalization</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="model-card">
            <h3>🔮 MotionFormer</h3>
            <h4>Encoder-Decoder Transformer</h4>
            <p><b>Purpose:</b> Motion Prediction</p>
            <p><b>Input:</b> 20 Past Frames</p>
            <p><b>Output:</b> 10 Future Frames</p>
            <hr>
            <p><b>Architecture:</b></p>
            <ul>
            <li>d_model=256, 8 Heads</li>
            <li>4 Encoder Layers</li>
            <li>4 Decoder Layers</li>
            <li>Learnable Positional Encoding</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Config details
        st.subheader("Configuration Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.json({
                "n_stacks": posenet_config.n_stacks,
                "n_features": posenet_config.n_features,
                "heatmap_size": posenet_config.heatmap_size,
                "num_keypoints": posenet_config.num_keypoints
            })
        
        with col2:
            st.json({
                "d_model": classifier_config.d_model,
                "n_layers": classifier_config.n_layers,
                "n_heads": classifier_config.n_heads,
                "sequence_length": classifier_config.sequence_length
            })
        
        with col3:
            st.json({
                "d_model": predictor_config.d_model,
                "n_heads": predictor_config.n_heads,
                "past_len": predictor_config.past_len,
                "future_len": predictor_config.future_len
            })
    
    with tab4:
        st.header("ℹ️ About")
        
        st.markdown("""
        ## Human Motion Intelligence System
        
        A complete AI system for real-time human motion analysis, built from scratch with:
        
        - **PyTorch** for deep learning models
        - **Streamlit** for interactive visualization  
        - **NumPy** for numerical operations
        - **Plotly** for interactive charts
        
        ### Key Features
        
        | Feature | Description |
        |---------|-------------|
        | 🎯 **Pose Estimation** | Extract 17 body keypoints from images using Stacked Hourglass Network |
        | 🏃 **Movement Classification** | Classify 15 different movement types using BiLSTM + Attention |
        | 🔮 **Motion Prediction** | Predict future motion trajectories using Transformer |
        
        ### Supported Movement Classes (15)
        """)
        
        # Display movement classes in a nice grid
        cols = st.columns(5)
        for i, cls in enumerate(MOVEMENT_CLASSES):
            with cols[i % 5]:
                st.markdown(f"- {cls.replace('_', ' ').title()}")
        
        st.markdown("""
        ### COCO 17 Keypoint Format
        
        | Index | Keypoint | Index | Keypoint |
        |-------|----------|-------|----------|
        | 0 | Nose | 9 | Left Wrist |
        | 1 | Left Eye | 10 | Right Wrist |
        | 2 | Right Eye | 11 | Left Hip |
        | 3 | Left Ear | 12 | Right Hip |
        | 4 | Right Ear | 13 | Left Knee |
        | 5 | Left Shoulder | 14 | Right Knee |
        | 6 | Right Shoulder | 15 | Left Ankle |
        | 7 | Left Elbow | 16 | Right Ankle |
        | 8 | Right Elbow | | |
        
        ### Quick Start
        
        ```bash
        # Train all models
        cd motion_ai
        python train_all.py
        
        # Run dashboard
        streamlit run app/streamlit_app.py
        ```
        """)


if __name__ == "__main__":
    main()
