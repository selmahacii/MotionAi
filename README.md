# Selma Motion Engine (SME)

Proprietary Engineering by Selma Haci

---

## Technical System Flow

```mermaid
graph TD
    subgraph "Data Acquisition"
        I[Input Stream] --> V[Visual Processing]
        V --> N[Normalization]
    end
    
    subgraph "Analytic Core (SME Engine)"
        N --> PE[Pose Estimation - Stage 1]
        PE --> PC[Activity Classification - Stage 2]
        PC --> FM[Temporal Prediction - Stage 3]
    end
    
    subgraph "Output Distribution"
        FM --> UI[Client Portal]
        PC --> DB[Secure Analytics Store]
        PE --> DP[Data Packet Serializer]
    end
```

---

## Core Engine Hierarchy

```mermaid
classDiagram
    class BaseEngine {
        <<Abstract>>
        +process_frame(frame) DataPacket
        +reset()
        +get_diagnostics() Dict
    }
    class AnalyticEngine {
        -SME_PoseEstimator
        -SME_ActivityClassifier
        -SME_TemporalPredictor
        +process_frame(frame) DataPacket
    }
    class SimulatedEngine {
        +process_frame(stub) DataPacket
    }
    BaseEngine <|-- AnalyticEngine
    BaseEngine <|-- SimulatedEngine
```

---

## Neural Processing Pipeline

```mermaid
sequenceDiagram
    participant V as Video Input
    participant S1 as Stage 1: PoseNet
    participant S2 as Stage 2: MoveClassifier
    participant S3 as Stage 3: MotionFormer
    participant O as SME DataPacket

    V->>S1: Raw Frame (RGB)
    S1->>S1: Heatmap Extraction
    S1->>S2: Anatomical Coordinates (17 Keypoints)
    S2->>S2: Temporal Self-Attention
    S2->>S3: Motion Profile + Features
    S3->>S3: Autoregressive Forecast
    S3->>O: Consolidated Analytic Packet
```

---

© 2026 Selma Haci. Proprietary Engineering.
