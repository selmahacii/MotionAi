# 🏃 Motion Engine by Selma Haci

##  System Architecture

```mermaid
graph TD
    UI[Next.js Frontend UI] <--> API[Next.js API Routes]
    API <--> DB[(Prisma / SQLite)]
    API <--> P[Python Motion AI API]
    
    subgraph "Human Motion Intelligence Engine"
        P --> PN[PoseNet: Hourglass]
        PN --> MC[MoveClassifier: BiLSTM]
        MC --> MF[MotionFormer: Transformer]
    end
    
    subgraph "AI Output"
        PN --> KP[17 Body Keypoints]
        MC --> CL[15 Movement Classes]
        MF --> FD[10 Future Frames]
    end
```

##  AI Pipeline Details

```mermaid
graph LR
    I[Input Frame] --> PN[PoseNet]
    PN -->|Heatmaps| K[Keypoints]
    K --> BC[Biomechanic Processor]
    BC --> MC[MoveClassifier]
    MC -->|Sequence Analysis| CL[Movement Class]
    BC --> MF[MotionFormer]
    MF -->|Autoregressive| P[Predicted Motion]
```

##  PoseNet Architecture (Stacked Hourglass)

```mermaid
graph LR
    Input[256x256 Image] --> Stack1[Hourglass Stack 1]
    Stack1 --> IS1[Intermediate Supervision 1]
    IS1 --> Stack2[Hourglass Stack 2]
    Stack2 --> IS2[Intermediate Supervision 2]
    IS2 --> Output[17 Keypoint Heatmaps]
    
    subgraph "Hourglass Core"
        direction TB
        E[Encoder: Pool/Conv] --> B[Bottleneck]
        B --> D[Decoder: Upsample/Conv]
        E -.->|Skip Connection| D
    end
```

##  Database Schema

```mermaid
erDiagram
    USER {
        string id PK
        string email UK
        string name
        datetime createdAt
        datetime updatedAt
    }
    POST {
        string id PK
        string title
        string content
        boolean published
        string authorId FK
        datetime createdAt
        datetime updatedAt
    }
    USER ||--o{ POST : writes
```

## Model Parameters Summary

```mermaid
pie title "Approximate Model Parameter Distribution"
    "PoseNet (Hourglass)" : 14
    "MotionFormer (Transformer)" : 8
    "MoveClassifier (BiLSTM)" : 1.2
```

##  Deployment Structure

```mermaid
graph TB
    subgraph "Client Side"
        Browser[Web Browser]
    end
    
    subgraph "Server Side"
        Caddy[Caddy Reverse Proxy]
        Next[Next.js Application]
        Prisma[Prisma Client]
        SQLite[SQLite Database]
    end
    
    subgraph "Processing Side"
        Streamlit[Streamlit Dashboard]
        Python[Python Inference Engine]
    end
    
    Browser <--> Caddy
    Caddy <--> Next
    Next <--> Prisma
    Prisma <--> SQLite
    Next <--> Python
    Python <--> Streamlit
```
