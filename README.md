# AR/VR Career Roadmap
**Goal:** Enter the AR/VR field as an AI/Computer Vision Engineer  
**Stack:** Python · C++ · OpenCV · PyTorch · Unity/Unreal · AR SDKs

---

## How to Use This Roadmap

Each topic is marked with one of two tiers:

| Label | Meaning |
|---|---|
| 🟢 Tier 1 — Go Deep | Build real proficiency. Do the mini project. Study until you can explain and implement it without reference. |
| 🟡 Tier 2 — Awareness | Understand what it is, why it exists, and when it is used. Read about it, run existing code, but do not build from scratch. |

Tier 1 topics align directly with the **Perceptual AR** direction — object identification, generative AI integration, vision-language understanding, gesture, features and object recognition, and on-device deployment. Tier 2 topics provide useful context and keep your options open without demanding deep investment.

---

## Year 1 — Foundations

### 🟢 * Mathematics for AI and Graphics
- Linear algebra — vectors, matrices, dot/cross products
- Matrix transformations — rotation, translation, scaling
- Homogeneous coordinates and projection matrices
- Quaternions — understanding and applying to 3D rotations
- Calculus — derivatives, partial derivatives, chain rule
- Multivariable optimization — gradient descent, learning rate
- Probability — distributions, expectation, variance
- Bayes' theorem and conditional probability
- Statistics — mean, variance, covariance, correlation
- 3D coordinate systems — world, camera, and screen space

> **Mini Project 1:** Build a Python script that applies 2D and 3D transformation matrices (rotate, scale, translate) to a set of points and visualizes the result using matplotlib.

---

### 🟢 * Python for Scientific Computing
- NumPy — arrays, broadcasting, linear algebra ops
- pandas — data loading, cleaning, manipulation
- matplotlib and seaborn — visualization and plotting
- SciPy — optimization, interpolation, signal processing
- Jupyter Notebooks — documentation and experimentation workflow
- Python packaging — virtual environments, pip, requirements.txt

> **Mini Project 2:** Load a dataset, perform exploratory data analysis (EDA), apply a transformation pipeline, and produce annotated plots summarizing the findings.

---

### 🟡 * C++ Basics for AR/CV Work
- Compilation model — headers, source files, linking
- Pointers and memory management
- Classes and objects — constructors, destructors
- STL containers — vector, map, unordered_map
- References and const correctness
- Basic CMake project setup

> **Mini Project 3 (optional):** Implement a simple image struct in C++ that stores pixel data in a 2D array and supports basic operations — invert, crop, and threshold — compiled with CMake.

---

### 🟢 * Machine Learning Fundamentals
- Supervised vs unsupervised vs reinforcement learning
- Linear regression and logistic regression from scratch
- Decision trees and random forests
- Support vector machines (SVM)
- k-Nearest Neighbors (kNN)
- Feature engineering and normalization
- Train/validation/test split
- Overfitting, underfitting, bias-variance tradeoff
- Cross-validation and hyperparameter tuning
- Scikit-learn workflow — fit, predict, evaluate

> **Mini Project 4:** Train and evaluate three classifiers (SVM, Random Forest, kNN) on a real dataset using scikit-learn. Compare accuracy, precision, recall, and F1 score. Visualize decision boundaries for a 2D subset.

---

### 🟢 * Computer Vision Introduction
- How digital images work — pixels, channels, color spaces (RGB, HSV, grayscale)
- Image read, write, and display with OpenCV
- Geometric transformations — resize, rotate, flip, warp
- Filtering — Gaussian blur, median filter, sharpening
- Edge detection — Sobel, Canny
- Thresholding and morphological operations — erosion, dilation
- Contour detection and shape analysis
- Color-based object segmentation in HSV space
- Histograms and histogram equalization

> **Mini Project 5:** Build a real-time color-based object tracker using OpenCV and a webcam. The program detects a colored object, draws a bounding box, and tracks its position across frames.

---

### 🟢 * Version Control and Development Workflow
- Git fundamentals — init, add, commit, push, pull
- Branching and merging
- GitHub — repository management, README writing
- .gitignore and managing binary files
- Commit discipline — meaningful messages and atomic commits

> **Mini Project 6:** Publish all Year 1 projects to a structured GitHub repository with individual READMEs, setup instructions, and a top-level portfolio README. 

---

### 🟢 * AR Exploration — Getting Started
- What is AR — marker-based, markerless, projection-based
- Overview of AR hardware — HoloLens 2, Meta Quest, ARCore phones
- Install Unity and set up an AR Foundation project
- ARCore (Android) or ARKit (iOS) SDK setup
- Understand the AR development loop — build, deploy, test on device
- Surface detection, raycasting, and object placement

> **Mini Project 7:** Create a basic AR app in Unity that detects a flat surface and lets the user tap to place a 3D model. Export and run it on a real device.

---

## Year 2 — Intermediate AI and Vision

### 🟢 * Neural Networks from the Ground Up
- Perceptron and multilayer perceptron (MLP)
- Activation functions — ReLU, sigmoid, tanh, GELU
- Forward pass and loss computation
- Backpropagation — deriving and implementing manually
- Weight initialization — Xavier, He
- Optimizers — SGD, Adam, RMSProp
- Batch, mini-batch, and stochastic gradient descent
- Regularization — L1, L2, dropout
- Batch normalization and layer normalization
- Learning rate schedules — step decay, cosine annealing

> **Mini Project 8:** Implement a fully connected neural network using only NumPy — no frameworks. Train it on MNIST and reach at least 95% test accuracy. Plot training and validation loss curves.

---

### 🟢 * PyTorch Core
- Tensors — creation, indexing, operations, GPU transfer
- Autograd — computational graph and gradient computation
- nn.Module — defining custom layers and models
- DataLoader and Dataset — loading and batching custom data
- Training loop — forward pass, loss, backward, optimizer step
- Model checkpointing — saving and loading state_dict
- torchvision — transforms, datasets, pre-trained models
- GPU training with CUDA

> **Mini Project 9:** Reproduce Mini Project 8 using PyTorch. Extend it to train on CIFAR-10 using a deeper MLP. Log training metrics and visualize weight histograms per epoch.

---

### 🟢 * Convolutional Neural Networks (CNNs)
- Convolution operation — kernel, stride, padding, dilation
- Pooling — max pooling, average pooling, global average pooling
- Classic architectures — LeNet, AlexNet, VGG, ResNet, EfficientNet
- Residual connections and skip connections
- Feature maps and receptive field
- Transfer learning — feature extraction vs fine-tuning
- Data augmentation — flips, crops, color jitter, mixup
- Class activation maps (CAM) for interpretability

> **Mini Project 10:** Fine-tune a pre-trained ResNet-18 on a custom 5-class image dataset you collect yourself (e.g., five household objects). Achieve at least 90% validation accuracy. Visualize CAM heatmaps for each class.

---

### 🟡 * Feature Detection and Matching
- Keypoint detectors — SIFT, ORB, FAST, AKAZE
- Descriptors and descriptor matching — brute-force, FLANN
- Homography estimation — RANSAC
- Image stitching and panorama construction
- Template matching

> *(Awareness only — read and run existing examples. Skip the mini project unless time allows.)*

---

### 🟡 * Camera Models and Calibration
- Pinhole camera model — intrinsic matrix, focal length, principal point
- Lens distortion — radial and tangential
- Camera calibration using a checkerboard
- Undistortion and rectification
- Stereo camera geometry — baseline, disparity, depth
- Epipolar geometry — epipoles, epipolar lines, essential and fundamental matrix

> *(Awareness only — understand the concepts. Run the OpenCV calibration demo once to see it in practice.)*

---

### 🟡 * Introduction to 3D Vision
- Depth from stereo — disparity maps and depth maps
- Monocular depth estimation — learned methods overview
- Point clouds — representation and visualization with Open3D
- 3D transformations — rigid body motions, SO(3), SE(3)
- Coordinate frame changes — sensor to world transforms

> *(Awareness only — understand depth maps and point clouds conceptually. You will encounter them in SLAM and reconstruction contexts.)*

---

### 🟢 * Intermediate AR Development
- Plane detection and spatial anchors in AR Foundation
- Marker-based AR — image targets with Vuforia or AR Foundation
- Raycasting in AR — placing objects at hit points
- Light estimation for realistic rendering
- Occlusion handling basics
- AR session lifecycle management

> **Mini Project 11:** Build a marker-based AR app that detects a printed image target and overlays an animated 3D model on top of it. Add basic lighting estimation so the model reacts to real-world illumination.

---

## Year 3 — Applied Computer Vision for AR

### 🟢 * Object Detection
- Detection paradigms — one-stage vs two-stage
- Anchor boxes, IoU, non-maximum suppression (NMS)
- Faster R-CNN — region proposal network, ROI pooling
- YOLO series — YOLOv5, YOLOv8 architecture and custom training
- DETR — transformer-based end-to-end detection
- Evaluation metrics — mAP, precision-recall curve
- Dataset labeling — LabelImg or Roboflow
- Data augmentation for detection — mosaic, flips, scaling

> **Mini Project 12:** Train YOLOv8 on a custom-labeled dataset of 5–10 object classes. Deploy it with OpenCV for real-time inference on a webcam feed. Display class labels, confidence scores, and bounding boxes.

---

### 🟢 * Image Segmentation
- Semantic segmentation — pixel-wise class labels
- Instance segmentation — per-object masks
- U-Net — encoder-decoder architecture with skip connections
- Mask R-CNN — combining detection and segmentation
- Segment Anything Model (SAM) — prompting with points and boxes
- DeepLabV3+ — atrous convolution and ASPP
- Evaluation — mIoU, Dice coefficient

> **Mini Project 13:** Use SAM to segment objects in a live camera feed interactively. Allow the user to click on an object to generate its mask. Overlay a colored mask on the original frame in real time.

---

### 🟡 * SLAM — Simultaneous Localization and Mapping
- SLAM problem definition — pose estimation and map building
- Visual odometry — estimating camera motion from image sequences
- Feature-based SLAM — ORB-SLAM3 overview and setup
- Dense vs sparse maps
- Loop closure detection
- Pose graphs and graph optimization
- Introduction to LiDAR SLAM concepts

> *(Awareness only — understand what SLAM solves and why AR devices use it. Run ORB-SLAM3 on a sample video once to observe the output. Do not build from scratch.)*

---

### 🟢 * Pose Estimation
- 2D pose estimation — heatmap-based keypoint detection
- HRNet and OpenPose architectures
- 3D pose estimation — lifting 2D keypoints to 3D
- Hand pose — MediaPipe Hands pipeline and joint landmarks
- Full body — MediaPipe Holistic
- Head pose estimation — solving the PnP problem
- Pose-driven AR interaction — mapping skeleton joints to virtual controls

> **Mini Project 14:** Build a gesture-controlled AR interface using MediaPipe Hands. Define 3–5 hand gestures (e.g., pinch, open palm, point) and map each to an AR action such as scale, rotate, or place an object.

---

### 🟢 * Sign Language Recognition
- Sign language datasets — ASL, FSL, and others
- Hand landmark extraction using MediaPipe
- Sequence modeling for dynamic signs — LSTM, GRU, Transformer
- Static vs dynamic sign classification
- Real-time inference pipeline — webcam to text output
- Handling signer variation and lighting conditions
- Evaluation — per-class accuracy, confusion matrix

> **Mini Project 15:** Build a real-time sign language recognition system using MediaPipe Hands for landmark extraction and an LSTM for sequence classification. Display the recognized word or letter as text overlay on the camera feed.

---

### 🟡 * Optical Flow and Motion Tracking
- Optical flow definition — apparent motion field
- Lucas-Kanade sparse optical flow
- Farneback dense optical flow
- Motion segmentation — separating moving objects from background
- Multi-object tracking — DeepSORT, ByteTrack
- Kalman filter for trajectory smoothing

> *(Awareness only — understand optical flow and how trackers like DeepSORT work. You will use ByteTrack as a library in later projects without needing to implement it.)*

---

### 🟡 * 3D Reconstruction
- Structure from Motion (SfM) pipeline — feature match, pose estimation, triangulation
- Multi-view stereo (MVS)
- Neural Radiance Fields (NeRF) — concept, volume rendering, training
- Gaussian Splatting — 3D Gaussian representation and real-time rendering
- COLMAP for photogrammetry
- Mesh processing with Open3D — smoothing, simplification, export

> *(Awareness only — understand NeRF and Gaussian Splatting conceptually. Run a pre-trained demo to observe the output. Relevant if you later move toward neural rendering, but not a priority for the assistant track.)*

---

### 🟢 * AR with AI Integration — Year 3 Milestone App
- Integrating a PyTorch model into Unity via Barracuda or a REST API
- Real-time inference pipeline — capture, preprocess, infer, overlay
- Latency profiling and optimization
- Async inference to avoid blocking the main AR thread
- Overlaying labels, bounding boxes, and 3D anchors in AR

> **Milestone Project:** Build an AR object recognition app that runs real-time YOLOv8 detection on the camera feed, identifies objects in the scene, and overlays 3D information panels anchored to each detected object. Deploy on a mobile device.

---

## Year 4 — Specialization and Capstone

### 🟢 * Transformers in Computer Vision
- Attention mechanism — scaled dot-product, multi-head attention
- Vision Transformer (ViT) — patch embeddings, position encoding
- Swin Transformer — hierarchical windowed attention
- DINO and DINOv2 — self-supervised vision features
- CLIP — contrastive language-image pre-training
- Grounded vision-language models — Grounding DINO, LLaVA
- Applying vision-language models to AR scene description

> **Mini Project 16:** Use CLIP to build an AR scene query system. The user speaks an object name, the system uses CLIP to locate the best-matching region in the live frame, and places an AR highlight overlay on it.

---

### 🟢 * Vision-Language Models and Generative AI Integration
- Large vision-language models (VLMs) — GPT-4V, LLaVA, InstructBLIP
- Prompt engineering for visual question answering
- Retrieval-augmented generation (RAG) for knowledge grounding
- Object identification with detail generation — name, description, usage
- OCR integration — reading text from real-world surfaces
- On-device vs API-based inference trade-offs
- Chaining vision and language outputs into a coherent AR response

> **Mini Project 17:** Build an AR object inspector. Point the camera at any object, the system detects it with YOLO, crops the region, sends it to a VLM API, and overlays a generated description panel in AR — name, what it is, and any relevant details.

---

### 🟡 * Advanced Detection and Tracking
- Real-time multi-object tracking at scale
- Re-identification (ReID) — tracking across camera cuts
- 3D object detection — PointPillars, BEV detection methods
- Panoptic segmentation — combining semantic and instance outputs
- Open-vocabulary detection — OWL-ViT, Grounding DINO
- Sparse convolutions for 3D data — spconv

> *(Awareness only — understand open-vocabulary detection and panoptic segmentation. Use Grounding DINO as a library. Deep implementation of 3D detection is not a priority.)*

---

### 🟢 * Lightweight Models and Edge Deployment
- Model compression — quantization (INT8, FP16), pruning, knowledge distillation
- TensorFlow Lite — conversion, optimization, benchmarking
- ONNX — model interchange format and ONNX Runtime
- Core ML — deploying models on iOS devices
- TensorRT — NVIDIA edge GPU acceleration
- TinyML — inference on microcontrollers
- Benchmarking — latency, throughput, memory footprint

> **Mini Project 18:** Take a trained YOLOv8 model and convert it to TFLite with INT8 quantization. Deploy it on a mobile device. Benchmark FPS, latency, and accuracy degradation versus the full-precision model.

---

### 🟡 * Neural Rendering and AR Visuals
- NeRF in depth — volume rendering, positional encoding, training pipeline
- Instant-NGP — fast hash-encoded NeRF training
- 3D Gaussian Splatting — training pipeline and real-time viewer
- Neural rendering for AR — inserting synthetic objects into real scenes
- Relighting and material estimation
- Diffusion models for AR asset and texture generation

> *(Awareness only — understand the landscape. Run a Gaussian Splatting demo on your own data once. Relevant if you expand toward the rendering track later.)*

---

### 🟢 * Multi-Modal Learning
- Fusion strategies — early, late, and cross-attention fusion
- Vision-language models — BLIP-2, InstructBLIP
- Speech-to-text integration — Whisper for on-device voice input
- Text-to-speech for AR voice output
- Language-conditioned object detection and grounding
- Chaining modalities — voice in, vision process, AR overlay out

> **Mini Project 19:** Build a voice-activated AR labeling system. The user speaks an object name, the system runs Grounding DINO to locate it in the frame, and places a labeled AR anchor on the detected object. Add Whisper for on-device speech-to-text input.

---

### 🟢 * Capstone Project

Choose one and develop it to a production-ready, documented, and deployable state.

#### Option A — Perceptual AR Assistant
- Real-time object detection and identification using YOLO + VLM
- Generative AI overlay — name, description, usage, and fun facts
- Voice query interface using Whisper for speech-to-text
- OCR for reading text on surfaces, signs, and labels
- Sign language recognition module for accessibility

#### Option B — AR Sign Language Translator
- Real-time hand landmark extraction with MediaPipe
- Sequence classification for dynamic signs using a Transformer model
- Live translation output overlaid as text in AR
- Support for multiple sign vocabularies
- Confidence display and fallback handling for ambiguous signs

#### Option C — AR Accessibility Assistant
- Real-time object and text recognition (OCR) at high accuracy
- Natural language scene description using a vision-language model
- Spatial audio feedback — sound cues mapped to object positions
- Hands-free gesture control for navigation and query
- Voice output for identified objects and scene context

#### Option D — AI-Powered AR Navigation Assistant
- Real-time obstacle and object detection in the camera stream
- Depth estimation for distance measurement to obstacles
- Path planning with AR directional overlays
- Voice command interface using Whisper
- Context-aware scene understanding (e.g., "door ahead on the left")

> **Deliverables (all options)**
> - [ ] Working application deployable on a real device
> - [ ] Clean, documented codebase on GitHub
> - [ ] Written technical report explaining design decisions
> - [ ] Video demonstration (3–5 minutes)
> - [ ] Slide deck suitable for a portfolio or job interview presentation

---

## Ongoing — Throughout All Years

### 🟢 * Git and Portfolio Hygiene
- Maintain a clean GitHub profile with pinned repositories
- Write READMEs with demo GIFs, setup instructions, and usage examples
- Use semantic commit messages and branch-per-feature workflow
- Tag releases for major project milestones

### 🟢 * Technical Writing
- Document every Tier 1 mini project with a short write-up — problem, approach, and result
- Publish 2–3 technical blog posts per year on your work
- Keep a personal learning journal tracking what you studied and what you built

### 🟢 * Open Source Contribution
- Year 2 — file issues or improve documentation on a CV or ML repository
- Year 3 — submit a pull request with a bug fix or small feature
- Year 4 — contribute meaningfully to an AR, VLM, or sign language open-source project

### 🟢 * Personal Website
- Year 1 — set up a portfolio site with a project list
- Year 2 — add demo videos and project write-ups
- Year 3 — add a blog section with technical articles
- Year 4 — polish everything and prepare for job applications

---

**Last Updated:** 2026-03-15