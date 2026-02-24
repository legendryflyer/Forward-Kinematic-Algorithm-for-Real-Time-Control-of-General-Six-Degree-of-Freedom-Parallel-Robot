# Forward Kinematic Algorithm for Real-Time Control of General Six Degree of Freedom Parallel Robot

This repository contains the implementation and data for a forward kinematics solution designed for a general 6-DOF parallel robot. Forward kinematics refers to computing the end-effector pose (position and orientation) from known joint values. In parallel robots like the Stewart platform, this problem is mathematically more complex than in serial manipulators. :contentReference[oaicite:0]{index=0}

## Overview

A **parallel robot** connects a moving platform to a fixed base using multiple kinematic chains. This architecture offers high stiffness, precision, and dynamic performance. However, calculating the end-effector pose given actuator inputs (forward kinematics) can be nontrivial due to closed-loop geometry. :contentReference[oaicite:1]{index=1}

This project provides:

- Python scripts for data handling and model inference
- Training and testing code for kinematic estimation
- Example datasets and model artifacts
- Utilities for performance evaluation

## Features

- Real-time forward kinematics computation
- Data-driven or model-based approach
- Scripts to train, test, and validate performance
- Works with general parallel robots having six degrees of freedom

## File Structure

```plaintext
.
├── LICENSE                          # MIT License
├── arm_min_max.txt                  # Range of arm joint values
├── arm_stats.py                     # Statistics and preprocessing
├── data.py                          # Dataset loader
├── fk_inference_and_test.py         # Inference & evaluation
├── train_fk_ai.py                  # Training script
├── fk_model_best.pth                # Best trained model
├── fk_model_full_best.pth           # Full best model
├── x_scaler.pkl                     # Input scaler
├── y_scaler.pkl                     # Output scaler
└── other utils / test files         # Extra scripts and test data
