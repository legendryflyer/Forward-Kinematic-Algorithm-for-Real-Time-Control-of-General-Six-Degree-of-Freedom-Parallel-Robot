# Forward-Kinematic-Algorithm-for-Real-Time-Control-of-General-Six-Degree-of-Freedom-Parallel-Robot

This repository implements an **AI-based Forward Kinematics (FK) system** for a general 6-DOF parallel robot (Stewart Platform). 



Instead of using traditional mathematical forward kinematics equations and numerical solvers, this system uses **Deep Learning** to learn the forward kinematic mapping directly from data.

## 📌 Project Overview
A Stewart platform consists of 6 arms (legs) and a moving platform with complex, highly nonlinear kinematics. 

### The Challenge with Traditional FK
Traditional forward kinematics requires:
* **Complex Equations:** Transcendental equations that are hard to solve analytically.
* **Iterative Solvers:** Methods like Newton-Raphson which are computationally expensive.
* **Jacobian Matrices:** Requires constant recalculation of the robot's Jacobian.
* **Stability Issues:** Risk of divergence, numerical instability, and sensitivity to modeling errors or sensor noise.

### The AI Solution
This project replaces the entire numerical pipeline with a neural network that learns the physical behavior of the robot:
**6 Arm Lengths → Neural Network → X, Y, Z, Roll, Pitch, Yaw**

---

## 🎯 Goals
* **Real-time Forward Kinematics:** Instantaneous pose estimation for high-speed control.
* **Sub-millimeter Accuracy:** Translational error $< 0.1$ mm.
* **High Precision Orientation:** Rotation estimation error $< 0.1$ degree.
* **Numerical Stability:** No divergence or "solver failed" states.
* **Hardware Tolerance:** Ability to learn real-world imperfections like backlash, compliance, and leg flexibility.
* **Foundation:** Readiness for Inverse Kinematics (IK) and Digital Twin integration.

---

