# Dyn_Sim_DS - Dynamic Simulation with Different RRT Algorithms

This project implements various Rapidly-exploring Random Tree (RRT) algorithms for robot motion planning in a dynamic environment with moving obstacles.

## Installation

### 1. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 2. Install required packages

```bash
pip install numpy
pip install pybullet
pip install scipy
```

## Project Structure

The project contains several Python files implementing different RRT algorithms:

### 1. `sim.py`
- Core simulation environment using PyBullet
- Implements the `PyBulletSim` class for robot and environment simulation
- Handles robot control, collision detection, and moving obstacles
- Provides visualization tools and utility functions

### 2. `rrt.py`
- Basic RRT (Rapidly-exploring Random Tree) implementation
- Single-directional tree growth from start to goal
- Handles static and moving obstacle collision checking
- Includes visualization of the search tree and final path

### 3. `rrt_apf.py`
- RRT with Artificial Potential Field (APF) integration
- Uses attractive forces towards goal and repulsive forces from obstacles
- Improved path planning with obstacle avoidance
- Combines RRT's exploration with APF's obstacle avoidance capabilities

### 4. `bi_rrt.py`
- Bidirectional RRT implementation
- Grows trees from both start and goal configurations
- Faster convergence compared to basic RRT
- Better handling of narrow passages

### 5. `bi_rrt_apf.py`
- Bidirectional RRT with APF integration
- Combines benefits of bidirectional search and APF
- Most sophisticated implementation with best performance
- Handles complex dynamic environments effectively

## Usage

To run any of the RRT implementations:

```bash
# Basic RRT
python rrt.py

# RRT with APF
python rrt_apf.py

# Bidirectional RRT
python bi_rrt.py

# Bidirectional RRT with APF
python bi_rrt_apf.py
```

## Features

- Dynamic obstacle avoidance
- Real-time visualization
- Multiple RRT algorithm implementations
- Robot arm control and path planning
- Collision detection and prevention
- Gripper control for object manipulation

## Requirements

- Python 3.10
- PyBullet
- NumPy
- SciPy

## Note

Make sure to have the required asset files in the correct directories:
- Robot models in `assets/doosan/`
- Object models in `assets/objects/`
- Obstacle models in `assets/obstacles/`
- Gripper model in `assets/gripper/`