# Missile Intercept Simulation

A 3D simulation of a missile interception scenario, featuring realistic physics and visualization.

## Features

- Realistic physics simulation with gravity and drag
- 3D visualization using PyOpenGL
- Interactive camera controls
- Trajectory visualization
- Pause/Resume and Reset functionality

## Requirements

- Python 3.7+
- Pygame
- PyOpenGL
- NumPy

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Simulation

Run the simulation with:
```
python visualization.py
```

## Controls

- **Left-click + Drag**: Rotate camera
- **Right-click + Drag**: Zoom in/out
- **R**: Reset simulation
- **SPACE**: Pause/Resume simulation
- **ESC**: Quit

## How It Works

The simulation models two objects:
1. A target missile moving along a ballistic trajectory
2. An interceptor missile that tries to intercept the target

The physics engine calculates the motion of both objects, taking into account:
- Gravity
- Air resistance (drag)
- Interceptor guidance system

## Extensions

Potential extensions to this project:
- Add multiple interceptors
- Implement no-fly zones
- Add radar simulation with detection limitations
- Improve collision detection
- Add sound effects and more visual feedback
