# Search-and-Rescue-Swarm-using-Webots
Multi-robot search &amp; rescue simulation in Webots R2025a using two e-puck robots. Features two stages that show cooperative exploration, inter-robot communication, and rescue mode, demonstrating efficient navigation and coordination in unknown environments.

# Multi-Robot Search & Rescue Simulation (Webots R2025a)

## Overview

This project presents a multi-robot search and rescue (S&R) simulation developed using Webots R2025a. It demonstrates how a team of autonomous robots can collaboratively explore an unknown environment, detect objects, and build a shared map in real time.

The system is designed to highlight coordination, communication, and intelligent navigation strategies in a simulated disaster scenario.

## Key Features

* **Multi-Robot Cooperation**: Two e-puck robots (epuck0 and epuck1) work together to explore the environment efficiently.
* **Shared Mapping System**: Robots continuously exchange data to build and update a common occupancy map.
* **Search Strategy**: Implements structured exploration (e.g., priority-based wall following) instead of random movement.
* **Object Detection Integration**: Robots can detect and report objects within the environment.
* **Inter-Robot Communication**: Custom communication system for synchronization and data sharing.

## Technologies Used

* Webots R2025a
* Python
* NumPy

## Project Structure

* `controllers/` – Robot control logic
* `mapping/` – Cooperative mapping system
* `communication/` – Data exchange between robots
* `visualization/` – Real-time monitoring tools

## Use Cases

* Robotics research and education
* Multi-agent system simulation
* Search and rescue algorithm testing
* Autonomous exploration studies

## Getting Started

1. Install Webots R2025a
2. Clone this repository
3. Open the project world file in Webots
4. Run the simulation

## Future Improvements

* Scaling to more robots
* Improved path planning (e.g., SLAM integration)
* Advanced AI-based object recognition
* Dynamic obstacle handling

## Authors

Developed as a university graduation project by a team of Automation students at College of Ellectronic Technology in Libya.
Mhammed Hassan Al-Turkman.
Sofian Mohammed Al-sowayah.
Yahya Fuad Bin-Khalaf.
