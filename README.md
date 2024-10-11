# Ground Bot Navigation Using Extended Kalman Filter (EKF)

## Overview
This project implements an Extended Kalman Filter (EKF) for a Ground Bot navigating through unknown territory. The EKF combines measurements from Lidar and Radar sensors to estimate the bot's position and velocity in real-time. The estimated values are compared with ground truth data to evaluate the accuracy of the navigation system.

## Features
- **Data Parsing**: Reads sensor measurements and ground truth values from an input file.
- **EKF Implementation**: Predicts and updates the state of the bot using sensor data.
- **Result Comparison**: Outputs the estimated positions and velocities alongside the measurements and ground truth data.

## Input and Output Format
### Input File Format
The input file contains sensor measurements and ground truth values in the following format:
- **Lidar Measurement (L)**: `L meas_px meas_py timestamp gt_px gt_py gt_vx gt_vy`
- **Radar Measurement (R)**: `R meas_rho meas_phi meas_rho_dot timestamp gt_px gt_py gt_vx gt_vy`

**Example**:
```
R 8.60363 0.0290616 -2.99903 1477010443399637 8.6 0.25 -3.00029 0
L 8.45 0.25 1477010443349642 8.45 0.25 -3.00027 0
```

### Output File Format
The output file will contain the estimated positions and velocities along with the measurements and ground truth values:
```
est_px est_py est_vx est_vy meas_px meas_py gt_px gt_py gt_vx gt_vy
```

**Example**:
```
4.53271 0.279 -0.842172 53.1339 4.29136 0.215312 2.28434 0.226323
```
When collecting results, placeholders (None) are used for Lidar values in Radar entries to maintain consistent formatting.

## Requirements

This project requires Python 3.10.12 and the following libraries:

    numpy
    math

You can install the required library using:

   ```bash
    pip install numpy
    pip install math
   ```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/dej0y/Extended-Kalman-Filter-GroundBot
   cd Extended-Kalman-Filter-GroundBot
   ```

2. Place your input file (e.g., `Input.txt`) in the Data directory.

3. Run the program:
   ```bash
   python3 main.py
   ```

4. The output will be saved in a file named `results.txt`.
