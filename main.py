import numpy as np
import math

class ExtendedKalmanFilter:
    def __init__(self):
        # Initial state (position and velocity along x and y) [px, py, vx, vy]
        self.x = np.zeros(4)  
        
        # State covariance matrix
        self.P = np.eye(4) * 1000
        
        # Process noise covariance
        self.Q = np.eye(4) * 5  # Adjust as necessary

        # Measurement noise covariance for Lidar
        self.R_lidar = np.array([[0.1, 0],
                                  [0, 0.1]])

        # Measurement noise covariance for Radar
        self.R_radar = np.array([[0.3, 0, 0],
                                  [0, 0.03, 0],
                                  [0, 0, 0.3]])
    
    def compute_rho_dot(self, px, py, vx, vy):
        c1 = px**2 + py**2
        if c1 < 1e-6:
            return 0.0  # Avoid division by zero
        return (px * vx + py * vy) / math.sqrt(c1)

    def predict(self, dt):
        # State transition model
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        # Predict state
        self.x = F @ self.x
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update_lidar(self, meas):
        z = np.array(meas)
        z_pred = self.x[:2]  # Predicted position (px, py)

        # Measurement residual
        y = z - z_pred

        # Calculate Kalman gain
        S = self.R_lidar + self.P[:2, :2]
        K = self.P[:2, :2] @ np.linalg.inv(S)

        # Update state
        self.x[:2] += K @ y

        # Update covariance
        # Calculate the identity matrix and apply the update only to the relevant part
        I = np.eye(4)
        self.P[:2, :2] -= K @ S @ K.T  # Correctly update the position part of the covariance matrix


    def update_radar(self, meas):
        rho, phi, rho_dot = meas
        px, py, vx, vy = self.x

        # Convert radar measurement to Cartesian
        z_pred = np.array([
            math.sqrt(px**2 + py**2),
            math.atan2(py, px),
            self.compute_rho_dot(px, py, vx, vy)  # Use the method correctly
        ])

        # Measurement residual
        y = np.array([rho, phi, rho_dot]) - z_pred

        # Normalize the angle
        y[1] = self.normalize_angle(y[1])

        # Calculate Jacobian
        H_jacobian = self.jacobian_matrix(px, py, vx, vy)

        # Calculate Kalman gain
        S = H_jacobian @ self.P @ H_jacobian.T + self.R_radar
        K = self.P @ H_jacobian.T @ np.linalg.inv(S)

        # Update state
        self.x += K @ y

        # Update covariance
        self.P = (np.eye(4) - K @ H_jacobian) @ self.P

    def jacobian_matrix(self, px, py, vx, vy):
        c1 = px**2 + py**2
        if c1 < 1e-6:
            return np.zeros((3, 4))  # Return zero matrix if close to zero

        sqrt_c1 = math.sqrt(c1)
        c2 = c1 * sqrt_c1

        H_jacobian = np.array([[px / sqrt_c1, py / sqrt_c1, 0, 0],
                                [-py / c1, px / c1, 0, 0],
                                [py * (vx * py - vy * px) / c2, 
                                 px * (vy * px - vx * py) / c2, 
                                 px / sqrt_c1, py / sqrt_c1]])
        
        return H_jacobian

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split() for line in lines]

def write_results(file_path, results):
    with open(file_path, 'w') as f:
        for result in results:
            f.write(' '.join(map(str, result)) + '\n')

def calculate_error(est, gt):
    return [(est[i] - gt[i]) / gt[i] * 100 if gt[i] != 0 else 0 for i in range(len(est))]

def main(input_file, output_file):
    ekf = ExtendedKalmanFilter()
    results = []

    measurements = read_data(input_file)

    previous_timestamp = 0
    for measurement in measurements:
        sensor_type = measurement[0]
        if sensor_type == 'L':        # Lidar
            meas_px = float(measurement[1])
            meas_py = float(measurement[2])
            timestamp = float(measurement[3])
            gt = list(map(float, measurement[4:]))
            
            # Time difference
            dt = (timestamp - previous_timestamp) / 1e6
            ekf.predict(dt)
            ekf.update_lidar([meas_px, meas_py])
            previous_timestamp = timestamp

        elif sensor_type == 'R':
            # Radar
            meas_rho = float(measurement[1])
            meas_phi = float(measurement[2])
            meas_rho_dot = float(measurement[3])
            timestamp = float(measurement[4])
            gt = list(map(float, measurement[5:]))
            
            # Time difference
            dt = (timestamp - previous_timestamp) / 1e6
            ekf.predict(dt)
            ekf.update_radar([meas_rho, meas_phi, meas_rho_dot])
            previous_timestamp = timestamp

        # Collect results
        est_px, est_py, est_vx, est_vy = ekf.x
        results.append([est_px, est_py, est_vx, est_vy,
                        meas_px if sensor_type == 'L' else None,
                        meas_py if sensor_type == 'L' else None,
                        # meas_rho if sensor_type == 'R' else None,
                        # meas_phi if sensor_type == 'R' else None,
                        # meas_rho_dot if sensor_type == 'R' else None,
                        gt[0], gt[1], gt[2], gt[3]])

        # Calculating and printing error percentage
        if sensor_type == 'L':  # Only calculate for Lidar measurements
            est = [est_px, est_py, est_vx, est_vy]
            error_percentage = calculate_error(est, gt)
            print(f"Error % for Lidar: Position: ({error_percentage[0]:.2f}%, {error_percentage[1]:.2f}%), "
                  f"Velocity: ({error_percentage[2]:.2f}%, {error_percentage[3]:.2f}%)")

    write_results(output_file, results)



if __name__ == "__main__":
    input_file = "/Data/Input.txt"  # Input file path
    output_file = "results.txt"  # Output file for estimated results
    main(input_file, output_file)
