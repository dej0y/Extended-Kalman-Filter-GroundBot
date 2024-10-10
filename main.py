import numpy as np
import math

class ExtendedKalmanFilter:
    def __init__(self):
        self.x = np.zeros(4)  # Initial state [px, py, vx, vy]
        self.P = np.eye(4) * 1000  # State covariance matrix
        self.Q = np.eye(4) * 9  # Process noise covariance

        # Measurement noise covariance for Lidar
        self.R_lidar = np.array([[2, 0],
                                  [0, 5]])

        # Measurement noise covariance for Radar
        self.R_radar = np.array([[0.3, 0, 0],
                                  [0, 0.9, 0],
                                  [0, 0, 0.3]])
    
    def compute_rho_dot(self, px, py, vx, vy):
        c1 = px**2 + py**2
        if c1 < 1e-6:
            return 0.0  # Avoid division by zero
        return (px * vx + py * vy) / math.sqrt(c1)

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update_lidar(self, meas):
        if len(meas) != 2:
            print("Invalid Lidar measurement; expected 2 values.")
            return

        z = np.array(meas)
        z_pred = self.x[:2]  # Predicted position

        y = z - z_pred  # Measurement residual

        S = self.R_lidar + self.P[:2, :2]
        K = self.P[:2, :2] @ np.linalg.inv(S)

        self.x[:2] += K @ y

        I = np.eye(4)
        self.P[:2, :2] -= K @ S @ K.T

    def update_radar(self, meas):
        if len(meas) != 3:
            print("Invalid Radar measurement; expected 3 values.")
            return
        
        rho, phi, rho_dot = meas
        px, py, vx, vy = self.x

        z_pred = np.array([
            math.sqrt(px**2 + py**2),
            math.atan2(py, px),
            self.compute_rho_dot(px, py, vx, vy)
        ])

        y = np.array([rho, phi, rho_dot]) - z_pred
        y[1] = self.normalize_angle(y[1])  # Normalize angle

        H_jacobian = self.jacobian_matrix(px, py, vx, vy)

        S = H_jacobian @ self.P @ H_jacobian.T + self.R_radar
        K = self.P @ H_jacobian.T @ np.linalg.inv(S)

        self.x += K @ y
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
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [line.strip().split() for line in lines]
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []

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
        if len(measurement) < 5:  # Check for sufficient data
            print("Insufficient measurement data; skipping this entry.")
            continue

        sensor_type = measurement[0]
        timestamp = float(measurement[-1])
        
        # Handle time difference
        dt = (timestamp - previous_timestamp) / 1e6 if previous_timestamp else 0
        previous_timestamp = timestamp

        if sensor_type == 'L':  # Lidar
            if len(measurement) < 6:
                print("Incomplete Lidar measurement; skipping.")
                continue
            meas_px = float(measurement[1])
            meas_py = float(measurement[2])
            gt = list(map(float, measurement[4:]))
            ekf.predict(dt)
            ekf.update_lidar([meas_px, meas_py])

            # Collect results
            est_px, est_py, est_vx, est_vy = ekf.x
            results.append([est_px, est_py, est_vx, est_vy,
                            meas_px, meas_py,
                            gt[0], gt[1], gt[2], gt[3]])

            # Calculating and printing error percentage for Lidar
            est = [est_px, est_py, est_vx, est_vy]
            error_percentage = calculate_error(est, gt)
            print(f"Error % for Lidar: Position: ({error_percentage[0]:.2f}%, {error_percentage[1]:.2f}%), "
                  f"Velocity: ({error_percentage[2]:.2f}%, {error_percentage[3]:.2f}%)")

        elif sensor_type == 'R':  # Radar
            if len(measurement) < 8:
                print("Incomplete Radar measurement; skipping.")
                continue
            meas_rho = float(measurement[1])
            meas_phi = float(measurement[2])
            meas_rho_dot = float(measurement[3])
            gt = list(map(float, measurement[5:]))
            ekf.predict(dt)
            ekf.update_radar([meas_rho, meas_phi, meas_rho_dot])

            # Collect results
            est_px, est_py, est_vx, est_vy = ekf.x
            results.append([est_px, est_py, est_vx, est_vy,
                            None, None,  # Placeholder for Lidar values
                            gt[0], gt[1], gt[2], gt[3]])

            # Calculating and printing error percentage for Radar
            est = [est_px, est_py, est_vx, est_vy]
            error_percentage = calculate_error(est, gt)
            print(f"Error % for Radar: Position: ({error_percentage[0]:.2f}%, {error_percentage[1]:.2f}%), "
                  f"Velocity: ({error_percentage[2]:.2f}%, {error_percentage[3]:.2f}%)")

    write_results(output_file, results)

if __name__ == "__main__":
    input_file = "Data/Input.txt"  # Input file path
    output_file = "results.txt"  # Output file for estimated results
    main(input_file, output_file)
