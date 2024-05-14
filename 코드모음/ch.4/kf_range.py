import numpy as np
import matplotlib.pyplot as plt

# time step
dt = 1

# process and measurement noise
Qc = 150
Q = Qc * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
R = 30

# state initialization
x0 = np.array([[2000], [10]])  # 수정: 2x1 열 벡터로 초기화
P0 = 30 * np.eye(2)

# KF initialization
xbar = x0 + np.sqrt(P0) @ np.random.randn(2, 1)
Pbar = P0
Qf = Q

# collections
X = []
XHAT = []
PHAT = []
KK = []
Z = []
ZHAT = []
SBAR = []
TIME = []

# system matrix
F = np.array([[1, dt], [0, 1]])
H = np.array([[1, 0]])

for time in range(101):
    
    # measurement
    z = H @ x0 + np.sqrt(R) * np.random.randn(1)  # 수정: np.random.randn(1,2) -> np.random.randn(1)
    
    # Measurement Update (MU)
    zhat = H @ xbar
    S = H @ Pbar @ H.T + R
    Phat = Pbar - Pbar @ H.T @ np.linalg.inv(S) @ H @ Pbar
    K = Pbar @ H.T @ np.linalg.inv(S)
    xhat = xbar + K @ (z - zhat)
    
    # Time Update (TU)
    xbar = F @ xhat
    Pbar = F @ Phat @ F.T + Qf
    
    # system dynamics
    x = F @ x0 + np.sqrt(Q) @ np.random.randn(2, 1)
       
    # collections
    X.append(x0.flatten())  # 누적할 때는 flatten() 사용하여 2D 배열을 1D로 변환
    XHAT.append(xhat.flatten())
    PHAT.append(np.diag(Phat))
    Z.append(z)
    ZHAT.append(zhat)
    SBAR.append(np.diag(S))
    TIME.append(time)
    KK.append(K.flatten())
    
    # for next step
    x0 = x

# converting lists to numpy arrays for plotting
X = np.array(X)
XHAT = np.array(XHAT)
PHAT = np.array(PHAT)
Z = np.array(Z)
ZHAT = np.array(ZHAT)
SBAR = np.array(SBAR)
TIME = np.array(TIME)
KK = np.array(KK)

# plotting
plt.figure()
plt.plot(TIME, X[:, 0], 'r', label='x1 True')
plt.plot(TIME, XHAT[:, 0], 'b', label='x1 Estimate')
plt.title('x1')
plt.legend()

plt.figure()
plt.plot(TIME, X[:, 1], 'r', label='x2 True')
plt.plot(TIME, XHAT[:, 1], 'b', label='x2 Estimate')
plt.title('x2')
plt.legend()

plt.figure()
plt.plot(TIME, X[:, 0] - XHAT[:, 0], 'r', label='x1 Error')
plt.plot(TIME, np.sqrt(PHAT[:, 0]), 'b', label='x1 Error Bound')
plt.plot(TIME, -np.sqrt(PHAT[:, 0]), 'b')
plt.title('x1 Error and Bound')
plt.legend()

plt.figure()
plt.plot(TIME, X[:, 1] - XHAT[:, 1], 'r', label='x2 Error')
plt.plot(TIME, np.sqrt(PHAT[:, 1]), 'b', label='x2 Error Bound')
plt.plot(TIME, -np.sqrt(PHAT[:, 1]), 'b')
plt.title('x2 Error and Bound')
plt.legend()

plt.figure()
plt.plot(TIME, (Z - ZHAT).squeeze(), 'r', label='Measurement Error')
plt.plot(TIME, np.sqrt(SBAR), 'b', label='Measurement Error Bound')
plt.plot(TIME, -np.sqrt(SBAR), 'b')
plt.title('Measurement Error and Bound')
plt.legend()

plt.figure()
plt.plot(TIME, KK[:, 0], 'r', label='K Gain 1')
plt.plot(TIME, KK[:, 1], 'b', label='K Gain 2')
plt.title('Kalman Gain')
plt.legend()

plt.show()

