import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# R1, R2의 범위 설정
R1 = np.linspace(-1000, 1000, 1000)
R2 = np.linspace(-1000, 1000, 100)

# R1, R2의 그리드 생성
R1, R2 = np.meshgrid(R1, R2)

# R3 값을 임의로 설정 (예: R3 = -3)
R3 = -300

# 함수 계산
Z = np.arctan2(-R3, np.sqrt(R1**2 + R2**2))

# 3D 그래프 그리기
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R1, R2, Z, cmap='viridis')

ax.set_xlabel('R1 Axis')
ax.set_ylabel('R2 Axis')
ax.set_zlabel('Function Value')

plt.show()



# 원 함수
def r_function(r1, r2, r3):
    return np.sqrt(r1**2 + r2**2 + r3**2)

# 자코비안 계산
def jacobian(r1, r2, r3):
    r = np.sqrt(r1**2 + r2**2 + r3**2)
    return np.array([r1/r, r2/r, r3/r])

# 주어진 점
r1_0, r2_0, r3_0 = 1, 1, 1  # 예시 점
r_0 = r_function(r1_0, r2_0, r3_0)
J = jacobian(r1_0, r2_0, r3_0)

# 변화 범위
delta_r1 = np.linspace(-1, 1, 100)
delta_r2 = np.linspace(-1, 1, 100)

# 선형 근사치 및 실제 함수 값 계산
DeltaR1, DeltaR2 = np.meshgrid(delta_r1, delta_r2)
Linear_Approx = r_0 + J[0]*DeltaR1 + J[1]*DeltaR2
Original = r_function(r1_0 + DeltaR1, r2_0 + DeltaR2, r3_0)

# 오차 계산
Error = np.abs(Original - Linear_Approx)

# 오차 시각화
plt.figure(figsize=(10, 7))
plt.contourf(DeltaR1, DeltaR2, Error, 20, cmap='viridis')
plt.colorbar


plt.show()
