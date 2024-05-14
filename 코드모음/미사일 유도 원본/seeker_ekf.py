import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def missile_guidance(r, v, vM):
    """
    r : 목표물까지의 상대 위치 벡터입니다.
    v : 목표물의 상대 속도 벡터입니다.
    vM : 미사일의 속도 벡터입니다.
    a_cmd : 미사일의 가속 명령 벡터입니다.
    """
    N = 3
    R2 = np.dot(r, r)
    OmL = np.cross(r, v) / R2  # 회전율 벡터
    a_cmd = N * np.cross(OmL, vM) # 회전율과 미사일 속도 벡터의 외적으로 가속명령 계산

    return a_cmd

def seeker_ekf_tu(xhat, Phat, a_cmd, Qd, dt):
    """
    seeker EKF Time Update
    one-step propagation
    """
    # system model
    # F = np.vstack([np.hstack([np.eye(3), dt * np.eye(3)]),
    #                np.hstack([np.zeros((3, 3)), np.eye(3)])])
    # G = np.vstack([np.zeros((3, 3)), -dt * np.eye(3)])

    # Gw = np.vstack([np.zeros((3, 3)),
    #                 np.eye(3)])

    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])

    G = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [-dt, 0, 0],
                  [0, -dt, 0],
                  [0, 0, -dt]])

    Gw = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    # Predict #
    # time update
    xbar = F @ xhat + G @ a_cmd
    Pbar = F @ Phat @ F.T + Gw @ Qd @ Gw.T

    return xbar, Pbar

def missile_dyn(r, v, a_cmd, dt):
    """
    Missile model with constant speed
    One-step propagation of position and velocity of target
    
    Parameters:
    r : Current position vector
    v : Current velocity vector
    a_cmd : Commanded acceleration vector
    dt : Time step
    
    Returns:
    r_next : Next position vector
    v_next : Next velocity vector
    Cbn : Direction cosine matrix (DCM) from body frame (b) to navigation frame (n)
    """

    # One-step propagation
    r_next = r + dt * v
    v_next = v + dt * a_cmd

    # DCM b->n calculation  몸체 좌표계에서 관성 좌표계로의 방향 코사인 행렬 계산
    the = np.arctan2(-v[2], np.sqrt(v[0] ** 2 + v[1] ** 2))
    psi = np.arctan2(v[1], v[0])
    cp = np.cos(psi)
    ct = np.cos(the)
    sp = np.sin(psi)
    st = np.sin(the)
    Cbn = np.array([[cp * ct, sp * ct, -st],
                    [-sp, cp, 0],
                    [cp * st, sp * st, ct]])

    return r_next, v_next, Cbn


def target_dyn(r, v, angle, Qd, dt, status):
    
    # # Process noise
    # if status == 'sy':
    #     w = np.sqrt(Qd) @ np.random.randn(3, 1)
    # else:
    #     w = np.zeros((3, 1))
    
    # r_next = r + dt * v  # v는 (3,) 차원
    # v_next = v + w.flatten()  # (3,1) -> (3,)


    v_next = np.zeros_like(v)

    # Process noise
    if status == 'sy':
        noise_angle = 0.02 * np.random.randn()
    else:
        noise_angle = 0
    
    # Update angle with some randomness
    angle_next = angle + noise_angle
    angle_next = angle_next % (2 * np.pi)

    V =  np.linalg.norm(v)
    v_next[0] = V * np.cos(angle_next)
    v_next[1] = V * np.sin(angle_next) 
    v_next[2] = 0

    # Calculate new position
    dx = v_next[0] * dt
    dy = v_next[1] * dt
    dz = 0
    r_next = r + np.array([dx, dy, dz])

    return r_next, v_next, angle_next



def seeker_meas(r_rel, Cbn, Rd, status):
    """
    추적 장비에서의 측정값을 계산합니다.

    Parameters:
    r_rel : 상대 위치 벡터입니다.
    Cbn : 몸체 좌표계로의 변환 행렬입니다.
    Rd : 측정 잡음의 분산입니다.
    status : 시뮬레이션 상태('sy'는 시뮬레이션, 그 외의 값은 실제 동작을 의미합니다).

    Returns:
    z : 측정 거리, 고각, 방위각을 포함하는 벡터입니다.
    """
    rb = np.dot(Cbn, r_rel)

    if status == 'sy':
        v = np.sqrt(Rd) @ np.random.randn(3, 1)
    else:
        v = np.zeros((3, 1))

    R = np.sqrt(np.dot(rb.T, rb)) + v[0][0]  # 측정 거리 R
    the_g = np.arctan2(-rb[2], np.sqrt(rb[0]**2 + rb[1]**2)) + v[1][0]  # 고각
    psi_g = np.arctan2(rb[1], rb[0]) + v[2][0] # 방위각

    z = np.array([R, the_g, psi_g])

    return z.squeeze() # `.squeeze()`를 사용하여 차원 축소를 통해 (3,) 형태로 반환합니다.

def seeker_ekf_mu(z, xbar, Pbar, Rd, Cbn):
    """
    추적 장비를 위한 확장 칼만 필터(EKF)의 측정 업데이트 단계.

    Parameters:
    z : 측정값 벡터.
    xbar : 예측 상태 벡터.
    Pbar : 예측 공분산 행렬.
    Rd : 측정 잡음의 분산.
    Cbn : 몸체 좌표계로의 변환 행렬.

    Returns:
    xhat : 업데이트된 상태 추정값.
    Phat : 업데이트된 공분산.
    zhat : 측정 예측값.
    S : 측정 업데이트의 공분산.
    """
    rb = np.dot(Cbn, xbar[:3])
    r1, r2, r3 = rb
    r_mag = np.sqrt(np.dot(rb.T, rb))
    Rtmp = (r1**2 + r2**2) * np.sqrt(r1**2 + r2**2) + r3**2 * np.sqrt(r1**2 + r2**2)
    dhdrb = np.array([
        [r1 / r_mag, r2 / r_mag, r3 / r_mag],
        [r1 * r3 / Rtmp, r2 * r3 / Rtmp, -np.sqrt(r1**2 + r2**2) / r_mag**2],
        [-r2 / (r1**2 + r2**2), r1 / (r1**2 + r2**2), 0]
    ])
    
    H = np.hstack([np.dot(dhdrb, Cbn), np.zeros((3, 3))])
    r_rel = xbar[:3]
    zhat = seeker_meas(r_rel, Cbn, Rd, 'kf')

    S = np.dot(H, np.dot(Pbar, H.T)) + Rd * np.eye(3)
    K = np.dot(np.dot(Pbar, H.T), inv(S))
    xhat = xbar + np.dot(K, (z - zhat))
    Phat = Pbar - np.dot(np.dot(K, S), K.T)

    return xhat, Phat, zhat, S


dt = 0.01
# process noise, measurement noise
Qc = (5**2) * np.diag([1, 1, 0])
Qd = Qc * dt
Rd = np.diag([5**2,  (np.pi/180)**2, (np.pi/180)**2])



############## target initial state ###############
rT0 = np.array([20000, 30000, 0])
VT = 20  # target speed
thetaT = 0
angle_T0 = np.random.randint(0,181) * np.pi/180
cp = np.cos(angle_T0)
ct = np.cos(thetaT)
sp = np.sin(angle_T0)
st = np.sin(thetaT)
vT0 = VT * np.array([cp*ct, sp*ct, -st])  # 1차원 배열 (3,)

P0 = np.diag([200**2, 200**2, 200**2, 10**2, 10**2, 0])
rvT0 = np.concatenate((rT0, vT0))


############# missile initial state ###############
rM0 = np.array([0, 0, -500])
VM = 270 # missile speed
thetaM = 0; psiM = 0

cp = np.cos(psiM); ct = np.cos(thetaM)
sp = np.sin(psiM); st = np.sin(thetaM)
vM0 = VM * np.array([cp*ct, sp*ct, -st])
rvM0 = np.concatenate((rM0, vM0))


############# EKF initialization ###################
xhat = (rvT0 - rvM0) + (np.sqrt(P0) @ np.random.randn(6, 1)).flatten()
Phat = np.diag([200**2, 200**2, 200**2, 100**2, 100**2, 100**2])


############# initial guidance law #################
rhat = xhat[:3].flatten() 
vhat = xhat[3:6].flatten() 
# print("xhat shape:", xhat.shape)
# print("rhat shape:", rhat.shape)
# print("vhat shape:", vhat.shape)
a_cmd0 = missile_guidance(rhat, vhat, vM0)


# 컬렉션 초기화
POST = []
POSM = []
XR = []
XV = []
XHAT = []
PHAT = []
Z = []
ZHAT = []
SBAR = []
ACMD = []
TIME = []


################# EKF propagation ###################
for k in range(1, 100001):
    t = dt * k
    
    # Time Update (TU)
    xbar, Pbar = seeker_ekf_tu(xhat, Phat, a_cmd0, Qd, dt)
    
    # 미사일 동역학
    rM, vM, Cbn = missile_dyn(rM0, vM0, a_cmd0, dt)
    
    # 타겟 동역학
    rT, vT, angle_T = target_dyn(rT0, vT0, angle_T0, Qd, dt, 'sy')
    
    # 추적기 측정
    r_rel = rT - rM
    z = seeker_meas(r_rel, Cbn, Rd, 'sy')
    
    # Measurement Update (MU)
    xhat, Phat, zhat, S = seeker_ekf_mu(z, xbar, Pbar, Rd, Cbn)
    
    # 가이던스 법칙
    rhat = xhat[:3].flatten() 
    vhat = xhat[3:6].flatten() 
    a_cmd = missile_guidance(rhat, vhat, vM)
    a_cmd_b = np.dot(Cbn, a_cmd)  # 몸체 좌표계에서의 가속도 명령
    
    # 데이터 수집
    XR.append(rT - rM)
    XV.append(vT - vM)
    XHAT.append(xhat)
    PHAT.append(np.diag(Phat))
    Z.append(z)
    ZHAT.append(zhat)
    SBAR.append(np.diag(S))
    ACMD.append(a_cmd_b)
    TIME.append(t)
    POST.append(rT)
    POSM.append(rM)
    
    # 다음 단계를 위한 준비
    rM0, vM0, rT0, vT0, angle_T0, a_cmd0 = rM, vM, rT, vT, angle_T, a_cmd
    
    # 종료 조건
    if np.linalg.norm(rT - rM) < 5:
        break


POST = np.array(POST)
POSM = np.array(POSM)
XR = np.array(XR)
XV = np.array(XV)
X = np.concatenate((XR, XV), axis = 1)
XHAT = np.array(XHAT)
PHAT = np.array(PHAT)
Z = np.array(Z)
ZHAT = np.array(ZHAT)
SBAR = np.array(SBAR)
ACMD = np.array(ACMD)
TIME = np.array(TIME)



# 첫 번째 그림: 3D 플롯
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot(POST[:, 0], POST[:, 1], -POST[:, 2], 'r', label='Position Target')
ax1.plot(POSM[:, 0], POSM[:, 1], -POSM[:, 2], 'b', label='Position Missile')
ax1.set_title('3D Track')
ax1.grid(True)
ax1.legend()



# X와 XHAT의 차이를 서브플롯으로 표시
fig, ax2 = plt.subplots(3, 2, figsize=(10, 15))  # 3행 2열 서브플롯
fig.suptitle('State error')
index = 0
for i in range(3):
    for j in range(2):
        ax2[i, j].plot(TIME, X[:, index] - XHAT[:, index], 'orangered', linewidth=1.3)
        ax2[i, j].plot(TIME, np.sqrt(PHAT[:, index]), 'olive', linestyle='dashed', linewidth=0.8)
        ax2[i, j].plot(TIME, -np.sqrt(PHAT[:, index]), 'olive', linestyle='dashed', linewidth=0.8)
        ax2[i, j].set_title(f'x{index+1}')
        ax2[i, j].grid(True)
        index += 1



# # Z와 ZHAT의 차이를 서브플롯으로 표시
# fig, ax3 = plt.subplots(3, 1, figsize=(10, 15))
# fig.suptitle('Measurements error')
# measurements = ['R', 'Theta_g', 'Psi_g']
# for i in range(3):
#     ax3[i].plot(TIME, (Z[:, i] - ZHAT[:, i]) * (180 / np.pi if i > 0 else 1), 'orangered', linewidth=1)
#     ax3[i].plot(TIME, np.sqrt(SBAR[:, i]) * (180 / np.pi if i > 0 else 1), 'olive', linestyle='dashed', linewidth=1)
#     ax3[i].plot(TIME, -np.sqrt(SBAR[:, i]) * (180 / np.pi if i > 0 else 1), 'olive', linestyle='dashed', linewidth=1)
#     ax3[i].set_title(measurements[i])
#     ax3[i].grid(True)



# # 상대 거리 플롯
# plt.figure()
# relative_distance = np.sqrt((POST[:, 0] - POSM[:, 0])**2 + (POST[:, 1] - POSM[:, 1])**2 + (POST[:, 2] - POSM[:, 2])**2)
# plt.plot(TIME, relative_distance, 'r')
# plt.title('Relative Distance')
# plt.xlabel('Time')
# plt.ylabel('Distance')
# plt.grid(True)



# # 가속도 명령 신호 플롯
# plt.figure() 
# cmd_labels = ['Command 1', 'Command 2', 'Command 3']
# for i in range(3):
#     plt.plot(TIME, ACMD[:, i], label=cmd_labels[i])
# plt.title('Command Signals')
# plt.xlabel('Time')
# plt.ylabel('Command Value')
# plt.legend()
# plt.grid(True)

plt.tight_layout()
plt.show()
