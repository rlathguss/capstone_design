from keras.models import load_model
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import joblib


# PNG, proportional navigation guidance 비례항법
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


def target_dyn(r, v, azimuth, elevation, k, dt):
    global azimuth_change, elevation_change
    change_interval = 500
    max_angular_velocity = 15 * np.pi / 180
    max_angle_change = 10 * np.pi / 180

    if k == 1:
        global tq
        tq = np.random.randint(1, change_interval+1)
        azimuth_change = 0
        elevation_change = 0

    if k == tq:
        # 각속도 변화를 시간 간격에 따라 제한
        angular_velocity_change_azimuth = np.random.uniform(-max_angular_velocity, max_angular_velocity)
        angular_velocity_change_elevation = np.random.uniform(-max_angular_velocity, max_angular_velocity)

        # 실제 각도 변화 계산 (이번에는 각속도 변화량을 유지합니다.)
        azimuth_change = angular_velocity_change_azimuth * dt
        elevation_change = angular_velocity_change_elevation * dt

        # 한 번의 제어동안 변화 가능한 최대 각도를 초과하지 않도록 제한
        azimuth_change = np.clip(azimuth_change, -max_angle_change, max_angle_change)
        elevation_change = np.clip(elevation_change, -max_angle_change, max_angle_change)

        tq += np.random.randint(200, change_interval+1)

    # 각도 업데이트 (이전 각속도 변화량을 사용하여 업데이트)
    azimuth = (azimuth + azimuth_change) % (2 * np.pi)
    elevation = (elevation + elevation_change) % (2 * np.pi)

    V = np.linalg.norm(v)
    v_next = np.zeros_like(v)

    v_next[0] = V * np.cos(elevation) * np.cos(azimuth)
    v_next[1] = V * np.cos(elevation) * np.sin(azimuth)
    v_next[2] = V * np.sin(elevation)

    dx = v_next[0] * dt
    dy = v_next[1] * dt
    dz = v_next[2] * dt
    r_next = r + np.array([dx, dy, dz])

    return r_next, v_next, azimuth, elevation


def seeker_meas(r_rel, Cbn, Rd, status):
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

def calculate_rmse(true_position, position):
    return np.sqrt(np.mean((true_position - position.astype(float)) ** 2))


dt = 0.01
# process noise, measurement noise
Qc = (4**2) * np.diag([1, 1, 1])
Qd = Qc * dt
Rd = np.diag([5**2,  (np.pi/180)**2, (np.pi/180)**2])


############## target initial state ###############
#rT0 = np.array([1500, 1500, 1000])
rT0 = np.array([np.random.randint(500,3000), np.random.randint(500,3000), np.random.randint(500,3000)])
VT = 150  # target speed
elevation0 = np.random.randint(-60,61) * np.pi / 180 ## theta
azimuth0 = np.random.randint(-60,61) * np.pi / 180  ## psi

rT0_sim2 = rT0
elevation0_sim2= elevation0
azimuth0_sim2 = azimuth0


print("elevation0:", elevation0)
print("azimuth0:", azimuth0)

cp = np.cos(azimuth0)       # psi의 코싸인 값, 방위각의 코사인
ct = np.cos(elevation0)     # theta의 코사인 값, 고도각의 코사인
sp = np.sin(azimuth0)       # psi의 사인 값, 방위각의 사인
st = np.sin(elevation0)     # theta의 사인 값, 고도각의 사인
vT0 = VT * np.array([cp*ct, sp*ct, -st])  # 1차원 배열 (3,)

P0 = np.diag([200**2, 200**2, 200**2, 10**2, 10**2, 10**2])
rvT0 = np.concatenate((rT0, vT0))



############# missile initial state ###############
rM0 = np.array([0, 0, -500])
VM = 200 # missile speed
thetaM = 0; psiM = 0

cp = np.cos(psiM); ct = np.cos(thetaM)
sp = np.sin(psiM); st = np.sin(thetaM)
vM0 = VM * np.array([cp*ct, sp*ct, -st])
rvM0 = np.concatenate((rM0, vM0))


############# EKF initialization ###################
xhat = (rvT0 - rvM0) + (np.sqrt(P0) @ np.random.randn(6, 1)).flatten()
xhat_sim2 = xhat
Phat = np.diag([300**2, 300**2, 300**2, 100**2, 100**2, 100**2])


############# initial guidance law #################
rhat = xhat[:3].flatten() 
vhat = xhat[3:6].flatten() 
a_cmd0 = missile_guidance(rhat, vhat, vM0)


# 컬렉션 초기화
POST = []
VELT= []
POSM = []
XR = []
XV = []
XHAT = []
PHAT = []

ACMD = []
TIME = []


################# EKF propagation ###################
for k in range(1, 10001):
    t = dt * k
    
    # Time Update (TU)
    xbar, Pbar = seeker_ekf_tu(xhat, Phat, a_cmd0, Qd, dt)
    
    # 미사일 동역학
    rM, vM, Cbn = missile_dyn(rM0, vM0, a_cmd0, dt)
    
    # 타겟 동역학
    rT, vT, azimuth, elevation = target_dyn(rT0, vT0, azimuth0, elevation0, k, dt)
    
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
    POST.append(rT)
    VELT.append(vT)
    POSM.append(rM)

    XR.append(rT - rM)
    XV.append(vT - vM)

    XHAT.append(xhat)
    PHAT.append(np.diag(Phat))
    
    ACMD.append(a_cmd_b)
    TIME.append(t)

    rM0, vM0, rT0, vT0, azimuth0, elevation0, a_cmd0 = rM, vM, rT, vT, azimuth, elevation, a_cmd
    
    # 종료 조건
    if np.linalg.norm(rT - rM) < 5:
        break


POST = np.array(POST)
VELT = np.array(VELT)
POSM = np.array(POSM)

XR = np.array(XR)
XV = np.array(XV)
X = np.concatenate((XR, XV), axis = 1)

XHAT = np.array(XHAT)
PHAT = np.array(PHAT)

ACMD = np.array(ACMD)
TIME = np.array(TIME)




# ######################################################## Sim2 검증###################################################################################
# 컬렉션 초기화
POST_sim2 = []
POSM_sim2 = []

XR_sim2 = []
XV_sim2 = []

XHAT_sim2 = []
PHAT_sim2 = []
Z_sim2 = []

ACMD_sim2 = []
TIME_sim2 = []
XR_sim2_corrected = []


dt = 0.01
# process noise, measurement noise
Qc = (4**2) * np.diag([1, 1, 1])
Qd = Qc * dt
Rd = np.diag([5**2,  (np.pi/180)**2, (np.pi/180)**2])


############## target initial state ###############
# rT0 = np.array([2000, 2000, 0])
rT0 = rT0_sim2
VT = 150  # target speed
elevation0 = elevation0_sim2  ## theta
azimuth0 = azimuth0_sim2  ## psi


print("Sim2 elevation0:", elevation0)
print("Sim2 azimuth0:", azimuth0)

cp = np.cos(azimuth0)       # psi의 코싸인 값, 방위각의 코사인
ct = np.cos(elevation0)     # theta의 코사인 값, 고도각의 코사인
sp = np.sin(azimuth0)       # psi의 사인 값, 방위각의 사인
st = np.sin(elevation0)     # theta의 사인 값, 고도각의 사인
vT0 = VT * np.array([cp*ct, sp*ct, -st])  # 1차원 배열 (3,)

P0 = np.diag([200**2, 200**2, 200**2, 10**2, 10**2, 10**2])
rvT0 = np.concatenate((rT0, vT0))



############# missile initial state ###############
rM0 = np.array([0, 0, -500])
VM = 200 # missile speed
thetaM = 0; psiM = 0

cp = np.cos(psiM); ct = np.cos(thetaM)
sp = np.sin(psiM); st = np.sin(thetaM)
vM0 = VM * np.array([cp*ct, sp*ct, -st])
rvM0 = np.concatenate((rM0, vM0))


############# EKF initialization ###################
xhat = xhat_sim2
Phat = np.diag([300**2, 300**2, 300**2, 100**2, 100**2, 100**2])


############# initial guidance law #################
rhat = xhat[:3].flatten() 
vhat = xhat[3:6].flatten() 
a_cmd0 = missile_guidance(rhat, vhat, vM0)

WL = 120
EG = 100

model = load_model('seeker.WL120-Q4-128-32')
scaler0 = joblib.load('scalerX_seeker_WL120_Q4.pkl')
scaler1 = joblib.load('scalerY_seeker_WL120_Q4.pkl')

################# EKF propagation ###################
for k in range(1, 10001):
    t = dt * k
    
    # Time Update (TU)
    xbar, Pbar = seeker_ekf_tu(xhat, Phat, a_cmd0, Qd, dt)
    
    # 미사일 동역학
    rM, vM, Cbn = missile_dyn(rM0, vM0, a_cmd0, dt)
    
    # 타겟 동역학
    rT = POST[k-1,:]
    vT = VELT[k-1,:]
    
    # 추적기 측정
    r_rel = rT - rM
    z = seeker_meas(r_rel, Cbn, Rd, 'sy')
    
    # Measurement Update (MU)
    xhat, Phat, zhat, S = seeker_ekf_mu(z, xbar, Pbar, Rd, Cbn)
    
    # 가이던스 법칙
    rhat = xhat[:3].flatten() 
    vhat = xhat[3:6].flatten()

    # 데이터 수집
    POST_sim2.append(rT)
    POSM_sim2.append(rM)

    XR_sim2.append(rT - rM) # xhat 상대좌표
    XV_sim2.append(vT - vM) # xhat 상대좌표
    
    XHAT_sim2.append(xhat)
    PHAT_sim2.append(np.diag(Phat))
    Z_sim2.append(z)

    TIME_sim2.append(t)


    if k > (WL + EG- 1): 
        x_test = np.hstack((np.array(XHAT_sim2[-WL:]), np.array(Z_sim2[-WL:])))
        x_test_scaled = scaler0.transform(x_test)                        ## [WL,9]로 만들기
        x_test_scaled_reshaped = np.expand_dims(x_test_scaled, axis=0)   ## [1,WL,9]로 만들기

        model_prediction_scaled = model.predict(x_test_scaled_reshaped)
        rhat_corrected = rhat + scaler1.inverse_transform(model_prediction_scaled)
        rhat_corrected = rhat_corrected.flatten()
        XR_sim2_corrected.append(rhat_corrected)
        a_cmd = missile_guidance(rhat_corrected, vhat, vM)
        a_cmd_b = np.dot(Cbn, a_cmd)  # 몸체 좌표계에서의 가속도 명령
    
    else :
        XR_sim2_corrected.append(rhat)
        a_cmd = missile_guidance(rhat, vhat, vM)
        a_cmd_b = np.dot(Cbn, a_cmd)  # 몸체 좌표계에서의 가속도 명령

    ACMD_sim2.append(a_cmd_b)
    

    # 다음 단계를 위한 준비
    rM0, vM0, rT0, vT0, a_cmd0 = rM, vM, rT, vT, a_cmd
    
    # 종료 조건
    if np.linalg.norm(rT - rM) < 5:
        break


POST_sim2 = np.array(POST_sim2)
POSM_sim2 = np.array(POSM_sim2)

XR_sim2 = np.array(XR_sim2)
XV_sim2 = np.array(XV_sim2)
X_sim2 = np.concatenate((XR_sim2, XV_sim2), axis = 1)

XHAT_sim2 = np.array(XHAT_sim2)
PHAT_sim2 = np.array(PHAT_sim2)


ACMD_sim2 = np.array(ACMD_sim2)
TIME_sim2 = np.array(TIME_sim2)
XR_sim2_corrected = np.array(XR_sim2_corrected)


print(f"EKF simulation time : {TIME[-1]}")
print(f"EKF + AI simulation time : {TIME_sim2[-1]}")



x_rmse = calculate_rmse(XR[EG:,0],XHAT[EG:,0])
y_rmse = calculate_rmse(XR[EG:,1],XHAT[EG:,1])
z_rmse = calculate_rmse(XR[EG:,2],XHAT[EG:,2])
print(f"x rmse : {x_rmse}")
print(f"y rmse : {y_rmse}")
print(f"z rmse : {z_rmse}")


x_corrected_rmse = calculate_rmse(XR_sim2[EG:,0],XR_sim2_corrected[EG:,0])
y_corrected_rmse = calculate_rmse(XR_sim2[EG:,1],XR_sim2_corrected[EG:,1])
z_corrected_rmse = calculate_rmse(XR_sim2[EG:,2],XR_sim2_corrected[EG:,2])
print(f"x corrected_rmse : {x_corrected_rmse}")
print(f"y corrected_rmse : {y_corrected_rmse}")
print(f"z corrected_rmse : {z_corrected_rmse}")

################################# 3D 트랙 플롯 (애니메이션)##################################

fig = plt.figure(figsize=(20, 10))



# 3D 트랙 플롯 (애니메이션)
ax1 = fig.add_subplot(2, 4, 1, projection='3d')
ax1.set_title('3D Track')

# 상대 거리 플롯 (정적)
ax2 = fig.add_subplot(2, 4, 5)
relative_distance = np.sqrt((POST[:, 0] - POSM[:, 0])**2 + (POST[:, 1] - POSM[:, 1])**2 + (POST[:, 2] - POSM[:, 2])**2)
relative_distance2 = np.sqrt((POST_sim2[:, 0] - POSM_sim2[:, 0])**2 + (POST_sim2[:, 1] - POSM_sim2[:, 1])**2 + (POST_sim2[:, 2] - POSM_sim2[:, 2])**2)
ax2.plot(TIME, relative_distance, 'r')
ax2.plot(TIME_sim2, relative_distance2, 'b')
ax2.set_title('Relative Distance')
ax2.set_xlabel('Time')
ax2.set_ylabel('Distance', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 가속도 명령 신호를 같은 그래프에 추가하지만, 다른 Y축 사용
ax2b = ax2.twinx()
cmd_labels = ['Command 1', 'Command 2', 'Command 3' ]
for i in range(3):
    ax2b.plot(TIME, ACMD[:, i], label=cmd_labels[i], linestyle='--')
cmd_labels2 = ['AI + Command 1', 'AI + Command 2', 'AI + Command 3' ]
for i in range(3):
    ax2b.plot(TIME_sim2, ACMD_sim2[:, i], label=cmd_labels2[i], linestyle='--')
ax2b.set_ylabel('Command Value', color='b')
ax2b.tick_params(axis='y', labelcolor='b')


# X1, X2, X3, X4, X5, X6에 대한 플롯
for i in range(6):
    if i < 3:
        ax = fig.add_subplot(2, 4, i+2)
        ax.plot(TIME_sim2[:], X_sim2[:, i] - XR_sim2_corrected[:, i], 'blue', linewidth=1.3, label='Corrected')
    else:
        ax = fig.add_subplot(2, 4, i+3)
    ax.plot(TIME, X[:, i] - XHAT[:, i], 'orangered', linewidth=1.3)
    ax.plot(TIME, np.sqrt(PHAT[:, i]), 'olive', linestyle='dashed', linewidth=0.8)
    ax.plot(TIME, -np.sqrt(PHAT[:, i]), 'olive', linestyle='dashed', linewidth=0.8)
    ax.set_title(f'X{i+1} Error')
    ax.grid(True)

# 데이터 희석
sparse_index = np.arange(0, len(TIME), 150)  # 150개당 하나씩 데이터 선택
sparse_POST = POST[sparse_index]
sparse_POSM = POSM[sparse_index]
sparse_POSM_sim2 = POSM_sim2[sparse_index]
sparse_TIME = TIME[sparse_index]


# 애니메이션 업데이트 함수
def update(frame):
    ax1.clear()
    # 희석된 데이터 사용
    ax1.plot(sparse_POST[:frame, 0], sparse_POST[:frame, 1], -sparse_POST[:frame, 2], 'r', label='Position Target')
    ax1.plot(sparse_POSM[:frame, 0], sparse_POSM[:frame, 1], -sparse_POSM[:frame, 2], 'b', label='Position Missile')
    ax1.plot(sparse_POSM_sim2[:frame, 0], sparse_POSM_sim2[:frame, 1], -sparse_POSM_sim2[:frame, 2], 'olive', label='Position AI Missile')
    ax1.legend()
    ax1.set_title('3D Track')

# 애니메이션 실행
ani = FuncAnimation(fig, update, frames=len(sparse_TIME), interval=5)

plt.tight_layout()
plt.show()