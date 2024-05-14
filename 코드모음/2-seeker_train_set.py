import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# PNG, proportional navigation guidance 비례항법
def missile_guidance(r, v, vM):
    """
    r : 목표물까지의 상대 위치 벡터입니다.
    v : 목표물의 상대 속도 벡터입니다.
    vM : 미사일의 속도 벡터입니다.
    a_cmd : 미사일의 가속 명령 벡터입니다.
    """
    N = 4
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


def target_dyn(r, v, azimuth, elevation, k, dt):
    global azimuth_change, elevation_change,tq, tq_past,angular_velocity_change_azimuth, angular_velocity_change_elevation, azimuth_change_sum,elevation_change_sum
    change_interval = 400
    max_angular_velocity = 30 * np.pi / 180
    max_angle_change = 25 * np.pi / 180

    if k == 1:
        tq_past = 1
        tq = np.random.randint(1, change_interval+1)
        azimuth_change = 0
        elevation_change = 0
        angular_velocity_change_azimuth = 0
        angular_velocity_change_elevation = 0
        azimuth_change_sum = 0
        elevation_change_sum = 0

    elif k == tq:
        azimuth_change_sum = 0
        elevation_change_sum = 0
        angular_velocity_change_azimuth = np.random.uniform(-max_angular_velocity, max_angular_velocity)
        angular_velocity_change_elevation = np.random.uniform(-max_angular_velocity, max_angular_velocity)
        azimuth_change = angular_velocity_change_azimuth * dt
        elevation_change = angular_velocity_change_elevation * dt

        tq_past = tq 
        tq += np.random.randint(200, change_interval+1)


    elif tq_past < k < tq :
        azimuth_change_sum += angular_velocity_change_azimuth * dt
        elevation_change_sum += angular_velocity_change_elevation * dt
        if abs(azimuth_change_sum) > max_angle_change:
            azimuth_change = 0
        if abs(elevation_change_sum) > max_angle_change:
            elevation_change = 0


    azimuth = (azimuth + azimuth_change) % (2 * np.pi)
    elevation = (elevation + elevation_change) % (2 * np.pi)

    V = np.linalg.norm(v)
    v_next = np.array([V * np.cos(elevation) * np.cos(azimuth), V * np.cos(elevation) * np.sin(azimuth), V * np.sin(elevation)])
    r_next = r + v_next * dt

    return r_next, v_next, azimuth, elevation


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

    return xhat, Phat




def sim():

    # process noise, measurement noise
    dt = 0.01
    Qc = (4**2) * np.diag([1, 1, 1])
    Qd = Qc * dt
    Rd = np.diag([5**2,  (np.pi/180)**2, (np.pi/180)**2])

    ############## target initial state ###############
    rT0 = np.array([np.random.randint(500,3000), np.random.randint(500,3000), np.random.randint(500,3000)])
    VT = 150  # target speed
    elevation0 = np.random.randint(-60,61) * np.pi / 180 ## theta 고도각
    azimuth0 = np.random.randint(-60,61) * np.pi / 180  ## psi 방위각


    cp = np.cos(azimuth0)
    ct = np.cos(elevation0)
    sp = np.sin(azimuth0) 
    st = np.sin(elevation0)   
    vT0 = VT * np.array([cp*ct, sp*ct, -st])  # 1차원 배열 (3,)

    P0 = np.diag([200**2, 200**2, 200**2, 10**2, 10**2, 10**2])
    rvT0 = np.concatenate((rT0, vT0))


    ############# missile initial state ###############
    rM0 = np.array([0, 0, -500])
    VM = 220 # missile speed
    thetaM = 0; psiM = 0

    cp = np.cos(psiM); ct = np.cos(thetaM)
    sp = np.sin(psiM); st = np.sin(thetaM)
    vM0 = VM * np.array([cp*ct, sp*ct, -st])
    rvM0 = np.concatenate((rM0, vM0))


    ############# EKF initialization ###################
    xhat = (rvT0 - rvM0) + (np.sqrt(P0) @ np.random.randn(6, 1)).flatten()
    Phat = np.diag([300**2, 300**2, 300**2, 100**2, 100**2, 100**2])


    ############# initial guidance law #################
    rhat = xhat[:3].flatten() 
    vhat = xhat[3:6].flatten() 
    a_cmd0 = missile_guidance(rhat, vhat, vM0)

    # 컬렉션 초기화
    XR = []
    XV = []
    XHAT = []
    Z = []
    success = 0 

    ################# EKF propagation ###################
    for k in range(1, 10001):
        
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
        xhat, Phat = seeker_ekf_mu(z, xbar, Pbar, Rd, Cbn)
        
        # 가이던스 법칙
        rhat = xhat[:3].flatten() 
        vhat = xhat[3:6].flatten() 
        a_cmd = missile_guidance(rhat, vhat, vM)
        
        # 데이터 수집
        XR.append(rT - rM)
        XV.append(vT - vM)
        XHAT.append(xhat)
        Z.append(z)

        
        # 다음 단계를 위한 준비
        rM0, vM0, rT0, vT0, azimuth0, elevation0, a_cmd0 = rM, vM, rT, vT, azimuth, elevation, a_cmd
        
        # 종료 조건
        if np.linalg.norm(rT - rM) < 5:
            success = 1
            break

    XREAL = np.concatenate((XR, XV), axis = 1)

    return XREAL, XHAT, Z, success



def generate_train_val_data(window_length, step):
    
    x_data = []
    y_data = []
    mission_failed_count = 0
    mission_success_count = 0
    for _ in range(600):
        XREAL, XHAT, Z, success= sim() 
        if success == 1:
            mission_success_count += 1
            print(f'mission success : {mission_success_count}')
            XREAL = np.array(XREAL)
            XHAT = np.array(XHAT)
            Z = np.array(Z)
            # print(f"XREAL.shape: {XREAL.shape}")
            # print(f"XHAT.shape: {XHAT.shape}")
            # print(f"Z.shape: {Z.shape}")

            result_real_3d = []
            result_hat_3d = []
            for k in range(0, len(XHAT) - window_length + 1, step):

                temp = np.concatenate((XHAT[k:k+window_length], Z[k:k+window_length]), axis=1)
                result_hat_3d.append(temp)
                result_real_3d.append(XREAL[k:k+window_length])

            result_hat_3d = np.array(result_hat_3d, dtype=np.float32)[5:]
            result_real_3d = np.array(result_real_3d, dtype=np.float32)[5:]

            # Calculate target values
            target_values = result_real_3d[:, -1, :3] - result_hat_3d[:, -1, :3]
            x_data.extend(result_hat_3d)
            y_data.extend(target_values)
        else :
            mission_failed_count += 1
            print(f'mission failed : {mission_failed_count}','failedfailedfailedfailedfailedfailedfailed')

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    print(f'x_train.shape: {x_train.shape}')
    print(f'y_train.shape: {y_train.shape}') 
    print(f'x_val.shape: {x_val.shape}')
    print(f'y_val.shape: {y_val.shape}') 

    return x_train, y_train, x_val, y_val



######################################################학습###################################################################################
WL = 120
step = 20
x_train, y_train, x_val, y_val= generate_train_val_data(WL, step)

# 각 축에 대한 스케일러를 생성
scalerX = StandardScaler()
scalerY = StandardScaler()
num_samples, num_timesteps, num_features = x_train.shape


x_train_reshaped = x_train.reshape(-1, num_features)
x_val_reshaped = x_val.reshape(-1, num_features)

x_train_scaled = scalerX.fit_transform(x_train_reshaped)
x_val_scaled = scalerX.transform(x_val_reshaped)

x_train_scaled = x_train_scaled.reshape(num_samples, num_timesteps, num_features)
x_val_scaled = x_val_scaled.reshape(-1, num_timesteps, num_features)



y_train_scaled = scalerY.fit_transform(y_train)
y_val_scaled = scalerY.transform(y_val)

joblib.dump(scalerX, 'scalerX_seeker_WL120_Q4.pkl')
joblib.dump(scalerY, 'scalerY_seeker_WL120_Q4.pkl')
np.savez('train_set_seeker_WL120_Q4.npz', x_train=x_train_scaled, y_train=y_train_scaled, x_val=x_val_scaled, y_val=y_val_scaled)


print(f'x_train_scaled.shape: {x_train_scaled.shape}')
print(f'y_train_scaled.shape: {y_train_scaled.shape}') 
print(f'x_val_scaled.shape: {x_val_scaled.shape}')
print(f'y_val_scaled.shape: {y_val_scaled.shape}') 