from keras.models import load_model
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import joblib
# set TF_GPU_ALLOCATOR=cuda_malloc_async

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
    """
    r: 현재 위치 벡터 (3차원)
    v: 현재 속도 벡터 (3차원)
    azimuth: 현재 방위각
    elevation: 현재 고각
    k: 현재 시뮬레이션 단계
    dt: 시간 간격
    change_interval: 각속도 변경 간격 (시뮬레이션 단계 기준)
    max_angular_velocity: 최대 각속도 변화량 (라디안/초 단위)
    max_angle_change: 한 번의 제어동안 변화 가능한 최대 각도 (라디안 단위)
    """
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

def calculate_rmse(true_position, position):
    return np.sqrt(np.mean((true_position - position.astype(float)) ** 2))


def remove_outliers(data):
    # 사분위수 계산
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # 이상치가 아닌 데이터의 인덱스를 찾음
    non_outlier_indices = (data >= (Q1 - 1.5 * IQR)) & (data <= (Q3 + 1.5 * IQR))
    
    # 이상치가 아닌 데이터만 반환
    return data[non_outlier_indices]


model1 = load_model('seeker.WL100-100-32')
model2 = load_model('seeker.WL100-128-32')
model3 = load_model('seeker.WL100-128-64')

scaler0 = joblib.load('scalerX_seeker_WL100.pkl')
scaler1 = joblib.load('scalerY_seeker_WL100.pkl')

WL=120
EG=100
num_paths = 200
original_rms= []
rms_model1 = []
rms_model2 = []
rms_model3 = []
mission_failed_count = 0
mission_success_count = 0


########################################################검증##################################################################################
for _ in range(num_paths):
    
    dt = 0.01
    # process noise, measurement noise
    Qc = (4**2) * np.diag([1, 1, 1])
    Qd = Qc * dt
    Rd = np.diag([5**2,  (np.pi/180)**2, (np.pi/180)**2])


    ############## target initial state ###############
    rT0 = np.array([np.random.randint(1000,4000), np.random.randint(1000,4000), np.random.randint(1000,4000)])
    VT = 150  # target speed
    elevation0 = np.random.randint(-60,61) * np.pi / 180 ## theta
    azimuth0 = np.random.randint(-120,121) * np.pi / 180  ## psi

    # print("elevation0:", elevation0)
    # print("azimuth0:", azimuth0)

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
    Phat = np.diag([300**2, 300**2, 300**2, 100**2, 100**2, 100**2])


    ############# initial guidance law #################
    rhat = xhat[:3].flatten() 
    vhat = xhat[3:6].flatten() 
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
    success = 0 
    
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
        rM0, vM0, rT0, vT0, azimuth0, elevation0, a_cmd0 = rM, vM, rT, vT, azimuth, elevation, a_cmd
        
        # 종료 조건
        if np.linalg.norm(rT - rM) < 5:
            success = 1
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

    
    
    
    
    if success == 1:
        mission_success_count += 1
        print(f'mission success : {mission_success_count}')
        INPUT_3d = []
        for k in range(0, len(XHAT) - WL + 1, 1):
            temp = np.concatenate((XHAT[k:k+WL], Z[k:k+WL]), axis=1)
            INPUT_3d.append(temp)
        x_test = np.array(INPUT_3d,dtype = np.float32)
        before_corrected_position = x_test[:,-1,:3]
        
        
        num_samples, num_timesteps, num_features = x_test.shape
        x_test_reshaped = x_test.reshape(-1, num_features)  # (샘플 수 * 시간 단계 수, 특성 수)


        x_test_scaled = scaler0.transform(x_test_reshaped)
        x_test_scaled = x_test_scaled.reshape(num_samples, num_timesteps, num_features)
        
        
        model_prediction_scaled1 = model1.predict(x_test_scaled)
        predicted_position1 = np.zeros(model_prediction_scaled1.shape)
        predicted_position1 = scaler1.inverse_transform(model_prediction_scaled1)
        corrected_position1 =  before_corrected_position + predicted_position1
        
        model_prediction_scaled2 = model2.predict(x_test_scaled)
        predicted_position2 = np.zeros(model_prediction_scaled2.shape)
        predicted_position2 = scaler1.inverse_transform(model_prediction_scaled2)
        corrected_position2 =  before_corrected_position + predicted_position2

        model_prediction_scaled3 = model3.predict(x_test_scaled)
        predicted_position3 = np.zeros(model_prediction_scaled3.shape)
        predicted_position3 = scaler1.inverse_transform(model_prediction_scaled3)
        corrected_position3 =  before_corrected_position + predicted_position3
        


        
        rms= calculate_rmse(XR[EG:], XHAT[EG:,:3])
        #print(f"rms : {rms}")
        corrected_rms1 = calculate_rmse(XR[WL-1+EG:,:],corrected_position1[EG:,:] )
        #print(f"corrected_rms1 : {corrected_rms1}")
        corrected_rms2 = calculate_rmse(XR[WL-1+EG:,:],corrected_position2[EG:,:] )
        #print(f"corrected_rms2 : {corrected_rms2}")
        corrected_rms3 = calculate_rmse(XR[WL-1+EG:,:],corrected_position3[EG:,:] )
        #print(f"corrected_rms3 : {corrected_rms3}")




        original_rms.append(rms)
        rms_model1.append(corrected_rms1)
        rms_model2.append(corrected_rms2)
        rms_model3.append(corrected_rms3)
    else :
        mission_failed_count += 1
        print(f'mission failed : {mission_failed_count}')




print(f"Average RMS error for Observed: {np.mean(original_rms)}")
print(f"Average RMS error for Model1: {np.mean(rms_model1)}")
print(f"Average RMS error for Model2: {np.mean(rms_model2)}")
print(f"Average RMS error for Model3: {np.mean(rms_model3)}")


# ########################################################그래프###################################################################################
# 평균 RMS 오차
rms_errors_data = [original_rms, rms_model1, rms_model2, rms_model3] 
rms_errors_data_cleaned = [remove_outliers(np.array(errors)) for errors in rms_errors_data]

labels = ['Original', '100-32','128-32', '128-64']
colors = ['blue', 'orange', 'gray', 'olive']

# RMSE 감소 비율 계산
original_reduction_percentages = [(1 - np.mean(rms_error) / np.mean(original_rms)) * 100 if i > 0 else 0 for i, rms_error in enumerate(rms_errors_data_cleaned)]


fig, axs = plt.subplots(1, 2, figsize=(12, 10))
bar_plot = axs[0].bar(labels, [np.mean(errors) for errors in rms_errors_data_cleaned], color=colors)

# 각 바 위에 텍스트 표시
for idx, rect in enumerate(bar_plot):
    height = rect.get_height()
    text = f'RMSE : {np.mean(rms_errors_data_cleaned[idx]):.2f}'
    if idx > 0: 
        text += f'\noriginal: {original_reduction_percentages[idx]:.2f}'
    axs[0].text(rect.get_x() + rect.get_width()/2., 1.01*height, text, ha='center', va='bottom')

# 'Observed'를 제외한 나머지 모델들에 대해 박스 플롯 생성
box_plot = plt.boxplot(rms_errors_data_cleaned, positions=range(len(labels)), widths=0.3, patch_artist=True, showfliers=False)  



# 박스 색상 설정
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

# 박스 플롯 위에 평균과 표준편차 표시
for i, line in enumerate(box_plot['boxes']):
    median = np.median(rms_errors_data_cleaned[i])
    std = np.std(rms_errors_data_cleaned[i])
    if i == 4:
        axs[1].text(i-0.5, median, f'Median: {median:.2f}\nStd: {std:.2f}', horizontalalignment='center', verticalalignment='center')
    else:
        axs[1].text(i+0.5, median, f'Median: {median:.2f}\nStd: {std:.2f}', horizontalalignment='center', verticalalignment='center')

# 박스 플롯 x축 눈금 라벨 설정
axs[1].set_xticks(range(len(labels)))
axs[1].set_xticklabels(labels)

axs[0].set_title(f'Average RMSE for each model (based on {num_paths} paths)')
axs[0].set_xlabel('Model')
axs[0].set_ylabel('Average RMSE')

axs[1].set_title('Box plot of RMSE')
axs[1].set_xlabel('Model')
axs[1].set_ylabel('RMSE')

plt.tight_layout()
plt.show()


