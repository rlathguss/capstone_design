from dof3_dynamics import *
from seeker_meas import *

import numpy as np
import matplotlib.pyplot as plt

ng_limit = 6 * g

def tpng(N, OmL, Vc, LOS_hat):
    a_cmd = N * Vc * np.cross(OmL, LOS_hat) - np.array([0,0,-g])
    a_cmd_mag = np.sqrt(a_cmd.dot(a_cmd))
    if a_cmd_mag > (g * ng_limit):
        a_cmd = (g * ng_limit) * a_cmd / a_cmd_mag

    return a_cmd

def ppng(N, OmL, vM):
    vM_hat = vM / np.sqrt(vM.dot(vM))
    gp = np.array([0,0,-g]) - (np.array([0,0,-g]).dot(vM_hat))*vM_hat
    a_cmd = N * np.cross(OmL, vM) - gp
    a_cmd_mag = np.sqrt(a_cmd.dot(a_cmd))
    if a_cmd_mag > (g * ng_limit):
        a_cmd = (g * ng_limit) * a_cmd / a_cmd_mag

    return a_cmd

#최종 속도가 미설정된 최적유도법칙 
def optg(N, r, v, Vc, R, t):
    t_go = R/Vc
    #t_go = tf-t
    a_cmd = N/(t_go*t_go) * (r + t_go*v) - np.array([0,0,-g])
    a_cmd_mag = np.sqrt(a_cmd.dot(a_cmd))
    if a_cmd_mag > (g * ng_limit):
        a_cmd = (g * ng_limit) * a_cmd / a_cmd_mag

    return a_cmd

#최종 속도가 설정된 최적유도법칙 
def optg_ivc(N, r, v, Vc, R, vF, vM, t):
    t_go = R/Vc
    #t_go = tf-t
    a_cmd = N/(t_go*t_go) * (r + t_go*v) - (2/t_go) * (vF-vM) - np.array([0,0,-g])

    a_cmd_mag = np.sqrt(a_cmd.dot(a_cmd))
    if a_cmd_mag > (g * ng_limit):
        a_cmd = (g * ng_limit) * a_cmd / a_cmd_mag

    return a_cmd


""" create target objects """
target = Dof3Dynamics()
# set initial position and velocity of target
posT = [10000, 10000, 3000]
v_magT = 300
azimuthT = 0*np.pi/180
elevationT = 0*np.pi/180

rT0, vT0 = target.reset(posT, v_magT, azimuthT, elevationT)

""" create missile objects """
missile = Dof3Dynamics()
# set position and velocity of missile
posM = [0, 0, 0]
v_magM = 400
azimuthM = 30*np.pi/180
elevationM = 30*np.pi/180
impact_dir = np.array([0, 1, 0])
impact_dir = impact_dir / np.sqrt(impact_dir.dot(impact_dir))
vF = v_magM*impact_dir

rM0, vM0 = missile.reset(posM, v_magM, azimuthM, elevationM)


""" processing """
# save the results
pos_target = []
pos_missile = []
vel_missile = []
sp_missile = []
rel_distance = []
command = []
look_angle = []
run_time = []


for k in range(100000):
    t = k*dt
    print(f't= {t:.3f}')

    """ seeker measurement """
    OmL, LOS_hat, R, Vc = seeker_meas(rT0, vT0, rM0, vM0)

    """ guidance law """
    # a_cmd = optg(3, rT0-rM0, vT0-vM0, Vc, R, t)
    # a_cmd = optg_ivc(6, rT0 - rM0, vT0 - vM0, Vc, R, vF, vM0, t)

    # a_cmd = tpng(3, OmL, Vc, LOS_hat)
    a_cmd = ppng(3, OmL, vM0)

    """ missile dynamics """
    rM, vM = missile.step(a_cmd, 'missile')

    """ target dynamics """
    a_cmd_target = np.array([0, 0, 0])
    rT, vT = target.step(a_cmd_target, 'target')


    """ collections """
    pos_target.append(rT0)
    pos_missile.append(rM0)
    vel_missile.append(vM0)
    sp_missile.append(np.sqrt(vM.dot(vM)))
    rel_distance.append(R)
    command.append(a_cmd)
    run_time.append(t)


    """ for the next step """
    rM0 = rM
    vM0 = vM
    rT0 = rT
    vT0 = vT

    """ stop condition """
    if R < 5:
        break

print(f'simulation done at time: {t:.3f} sec, and R: {R:.3f} m')

# convert lists to numpy arrays
pos_target = np.array(pos_target)
pos_missile = np.array(pos_missile)
vel_missile = np.array(vel_missile)
rel_distance = np.array(rel_distance)
command = np.array(command)
run_time = np.array(run_time)


# plot target and missile positions
fig1, ax1 = plt.subplots(subplot_kw={'projection': '3d'})
ax1.plot(pos_target[:, 0]/1000, pos_target[:, 1]/1000, pos_target[:, 2]/1000, label='Target')
ax1.plot(pos_missile[:, 0]/1000, pos_missile[:, 1]/1000, pos_missile[:, 2]/1000, label='Missile')
ax1.set_xlabel('x (km)')
ax1.set_ylabel('y (km)')
ax1.set_zlabel('z (km)')
ax1.legend()
ax1.set_title('Target and Missile Positions')

# plot relative distance
fig2, ax2 = plt.subplots()
ax2.plot(run_time, rel_distance)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Relative Distance [m]')
ax2.set_title('Relative Distance vs Time')

# plot control commands
fig3, ax3 = plt.subplots()
ax3.plot(run_time, command[:, 0], label='a_x')
ax3.plot(run_time, command[:, 1], label='a_y')
ax3.plot(run_time, command[:, 2], label='a_z')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Control Commands (a_cmd)')
ax3.legend()
ax3.set_title('Control Commands a_cmd vs Time')

# plot missile velocity
fig4, ax4 = plt.subplots()
ax4.plot(run_time, vel_missile[:, 0], label='vx')
ax4.plot(run_time, vel_missile[:, 1], label='vy')
ax4.plot(run_time, vel_missile[:, 2], label='vz')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('missile velocity (m/s)')
ax4.legend()
ax4.set_title('Missile Velocity')


# plot missile speed
fig5, ax5 = plt.subplots()
ax5.plot(run_time, sp_missile)
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('missile velocity (m/s)')
ax5.set_title('Missile Velocity')

# display all plots
plt.show()