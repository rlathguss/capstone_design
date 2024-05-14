import numpy as np

dt= 0.01
g = 9.81
class Dof3Dynamics:

    def __init__(self):
        pass

    """ initialization """
    def reset(self, pos, v_mag, azimuth, elevation):

        self.p_vec = np.array(pos).reshape(3, )

        vx0 = v_mag * np.cos(elevation) * np.cos(azimuth)
        vy0 = v_mag * np.cos(elevation) * np.sin(azimuth)
        vz0 = v_mag * np.sin(elevation)
        self.v_vec = np.array([vx0, vy0, vz0]).reshape(3, )

        return np.copy(self.p_vec), np.copy(self.v_vec)  # shape (3,)


    def step(self, a_cmd, option):

        """ compute numerical integration(RK4) """
        xx = np.append(self.p_vec, self.v_vec)

        k1 = dt * self.fn(xx, a_cmd, option)
        k2 = dt * self.fn(xx + dt / 2 * k1, a_cmd, option)
        k3 = dt * self.fn(xx + dt / 2 * k2, a_cmd, option)
        k4 = dt * self.fn(xx + dt * k3, a_cmd, option)
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        xx = xx + k

        self.p_vec = xx[0:3]
        self.v_vec = xx[3:6]

        return np.copy(self.p_vec), np.copy(self.v_vec)

    """ 3DOF dynamics """
    def fn(self, xx, ctrl, option):
        if option == 'target':
            with_g = 0
        else:
            with_g = 1

        v = xx[3:6]

        p_dot = v
        v_dot = ctrl + with_g*np.array(([0, 0, -g]))
        xx_dot = np.append(p_dot, v_dot)
        return xx_dot