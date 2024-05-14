import numpy as np

def seeker_meas(rT, vT, rM, vM):
    """ relative position and velocity vector """
    rTM = rT - rM
    vTM = vT - vM

    R2 = rTM.dot(rTM)
    R = np.sqrt(R2)

    """ LOS rate """
    OmL = np.cross(rTM, vTM)/R2

    """ approaching speed """
    Vc = -rTM.dot(vTM) / R

    LOS_hat = rTM / R  # LOS direction vector

    return OmL, LOS_hat, R, Vc