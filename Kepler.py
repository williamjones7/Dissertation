import numpy as np

def SolveKepler(e, M):
    E0 = M
    while True:
        E1 = E0 - (E0 - e * np.sin(E0) - M) / (1 - e * np.cos(E0))
        if np.linalg.norm(E1 - E0) < 1e-8:
            return E1
        E0 = E1

def KeplerOrbit(a, e, mu, t):
    T = KeplerPeriod(a, mu)
    M = 2 * np.pi * t / T
    E = SolveKepler(e, M)
    theta = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), np.sqrt(1 - e) * np.cos(E/2))
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)
    return np.column_stack((x, y, z))

def SemiMajorAxis(mu, r0s, v0s):
    r = r0s[1, :]
    v = v0s[1, :]
    R = np.linalg.norm(r)
    a = mu * R / (2 * mu - R * (v[0] ** 2 + v[1] ** 2))
    return a

def KeplerPeriod(a, mu): return 2 * np.pi * np.sqrt(a**3 / mu)