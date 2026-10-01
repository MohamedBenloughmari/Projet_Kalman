import kalman
import numpy as np
import graph_helper as gh;
import matplotlib.pyplot as plt

N = 40

has_measure = np.ones(N + 1, dtype=bool)
has_measure[11:21] = False 
XTrue = np.zeros((N + 1, 6))
XTrue[0] = np.array([20.,
                     -10.,
                     5.0,
                     0.,
                     5.,
                     0.])
Y = np.zeros((N+1, 3))
XEst = np.zeros((N + 1, 6))
PEst = np.zeros((N + 1, 6, 6))

X0 = np.zeros((6,))
P0 = np.zeros((6,6))
P0[0,0] =10000
P0[1,1] =10000
P0[2,2] =10000
P0[3,3] =100
P0[4,4] =100
P0[5,5] =10

XEst[0] = X0
PEst[0] = P0

DeltaT = 1

Fk = np.eye(6,6)
Fk[0,3] = DeltaT
Fk[1,4] = DeltaT
Fk[2,5] = DeltaT

Qk = np.eye(6,6)
Qk[0,0] = 0.0001
Qk[1,1] = 0.0001
Qk[2,2] = 0.0001
Qk[3,3] = 0.01
Qk[4,4] = 0.01
Qk[5,5] = 0.01

Hk = np.zeros((3,6))
Hk[0,0] = 1
Hk[1,1] = 1
Hk[2,2] = 1

Rk = np.array([
    [100.0, 0.0, 0.0],
    [0.0, 100.0, 0.0],
    [0.0,0.0,100.0]])
Qk=1*Qk
Rk=10*Rk
print("XTrue0",XTrue[0])
print("X0",X0)
print("P0",P0)
print("Fk",Fk)
print("Qk",Qk)
print("Hk",Hk)
print("Rk",Rk)

kf = kalman.KalmanFilter(X0, P0)

def F(k):
    return Fk

def Q(k):
    return Qk

def H(k):
    return Hk

def R(k):
    return Rk


def simulate_x_step(k):
    global XTrue
    noise = np.linalg.cholesky(Qk) @ np.random.randn(6)
    XTrue[k+1] = Fk @ XTrue[k] + noise

def simulate_y(k):
    global XTrue, Y
    Y[k] = Hk @ XTrue[k] + np.linalg.cholesky(Rk) @ np.random.randn(3)


for k in range(N):
    simulate_x_step(k)

    kf.predict(F(k), Q(k))
    if has_measure[k+1]:
        simulate_y(k+1)
        innovation, S, K = kf.update(Y[k+1], H(k+1), R(k+1))

    XEst[k+1] = kf.x
    PEst[k+1] = kf.P

fig, axes = plt.subplots(2, 3, sharex=True)


t = np.arange(N+1) - 1

x_true = XTrue[:,0]
y_true = XTrue[:,1]
z_true = XTrue[:,2]
vx_true = XTrue[:,3]
vy_true = XTrue[:,4]
vz_true = XTrue[:,5]

x_est = XEst[:,0]
y_est = XEst[:,1]
z_est = XEst[:,2]
vx_est = XEst[:,3]
vy_est = XEst[:,4]
vz_est = XEst[:,5]

sigma2_x = PEst[:,0,0]
sigma2_y = PEst[:,1,1]
sigma2_z = PEst[:,2,2]
sigma2_vx = PEst[:,3,3]
sigma2_vy = PEst[:,4,4]
sigma2_vz = PEst[:,5,5]
sigma_x = sigma2_x ** 0.5
sigma_y = sigma2_y ** 0.5
sigma_z = sigma2_z ** 0.5
sigma_vx = sigma2_vx ** 0.5
sigma_vy = sigma2_vy ** 0.5
sigma_vz = sigma2_vz ** 0.5


gh.plot_with_interval(
    t,
    curves=[
        {"y": x_true, "label": "x réelle", "color": "blue"},
        {"y": x_est,  "label": "x estimée", "color": "red"}
    ],
    lower=x_est - 3 * sigma_x,
    upper=x_est + 3 * sigma_x,
    interval_label=r"incertitude ($3\sigma$)",
    ylabel="position (m)",
    ax=axes[0,0]
)

gh.plot_with_interval(
    t,
    curves=[
        {"y": y_true, "label": "y réelle", "color": "blue"},
        {"y": y_est,  "label": "y estimée", "color": "red"}
    ],
    lower=y_est - 3 * sigma_y,
    upper=y_est + 3 * sigma_y,
    interval_label=r"incertitude ($3\sigma$)",
    ylabel="position (m)",
    ax=axes[0,1]
)

gh.plot_with_interval(
    t,
    curves=[
        {"y": z_true, "label": "z réelle", "color": "blue"},
        {"y": z_est,  "label": "z estimée", "color": "red"}
    ],
    lower=z_est - 3 * sigma_z,
    upper=z_est + 3 * sigma_z,
    interval_label=r"incertitude ($3\sigma$)",
    ylabel="position (m)",
    ax=axes[0,2]
)

gh.plot_with_interval(
    t,
    curves=[
        {"y": vx_true, "label": "vx réelle", "color": "blue"},
        {"y": vx_est,  "label": "vx estimée", "color": "red"}
    ],
    lower=vx_est - 3 * sigma_vx,
    upper=vx_est + 3 * sigma_vx,
    interval_label=r"incertitude ($3\sigma$)",
    ylabel="position (m)",
    ax=axes[1,0]
)

gh.plot_with_interval(
    t,
    curves=[
        {"y": vy_true, "label": "vy réelle", "color": "blue"},
        {"y": vy_est,  "label": "vy estimée", "color": "red"}
    ],
    lower=vy_est - 3 * sigma_vy,
    upper=vy_est + 3 * sigma_vy,
    interval_label=r"incertitude ($3\sigma$)",
    ylabel="position (m)",
    ax=axes[1,1]
)

gh.plot_with_interval(
    t,
    curves=[
        {"y": vz_true, "label": "vz réelle", "color": "blue"},
        {"y": vz_est,  "label": "vz estimée", "color": "red"}
    ],
    lower=vz_est - 3 * sigma_vz,
    upper=vz_est + 3 * sigma_vz,
    interval_label=r"incertitude ($3\sigma$)",
    ylabel="position (m)",
    ax=axes[1,2]
)

plt.tight_layout()
plt.show()
