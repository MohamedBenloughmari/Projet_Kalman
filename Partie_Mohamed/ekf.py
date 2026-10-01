import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, x0, P0):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

    def predict(self, f, F_jacobian, Q):
        """
        Modèle:
            x_k = f_k(x_k-1) + w_k

        F_k = df_k/dx évaluée en x_k-1
        """
        self.x=f(self.x)
        self.P=F_jacobian(self.x)@self.P@F_jacobian(self.x).T +Q


    def update(self, y, h, H_jacobian, R):
        """
        Modèle de mesure:
            y_k = h_k(x_k) + v_k

        H_k = dh_k/dx évaluée en x_k|k-1
        """

        innovation = y-h(self.x)
        S = H_jacobian(self.x)@self.P@H_jacobian(self.x).T+R
        K = self.P@H_jacobian(self.x).T@np.linalg.inv(S)
        self.x=self.x+K@innovation
        KH=K@H_jacobian(self.x)
        self.P=(np.eye(len(KH))-KH)@self.P
        self.P=0.5*(self.P+self.P.T)
        return innovation, S, K
