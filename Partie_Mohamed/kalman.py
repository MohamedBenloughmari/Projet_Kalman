import numpy as np


class KalmanFilter:
    def __init__(self, x0, P0):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)

    def predict(self, F, Q):
        """
        Prediction:
            x_k|k-1 = F x_k-1
            P_k|k-1 = F P_k-1 F.T + Q
        """
        self.x=F@self.x
        self.P=F@self.P@F.T+Q

        
    def update(self, y, H, R):
        """
        Correction:
            innovation = y - H x
            S = H P H.T + R
            K = P H.T S^-1
        """

        innovation = y-H@self.x
        S = H@self.P@H.T+R
        K = self.P@H.T@np.linalg.inv(S)
        self.x=self.x+K@innovation
        #self.P=(np.eye(len(K@H))-K@H)@self.P
        self.P=(np.eye(len(K@H))-K@H)@self.P@(np.eye(len(K@H))-K@H).T+K@R@K.T

        return innovation, S, K
