import numpy as np

class KerrMetric:
    def __init__(self, M, S):
        self.mass = M
        self.spin = S
        self.a = S/M
        self.rHorizon = self.mass + np.sqrt(self.mass - self.a**2)

    def HorizonMetric(self, theta, phi):
        sigma = self.rHorizon**2 + self.a**2*np.cos(theta)**2
        gphph = (self.rHorizon**2 + self.a**2)**2/sigma*np.sin(theta)**2
        gthth = sigma
        gthph = np.zeros(theta.shape)
        return gthth, gphph, gthph

    def HorizonRicci(self, theta, phi):
        return 2*(self.rHorizon**2 + self.a**2)*(self.rHorizon**2 - 3*self.a**2*np.cos(theta)**2)/((self.rHorizon**2.0 + (self.a*np.cos(theta))**2.0)**3.0)  #http://arxiv.org/pdf/0706.0622.pdf
