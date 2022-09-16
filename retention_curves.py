import numpy as np


class Curve:
    def calculate(self, s: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def plot(self, function):
        pass  # TODO


class RetentionCurve(Curve):
    MAX_SATURATION = 0.999
    MIN_SATURATION = 0.001

    OFFSET = NotImplemented

    def __init__(self, a: float):
        self.A = a

    def calculate(self, s: np.ndarray) -> np.ndarray:
        """
        Logistic retention curve

        If the saturation is greater than 0.999 or lower than 0.001, some
        modification of the function should be used due to a very high slope.
        Parameter Kps should not be lower than the slope of the retention curve.
        Therefore, the van Genuchten retention curve is recommended.

        :param s: saturation
        :param a: multiplication used for the scaling of the retention curve
        :return: pressure
        """

        # TODO nezpomalí se to kvůli tomuto?
        condition = self.MAX_SATURATION > s > self.MIN_SATURATION
        assert condition.any(), "Saturation out of range"

        c = -100 * np.log(1. / self.MAX_SATURATION - 1) - 1200  # c = -509.3245
        pressure = self.A * (-100 * np.log(1 / s - 1))
        pressure[condition] = -125 * c * (s[condition] - 1)  # slope: 8.97e4

        return pressure - self.OFFSET


class RetentionCurveWet(RetentionCurve):
    """Logistic retention curve for the main wetting branch."""
    OFFSET = 700


class RetentionCurveDrain(RetentionCurve):
    """Logistic retention curve for the main draining branch."""
    OFFSET = 1300


class VanGenuchten(Curve):
    """
    Van Genuchten retention curve.

    ALFA: first parameter of the retention curve (the main draining or wetting branch)
    N: second parameter of the retention curve (the main draining or wetting branch)
    A: multiplication used for the scaling of the retention curve
    RHO_G: density of water * acceleration due to gravity
    :return: pressure [Pa]
    """

    ALFA: float = NotImplemented
    N: float = NotImplemented

    def __init__(self, a: float, rho_g: float) -> None:
        self.A = a
        self.RHO_G = rho_g / 100  # TODO zde se možná ubere přesnost??
        self.N_INVERSE = 1 / self.N
        self.M = -1. / (1 - self.N_INVERSE)
        self.T = ((0.5 ** self.M - 1) ** self.N_INVERSE)

        # The scaling of the retention curve around the point S = 0.5.
        self.A_ALFA = -(self.A / self.ALFA)

        value1 = -(1 / self.ALFA) * self.T
        value2 = self.A_ALFA * self.T

        self.VALUE_DIFF = (value1 - value2) * self.RHO_G
        self.A_ALFA_RHO_G = self.A_ALFA * self.RHO_G

    def calculate(self, s: np.ndarray) -> np.ndarray:
        """
        Van Genuchten retention curve.
        :param s: saturation
        :return: pressure [Pa]
        """
        return self.A_ALFA_RHO_G * ((s ** self.M - 1) ** self.N_INVERSE) + self.VALUE_DIFF


class VanGenuchtenDrain(VanGenuchten):
    ALFA = 0.0744
    N = 8.47


class VanGenuchtenWet(VanGenuchten):
    ALFA = 0.177
    N = 6.23


if __name__ == "__main__":  # TODO sanity check - visualize the retention curve
    pass
