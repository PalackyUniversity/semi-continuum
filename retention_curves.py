import numpy as np


class RetentionCurves:
    MAX_SATURATION = 0.999
    MIN_SATURATION = 0.001

    def __curve(self, s: np.ndarray, a: float) -> np.ndarray:
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
        pressure = a * (-100 * np.log(1 / s - 1))
        pressure[condition] = -125 * c * (s[condition] - 1)  # slope: 8.97e4

        return pressure

    def wet(self, s: np.ndarray, a: float) -> np.ndarray:
        """Logistic retention curve for the main wetting branch."""
        return self.__curve(s, a) - 700

    def drain(self, s: np.ndarray, a: float) -> np.ndarray:
        """Logistic retention curve for the main draining branch."""
        return self.__curve(s, a) - 1300

    @staticmethod
    def van_genuchten(s: np.ndarray, alfa, n, a: float, rho, g: float) -> np.ndarray:
        """
        Van Genuchten retention curve.

        :param s: saturation
        :param alfa: first parameter of the retention curve (the main draining or wetting branch)
        :param n: second parameter of the retention curve (the main draining or wetting branch)
        :param a: multiplication used for the scaling of the retention curve
        :param rho: density of water
        :param g: acceleration due to gravity
        :return: pressure [Pa
        """

        m = 1 - 1 / n

        # The scaling of the retention curve around the point S = 0.5.
        value1 = -(1 / alfa) * ((0.5 ** (-1. / m) - 1) ** (1. / n))
        value2 = -(a / alfa) * ((0.5 ** (-1. / m) - 1) ** (1. / n))

        pressure = -(a / alfa) * ((s ** (-1. / m) - 1) ** (1. / n)) + (value1 - value2)
        pressure = pressure * (rho * g) / 100

        return pressure

    def plot(self, function):
        pass


if __name__ == "__main__":  # TODO sanity check - visualize the retention curve
    pass
