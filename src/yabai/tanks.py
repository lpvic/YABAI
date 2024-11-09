# This file is part of YABAI.

# YABAI is free software: you can redistribute it and/or modify it under the terms of the Affero GNU General Public
# License # as published by the Free Software Foundation, either version 3 of the License, or any later version.

# YABAI is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Affero GNU General Public License for more details.

# You should have received a copy of the Affero GNU General Public License along with YABAI.
# If not, see <https://www.gnu.org/licenses/>.

class Gas:
    """Gas mix contained in the scuba tank.

    Args:
        o2: Oxigen content in the gas mixture, expressed in percentage, defaults to 21.
        he: Hellium content in the gas mixture, expressed in percentage, defaults to 0.
    """

    def __init__(self, o2: int = 21, he: int = 0) -> None:
        self._O2: int = o2
        self._He: int = he
        self._N2: int = 100 - o2 - he

    def ppO2(self, depth: float) -> float:
        """Calculates the oxigen partial pressure of the gas mix at a given depth.

        Args:
            depth: Depth to calculate oxigen partial pressure of the gas mix (in meters).

        Returns:
            The oxigen partial pressure of the gas mix at the given depth.
        """
        pabs = (depth / 10) + 1
        return pabs * self._O2 / 100

    def ppN2(self, depth: float) -> float:
        """Calculates the nitrogen partial pressure of the gas mix at a given depth.

        Args:
            depth: Depth to calculate nitrogen partial pressure of the gas mix (in meters).

        Returns:
            The nitrogen partial pressure of the gas mix at the given depth.
        """
        pabs = (depth / 10) + 1
        return pabs * self._N2 / 100

    def ppHe(self, depth: float) -> float:
        """Calculates the hellium partial pressure of the gas mix at a given depth.

        Args:
            depth: Depth to calculate hellium partial pressure of the gas mix (in meters).

        Returns:
            The hellium partial pressure of the gas mix at the given depth.
        """
        pabs = (depth / 10) + 1
        return pabs * self._He / 100

    def mod(self, pp_o2=1.4) -> float:
        """Calculates the Maximum Operating Depth (MOD) of the gas mix at a given oxigen partial pressure.

        Args:
            pp_o2: Oxigen partial pressure to calculate MOD.

        Returns:
            The maximum operating depth of the gas mix at the given oxigen partial pressure.
        """
        return 10 * ((pp_o2 / (self._O2 / 100)) - 1)

    @property
    def O2(self) -> int:
        """Oxigen content of the gas mix (in percentage)."""
        return self._O2

    @property
    def He(self) -> int:
        """Hellium content of the gas mix (in percentage)."""
        return self._He

    @property
    def N2(self) -> int:
        """Nitrogen content of the gas mix (in percentage)."""
        return self._N2

    @property
    def fO2(self) -> float:
        """Oxigen fraction of the gas mix (decimal)."""
        return self._O2 / 100.

    @property
    def fN2(self) -> float:
        """Nitrogen fraction of the gas mix (decimal)."""
        return self._N2 / 100.

    @property
    def fHe(self) -> float:
        """Hellium fraction of the gas mix (decimal)."""
        return self._He / 100.

    def __str__(self) -> str:
        if (self._O2 == 21) and (self._He == 0):
            return 'Air'
        elif self._He == 0:
            return 'EAN{}'.format(self._O2)
        else:
            return 'Trimix{}/{}'.format(self._O2, self._He)

    def __repr__(self) -> str:
        return '<Gas Mixture: O2: {} N2: {} He: {}>'.format(self._O2, self._N2, self._He)


class Tank:
    """A scuba tank.

    Args:
        start_pressure: Tank start pressure.
        gas: Gas mix in the tank.
        size: Size of the tank, in liters.
    """

    def __init__(self, start_pressure: int = 200, gas: Gas = Gas(), size: int = 15) -> None:
        self._gas = gas
        self.start_pressure = start_pressure
        self._size = size

    @property
    def gas(self) -> Gas:
        """Gas mix in the tank."""
        return self._gas

    @property
    def size(self) -> int:
        """Size of the tank, in liters."""
        return self._size
