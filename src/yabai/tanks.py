class Gas:
    def __init__(self, o2: int = 21, he: int = 0) -> None:
        self._O2: int = o2
        self._He: int = he
        self._N2: int = 100 - o2 - he

    def ppO2(self, depth: float) -> float:
        pabs = (depth / 10) + 1
        return pabs * self._O2 / 100

    def ppN2(self, depth: float) -> float:
        pabs = (depth / 10) + 1
        return pabs * self._N2 / 100

    def ppHe(self, depth: float) -> float:
        pabs = (depth / 10) + 1
        return pabs * self._He / 100

    def mod(self, pp_o2=1.4) -> float:
        return 10 * ((pp_o2 / (self._O2 / 100)) - 1)

    @property
    def O2(self) -> int:
        return self._O2

    @property
    def He(self) -> int:
        return self._He

    @property
    def N2(self) -> int:
        return self._N2

    @property
    def fO2(self) -> float:
        return self._O2 / 100.

    @property
    def fN2(self) -> float:
        return self._N2 / 100.

    @property
    def fHe(self) -> float:
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
    def __init__(self, start_pressure: int = 200, gas: Gas = Gas(), size: int = 15) -> None:
        self._gas = gas
        self.start_pressure = start_pressure
        self._size = size

    @property
    def gas(self) -> Gas:
        return self._gas

    @property
    def size(self) -> int:
        return self._size
