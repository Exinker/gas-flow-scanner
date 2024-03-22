import os
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np

from pyspectrum.device_factory import UsbID
from pyspectrum.spectrometer import Spectrometer, FactoryConfig

from scanner.data import Data, DataMeta
from scanner.typing import Array, Digit, Hz, MilliSecond


class DeviceConfig:

    def __init__(self, omega: Hz, tau: MilliSecond = 2) -> None:
        assert isinstance(omega, (int, float)), 'Частота регистрации `omega` должно быть числом!'
        assert 1 <= omega <= 500, 'Частота регистрации `omega` должна лежать в диапазоне [1; 500] Гц!'
        assert isinstance(tau, (int, float)), 'Базовое время экспозиции `tau` должно быть числом!'
        assert 2 <= tau <= 1_000, 'Базовое время экспозиции `tau` должно лежать в диапазоне [2; 1_000] мс!'

        self._omega = omega
        self._tau = tau
        self._buffer_size = self.calculate_buffer_size(omega=omega, tau=tau)
        self._factor = 1

    @property
    def omega(self) -> float:
        """Частота регистрации (Гц)."""
        return self._omega

    @property
    def tau(self) -> MilliSecond:
        """Базовое время экспозиции (мс)."""
        return self._tau

    @property
    def buffer_size(self) -> int:
        """Количество накоплений по времени."""
        return self._buffer_size

    @staticmethod
    def calculate_buffer_size(omega: Hz, tau: MilliSecond) -> int:
        """Рассчитать размер буфера."""
        assert Decimal(1e+3) / Decimal(omega) % Decimal(tau) == 0, 'Частота регистрации `omega` должна быть кратна базовому времени экспозиции `tau`!'

        return int(Decimal(1e+3) / Decimal(omega) / Decimal(tau))

    @property
    def scale(self) -> float:
        """Коэффициент перевода выходного сигнала (`Digit`) в интенсивность (`Percent`)."""
        return 100 / (2**16 - 1)


class Device:

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._device = Spectrometer(
            UsbID(),
            factory_config=FactoryConfig.load(os.path.join(os.path.split(os.path.abspath(__file__))[0], 'factory_config.json'))
        )

        self._wavelength = None
        self._dark_data = None

    @property
    def config(self) -> DeviceConfig:
        return self._config

    # --------        read        --------
    def read(self, exposure: MilliSecond) -> Data:
        """Начать чтение в течение `exposure` мс."""
        assert isinstance(exposure, int), 'Время экспозиции `exposure` должно быть целым числом!'
        assert exposure % self.config.tau == 0, 'Время экспозиции `exposure` должно быть кратно базовой экспозиции `tau`!'
        assert exposure // self.config.tau % self.config.buffer_size == 0, 'Время экспозиции `exposure` должно быть кратно частоте регистрации `omega`!'

        n_frames = exposure // self.config.tau

        # setup
        self._device.set_config(
            exposure=self.config.tau,
            n_times=n_frames,
        )

        # read
        data = self._device.read_raw()

        intensity = data.intensity.reshape((-1, data.n_numbers, self.config.buffer_size)).mean(axis=2)
        clipped = data.clipped.reshape((-1, data.n_numbers, self.config.buffer_size)).max(axis=2)

        #
        return Data(
            intensity=intensity,
            clipped=clipped,
            meta=DataMeta(
                tau=self.config.tau,
                factor=self.config.buffer_size,
            ),
        )
