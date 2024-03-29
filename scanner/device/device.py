import os
from decimal import Decimal

import matplotlib.pyplot as plt
from IPython import display

from pyspectrum.data import Data as Raw
from pyspectrum.device_factory import UsbID
from pyspectrum.spectrometer import FactoryConfig, Spectrometer

from scanner.data import Data, DataMeta
from scanner.typing import Digit, Hz, MilliSecond


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
    def value_max(self) -> Digit:
        """Макс значение АЦП."""
        return 2**16 - 1

    @property
    def scale(self) -> float:
        """Коэффициент перевода выходного сигнала (`Digit`) в интенсивность (`Percent`)."""
        return 100 / self.value_max


class Device:

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._device = Spectrometer(
            UsbID(),
            factory_config=FactoryConfig.load(os.path.join(os.path.split(os.path.abspath(__file__))[0], 'factory_config.json')),
        )

        self._wavelength = None
        self._dark = None

    @property
    def config(self) -> DeviceConfig:
        return self._config

    @property
    def dark(self) -> Data:
        return self._dark

    # --------        handler        --------
    def view(self, n_frames: int = 1) -> None:

        while True:

            # raw
            raw = self._read(n_frames)

            intensity = raw.intensity.mean(axis=0) * self.config.scale
            if self.dark:
                intensity -= self.dark.intensity

            clipped = raw.clipped.max(axis=0)
            if self.dark:
                clipped = clipped | self.dark.clipped

            data = Data(
                intensity=intensity,
                clipped=clipped,
                meta=DataMeta(
                    tau=self.config.tau,
                    factor=n_frames,
                ),
            )

            # show
            display.clear_output(wait=True)

            figure, ax = plt.subplots(figsize=(8, 4), tight_layout=True)

            plt.plot(
                intensity,
                color='black', linestyle='-',
            )

            mask = clipped == True
            plt.plot(
                data.number[mask], data.intensity[mask],
                color='red', linestyle='none', marker='.', markersize=4,
            )

            plt.xlabel('number')
            plt.ylabel('$I$, %')

            plt.grid(color='grey', linestyle=':')
            plt.pause(.001)

    def read(
        self,
        exposure: MilliSecond,
        velocity: float | None = None,  # in mm/s
        comment: str = None,
    ) -> Data:
        """Начать чтение в течение `exposure` мс."""
        assert isinstance(exposure, int), 'Время экспозиции `exposure` должно быть целым числом!'
        assert exposure % self.config.tau == 0, 'Время экспозиции `exposure` должно быть кратно базовой экспозиции `tau`!'
        assert exposure // self.config.tau % self.config.buffer_size == 0, 'Время экспозиции `exposure` должно быть кратно частоте регистрации `omega`!'

        n_frames = exposure // self.config.tau

        # read
        raw = self._read(n_frames)

        intensity = raw.intensity.reshape((-1, self.config.buffer_size, raw.n_numbers)).mean(axis=1) * self.config.scale
        clipped = raw.clipped.reshape((-1, self.config.buffer_size, raw.n_numbers)).max(axis=1)

        if self.dark:
            intensity = intensity - self.dark.intensity
            clipped = clipped | self.dark.clipped

        #
        return Data(
            intensity=intensity,
            clipped=clipped,
            meta=DataMeta(
                tau=self.config.tau,
                factor=self.config.buffer_size,
                velocity=velocity,
                comment=comment,
            ),
        )

    def calibrate_dark(self, n_frames: int = 1_000, show: bool = True) -> None:

        # read
        raw = self._read(n_frames)

        # dark
        intensity = raw.intensity.mean(axis=0) * self.config.scale
        clipped = raw.clipped.max(axis=0)

        dark = Data(
            intensity=intensity,
            clipped=clipped,
            meta=DataMeta(
                tau=self.config.tau,
                factor=n_frames,
            ),
        )

        # show
        if show:
            figure, ax = plt.subplots(figsize=(8, 4), tight_layout=True)

            plt.plot(
                dark.number, dark.intensity,
                color='black', linestyle='-',
            )

            mask = clipped == True
            plt.plot(
                dark.number[mask], dark.intensity[mask],
                color='red', linestyle='none', marker='.', markersize=4,
            )

            plt.xlabel('number')
            plt.ylabel('$I_d$, %')

            plt.grid(color='grey', linestyle=':')
            plt.show()

        #
        self._dark = dark

    # --------        private        --------
    def _read(self, n_frames: int = 1) -> Raw:
        """Начать чтение в течение `exposure` мс."""

        # setup
        self._device.set_config(self.config.tau, n_frames)

        # read
        raw = self._device.read_raw()

        #
        return raw
