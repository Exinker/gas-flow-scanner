import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from scanner.exception import LoadError
from scanner.typing import Array, Inch, MicroMeter, MilliMeter, MilliSecond, Hz, Percent, Second


WIDTH: MicroMeter = 7  # detector's pixel width


@dataclass(frozen=True)
class DataMeta:
    tau: MilliSecond
    factor: int
    dt: datetime = field(default_factory=datetime.now)

    velocity: float = field(default=None)  # in mm/s
    comment: str = field(default=None)

    @property
    def omega(self) -> Hz:
        return 1e+3 / (self.tau * self.factor)

    @property
    def label(self) -> str:
        if self.comment is None:
            return self.dt.strftime('%y.%m.%d %H.%M.%S')

        return '{dt} ({comment})'.format(
            dt=self.dt.strftime('%y.%m.%d %H.%M.%S'),
            comment=self.comment,
        )

    # --------        private        --------
    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}(tau={self.tau}, factor={self.factor}, velocity={self.velocity}, label={repr(self.label)})'


@dataclass
class Data:
    """Сырые данные, полученные c детектора излучения."""
    intensity: Array[Percent]  # Двумерный массив данных измерения. Первый индекс - номер кадра, второй - номер отсчета в кадре
    clipped: Array[bool]  # Двумерный массив boolean значений. Если `clipped[i,j] == True`, то `intensity[i,j]` содержит зашкаленное значение
    meta: DataMeta

    def __post_init__(self):
        self._time = self._time = np.arange(self.n_times) / self.meta.omega
        self._number = np.arange(self.n_numbers)

    @property
    def n_times(self) -> int:
        """Количество измерений."""
        if self.intensity.ndim == 1:
            return 1
        return self.intensity.shape[0]

    @property
    def time(self) -> Array[Second]:
        return self._time

    @property
    def distance(self) -> Array[float] | Array[MilliMeter]:
        return self.time * self.meta.velocity

    @property
    def n_numbers(self) -> int:
        """Количество отсчетов."""
        if self.intensity.ndim == 1:
            return self.intensity.shape[0]
        return self.intensity.shape[1]

    @property
    def number(self) -> Array[int]:
        return self._number

    @property
    def shape(self) -> tuple[int, int]:
        """Размерность данынх."""
        return self.intensity.shape

    # --------        handler        --------
    def show(self, figsize: tuple[Inch, Inch] = (8, 4), n_xticks: int = 11, n_yticks: int = 5) -> None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        # imshow
        image = plt.imshow(
            self.intensity.T,
            origin='lower',
            # interpolation='none',
            # cmap=cmap, clim=(-.01, .5),
            aspect='auto',
        )
        fig.colorbar(image, ax=ax)

        # ticks
        xarray = self.time if self.meta.velocity is None else self.distance
        ax.set_xticks(np.arange(0, self.n_times, self.n_times//(n_xticks - 1)))
        ax.set_xticklabels([f'{xarray[n]}' for n in ax.get_xticks()])

        yarray = WIDTH*self.number/1000
        ax.set_yticks(np.arange(0, self.n_numbers, self.n_numbers//(n_yticks - 1)))
        ax.set_yticklabels([f'{yarray[n]:.1f}' for n in ax.get_yticks()])

        # labels
        if self.meta.velocity is None:
            plt.xlabel(r'time [$s$]')
        else:
            plt.xlabel(r'$h$ [$mm$]')

        plt.ylabel(r'$x$ [$mm$]')

        #
        plt.show()

    def save(self, path: str):
        """Сохранить объект в файл."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    # --------        fabric        --------
    @classmethod
    def load(cls, path: str) -> 'Data':
        """Прочитать объект из файла."""

        with open(path, 'rb') as f:
            result = pickle.load(f)

        if not isinstance(result, cls):
            raise LoadError(path)

        return result

    # --------        private        --------
    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}(n_times={self.n_times}, n_numbers={self.n_numbers})'
