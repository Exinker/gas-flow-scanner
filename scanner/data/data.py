import os
import pickle
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from scanner.data.utils import fetch_filedir
from scanner.exception import LoadError
from scanner.typing import Array, Inch, MicroMeter, MilliMeter, MilliSecond, Hz, Percent, Second


WIDTH: MicroMeter = 7  # detector's pixel width


@dataclass(frozen=False)
class DataMeta:
    tau: MilliSecond
    factor: int
    dt: datetime = field(default_factory=datetime.now)

    xoffset: MicroMeter = field(default=0)
    xscale: float = field(default=1)
    zoffset: MicroMeter = field(default=0)
    zscale: float = field(default=1)
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
        return f'{cls.__name__}(tau={self.tau}, factor={self.factor}, label={repr(self.label)})'


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
    def xvalue(self) -> Array[MilliMeter]:
        return self.meta.xoffset + self.time*self.meta.xscale

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
    def zvalue(self) -> Array[MilliMeter]:
        return self.meta.zoffset + self.number*WIDTH/1000*self.meta.zscale

    @property
    def shape(self) -> tuple[int, int]:
        """Размерность данынх."""
        return self.intensity.shape

    # --------        handler        --------
    def show(
        self,
        levels: int | Sequence[float] | None = None,
        figsize: tuple[Inch, Inch] = (8, 4),
        n_xticks: int = 11,
        n_yticks: int = 7,
        clim: tuple[float, float] | None = None,
        cmap: mpl.colors.LinearSegmentedColormap | None = None,
        save: bool = True,
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        # image
        clim = clim or (-.1, 100)
        cmap = cmap or cm.viridis

        if levels is None:
            plt.imshow(
                self.intensity.T,
                origin='lower',
                interpolation='none',
                cmap=cmap,
                clim=clim,
                aspect='auto',
            )
        else:
            plt.contourf(
                self.intensity.T,
                levels=levels,
                cmap=cmap,
            )

        # colorbar
        plt.colorbar()

        # ticks
        ax.set_xticks(np.arange(0, self.n_times, self.n_times//(n_xticks - 1)))
        ax.set_xticklabels([f'{self.xvalue[n]}' for n in ax.get_xticks()])

        ax.set_yticks(np.arange(0, self.n_numbers, self.n_numbers//(n_yticks - 1)))
        ax.set_yticklabels([f'{self.zvalue[n]:.1f}' for n in ax.get_yticks()])

        # labels
        plt.xlabel(r'$x$, $mm$')
        plt.ylabel(r'$z$, $mm$')

        #
        if save:
            filedir = fetch_filedir(kind='img')
            filepath = os.path.join(filedir, f'{self.meta.label}.png')
            plt.savefig(filepath, dpi=300)

        #
        plt.show()

    def save(self, mode: Literal['binary', 'txt'] = 'binary'):
        """Сохранить объект в файл."""

        filedir = fetch_filedir(kind='data')
        match mode:
            case 'binary':
                filepath = os.path.join(filedir, f'{self.meta.label}.pkl')

                with open(filepath, 'wb') as file:
                    pickle.dump(self, file)
            case 'txt':
                filepath = os.path.join(filedir, f'{self.meta.label}.txt')

                np.savetxt(
                    filepath,
                    self.intensity,
                )
            case _:
                raise ValueError(f'Mode: {mode} is not supported yet!')

    # --------        fabric        --------
    @classmethod
    def load(cls, filepath: str) -> 'Data':
        """Прочитать объект из файла."""

        with open(filepath, 'rb') as file:
            result = pickle.load(file)

        if not isinstance(result, cls):
            raise LoadError(filepath)

        return result

    # --------        private        --------

    def __repr__(self) -> str:
        cls = self.__class__
        return f'{cls.__name__}(n_times={self.n_times}, n_numbers={self.n_numbers})'
