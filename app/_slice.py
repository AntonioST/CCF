import abc

import numpy as np

__all__ = ['SliceView', 'CoronalView', 'SagittalView', 'TransverseView']


class SliceView(metaclass=abc.ABCMeta):
    def __init__(self, name: str, reference: np.ndarray):
        self.name = name
        self.reference = reference
        self.grid_y, self.grid_x = np.mgrid[0:self.height, 0:self.width]

    @property
    def n_ap(self) -> int:
        return self.reference.shape[0]

    @property
    def n_dv(self) -> int:
        return self.reference.shape[1]

    @property
    def n_ml(self) -> int:
        return self.reference.shape[2]

    @property
    @abc.abstractmethod
    def n_frame(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def width(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def height(self) -> int:
        pass

    @abc.abstractmethod
    def plane(self, o: np.ndarray) -> np.ndarray:
        pass

    def offset(self, h: int, v: int) -> np.ndarray:
        x_frame = np.round(np.linspace(-h, h, self.width)).astype(int)
        y_frame = np.round(np.linspace(-v, v, self.height)).astype(int)

        return x_frame[None, :] + y_frame[:, None]


class CoronalView(SliceView):

    def __init__(self, reference: np.ndarray):
        super().__init__('Coronal', reference)

    @property
    def n_frame(self) -> int:
        return self.n_ap

    @property
    def width(self) -> int:
        return self.n_ml

    @property
    def height(self) -> int:
        return self.n_dv

    def plane(self, o: np.ndarray) -> np.ndarray:
        return self.reference[o, self.grid_y, self.grid_x]


class SagittalView(SliceView):
    def __init__(self, reference: np.ndarray):
        super().__init__('Sagittal', reference)

    @property
    def n_frame(self) -> int:
        return self.n_ml

    @property
    def width(self) -> int:
        return self.n_ap

    @property
    def height(self) -> int:
        return self.n_dv

    def plane(self, o: np.ndarray) -> np.ndarray:
        return self.reference[self.grid_x, self.grid_y, o]


class TransverseView(SliceView):
    def __init__(self, reference: np.ndarray):
        super().__init__('Transverse', reference)

    @property
    def n_frame(self) -> int:
        return self.n_dv

    @property
    def width(self) -> int:
        return self.n_ml

    @property
    def height(self) -> int:
        return self.n_ap

    def plane(self, o: np.ndarray) -> np.ndarray:
        return self.reference[self.grid_y, o, self.grid_x]
