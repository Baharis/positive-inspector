import abc
import unittest
from typing import List
from decimal import Decimal
import numpy as np


class MostlyDefaultDataclass(abc.ABC):
    DEFAULTS: dict

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.DEFAULTS.keys():
                raise KeyError(f'Unknown attribute "{k}"')
        for k, v in self.DEFAULTS:
            if k in kwargs.keys():
                v_class = self.DEFAULTS[k].__class__
                self.__setattr__(k, v_class(kwargs[k]))
            else:
                self.__setattr__(k, v)


class AtomSettingsCase(MostlyDefaultDataclass):
    """
    This class stores individual atoms parameters independent of unit cell,
    including atom type as well as individual xyz, Uij, Cijk, and Dijkl values.
    """

    DEFAULTS = {
        'Z': 42,
        'x': Decimal('0.5'),
        'y': Decimal('0.5'),
        'z': Decimal('0.5'),
        'U11': Decimal('0.1'),
        'U22': Decimal('0.1'),
        'U33': Decimal('0.1'),
        'U12': Decimal('0.0'),
        'U13': Decimal('0.0'),
        'U23': Decimal('0.0'),
        'C111': Decimal('0.0'),
        'C222': Decimal('0.0'),
        'C333': Decimal('0.0'),
        'C112': Decimal('0.0'),
        'C122': Decimal('0.0'),
        'C113': Decimal('0.0'),
        'C133': Decimal('0.0'),
        'C223': Decimal('0.0'),
        'C233': Decimal('0.0'),
        'C123': Decimal('0.0'),
        'D1111': Decimal('0.0'),
        'D2222': Decimal('0.0'),
        'D3333': Decimal('0.0'),
        'D1112': Decimal('0.0'),
        'D1222': Decimal('0.0'),
        'D1113': Decimal('0.0'),
        'D1333': Decimal('0.0'),
        'D2223': Decimal('0.0'),
        'D2333': Decimal('0.0'),
        'D1122': Decimal('0.0'),
        'D1133': Decimal('0.0'),
        'D2233': Decimal('0.0'),
        'D1123': Decimal('0.0'),
        'D1223': Decimal('0.0'),
        'D1233': Decimal('0.0'),
    }


class CellSettingsCase(MostlyDefaultDataclass):
    """
    This class stores all parameters concerning unit cell to be tested
    for positiveness, including unit cell parameters and atoms inside it.
    """

    DEFAULTS = {
        'a': 42,
        'b': Decimal('0.5'),
        'c': Decimal('0.5'),
        'al': Decimal('0.5'),
        'be': Decimal('0.1'),
        'ga': Decimal('0.1'),
        'atoms': [AtomSettingsCase()]
    }

    @property
    def xd_par_file_contents(self) -> str:
        return str()

    @property
    def xd_mas_file_contents(self) -> str:
        return str()

    @property
    def olex2_res_file_contents(self) -> str:
        return str()


class PDFGrid(object):
    def __init__(self):
        values: np.ndarray = np.zeros(1)
        negative_volume: float = 0.0

    @classmethod
    def from_xd_cube(cls, csc: CellSettingsCase):
        return cls()

    @classmethod
    def from_nosphera2_cube(cls, csc: CellSettingsCase):
        return cls()

    @property
    def is_positive_definite(self):
        return bool()


class PositiveInspector(unittest.TestCase):
    def test_grids_have_same_sign(self):
        asc_list: List[AtomSettingsCase]
        for asc in asc_list:
            with self.subTest('XD and olex2 report different positivity',
                              asc_=asc):
                csc = CellSettingsCase()
                csc.atoms = [asc]
                grid_xd = PDFGrid.from_xd_cube(csc=csc)
                grid_nosphera2 = PDFGrid.from_nosphera2_cube(csc=csc)
                self.assertEqual(grid_xd.is_positive_definite,
                                 grid_nosphera2.is_positive_definite)
