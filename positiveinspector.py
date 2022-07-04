import unittest
from typing import List
from decimal import Decimal
import numpy as np


class AtomSettingsCase(object):
    """
    This class stores individual atoms parameters independent of unit cell,
    including atom type as well as individual xyz, Uij, Cijk, and Dijkl values.
    """
    def __init__(self, **kwargs):
        self.Z: int = 42
        self.x: Decimal = Decimal('0.5')
        self.y: Decimal = Decimal('0.5')
        self.z: Decimal = Decimal('0.5')
        self.U11: Decimal = Decimal('0.1')
        self.U22: Decimal = Decimal('0.1')
        self.U33: Decimal = Decimal('0.1')
        self.U12: Decimal = Decimal('0.0')
        self.U13: Decimal = Decimal('0.0')
        self.U23: Decimal = Decimal('0.0')
        self.C111: Decimal = Decimal('0.0')
        self.C222: Decimal = Decimal('0.0')
        self.C333: Decimal = Decimal('0.0')
        self.C112: Decimal = Decimal('0.0')
        self.C122: Decimal = Decimal('0.0')
        self.C113: Decimal = Decimal('0.0')
        self.C133: Decimal = Decimal('0.0')
        self.C223: Decimal = Decimal('0.0')
        self.C233: Decimal = Decimal('0.0')
        self.C123: Decimal = Decimal('0.0')
        self.D1111: Decimal = Decimal('0.0')
        self.D2222: Decimal = Decimal('0.0')
        self.D3333: Decimal = Decimal('0.0')
        self.D1112: Decimal = Decimal('0.0')
        self.D1222: Decimal = Decimal('0.0')
        self.D1113: Decimal = Decimal('0.0')
        self.D1333: Decimal = Decimal('0.0')
        self.D2223: Decimal = Decimal('0.0')
        self.D2333: Decimal = Decimal('0.0')
        self.D1122: Decimal = Decimal('0.0')
        self.D1133: Decimal = Decimal('0.0')
        self.D2233: Decimal = Decimal('0.0')
        self.D1123: Decimal = Decimal('0.0')
        self.D1223: Decimal = Decimal('0.0')
        self.D1233: Decimal = Decimal('0.0')


class CellSettingsCase(object):
    """
    This class stores all parameters concerning unit cell to be tested
    for positiveness, including unit cell parameters and atoms inside it.
    """
    def __init__(self):
        self.a: Decimal = Decimal(10)
        self.b: Decimal = Decimal(10)
        self.c: Decimal = Decimal(10)
        self.al: Decimal = Decimal(90)
        self.be: Decimal = Decimal(90)
        self.ga: Decimal = Decimal(90)
        self.atoms: List[AtomSettingsCase]

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
