import abc
import itertools
import os
import pathlib
import subprocess
import tempfile
import unittest
from collections import UserList, UserDict
from decimal import Decimal
from typing import Iterable
import numpy as np


CURRENT_DIRECTORY = pathlib.Path(__file__).resolve().parent


class MostlyDefaultDict(UserDict, abc.ABC):
    DEFAULTS: dict

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if k not in self.DEFAULTS.keys():
                raise KeyError(f'Unknown attribute "{k}"')
        for k, v in self.DEFAULTS.items():
            if k in kwargs.keys():
                v_class = self.DEFAULTS[k].__class__
                self[k] = v_class(kwargs[k])
            else:
                self[k] = v


class SettingCase(MostlyDefaultDict):
    """
    This class stores unit cell parameters and central Mo(1) atom parameters,
    including its type as well as individual xyz, Uij, Cijk, and Dijkl values.
    """

    DEFAULTS = {
        # UNIT CELL SETTING
        'a': Decimal('10.0'),
        'b': Decimal('10.0'),
        'c': Decimal('10.0'),
        'al': Decimal('90'),
        'be': Decimal('90'),
        'ga': Decimal('90'),
        # ATOM SETTING
        'x': Decimal('0.5'),
        'y': Decimal('0.5'),
        'z': Decimal('0.5'),
        'U11': Decimal('0.01'),
        'U22': Decimal('0.01'),
        'U33': Decimal('0.01'),
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
        # GRID SETTING
        'grid_radius': Decimal('1.0'),
        'grid_steps': 21,
    }
    XD_TEMPLATE_INP_PATH = CURRENT_DIRECTORY.joinpath('xd_template.inp')
    XD_TEMPLATE_MAS_PATH = CURRENT_DIRECTORY.joinpath('xd_template.mas')

    # TODO recompute Uijk and Uijkm from Cijk and Dijkm

    @property
    def xd_inp_file_contents(self) -> str:
        with open(self.XD_TEMPLATE_INP_PATH, 'r') as file:
            return file.read().format(**self)

    @property
    def xd_mas_file_contents(self) -> str:
        with open(self.XD_TEMPLATE_MAS_PATH, 'r') as file:
            return file.read().format(**self)

    @property
    def olex2_res_file_contents(self) -> str:
        return str()


class SettingList(UserList):
    @classmethod
    def where(cls, **kwargs: Iterable):
        """
        Return an `AtomSettingsList` where every `kwarg`-key assumes values of
        `kwarg`-value. Names of kwargs must match those of  `AtomSettingsCase`.
        For three `kwargs` of length 1, 3, and 5, return an instance of
        `AtomSettingsList` with 1*3*5=15 distinct `AtomSettingsCase`s.
        """
        new = SettingList()
        for value_combination in itertools.product(*kwargs.values()):
            new_kwargs = {k: v for k, v in zip(kwargs.keys(), value_combination)}
            new.append(SettingCase(**new_kwargs))
        return new


class PDFGrid(object):
    def __init__(self):
        values: np.ndarray = np.zeros(1)
        negative_volume: float = 0.0

    @classmethod
    def from_xd_cube(cls, setting: SettingCase):
        with tempfile.TemporaryDirectory() as temp_dir:
            xd_inp_file_path = pathlib.Path(temp_dir).joinpath('xd.inp')
            xd_mas_file_path = pathlib.Path(temp_dir).joinpath('xd.mas')
            xd_out_file_path = pathlib.Path(temp_dir).joinpath('xd_pdf.out')
            xd_grd_file_path = pathlib.Path(temp_dir).joinpath('xd_pdf.grd')
            with open(xd_inp_file_path, 'w') as xd_inp_file:
                xd_inp_file.write(setting.xd_inp_file_contents)
            with open(xd_mas_file_path, 'w') as xd_mas_file:
                xd_mas_file.write(setting.xd_mas_file_contents)
            my_env = os.environ
            my_env['XD_DATADIR'] = '/home/dtchon/XD'
            my_env['PATH'] += os.pathsep + '/home/dtchon/XD/bin'
            process = subprocess.Popen("xdpdf", shell=True, cwd=temp_dir,
                                       env=my_env, stdout=subprocess.DEVNULL)
            process.wait(timeout=5)
            with open(xd_out_file_path, 'r') as xd_out_file:
                xd_out_file_contents = xd_out_file.read()
            with open(xd_grd_file_path, 'r') as xd_grd_file:
                xd_grd_file_contents = xd_grd_file.read()
                print(setting)
                print(xd_out_file_contents)

            # TODO: TEMPORARY PART
            for line in xd_out_file_contents.splitlines():
                if 'PDF values range from' in line:
                    minimum_value = float(line.split()[-4])
                    if minimum_value < 0.0:
                        print('negative minimum found')
                        print(setting)
                        assert False
        return cls()

    @classmethod
    def from_nosphera2_cube(cls, setting: SettingCase):
        return cls()

    @property
    def is_positive_definite(self):
        return bool()


class PositiveInspector(unittest.TestCase):
    def test_grids_have_same_sign(self):
        setting_list: SettingList[SettingCase]
        for setting in setting_list:
            with self.subTest('XD and olex2 report different positivity',
                              setting=setting):
                grid_xd = PDFGrid.from_xd_cube(setting=setting)
                grid_nosphera2 = PDFGrid.from_nosphera2_cube(setting=setting)
                self.assertEqual(grid_xd.is_positive_definite,
                                 grid_nosphera2.is_positive_definite)


if __name__ == '__main__':
    setting_list = SettingList.where(
        C111=[-1, 1],
        C222=[-1, 1],
        C333=[-1, 1],
        C112=[-1, 1],
        C122=[-1, 1],
        C113=[-1, 1],
        C133=[-1, 1],
        C223=[-1, 1],
        C233=[-1, 1],
        C123=[-1, 1],
        a=[1],
        b=[1],
        c=[1],
        # D1111=[-10000, 10000],
        # D2222=[-10000, 10000],
        # D3333=[-10000, 10000],
        # D1112=[-10000, 10000],
        # D1222=[-10000, 10000],
        # D1113=[-10000, 10000],
        # D1333=[-10000, 10000],
        # D2223=[-10000, 10000],
        # D2333=[-10000, 10000],
        # D1122=[-10000, 10000],
        # D1133=[-10000, 10000],
        # D2233=[-10000, 10000],
        # D1123=[-10000, 10000],
        # D1223=[-10000, 10000],
        # D1233=[-10000, 10000],
    )
    for setting_number, setting in enumerate(setting_list):
        print(f'{setting_number} / {len(setting_list)}')
        PDFGrid.from_xd_cube(setting)

