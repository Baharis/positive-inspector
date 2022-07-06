import abc
import enum
import itertools
import os
import pathlib
import re
import subprocess
import tempfile
import unittest
from collections import UserList, UserDict
from decimal import Decimal
from typing import Iterable, Union
import numpy as np


CURRENT_DIRECTORY = pathlib.Path(__file__).resolve().parent


class MostlyDefaultDict(UserDict, abc.ABC):
    """Dictionary where most values have types and defaults pre-defined"""
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


class Environ(MostlyDefaultDict):
    """Dictionary with current environment variables and helper functions"""
    DEFAULTS = os.environ

    def append(self, **kwargs: str) -> None:
        """Set or append `kwargs.values` to current values of `kwargs.keys`"""
        for k, v in kwargs.items():
            self[k] = self.get(k, '') + os.pathsep + v


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
    OLEX2_TEMPLATE_INS_PATH = CURRENT_DIRECTORY.joinpath('olex2_template.ins')
    OLEX2_TEMPLATE_HKL_PATH = CURRENT_DIRECTORY.joinpath('olex2_template.hkl')

    @property
    def has_third_order_moments(self) -> bool:
        """True if any of the Cijk elements is non-zero, False otherwise"""
        cijk_regex = re.compile(r'^C[1-3]{3}$')
        return any([v != 0 for k, v in self.items() if cijk_regex.match(k)])

    @property
    def has_fourth_order_moments(self) -> bool:
        """True if any of the Dijkl elements is non-zero, False otherwise"""
        dijkl_regex = re.compile(r'^D[1-3]{4}$')
        return any([v != 0 for k, v in self.items() if dijkl_regex.match(k)])

    @property
    def format_dict(self) -> dict:
        """`self` dictionary with additional keywords used in template files"""
        d = dict(self)
        d['third_star'] = '*' if self.has_third_order_moments else ' '
        d['fourth_star'] = '*' if self.has_fourth_order_moments else ' '
        return d

    @property
    def xd_inp_file_contents(self) -> str:
        """string representation of `self`-based "xd.inp" file"""
        with open(self.XD_TEMPLATE_INP_PATH, 'r') as file:
            file_contents = file.read().format(**self.format_dict)
        return file_contents

    @property
    def xd_mas_file_contents(self) -> str:
        """string representation of `self`-based "xd.mas" file"""
        with open(self.XD_TEMPLATE_MAS_PATH, 'r') as file:
            file_contents = file.read().format(**self.format_dict)
        return file_contents

    @property
    def olex2_ins_file_contents(self) -> str:
        """string representation of `self`-based "olex2.res" file"""
        with open(self.OLEX2_TEMPLATE_INS_PATH, 'r') as file:
            file_contents = file.read().format(**self.format_dict)
        return file_contents

    @property
    def olex2_hkl_file_contents(self) -> str:
        """string representation of `self`-based "olex2.hkl" file"""
        with open(self.OLEX2_TEMPLATE_HKL_PATH, 'r') as file:
            file_contents = file.read().format(**self.format_dict)
        return file_contents


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
    """Class handling reading, writing, and analysing grid and cube files"""

    class Backend(enum.Enum):
        XD = 'xd'
        olex2 = 'olex2'

    GRD_COMMENT_LINE_REGEX = re.compile(r'^!.+$', re.MULTILINE)

    @classmethod
    def generate_from_setting(cls, setting: SettingCase, backend: str = 'xd'):
        """Create an instance based `SettingCase` objects using `backend`"""
        if cls.Backend(backend.lower()) is cls.Backend.XD:
            return cls._generate_from_setting_using_xd(setting=setting)
        elif cls.Backend(backend.lower()) is cls.Backend.olex2:
            return cls._generate_from_setting_using_olex2(setting=setting)
        else:
            raise NotImplementedError

    @classmethod
    def _generate_from_setting_using_xd(cls, setting: SettingCase):
        with tempfile.TemporaryDirectory() as temp_dir:
            xd_inp_file_path = pathlib.Path(temp_dir).joinpath('xd.inp')
            xd_mas_file_path = pathlib.Path(temp_dir).joinpath('xd.mas')
            xd_grd_file_path = pathlib.Path(temp_dir).joinpath('xd_pdf.grd')
            with open(xd_inp_file_path, 'w') as xd_inp_file:
                xd_inp_file.write(setting.xd_inp_file_contents)
            with open(xd_mas_file_path, 'w') as xd_mas_file:
                xd_mas_file.write(setting.xd_mas_file_contents)

            my_env = Environ()
            my_env.append(PATH='/home/dtchon/XD/bin')
            my_env.append(XD_DATADIR='/home/dtchon/XD')
            process = subprocess.Popen("xdpdf", shell=True, cwd=temp_dir,
                                       env=my_env, stdout=subprocess.DEVNULL)
            process.wait()
            return cls._read_from_grid_file(xd_grd_file_path)

    @classmethod
    def _generate_from_setting_using_olex2(cls, setting: SettingCase):
        raise NotImplementedError

    @classmethod
    def _read_from_cube_file(cls, path: Union[str, pathlib.Path]):
        raise NotImplementedError

    @classmethod
    def _read_from_grid_file(cls, path: Union[str, pathlib.Path]):
        with open(path, 'r') as grd_file:
            grd_file_lines = grd_file.readlines()
        grd_non_empty_lines = [line for line in grd_file_lines if line.strip()
                               and not cls.GRD_COMMENT_LINE_REGEX.match(line)]
        x_steps, y_steps, z_steps = map(int, grd_non_empty_lines[2].split())
        x_min, y_min, z_min = map(float, grd_non_empty_lines[3].split())
        x_max, y_max, z_max = map(float, grd_non_empty_lines[4].split())
        grd_entries = ' '.join(grd_non_empty_lines[11:]).split()
        grd_values = np.array(grd_entries, dtype=float)
        grd_array = grd_values.reshape((x_steps, y_steps, z_steps), order='F')
        return cls(array=grd_array, x_lims=(x_min, x_max),
                   y_lims=(y_min, y_max), z_lims=(z_min, z_max))

    def __init__(self,
                 array: np.ndarray,
                 x_lims: Iterable = (0., 1.),
                 y_lims: Iterable = (0., 1.),
                 z_lims: Iterable = (0., 1.)):
        self.array = array
        self.x_lims = np.array(x_lims[:2])
        self.y_lims = np.array(y_lims[:2])
        self.z_lims = np.array(z_lims[:2])

    @property
    def voxel_volume(self):
        return self.x_lims.ptp() / (self.array.shape[0] - 1) * \
               self.y_lims.ptp() / (self.array.shape[1] - 1) * \
               self.z_lims.ptp() / (self.array.shape[2] - 1)

    @property
    def integrated_probability(self):
        return np.sum(self.array) * self.voxel_volume

    @property
    def integrated_positive_probability(self):
        return np.sum(self.array[self.array > 0]) * self.voxel_volume

    @property
    def integrated_negative_probability(self):
        return np.sum(self.array[self.array < 0]) * self.voxel_volume

    @property
    def is_positive_definite(self):
        return np.all(self.array >= 0)

    @property
    def summary(self) -> str:
        return f'Σp|p>0= {self.integrated_positive_probability: 6.4e} \n'\
               f'Σp|p<0= {self.integrated_negative_probability: 6.4e} \n'\
               f'Σp=     {self.integrated_probability: 6.4e} \n'\
               f'max(p)= {np.max(self.array): 6.4e} \n'\
               f'min(p)= {np.min(self.array): 6.4e} '


class PositiveInspector(unittest.TestCase):
    """Test suite responsible for finding grids with specific values"""
    def test_grids_have_same_sign(self):
        setting_list: SettingList[SettingCase]
        for setting in setting_list:
            with self.subTest('XD and olex2 report different positivity',
                              setting=setting):
                grid_xd = PDFGrid.generate_from_setting(setting=setting)
                grid_olex2 = PDFGrid.from_cube(setting=setting)
                self.assertEqual(grid_xd.is_positive_definite,
                                 grid_olex2.is_positive_definite)


if __name__ == '__main__':
    setting_list = SettingList.where(
        C111=[0.01],
        C222=[0.01],
        C333=[0.01],
        C112=[0.01],
        C122=[0.01],
        C113=[0.01],
        C133=[0.01],
        C223=[0.01],
        C233=[0.01],
        C123=[0.01],
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
        g = PDFGrid.generate_from_setting(setting, backend='xd')
        print(g.voxel_volume)
        print(f'{setting_number} / {len(setting_list)}:')
        print(g.summary)
