import abc
import enum
import itertools
import os
import pathlib
import re
import subprocess
import tempfile
from collections import UserList, UserDict
from decimal import Decimal
from typing import Iterable, Union
from unittest.mock import Mock
import numpy as np

try:
    import olex
    from olexFunctions import OV
    from cubes_maps import PDF_map
except ImportError:  # Mock modules in development environment if not available
    olex = Mock()
    OV = Mock()
    PDF_map = Mock()


TEMP_DIR = tempfile.TemporaryDirectory()

OLEX2_TEMPLATE_HKL = """
   1   0   0    1.00    1.00
   0   1   0    1.00    1.00
   0   0   1    1.00    1.00
   0   0   0    0.00    0.00
""".strip('\n')

OLEX2_TEMPLATE_INS = """
TITL PI
CELL 0.71073 {a:9.6f} {b:9.6f} {c:9.6f} {al:9.6g} {be:9.6g} {ga:9.6g}
ZERR 4 0.0 0.0 0.0 0.0 0.0 0.0
LATT -1
SFAC Fe
UNIT 1

L.S. 20
PLAN  5
CONF
list 4
MORE -1

WGHT   0.00000
FVAR   0.10000
Fe1  1 {x:9.6f} {y:9.6f} {z:9.6f} 11.0 {U11:9.6f} {U22:9.6f} {U33:9.6f} {U12:9.6f} {U13:9.6f} {U23:9.6f}
HKLF 4
REM <olex2.extras>
REM <anharmonics
REM  <Fe1 Cijk="{C111:6.4e} {C112:6.4e} {C113:6.4e} {C122:6.4e} {C123:6.4e}
REM  {C133:6.4e} {C222:6.4e} {C223:6.4e} {C233:6.4e} {C333:6.4e}" Dijkl="{D1111:6.4e}
REM  {D1112:6.4e} {D1113:6.4e} {D1122:6.4e} {D1123:6.4e} {D1133:6.4e} {D1222:6.4e}
REM  {D1223:6.4e} {D1233:6.4e} {D1333:6.4e} {D2222:6.4e} {D2223:6.4e} {D2233:6.4e}
REM  {D2333:6.4e} {D3333:6.4e}">
REM >
REM </olex2.extras>
""".strip('\n')

XD_TEMPLATE_INP = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! <<< X D PARAMETER FILE >>> $Revision: 2016.01 (Jul 08 2016)$                !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
XDPARFILE VERSION 2
PI   MODEL  -1  4  0  0
LIMITS nat  2000 ntx  31 lmx  4 nzz  30 nto  0 nsc  20 ntb  20 nov   2020
USAGE        1  4  0  1  0  1  0  0  1  0  0  0  0  0
  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000 0.000E+00
  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
Fe(1)     3 2    0   1   0 4  1  1 0  0   0  {x:9.6f}  {y:9.6f}  {z:9.6f} 1.0000
  {U11:9.6f} {U22:9.6f} {U33:9.6f} {U12:9.6f} {U13:9.6f} {U23:9.6f}
  {C111:9.6f} {C222:9.6f} {C333:9.6f} {C112:9.6f} {C122:9.6f}
  {C113:9.6f} {C133:9.6f} {C223:9.6f} {C233:9.6f} {C123:9.6f}
  {D1111:9.6f} {D2222:9.6f} {D3333:9.6f} {D1112:9.6f} {D1222:9.6f}
  {D1113:9.6f} {D1333:9.6f} {D2223:9.6f} {D2333:9.6f} {D1122:9.6f}
  {D1133:9.6f} {D2233:9.6f} {D1123:9.6f} {D1223:9.6f} {D1233:9.6f}
 1  1.000000  1.000000  1.000000  1.000000  1.000000  1.000000
 0.0000E+00 0.0000E+00 0.0000E+00 0.0000E+00 0.0000E+00 0.0000E+00 0.0000E+00
 0.0000E+00
 0.1000E+01
""".strip('\n')

XD_TEMPLATE_MAS = """
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! <<< X D MASTER FILE >>> $Revision: 2016.01 (Jul 08 2016)$                   !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
TITLE PI
CELL     {a:9.6f}  {b:9.6f}  {c:9.6f}  {al:9.6g}  {be:9.6g}  {ga:9.6g}
WAVE     0.71073
CELLSD    0.0000   0.0000   0.0000    0.000    0.000    0.000
LATT   A P
BANK   CR
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
MODULE *XDPDF
SELECT atom Fe(1) scale 1.0 orth angstrom
CUMORD *second {third_star}third {fourth_star}fourth
GRID 3-points *cryst
LIMITS xmin -{grid_radius:8.6f} xmax {grid_radius:8.6f} nx {grid_steps:d}
LIMITS ymin -{grid_radius:8.6f} ymax {grid_radius:8.6f} ny {grid_steps:d}
LIMITS zmin -{grid_radius:8.6f} zmax {grid_radius:8.6f} nz {grid_steps:d}
END XDPDF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
""".strip('\n')


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
    def olex2_hkl_file_contents(self) -> str:
        """string representation of `self`-based "olex2.hkl" file"""
        return OLEX2_TEMPLATE_HKL.format(**self.format_dict)

    @property
    def olex2_ins_file_contents(self) -> str:
        """string representation of `self`-based "olex2.res" file"""
        return OLEX2_TEMPLATE_INS.format(**self.format_dict)

    @property
    def xd_inp_file_contents(self) -> str:
        """string representation of `self`-based "xd.inp" file"""
        return XD_TEMPLATE_INP.format(**self.format_dict)

    @property
    def xd_mas_file_contents(self) -> str:
        """string representation of `self`-based "xd.mas" file"""
        return XD_TEMPLATE_MAS.format(**self.format_dict)


class SettingList(UserList):
    """This class generates and stores individual `SettingCase`s"""
    @classmethod
    def where(cls, **kwargs: Iterable):
        """
        Return a `SettingsList` with all possible `SettingCase`s such that
        `SettingCase[k] == v[i]` for `k` in `kwargs` keys and
        `v[i]` in iterable `v` of respective `kwargs` value.
        `kwargs` keys must exactly match those of `SettingsCase` keys.
        For example `SettingsList(**{'U11': [0,1], 'U22': [0,1,2], 'U33': [1]})`
        will return a `SettingList` with 2 * 3 * 1 = 6 distinct `SettingsCase`s.
        """
        new = SettingList()
        for value_combination in itertools.product(*kwargs.values()):
            new_kwargs = {k: v for k, v in zip(kwargs.keys(), value_combination)}
            new.append(SettingCase(**new_kwargs))
        return new

    @classmethod
    def wheregex(cls, **kwargs: Iterable):
        """
        Return a `SettingsList` with all possible `SettingCase`s such that
        `SettingCase[k] == v[i]` for `k` in `kwargs` keys and
        `v[i]` in iterable `v` of respective `kwargs` value.
        `kwargs` keys must be valid regex expression which match at least one
        (but possibly many) `SettingsCase` keys.
        For example `SettingsList(**{'C..2': [0,1], 'D[12]{4}': [0,1,2]})` will
        return a `SettingList` with 2**3 * 3**5 = 1944 distinct `SettingsCase`s.
        """
        full_kwargs = {}
        for kwarg_key, kwarg_value_iterable in kwargs.items():
            for setting_case_key in SettingCase.DEFAULTS.keys():
                if re.fullmatch(kwarg_key, setting_case_key):
                    full_kwargs.update({setting_case_key: kwarg_value_iterable})
        return cls.where(**full_kwargs)


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
        xd_inp_file_path = pathlib.Path(TEMP_DIR.name).joinpath('xd.inp')
        xd_mas_file_path = pathlib.Path(TEMP_DIR.name).joinpath('xd.mas')
        xd_grd_file_path = pathlib.Path(TEMP_DIR.name).joinpath('xd_pdf.grd')
        with open(xd_inp_file_path, 'w') as xd_inp_file:
            xd_inp_file.write(setting.xd_inp_file_contents)
        with open(xd_mas_file_path, 'w') as xd_mas_file:
            xd_mas_file.write(setting.xd_mas_file_contents)

        my_env = Environ()
        my_env.append(PATH=str(pathlib.Path.home().joinpath('XD', 'bin')))
        my_env.append(XD_DATADIR=str(pathlib.Path.home().joinpath('XD')))
        process = subprocess.Popen("xdpdf", shell=True, cwd=TEMP_DIR.name,
                                   env=my_env, stdout=subprocess.DEVNULL)
        process.wait()
        return cls._read_from_grid_file(xd_grd_file_path)

    @classmethod
    def _generate_from_setting_using_olex2(cls, setting: SettingCase):
        olex2_hkl_file_path = pathlib.Path(TEMP_DIR.name).joinpath('olex2.hkl')
        olex2_ins_file_path = pathlib.Path(TEMP_DIR.name).joinpath('olex2.ins')
        olex2_cube_file_path = pathlib.Path(TEMP_DIR.name).joinpath('PDF.cube')
        with open(olex2_hkl_file_path, 'w') as olex2_hkl_file:
            olex2_hkl_file.write(setting.olex2_hkl_file_contents)
        with open(olex2_ins_file_path, 'w') as olex2_ins_file:
            olex2_ins_file.write(setting.olex2_ins_file_contents)
        OV.Reap(str(olex2_ins_file_path))
        gss = 2 * setting['grid_radius'] / (setting['grid_steps'] - 1)
        PDF_map(gss, setting['grid_radius'], True, True, True, False, True)
        return cls._read_from_cube_file(olex2_cube_file_path)

    @classmethod
    def _read_from_cube_file(cls, path: Union[str, pathlib.Path]):
        with open(path, 'r') as cube_file:
            cube_file_lines = cube_file.readlines()
        atom_count = int(cube_file_lines[2].split()[0])
        _, x_min, y_min, z_min = map(float, cube_file_lines[2].split())
        x_steps = int(cube_file_lines[3].split()[0])
        y_steps = int(cube_file_lines[4].split()[0])
        z_steps = int(cube_file_lines[5].split()[0])
        x_step = float(cube_file_lines[3].split()[1])
        y_step = float(cube_file_lines[4].split()[2])
        z_step = float(cube_file_lines[5].split()[3])
        array = cls._read_array_from_lines(lines=cube_file_lines[6+atom_count:],
                                           shape=(x_steps, y_steps, z_steps),
                                           order='C')
        x_lims = (x_min, x_min + x_steps * x_step)
        y_lims = (y_min, y_min + y_steps * y_step)
        z_lims = (z_min, z_min + z_steps * z_step)
        return cls(array=array, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims)

    @classmethod
    def _read_from_grid_file(cls, path: Union[str, pathlib.Path]):
        with open(path, 'r') as grd_file:
            grd_file_lines = grd_file.readlines()
        grd_non_empty_lines = [line for line in grd_file_lines if line.strip()
                               and not cls.GRD_COMMENT_LINE_REGEX.match(line)]
        x_steps, y_steps, z_steps = map(int, grd_non_empty_lines[2].split())
        x_min, y_min, z_min = map(float, grd_non_empty_lines[3].split())
        x_max, y_max, z_max = map(float, grd_non_empty_lines[4].split())
        array = cls._read_array_from_lines(lines=grd_non_empty_lines[11:],
                                           shape=(x_steps, y_steps, z_steps),
                                           order='F')
        return cls(array=array, x_lims=(x_min, x_max),
                   y_lims=(y_min, y_max), z_lims=(z_min, z_max))

    @staticmethod
    def _read_array_from_lines(lines: Iterable[str],
                               order: str = 'C',
                               shape: Iterable[int] = None) -> np.array:
        entries = ' '.join(lines).split()
        if shape is None:
            shape = [int(round(len(entries) ** (1/3), 0)), ] * 3
        if shape[0] * shape[1] * shape[2] != len(entries):
            raise IndexError(f'Wrong shape {shape} for length {len(entries)}!')
        values = np.array(entries, dtype=np.float64)
        return values.reshape(shape, order=order)

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
    def voxel_volume(self) -> float:
        return self.x_lims.ptp() / (self.array.shape[0] - 1) * \
               self.y_lims.ptp() / (self.array.shape[1] - 1) * \
               self.z_lims.ptp() / (self.array.shape[2] - 1)

    @property
    def integrated_probability(self) -> float:
        return np.sum(self.array) * self.voxel_volume

    @property
    def integrated_positive_probability(self) -> float:
        return np.sum(self.array[self.array > 0]) * self.voxel_volume

    @property
    def integrated_negative_probability(self) -> float:
        return np.sum(self.array[self.array < 0]) * self.voxel_volume

    @property
    def is_positive_definite(self) -> bool:
        return np.all(self.array >= 0)

    @property
    def summary(self) -> str:
        return f'Σp|p>0= {self.integrated_positive_probability: 6.4e} \n'\
               f'Σp|p<0= {self.integrated_negative_probability: 6.4e} \n'\
               f'Σp=     {self.integrated_probability: 6.4e} \n'\
               f'max(p)= {np.max(self.array): 6.4e} \n'\
               f'min(p)= {np.min(self.array): 6.4e} '


def parse_test_args_into_kwargs(args):
    raise NotImplementedError


def test_pdf_map_where(*args):
    raise NotImplementedError


def test_pdf_map_wheregex(*args):
    raise NotImplementedError


def test_pdf_map_third_order():
    test_setting_list = SettingList.wheregex(**{'C[123]{3}': [0.000005, 0.0]})
    results = [None, ] * len(test_setting_list)
    print(f'Testing {len(results)} individual maps against each other')
    for i, s in enumerate(test_setting_list):
        g1 = PDFGrid.generate_from_setting(setting=s, backend='xd')
        g2 = PDFGrid.generate_from_setting(setting=s, backend='olex2')
        results[i] = g1.is_positive_definite is g2.is_positive_definite
        if (i + 1) % 10 == int(len(results) / 10):
            print(f'Checked {i + 1:7d} / {len(results)} map pairs: '
                  f'{len([r for r in results if r is True])} agree, '
                  f'{len([r for r in results if r is False])} disagree.')


def test_pdf_map_fourth_order():
    test_setting_list = SettingList.wheregex(**{'D[123]{4}': [0.000005, 0.0]})
    results = [None, ] * len(test_setting_list)
    print(f'Testing {len(results)} individual maps against each other')
    for i, s in enumerate(test_setting_list):
        g1 = PDFGrid.generate_from_setting(setting=s, backend='xd')
        g2 = PDFGrid.generate_from_setting(setting=s, backend='olex2')
        results[i] = g1.is_positive_definite is g2.is_positive_definite
        if (i + 1) % 10 == int(len(results) / 10):
            print(f'Checked {i + 1:7d} / {len(results)} map pairs: '
                  f'{len([r for r in results if r is True])} agree, '
                  f'{len([r for r in results if r is False])} disagree.')


namespace = 'NoSpherA2'
OV.registerFunction(test_pdf_map_third_order, False, namespace)
OV.registerFunction(test_pdf_map_fourth_order, False, namespace)


if __name__ == '__main__':
    setting_list = SettingList.wheregex(**{'[C][12]+': [0.000005, 0.0]})
    for setting_number, setting in enumerate(setting_list):
        g = PDFGrid.generate_from_setting(setting, backend='xd')
        print(f'{setting_number} / {len(setting_list)}:')
        print(g.summary)
