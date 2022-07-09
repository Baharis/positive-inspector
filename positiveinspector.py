import abc
import enum
import itertools
import os
import pathlib
import re
import subprocess
import tempfile
from collections import UserList, UserDict
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
  {U11:6.4e} {U22:6.4e} {U33:6.4e} {U12:6.4e} {U13:6.4e} {U23:6.4e}
  {C111:6.4e} {C222:6.4e} {C333:6.4e} {C112:6.4e} {C122:6.4e}
  {C113:6.4e} {C133:6.4e} {C223:6.4e} {C233:6.4e} {C123:6.4e}
  {D1111:6.4e} {D2222:6.4e} {D3333:6.4e} {D1112:6.4e} {D1222:6.4e}
  {D1113:6.4e} {D1333:6.4e} {D2223:6.4e} {D2333:6.4e} {D1122:6.4e}
  {D1133:6.4e} {D2233:6.4e} {D1123:6.4e} {D1223:6.4e} {D1233:6.4e}
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
SELECT atom Fe(1) scale 1.0 orth *angstrom
CUMORD {star2}second {star3}third {star4}fourth
GRID 3-points *cryst
LIMITS xmin -{grid_radius:8.6f} xmax {grid_radius:8.6f} nx {grid_steps:d}
LIMITS ymin -{grid_radius:8.6f} ymax {grid_radius:8.6f} ny {grid_steps:d}
LIMITS zmin -{grid_radius:8.6f} zmax {grid_radius:8.6f} nz {grid_steps:d}
END XDPDF
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
""".strip('\n')


def a2b(value: Union[int, float, np.ndarray] = 1) -> Union[float, np.ndarray]:
    """Convert `value` in Angstrom to Bohr"""
    return value * 1.8897259886


def b2a(value: Union[int, float, np.ndarray] = 1) -> Union[float, np.ndarray]:
    """Convert `value` in Bohr to Angstrom"""
    return value * 0.529177249


def hstack_strings(*strings: str) -> str:
    """Position two multiline strings next to each other"""
    widths = [max([len(l_) for l_ in string.split('\n')]) for string in strings]
    heights = [len(string.split('\n')) for string in strings]
    format_string = '     '.join(f'{{:{w}}}' for w in widths)
    columns = [s.split('\n') for s in strings]
    zip_ = itertools.zip_longest(*columns, fillvalue='')
    rows = [format_string.format(*cells) for cells in zip_]
    return '\n'.join(rows)


def interpret_string_booleans(s: str) -> Union[str, bool]:
    return True if s.lower() == 'true' else False if s.lower() == 'false' else s


def moment(array: np.ndarray, moment_: int, axis: int = 0) \
        -> Union[float, np.ndarray]:
    """Calculate `moment`-th moment of `array` along `axis`. Source: scipy."""
    if moment_ == 0 or moment_ == 1:
        s = list(array.shape)
        del s[axis]
        d = array.dtype.type if array.dtype.kind in 'fc' else np.float64
        if len(s) == 0:
            return d(1.0 if moment_ == 0 else 0.0)
        else:
            return np.ones(s, dtype=d) if moment_ == 0 else np.zeros(s, dtype=d)
    else:
        n_list = [moment_]
        current_n = moment_
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n - 1) / 2
            else:
                current_n /= 2
            n_list.append(current_n)

        a_zero_mean = array - array.mean(axis, keepdims=True)
        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            s = a_zero_mean**2

        for n in n_list[-2::-1]:
            s = s**2
            if n % 2:
                s *= a_zero_mean
        return np.mean(s, axis)


def kurtosis(array: np.ndarray, axis: int = 0) -> Union[float, np.ndarray]:
    """Calculate kurtosis of `array` along `axis`. Source: scipy."""
    mean = array.mean(axis=axis, keepdims=True)
    m2 = moment(array, 2, axis)
    m4 = moment(array, 4, axis)
    zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(axis)) ** 2)
    values = np.where(zero, 0, m4 / m2 ** 2.0)
    if values.ndim == 0:
        values = values.item()
    return values - 3


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
        'a': 10.0,
        'b': 10.0,
        'c': 10.0,
        'al': 10.0,
        'be': 10.0,
        'ga': 10.0,
        # ATOM SETTING
        'x': 0.5,
        'y': 0.5,
        'z': 0.5,
        'U11': 0.01,
        'U22': 0.01,
        'U33': 0.01,
        'U12': 0.0,
        'U13': 0.0,
        'U23': 0.0,
        'C111': 0.0,
        'C222': 0.0,
        'C333': 0.0,
        'C112': 0.0,
        'C122': 0.0,
        'C113': 0.0,
        'C133': 0.0,
        'C223': 0.0,
        'C233': 0.0,
        'C123': 0.0,
        'D1111': 0.0,
        'D2222': 0.0,
        'D3333': 0.0,
        'D1112': 0.0,
        'D1222': 0.0,
        'D1113': 0.0,
        'D1333': 0.0,
        'D2223': 0.0,
        'D2333': 0.0,
        'D1122': 0.0,
        'D1133': 0.0,
        'D2233': 0.0,
        'D1123': 0.0,
        'D1223': 0.0,
        'D1233': 0.0,
        # GRID SETTING
        'grid_radius': 1.0,
        'grid_steps': 21,
        # QUASI-MOMENTUM ORDER USED
        'use_second': True,
        'use_third': True,
        'use_fourth': True,
    }

    @property
    def has_third_order_moment_parameters(self) -> bool:
        """True if any of the Cijk elements is non-zero, False otherwise"""
        cijk_regex = re.compile(r'^C[1-3]{3}$')
        return any([v != 0 for k, v in self.items() if cijk_regex.match(k)])

    @property
    def has_fourth_order_moment_parameters(self) -> bool:
        """True if any of the Dijkl elements is non-zero, False otherwise"""
        dijkl_regex = re.compile(r'^D[1-3]{4}$')
        return any([v != 0 for k, v in self.items() if dijkl_regex.match(k)])

    @property
    def format_dict(self) -> dict:
        """`self` dictionary with additional keywords used in template files"""
        d = dict(self)
        star2 = self['use_second']
        star3 = self.has_third_order_moment_parameters and self['use_third']
        star4 = self.has_fourth_order_moment_parameters and self['use_fourth']
        d['star2'] = '*' if star2 else ' '
        d['star3'] = '*' if star3 else ' '
        d['star4'] = '*' if star4 else ' '
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
            new_kwargs = {k: v for k, v in
                          zip(kwargs.keys(), value_combination)}
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
        new = cls._read_from_grid_file(xd_grd_file_path)
        origin_shift = np.array([
            float(setting['x'] * setting['a'] - setting['grid_radius']),
            float(setting['y'] * setting['b'] - setting['grid_radius']),
            float(setting['z'] * setting['c'] - setting['grid_radius'])
        ], dtype=np.float64)
        new.origin += origin_shift
        return new

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
        grid_step_size = 2 * setting['grid_radius'] / \
                         (setting['grid_steps'] - 1) + 1e-6
        # this step size makes olex2 create a 100-steps grid when a=b=c=10, but
        # only with PDF gridding "mandatory_factors=[5, 5, 5], max_prime=1000"
        PDF_map(grid_step_size, setting['grid_radius'], setting['use_second'],
                setting['use_third'], setting['use_fourth'], True, True)
        new = cls._read_from_cube_file(olex2_cube_file_path)
        new.array = new.array if setting['use_second'] else new.array / 1000.
        center = np.array([setting['a'] * setting['x'],
                           setting['b'] * setting['y'],
                           setting['c'] * setting['z']], dtype=np.float64)
        new = new.trim_around(center, radius=float(setting['grid_radius']))
        return new

    @classmethod
    def _read_from_cube_file(cls, path: Union[str, pathlib.Path]):
        with open(path, 'r') as cube_file:
            file_lines = cube_file.readlines()
        atom_count = int(file_lines[2].split()[0])
        origin = b2a(np.array(file_lines[2].split()[1:], dtype=np.float64))
        steps = [int(l_.split()[0]) for l_ in file_lines[3:6]]
        x_vector = b2a(np.array(file_lines[3].split()[1:], dtype=np.float64))
        y_vector = b2a(np.array(file_lines[4].split()[1:], dtype=np.float64))
        z_vector = b2a(np.array(file_lines[5].split()[1:], dtype=np.float64))
        array = cls._read_array_from_lines(file_lines[6 + atom_count:], steps, 'C')
        return cls(array, origin, x_vector, y_vector, z_vector)

    @classmethod
    def _read_from_grid_file(cls, path: Union[str, pathlib.Path]):
        with open(path, 'r') as grd_file:
            file_lines = grd_file.readlines()
        non_empty_lines = [line for line in file_lines if line.strip()
                           and not cls.GRD_COMMENT_LINE_REGEX.match(line)]
        steps = np.array(non_empty_lines[2].split(), dtype=int)
        origin = np.array(non_empty_lines[3].split(), dtype=np.float64)
        lengths = np.array(non_empty_lines[4].split(), dtype=np.float64)
        o_vec = np.array(non_empty_lines[6].split()[1:4], dtype=np.float64)
        x_dir = np.array(non_empty_lines[7].split()[1:4], dtype=np.float64)
        y_dir = np.array(non_empty_lines[8].split()[1:4], dtype=np.float64)
        z_dir = np.array(non_empty_lines[9].split()[1:4], dtype=np.float64)
        x_vector = x_dir * lengths[0] / (steps[0] - 1)
        y_vector = y_dir * lengths[1] / (steps[1] - 1)
        z_vector = z_dir * lengths[2] / (steps[2] - 1)
        array = cls._read_array_from_lines(non_empty_lines[11:], steps, 'F')
        return cls(array, origin + o_vec, x_vector, y_vector, z_vector)

    @staticmethod
    def _read_array_from_lines(lines: Iterable[str],
                               shape: Iterable[int] = None,
                               order: str = 'C') -> np.array:
        entries = ' '.join(lines).split()
        if shape is None:
            shape = [int(round(len(entries) ** (1 / 3), 0)), ] * 3
        if shape[0] * shape[1] * shape[2] != len(entries):
            raise IndexError(f'Wrong shape {shape} for length {len(entries)}!')
        values = np.array(entries, dtype=np.float64)
        return values.reshape(shape, order=order)

    def __init__(self,
                 array: np.ndarray,
                 origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
                 x_vector: np.ndarray = np.array([1.0, 0.0, 0.0]),
                 y_vector: np.ndarray = np.array([1.0, 0.0, 0.0]),
                 z_vector: np.ndarray = np.array([1.0, 0.0, 0.0])):
        """
        :param array: 3-dimensional array with PDF values at lattice points
        :param origin: 3-element vector describing position of the 000 point
        :param x_vector: 3-el. vector between subsequent lattice nodes in x dir.
        :param y_vector: 3-el. vector between subsequent lattice nodes in y dir.
        :param z_vector: 3-el. vector between subsequent lattice nodes in z dir.
        """
        self._array = np.zeros(shape=(10, 10, 10))
        self._origin = np.array([0.0, 0.0, 0.0])
        self._basis = np.eye(3)
        self.array = array
        self.origin = origin
        self.basis = np.vstack([x_vector, y_vector, z_vector])

    @property
    def array(self) -> np.ndarray:
        return self._array

    @array.setter
    def array(self, value: np.ndarray):
        self._array = value
        self._update_xyz()

    @property
    def origin(self) -> np.ndarray:
        return self._origin

    @origin.setter
    def origin(self, value: np.ndarray):
        self._origin = value
        self._update_xyz()

    @property
    def basis(self) -> np.ndarray:
        return self._basis

    @basis.setter
    def basis(self, value: np.ndarray):
        self._basis = value
        self._update_xyz()

    @property
    def voxel_volume(self) -> float:
        return np.linalg.det(self.basis)

    def _update_xyz(self):
        indices = np.indices(self._array.shape).reshape(3, -1).T  # [x,y,z] col
        vectors = indices @ self._basis
        self._x = vectors[:, 0].reshape(self._array.shape)
        self._y = vectors[:, 1].reshape(self._array.shape)
        self._z = vectors[:, 2].reshape(self._array.shape)

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def z(self) -> np.ndarray:
        return self._z

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

    def indices2position(self, indices: np.ndarray) -> np.ndarray:
        return indices @ self.basis + self.origin

    def position2indices(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz - self.origin) @ np.linalg.pinv(self.basis)  # inv crashes!

    @property
    def positive_peak_position(self):
        ind = np.unravel_index(np.argmax(self.array), self.array.shape)
        return self.indices2position(np.array(ind))

    @property
    def negative_peak_position(self):
        ind = np.unravel_index(np.argmin(self.array), self.array.shape)
        return self.indices2position(np.array(ind))

    @property
    def absolute_peak_position(self):
        ind = np.unravel_index(np.argmax(np.abs(self.array)), self.array.shape)
        return self.indices2position(np.array(ind))

    def trim_around(self,
             center: np.ndarray,
             radius: float,
             tolerance: float = 1e-4):
        """Trim to `radius` with maximum norm of `self.basis` around `center`"""
        x_0i, y_0i, z_0i = self.position2indices(center)
        x_ri = radius / np.linalg.norm(self.basis[0]) + tolerance
        y_ri = radius / np.linalg.norm(self.basis[1]) + tolerance
        z_ri = radius / np.linalg.norm(self.basis[2]) + tolerance
        x_mini = max(np.ceil(x_0i - x_ri).astype(int), 0)
        x_maxi = np.ceil(x_0i + x_ri).astype(int)
        y_mini = max(np.ceil(y_0i - y_ri).astype(int), 0)
        y_maxi = np.ceil(y_0i + y_ri).astype(int)
        z_mini = max(np.ceil(z_0i - z_ri).astype(int), 0)
        z_maxi = np.ceil(z_0i + z_ri).astype(int)
        new_array = self.array[x_mini:x_maxi, y_mini:y_maxi, z_mini:z_maxi]
        new_origin = self.indices2position(np.array([x_mini, y_mini, z_mini]))
        return self.__class__(new_array, new_origin, *self.basis)

    @property
    def summary(self) -> str:
        t = '               |       for p>0 |       for p<0 |     for all p\n' \
            '  PDF Integral | {intpp:13.5e} | {intpn:13.5e} | {intpa:13.5e}\n' \
            '  PDF peak val | {valpp:13.5e} | {valpn:13.5e} | {valpa:13.5e}\n' \
            ' PDF peak xpos | {posxp:13.5e} | {posxn:13.5e} | {posxa:13.5e}\n' \
            ' PDF peak ypos | {posyp:13.5e} | {posyn:13.5e} | {posya:13.5e}\n' \
            ' PDF peak zpos | {poszp:13.5e} | {poszn:13.5e} | {posza:13.5e}\n' \
            '               |             x |             y |             z\n' \
            '  PDF variance | {varpx:13.5e} | {varpy:13.5e} | {varpz:13.5e}\n' \
            '  PDF kurtosis | {kurpx:13.5e} | {kurpy:13.5e} | {kurpz:13.5e}\n' \
            ' map origin    | {ori_x:13.5e} | {ori_y:13.5e} | {ori_z:13.5e}\n' \
            ' map limit min | {lim0x:13.5e} | {lim0y:13.5e} | {lim0z:13.5e}\n' \
            ' map limit max | {lim1x:13.5e} | {lim1y:13.5e} | {lim1z:13.5e}'
        posp = self.positive_peak_position
        posn = self.negative_peak_position
        posa = self.absolute_peak_position
        return t.format(
            intpp=self.integrated_positive_probability,
            intpn=self.integrated_negative_probability,
            intpa=self.integrated_probability,
            valpp=np.max(self.array),
            valpn=np.max(-self.array),
            valpa=np.max(np.abs(self.array)),
            posxp=posp[0],
            posyp=posp[1],
            poszp=posp[2],
            posxn=posn[0],
            posyn=posn[1],
            poszn=posn[2],
            posxa=posa[0],
            posya=posa[1],
            posza=posa[2],
            varpx=np.var(self.array.mean(axis=(1, 2))),
            varpy=np.var(self.array.mean(axis=(2, 0))),
            varpz=np.var(self.array.mean(axis=(0, 1))),
            kurpx=kurtosis(array=self.array.mean(axis=(1, 2))),
            kurpy=kurtosis(array=self.array.mean(axis=(2, 0))),
            kurpz=kurtosis(array=self.array.mean(axis=(0, 1))),
            ori_x=self.origin[0],
            ori_y=self.origin[1],
            ori_z=self.origin[2],
            lim0x=np.amin(self.x),
            lim1x=np.amax(self.x),
            lim0y=np.amin(self.y),
            lim1y=np.amax(self.y),
            lim0z=np.amin(self.z),
            lim1z=np.amax(self.z),
        )


def _parse_test_pdf_map_args(args) -> dict:
    matching_brackets_regex = re.compile(r"""^\([^)]+\)$|^\[[^]]+]$""")
    quote_key_quote_value_regex = \
        re.compile(r"""^(['"]?)([^"=]*)(\1)=(\([^)]+\)|\[[^]]+]|[^()[\]]+)$""")
    kwargs = {}
    for arg in args:
        match = quote_key_quote_value_regex.fullmatch(arg)
        if not match:
            raise ValueError(f'Cannot interpret argument: {arg}')
        kwargs_key_string, kwargs_value_string = match.group(2, 4)
        if matching_brackets_regex.fullmatch(kwargs_value_string):
            kwargs_value_string = kwargs_value_string[1:-1]
        kwargs_value_list = [interpret_string_booleans(kwarg_value.strip())
                             for kwarg_value in kwargs_value_string.split(',')]
        kwargs[kwargs_key_string] = kwargs_value_list
    return kwargs


def _run_test_pdf_map(setting_list: SettingList) -> None:
    passed = [None, ] * len(setting_list)
    print(f'Testing {len(passed)} individual maps against each other')
    for i, s in enumerate(setting_list):
        g1 = PDFGrid.generate_from_setting(setting=s, backend='xd')
        g2 = PDFGrid.generate_from_setting(setting=s, backend='olex2')
        passed[i] = g1.is_positive_definite is g2.is_positive_definite
        print('XD summary:' + ' ' * 57 + 'olex2 summary:')
        print(hstack_strings(g1.summary, g2.summary))
        print(f'Checked {i + 1:7d} / {len(passed)} map pairs: '
              f'{len([r for r in passed if r is True])} agree, '
              f'{len([r for r in passed if r is False])} disagree.')


def test_pdf_map_where(*args) -> None:
    kwargs = _parse_test_pdf_map_args(args)
    test_setting_list = SettingList.where(**kwargs)
    _run_test_pdf_map(setting_list=test_setting_list)


def test_pdf_map_wheregex(*args) -> None:
    kwargs = _parse_test_pdf_map_args(args)
    test_setting_list = SettingList.wheregex(**kwargs)
    _run_test_pdf_map(setting_list=test_setting_list)


def test_pdf_map_single_case() -> None:
    test_setting_list = SettingList.wheregex(**{'C111': [0.00001]})
    _run_test_pdf_map(setting_list=test_setting_list)


def test_pdf_map_third_order() -> None:
    test_setting_list = SettingList.wheregex(**{'C[123]{3}': [0.000005, 0.0]})
    _run_test_pdf_map(setting_list=test_setting_list)


def test_pdf_map_fourth_order() -> None:
    test_setting_list = SettingList.wheregex(**{'D[123]{4}': [0.000005, 0.0]})
    _run_test_pdf_map(setting_list=test_setting_list)


namespace = 'NoSpherA2'
OV.registerFunction(test_pdf_map_where, False, namespace)
OV.registerFunction(test_pdf_map_wheregex, False, namespace)
OV.registerFunction(test_pdf_map_single_case, False, namespace)
# OV.registerFunction(test_pdf_map_third_order, False, namespace)
# OV.registerFunction(test_pdf_map_fourth_order, False, namespace)
