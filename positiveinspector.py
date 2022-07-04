from typing import List


class AtomSettingsCase(object):
    """
    This class stores individual atoms parameters independent of unit cell,
    including atom type as well as individual xyz, Uij, Cijk, and Dijkl values.
    """
    Z: int
    x: float
    y: float
    z: float
    C111
    C222
    C333
    C112
    C122
    C113
    C133
    C223
    C233
    C123
    D1111
    D2222
    D3333
    D1112
    D1222
    D1113
    D1333
    D2223
    D2333
    D1122
    D1133
    D2233
    D1123
    D1223
    D1233


class CellSettingsCase(object):
    """
    This class stores all parameters concerning unit cell to be tested
    for positiveness, including unit cell parameters and atoms inside it.
    """
    a: float
    b: float
    c: float
    al: float
    be: float
    ga: float
    atoms: List[AtomSettingsCase]


def write_xd_input():
    pass


def write_olex2_input():
    pass

