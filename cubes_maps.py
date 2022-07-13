import os
import olex
import olx
import olex_core
import math
import numpy as np
from scipy import linalg

from olexFunctions import OV
from cctbx_olex_adapter import OlexCctbxAdapter
from smtbx.structure_factors import direct
from cctbx import sgtbx
from cctbx.array_family import flex
from cctbx_olex_adapter import OlexCctbxMasks
import olex_xgrid

import NoSpherA2
import Wfn_Job

types = [ "Q", "H",                                                                                                                                                 "He",
         "Li","Be",                                                                                                                         "B", "C", "N", "O", "F","Ne",
         "Na","Mg",                                                                                                                        "Al","Si", "P", "S","Cl","Ar",
          "K","Ca",                                                                      "Sc","Ti", "V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
         "Rb","Sr",                                                                       "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te", "I","Xe",
         "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta", "W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
         "Fr","Ra","Ac","Th","Pa", "U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr"]

Cs = [[0, 0, 0],
      [0, 0, 1],
      [0, 0, 2],
      [0, 1, 1],
      [0, 1, 2],
      [0, 2, 2],
      [1, 1, 1],
      [1, 1, 2],
      [1, 2, 2],
      [2, 2, 2]]

Ds = [[0, 0, 0, 0],
      [0, 0, 0, 1],
      [0, 0, 0, 2],
      [0, 0, 1, 1],
      [0, 0, 1, 2],
      [0, 0, 2, 2],
      [0, 1, 1, 1],
      [0, 1, 1, 2],
      [0, 1, 2, 2],
      [0, 2, 2, 2],
      [1, 1, 1, 1],
      [1, 1, 1, 2],
      [1, 1, 2, 2],
      [1, 2, 2, 2],
      [2, 2, 2, 2]]

U_map = [[0, 3, 4],
         [3, 1, 5],
         [4, 5, 2]]

a2b = 0.52917749
def unique_permutations(v):
  f = np.math.factorial
  return f(len(v)) / np.prod([f(v.count(i)) for i in range(max(v) + 1)])

def Hjkl(jkl, diff, sigma):
  wj = sum(sigma[U_map[jkl[0]][i]] * diff[i] for i in range(3))
  wk = sum(sigma[U_map[jkl[1]][i]] * diff[i] for i in range(3))
  wl = sum(sigma[U_map[jkl[2]][i]] * diff[i] for i in range(3))
  result = unique_permutations(jkl) * (wj * wl * wk \
         - wj * sigma[U_map[jkl[1]][jkl[2]]] \
         - wk * sigma[U_map[jkl[2]][jkl[0]]] \
         - wl * sigma[U_map[jkl[0]][jkl[1]]])
  return result

def Hjklm(jklm, diff, sigma):
  wj = sum(sigma[U_map[jklm[0]][i]] * diff[i] for i in range(3))
  wk = sum(sigma[U_map[jklm[1]][i]] * diff[i] for i in range(3))
  wl = sum(sigma[U_map[jklm[2]][i]] * diff[i] for i in range(3))
  wm = sum(sigma[U_map[jklm[3]][i]] * diff[i] for i in range(3))
  result = unique_permutations(jklm) * (wj * wk * wl * wm \
         - wj * wk * sigma[U_map[jklm[2]][jklm[3]]] \
         - wj * wl * sigma[U_map[jklm[1]][jklm[3]]] \
         - wj * wm * sigma[U_map[jklm[1]][jklm[2]]] \
         - wk * wl * sigma[U_map[jklm[3]][jklm[0]]] \
         - wk * wm * sigma[U_map[jklm[2]][jklm[0]]] \
         - wl * wm * sigma[U_map[jklm[0]][jklm[1]]] \
         + sigma[U_map[jklm[0]][jklm[1]]] * sigma[U_map[jklm[2]][jklm[3]]] \
         + sigma[U_map[jklm[0]][jklm[2]]] * sigma[U_map[jklm[1]][jklm[3]]] \
         + sigma[U_map[jklm[0]][jklm[3]]] * sigma[U_map[jklm[1]][jklm[2]]])
  return result

def z_slice(z, x, y, vecs, posn, sigmas, pre, n_atoms, anharms, s, t, f, only_anh):
  pos = [(x) * vecs[0][0] + (y) * vecs[0][1] + (z) * vecs[0][2],
         (x) * vecs[1][0] + (y) * vecs[1][1] + (z) * vecs[1][2],
         (x) * vecs[2][0] + (y) * vecs[2][1] + (z) * vecs[2][2]]
  result = 0.0
  for a in range(n_atoms):
    if only_anh == True and anharms[a] == None:
      continue
    if pre[a] < 0:
      continue
    diff = [(pos[0] - posn[a][0]) * a2b, (pos[1] - posn[a][1]) * a2b, (pos[2] - posn[a][2]) * a2b]
    mhalfuTUu = np.fmin(-0.5 * (diff[0] * (diff[0] * sigmas[a][0] \
                                         + diff[1] * sigmas[a][3] \
                                         + diff[2] * sigmas[a][4]) \
                              + diff[1] * (diff[0] * sigmas[a][3] \
                                         + diff[1] * sigmas[a][1] \
                                         + diff[2] * sigmas[a][5]) \
                              + diff[2] * (diff[0] * sigmas[a][4] \
                                         + diff[1] * sigmas[a][5] \
                                         + diff[2] * sigmas[a][2])), np.full((z.size), 0))
    P0 = pre[a] * np.exp(mhalfuTUu)
    P0[abs(P0) < 1E-30] = 0
    fact = float(s)
    if anharms[a] != None:
      if t == True:
        for i in range(10):
          fact += anharms[a][i] * Hjkl(Cs[i], diff, sigmas[a]) / 6
      if f == True:
        for i in range(10, 25):
          fact += anharms[a][i] * Hjklm(Ds[i - 10], diff, sigmas[a]) / 24
    atom_P = P0 * fact
    result += atom_P
  return result

try:
  from_outside = False
  p_path = os.path.dirname(os.path.abspath(__file__))
except:
  from_outside = True
  p_path = os.path.dirname(os.path.abspath("__file__"))

def calculate_cubes():
  if NoSpherA2.is_disordered == True:
    print("Disordered structures not implemented!")
    return

  wfn2fchk = OV.GetVar("Wfn2Fchk")
  args = []

  args.append(wfn2fchk)
  cpus = OV.GetParam('snum.NoSpherA2.ncpus')
  args.append("-cpus")
  args.append(cpus)
  args.append("-wfn")
  if os.path.exists(OV.ModelSrc() + ".wfx"):
    args.append(OV.ModelSrc() + ".wfx")
  else:
    args.append(OV.ModelSrc() + ".wfn")
  Lap = OV.GetParam('snum.NoSpherA2.Property_Lap')
  Eli = OV.GetParam('snum.NoSpherA2.Property_Eli')
  Elf = OV.GetParam('snum.NoSpherA2.Property_Elf')
  RDG = OV.GetParam('snum.NoSpherA2.Property_RDG')
  ESP = OV.GetParam('snum.NoSpherA2.Property_ESP')
  MO  = OV.GetParam('snum.NoSpherA2.Property_MO')
  ATOM = OV.GetParam('snum.NoSpherA2.Property_ATOM')
  DEF = OV.GetParam('snum.NoSpherA2.Property_DEF')
  all_MOs = OV.GetParam('snum.NoSpherA2.Property_all_MOs')
  if Lap == True:
    args.append("-lap")
  if Eli == True:
    args.append("-eli")
  if Elf == True:
    args.append("-elf")
  if RDG == True:
    args.append("-rdg")
  if ESP == True:
    args.append("-esp")
  if ATOM == True:
    args.append("-HDEF")
  if DEF == True:
    args.append("-def")
  if MO == True:
    args.append("-MO")
    if all_MOs == True:
      args.append("all")
    else:
      args.append(str(int(OV.GetParam('snum.NoSpherA2.Property_MO_number'))-1))
  if OV.GetParam('snum.NoSpherA2.wfn2fchk_debug') == True:
    args.append("-v")

  radius = OV.GetParam('snum.NoSpherA2.map_radius')
  res = OV.GetParam('snum.NoSpherA2.map_resolution')
  args.append("-resolution")
  args.append(res)
  args.append("-radius")
  args.append(radius)
  args.append("-cif")
  args.append(OV.ModelSrc() + ".cif")

  os.environ['cube_cmd'] = '+&-'.join(args)
  os.environ['cube_file'] = OV.ModelSrc()
  os.environ['cube_dir'] = OV.FilePath()

  import subprocess
  pyl = OV.getPYLPath()
  if not pyl:
    print("A problem with pyl is encountered, aborting.")
    return
  subprocess.Popen([pyl, os.path.join(p_path, "cube-launch.py")])

OV.registerFunction(calculate_cubes,True,'NoSpherA2')

def get_map_types():
  name = OV.ModelSrc()
  folder = OV.FilePath()
  list = ";Residual<-diff;Deformation<-fcfmc;2Fo-Fc<-tomc;Fobs<-fobs;Fcalc<-fcalc;"
  if os.path.isfile(os.path.join(folder,name+"_eli.cube")):
    list += "ELI-D;"
  if os.path.isfile(os.path.join(folder,name+"_lap.cube")):
    list += "Laplacian;"
  if os.path.isfile(os.path.join(folder,name+"_elf.cube")):
    list += "ELF;"
  if os.path.isfile(os.path.join(folder,name+"_esp.cube")):
    list += "ESP;"
  if os.path.isfile(os.path.join(folder,name+"_rdg.cube")):
    list += "RDG;"
  if os.path.isfile(os.path.join(folder,name+"_def.cube")):
    list += "Stat. Def.;"
  if os.path.isfile(os.path.join(folder,name+"_rdg.cube")) and os.path.isfile(os.path.join(folder,name+"_signed_rho.cube")):
    list += "NCI;"
  if os.path.isfile(os.path.join(folder,name+"_rho.cube")) and os.path.isfile(os.path.join(folder,name+"_esp.cube")):
    list += "Rho + ESP;"
  nmo = Wfn_Job.get_nmo()
  if nmo != -1:
    exists = False
    for i in range(int(nmo)+1):
      if os.path.isfile(os.path.join(folder,name+"_MO_"+str(i)+".cube")):
        exists = True
    if exists == True:
      list += "MO;"
  ncen = Wfn_Job.get_ncen()
  if ncen != -1:
    exists = False
    for i in range(int(ncen)+1):
      if os.path.isfile(os.path.join(folder,name+"_HDEF_"+str(i)+".cube")):
        exists = True
    if exists == True:
      list += "HDEF;"
  if list == "":
    return "None;"
  return list
OV.registerFunction(get_map_types,True,'NoSpherA2')

def change_map():
  Type = OV.GetParam('snum.NoSpherA2.map_type')
  if Type == "None" or Type == "":
    return
  name = OV.ModelSrc()
  if Type == "ELI-D":
    plot_cube(name+"_eli.cube",None)
  elif Type == "Laplacian":
    plot_cube(name+"_lap.cube",None)
  elif Type == "ELF":
    plot_cube(name+"_elf.cube",None)
  elif Type == "ESP":
    plot_cube(name+"_esp.cube",None)
  elif Type == "Stat. Def.":
    plot_cube(name+"_def.cube",None)
  elif Type == "NCI":
    OV.SetParam('snum.NoSpherA2.map_scale_name',"RGB")
    plot_cube(name+"_rdg.cube",name+"_signed_rho.cube")
  elif Type == "RDG":
    plot_cube(name+"_rdg.cube",None)
  elif Type == "Rho + ESP":
    OV.SetParam('snum.NoSpherA2.map_scale_name',"BWR")
    plot_cube(name+"_rho.cube",name+"_esp.cube")
  elif Type == "fcfmc" or Type == "diff" or Type == "tomc" or Type == "fobs" or Type == "fcalc":
    OV.SetVar('map_slider_scale',-50)
    OV.SetParam('snum.map.type',Type)
    show_fft_map(float(OV.GetParam('snum.NoSpherA2.map_resolution')),map_type=Type)
    minimal = float(olx.xgrid.GetMin())
    maximal = float(olx.xgrid.GetMax())
    if -minimal > maximal:
      maximal = -minimal
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',maximal*50)
    olex.m("html.Update()")
  elif Type == "MO":
    number = int(OV.GetParam('snum.NoSpherA2.Property_MO_number')) -1
    plot_cube(name+"_MO_"+str(number)+".cube",None)
  elif Type == "HDEF":
    number = int(OV.GetParam('snum.NoSpherA2.Property_ATOM_number')) -1
    plot_cube(name+"_HDEF_"+str(number)+".cube",None)
  else:
    print("Sorry, no map type available or selected map type not correct!")
    return
OV.registerFunction(change_map,True,'NoSpherA2')

def change_pointsize():
  PS = OV.GetParam('snum.NoSpherA2.gl_pointsize')
  olex.m('gl.PointSize ' + PS)
OV.registerFunction(change_pointsize,True,'NoSpherA2')

def plot_cube(name,color_cube):
  if not os.path.isfile(name):
    print("Cube file does not exist!")
    return
  # olex.m("html.Update()")
  with open(name) as cub:
    cube = cub.readlines()

  run = 0
  na = 0
  x_size = 0
  y_size = 0
  z_size = 0
  x_run = 0
  y_run = 0
  z_run = 0
  data = None

  #min = 100000
  #max = 0

  for line in cube:
    run += 1
    if (run==3):
      values = line.split()
      na = int(values[0])
    if (run==4):
      values = line.split()
      x_size = int(values[0])
    if (run==5):
      values = line.split()
      y_size = int(values[0])
    if (run==6):
      values = line.split()
      z_size = int(values[0])
      data = flex.double(x_size * y_size * z_size)
      data.reshape(flex.grid(x_size, y_size, z_size))
    if (run > na + 6):
      values = line.split()
      for i in range(len(values)):
        v = float(values[i])
        data[(x_run * y_size + y_run) * z_size + z_run] = v
        z_run += 1
        if z_run == z_size:
          y_run += 1
          z_run = 0
          if y_run == y_size:
            x_run += 1
            y_run = 0
        if x_run > x_size:
          print("ERROR! Mismatched indices while reading!")
          return

  cube = None

  make_colorfull = (color_cube != None)
  if make_colorfull == True:
    with open(color_cube) as cub:
      cube2 = cub.readlines()

    run = 0
    #na2 = 0
    x_size2 = 0
    y_size2 = 0
    z_size2 = 0
    x_run = 0
    y_run = 0
    z_run = 0
    data2 = None

    for line in cube2:
      run += 1
      if (run==3):
        values = line.split()
        #na2 = int(values[0])
      if (run==4):
        values = line.split()
        x_size2 = int(values[0])
      if (run==5):
        values = line.split()
        y_size2 = int(values[0])
      if (run==6):
        values = line.split()
        z_size2 = int(values[0])
        data2 = flex.double(x_size2 * y_size2 * z_size2)
        data2.reshape(flex.grid(x_size2, y_size2, z_size2))
      if (run > na + 6):
        values = line.split()
        for i in range(len(values)):
          data2[x_run][y_run][z_run] = float(values[i])
          z_run += 1
          if z_run == z_size2:
            y_run += 1
            z_run = 0
            if y_run == y_size2:
              x_run += 1
              y_run = 0
          if x_run > x_size2:
            print("ERROR! Mismatched indices while reading!")
            return

    cube2 = None
    values = None
    z_run = None
    y_run = None
    x_run = None
    na = None
    #na2 = None
    line = None
    run = None
    olex_xgrid.Init(x_size,y_size,z_size,True)

    def interpolate(x,y,z):
      #trilinear interpolation between the points... sorry for the mess
      x_1 = x/x_size
      y_1 = y/y_size
      z_1 = z/z_size
      x_2 = x_1 * x_size2
      y_2 = y_1 * y_size2
      z_2 = z_1 * z_size2
      ix2 = int(x_2)
      iy2 = int(y_2)
      iz2 = int(z_2)
      ix21 = ix2 + 1
      iy21 = iy2 + 1
      iz21 = iz2 + 1
      a_0 = data2[ix2][iy2][iz2]*ix21*iy21*iz21 - data2[ix2][iy2][iz21]*ix21*iy21*iz2 - data2[ix2][iy21][iz2]*ix21*iy2*iz21 + data2[ix2][iy21][iz21]*ix21*iy2*iz2 - data2[ix21][iy2][iz2]*ix2*iy21*iz21 + data2[ix21][iy2][iz21]*ix2*iy21*iz2 + data2[ix21][iy21][iz2]*ix2*iy2*iz21 - data2[ix21][iy21][iz21]*ix2*iy2*iz2
      a_1 = - data2[ix2][iy2][iz2] * iy21 * iz21 + data2[ix2][iy2][iz21] * iy21 * iz2 + data2[ix2][iy21][iz2] * iy2 * iz21 - data2[ix2][iy21][iz21] * iy2 * iz2 + data2[ix21][iy2][iz2] * iy21 * iz21 - data2[ix21][iy2][iz21] * iy21 * iz2 - data2[ix21][iy21][iz2] * iy2 * iz21 + data2[ix21][iy21][iz21] * iy2 * iz2
      a_2 = - data2[ix2][iy2][iz2] * ix21 * iz21 + data2[ix2][iy2][iz21] * ix21 * iz2 + data2[ix2][iy21][iz2] * ix2 * iz21 - data2[ix2][iy21][iz21] * ix2 * iz2 + data2[ix21][iy2][iz2] * ix21 * iz21 - data2[ix21][iy2][iz21] * ix21 * iz2 - data2[ix21][iy21][iz2] * ix2 * iz21 + data2[ix21][iy21][iz21] * ix2 * iz2
      a_3 = - data2[ix2][iy2][iz2] * ix21 * iy21 + data2[ix2][iy2][iz21] * ix21 * iy2 + data2[ix2][iy21][iz2] * ix2 * iy21 - data2[ix2][iy21][iz21] * ix2 * iy2 + data2[ix21][iy2][iz2] * ix21 * iy21 - data2[ix21][iy2][iz21] * ix21 * iy2 - data2[ix21][iy21][iz2] * ix2 * iy21 + data2[ix21][iy21][iz21] * ix2 * iy2
      a_4 = data2[ix2][iy2][iz2] * iz21 - data2[ix2][iy2][iz21] * iz2 - data2[ix2][iy21][iz2] * iz21 + data2[ix2][iy2][iz21] * iz2 - data2[ix21][iy2][iz2] * iz21 + data2[ix21][iy2][iz21] * iz2 + data2[ix21][iy21][iz2] * iz21 - data2[ix21][iy21][iz21] * iz2
      a_5 = data2[ix2][iy2][iz2] * iy21 - data2[ix2][iy2][iz21] * iy2 - data2[ix2][iy21][iz2] * iy21 + data2[ix2][iy2][iz21] * iy2 - data2[ix21][iy2][iz2] * iy21 + data2[ix21][iy2][iz21] * iy2 + data2[ix21][iy21][iz2] * iy21 - data2[ix21][iy21][iz21] * iy2
      a_6 = data2[ix2][iy2][iz2] * ix21 - data2[ix2][iy2][iz21] * ix2 - data2[ix2][iy21][iz2] * ix21 + data2[ix2][iy2][iz21] * ix2 - data2[ix21][iy2][iz2] * ix21 + data2[ix21][iy2][iz21] * ix2 + data2[ix21][iy21][iz2] * ix21 - data2[ix21][iy21][iz21] * ix2
      a_7 = -(data2[ix2][iy2][iz2] - data2[ix2][iy2][iz21] - data2[ix2][iy21][iz2] + data2[ix2][iy2][iz21] - data2[ix21][iy2][iz2] + data2[ix21][iy2][iz21] + data2[ix21][iy21][iz2] - data2[ix21][iy21][iz21])

      return a_0 + a_1 * ix2 + a_2 * iy2 + a_3 * z_2 + a_4 * x_2 * y_2 + a_5 * x_2 * z_2 + a_6 * y_2 * z_2 + a_7 * x_2 * y_2 * z_2


    value = [[[float(0.0) for k in range(z_size)] for j in range(y_size)] for i in range(x_size)]
    i=None
    if x_size == x_size2 and y_size == y_size2 and z_size == z_size2:
      for x in range(x_size):
        for y in range(y_size):
          for z in range(z_size):
            value[x][y][z] = data2[x][y][z]
    else:
      print("Interpolating...")
      #from dask import delayed
      #from multiprocessing import Pool
      #nproc = int(OV.GetParam("snum.NoSpherA2.ncpus"))
      #pool = Pool(processes=nproc)
      for x in range(x_size):
        for y in range(y_size):
          for z in range(z_size):
            res = interpolate(x,y,z)
            value[x][y][z] = res
    data2 = None
    for x in range(x_size):
      for y in range(y_size):
        for z in range(z_size):
          colour = int(get_color(value[x][y][z]))
          olex_xgrid.SetValue(x,y,z,data[x][y][z],colour)
  else:
    gridding = data.accessor()
    isint = isinstance(data, flex.int)
    a1 = gridding.all()
    a2 = gridding.focus()
    olex_xgrid.Import(a1, a2, data.copy_to_byte_str(), isint)
  Type = OV.GetParam('snum.NoSpherA2.map_type')
  if Type == "Laplacian":
    OV.SetVar('map_min', 0)
    OV.SetVar('map_max', 40)
    OV.SetVar('map_slider_scale',40)
  elif Type == "ELI-D":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',60)
    OV.SetVar('map_slider_scale',20)
  elif Type == "ELF":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',40)
    OV.SetVar('map_slider_scale',40)
  elif Type == "ESP":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',50)
    OV.SetVar('map_slider_scale',50)
  elif Type == "NCI":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',50)
    OV.SetVar('map_slider_scale',100)
  elif Type == "RDG":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',50)
    OV.SetVar('map_slider_scale',100)
  elif Type == "Rho + ESP":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',50)
    OV.SetVar('map_slider_scale',100)
  elif Type == "MO":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',50)
    OV.SetVar('map_slider_scale',100)
  elif Type == "HDEF":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',50)
    OV.SetVar('map_slider_scale',100)
  elif Type == "Stat. Def.":
    OV.SetVar('map_min',0)
    OV.SetVar('map_max',50)
    OV.SetVar('map_slider_scale', 100)
  mmm = data.as_1d().min_max_mean()
  mi = mmm.min
  ma = mmm.max
  olex_xgrid.SetMinMax(mmm.min, mmm.max)
  olex_xgrid.SetVisible(True)
  olex_xgrid.InitSurface(True,1)
  iso = float((abs(mi)+abs(ma))*2/3)
  olex_xgrid.SetSurfaceScale(iso)
  OV.SetParam('snum.xgrid.scale',"{:.3f}".format(iso))

OV.registerFunction(plot_cube,True,'NoSpherA2')

def plot_cube_single(name):
  if not os.path.isfile(name):
    print("Cube file does not exist!")
    return
  olex.m("html.Update()")
  with open(name) as cub:
    cube = cub.readlines()

  run = 0
  na = 0
  x_size = 0
  y_size = 0
  z_size = 0
  x_run = 0
  y_run = 0
  z_run = 0
  data = None

  min = 100000
  max = 0

  for line in cube:
    run += 1
    if (run==3):
      values = line.split()
      na = int(values[0])
    if (run==4):
      values = line.split()
      x_size = int(values[0])
    if (run==5):
      values = line.split()
      y_size = int(values[0])
    if (run==6):
      values = line.split()
      z_size = int(values[0])
      data = [[[float(0.0) for k in range(z_size)] for j in range(y_size)] for i in range(x_size)]
    if (run > na + 6):
      values = line.split()
      for i in range(len(values)):
        data[x_run][y_run][z_run] = float(values[i])
        if data[x_run][y_run][z_run] > max:
          max = data[x_run][y_run][z_run]
        if data[x_run][y_run][z_run] < min:
          min = data[x_run][y_run][z_run]
        z_run += 1
        if z_run == z_size:
          y_run += 1
          z_run = 0
          if y_run == y_size:
            x_run += 1
            y_run = 0
        if x_run > x_size:
          print("ERROR! Mismatched indices while reading!")
          return

  cube = None

  olex_xgrid.Init(x_size,y_size,z_size)
  for x in range(x_size):
    for y in range(y_size):
      for z in range(z_size):
        olex_xgrid.SetValue(x,y,z,data[x][y][z])
  data = None
  OV.SetVar('map_min',0)
  OV.SetVar('map_max',40)
  OV.SetVar('map_slider_scale',100)
  olex_xgrid.SetMinMax(min, max)
  olex_xgrid.SetVisible(True)
  olex_xgrid.InitSurface(True,1)
  iso = float((abs(min)+abs(max))*2/3)
  olex_xgrid.SetSurfaceScale(iso)
  OV.SetParam('snum.xgrid.scale',"{:.3f}".format(iso))

OV.registerFunction(plot_cube_single,True,'NoSpherA2')

def plot_map_cube(map_type,resolution):
  olex.m('CalcFourier -fcf -%s -r=%s'%(map_type,resolution))
  import math
  cctbx_adapter = OlexCctbxAdapter()
  xray_structure = cctbx_adapter.xray_structure()
  uc = xray_structure.unit_cell()
  temp = olex_xgrid.GetSize()
  size = [int(temp[0]),int(temp[1]),int(temp[2])]
  name = OV.ModelSrc()

  n_atoms = int(olx.xf.au.GetAtomCount())
  positions = [[float(0.0) for k in range(3)] for l in range(n_atoms)]
  cm = list(uc.orthogonalization_matrix())
  for i in range(9):
    cm[i] /= a2b
  cm = tuple(cm)  
  for a in range(n_atoms):
    coord = olx.xf.au.GetAtomCrd(a)
    pos = olx.xf.au.Orthogonalise(coord).split(',')
    positions[a] = [float(pos[0]) / a2b, float(pos[1]) / a2b, float(pos[2]) / a2b]

  vecs = [(cm[0] / (size[0] - 1), cm[1] / (size[0] - 1), cm[2] / (size[0] - 1)),
          (cm[3] / (size[1] - 1), cm[4] / (size[1] - 1), cm[5] / (size[1] - 1)),
          (cm[6] / (size[2] - 1), cm[7] / (size[2] - 1), cm[8] / (size[2] - 1))]

  print("start writing a %4d x %4d x %4d cube" % (size[0], size[1], size[2]))

  with open("%s_%s.cube" % (name, map_name), 'w') as cube:
    cube.write("Fourier synthesis map created by Olex2\n")
    cube.write("Model name: %s\n" % name)
    # Origin of cube
    cube.write("%6d %12.8f %12.8f %12.8f\n" % (n_atoms, 0.0, 0.0, 0.0))
    # need to write vectors!
    cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[0], vecs[0][0], vecs[0][1], vecs[0][2]))
    cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[1], vecs[1][0], vecs[1][1], vecs[1][2]))
    cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[2], vecs[2][0], vecs[2][1], vecs[2][2]))
    for i in range(n_atoms):
      atom_type = olx.xf.au.GetAtomType(i)
      charge = 200
      for j in range(104):
        if types[j] == atom_type:
          charge = j
          break
      if charge == 200:
        print("ATOM NOT FOUND!")
      cube.write("%6d %6d.00000 %12.8f %12.8f %12.8f\n" % (charge, charge, positions[i][0], positions[i][1], positions[i][2]))
    for x in range(size[0]):
      for y in range(size[1]):
        string = ""
        for z in range(size[2]):
          value = olex_xgrid.GetValue(x,y,z)
          string += ("%15.7e"%value)
          if (z+1) % 6 == 0 and (z+1) != size[2]:
            string += '\n'
        if (y != (size[1] - 1)):
          string += '\n'
        cube.write(string)
      if(x != (size[0] -1)):
        cube.write('\n')

    cube.close()

  print("Saved Fourier map successfully")

OV.registerFunction(plot_map_cube,True,'NoSpherA2')

def get_color(value):
  a = 127
  b = 0
  g = 0
  r = 0
  scale_min = OV.GetParam('snum.NoSpherA2.map_scale_min')
  scale_max = OV.GetParam('snum.NoSpherA2.map_scale_max')
  scale = OV.GetParam('snum.NoSpherA2.map_scale_name') #BWR = Blue White Red; RGB = Red Green Blue
  x = 0
  if value <= float(scale_min):
    x = 0
  elif value >= float(scale_max):
    x = 1
  else:
    x = (value - float(scale_min)) / (float(scale_max) - float (scale_min))
  if scale == "RWB":
    x = 1 - x
    scale = "BWR"
  if scale == "RGB":
    x = 1 - x
    scale = "BGR"
  if scale == "BWR":
    if x <= 0.5:
      h = 2 * x
      b = 255
      g = int(255 * h)
      r = int(255 * h)
    else:
      h = -2*x + 2
      b = int(255 * h)
      g = int(255 * h)
      r = 255
  elif scale == "BGR":
    if x <= 0.5:
      h = 2*x
      b = int(255*1-h)
      g = int(255* h)
      r = 0
    elif x > 0.5:
      b = 0
      g = int(255 * (-2 * x + 2))
      r = int(255 * ( 2 * x - 1))
  rgba = (a << 24) | (b << 16) | (g << 8) | r
  if value == "0.00101":
    print(rgba)
  return rgba
OV.registerFunction(get_color,True,'NoSpherA2')

def is_colored():
  Type = OV.GetParam('snum.NoSpherA2.map_type')
  if Type == "NCI":
    return True
  elif Type == "Rho + ESP":
    return True
  else:
    return False
OV.registerFunction(is_colored,True,'NoSpherA2')

def plot_fft_map(fft_map):
  data = fft_map.real_map_unpadded()
  gridding = data.accessor()
  from cctbx.array_family import flex
  type = isinstance(data, flex.int)
  olex_xgrid.Import(
    gridding.all(), gridding.focus(), data.copy_to_byte_str(), type)
  statistics = fft_map.statistics()
  min_v = statistics.min()
  max_v = statistics.max()
  sigma = statistics.sigma()
  data = None
  olex_xgrid.SetMinMax(min_v, max_v)
  olex_xgrid.SetVisible(True)
  olex_xgrid.InitSurface(False, 2.0)
  iso = float(-sigma*3.3)
  olex_xgrid.SetSurfaceScale(iso)
  #OV.SetParam('snum.xgrid.scale',"{:.3f}".format(iso))
  print("Map max val %.3f min val %.3f RMS: %.3f"%(max_v,min_v,sigma))
  print("Map size: %d x %d x %d"%(fft_map.n_real()[0],fft_map.n_real()[1],fft_map.n_real()[2]))

OV.registerFunction(plot_fft_map, True, 'NoSpherA2')

def plot_map(data, iso, dist=1.0, min_v=0, max_v=20):
  gridding = data.accessor()
  from cctbx.array_family import flex
  type = isinstance(data, flex.int)
  olex_xgrid.Import(
    gridding.all(), gridding.focus(), data.copy_to_byte_str(), type)
  olex_xgrid.SetMinMax(min_v, max_v)
  olex_xgrid.InitSurface(True, dist)
  olex.m("xgrid.RenderMode(line)")
  olex_xgrid.SetSurfaceScale(iso)
  olex_xgrid.SetVisible(True)

def plot_fft_map_cube(fft_map, map_name):
  cctbx_adapter = OlexCctbxAdapter()
  xray_structure = cctbx_adapter.xray_structure()
  uc = xray_structure.unit_cell()  
  temp = fft_map.n_real()
  size = [int(temp[0]),int(temp[1]),int(temp[2])]
  name = OV.ModelSrc()

  n_atoms = int(olx.xf.au.GetAtomCount())
  positions = [[float(0.0) for k in range(3)] for l in range(n_atoms)]
  cm = list(uc.orthogonalization_matrix())
  for i in range(9):
    cm[i] /= a2b
  cm = tuple(cm)  
  for a in range(n_atoms):
      coord = olx.xf.au.GetAtomCrd(a)
      pos = olx.xf.au.Orthogonalise(coord).split(',')
      positions[a] = [float(pos[0]) / a2b, float(pos[1]) / a2b, float(pos[2]) / a2b]

  vecs = [(cm[0] / (size[0]), cm[1] / (size[0]), cm[2] / (size[0])),
          (cm[3] / (size[1]), cm[4] / (size[1]), cm[5] / (size[1])),
          (cm[6] / (size[2]), cm[7] / (size[2]), cm[8] / (size[2]))]

  print ("start writing a %4d x %4d x %4d cube"%(size[0],size[1],size[2]))
  values = fft_map.real_map_unpadded()

  with open("%s_%s.cube"%(name,map_name),'w') as cube:
    cube.write("Fourier synthesis map created by Olex2\n")
    cube.write("Model name: %s\n"%name)
    # Origin of cube
    cube.write("%6d %12.8f %12.8f %12.8f\n"%(n_atoms,0.0,0.0,0.0))
    # need to write vectors!
    cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[0], vecs[0][0], vecs[0][1], vecs[0][2]))
    cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[1], vecs[1][0], vecs[1][1], vecs[1][2]))
    cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[2], vecs[2][0], vecs[2][1], vecs[2][2]))
    for i in range(n_atoms):
      atom_type = olx.xf.au.GetAtomType(i)
      charge = 200
      for j in range(104):
        if types[j] == atom_type:
          charge = j
          break
      if charge == 200:
        print("ATOM NOT FOUND!")
      cube.write("%6d %6d.00000 %12.8f %12.8f %12.8f\n" % (charge, charge, positions[i][0], positions[i][1], positions[i][2]))
    for x in range(size[0]):
      for y in range(size[1]):
        string = ""
        for z in range(size[2]):
          # value = fft_map.real_map_unpadded()[(x*size[1]+y)*size[2]+z]
          string += ("%15.7e"%values[(x*size[1]+y)*size[2]+z])
          if (z+1) % 6 == 0 and (z+1) != size[2]:
            string += '\n'
        if (y != (size[1] - 1)):
          string += '\n'
        cube.write(string)
      if(x != (size[0] -1)):
        cube.write('\n')

    cube.close()

  print("Saved Fourier map successfully")

def residual_map(resolution=0.1,return_map=False,print_peaks=False):
  cctbx_adapter = OlexCctbxAdapter()
  xray_structure = cctbx_adapter.xray_structure()
  use_tsc = OV.GetParam('snum.NoSpherA2.use_aspherical')
  NoSpherA2_instance = NoSpherA2.get_NoSpherA2_instance()
  if use_tsc == True:
    table_name = str(OV.GetParam("snum.NoSpherA2.file"))
    time = os.path.getmtime(table_name)
    if NoSpherA2_instance.reflection_date is None or time < NoSpherA2_instance.reflection_date:
      print("Calculating Structure Factors from files...")
      one_h = direct.f_calc_modulus_squared(
        xray_structure, table_file_name=table_name)
      f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc(one_h_function=one_h)
      NoSpherA2_instance.set_f_calc_obs_sq_one_h_linearisation(f_calc, f_sq_obs, one_h)
    else:
      print("Calculating Structure Factors from memory...")
      f_sq_obs, f_calc = NoSpherA2_instance.f_obs_sq, NoSpherA2_instance.f_calc
  else:
    f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc()
  if OV.GetParam("snum.refinement.use_solvent_mask"):
    f_mask = cctbx_adapter.load_mask()
    if not f_mask:
      OlexCctbxMasks()
      if olx.current_mask.flood_fill.n_voids() > 0:
        f_mask = olx.current_mask.f_mask()      
    if f_mask:
      if not f_sq_obs.space_group().is_centric() and f_sq_obs.anomalous_flag():
        f_mask = f_mask.generate_bijvoet_mates()
      f_mask = f_mask.common_set(f_sq_obs)
      f_obs = f_sq_obs.f_sq_as_f()
      f_calc = f_calc.array(data=(f_calc.data() + f_mask.data()))
      k = math.sqrt(OV.GetOSF())
      f_diff = f_obs.f_obs_minus_f_calc(1.0/k, f_calc)
  else:
    f_obs = f_sq_obs.f_sq_as_f()
    k = math.sqrt(OV.GetOSF())
    f_diff = f_obs.f_obs_minus_f_calc(1.0/k, f_calc)
  f_diff = f_diff.expand_to_p1()
  print("Using %d reflections for Fourier synthesis"%f_diff.size())
  diff_map = f_diff.fft_map(symmetry_flags=sgtbx.search_symmetry_flags(use_space_group_symmetry=False),
                            resolution_factor=1,grid_step=float(resolution)).apply_volume_scaling()
  if print_peaks == True or print_peaks == "True":
    from cctbx import maptbx
    max_peaks=10
    peaks = diff_map.peak_search(
      parameters=maptbx.peak_search_parameters(
        peak_search_level=2,
        interpolate=False,
        min_distance_sym_equiv=1.0,
        max_clusters=max_peaks+len(xray_structure.scatterers())),
      verify_symmetry=True
      ).all()
    i = 0
    olx.Kill('$Q', au=True) #HP-JUL18 -- Why kill the peaks? -- cause otherwise they accumulate! #HP4/9/18
    for xyz, height in zip(peaks.sites(), peaks.heights()):
      if i < max_peaks:
        a = olx.xf.uc.Closest(*xyz).split(',')
        pi = "Peak %s = (%.3f, %.3f, %.3f), Height = %.3f e/A^3, %.3f A away from %s" %(
            i+1, xyz[0], xyz[1], xyz[2], height, float(a[1]), a[0])
        print(pi)
      id = olx.xf.au.NewAtom("%.2f" %(height), *xyz)
      if id != '-1':
        olx.xf.au.SetAtomU(id, "0.06")
        i = i+1
      if i == 100 or i >= max_peaks:
        break
    if OV.HasGUI():
      basis = olx.gl.Basis()
      frozen = olx.Freeze(True)
    olx.xf.EndUpdate(True) #clear LST
    olx.Compaq(q=True)
    if OV.HasGUI():
      olx.gl.Basis(basis)
      olx.Freeze(frozen)
      OV.Refresh()
  if return_map==True:
    return diff_map
  plot_fft_map_cube(diff_map,"diff")

OV.registerFunction(residual_map, False, "NoSpherA2")


def det(U):
  return U[0] * U[1] * U[2] + U[3] * U[4] * U[5] * 2 - U[1] * U[4] * U[4] - U[3] * U[2] * U[3] - U[5] * U[0] * U[5]
def U_to_sigma(U):
  # wich turns out to be a matrix inversion...
  U_loc = linalg.inv(np.array([U[0],U[3],U[4],U[3],U[1],U[5],U[4],U[5],U[2]]).reshape(3,3))
  return [U_loc[0][0],U_loc[1][1],U_loc[2][2],U_loc[0][1],U_loc[0][2],U_loc[1][2]]

def digest_boolinput(i):
  if i == False or i == "False" or i == "0":
    return False
  else:
    return True

def PDF_map(resolution=0.1, distance=1.0, second=True, third=True, fourth=True, only_anh=True, do_plot=True, save_cube=False):
  second = digest_boolinput(second)
  third = digest_boolinput(third)
  fourth = digest_boolinput(fourth)
  do_plot = digest_boolinput(do_plot)
  save_cube = digest_boolinput(save_cube)
  only_anh = digest_boolinput(only_anh)
  from cctbx import adptbx
  import numpy as np
  from cctbx.array_family import flex
  if second == False and third == False and fourth == False:
    print("Well, what should I print then? Please decide what you want to see!")
    return
  olex.m("kill $Q")
  OV.CreateBitmap("working")
  try:
    distance = float(distance)
    cctbx_adapter = OlexCctbxAdapter()
    uc = cctbx_adapter.xray_structure().unit_cell()
    fixed = math.pow(2 * math.pi, 1.5)
    name = OV.ModelSrc()
    Us = []
    Us_cart = []
    sigmas = []
    pre = []
    posn = []
    anharms = []
    for atom in cctbx_adapter.xray_structure()._scatterers:
      temp = list(atom.site)
      coord = list(uc.orthogonalize(temp))
      coord[0] /= a2b
      coord[1] /= a2b
      coord[2] /= a2b
      posn.append(coord)
      adp = atom.u_star
      if adp != (-1., -1., -1., -1., -1., -1.):
        Us.append(atom.u_star)
        adp_cart = adptbx.u_star_as_u_cart(uc, atom.u_star)
      else:
        Us.append(atom.u_iso)
        adp_cart = adptbx.u_iso_as_u_cart(atom.u_iso)
      if atom.anharmonic_adp == None:
        anharms.append(None)
      else:
        anharms.append(atom.anharmonic_adp.data())
      Us_cart.append(adp_cart)
      sigmas.append(U_to_sigma(adp_cart))
      pre_temp = det(sigmas[-1])
      if pre_temp < 0:
        pre_temp = -math.sqrt(-pre_temp) / fixed
      else:
        pre_temp = math.sqrt(pre_temp) / fixed
      pre.append(pre_temp)

    gridding = cctbx_adapter.xray_structure().gridding(step=float(resolution))
    size = list(gridding.n_real())
    data = flex.double(size[0] * size[1] * size[2])

    n_atoms = len(posn)
    cm = list(uc.orthogonalization_matrix())
    for i in range(9):
      cm[i] /= a2b
    cm = tuple(cm)
    
    fm = list(uc.fractionalization_matrix())
    
    vecs = [(cm[0] / (size[0]), cm[1] / (size[1]), cm[2] / (size[2])),
            (cm[3] / (size[0]), cm[4] / (size[1]), cm[5] / (size[2])),
            (cm[6] / (size[0]), cm[7] / (size[1]), cm[8] / (size[2]))]

    print("Calculating Grid...")
    olx.xf.EndUpdate()
    if OV.HasGUI():
      olx.Refresh()

    limits = [[cm[0], 0],
              [cm[4], 0],
              [cm[8], 0]]

# determine piece of grid that really needs evaluation
    dist_bohr = distance / a2b
    for a in range(n_atoms):
      if second == False or only_anh == True:
        if anharms[a] == None:
          continue
      cart_minmax = [posn[a][0] - dist_bohr,
                posn[a][1] - dist_bohr,
                posn[a][2] - dist_bohr,
                posn[a][0] + dist_bohr,
                posn[a][1] + dist_bohr,
                posn[a][2] + dist_bohr]
      minmax = [0 for x in range(6)]
      for i in range(3):
        for j in range(3):
          for k in range(2):
            minmax[i+k*3] += cart_minmax[j+k*3] * fm[i*3+j] * size[i] * a2b      
      if minmax[0] < limits[0][0]:
        limits[0][0] = minmax[0]
      if minmax[1] < limits[1][0]:
        limits[1][0] = minmax[1]
      if minmax[2] < limits[2][0]:
        limits[2][0] = minmax[2]
      if minmax[3] > limits[0][1]:
        limits[0][1] = minmax[3]
      if minmax[4] > limits[1][1]:
        limits[1][1] = minmax[4]
      if minmax[5] > limits[2][1]:
        limits[2][1] = minmax[5]
    for i in range(3):
      limits[i][0] = math.floor(limits[i][0])
      limits[i][1] = math.ceil(limits[i][1])
    step = 5
    last_percentage = step - 1
    x_size = limits[0][1] - limits[0][0]
    for x in range(limits[0][0], limits[0][1]):
      for y in range(limits[1][0], limits[1][1]):
        x_loc = x
        if x < 0:
          x_loc += size[0]
        y_loc = y
        if y < 0:
          y_loc += size[1]
        start = ((x_loc % size[0]) * size[1] + (y_loc % size[1])) * size[2]
        zs = np.array(range(limits[2][0], limits[2][1]))
        res = z_slice(zs, x, y, vecs, posn, sigmas, pre, n_atoms, anharms, bool(second), bool(third), bool(fourth), only_anh)
        for i, val in enumerate(res):
          z_loc = zs[i]
          if z_loc < 0:
            z_loc += size[2]
          data[start + (z_loc % size[2])] += val
      num = int((x-limits[0][0]) / x_size * 100)
      if num > last_percentage:
        while last_percentage < num:
          last_percentage += step
        print("{:4d}%".format(num))
        olx.xf.EndUpdate()
        if OV.HasGUI():
          olx.Refresh()
    if second == False:
      print("Multiplying grid values with 1000 to get on visible scale")
      data = data * 1000
    stats = data.min_max_mean()
    if stats.min < -0.05:
      index = (data == stats.min).iselection()[0]
      x = math.floor(index / (size[2] * size[1]))
      index -= x * size[2] * size[1]
      y = math.floor(index / size[2])
      z = index % size[2]
      pos = [(x) * vecs[0][0] + (y) * vecs[0][1] + (z) * vecs[0][2],
             (x) * vecs[1][0] + (y) * vecs[1][1] + (z) * vecs[1][2],
             (x) * vecs[2][0] + (y) * vecs[2][1] + (z) * vecs[2][2]]
      min_dist = cm[0] + cm[4] + cm[8]
      atom_nr = 0
      for i in range(n_atoms):
        diff = [(pos[0] - posn[i][0]), (pos[1] - posn[i][1]), (pos[2] - posn[i][2])]
        dist = np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) * a2b
        if dist < min_dist:
          min_dist = dist
          atom_nr = i
      print("WARNING! Significant negative PDF for Atom: " + str(cctbx_adapter.xray_structure()._scatterers[atom_nr].label))
      print("WARNING! At a distance of {:8.3f} Angs".format(min_dist))
    data.reshape(flex.grid(size[0], size[1], size[2]))
    if save_cube:
      print("start writing a %4d x %4d x %4d cube" % (size[0], size[1], size[2]))
      olx.xf.EndUpdate()
      if OV.HasGUI():
        olx.Refresh()
      with open("PDF.cube", 'w') as cube:
        cube.write("PDF map created by Olex2\n")
        cube.write("Model name: %s\n" % name)
        # Origin of cube
        cube.write("%6d %12.8f %12.8f %12.8f\n" % (n_atoms, 0.0, 0.0, 0.0))
        # need to write vectors!
        cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[0], vecs[0][0], vecs[0][1], vecs[0][2]))
        cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[1], vecs[1][0], vecs[1][1], vecs[1][2]))
        cube.write("%6d %12.8f %12.8f %12.8f\n" % (size[2], vecs[2][0], vecs[2][1], vecs[2][2]))
        for i in range(n_atoms):
          atom_type=olx.xf.au.GetAtomType(i)
          charge=200
          for j in range(104):
            if types[j] == atom_type:
              charge=j
              break
          if charge == 200:
            print("ATOM NOT FOUND!")
          cube.write("%6d %6d.00000 %12.8f %12.8f %12.8f\n" % (charge, charge, posn[i][0], posn[i][1], posn[i][2]))
        for x in range(size[0]):
          for y in range(size[1]):
            string = ""
            for z in range(size[2]):
              v = data[(x * size[1] + y) * size[2] + z]
              if abs(v) > 1E-99:
                string += ("%15.5e" % data[(x * size[1] + y) * size[2] + z])
              else:
                string += ("%15.5e" % 0)
              if (z + 1) % 6 == 0 and (z + 1) != size[2]:
                string += '\n'
            if (y != (size[1] - 1)):
              string += '\n'
            cube.write(string)
          if(x != (size[0] - 1)):
            cube.write('\n')
        cube.close()
      print("Saved PDF map successfully")

    if do_plot:
      print("Grid Size: %4d x %4d x %4d" % (size[0], size[1], size[2]))
      iso = -3.1415
      if second == False:
        iso = -0.05
      plot_map(data, iso, distance, min_v=stats.min, max_v=stats.max)
  except Exception as e:
    OV.DeleteBitmap("working")
    raise(e)

  OV.DeleteBitmap("working")
  print("WARNING: PDF Maps are at the moment just to be understood \nindicative and not benchmarked for numerical correctness!")
OV.registerFunction(PDF_map, False, "NoSpherA2")

def tomc_map(resolution=0.1, return_map=False, use_f000=False):
  cctbx_adapter = OlexCctbxAdapter()
  use_tsc = OV.GetParam('snum.NoSpherA2.use_aspherical')
  if use_tsc == True:
    table_name = str(OV.GetParam("snum.NoSpherA2.file"))
    time = os.path.getmtime(table_name)
    NoSpherA2_instance = NoSpherA2.get_NoSpherA2_instance()
    if NoSpherA2_instance.reflection_date is None or time < NoSpherA2_instance.reflection_date:
      xray_structure = cctbx_adapter.xray_structure()
      one_h = direct.f_calc_modulus_squared(
                       xray_structure, table_file_name=table_name)
      f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc(one_h_function=one_h)
      NoSpherA2_instance.set_f_calc_obs_sq_one_h_linearisation(f_calc, f_sq_obs, one_h)
    else:
      f_sq_obs, f_calc = NoSpherA2_instance.f_obs_sq, NoSpherA2_instance.f_calc
  else:
    f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc()
  if OV.GetParam("snum.refinement.use_solvent_mask"):
    f_mask = cctbx_adapter.load_mask()
    if not f_mask:
      OlexCctbxMasks()
      if olx.current_mask.flood_fill.n_voids() > 0:
        f_mask = olx.current_mask.f_mask()      
    if f_mask:
      if not f_sq_obs.space_group().is_centric() and f_sq_obs.anomalous_flag():
        f_mask = f_mask.generate_bijvoet_mates()
      f_mask = f_mask.common_set(f_sq_obs)
      f_obs = f_sq_obs.f_sq_as_f()
      f_calc = f_calc.array(data=(f_calc.data() + f_mask.data()))
      k = math.sqrt(OV.GetOSF())
      f_diff = f_obs.f_obs_minus_f_calc(2.0/k, f_calc)
  else:
    f_obs = f_sq_obs.f_sq_as_f()
    k = math.sqrt(OV.GetOSF())
    f_diff = f_obs.f_obs_minus_f_calc(2.0/k, f_calc)
  
  f_diff = f_diff.expand_to_p1()
  if use_f000 == True or use_f000 == "True":
    f000 = float(olx.xf.GetF000())
    tomc_map = f_diff.fft_map(symmetry_flags=sgtbx.search_symmetry_flags(use_space_group_symmetry=False),
                              resolution_factor=1,grid_step=float(resolution),
                              f_000=f000).apply_volume_scaling()
  else:
    tomc_map = f_diff.fft_map(symmetry_flags=sgtbx.search_symmetry_flags(use_space_group_symmetry=False),
                              resolution_factor=1, grid_step=float(resolution)).apply_volume_scaling()
  if return_map == True:
    return tomc_map
  plot_fft_map_cube(tomc_map,"tomc")

OV.registerFunction(tomc_map, False, "NoSpherA2")

def deformation_map(resolution=0.1, return_map=False):
  use_tsc = OV.GetParam('snum.NoSpherA2.use_aspherical')
  if use_tsc == False:
    print("ERROR! Deformation is only available when using a .tsc file!")
    return
  cctbx_adapter = OlexCctbxAdapter()
  table_name = str(OV.GetParam("snum.NoSpherA2.file"))
  time = os.path.getmtime(table_name)
  NoSpherA2_instance = NoSpherA2.get_NoSpherA2_instance()
  if NoSpherA2_instance.reflection_date is None or time < NoSpherA2_instance.reflection_date:
    xray_structure = cctbx_adapter.xray_structure()
    one_h = direct.f_calc_modulus_squared(
        xray_structure, table_file_name=table_name)
    f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc(one_h_function=one_h)
    NoSpherA2_instance.set_f_calc_obs_sq_one_h_linearisation(f_calc, f_sq_obs, one_h)
  else:
    f_sq_obs, f_calc = NoSpherA2_instance.f_obs_sq, NoSpherA2_instance.f_calc
  f_sq_obs, f_calc_spher = cctbx_adapter.get_fo_sq_fc()
  f_diff = f_calc.f_obs_minus_f_calc(1, f_calc_spher)
  f_diff = f_diff.expand_to_p1()
  def_map = f_diff.fft_map(symmetry_flags=sgtbx.search_symmetry_flags(use_space_group_symmetry=False),
                           resolution_factor=1,grid_step=float(resolution)).apply_volume_scaling()
  if return_map==True:
    return def_map
  plot_fft_map_cube(def_map,"deform")

OV.registerFunction(deformation_map, False, "NoSpherA2")

def obs_map(resolution=0.1, return_map=False, use_f000=False):
  cctbx_adapter = OlexCctbxAdapter()
  use_tsc = OV.GetParam('snum.NoSpherA2.use_aspherical')
  if use_tsc == True:
    table_name = str(OV.GetParam("snum.NoSpherA2.file"))
    time = os.path.getmtime(table_name)
    NoSpherA2_instance = NoSpherA2.get_NoSpherA2_instance()
    if NoSpherA2_instance.reflection_date is None or time < NoSpherA2_instance.reflection_date:
      
      xray_structure = cctbx_adapter.xray_structure()
      one_h = direct.f_calc_modulus_squared(
        xray_structure, table_file_name=table_name)
      f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc(one_h_function=one_h)
      NoSpherA2_instance.set_f_calc_obs_sq_one_h_linearisation(f_calc, f_sq_obs, one_h)
    else:
      f_sq_obs, f_calc = NoSpherA2_instance.f_obs_sq, NoSpherA2_instance.f_calc
  else:
    f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc()
  f_obs = f_sq_obs.f_sq_as_f()
  k = math.sqrt(OV.GetOSF())
  f_obs.apply_scaling(factor=1./k)
  f_obs = f_obs.phase_transfer(f_calc)
  if use_f000 == True or use_f000 == "True":
    f000 = float(olx.xf.GetF000())
    obs_map = f_obs.fft_map(symmetry_flags=sgtbx.search_symmetry_flags(use_space_group_symmetry=False),
                              resolution_factor=1,
                              grid_step=float(resolution),
                              f_000=f000).apply_volume_scaling()
  else:
    obs_map = f_obs.fft_map(symmetry_flags=sgtbx.search_symmetry_flags(use_space_group_symmetry=False),
                              resolution_factor=1,grid_step=float(resolution)).apply_volume_scaling()
  if return_map==True:
    return obs_map
  plot_fft_map_cube(obs_map,"obs")

OV.registerFunction(obs_map, False, "NoSpherA2")

def calc_map(resolution=0.1,return_map=False, use_f000=False):
  cctbx_adapter = OlexCctbxAdapter()
  use_tsc = OV.GetParam('snum.NoSpherA2.use_aspherical')
  if use_tsc == True:
    table_name = str(OV.GetParam("snum.NoSpherA2.file"))
    time = os.path.getmtime(table_name)
    NoSpherA2_instance = NoSpherA2.get_NoSpherA2_instance()
    if NoSpherA2_instance.reflection_date is None or time < NoSpherA2_instance.reflection_date:
      xray_structure = cctbx_adapter.xray_structure()
      one_h = direct.f_calc_modulus_squared(
        xray_structure, table_file_name=table_name)
      f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc(one_h_function=one_h)
      NoSpherA2_instance.set_f_calc_obs_sq_one_h_linearisation(f_calc, f_sq_obs, one_h)
    else:
      f_sq_obs, f_calc = NoSpherA2_instance.f_obs_sq, NoSpherA2_instance.f_calc
  else:
    f_sq_obs, f_calc = cctbx_adapter.get_fo_sq_fc()
  if use_f000 == True or use_f000 == "True":
    f000 = float(olx.xf.GetF000())
    calc_map = f_calc.fft_map(symmetry_flags=sgtbx.search_symmetry_flags(use_space_group_symmetry=False),
                              resolution_factor=1, grid_step=float(resolution),
                              f_000=f000).apply_volume_scaling()
  else:
    calc_map = f_calc.fft_map(symmetry_flags=sgtbx.search_symmetry_flags(use_space_group_symmetry=False),
                              resolution_factor=1, grid_step=float(resolution)).apply_volume_scaling()
  if return_map==True:
    return calc_map
  plot_fft_map_cube(calc_map,"calc")

OV.registerFunction(calc_map, False, "NoSpherA2")

def show_fft_map(resolution=0.1,map_type="diff",use_f000=False,print_peaks=False):
  if map_type == "diff":
    plot_fft_map(residual_map(resolution, return_map=True,print_peaks=print_peaks))
  elif map_type == "fcfmc":
    plot_fft_map(deformation_map(resolution, return_map=True))
  elif map_type == "obs":
    plot_fft_map(obs_map(resolution,return_map=True,use_f000=use_f000))
  elif map_type == "calc":
    plot_fft_map(calc_map(resolution,return_map=True,use_f000=use_f000))
  elif map_type == "tomc":
    plot_fft_map(tomc_map(resolution,return_map=True,use_f000=use_f000))

OV.registerFunction(show_fft_map, False, "NoSpherA2")