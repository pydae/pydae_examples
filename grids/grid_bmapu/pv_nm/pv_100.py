import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support
from io import BytesIO
import pkgutil

dae_file_mode = 'local'

ffi = cffi.FFI()

if dae_file_mode == 'local':
    import pv_100_ini_cffi as jacs_ini
    import pv_100_run_cffi as jacs_run
    import pv_100_trap_cffi as jacs_trap

if dae_file_mode == 'enviroment':
    import envus.no_enviroment.pv_100_cffi as jacs
if dae_file_mode == 'colab':
    import pv_100_cffi as jacs
    
cffi_support.register_module(jacs_ini)
cffi_support.register_module(jacs_run)
cffi_support.register_module(jacs_trap)

f_ini_eval = jacs_ini.lib.f_ini_eval
g_ini_eval = jacs_ini.lib.g_ini_eval
f_run_eval = jacs_run.lib.f_run_eval
g_run_eval = jacs_run.lib.g_run_eval
h_eval  = jacs_ini.lib.h_eval

sparse = False

de_jac_ini_xy_eval = jacs_ini.lib.de_jac_ini_xy_eval
de_jac_ini_up_eval = jacs_ini.lib.de_jac_ini_up_eval
de_jac_ini_num_eval = jacs_ini.lib.de_jac_ini_num_eval

if sparse:
    sp_jac_ini_xy_eval = jacs.lib.sp_jac_ini_xy_eval
    sp_jac_ini_up_eval = jacs.lib.sp_jac_ini_up_eval
    sp_jac_ini_num_eval = jacs.lib.sp_jac_ini_num_eval

de_jac_run_xy_eval = jacs_run.lib.de_jac_run_xy_eval
de_jac_run_up_eval = jacs_run.lib.de_jac_run_up_eval
de_jac_run_num_eval = jacs_run.lib.de_jac_run_num_eval

if sparse:
    sp_jac_run_xy_eval = jacs.lib.sp_jac_run_xy_eval
    sp_jac_run_up_eval = jacs.lib.sp_jac_run_up_eval
    sp_jac_run_num_eval = jacs.lib.sp_jac_run_num_eval

de_jac_trap_xy_eval= jacs_trap.lib.de_jac_trap_xy_eval            
de_jac_trap_up_eval= jacs_trap.lib.de_jac_trap_up_eval        
de_jac_trap_num_eval= jacs_trap.lib.de_jac_trap_num_eval

if sparse:
    sp_jac_trap_xy_eval= jacs.lib.sp_jac_trap_xy_eval            
    sp_jac_trap_up_eval= jacs.lib.sp_jac_trap_up_eval        
    sp_jac_trap_num_eval= jacs.lib.sp_jac_trap_num_eval





import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class model: 

    def __init__(self): 
        
        self.matrices_folder = 'build'
        self.sparse = False
        self.dae_file_mode = 'local'
        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 4
        self.N_y = 913 
        self.N_z = 609 
        self.N_store = 100000 
        self.params_list = ['S_base', 'g_POI_MV_POI', 'b_POI_MV_POI', 'bs_POI_MV_POI', 'g_POI_GRID', 'b_POI_GRID', 'bs_POI_GRID', 'g_LV001_MV001', 'b_LV001_MV001', 'bs_LV001_MV001', 'g_MV001_POI_MV', 'b_MV001_POI_MV', 'bs_MV001_POI_MV', 'g_LV002_MV002', 'b_LV002_MV002', 'bs_LV002_MV002', 'g_MV002_POI_MV', 'b_MV002_POI_MV', 'bs_MV002_POI_MV', 'g_LV003_MV003', 'b_LV003_MV003', 'bs_LV003_MV003', 'g_MV003_POI_MV', 'b_MV003_POI_MV', 'bs_MV003_POI_MV', 'g_LV004_MV004', 'b_LV004_MV004', 'bs_LV004_MV004', 'g_MV004_POI_MV', 'b_MV004_POI_MV', 'bs_MV004_POI_MV', 'g_LV005_MV005', 'b_LV005_MV005', 'bs_LV005_MV005', 'g_MV005_POI_MV', 'b_MV005_POI_MV', 'bs_MV005_POI_MV', 'g_LV006_MV006', 'b_LV006_MV006', 'bs_LV006_MV006', 'g_MV006_POI_MV', 'b_MV006_POI_MV', 'bs_MV006_POI_MV', 'g_LV007_MV007', 'b_LV007_MV007', 'bs_LV007_MV007', 'g_MV007_POI_MV', 'b_MV007_POI_MV', 'bs_MV007_POI_MV', 'g_LV008_MV008', 'b_LV008_MV008', 'bs_LV008_MV008', 'g_MV008_POI_MV', 'b_MV008_POI_MV', 'bs_MV008_POI_MV', 'g_LV009_MV009', 'b_LV009_MV009', 'bs_LV009_MV009', 'g_MV009_POI_MV', 'b_MV009_POI_MV', 'bs_MV009_POI_MV', 'g_LV010_MV010', 'b_LV010_MV010', 'bs_LV010_MV010', 'g_MV010_POI_MV', 'b_MV010_POI_MV', 'bs_MV010_POI_MV', 'g_LV011_MV011', 'b_LV011_MV011', 'bs_LV011_MV011', 'g_MV011_POI_MV', 'b_MV011_POI_MV', 'bs_MV011_POI_MV', 'g_LV012_MV012', 'b_LV012_MV012', 'bs_LV012_MV012', 'g_MV012_POI_MV', 'b_MV012_POI_MV', 'bs_MV012_POI_MV', 'g_LV013_MV013', 'b_LV013_MV013', 'bs_LV013_MV013', 'g_MV013_POI_MV', 'b_MV013_POI_MV', 'bs_MV013_POI_MV', 'g_LV014_MV014', 'b_LV014_MV014', 'bs_LV014_MV014', 'g_MV014_POI_MV', 'b_MV014_POI_MV', 'bs_MV014_POI_MV', 'g_LV015_MV015', 'b_LV015_MV015', 'bs_LV015_MV015', 'g_MV015_POI_MV', 'b_MV015_POI_MV', 'bs_MV015_POI_MV', 'g_LV016_MV016', 'b_LV016_MV016', 'bs_LV016_MV016', 'g_MV016_POI_MV', 'b_MV016_POI_MV', 'bs_MV016_POI_MV', 'g_LV017_MV017', 'b_LV017_MV017', 'bs_LV017_MV017', 'g_MV017_POI_MV', 'b_MV017_POI_MV', 'bs_MV017_POI_MV', 'g_LV018_MV018', 'b_LV018_MV018', 'bs_LV018_MV018', 'g_MV018_POI_MV', 'b_MV018_POI_MV', 'bs_MV018_POI_MV', 'g_LV019_MV019', 'b_LV019_MV019', 'bs_LV019_MV019', 'g_MV019_POI_MV', 'b_MV019_POI_MV', 'bs_MV019_POI_MV', 'g_LV020_MV020', 'b_LV020_MV020', 'bs_LV020_MV020', 'g_MV020_POI_MV', 'b_MV020_POI_MV', 'bs_MV020_POI_MV', 'g_LV021_MV021', 'b_LV021_MV021', 'bs_LV021_MV021', 'g_MV021_POI_MV', 'b_MV021_POI_MV', 'bs_MV021_POI_MV', 'g_LV022_MV022', 'b_LV022_MV022', 'bs_LV022_MV022', 'g_MV022_POI_MV', 'b_MV022_POI_MV', 'bs_MV022_POI_MV', 'g_LV023_MV023', 'b_LV023_MV023', 'bs_LV023_MV023', 'g_MV023_POI_MV', 'b_MV023_POI_MV', 'bs_MV023_POI_MV', 'g_LV024_MV024', 'b_LV024_MV024', 'bs_LV024_MV024', 'g_MV024_POI_MV', 'b_MV024_POI_MV', 'bs_MV024_POI_MV', 'g_LV025_MV025', 'b_LV025_MV025', 'bs_LV025_MV025', 'g_MV025_POI_MV', 'b_MV025_POI_MV', 'bs_MV025_POI_MV', 'g_LV026_MV026', 'b_LV026_MV026', 'bs_LV026_MV026', 'g_MV026_POI_MV', 'b_MV026_POI_MV', 'bs_MV026_POI_MV', 'g_LV027_MV027', 'b_LV027_MV027', 'bs_LV027_MV027', 'g_MV027_POI_MV', 'b_MV027_POI_MV', 'bs_MV027_POI_MV', 'g_LV028_MV028', 'b_LV028_MV028', 'bs_LV028_MV028', 'g_MV028_POI_MV', 'b_MV028_POI_MV', 'bs_MV028_POI_MV', 'g_LV029_MV029', 'b_LV029_MV029', 'bs_LV029_MV029', 'g_MV029_POI_MV', 'b_MV029_POI_MV', 'bs_MV029_POI_MV', 'g_LV030_MV030', 'b_LV030_MV030', 'bs_LV030_MV030', 'g_MV030_POI_MV', 'b_MV030_POI_MV', 'bs_MV030_POI_MV', 'g_LV031_MV031', 'b_LV031_MV031', 'bs_LV031_MV031', 'g_MV031_POI_MV', 'b_MV031_POI_MV', 'bs_MV031_POI_MV', 'g_LV032_MV032', 'b_LV032_MV032', 'bs_LV032_MV032', 'g_MV032_POI_MV', 'b_MV032_POI_MV', 'bs_MV032_POI_MV', 'g_LV033_MV033', 'b_LV033_MV033', 'bs_LV033_MV033', 'g_MV033_POI_MV', 'b_MV033_POI_MV', 'bs_MV033_POI_MV', 'g_LV034_MV034', 'b_LV034_MV034', 'bs_LV034_MV034', 'g_MV034_POI_MV', 'b_MV034_POI_MV', 'bs_MV034_POI_MV', 'g_LV035_MV035', 'b_LV035_MV035', 'bs_LV035_MV035', 'g_MV035_POI_MV', 'b_MV035_POI_MV', 'bs_MV035_POI_MV', 'g_LV036_MV036', 'b_LV036_MV036', 'bs_LV036_MV036', 'g_MV036_POI_MV', 'b_MV036_POI_MV', 'bs_MV036_POI_MV', 'g_LV037_MV037', 'b_LV037_MV037', 'bs_LV037_MV037', 'g_MV037_POI_MV', 'b_MV037_POI_MV', 'bs_MV037_POI_MV', 'g_LV038_MV038', 'b_LV038_MV038', 'bs_LV038_MV038', 'g_MV038_POI_MV', 'b_MV038_POI_MV', 'bs_MV038_POI_MV', 'g_LV039_MV039', 'b_LV039_MV039', 'bs_LV039_MV039', 'g_MV039_POI_MV', 'b_MV039_POI_MV', 'bs_MV039_POI_MV', 'g_LV040_MV040', 'b_LV040_MV040', 'bs_LV040_MV040', 'g_MV040_POI_MV', 'b_MV040_POI_MV', 'bs_MV040_POI_MV', 'g_LV041_MV041', 'b_LV041_MV041', 'bs_LV041_MV041', 'g_MV041_POI_MV', 'b_MV041_POI_MV', 'bs_MV041_POI_MV', 'g_LV042_MV042', 'b_LV042_MV042', 'bs_LV042_MV042', 'g_MV042_POI_MV', 'b_MV042_POI_MV', 'bs_MV042_POI_MV', 'g_LV043_MV043', 'b_LV043_MV043', 'bs_LV043_MV043', 'g_MV043_POI_MV', 'b_MV043_POI_MV', 'bs_MV043_POI_MV', 'g_LV044_MV044', 'b_LV044_MV044', 'bs_LV044_MV044', 'g_MV044_POI_MV', 'b_MV044_POI_MV', 'bs_MV044_POI_MV', 'g_LV045_MV045', 'b_LV045_MV045', 'bs_LV045_MV045', 'g_MV045_POI_MV', 'b_MV045_POI_MV', 'bs_MV045_POI_MV', 'g_LV046_MV046', 'b_LV046_MV046', 'bs_LV046_MV046', 'g_MV046_POI_MV', 'b_MV046_POI_MV', 'bs_MV046_POI_MV', 'g_LV047_MV047', 'b_LV047_MV047', 'bs_LV047_MV047', 'g_MV047_POI_MV', 'b_MV047_POI_MV', 'bs_MV047_POI_MV', 'g_LV048_MV048', 'b_LV048_MV048', 'bs_LV048_MV048', 'g_MV048_POI_MV', 'b_MV048_POI_MV', 'bs_MV048_POI_MV', 'g_LV049_MV049', 'b_LV049_MV049', 'bs_LV049_MV049', 'g_MV049_POI_MV', 'b_MV049_POI_MV', 'bs_MV049_POI_MV', 'g_LV050_MV050', 'b_LV050_MV050', 'bs_LV050_MV050', 'g_MV050_POI_MV', 'b_MV050_POI_MV', 'bs_MV050_POI_MV', 'g_LV051_MV051', 'b_LV051_MV051', 'bs_LV051_MV051', 'g_MV051_POI_MV', 'b_MV051_POI_MV', 'bs_MV051_POI_MV', 'g_LV052_MV052', 'b_LV052_MV052', 'bs_LV052_MV052', 'g_MV052_POI_MV', 'b_MV052_POI_MV', 'bs_MV052_POI_MV', 'g_LV053_MV053', 'b_LV053_MV053', 'bs_LV053_MV053', 'g_MV053_POI_MV', 'b_MV053_POI_MV', 'bs_MV053_POI_MV', 'g_LV054_MV054', 'b_LV054_MV054', 'bs_LV054_MV054', 'g_MV054_POI_MV', 'b_MV054_POI_MV', 'bs_MV054_POI_MV', 'g_LV055_MV055', 'b_LV055_MV055', 'bs_LV055_MV055', 'g_MV055_POI_MV', 'b_MV055_POI_MV', 'bs_MV055_POI_MV', 'g_LV056_MV056', 'b_LV056_MV056', 'bs_LV056_MV056', 'g_MV056_POI_MV', 'b_MV056_POI_MV', 'bs_MV056_POI_MV', 'g_LV057_MV057', 'b_LV057_MV057', 'bs_LV057_MV057', 'g_MV057_POI_MV', 'b_MV057_POI_MV', 'bs_MV057_POI_MV', 'g_LV058_MV058', 'b_LV058_MV058', 'bs_LV058_MV058', 'g_MV058_POI_MV', 'b_MV058_POI_MV', 'bs_MV058_POI_MV', 'g_LV059_MV059', 'b_LV059_MV059', 'bs_LV059_MV059', 'g_MV059_POI_MV', 'b_MV059_POI_MV', 'bs_MV059_POI_MV', 'g_LV060_MV060', 'b_LV060_MV060', 'bs_LV060_MV060', 'g_MV060_POI_MV', 'b_MV060_POI_MV', 'bs_MV060_POI_MV', 'g_LV061_MV061', 'b_LV061_MV061', 'bs_LV061_MV061', 'g_MV061_POI_MV', 'b_MV061_POI_MV', 'bs_MV061_POI_MV', 'g_LV062_MV062', 'b_LV062_MV062', 'bs_LV062_MV062', 'g_MV062_POI_MV', 'b_MV062_POI_MV', 'bs_MV062_POI_MV', 'g_LV063_MV063', 'b_LV063_MV063', 'bs_LV063_MV063', 'g_MV063_POI_MV', 'b_MV063_POI_MV', 'bs_MV063_POI_MV', 'g_LV064_MV064', 'b_LV064_MV064', 'bs_LV064_MV064', 'g_MV064_POI_MV', 'b_MV064_POI_MV', 'bs_MV064_POI_MV', 'g_LV065_MV065', 'b_LV065_MV065', 'bs_LV065_MV065', 'g_MV065_POI_MV', 'b_MV065_POI_MV', 'bs_MV065_POI_MV', 'g_LV066_MV066', 'b_LV066_MV066', 'bs_LV066_MV066', 'g_MV066_POI_MV', 'b_MV066_POI_MV', 'bs_MV066_POI_MV', 'g_LV067_MV067', 'b_LV067_MV067', 'bs_LV067_MV067', 'g_MV067_POI_MV', 'b_MV067_POI_MV', 'bs_MV067_POI_MV', 'g_LV068_MV068', 'b_LV068_MV068', 'bs_LV068_MV068', 'g_MV068_POI_MV', 'b_MV068_POI_MV', 'bs_MV068_POI_MV', 'g_LV069_MV069', 'b_LV069_MV069', 'bs_LV069_MV069', 'g_MV069_POI_MV', 'b_MV069_POI_MV', 'bs_MV069_POI_MV', 'g_LV070_MV070', 'b_LV070_MV070', 'bs_LV070_MV070', 'g_MV070_POI_MV', 'b_MV070_POI_MV', 'bs_MV070_POI_MV', 'g_LV071_MV071', 'b_LV071_MV071', 'bs_LV071_MV071', 'g_MV071_POI_MV', 'b_MV071_POI_MV', 'bs_MV071_POI_MV', 'g_LV072_MV072', 'b_LV072_MV072', 'bs_LV072_MV072', 'g_MV072_POI_MV', 'b_MV072_POI_MV', 'bs_MV072_POI_MV', 'g_LV073_MV073', 'b_LV073_MV073', 'bs_LV073_MV073', 'g_MV073_POI_MV', 'b_MV073_POI_MV', 'bs_MV073_POI_MV', 'g_LV074_MV074', 'b_LV074_MV074', 'bs_LV074_MV074', 'g_MV074_POI_MV', 'b_MV074_POI_MV', 'bs_MV074_POI_MV', 'g_LV075_MV075', 'b_LV075_MV075', 'bs_LV075_MV075', 'g_MV075_POI_MV', 'b_MV075_POI_MV', 'bs_MV075_POI_MV', 'g_LV076_MV076', 'b_LV076_MV076', 'bs_LV076_MV076', 'g_MV076_POI_MV', 'b_MV076_POI_MV', 'bs_MV076_POI_MV', 'g_LV077_MV077', 'b_LV077_MV077', 'bs_LV077_MV077', 'g_MV077_POI_MV', 'b_MV077_POI_MV', 'bs_MV077_POI_MV', 'g_LV078_MV078', 'b_LV078_MV078', 'bs_LV078_MV078', 'g_MV078_POI_MV', 'b_MV078_POI_MV', 'bs_MV078_POI_MV', 'g_LV079_MV079', 'b_LV079_MV079', 'bs_LV079_MV079', 'g_MV079_POI_MV', 'b_MV079_POI_MV', 'bs_MV079_POI_MV', 'g_LV080_MV080', 'b_LV080_MV080', 'bs_LV080_MV080', 'g_MV080_POI_MV', 'b_MV080_POI_MV', 'bs_MV080_POI_MV', 'g_LV081_MV081', 'b_LV081_MV081', 'bs_LV081_MV081', 'g_MV081_POI_MV', 'b_MV081_POI_MV', 'bs_MV081_POI_MV', 'g_LV082_MV082', 'b_LV082_MV082', 'bs_LV082_MV082', 'g_MV082_POI_MV', 'b_MV082_POI_MV', 'bs_MV082_POI_MV', 'g_LV083_MV083', 'b_LV083_MV083', 'bs_LV083_MV083', 'g_MV083_POI_MV', 'b_MV083_POI_MV', 'bs_MV083_POI_MV', 'g_LV084_MV084', 'b_LV084_MV084', 'bs_LV084_MV084', 'g_MV084_POI_MV', 'b_MV084_POI_MV', 'bs_MV084_POI_MV', 'g_LV085_MV085', 'b_LV085_MV085', 'bs_LV085_MV085', 'g_MV085_POI_MV', 'b_MV085_POI_MV', 'bs_MV085_POI_MV', 'g_LV086_MV086', 'b_LV086_MV086', 'bs_LV086_MV086', 'g_MV086_POI_MV', 'b_MV086_POI_MV', 'bs_MV086_POI_MV', 'g_LV087_MV087', 'b_LV087_MV087', 'bs_LV087_MV087', 'g_MV087_POI_MV', 'b_MV087_POI_MV', 'bs_MV087_POI_MV', 'g_LV088_MV088', 'b_LV088_MV088', 'bs_LV088_MV088', 'g_MV088_POI_MV', 'b_MV088_POI_MV', 'bs_MV088_POI_MV', 'g_LV089_MV089', 'b_LV089_MV089', 'bs_LV089_MV089', 'g_MV089_POI_MV', 'b_MV089_POI_MV', 'bs_MV089_POI_MV', 'g_LV090_MV090', 'b_LV090_MV090', 'bs_LV090_MV090', 'g_MV090_POI_MV', 'b_MV090_POI_MV', 'bs_MV090_POI_MV', 'g_LV091_MV091', 'b_LV091_MV091', 'bs_LV091_MV091', 'g_MV091_POI_MV', 'b_MV091_POI_MV', 'bs_MV091_POI_MV', 'g_LV092_MV092', 'b_LV092_MV092', 'bs_LV092_MV092', 'g_MV092_POI_MV', 'b_MV092_POI_MV', 'bs_MV092_POI_MV', 'g_LV093_MV093', 'b_LV093_MV093', 'bs_LV093_MV093', 'g_MV093_POI_MV', 'b_MV093_POI_MV', 'bs_MV093_POI_MV', 'g_LV094_MV094', 'b_LV094_MV094', 'bs_LV094_MV094', 'g_MV094_POI_MV', 'b_MV094_POI_MV', 'bs_MV094_POI_MV', 'g_LV095_MV095', 'b_LV095_MV095', 'bs_LV095_MV095', 'g_MV095_POI_MV', 'b_MV095_POI_MV', 'bs_MV095_POI_MV', 'g_LV096_MV096', 'b_LV096_MV096', 'bs_LV096_MV096', 'g_MV096_POI_MV', 'b_MV096_POI_MV', 'bs_MV096_POI_MV', 'g_LV097_MV097', 'b_LV097_MV097', 'bs_LV097_MV097', 'g_MV097_POI_MV', 'b_MV097_POI_MV', 'bs_MV097_POI_MV', 'g_LV098_MV098', 'b_LV098_MV098', 'bs_LV098_MV098', 'g_MV098_POI_MV', 'b_MV098_POI_MV', 'bs_MV098_POI_MV', 'g_LV099_MV099', 'b_LV099_MV099', 'bs_LV099_MV099', 'g_MV099_POI_MV', 'b_MV099_POI_MV', 'bs_MV099_POI_MV', 'g_LV100_MV100', 'b_LV100_MV100', 'bs_LV100_MV100', 'g_MV100_POI_MV', 'b_MV100_POI_MV', 'bs_MV100_POI_MV', 'U_POI_MV_n', 'U_POI_n', 'U_GRID_n', 'U_LV001_n', 'U_MV001_n', 'U_LV002_n', 'U_MV002_n', 'U_LV003_n', 'U_MV003_n', 'U_LV004_n', 'U_MV004_n', 'U_LV005_n', 'U_MV005_n', 'U_LV006_n', 'U_MV006_n', 'U_LV007_n', 'U_MV007_n', 'U_LV008_n', 'U_MV008_n', 'U_LV009_n', 'U_MV009_n', 'U_LV010_n', 'U_MV010_n', 'U_LV011_n', 'U_MV011_n', 'U_LV012_n', 'U_MV012_n', 'U_LV013_n', 'U_MV013_n', 'U_LV014_n', 'U_MV014_n', 'U_LV015_n', 'U_MV015_n', 'U_LV016_n', 'U_MV016_n', 'U_LV017_n', 'U_MV017_n', 'U_LV018_n', 'U_MV018_n', 'U_LV019_n', 'U_MV019_n', 'U_LV020_n', 'U_MV020_n', 'U_LV021_n', 'U_MV021_n', 'U_LV022_n', 'U_MV022_n', 'U_LV023_n', 'U_MV023_n', 'U_LV024_n', 'U_MV024_n', 'U_LV025_n', 'U_MV025_n', 'U_LV026_n', 'U_MV026_n', 'U_LV027_n', 'U_MV027_n', 'U_LV028_n', 'U_MV028_n', 'U_LV029_n', 'U_MV029_n', 'U_LV030_n', 'U_MV030_n', 'U_LV031_n', 'U_MV031_n', 'U_LV032_n', 'U_MV032_n', 'U_LV033_n', 'U_MV033_n', 'U_LV034_n', 'U_MV034_n', 'U_LV035_n', 'U_MV035_n', 'U_LV036_n', 'U_MV036_n', 'U_LV037_n', 'U_MV037_n', 'U_LV038_n', 'U_MV038_n', 'U_LV039_n', 'U_MV039_n', 'U_LV040_n', 'U_MV040_n', 'U_LV041_n', 'U_MV041_n', 'U_LV042_n', 'U_MV042_n', 'U_LV043_n', 'U_MV043_n', 'U_LV044_n', 'U_MV044_n', 'U_LV045_n', 'U_MV045_n', 'U_LV046_n', 'U_MV046_n', 'U_LV047_n', 'U_MV047_n', 'U_LV048_n', 'U_MV048_n', 'U_LV049_n', 'U_MV049_n', 'U_LV050_n', 'U_MV050_n', 'U_LV051_n', 'U_MV051_n', 'U_LV052_n', 'U_MV052_n', 'U_LV053_n', 'U_MV053_n', 'U_LV054_n', 'U_MV054_n', 'U_LV055_n', 'U_MV055_n', 'U_LV056_n', 'U_MV056_n', 'U_LV057_n', 'U_MV057_n', 'U_LV058_n', 'U_MV058_n', 'U_LV059_n', 'U_MV059_n', 'U_LV060_n', 'U_MV060_n', 'U_LV061_n', 'U_MV061_n', 'U_LV062_n', 'U_MV062_n', 'U_LV063_n', 'U_MV063_n', 'U_LV064_n', 'U_MV064_n', 'U_LV065_n', 'U_MV065_n', 'U_LV066_n', 'U_MV066_n', 'U_LV067_n', 'U_MV067_n', 'U_LV068_n', 'U_MV068_n', 'U_LV069_n', 'U_MV069_n', 'U_LV070_n', 'U_MV070_n', 'U_LV071_n', 'U_MV071_n', 'U_LV072_n', 'U_MV072_n', 'U_LV073_n', 'U_MV073_n', 'U_LV074_n', 'U_MV074_n', 'U_LV075_n', 'U_MV075_n', 'U_LV076_n', 'U_MV076_n', 'U_LV077_n', 'U_MV077_n', 'U_LV078_n', 'U_MV078_n', 'U_LV079_n', 'U_MV079_n', 'U_LV080_n', 'U_MV080_n', 'U_LV081_n', 'U_MV081_n', 'U_LV082_n', 'U_MV082_n', 'U_LV083_n', 'U_MV083_n', 'U_LV084_n', 'U_MV084_n', 'U_LV085_n', 'U_MV085_n', 'U_LV086_n', 'U_MV086_n', 'U_LV087_n', 'U_MV087_n', 'U_LV088_n', 'U_MV088_n', 'U_LV089_n', 'U_MV089_n', 'U_LV090_n', 'U_MV090_n', 'U_LV091_n', 'U_MV091_n', 'U_LV092_n', 'U_MV092_n', 'U_LV093_n', 'U_MV093_n', 'U_LV094_n', 'U_MV094_n', 'U_LV095_n', 'U_MV095_n', 'U_LV096_n', 'U_MV096_n', 'U_LV097_n', 'U_MV097_n', 'U_LV098_n', 'U_MV098_n', 'U_LV099_n', 'U_MV099_n', 'U_LV100_n', 'U_MV100_n', 'S_n_GRID', 'F_n_GRID', 'X_v_GRID', 'R_v_GRID', 'K_delta_GRID', 'K_alpha_GRID', 'K_rocov_GRID', 'I_sc_LV001', 'I_mp_LV001', 'V_mp_LV001', 'V_oc_LV001', 'N_pv_s_LV001', 'N_pv_p_LV001', 'K_vt_LV001', 'K_it_LV001', 'v_lvrt_LV001', 'S_n_LV001', 'F_n_LV001', 'U_n_LV001', 'X_s_LV001', 'R_s_LV001', 'I_sc_LV002', 'I_mp_LV002', 'V_mp_LV002', 'V_oc_LV002', 'N_pv_s_LV002', 'N_pv_p_LV002', 'K_vt_LV002', 'K_it_LV002', 'v_lvrt_LV002', 'S_n_LV002', 'F_n_LV002', 'U_n_LV002', 'X_s_LV002', 'R_s_LV002', 'I_sc_LV003', 'I_mp_LV003', 'V_mp_LV003', 'V_oc_LV003', 'N_pv_s_LV003', 'N_pv_p_LV003', 'K_vt_LV003', 'K_it_LV003', 'v_lvrt_LV003', 'S_n_LV003', 'F_n_LV003', 'U_n_LV003', 'X_s_LV003', 'R_s_LV003', 'I_sc_LV004', 'I_mp_LV004', 'V_mp_LV004', 'V_oc_LV004', 'N_pv_s_LV004', 'N_pv_p_LV004', 'K_vt_LV004', 'K_it_LV004', 'v_lvrt_LV004', 'S_n_LV004', 'F_n_LV004', 'U_n_LV004', 'X_s_LV004', 'R_s_LV004', 'I_sc_LV005', 'I_mp_LV005', 'V_mp_LV005', 'V_oc_LV005', 'N_pv_s_LV005', 'N_pv_p_LV005', 'K_vt_LV005', 'K_it_LV005', 'v_lvrt_LV005', 'S_n_LV005', 'F_n_LV005', 'U_n_LV005', 'X_s_LV005', 'R_s_LV005', 'I_sc_LV006', 'I_mp_LV006', 'V_mp_LV006', 'V_oc_LV006', 'N_pv_s_LV006', 'N_pv_p_LV006', 'K_vt_LV006', 'K_it_LV006', 'v_lvrt_LV006', 'S_n_LV006', 'F_n_LV006', 'U_n_LV006', 'X_s_LV006', 'R_s_LV006', 'I_sc_LV007', 'I_mp_LV007', 'V_mp_LV007', 'V_oc_LV007', 'N_pv_s_LV007', 'N_pv_p_LV007', 'K_vt_LV007', 'K_it_LV007', 'v_lvrt_LV007', 'S_n_LV007', 'F_n_LV007', 'U_n_LV007', 'X_s_LV007', 'R_s_LV007', 'I_sc_LV008', 'I_mp_LV008', 'V_mp_LV008', 'V_oc_LV008', 'N_pv_s_LV008', 'N_pv_p_LV008', 'K_vt_LV008', 'K_it_LV008', 'v_lvrt_LV008', 'S_n_LV008', 'F_n_LV008', 'U_n_LV008', 'X_s_LV008', 'R_s_LV008', 'I_sc_LV009', 'I_mp_LV009', 'V_mp_LV009', 'V_oc_LV009', 'N_pv_s_LV009', 'N_pv_p_LV009', 'K_vt_LV009', 'K_it_LV009', 'v_lvrt_LV009', 'S_n_LV009', 'F_n_LV009', 'U_n_LV009', 'X_s_LV009', 'R_s_LV009', 'I_sc_LV010', 'I_mp_LV010', 'V_mp_LV010', 'V_oc_LV010', 'N_pv_s_LV010', 'N_pv_p_LV010', 'K_vt_LV010', 'K_it_LV010', 'v_lvrt_LV010', 'S_n_LV010', 'F_n_LV010', 'U_n_LV010', 'X_s_LV010', 'R_s_LV010', 'I_sc_LV011', 'I_mp_LV011', 'V_mp_LV011', 'V_oc_LV011', 'N_pv_s_LV011', 'N_pv_p_LV011', 'K_vt_LV011', 'K_it_LV011', 'v_lvrt_LV011', 'S_n_LV011', 'F_n_LV011', 'U_n_LV011', 'X_s_LV011', 'R_s_LV011', 'I_sc_LV012', 'I_mp_LV012', 'V_mp_LV012', 'V_oc_LV012', 'N_pv_s_LV012', 'N_pv_p_LV012', 'K_vt_LV012', 'K_it_LV012', 'v_lvrt_LV012', 'S_n_LV012', 'F_n_LV012', 'U_n_LV012', 'X_s_LV012', 'R_s_LV012', 'I_sc_LV013', 'I_mp_LV013', 'V_mp_LV013', 'V_oc_LV013', 'N_pv_s_LV013', 'N_pv_p_LV013', 'K_vt_LV013', 'K_it_LV013', 'v_lvrt_LV013', 'S_n_LV013', 'F_n_LV013', 'U_n_LV013', 'X_s_LV013', 'R_s_LV013', 'I_sc_LV014', 'I_mp_LV014', 'V_mp_LV014', 'V_oc_LV014', 'N_pv_s_LV014', 'N_pv_p_LV014', 'K_vt_LV014', 'K_it_LV014', 'v_lvrt_LV014', 'S_n_LV014', 'F_n_LV014', 'U_n_LV014', 'X_s_LV014', 'R_s_LV014', 'I_sc_LV015', 'I_mp_LV015', 'V_mp_LV015', 'V_oc_LV015', 'N_pv_s_LV015', 'N_pv_p_LV015', 'K_vt_LV015', 'K_it_LV015', 'v_lvrt_LV015', 'S_n_LV015', 'F_n_LV015', 'U_n_LV015', 'X_s_LV015', 'R_s_LV015', 'I_sc_LV016', 'I_mp_LV016', 'V_mp_LV016', 'V_oc_LV016', 'N_pv_s_LV016', 'N_pv_p_LV016', 'K_vt_LV016', 'K_it_LV016', 'v_lvrt_LV016', 'S_n_LV016', 'F_n_LV016', 'U_n_LV016', 'X_s_LV016', 'R_s_LV016', 'I_sc_LV017', 'I_mp_LV017', 'V_mp_LV017', 'V_oc_LV017', 'N_pv_s_LV017', 'N_pv_p_LV017', 'K_vt_LV017', 'K_it_LV017', 'v_lvrt_LV017', 'S_n_LV017', 'F_n_LV017', 'U_n_LV017', 'X_s_LV017', 'R_s_LV017', 'I_sc_LV018', 'I_mp_LV018', 'V_mp_LV018', 'V_oc_LV018', 'N_pv_s_LV018', 'N_pv_p_LV018', 'K_vt_LV018', 'K_it_LV018', 'v_lvrt_LV018', 'S_n_LV018', 'F_n_LV018', 'U_n_LV018', 'X_s_LV018', 'R_s_LV018', 'I_sc_LV019', 'I_mp_LV019', 'V_mp_LV019', 'V_oc_LV019', 'N_pv_s_LV019', 'N_pv_p_LV019', 'K_vt_LV019', 'K_it_LV019', 'v_lvrt_LV019', 'S_n_LV019', 'F_n_LV019', 'U_n_LV019', 'X_s_LV019', 'R_s_LV019', 'I_sc_LV020', 'I_mp_LV020', 'V_mp_LV020', 'V_oc_LV020', 'N_pv_s_LV020', 'N_pv_p_LV020', 'K_vt_LV020', 'K_it_LV020', 'v_lvrt_LV020', 'S_n_LV020', 'F_n_LV020', 'U_n_LV020', 'X_s_LV020', 'R_s_LV020', 'I_sc_LV021', 'I_mp_LV021', 'V_mp_LV021', 'V_oc_LV021', 'N_pv_s_LV021', 'N_pv_p_LV021', 'K_vt_LV021', 'K_it_LV021', 'v_lvrt_LV021', 'S_n_LV021', 'F_n_LV021', 'U_n_LV021', 'X_s_LV021', 'R_s_LV021', 'I_sc_LV022', 'I_mp_LV022', 'V_mp_LV022', 'V_oc_LV022', 'N_pv_s_LV022', 'N_pv_p_LV022', 'K_vt_LV022', 'K_it_LV022', 'v_lvrt_LV022', 'S_n_LV022', 'F_n_LV022', 'U_n_LV022', 'X_s_LV022', 'R_s_LV022', 'I_sc_LV023', 'I_mp_LV023', 'V_mp_LV023', 'V_oc_LV023', 'N_pv_s_LV023', 'N_pv_p_LV023', 'K_vt_LV023', 'K_it_LV023', 'v_lvrt_LV023', 'S_n_LV023', 'F_n_LV023', 'U_n_LV023', 'X_s_LV023', 'R_s_LV023', 'I_sc_LV024', 'I_mp_LV024', 'V_mp_LV024', 'V_oc_LV024', 'N_pv_s_LV024', 'N_pv_p_LV024', 'K_vt_LV024', 'K_it_LV024', 'v_lvrt_LV024', 'S_n_LV024', 'F_n_LV024', 'U_n_LV024', 'X_s_LV024', 'R_s_LV024', 'I_sc_LV025', 'I_mp_LV025', 'V_mp_LV025', 'V_oc_LV025', 'N_pv_s_LV025', 'N_pv_p_LV025', 'K_vt_LV025', 'K_it_LV025', 'v_lvrt_LV025', 'S_n_LV025', 'F_n_LV025', 'U_n_LV025', 'X_s_LV025', 'R_s_LV025', 'I_sc_LV026', 'I_mp_LV026', 'V_mp_LV026', 'V_oc_LV026', 'N_pv_s_LV026', 'N_pv_p_LV026', 'K_vt_LV026', 'K_it_LV026', 'v_lvrt_LV026', 'S_n_LV026', 'F_n_LV026', 'U_n_LV026', 'X_s_LV026', 'R_s_LV026', 'I_sc_LV027', 'I_mp_LV027', 'V_mp_LV027', 'V_oc_LV027', 'N_pv_s_LV027', 'N_pv_p_LV027', 'K_vt_LV027', 'K_it_LV027', 'v_lvrt_LV027', 'S_n_LV027', 'F_n_LV027', 'U_n_LV027', 'X_s_LV027', 'R_s_LV027', 'I_sc_LV028', 'I_mp_LV028', 'V_mp_LV028', 'V_oc_LV028', 'N_pv_s_LV028', 'N_pv_p_LV028', 'K_vt_LV028', 'K_it_LV028', 'v_lvrt_LV028', 'S_n_LV028', 'F_n_LV028', 'U_n_LV028', 'X_s_LV028', 'R_s_LV028', 'I_sc_LV029', 'I_mp_LV029', 'V_mp_LV029', 'V_oc_LV029', 'N_pv_s_LV029', 'N_pv_p_LV029', 'K_vt_LV029', 'K_it_LV029', 'v_lvrt_LV029', 'S_n_LV029', 'F_n_LV029', 'U_n_LV029', 'X_s_LV029', 'R_s_LV029', 'I_sc_LV030', 'I_mp_LV030', 'V_mp_LV030', 'V_oc_LV030', 'N_pv_s_LV030', 'N_pv_p_LV030', 'K_vt_LV030', 'K_it_LV030', 'v_lvrt_LV030', 'S_n_LV030', 'F_n_LV030', 'U_n_LV030', 'X_s_LV030', 'R_s_LV030', 'I_sc_LV031', 'I_mp_LV031', 'V_mp_LV031', 'V_oc_LV031', 'N_pv_s_LV031', 'N_pv_p_LV031', 'K_vt_LV031', 'K_it_LV031', 'v_lvrt_LV031', 'S_n_LV031', 'F_n_LV031', 'U_n_LV031', 'X_s_LV031', 'R_s_LV031', 'I_sc_LV032', 'I_mp_LV032', 'V_mp_LV032', 'V_oc_LV032', 'N_pv_s_LV032', 'N_pv_p_LV032', 'K_vt_LV032', 'K_it_LV032', 'v_lvrt_LV032', 'S_n_LV032', 'F_n_LV032', 'U_n_LV032', 'X_s_LV032', 'R_s_LV032', 'I_sc_LV033', 'I_mp_LV033', 'V_mp_LV033', 'V_oc_LV033', 'N_pv_s_LV033', 'N_pv_p_LV033', 'K_vt_LV033', 'K_it_LV033', 'v_lvrt_LV033', 'S_n_LV033', 'F_n_LV033', 'U_n_LV033', 'X_s_LV033', 'R_s_LV033', 'I_sc_LV034', 'I_mp_LV034', 'V_mp_LV034', 'V_oc_LV034', 'N_pv_s_LV034', 'N_pv_p_LV034', 'K_vt_LV034', 'K_it_LV034', 'v_lvrt_LV034', 'S_n_LV034', 'F_n_LV034', 'U_n_LV034', 'X_s_LV034', 'R_s_LV034', 'I_sc_LV035', 'I_mp_LV035', 'V_mp_LV035', 'V_oc_LV035', 'N_pv_s_LV035', 'N_pv_p_LV035', 'K_vt_LV035', 'K_it_LV035', 'v_lvrt_LV035', 'S_n_LV035', 'F_n_LV035', 'U_n_LV035', 'X_s_LV035', 'R_s_LV035', 'I_sc_LV036', 'I_mp_LV036', 'V_mp_LV036', 'V_oc_LV036', 'N_pv_s_LV036', 'N_pv_p_LV036', 'K_vt_LV036', 'K_it_LV036', 'v_lvrt_LV036', 'S_n_LV036', 'F_n_LV036', 'U_n_LV036', 'X_s_LV036', 'R_s_LV036', 'I_sc_LV037', 'I_mp_LV037', 'V_mp_LV037', 'V_oc_LV037', 'N_pv_s_LV037', 'N_pv_p_LV037', 'K_vt_LV037', 'K_it_LV037', 'v_lvrt_LV037', 'S_n_LV037', 'F_n_LV037', 'U_n_LV037', 'X_s_LV037', 'R_s_LV037', 'I_sc_LV038', 'I_mp_LV038', 'V_mp_LV038', 'V_oc_LV038', 'N_pv_s_LV038', 'N_pv_p_LV038', 'K_vt_LV038', 'K_it_LV038', 'v_lvrt_LV038', 'S_n_LV038', 'F_n_LV038', 'U_n_LV038', 'X_s_LV038', 'R_s_LV038', 'I_sc_LV039', 'I_mp_LV039', 'V_mp_LV039', 'V_oc_LV039', 'N_pv_s_LV039', 'N_pv_p_LV039', 'K_vt_LV039', 'K_it_LV039', 'v_lvrt_LV039', 'S_n_LV039', 'F_n_LV039', 'U_n_LV039', 'X_s_LV039', 'R_s_LV039', 'I_sc_LV040', 'I_mp_LV040', 'V_mp_LV040', 'V_oc_LV040', 'N_pv_s_LV040', 'N_pv_p_LV040', 'K_vt_LV040', 'K_it_LV040', 'v_lvrt_LV040', 'S_n_LV040', 'F_n_LV040', 'U_n_LV040', 'X_s_LV040', 'R_s_LV040', 'I_sc_LV041', 'I_mp_LV041', 'V_mp_LV041', 'V_oc_LV041', 'N_pv_s_LV041', 'N_pv_p_LV041', 'K_vt_LV041', 'K_it_LV041', 'v_lvrt_LV041', 'S_n_LV041', 'F_n_LV041', 'U_n_LV041', 'X_s_LV041', 'R_s_LV041', 'I_sc_LV042', 'I_mp_LV042', 'V_mp_LV042', 'V_oc_LV042', 'N_pv_s_LV042', 'N_pv_p_LV042', 'K_vt_LV042', 'K_it_LV042', 'v_lvrt_LV042', 'S_n_LV042', 'F_n_LV042', 'U_n_LV042', 'X_s_LV042', 'R_s_LV042', 'I_sc_LV043', 'I_mp_LV043', 'V_mp_LV043', 'V_oc_LV043', 'N_pv_s_LV043', 'N_pv_p_LV043', 'K_vt_LV043', 'K_it_LV043', 'v_lvrt_LV043', 'S_n_LV043', 'F_n_LV043', 'U_n_LV043', 'X_s_LV043', 'R_s_LV043', 'I_sc_LV044', 'I_mp_LV044', 'V_mp_LV044', 'V_oc_LV044', 'N_pv_s_LV044', 'N_pv_p_LV044', 'K_vt_LV044', 'K_it_LV044', 'v_lvrt_LV044', 'S_n_LV044', 'F_n_LV044', 'U_n_LV044', 'X_s_LV044', 'R_s_LV044', 'I_sc_LV045', 'I_mp_LV045', 'V_mp_LV045', 'V_oc_LV045', 'N_pv_s_LV045', 'N_pv_p_LV045', 'K_vt_LV045', 'K_it_LV045', 'v_lvrt_LV045', 'S_n_LV045', 'F_n_LV045', 'U_n_LV045', 'X_s_LV045', 'R_s_LV045', 'I_sc_LV046', 'I_mp_LV046', 'V_mp_LV046', 'V_oc_LV046', 'N_pv_s_LV046', 'N_pv_p_LV046', 'K_vt_LV046', 'K_it_LV046', 'v_lvrt_LV046', 'S_n_LV046', 'F_n_LV046', 'U_n_LV046', 'X_s_LV046', 'R_s_LV046', 'I_sc_LV047', 'I_mp_LV047', 'V_mp_LV047', 'V_oc_LV047', 'N_pv_s_LV047', 'N_pv_p_LV047', 'K_vt_LV047', 'K_it_LV047', 'v_lvrt_LV047', 'S_n_LV047', 'F_n_LV047', 'U_n_LV047', 'X_s_LV047', 'R_s_LV047', 'I_sc_LV048', 'I_mp_LV048', 'V_mp_LV048', 'V_oc_LV048', 'N_pv_s_LV048', 'N_pv_p_LV048', 'K_vt_LV048', 'K_it_LV048', 'v_lvrt_LV048', 'S_n_LV048', 'F_n_LV048', 'U_n_LV048', 'X_s_LV048', 'R_s_LV048', 'I_sc_LV049', 'I_mp_LV049', 'V_mp_LV049', 'V_oc_LV049', 'N_pv_s_LV049', 'N_pv_p_LV049', 'K_vt_LV049', 'K_it_LV049', 'v_lvrt_LV049', 'S_n_LV049', 'F_n_LV049', 'U_n_LV049', 'X_s_LV049', 'R_s_LV049', 'I_sc_LV050', 'I_mp_LV050', 'V_mp_LV050', 'V_oc_LV050', 'N_pv_s_LV050', 'N_pv_p_LV050', 'K_vt_LV050', 'K_it_LV050', 'v_lvrt_LV050', 'S_n_LV050', 'F_n_LV050', 'U_n_LV050', 'X_s_LV050', 'R_s_LV050', 'I_sc_LV051', 'I_mp_LV051', 'V_mp_LV051', 'V_oc_LV051', 'N_pv_s_LV051', 'N_pv_p_LV051', 'K_vt_LV051', 'K_it_LV051', 'v_lvrt_LV051', 'S_n_LV051', 'F_n_LV051', 'U_n_LV051', 'X_s_LV051', 'R_s_LV051', 'I_sc_LV052', 'I_mp_LV052', 'V_mp_LV052', 'V_oc_LV052', 'N_pv_s_LV052', 'N_pv_p_LV052', 'K_vt_LV052', 'K_it_LV052', 'v_lvrt_LV052', 'S_n_LV052', 'F_n_LV052', 'U_n_LV052', 'X_s_LV052', 'R_s_LV052', 'I_sc_LV053', 'I_mp_LV053', 'V_mp_LV053', 'V_oc_LV053', 'N_pv_s_LV053', 'N_pv_p_LV053', 'K_vt_LV053', 'K_it_LV053', 'v_lvrt_LV053', 'S_n_LV053', 'F_n_LV053', 'U_n_LV053', 'X_s_LV053', 'R_s_LV053', 'I_sc_LV054', 'I_mp_LV054', 'V_mp_LV054', 'V_oc_LV054', 'N_pv_s_LV054', 'N_pv_p_LV054', 'K_vt_LV054', 'K_it_LV054', 'v_lvrt_LV054', 'S_n_LV054', 'F_n_LV054', 'U_n_LV054', 'X_s_LV054', 'R_s_LV054', 'I_sc_LV055', 'I_mp_LV055', 'V_mp_LV055', 'V_oc_LV055', 'N_pv_s_LV055', 'N_pv_p_LV055', 'K_vt_LV055', 'K_it_LV055', 'v_lvrt_LV055', 'S_n_LV055', 'F_n_LV055', 'U_n_LV055', 'X_s_LV055', 'R_s_LV055', 'I_sc_LV056', 'I_mp_LV056', 'V_mp_LV056', 'V_oc_LV056', 'N_pv_s_LV056', 'N_pv_p_LV056', 'K_vt_LV056', 'K_it_LV056', 'v_lvrt_LV056', 'S_n_LV056', 'F_n_LV056', 'U_n_LV056', 'X_s_LV056', 'R_s_LV056', 'I_sc_LV057', 'I_mp_LV057', 'V_mp_LV057', 'V_oc_LV057', 'N_pv_s_LV057', 'N_pv_p_LV057', 'K_vt_LV057', 'K_it_LV057', 'v_lvrt_LV057', 'S_n_LV057', 'F_n_LV057', 'U_n_LV057', 'X_s_LV057', 'R_s_LV057', 'I_sc_LV058', 'I_mp_LV058', 'V_mp_LV058', 'V_oc_LV058', 'N_pv_s_LV058', 'N_pv_p_LV058', 'K_vt_LV058', 'K_it_LV058', 'v_lvrt_LV058', 'S_n_LV058', 'F_n_LV058', 'U_n_LV058', 'X_s_LV058', 'R_s_LV058', 'I_sc_LV059', 'I_mp_LV059', 'V_mp_LV059', 'V_oc_LV059', 'N_pv_s_LV059', 'N_pv_p_LV059', 'K_vt_LV059', 'K_it_LV059', 'v_lvrt_LV059', 'S_n_LV059', 'F_n_LV059', 'U_n_LV059', 'X_s_LV059', 'R_s_LV059', 'I_sc_LV060', 'I_mp_LV060', 'V_mp_LV060', 'V_oc_LV060', 'N_pv_s_LV060', 'N_pv_p_LV060', 'K_vt_LV060', 'K_it_LV060', 'v_lvrt_LV060', 'S_n_LV060', 'F_n_LV060', 'U_n_LV060', 'X_s_LV060', 'R_s_LV060', 'I_sc_LV061', 'I_mp_LV061', 'V_mp_LV061', 'V_oc_LV061', 'N_pv_s_LV061', 'N_pv_p_LV061', 'K_vt_LV061', 'K_it_LV061', 'v_lvrt_LV061', 'S_n_LV061', 'F_n_LV061', 'U_n_LV061', 'X_s_LV061', 'R_s_LV061', 'I_sc_LV062', 'I_mp_LV062', 'V_mp_LV062', 'V_oc_LV062', 'N_pv_s_LV062', 'N_pv_p_LV062', 'K_vt_LV062', 'K_it_LV062', 'v_lvrt_LV062', 'S_n_LV062', 'F_n_LV062', 'U_n_LV062', 'X_s_LV062', 'R_s_LV062', 'I_sc_LV063', 'I_mp_LV063', 'V_mp_LV063', 'V_oc_LV063', 'N_pv_s_LV063', 'N_pv_p_LV063', 'K_vt_LV063', 'K_it_LV063', 'v_lvrt_LV063', 'S_n_LV063', 'F_n_LV063', 'U_n_LV063', 'X_s_LV063', 'R_s_LV063', 'I_sc_LV064', 'I_mp_LV064', 'V_mp_LV064', 'V_oc_LV064', 'N_pv_s_LV064', 'N_pv_p_LV064', 'K_vt_LV064', 'K_it_LV064', 'v_lvrt_LV064', 'S_n_LV064', 'F_n_LV064', 'U_n_LV064', 'X_s_LV064', 'R_s_LV064', 'I_sc_LV065', 'I_mp_LV065', 'V_mp_LV065', 'V_oc_LV065', 'N_pv_s_LV065', 'N_pv_p_LV065', 'K_vt_LV065', 'K_it_LV065', 'v_lvrt_LV065', 'S_n_LV065', 'F_n_LV065', 'U_n_LV065', 'X_s_LV065', 'R_s_LV065', 'I_sc_LV066', 'I_mp_LV066', 'V_mp_LV066', 'V_oc_LV066', 'N_pv_s_LV066', 'N_pv_p_LV066', 'K_vt_LV066', 'K_it_LV066', 'v_lvrt_LV066', 'S_n_LV066', 'F_n_LV066', 'U_n_LV066', 'X_s_LV066', 'R_s_LV066', 'I_sc_LV067', 'I_mp_LV067', 'V_mp_LV067', 'V_oc_LV067', 'N_pv_s_LV067', 'N_pv_p_LV067', 'K_vt_LV067', 'K_it_LV067', 'v_lvrt_LV067', 'S_n_LV067', 'F_n_LV067', 'U_n_LV067', 'X_s_LV067', 'R_s_LV067', 'I_sc_LV068', 'I_mp_LV068', 'V_mp_LV068', 'V_oc_LV068', 'N_pv_s_LV068', 'N_pv_p_LV068', 'K_vt_LV068', 'K_it_LV068', 'v_lvrt_LV068', 'S_n_LV068', 'F_n_LV068', 'U_n_LV068', 'X_s_LV068', 'R_s_LV068', 'I_sc_LV069', 'I_mp_LV069', 'V_mp_LV069', 'V_oc_LV069', 'N_pv_s_LV069', 'N_pv_p_LV069', 'K_vt_LV069', 'K_it_LV069', 'v_lvrt_LV069', 'S_n_LV069', 'F_n_LV069', 'U_n_LV069', 'X_s_LV069', 'R_s_LV069', 'I_sc_LV070', 'I_mp_LV070', 'V_mp_LV070', 'V_oc_LV070', 'N_pv_s_LV070', 'N_pv_p_LV070', 'K_vt_LV070', 'K_it_LV070', 'v_lvrt_LV070', 'S_n_LV070', 'F_n_LV070', 'U_n_LV070', 'X_s_LV070', 'R_s_LV070', 'I_sc_LV071', 'I_mp_LV071', 'V_mp_LV071', 'V_oc_LV071', 'N_pv_s_LV071', 'N_pv_p_LV071', 'K_vt_LV071', 'K_it_LV071', 'v_lvrt_LV071', 'S_n_LV071', 'F_n_LV071', 'U_n_LV071', 'X_s_LV071', 'R_s_LV071', 'I_sc_LV072', 'I_mp_LV072', 'V_mp_LV072', 'V_oc_LV072', 'N_pv_s_LV072', 'N_pv_p_LV072', 'K_vt_LV072', 'K_it_LV072', 'v_lvrt_LV072', 'S_n_LV072', 'F_n_LV072', 'U_n_LV072', 'X_s_LV072', 'R_s_LV072', 'I_sc_LV073', 'I_mp_LV073', 'V_mp_LV073', 'V_oc_LV073', 'N_pv_s_LV073', 'N_pv_p_LV073', 'K_vt_LV073', 'K_it_LV073', 'v_lvrt_LV073', 'S_n_LV073', 'F_n_LV073', 'U_n_LV073', 'X_s_LV073', 'R_s_LV073', 'I_sc_LV074', 'I_mp_LV074', 'V_mp_LV074', 'V_oc_LV074', 'N_pv_s_LV074', 'N_pv_p_LV074', 'K_vt_LV074', 'K_it_LV074', 'v_lvrt_LV074', 'S_n_LV074', 'F_n_LV074', 'U_n_LV074', 'X_s_LV074', 'R_s_LV074', 'I_sc_LV075', 'I_mp_LV075', 'V_mp_LV075', 'V_oc_LV075', 'N_pv_s_LV075', 'N_pv_p_LV075', 'K_vt_LV075', 'K_it_LV075', 'v_lvrt_LV075', 'S_n_LV075', 'F_n_LV075', 'U_n_LV075', 'X_s_LV075', 'R_s_LV075', 'I_sc_LV076', 'I_mp_LV076', 'V_mp_LV076', 'V_oc_LV076', 'N_pv_s_LV076', 'N_pv_p_LV076', 'K_vt_LV076', 'K_it_LV076', 'v_lvrt_LV076', 'S_n_LV076', 'F_n_LV076', 'U_n_LV076', 'X_s_LV076', 'R_s_LV076', 'I_sc_LV077', 'I_mp_LV077', 'V_mp_LV077', 'V_oc_LV077', 'N_pv_s_LV077', 'N_pv_p_LV077', 'K_vt_LV077', 'K_it_LV077', 'v_lvrt_LV077', 'S_n_LV077', 'F_n_LV077', 'U_n_LV077', 'X_s_LV077', 'R_s_LV077', 'I_sc_LV078', 'I_mp_LV078', 'V_mp_LV078', 'V_oc_LV078', 'N_pv_s_LV078', 'N_pv_p_LV078', 'K_vt_LV078', 'K_it_LV078', 'v_lvrt_LV078', 'S_n_LV078', 'F_n_LV078', 'U_n_LV078', 'X_s_LV078', 'R_s_LV078', 'I_sc_LV079', 'I_mp_LV079', 'V_mp_LV079', 'V_oc_LV079', 'N_pv_s_LV079', 'N_pv_p_LV079', 'K_vt_LV079', 'K_it_LV079', 'v_lvrt_LV079', 'S_n_LV079', 'F_n_LV079', 'U_n_LV079', 'X_s_LV079', 'R_s_LV079', 'I_sc_LV080', 'I_mp_LV080', 'V_mp_LV080', 'V_oc_LV080', 'N_pv_s_LV080', 'N_pv_p_LV080', 'K_vt_LV080', 'K_it_LV080', 'v_lvrt_LV080', 'S_n_LV080', 'F_n_LV080', 'U_n_LV080', 'X_s_LV080', 'R_s_LV080', 'I_sc_LV081', 'I_mp_LV081', 'V_mp_LV081', 'V_oc_LV081', 'N_pv_s_LV081', 'N_pv_p_LV081', 'K_vt_LV081', 'K_it_LV081', 'v_lvrt_LV081', 'S_n_LV081', 'F_n_LV081', 'U_n_LV081', 'X_s_LV081', 'R_s_LV081', 'I_sc_LV082', 'I_mp_LV082', 'V_mp_LV082', 'V_oc_LV082', 'N_pv_s_LV082', 'N_pv_p_LV082', 'K_vt_LV082', 'K_it_LV082', 'v_lvrt_LV082', 'S_n_LV082', 'F_n_LV082', 'U_n_LV082', 'X_s_LV082', 'R_s_LV082', 'I_sc_LV083', 'I_mp_LV083', 'V_mp_LV083', 'V_oc_LV083', 'N_pv_s_LV083', 'N_pv_p_LV083', 'K_vt_LV083', 'K_it_LV083', 'v_lvrt_LV083', 'S_n_LV083', 'F_n_LV083', 'U_n_LV083', 'X_s_LV083', 'R_s_LV083', 'I_sc_LV084', 'I_mp_LV084', 'V_mp_LV084', 'V_oc_LV084', 'N_pv_s_LV084', 'N_pv_p_LV084', 'K_vt_LV084', 'K_it_LV084', 'v_lvrt_LV084', 'S_n_LV084', 'F_n_LV084', 'U_n_LV084', 'X_s_LV084', 'R_s_LV084', 'I_sc_LV085', 'I_mp_LV085', 'V_mp_LV085', 'V_oc_LV085', 'N_pv_s_LV085', 'N_pv_p_LV085', 'K_vt_LV085', 'K_it_LV085', 'v_lvrt_LV085', 'S_n_LV085', 'F_n_LV085', 'U_n_LV085', 'X_s_LV085', 'R_s_LV085', 'I_sc_LV086', 'I_mp_LV086', 'V_mp_LV086', 'V_oc_LV086', 'N_pv_s_LV086', 'N_pv_p_LV086', 'K_vt_LV086', 'K_it_LV086', 'v_lvrt_LV086', 'S_n_LV086', 'F_n_LV086', 'U_n_LV086', 'X_s_LV086', 'R_s_LV086', 'I_sc_LV087', 'I_mp_LV087', 'V_mp_LV087', 'V_oc_LV087', 'N_pv_s_LV087', 'N_pv_p_LV087', 'K_vt_LV087', 'K_it_LV087', 'v_lvrt_LV087', 'S_n_LV087', 'F_n_LV087', 'U_n_LV087', 'X_s_LV087', 'R_s_LV087', 'I_sc_LV088', 'I_mp_LV088', 'V_mp_LV088', 'V_oc_LV088', 'N_pv_s_LV088', 'N_pv_p_LV088', 'K_vt_LV088', 'K_it_LV088', 'v_lvrt_LV088', 'S_n_LV088', 'F_n_LV088', 'U_n_LV088', 'X_s_LV088', 'R_s_LV088', 'I_sc_LV089', 'I_mp_LV089', 'V_mp_LV089', 'V_oc_LV089', 'N_pv_s_LV089', 'N_pv_p_LV089', 'K_vt_LV089', 'K_it_LV089', 'v_lvrt_LV089', 'S_n_LV089', 'F_n_LV089', 'U_n_LV089', 'X_s_LV089', 'R_s_LV089', 'I_sc_LV090', 'I_mp_LV090', 'V_mp_LV090', 'V_oc_LV090', 'N_pv_s_LV090', 'N_pv_p_LV090', 'K_vt_LV090', 'K_it_LV090', 'v_lvrt_LV090', 'S_n_LV090', 'F_n_LV090', 'U_n_LV090', 'X_s_LV090', 'R_s_LV090', 'I_sc_LV091', 'I_mp_LV091', 'V_mp_LV091', 'V_oc_LV091', 'N_pv_s_LV091', 'N_pv_p_LV091', 'K_vt_LV091', 'K_it_LV091', 'v_lvrt_LV091', 'S_n_LV091', 'F_n_LV091', 'U_n_LV091', 'X_s_LV091', 'R_s_LV091', 'I_sc_LV092', 'I_mp_LV092', 'V_mp_LV092', 'V_oc_LV092', 'N_pv_s_LV092', 'N_pv_p_LV092', 'K_vt_LV092', 'K_it_LV092', 'v_lvrt_LV092', 'S_n_LV092', 'F_n_LV092', 'U_n_LV092', 'X_s_LV092', 'R_s_LV092', 'I_sc_LV093', 'I_mp_LV093', 'V_mp_LV093', 'V_oc_LV093', 'N_pv_s_LV093', 'N_pv_p_LV093', 'K_vt_LV093', 'K_it_LV093', 'v_lvrt_LV093', 'S_n_LV093', 'F_n_LV093', 'U_n_LV093', 'X_s_LV093', 'R_s_LV093', 'I_sc_LV094', 'I_mp_LV094', 'V_mp_LV094', 'V_oc_LV094', 'N_pv_s_LV094', 'N_pv_p_LV094', 'K_vt_LV094', 'K_it_LV094', 'v_lvrt_LV094', 'S_n_LV094', 'F_n_LV094', 'U_n_LV094', 'X_s_LV094', 'R_s_LV094', 'I_sc_LV095', 'I_mp_LV095', 'V_mp_LV095', 'V_oc_LV095', 'N_pv_s_LV095', 'N_pv_p_LV095', 'K_vt_LV095', 'K_it_LV095', 'v_lvrt_LV095', 'S_n_LV095', 'F_n_LV095', 'U_n_LV095', 'X_s_LV095', 'R_s_LV095', 'I_sc_LV096', 'I_mp_LV096', 'V_mp_LV096', 'V_oc_LV096', 'N_pv_s_LV096', 'N_pv_p_LV096', 'K_vt_LV096', 'K_it_LV096', 'v_lvrt_LV096', 'S_n_LV096', 'F_n_LV096', 'U_n_LV096', 'X_s_LV096', 'R_s_LV096', 'I_sc_LV097', 'I_mp_LV097', 'V_mp_LV097', 'V_oc_LV097', 'N_pv_s_LV097', 'N_pv_p_LV097', 'K_vt_LV097', 'K_it_LV097', 'v_lvrt_LV097', 'S_n_LV097', 'F_n_LV097', 'U_n_LV097', 'X_s_LV097', 'R_s_LV097', 'I_sc_LV098', 'I_mp_LV098', 'V_mp_LV098', 'V_oc_LV098', 'N_pv_s_LV098', 'N_pv_p_LV098', 'K_vt_LV098', 'K_it_LV098', 'v_lvrt_LV098', 'S_n_LV098', 'F_n_LV098', 'U_n_LV098', 'X_s_LV098', 'R_s_LV098', 'I_sc_LV099', 'I_mp_LV099', 'V_mp_LV099', 'V_oc_LV099', 'N_pv_s_LV099', 'N_pv_p_LV099', 'K_vt_LV099', 'K_it_LV099', 'v_lvrt_LV099', 'S_n_LV099', 'F_n_LV099', 'U_n_LV099', 'X_s_LV099', 'R_s_LV099', 'I_sc_LV100', 'I_mp_LV100', 'V_mp_LV100', 'V_oc_LV100', 'N_pv_s_LV100', 'N_pv_p_LV100', 'K_vt_LV100', 'K_it_LV100', 'v_lvrt_LV100', 'S_n_LV100', 'F_n_LV100', 'U_n_LV100', 'X_s_LV100', 'R_s_LV100', 'K_p_agc', 'K_i_agc', 'K_xif'] 
        self.params_values_list  = [100000000.0, 0.0, -24.0, -0.0, 0.0, -60.0, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.5714285714285714, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.5454545454545455, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.5217391304347826, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.5, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.4799999999999999, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.46153846153846156, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.4444444444444444, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.4285714285714285, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.4137931034482759, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.4, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.3870967741935484, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.375, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.36363636363636365, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.3529411764705882, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.3428571428571428, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.33333333333333326, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.32432432432432423, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.31578947368421056, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.3076923076923077, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.3, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.2926829268292683, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.2857142857142857, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.27906976744186046, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.27272727272727276, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.26666666666666666, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.2608695652173913, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.2553191489361702, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.25, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.24489795918367352, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.23529411764705882, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.23076923076923078, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.2264150943396226, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.22222222222222218, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.21818181818181814, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.21428571428571433, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.2105263157894737, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.20689655172413796, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.20338983050847456, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.2, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.19672131147540983, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1935483870967742, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.19047619047619047, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1875, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1846153846153846, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.18181818181818182, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.17910447761194032, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1764705882352941, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.17391304347826086, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1714285714285714, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.16901408450704228, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.16666666666666669, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1643835616438356, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.16216216216216217, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.16, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.15789473684210528, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.15584415584415584, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.15384615384615385, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1518987341772152, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.15, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.14814814814814814, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.14634146341463414, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.14457831325301204, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.14285714285714285, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1411764705882353, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.13953488372093023, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.13793103448275862, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.13636363636363638, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.13483146067415727, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1333333333333333, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.13186813186813187, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.13043478260869565, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.12903225806451613, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1276595744680851, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.12631578947368421, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.125, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.12371134020618556, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.12244897959183676, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.12121212121212122, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11999999999999998, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11881188118811882, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11764705882352941, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11650485436893204, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11538461538461539, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11428571428571427, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11320754716981132, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11214953271028039, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.1111111111111111, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.11009174311926605, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10909090909090909, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10810810810810811, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10714285714285712, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10619469026548674, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10526315789473684, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10434782608695652, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10344827586206898, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10256410256410256, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10169491525423728, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.10084033613445378, -0.0, 0.0, -0.23999999999999996, -0.0, 0.0, -0.09999999999999999, -0.0, 20000.0, 132000.0, 132000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 1000000000.0, 50.0, 0.001, 0.0, 0.001, 1e-06, 1e-06, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 8, 3.56, 33.7, 42.1, 25, 250, -0.16, 0.065, 0.8, 1000000.0, 50.0, 400.0, 0.1, 0.01, 0.0, 0.0, 0.01] 
        self.inputs_ini_list = ['P_POI_MV', 'Q_POI_MV', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'P_LV001', 'Q_LV001', 'P_MV001', 'Q_MV001', 'P_LV002', 'Q_LV002', 'P_MV002', 'Q_MV002', 'P_LV003', 'Q_LV003', 'P_MV003', 'Q_MV003', 'P_LV004', 'Q_LV004', 'P_MV004', 'Q_MV004', 'P_LV005', 'Q_LV005', 'P_MV005', 'Q_MV005', 'P_LV006', 'Q_LV006', 'P_MV006', 'Q_MV006', 'P_LV007', 'Q_LV007', 'P_MV007', 'Q_MV007', 'P_LV008', 'Q_LV008', 'P_MV008', 'Q_MV008', 'P_LV009', 'Q_LV009', 'P_MV009', 'Q_MV009', 'P_LV010', 'Q_LV010', 'P_MV010', 'Q_MV010', 'P_LV011', 'Q_LV011', 'P_MV011', 'Q_MV011', 'P_LV012', 'Q_LV012', 'P_MV012', 'Q_MV012', 'P_LV013', 'Q_LV013', 'P_MV013', 'Q_MV013', 'P_LV014', 'Q_LV014', 'P_MV014', 'Q_MV014', 'P_LV015', 'Q_LV015', 'P_MV015', 'Q_MV015', 'P_LV016', 'Q_LV016', 'P_MV016', 'Q_MV016', 'P_LV017', 'Q_LV017', 'P_MV017', 'Q_MV017', 'P_LV018', 'Q_LV018', 'P_MV018', 'Q_MV018', 'P_LV019', 'Q_LV019', 'P_MV019', 'Q_MV019', 'P_LV020', 'Q_LV020', 'P_MV020', 'Q_MV020', 'P_LV021', 'Q_LV021', 'P_MV021', 'Q_MV021', 'P_LV022', 'Q_LV022', 'P_MV022', 'Q_MV022', 'P_LV023', 'Q_LV023', 'P_MV023', 'Q_MV023', 'P_LV024', 'Q_LV024', 'P_MV024', 'Q_MV024', 'P_LV025', 'Q_LV025', 'P_MV025', 'Q_MV025', 'P_LV026', 'Q_LV026', 'P_MV026', 'Q_MV026', 'P_LV027', 'Q_LV027', 'P_MV027', 'Q_MV027', 'P_LV028', 'Q_LV028', 'P_MV028', 'Q_MV028', 'P_LV029', 'Q_LV029', 'P_MV029', 'Q_MV029', 'P_LV030', 'Q_LV030', 'P_MV030', 'Q_MV030', 'P_LV031', 'Q_LV031', 'P_MV031', 'Q_MV031', 'P_LV032', 'Q_LV032', 'P_MV032', 'Q_MV032', 'P_LV033', 'Q_LV033', 'P_MV033', 'Q_MV033', 'P_LV034', 'Q_LV034', 'P_MV034', 'Q_MV034', 'P_LV035', 'Q_LV035', 'P_MV035', 'Q_MV035', 'P_LV036', 'Q_LV036', 'P_MV036', 'Q_MV036', 'P_LV037', 'Q_LV037', 'P_MV037', 'Q_MV037', 'P_LV038', 'Q_LV038', 'P_MV038', 'Q_MV038', 'P_LV039', 'Q_LV039', 'P_MV039', 'Q_MV039', 'P_LV040', 'Q_LV040', 'P_MV040', 'Q_MV040', 'P_LV041', 'Q_LV041', 'P_MV041', 'Q_MV041', 'P_LV042', 'Q_LV042', 'P_MV042', 'Q_MV042', 'P_LV043', 'Q_LV043', 'P_MV043', 'Q_MV043', 'P_LV044', 'Q_LV044', 'P_MV044', 'Q_MV044', 'P_LV045', 'Q_LV045', 'P_MV045', 'Q_MV045', 'P_LV046', 'Q_LV046', 'P_MV046', 'Q_MV046', 'P_LV047', 'Q_LV047', 'P_MV047', 'Q_MV047', 'P_LV048', 'Q_LV048', 'P_MV048', 'Q_MV048', 'P_LV049', 'Q_LV049', 'P_MV049', 'Q_MV049', 'P_LV050', 'Q_LV050', 'P_MV050', 'Q_MV050', 'P_LV051', 'Q_LV051', 'P_MV051', 'Q_MV051', 'P_LV052', 'Q_LV052', 'P_MV052', 'Q_MV052', 'P_LV053', 'Q_LV053', 'P_MV053', 'Q_MV053', 'P_LV054', 'Q_LV054', 'P_MV054', 'Q_MV054', 'P_LV055', 'Q_LV055', 'P_MV055', 'Q_MV055', 'P_LV056', 'Q_LV056', 'P_MV056', 'Q_MV056', 'P_LV057', 'Q_LV057', 'P_MV057', 'Q_MV057', 'P_LV058', 'Q_LV058', 'P_MV058', 'Q_MV058', 'P_LV059', 'Q_LV059', 'P_MV059', 'Q_MV059', 'P_LV060', 'Q_LV060', 'P_MV060', 'Q_MV060', 'P_LV061', 'Q_LV061', 'P_MV061', 'Q_MV061', 'P_LV062', 'Q_LV062', 'P_MV062', 'Q_MV062', 'P_LV063', 'Q_LV063', 'P_MV063', 'Q_MV063', 'P_LV064', 'Q_LV064', 'P_MV064', 'Q_MV064', 'P_LV065', 'Q_LV065', 'P_MV065', 'Q_MV065', 'P_LV066', 'Q_LV066', 'P_MV066', 'Q_MV066', 'P_LV067', 'Q_LV067', 'P_MV067', 'Q_MV067', 'P_LV068', 'Q_LV068', 'P_MV068', 'Q_MV068', 'P_LV069', 'Q_LV069', 'P_MV069', 'Q_MV069', 'P_LV070', 'Q_LV070', 'P_MV070', 'Q_MV070', 'P_LV071', 'Q_LV071', 'P_MV071', 'Q_MV071', 'P_LV072', 'Q_LV072', 'P_MV072', 'Q_MV072', 'P_LV073', 'Q_LV073', 'P_MV073', 'Q_MV073', 'P_LV074', 'Q_LV074', 'P_MV074', 'Q_MV074', 'P_LV075', 'Q_LV075', 'P_MV075', 'Q_MV075', 'P_LV076', 'Q_LV076', 'P_MV076', 'Q_MV076', 'P_LV077', 'Q_LV077', 'P_MV077', 'Q_MV077', 'P_LV078', 'Q_LV078', 'P_MV078', 'Q_MV078', 'P_LV079', 'Q_LV079', 'P_MV079', 'Q_MV079', 'P_LV080', 'Q_LV080', 'P_MV080', 'Q_MV080', 'P_LV081', 'Q_LV081', 'P_MV081', 'Q_MV081', 'P_LV082', 'Q_LV082', 'P_MV082', 'Q_MV082', 'P_LV083', 'Q_LV083', 'P_MV083', 'Q_MV083', 'P_LV084', 'Q_LV084', 'P_MV084', 'Q_MV084', 'P_LV085', 'Q_LV085', 'P_MV085', 'Q_MV085', 'P_LV086', 'Q_LV086', 'P_MV086', 'Q_MV086', 'P_LV087', 'Q_LV087', 'P_MV087', 'Q_MV087', 'P_LV088', 'Q_LV088', 'P_MV088', 'Q_MV088', 'P_LV089', 'Q_LV089', 'P_MV089', 'Q_MV089', 'P_LV090', 'Q_LV090', 'P_MV090', 'Q_MV090', 'P_LV091', 'Q_LV091', 'P_MV091', 'Q_MV091', 'P_LV092', 'Q_LV092', 'P_MV092', 'Q_MV092', 'P_LV093', 'Q_LV093', 'P_MV093', 'Q_MV093', 'P_LV094', 'Q_LV094', 'P_MV094', 'Q_MV094', 'P_LV095', 'Q_LV095', 'P_MV095', 'Q_MV095', 'P_LV096', 'Q_LV096', 'P_MV096', 'Q_MV096', 'P_LV097', 'Q_LV097', 'P_MV097', 'Q_MV097', 'P_LV098', 'Q_LV098', 'P_MV098', 'Q_MV098', 'P_LV099', 'Q_LV099', 'P_MV099', 'Q_MV099', 'P_LV100', 'Q_LV100', 'P_MV100', 'Q_MV100', 'alpha_GRID', 'v_ref_GRID', 'omega_ref_GRID', 'delta_ref_GRID', 'phi_GRID', 'rocov_GRID', 'irrad_LV001', 'temp_deg_LV001', 'lvrt_ext_LV001', 'p_s_ppc_LV001', 'q_s_ppc_LV001', 'i_sa_ref_LV001', 'i_sr_ref_LV001', 'irrad_LV002', 'temp_deg_LV002', 'lvrt_ext_LV002', 'p_s_ppc_LV002', 'q_s_ppc_LV002', 'i_sa_ref_LV002', 'i_sr_ref_LV002', 'irrad_LV003', 'temp_deg_LV003', 'lvrt_ext_LV003', 'p_s_ppc_LV003', 'q_s_ppc_LV003', 'i_sa_ref_LV003', 'i_sr_ref_LV003', 'irrad_LV004', 'temp_deg_LV004', 'lvrt_ext_LV004', 'p_s_ppc_LV004', 'q_s_ppc_LV004', 'i_sa_ref_LV004', 'i_sr_ref_LV004', 'irrad_LV005', 'temp_deg_LV005', 'lvrt_ext_LV005', 'p_s_ppc_LV005', 'q_s_ppc_LV005', 'i_sa_ref_LV005', 'i_sr_ref_LV005', 'irrad_LV006', 'temp_deg_LV006', 'lvrt_ext_LV006', 'p_s_ppc_LV006', 'q_s_ppc_LV006', 'i_sa_ref_LV006', 'i_sr_ref_LV006', 'irrad_LV007', 'temp_deg_LV007', 'lvrt_ext_LV007', 'p_s_ppc_LV007', 'q_s_ppc_LV007', 'i_sa_ref_LV007', 'i_sr_ref_LV007', 'irrad_LV008', 'temp_deg_LV008', 'lvrt_ext_LV008', 'p_s_ppc_LV008', 'q_s_ppc_LV008', 'i_sa_ref_LV008', 'i_sr_ref_LV008', 'irrad_LV009', 'temp_deg_LV009', 'lvrt_ext_LV009', 'p_s_ppc_LV009', 'q_s_ppc_LV009', 'i_sa_ref_LV009', 'i_sr_ref_LV009', 'irrad_LV010', 'temp_deg_LV010', 'lvrt_ext_LV010', 'p_s_ppc_LV010', 'q_s_ppc_LV010', 'i_sa_ref_LV010', 'i_sr_ref_LV010', 'irrad_LV011', 'temp_deg_LV011', 'lvrt_ext_LV011', 'p_s_ppc_LV011', 'q_s_ppc_LV011', 'i_sa_ref_LV011', 'i_sr_ref_LV011', 'irrad_LV012', 'temp_deg_LV012', 'lvrt_ext_LV012', 'p_s_ppc_LV012', 'q_s_ppc_LV012', 'i_sa_ref_LV012', 'i_sr_ref_LV012', 'irrad_LV013', 'temp_deg_LV013', 'lvrt_ext_LV013', 'p_s_ppc_LV013', 'q_s_ppc_LV013', 'i_sa_ref_LV013', 'i_sr_ref_LV013', 'irrad_LV014', 'temp_deg_LV014', 'lvrt_ext_LV014', 'p_s_ppc_LV014', 'q_s_ppc_LV014', 'i_sa_ref_LV014', 'i_sr_ref_LV014', 'irrad_LV015', 'temp_deg_LV015', 'lvrt_ext_LV015', 'p_s_ppc_LV015', 'q_s_ppc_LV015', 'i_sa_ref_LV015', 'i_sr_ref_LV015', 'irrad_LV016', 'temp_deg_LV016', 'lvrt_ext_LV016', 'p_s_ppc_LV016', 'q_s_ppc_LV016', 'i_sa_ref_LV016', 'i_sr_ref_LV016', 'irrad_LV017', 'temp_deg_LV017', 'lvrt_ext_LV017', 'p_s_ppc_LV017', 'q_s_ppc_LV017', 'i_sa_ref_LV017', 'i_sr_ref_LV017', 'irrad_LV018', 'temp_deg_LV018', 'lvrt_ext_LV018', 'p_s_ppc_LV018', 'q_s_ppc_LV018', 'i_sa_ref_LV018', 'i_sr_ref_LV018', 'irrad_LV019', 'temp_deg_LV019', 'lvrt_ext_LV019', 'p_s_ppc_LV019', 'q_s_ppc_LV019', 'i_sa_ref_LV019', 'i_sr_ref_LV019', 'irrad_LV020', 'temp_deg_LV020', 'lvrt_ext_LV020', 'p_s_ppc_LV020', 'q_s_ppc_LV020', 'i_sa_ref_LV020', 'i_sr_ref_LV020', 'irrad_LV021', 'temp_deg_LV021', 'lvrt_ext_LV021', 'p_s_ppc_LV021', 'q_s_ppc_LV021', 'i_sa_ref_LV021', 'i_sr_ref_LV021', 'irrad_LV022', 'temp_deg_LV022', 'lvrt_ext_LV022', 'p_s_ppc_LV022', 'q_s_ppc_LV022', 'i_sa_ref_LV022', 'i_sr_ref_LV022', 'irrad_LV023', 'temp_deg_LV023', 'lvrt_ext_LV023', 'p_s_ppc_LV023', 'q_s_ppc_LV023', 'i_sa_ref_LV023', 'i_sr_ref_LV023', 'irrad_LV024', 'temp_deg_LV024', 'lvrt_ext_LV024', 'p_s_ppc_LV024', 'q_s_ppc_LV024', 'i_sa_ref_LV024', 'i_sr_ref_LV024', 'irrad_LV025', 'temp_deg_LV025', 'lvrt_ext_LV025', 'p_s_ppc_LV025', 'q_s_ppc_LV025', 'i_sa_ref_LV025', 'i_sr_ref_LV025', 'irrad_LV026', 'temp_deg_LV026', 'lvrt_ext_LV026', 'p_s_ppc_LV026', 'q_s_ppc_LV026', 'i_sa_ref_LV026', 'i_sr_ref_LV026', 'irrad_LV027', 'temp_deg_LV027', 'lvrt_ext_LV027', 'p_s_ppc_LV027', 'q_s_ppc_LV027', 'i_sa_ref_LV027', 'i_sr_ref_LV027', 'irrad_LV028', 'temp_deg_LV028', 'lvrt_ext_LV028', 'p_s_ppc_LV028', 'q_s_ppc_LV028', 'i_sa_ref_LV028', 'i_sr_ref_LV028', 'irrad_LV029', 'temp_deg_LV029', 'lvrt_ext_LV029', 'p_s_ppc_LV029', 'q_s_ppc_LV029', 'i_sa_ref_LV029', 'i_sr_ref_LV029', 'irrad_LV030', 'temp_deg_LV030', 'lvrt_ext_LV030', 'p_s_ppc_LV030', 'q_s_ppc_LV030', 'i_sa_ref_LV030', 'i_sr_ref_LV030', 'irrad_LV031', 'temp_deg_LV031', 'lvrt_ext_LV031', 'p_s_ppc_LV031', 'q_s_ppc_LV031', 'i_sa_ref_LV031', 'i_sr_ref_LV031', 'irrad_LV032', 'temp_deg_LV032', 'lvrt_ext_LV032', 'p_s_ppc_LV032', 'q_s_ppc_LV032', 'i_sa_ref_LV032', 'i_sr_ref_LV032', 'irrad_LV033', 'temp_deg_LV033', 'lvrt_ext_LV033', 'p_s_ppc_LV033', 'q_s_ppc_LV033', 'i_sa_ref_LV033', 'i_sr_ref_LV033', 'irrad_LV034', 'temp_deg_LV034', 'lvrt_ext_LV034', 'p_s_ppc_LV034', 'q_s_ppc_LV034', 'i_sa_ref_LV034', 'i_sr_ref_LV034', 'irrad_LV035', 'temp_deg_LV035', 'lvrt_ext_LV035', 'p_s_ppc_LV035', 'q_s_ppc_LV035', 'i_sa_ref_LV035', 'i_sr_ref_LV035', 'irrad_LV036', 'temp_deg_LV036', 'lvrt_ext_LV036', 'p_s_ppc_LV036', 'q_s_ppc_LV036', 'i_sa_ref_LV036', 'i_sr_ref_LV036', 'irrad_LV037', 'temp_deg_LV037', 'lvrt_ext_LV037', 'p_s_ppc_LV037', 'q_s_ppc_LV037', 'i_sa_ref_LV037', 'i_sr_ref_LV037', 'irrad_LV038', 'temp_deg_LV038', 'lvrt_ext_LV038', 'p_s_ppc_LV038', 'q_s_ppc_LV038', 'i_sa_ref_LV038', 'i_sr_ref_LV038', 'irrad_LV039', 'temp_deg_LV039', 'lvrt_ext_LV039', 'p_s_ppc_LV039', 'q_s_ppc_LV039', 'i_sa_ref_LV039', 'i_sr_ref_LV039', 'irrad_LV040', 'temp_deg_LV040', 'lvrt_ext_LV040', 'p_s_ppc_LV040', 'q_s_ppc_LV040', 'i_sa_ref_LV040', 'i_sr_ref_LV040', 'irrad_LV041', 'temp_deg_LV041', 'lvrt_ext_LV041', 'p_s_ppc_LV041', 'q_s_ppc_LV041', 'i_sa_ref_LV041', 'i_sr_ref_LV041', 'irrad_LV042', 'temp_deg_LV042', 'lvrt_ext_LV042', 'p_s_ppc_LV042', 'q_s_ppc_LV042', 'i_sa_ref_LV042', 'i_sr_ref_LV042', 'irrad_LV043', 'temp_deg_LV043', 'lvrt_ext_LV043', 'p_s_ppc_LV043', 'q_s_ppc_LV043', 'i_sa_ref_LV043', 'i_sr_ref_LV043', 'irrad_LV044', 'temp_deg_LV044', 'lvrt_ext_LV044', 'p_s_ppc_LV044', 'q_s_ppc_LV044', 'i_sa_ref_LV044', 'i_sr_ref_LV044', 'irrad_LV045', 'temp_deg_LV045', 'lvrt_ext_LV045', 'p_s_ppc_LV045', 'q_s_ppc_LV045', 'i_sa_ref_LV045', 'i_sr_ref_LV045', 'irrad_LV046', 'temp_deg_LV046', 'lvrt_ext_LV046', 'p_s_ppc_LV046', 'q_s_ppc_LV046', 'i_sa_ref_LV046', 'i_sr_ref_LV046', 'irrad_LV047', 'temp_deg_LV047', 'lvrt_ext_LV047', 'p_s_ppc_LV047', 'q_s_ppc_LV047', 'i_sa_ref_LV047', 'i_sr_ref_LV047', 'irrad_LV048', 'temp_deg_LV048', 'lvrt_ext_LV048', 'p_s_ppc_LV048', 'q_s_ppc_LV048', 'i_sa_ref_LV048', 'i_sr_ref_LV048', 'irrad_LV049', 'temp_deg_LV049', 'lvrt_ext_LV049', 'p_s_ppc_LV049', 'q_s_ppc_LV049', 'i_sa_ref_LV049', 'i_sr_ref_LV049', 'irrad_LV050', 'temp_deg_LV050', 'lvrt_ext_LV050', 'p_s_ppc_LV050', 'q_s_ppc_LV050', 'i_sa_ref_LV050', 'i_sr_ref_LV050', 'irrad_LV051', 'temp_deg_LV051', 'lvrt_ext_LV051', 'p_s_ppc_LV051', 'q_s_ppc_LV051', 'i_sa_ref_LV051', 'i_sr_ref_LV051', 'irrad_LV052', 'temp_deg_LV052', 'lvrt_ext_LV052', 'p_s_ppc_LV052', 'q_s_ppc_LV052', 'i_sa_ref_LV052', 'i_sr_ref_LV052', 'irrad_LV053', 'temp_deg_LV053', 'lvrt_ext_LV053', 'p_s_ppc_LV053', 'q_s_ppc_LV053', 'i_sa_ref_LV053', 'i_sr_ref_LV053', 'irrad_LV054', 'temp_deg_LV054', 'lvrt_ext_LV054', 'p_s_ppc_LV054', 'q_s_ppc_LV054', 'i_sa_ref_LV054', 'i_sr_ref_LV054', 'irrad_LV055', 'temp_deg_LV055', 'lvrt_ext_LV055', 'p_s_ppc_LV055', 'q_s_ppc_LV055', 'i_sa_ref_LV055', 'i_sr_ref_LV055', 'irrad_LV056', 'temp_deg_LV056', 'lvrt_ext_LV056', 'p_s_ppc_LV056', 'q_s_ppc_LV056', 'i_sa_ref_LV056', 'i_sr_ref_LV056', 'irrad_LV057', 'temp_deg_LV057', 'lvrt_ext_LV057', 'p_s_ppc_LV057', 'q_s_ppc_LV057', 'i_sa_ref_LV057', 'i_sr_ref_LV057', 'irrad_LV058', 'temp_deg_LV058', 'lvrt_ext_LV058', 'p_s_ppc_LV058', 'q_s_ppc_LV058', 'i_sa_ref_LV058', 'i_sr_ref_LV058', 'irrad_LV059', 'temp_deg_LV059', 'lvrt_ext_LV059', 'p_s_ppc_LV059', 'q_s_ppc_LV059', 'i_sa_ref_LV059', 'i_sr_ref_LV059', 'irrad_LV060', 'temp_deg_LV060', 'lvrt_ext_LV060', 'p_s_ppc_LV060', 'q_s_ppc_LV060', 'i_sa_ref_LV060', 'i_sr_ref_LV060', 'irrad_LV061', 'temp_deg_LV061', 'lvrt_ext_LV061', 'p_s_ppc_LV061', 'q_s_ppc_LV061', 'i_sa_ref_LV061', 'i_sr_ref_LV061', 'irrad_LV062', 'temp_deg_LV062', 'lvrt_ext_LV062', 'p_s_ppc_LV062', 'q_s_ppc_LV062', 'i_sa_ref_LV062', 'i_sr_ref_LV062', 'irrad_LV063', 'temp_deg_LV063', 'lvrt_ext_LV063', 'p_s_ppc_LV063', 'q_s_ppc_LV063', 'i_sa_ref_LV063', 'i_sr_ref_LV063', 'irrad_LV064', 'temp_deg_LV064', 'lvrt_ext_LV064', 'p_s_ppc_LV064', 'q_s_ppc_LV064', 'i_sa_ref_LV064', 'i_sr_ref_LV064', 'irrad_LV065', 'temp_deg_LV065', 'lvrt_ext_LV065', 'p_s_ppc_LV065', 'q_s_ppc_LV065', 'i_sa_ref_LV065', 'i_sr_ref_LV065', 'irrad_LV066', 'temp_deg_LV066', 'lvrt_ext_LV066', 'p_s_ppc_LV066', 'q_s_ppc_LV066', 'i_sa_ref_LV066', 'i_sr_ref_LV066', 'irrad_LV067', 'temp_deg_LV067', 'lvrt_ext_LV067', 'p_s_ppc_LV067', 'q_s_ppc_LV067', 'i_sa_ref_LV067', 'i_sr_ref_LV067', 'irrad_LV068', 'temp_deg_LV068', 'lvrt_ext_LV068', 'p_s_ppc_LV068', 'q_s_ppc_LV068', 'i_sa_ref_LV068', 'i_sr_ref_LV068', 'irrad_LV069', 'temp_deg_LV069', 'lvrt_ext_LV069', 'p_s_ppc_LV069', 'q_s_ppc_LV069', 'i_sa_ref_LV069', 'i_sr_ref_LV069', 'irrad_LV070', 'temp_deg_LV070', 'lvrt_ext_LV070', 'p_s_ppc_LV070', 'q_s_ppc_LV070', 'i_sa_ref_LV070', 'i_sr_ref_LV070', 'irrad_LV071', 'temp_deg_LV071', 'lvrt_ext_LV071', 'p_s_ppc_LV071', 'q_s_ppc_LV071', 'i_sa_ref_LV071', 'i_sr_ref_LV071', 'irrad_LV072', 'temp_deg_LV072', 'lvrt_ext_LV072', 'p_s_ppc_LV072', 'q_s_ppc_LV072', 'i_sa_ref_LV072', 'i_sr_ref_LV072', 'irrad_LV073', 'temp_deg_LV073', 'lvrt_ext_LV073', 'p_s_ppc_LV073', 'q_s_ppc_LV073', 'i_sa_ref_LV073', 'i_sr_ref_LV073', 'irrad_LV074', 'temp_deg_LV074', 'lvrt_ext_LV074', 'p_s_ppc_LV074', 'q_s_ppc_LV074', 'i_sa_ref_LV074', 'i_sr_ref_LV074', 'irrad_LV075', 'temp_deg_LV075', 'lvrt_ext_LV075', 'p_s_ppc_LV075', 'q_s_ppc_LV075', 'i_sa_ref_LV075', 'i_sr_ref_LV075', 'irrad_LV076', 'temp_deg_LV076', 'lvrt_ext_LV076', 'p_s_ppc_LV076', 'q_s_ppc_LV076', 'i_sa_ref_LV076', 'i_sr_ref_LV076', 'irrad_LV077', 'temp_deg_LV077', 'lvrt_ext_LV077', 'p_s_ppc_LV077', 'q_s_ppc_LV077', 'i_sa_ref_LV077', 'i_sr_ref_LV077', 'irrad_LV078', 'temp_deg_LV078', 'lvrt_ext_LV078', 'p_s_ppc_LV078', 'q_s_ppc_LV078', 'i_sa_ref_LV078', 'i_sr_ref_LV078', 'irrad_LV079', 'temp_deg_LV079', 'lvrt_ext_LV079', 'p_s_ppc_LV079', 'q_s_ppc_LV079', 'i_sa_ref_LV079', 'i_sr_ref_LV079', 'irrad_LV080', 'temp_deg_LV080', 'lvrt_ext_LV080', 'p_s_ppc_LV080', 'q_s_ppc_LV080', 'i_sa_ref_LV080', 'i_sr_ref_LV080', 'irrad_LV081', 'temp_deg_LV081', 'lvrt_ext_LV081', 'p_s_ppc_LV081', 'q_s_ppc_LV081', 'i_sa_ref_LV081', 'i_sr_ref_LV081', 'irrad_LV082', 'temp_deg_LV082', 'lvrt_ext_LV082', 'p_s_ppc_LV082', 'q_s_ppc_LV082', 'i_sa_ref_LV082', 'i_sr_ref_LV082', 'irrad_LV083', 'temp_deg_LV083', 'lvrt_ext_LV083', 'p_s_ppc_LV083', 'q_s_ppc_LV083', 'i_sa_ref_LV083', 'i_sr_ref_LV083', 'irrad_LV084', 'temp_deg_LV084', 'lvrt_ext_LV084', 'p_s_ppc_LV084', 'q_s_ppc_LV084', 'i_sa_ref_LV084', 'i_sr_ref_LV084', 'irrad_LV085', 'temp_deg_LV085', 'lvrt_ext_LV085', 'p_s_ppc_LV085', 'q_s_ppc_LV085', 'i_sa_ref_LV085', 'i_sr_ref_LV085', 'irrad_LV086', 'temp_deg_LV086', 'lvrt_ext_LV086', 'p_s_ppc_LV086', 'q_s_ppc_LV086', 'i_sa_ref_LV086', 'i_sr_ref_LV086', 'irrad_LV087', 'temp_deg_LV087', 'lvrt_ext_LV087', 'p_s_ppc_LV087', 'q_s_ppc_LV087', 'i_sa_ref_LV087', 'i_sr_ref_LV087', 'irrad_LV088', 'temp_deg_LV088', 'lvrt_ext_LV088', 'p_s_ppc_LV088', 'q_s_ppc_LV088', 'i_sa_ref_LV088', 'i_sr_ref_LV088', 'irrad_LV089', 'temp_deg_LV089', 'lvrt_ext_LV089', 'p_s_ppc_LV089', 'q_s_ppc_LV089', 'i_sa_ref_LV089', 'i_sr_ref_LV089', 'irrad_LV090', 'temp_deg_LV090', 'lvrt_ext_LV090', 'p_s_ppc_LV090', 'q_s_ppc_LV090', 'i_sa_ref_LV090', 'i_sr_ref_LV090', 'irrad_LV091', 'temp_deg_LV091', 'lvrt_ext_LV091', 'p_s_ppc_LV091', 'q_s_ppc_LV091', 'i_sa_ref_LV091', 'i_sr_ref_LV091', 'irrad_LV092', 'temp_deg_LV092', 'lvrt_ext_LV092', 'p_s_ppc_LV092', 'q_s_ppc_LV092', 'i_sa_ref_LV092', 'i_sr_ref_LV092', 'irrad_LV093', 'temp_deg_LV093', 'lvrt_ext_LV093', 'p_s_ppc_LV093', 'q_s_ppc_LV093', 'i_sa_ref_LV093', 'i_sr_ref_LV093', 'irrad_LV094', 'temp_deg_LV094', 'lvrt_ext_LV094', 'p_s_ppc_LV094', 'q_s_ppc_LV094', 'i_sa_ref_LV094', 'i_sr_ref_LV094', 'irrad_LV095', 'temp_deg_LV095', 'lvrt_ext_LV095', 'p_s_ppc_LV095', 'q_s_ppc_LV095', 'i_sa_ref_LV095', 'i_sr_ref_LV095', 'irrad_LV096', 'temp_deg_LV096', 'lvrt_ext_LV096', 'p_s_ppc_LV096', 'q_s_ppc_LV096', 'i_sa_ref_LV096', 'i_sr_ref_LV096', 'irrad_LV097', 'temp_deg_LV097', 'lvrt_ext_LV097', 'p_s_ppc_LV097', 'q_s_ppc_LV097', 'i_sa_ref_LV097', 'i_sr_ref_LV097', 'irrad_LV098', 'temp_deg_LV098', 'lvrt_ext_LV098', 'p_s_ppc_LV098', 'q_s_ppc_LV098', 'i_sa_ref_LV098', 'i_sr_ref_LV098', 'irrad_LV099', 'temp_deg_LV099', 'lvrt_ext_LV099', 'p_s_ppc_LV099', 'q_s_ppc_LV099', 'i_sa_ref_LV099', 'i_sr_ref_LV099', 'irrad_LV100', 'temp_deg_LV100', 'lvrt_ext_LV100', 'p_s_ppc_LV100', 'q_s_ppc_LV100', 'i_sa_ref_LV100', 'i_sr_ref_LV100'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0] 
        self.inputs_run_list = ['P_POI_MV', 'Q_POI_MV', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'P_LV001', 'Q_LV001', 'P_MV001', 'Q_MV001', 'P_LV002', 'Q_LV002', 'P_MV002', 'Q_MV002', 'P_LV003', 'Q_LV003', 'P_MV003', 'Q_MV003', 'P_LV004', 'Q_LV004', 'P_MV004', 'Q_MV004', 'P_LV005', 'Q_LV005', 'P_MV005', 'Q_MV005', 'P_LV006', 'Q_LV006', 'P_MV006', 'Q_MV006', 'P_LV007', 'Q_LV007', 'P_MV007', 'Q_MV007', 'P_LV008', 'Q_LV008', 'P_MV008', 'Q_MV008', 'P_LV009', 'Q_LV009', 'P_MV009', 'Q_MV009', 'P_LV010', 'Q_LV010', 'P_MV010', 'Q_MV010', 'P_LV011', 'Q_LV011', 'P_MV011', 'Q_MV011', 'P_LV012', 'Q_LV012', 'P_MV012', 'Q_MV012', 'P_LV013', 'Q_LV013', 'P_MV013', 'Q_MV013', 'P_LV014', 'Q_LV014', 'P_MV014', 'Q_MV014', 'P_LV015', 'Q_LV015', 'P_MV015', 'Q_MV015', 'P_LV016', 'Q_LV016', 'P_MV016', 'Q_MV016', 'P_LV017', 'Q_LV017', 'P_MV017', 'Q_MV017', 'P_LV018', 'Q_LV018', 'P_MV018', 'Q_MV018', 'P_LV019', 'Q_LV019', 'P_MV019', 'Q_MV019', 'P_LV020', 'Q_LV020', 'P_MV020', 'Q_MV020', 'P_LV021', 'Q_LV021', 'P_MV021', 'Q_MV021', 'P_LV022', 'Q_LV022', 'P_MV022', 'Q_MV022', 'P_LV023', 'Q_LV023', 'P_MV023', 'Q_MV023', 'P_LV024', 'Q_LV024', 'P_MV024', 'Q_MV024', 'P_LV025', 'Q_LV025', 'P_MV025', 'Q_MV025', 'P_LV026', 'Q_LV026', 'P_MV026', 'Q_MV026', 'P_LV027', 'Q_LV027', 'P_MV027', 'Q_MV027', 'P_LV028', 'Q_LV028', 'P_MV028', 'Q_MV028', 'P_LV029', 'Q_LV029', 'P_MV029', 'Q_MV029', 'P_LV030', 'Q_LV030', 'P_MV030', 'Q_MV030', 'P_LV031', 'Q_LV031', 'P_MV031', 'Q_MV031', 'P_LV032', 'Q_LV032', 'P_MV032', 'Q_MV032', 'P_LV033', 'Q_LV033', 'P_MV033', 'Q_MV033', 'P_LV034', 'Q_LV034', 'P_MV034', 'Q_MV034', 'P_LV035', 'Q_LV035', 'P_MV035', 'Q_MV035', 'P_LV036', 'Q_LV036', 'P_MV036', 'Q_MV036', 'P_LV037', 'Q_LV037', 'P_MV037', 'Q_MV037', 'P_LV038', 'Q_LV038', 'P_MV038', 'Q_MV038', 'P_LV039', 'Q_LV039', 'P_MV039', 'Q_MV039', 'P_LV040', 'Q_LV040', 'P_MV040', 'Q_MV040', 'P_LV041', 'Q_LV041', 'P_MV041', 'Q_MV041', 'P_LV042', 'Q_LV042', 'P_MV042', 'Q_MV042', 'P_LV043', 'Q_LV043', 'P_MV043', 'Q_MV043', 'P_LV044', 'Q_LV044', 'P_MV044', 'Q_MV044', 'P_LV045', 'Q_LV045', 'P_MV045', 'Q_MV045', 'P_LV046', 'Q_LV046', 'P_MV046', 'Q_MV046', 'P_LV047', 'Q_LV047', 'P_MV047', 'Q_MV047', 'P_LV048', 'Q_LV048', 'P_MV048', 'Q_MV048', 'P_LV049', 'Q_LV049', 'P_MV049', 'Q_MV049', 'P_LV050', 'Q_LV050', 'P_MV050', 'Q_MV050', 'P_LV051', 'Q_LV051', 'P_MV051', 'Q_MV051', 'P_LV052', 'Q_LV052', 'P_MV052', 'Q_MV052', 'P_LV053', 'Q_LV053', 'P_MV053', 'Q_MV053', 'P_LV054', 'Q_LV054', 'P_MV054', 'Q_MV054', 'P_LV055', 'Q_LV055', 'P_MV055', 'Q_MV055', 'P_LV056', 'Q_LV056', 'P_MV056', 'Q_MV056', 'P_LV057', 'Q_LV057', 'P_MV057', 'Q_MV057', 'P_LV058', 'Q_LV058', 'P_MV058', 'Q_MV058', 'P_LV059', 'Q_LV059', 'P_MV059', 'Q_MV059', 'P_LV060', 'Q_LV060', 'P_MV060', 'Q_MV060', 'P_LV061', 'Q_LV061', 'P_MV061', 'Q_MV061', 'P_LV062', 'Q_LV062', 'P_MV062', 'Q_MV062', 'P_LV063', 'Q_LV063', 'P_MV063', 'Q_MV063', 'P_LV064', 'Q_LV064', 'P_MV064', 'Q_MV064', 'P_LV065', 'Q_LV065', 'P_MV065', 'Q_MV065', 'P_LV066', 'Q_LV066', 'P_MV066', 'Q_MV066', 'P_LV067', 'Q_LV067', 'P_MV067', 'Q_MV067', 'P_LV068', 'Q_LV068', 'P_MV068', 'Q_MV068', 'P_LV069', 'Q_LV069', 'P_MV069', 'Q_MV069', 'P_LV070', 'Q_LV070', 'P_MV070', 'Q_MV070', 'P_LV071', 'Q_LV071', 'P_MV071', 'Q_MV071', 'P_LV072', 'Q_LV072', 'P_MV072', 'Q_MV072', 'P_LV073', 'Q_LV073', 'P_MV073', 'Q_MV073', 'P_LV074', 'Q_LV074', 'P_MV074', 'Q_MV074', 'P_LV075', 'Q_LV075', 'P_MV075', 'Q_MV075', 'P_LV076', 'Q_LV076', 'P_MV076', 'Q_MV076', 'P_LV077', 'Q_LV077', 'P_MV077', 'Q_MV077', 'P_LV078', 'Q_LV078', 'P_MV078', 'Q_MV078', 'P_LV079', 'Q_LV079', 'P_MV079', 'Q_MV079', 'P_LV080', 'Q_LV080', 'P_MV080', 'Q_MV080', 'P_LV081', 'Q_LV081', 'P_MV081', 'Q_MV081', 'P_LV082', 'Q_LV082', 'P_MV082', 'Q_MV082', 'P_LV083', 'Q_LV083', 'P_MV083', 'Q_MV083', 'P_LV084', 'Q_LV084', 'P_MV084', 'Q_MV084', 'P_LV085', 'Q_LV085', 'P_MV085', 'Q_MV085', 'P_LV086', 'Q_LV086', 'P_MV086', 'Q_MV086', 'P_LV087', 'Q_LV087', 'P_MV087', 'Q_MV087', 'P_LV088', 'Q_LV088', 'P_MV088', 'Q_MV088', 'P_LV089', 'Q_LV089', 'P_MV089', 'Q_MV089', 'P_LV090', 'Q_LV090', 'P_MV090', 'Q_MV090', 'P_LV091', 'Q_LV091', 'P_MV091', 'Q_MV091', 'P_LV092', 'Q_LV092', 'P_MV092', 'Q_MV092', 'P_LV093', 'Q_LV093', 'P_MV093', 'Q_MV093', 'P_LV094', 'Q_LV094', 'P_MV094', 'Q_MV094', 'P_LV095', 'Q_LV095', 'P_MV095', 'Q_MV095', 'P_LV096', 'Q_LV096', 'P_MV096', 'Q_MV096', 'P_LV097', 'Q_LV097', 'P_MV097', 'Q_MV097', 'P_LV098', 'Q_LV098', 'P_MV098', 'Q_MV098', 'P_LV099', 'Q_LV099', 'P_MV099', 'Q_MV099', 'P_LV100', 'Q_LV100', 'P_MV100', 'Q_MV100', 'alpha_GRID', 'v_ref_GRID', 'omega_ref_GRID', 'delta_ref_GRID', 'phi_GRID', 'rocov_GRID', 'irrad_LV001', 'temp_deg_LV001', 'lvrt_ext_LV001', 'p_s_ppc_LV001', 'q_s_ppc_LV001', 'i_sa_ref_LV001', 'i_sr_ref_LV001', 'irrad_LV002', 'temp_deg_LV002', 'lvrt_ext_LV002', 'p_s_ppc_LV002', 'q_s_ppc_LV002', 'i_sa_ref_LV002', 'i_sr_ref_LV002', 'irrad_LV003', 'temp_deg_LV003', 'lvrt_ext_LV003', 'p_s_ppc_LV003', 'q_s_ppc_LV003', 'i_sa_ref_LV003', 'i_sr_ref_LV003', 'irrad_LV004', 'temp_deg_LV004', 'lvrt_ext_LV004', 'p_s_ppc_LV004', 'q_s_ppc_LV004', 'i_sa_ref_LV004', 'i_sr_ref_LV004', 'irrad_LV005', 'temp_deg_LV005', 'lvrt_ext_LV005', 'p_s_ppc_LV005', 'q_s_ppc_LV005', 'i_sa_ref_LV005', 'i_sr_ref_LV005', 'irrad_LV006', 'temp_deg_LV006', 'lvrt_ext_LV006', 'p_s_ppc_LV006', 'q_s_ppc_LV006', 'i_sa_ref_LV006', 'i_sr_ref_LV006', 'irrad_LV007', 'temp_deg_LV007', 'lvrt_ext_LV007', 'p_s_ppc_LV007', 'q_s_ppc_LV007', 'i_sa_ref_LV007', 'i_sr_ref_LV007', 'irrad_LV008', 'temp_deg_LV008', 'lvrt_ext_LV008', 'p_s_ppc_LV008', 'q_s_ppc_LV008', 'i_sa_ref_LV008', 'i_sr_ref_LV008', 'irrad_LV009', 'temp_deg_LV009', 'lvrt_ext_LV009', 'p_s_ppc_LV009', 'q_s_ppc_LV009', 'i_sa_ref_LV009', 'i_sr_ref_LV009', 'irrad_LV010', 'temp_deg_LV010', 'lvrt_ext_LV010', 'p_s_ppc_LV010', 'q_s_ppc_LV010', 'i_sa_ref_LV010', 'i_sr_ref_LV010', 'irrad_LV011', 'temp_deg_LV011', 'lvrt_ext_LV011', 'p_s_ppc_LV011', 'q_s_ppc_LV011', 'i_sa_ref_LV011', 'i_sr_ref_LV011', 'irrad_LV012', 'temp_deg_LV012', 'lvrt_ext_LV012', 'p_s_ppc_LV012', 'q_s_ppc_LV012', 'i_sa_ref_LV012', 'i_sr_ref_LV012', 'irrad_LV013', 'temp_deg_LV013', 'lvrt_ext_LV013', 'p_s_ppc_LV013', 'q_s_ppc_LV013', 'i_sa_ref_LV013', 'i_sr_ref_LV013', 'irrad_LV014', 'temp_deg_LV014', 'lvrt_ext_LV014', 'p_s_ppc_LV014', 'q_s_ppc_LV014', 'i_sa_ref_LV014', 'i_sr_ref_LV014', 'irrad_LV015', 'temp_deg_LV015', 'lvrt_ext_LV015', 'p_s_ppc_LV015', 'q_s_ppc_LV015', 'i_sa_ref_LV015', 'i_sr_ref_LV015', 'irrad_LV016', 'temp_deg_LV016', 'lvrt_ext_LV016', 'p_s_ppc_LV016', 'q_s_ppc_LV016', 'i_sa_ref_LV016', 'i_sr_ref_LV016', 'irrad_LV017', 'temp_deg_LV017', 'lvrt_ext_LV017', 'p_s_ppc_LV017', 'q_s_ppc_LV017', 'i_sa_ref_LV017', 'i_sr_ref_LV017', 'irrad_LV018', 'temp_deg_LV018', 'lvrt_ext_LV018', 'p_s_ppc_LV018', 'q_s_ppc_LV018', 'i_sa_ref_LV018', 'i_sr_ref_LV018', 'irrad_LV019', 'temp_deg_LV019', 'lvrt_ext_LV019', 'p_s_ppc_LV019', 'q_s_ppc_LV019', 'i_sa_ref_LV019', 'i_sr_ref_LV019', 'irrad_LV020', 'temp_deg_LV020', 'lvrt_ext_LV020', 'p_s_ppc_LV020', 'q_s_ppc_LV020', 'i_sa_ref_LV020', 'i_sr_ref_LV020', 'irrad_LV021', 'temp_deg_LV021', 'lvrt_ext_LV021', 'p_s_ppc_LV021', 'q_s_ppc_LV021', 'i_sa_ref_LV021', 'i_sr_ref_LV021', 'irrad_LV022', 'temp_deg_LV022', 'lvrt_ext_LV022', 'p_s_ppc_LV022', 'q_s_ppc_LV022', 'i_sa_ref_LV022', 'i_sr_ref_LV022', 'irrad_LV023', 'temp_deg_LV023', 'lvrt_ext_LV023', 'p_s_ppc_LV023', 'q_s_ppc_LV023', 'i_sa_ref_LV023', 'i_sr_ref_LV023', 'irrad_LV024', 'temp_deg_LV024', 'lvrt_ext_LV024', 'p_s_ppc_LV024', 'q_s_ppc_LV024', 'i_sa_ref_LV024', 'i_sr_ref_LV024', 'irrad_LV025', 'temp_deg_LV025', 'lvrt_ext_LV025', 'p_s_ppc_LV025', 'q_s_ppc_LV025', 'i_sa_ref_LV025', 'i_sr_ref_LV025', 'irrad_LV026', 'temp_deg_LV026', 'lvrt_ext_LV026', 'p_s_ppc_LV026', 'q_s_ppc_LV026', 'i_sa_ref_LV026', 'i_sr_ref_LV026', 'irrad_LV027', 'temp_deg_LV027', 'lvrt_ext_LV027', 'p_s_ppc_LV027', 'q_s_ppc_LV027', 'i_sa_ref_LV027', 'i_sr_ref_LV027', 'irrad_LV028', 'temp_deg_LV028', 'lvrt_ext_LV028', 'p_s_ppc_LV028', 'q_s_ppc_LV028', 'i_sa_ref_LV028', 'i_sr_ref_LV028', 'irrad_LV029', 'temp_deg_LV029', 'lvrt_ext_LV029', 'p_s_ppc_LV029', 'q_s_ppc_LV029', 'i_sa_ref_LV029', 'i_sr_ref_LV029', 'irrad_LV030', 'temp_deg_LV030', 'lvrt_ext_LV030', 'p_s_ppc_LV030', 'q_s_ppc_LV030', 'i_sa_ref_LV030', 'i_sr_ref_LV030', 'irrad_LV031', 'temp_deg_LV031', 'lvrt_ext_LV031', 'p_s_ppc_LV031', 'q_s_ppc_LV031', 'i_sa_ref_LV031', 'i_sr_ref_LV031', 'irrad_LV032', 'temp_deg_LV032', 'lvrt_ext_LV032', 'p_s_ppc_LV032', 'q_s_ppc_LV032', 'i_sa_ref_LV032', 'i_sr_ref_LV032', 'irrad_LV033', 'temp_deg_LV033', 'lvrt_ext_LV033', 'p_s_ppc_LV033', 'q_s_ppc_LV033', 'i_sa_ref_LV033', 'i_sr_ref_LV033', 'irrad_LV034', 'temp_deg_LV034', 'lvrt_ext_LV034', 'p_s_ppc_LV034', 'q_s_ppc_LV034', 'i_sa_ref_LV034', 'i_sr_ref_LV034', 'irrad_LV035', 'temp_deg_LV035', 'lvrt_ext_LV035', 'p_s_ppc_LV035', 'q_s_ppc_LV035', 'i_sa_ref_LV035', 'i_sr_ref_LV035', 'irrad_LV036', 'temp_deg_LV036', 'lvrt_ext_LV036', 'p_s_ppc_LV036', 'q_s_ppc_LV036', 'i_sa_ref_LV036', 'i_sr_ref_LV036', 'irrad_LV037', 'temp_deg_LV037', 'lvrt_ext_LV037', 'p_s_ppc_LV037', 'q_s_ppc_LV037', 'i_sa_ref_LV037', 'i_sr_ref_LV037', 'irrad_LV038', 'temp_deg_LV038', 'lvrt_ext_LV038', 'p_s_ppc_LV038', 'q_s_ppc_LV038', 'i_sa_ref_LV038', 'i_sr_ref_LV038', 'irrad_LV039', 'temp_deg_LV039', 'lvrt_ext_LV039', 'p_s_ppc_LV039', 'q_s_ppc_LV039', 'i_sa_ref_LV039', 'i_sr_ref_LV039', 'irrad_LV040', 'temp_deg_LV040', 'lvrt_ext_LV040', 'p_s_ppc_LV040', 'q_s_ppc_LV040', 'i_sa_ref_LV040', 'i_sr_ref_LV040', 'irrad_LV041', 'temp_deg_LV041', 'lvrt_ext_LV041', 'p_s_ppc_LV041', 'q_s_ppc_LV041', 'i_sa_ref_LV041', 'i_sr_ref_LV041', 'irrad_LV042', 'temp_deg_LV042', 'lvrt_ext_LV042', 'p_s_ppc_LV042', 'q_s_ppc_LV042', 'i_sa_ref_LV042', 'i_sr_ref_LV042', 'irrad_LV043', 'temp_deg_LV043', 'lvrt_ext_LV043', 'p_s_ppc_LV043', 'q_s_ppc_LV043', 'i_sa_ref_LV043', 'i_sr_ref_LV043', 'irrad_LV044', 'temp_deg_LV044', 'lvrt_ext_LV044', 'p_s_ppc_LV044', 'q_s_ppc_LV044', 'i_sa_ref_LV044', 'i_sr_ref_LV044', 'irrad_LV045', 'temp_deg_LV045', 'lvrt_ext_LV045', 'p_s_ppc_LV045', 'q_s_ppc_LV045', 'i_sa_ref_LV045', 'i_sr_ref_LV045', 'irrad_LV046', 'temp_deg_LV046', 'lvrt_ext_LV046', 'p_s_ppc_LV046', 'q_s_ppc_LV046', 'i_sa_ref_LV046', 'i_sr_ref_LV046', 'irrad_LV047', 'temp_deg_LV047', 'lvrt_ext_LV047', 'p_s_ppc_LV047', 'q_s_ppc_LV047', 'i_sa_ref_LV047', 'i_sr_ref_LV047', 'irrad_LV048', 'temp_deg_LV048', 'lvrt_ext_LV048', 'p_s_ppc_LV048', 'q_s_ppc_LV048', 'i_sa_ref_LV048', 'i_sr_ref_LV048', 'irrad_LV049', 'temp_deg_LV049', 'lvrt_ext_LV049', 'p_s_ppc_LV049', 'q_s_ppc_LV049', 'i_sa_ref_LV049', 'i_sr_ref_LV049', 'irrad_LV050', 'temp_deg_LV050', 'lvrt_ext_LV050', 'p_s_ppc_LV050', 'q_s_ppc_LV050', 'i_sa_ref_LV050', 'i_sr_ref_LV050', 'irrad_LV051', 'temp_deg_LV051', 'lvrt_ext_LV051', 'p_s_ppc_LV051', 'q_s_ppc_LV051', 'i_sa_ref_LV051', 'i_sr_ref_LV051', 'irrad_LV052', 'temp_deg_LV052', 'lvrt_ext_LV052', 'p_s_ppc_LV052', 'q_s_ppc_LV052', 'i_sa_ref_LV052', 'i_sr_ref_LV052', 'irrad_LV053', 'temp_deg_LV053', 'lvrt_ext_LV053', 'p_s_ppc_LV053', 'q_s_ppc_LV053', 'i_sa_ref_LV053', 'i_sr_ref_LV053', 'irrad_LV054', 'temp_deg_LV054', 'lvrt_ext_LV054', 'p_s_ppc_LV054', 'q_s_ppc_LV054', 'i_sa_ref_LV054', 'i_sr_ref_LV054', 'irrad_LV055', 'temp_deg_LV055', 'lvrt_ext_LV055', 'p_s_ppc_LV055', 'q_s_ppc_LV055', 'i_sa_ref_LV055', 'i_sr_ref_LV055', 'irrad_LV056', 'temp_deg_LV056', 'lvrt_ext_LV056', 'p_s_ppc_LV056', 'q_s_ppc_LV056', 'i_sa_ref_LV056', 'i_sr_ref_LV056', 'irrad_LV057', 'temp_deg_LV057', 'lvrt_ext_LV057', 'p_s_ppc_LV057', 'q_s_ppc_LV057', 'i_sa_ref_LV057', 'i_sr_ref_LV057', 'irrad_LV058', 'temp_deg_LV058', 'lvrt_ext_LV058', 'p_s_ppc_LV058', 'q_s_ppc_LV058', 'i_sa_ref_LV058', 'i_sr_ref_LV058', 'irrad_LV059', 'temp_deg_LV059', 'lvrt_ext_LV059', 'p_s_ppc_LV059', 'q_s_ppc_LV059', 'i_sa_ref_LV059', 'i_sr_ref_LV059', 'irrad_LV060', 'temp_deg_LV060', 'lvrt_ext_LV060', 'p_s_ppc_LV060', 'q_s_ppc_LV060', 'i_sa_ref_LV060', 'i_sr_ref_LV060', 'irrad_LV061', 'temp_deg_LV061', 'lvrt_ext_LV061', 'p_s_ppc_LV061', 'q_s_ppc_LV061', 'i_sa_ref_LV061', 'i_sr_ref_LV061', 'irrad_LV062', 'temp_deg_LV062', 'lvrt_ext_LV062', 'p_s_ppc_LV062', 'q_s_ppc_LV062', 'i_sa_ref_LV062', 'i_sr_ref_LV062', 'irrad_LV063', 'temp_deg_LV063', 'lvrt_ext_LV063', 'p_s_ppc_LV063', 'q_s_ppc_LV063', 'i_sa_ref_LV063', 'i_sr_ref_LV063', 'irrad_LV064', 'temp_deg_LV064', 'lvrt_ext_LV064', 'p_s_ppc_LV064', 'q_s_ppc_LV064', 'i_sa_ref_LV064', 'i_sr_ref_LV064', 'irrad_LV065', 'temp_deg_LV065', 'lvrt_ext_LV065', 'p_s_ppc_LV065', 'q_s_ppc_LV065', 'i_sa_ref_LV065', 'i_sr_ref_LV065', 'irrad_LV066', 'temp_deg_LV066', 'lvrt_ext_LV066', 'p_s_ppc_LV066', 'q_s_ppc_LV066', 'i_sa_ref_LV066', 'i_sr_ref_LV066', 'irrad_LV067', 'temp_deg_LV067', 'lvrt_ext_LV067', 'p_s_ppc_LV067', 'q_s_ppc_LV067', 'i_sa_ref_LV067', 'i_sr_ref_LV067', 'irrad_LV068', 'temp_deg_LV068', 'lvrt_ext_LV068', 'p_s_ppc_LV068', 'q_s_ppc_LV068', 'i_sa_ref_LV068', 'i_sr_ref_LV068', 'irrad_LV069', 'temp_deg_LV069', 'lvrt_ext_LV069', 'p_s_ppc_LV069', 'q_s_ppc_LV069', 'i_sa_ref_LV069', 'i_sr_ref_LV069', 'irrad_LV070', 'temp_deg_LV070', 'lvrt_ext_LV070', 'p_s_ppc_LV070', 'q_s_ppc_LV070', 'i_sa_ref_LV070', 'i_sr_ref_LV070', 'irrad_LV071', 'temp_deg_LV071', 'lvrt_ext_LV071', 'p_s_ppc_LV071', 'q_s_ppc_LV071', 'i_sa_ref_LV071', 'i_sr_ref_LV071', 'irrad_LV072', 'temp_deg_LV072', 'lvrt_ext_LV072', 'p_s_ppc_LV072', 'q_s_ppc_LV072', 'i_sa_ref_LV072', 'i_sr_ref_LV072', 'irrad_LV073', 'temp_deg_LV073', 'lvrt_ext_LV073', 'p_s_ppc_LV073', 'q_s_ppc_LV073', 'i_sa_ref_LV073', 'i_sr_ref_LV073', 'irrad_LV074', 'temp_deg_LV074', 'lvrt_ext_LV074', 'p_s_ppc_LV074', 'q_s_ppc_LV074', 'i_sa_ref_LV074', 'i_sr_ref_LV074', 'irrad_LV075', 'temp_deg_LV075', 'lvrt_ext_LV075', 'p_s_ppc_LV075', 'q_s_ppc_LV075', 'i_sa_ref_LV075', 'i_sr_ref_LV075', 'irrad_LV076', 'temp_deg_LV076', 'lvrt_ext_LV076', 'p_s_ppc_LV076', 'q_s_ppc_LV076', 'i_sa_ref_LV076', 'i_sr_ref_LV076', 'irrad_LV077', 'temp_deg_LV077', 'lvrt_ext_LV077', 'p_s_ppc_LV077', 'q_s_ppc_LV077', 'i_sa_ref_LV077', 'i_sr_ref_LV077', 'irrad_LV078', 'temp_deg_LV078', 'lvrt_ext_LV078', 'p_s_ppc_LV078', 'q_s_ppc_LV078', 'i_sa_ref_LV078', 'i_sr_ref_LV078', 'irrad_LV079', 'temp_deg_LV079', 'lvrt_ext_LV079', 'p_s_ppc_LV079', 'q_s_ppc_LV079', 'i_sa_ref_LV079', 'i_sr_ref_LV079', 'irrad_LV080', 'temp_deg_LV080', 'lvrt_ext_LV080', 'p_s_ppc_LV080', 'q_s_ppc_LV080', 'i_sa_ref_LV080', 'i_sr_ref_LV080', 'irrad_LV081', 'temp_deg_LV081', 'lvrt_ext_LV081', 'p_s_ppc_LV081', 'q_s_ppc_LV081', 'i_sa_ref_LV081', 'i_sr_ref_LV081', 'irrad_LV082', 'temp_deg_LV082', 'lvrt_ext_LV082', 'p_s_ppc_LV082', 'q_s_ppc_LV082', 'i_sa_ref_LV082', 'i_sr_ref_LV082', 'irrad_LV083', 'temp_deg_LV083', 'lvrt_ext_LV083', 'p_s_ppc_LV083', 'q_s_ppc_LV083', 'i_sa_ref_LV083', 'i_sr_ref_LV083', 'irrad_LV084', 'temp_deg_LV084', 'lvrt_ext_LV084', 'p_s_ppc_LV084', 'q_s_ppc_LV084', 'i_sa_ref_LV084', 'i_sr_ref_LV084', 'irrad_LV085', 'temp_deg_LV085', 'lvrt_ext_LV085', 'p_s_ppc_LV085', 'q_s_ppc_LV085', 'i_sa_ref_LV085', 'i_sr_ref_LV085', 'irrad_LV086', 'temp_deg_LV086', 'lvrt_ext_LV086', 'p_s_ppc_LV086', 'q_s_ppc_LV086', 'i_sa_ref_LV086', 'i_sr_ref_LV086', 'irrad_LV087', 'temp_deg_LV087', 'lvrt_ext_LV087', 'p_s_ppc_LV087', 'q_s_ppc_LV087', 'i_sa_ref_LV087', 'i_sr_ref_LV087', 'irrad_LV088', 'temp_deg_LV088', 'lvrt_ext_LV088', 'p_s_ppc_LV088', 'q_s_ppc_LV088', 'i_sa_ref_LV088', 'i_sr_ref_LV088', 'irrad_LV089', 'temp_deg_LV089', 'lvrt_ext_LV089', 'p_s_ppc_LV089', 'q_s_ppc_LV089', 'i_sa_ref_LV089', 'i_sr_ref_LV089', 'irrad_LV090', 'temp_deg_LV090', 'lvrt_ext_LV090', 'p_s_ppc_LV090', 'q_s_ppc_LV090', 'i_sa_ref_LV090', 'i_sr_ref_LV090', 'irrad_LV091', 'temp_deg_LV091', 'lvrt_ext_LV091', 'p_s_ppc_LV091', 'q_s_ppc_LV091', 'i_sa_ref_LV091', 'i_sr_ref_LV091', 'irrad_LV092', 'temp_deg_LV092', 'lvrt_ext_LV092', 'p_s_ppc_LV092', 'q_s_ppc_LV092', 'i_sa_ref_LV092', 'i_sr_ref_LV092', 'irrad_LV093', 'temp_deg_LV093', 'lvrt_ext_LV093', 'p_s_ppc_LV093', 'q_s_ppc_LV093', 'i_sa_ref_LV093', 'i_sr_ref_LV093', 'irrad_LV094', 'temp_deg_LV094', 'lvrt_ext_LV094', 'p_s_ppc_LV094', 'q_s_ppc_LV094', 'i_sa_ref_LV094', 'i_sr_ref_LV094', 'irrad_LV095', 'temp_deg_LV095', 'lvrt_ext_LV095', 'p_s_ppc_LV095', 'q_s_ppc_LV095', 'i_sa_ref_LV095', 'i_sr_ref_LV095', 'irrad_LV096', 'temp_deg_LV096', 'lvrt_ext_LV096', 'p_s_ppc_LV096', 'q_s_ppc_LV096', 'i_sa_ref_LV096', 'i_sr_ref_LV096', 'irrad_LV097', 'temp_deg_LV097', 'lvrt_ext_LV097', 'p_s_ppc_LV097', 'q_s_ppc_LV097', 'i_sa_ref_LV097', 'i_sr_ref_LV097', 'irrad_LV098', 'temp_deg_LV098', 'lvrt_ext_LV098', 'p_s_ppc_LV098', 'q_s_ppc_LV098', 'i_sa_ref_LV098', 'i_sr_ref_LV098', 'irrad_LV099', 'temp_deg_LV099', 'lvrt_ext_LV099', 'p_s_ppc_LV099', 'q_s_ppc_LV099', 'i_sa_ref_LV099', 'i_sr_ref_LV099', 'irrad_LV100', 'temp_deg_LV100', 'lvrt_ext_LV100', 'p_s_ppc_LV100', 'q_s_ppc_LV100', 'i_sa_ref_LV100', 'i_sr_ref_LV100'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.5, 0.0, 0.0, 0.0] 
        self.outputs_list = ['V_POI_MV', 'V_POI', 'V_GRID', 'V_LV001', 'V_MV001', 'V_LV002', 'V_MV002', 'V_LV003', 'V_MV003', 'V_LV004', 'V_MV004', 'V_LV005', 'V_MV005', 'V_LV006', 'V_MV006', 'V_LV007', 'V_MV007', 'V_LV008', 'V_MV008', 'V_LV009', 'V_MV009', 'V_LV010', 'V_MV010', 'V_LV011', 'V_MV011', 'V_LV012', 'V_MV012', 'V_LV013', 'V_MV013', 'V_LV014', 'V_MV014', 'V_LV015', 'V_MV015', 'V_LV016', 'V_MV016', 'V_LV017', 'V_MV017', 'V_LV018', 'V_MV018', 'V_LV019', 'V_MV019', 'V_LV020', 'V_MV020', 'V_LV021', 'V_MV021', 'V_LV022', 'V_MV022', 'V_LV023', 'V_MV023', 'V_LV024', 'V_MV024', 'V_LV025', 'V_MV025', 'V_LV026', 'V_MV026', 'V_LV027', 'V_MV027', 'V_LV028', 'V_MV028', 'V_LV029', 'V_MV029', 'V_LV030', 'V_MV030', 'V_LV031', 'V_MV031', 'V_LV032', 'V_MV032', 'V_LV033', 'V_MV033', 'V_LV034', 'V_MV034', 'V_LV035', 'V_MV035', 'V_LV036', 'V_MV036', 'V_LV037', 'V_MV037', 'V_LV038', 'V_MV038', 'V_LV039', 'V_MV039', 'V_LV040', 'V_MV040', 'V_LV041', 'V_MV041', 'V_LV042', 'V_MV042', 'V_LV043', 'V_MV043', 'V_LV044', 'V_MV044', 'V_LV045', 'V_MV045', 'V_LV046', 'V_MV046', 'V_LV047', 'V_MV047', 'V_LV048', 'V_MV048', 'V_LV049', 'V_MV049', 'V_LV050', 'V_MV050', 'V_LV051', 'V_MV051', 'V_LV052', 'V_MV052', 'V_LV053', 'V_MV053', 'V_LV054', 'V_MV054', 'V_LV055', 'V_MV055', 'V_LV056', 'V_MV056', 'V_LV057', 'V_MV057', 'V_LV058', 'V_MV058', 'V_LV059', 'V_MV059', 'V_LV060', 'V_MV060', 'V_LV061', 'V_MV061', 'V_LV062', 'V_MV062', 'V_LV063', 'V_MV063', 'V_LV064', 'V_MV064', 'V_LV065', 'V_MV065', 'V_LV066', 'V_MV066', 'V_LV067', 'V_MV067', 'V_LV068', 'V_MV068', 'V_LV069', 'V_MV069', 'V_LV070', 'V_MV070', 'V_LV071', 'V_MV071', 'V_LV072', 'V_MV072', 'V_LV073', 'V_MV073', 'V_LV074', 'V_MV074', 'V_LV075', 'V_MV075', 'V_LV076', 'V_MV076', 'V_LV077', 'V_MV077', 'V_LV078', 'V_MV078', 'V_LV079', 'V_MV079', 'V_LV080', 'V_MV080', 'V_LV081', 'V_MV081', 'V_LV082', 'V_MV082', 'V_LV083', 'V_MV083', 'V_LV084', 'V_MV084', 'V_LV085', 'V_MV085', 'V_LV086', 'V_MV086', 'V_LV087', 'V_MV087', 'V_LV088', 'V_MV088', 'V_LV089', 'V_MV089', 'V_LV090', 'V_MV090', 'V_LV091', 'V_MV091', 'V_LV092', 'V_MV092', 'V_LV093', 'V_MV093', 'V_LV094', 'V_MV094', 'V_LV095', 'V_MV095', 'V_LV096', 'V_MV096', 'V_LV097', 'V_MV097', 'V_LV098', 'V_MV098', 'V_LV099', 'V_MV099', 'V_LV100', 'V_MV100', 'p_line_POI_GRID', 'q_line_POI_GRID', 'p_line_GRID_POI', 'q_line_GRID_POI', 'alpha_GRID', 'Dv_GRID', 'm_ref_LV001', 'v_sd_LV001', 'v_sq_LV001', 'lvrt_LV001', 'm_ref_LV002', 'v_sd_LV002', 'v_sq_LV002', 'lvrt_LV002', 'm_ref_LV003', 'v_sd_LV003', 'v_sq_LV003', 'lvrt_LV003', 'm_ref_LV004', 'v_sd_LV004', 'v_sq_LV004', 'lvrt_LV004', 'm_ref_LV005', 'v_sd_LV005', 'v_sq_LV005', 'lvrt_LV005', 'm_ref_LV006', 'v_sd_LV006', 'v_sq_LV006', 'lvrt_LV006', 'm_ref_LV007', 'v_sd_LV007', 'v_sq_LV007', 'lvrt_LV007', 'm_ref_LV008', 'v_sd_LV008', 'v_sq_LV008', 'lvrt_LV008', 'm_ref_LV009', 'v_sd_LV009', 'v_sq_LV009', 'lvrt_LV009', 'm_ref_LV010', 'v_sd_LV010', 'v_sq_LV010', 'lvrt_LV010', 'm_ref_LV011', 'v_sd_LV011', 'v_sq_LV011', 'lvrt_LV011', 'm_ref_LV012', 'v_sd_LV012', 'v_sq_LV012', 'lvrt_LV012', 'm_ref_LV013', 'v_sd_LV013', 'v_sq_LV013', 'lvrt_LV013', 'm_ref_LV014', 'v_sd_LV014', 'v_sq_LV014', 'lvrt_LV014', 'm_ref_LV015', 'v_sd_LV015', 'v_sq_LV015', 'lvrt_LV015', 'm_ref_LV016', 'v_sd_LV016', 'v_sq_LV016', 'lvrt_LV016', 'm_ref_LV017', 'v_sd_LV017', 'v_sq_LV017', 'lvrt_LV017', 'm_ref_LV018', 'v_sd_LV018', 'v_sq_LV018', 'lvrt_LV018', 'm_ref_LV019', 'v_sd_LV019', 'v_sq_LV019', 'lvrt_LV019', 'm_ref_LV020', 'v_sd_LV020', 'v_sq_LV020', 'lvrt_LV020', 'm_ref_LV021', 'v_sd_LV021', 'v_sq_LV021', 'lvrt_LV021', 'm_ref_LV022', 'v_sd_LV022', 'v_sq_LV022', 'lvrt_LV022', 'm_ref_LV023', 'v_sd_LV023', 'v_sq_LV023', 'lvrt_LV023', 'm_ref_LV024', 'v_sd_LV024', 'v_sq_LV024', 'lvrt_LV024', 'm_ref_LV025', 'v_sd_LV025', 'v_sq_LV025', 'lvrt_LV025', 'm_ref_LV026', 'v_sd_LV026', 'v_sq_LV026', 'lvrt_LV026', 'm_ref_LV027', 'v_sd_LV027', 'v_sq_LV027', 'lvrt_LV027', 'm_ref_LV028', 'v_sd_LV028', 'v_sq_LV028', 'lvrt_LV028', 'm_ref_LV029', 'v_sd_LV029', 'v_sq_LV029', 'lvrt_LV029', 'm_ref_LV030', 'v_sd_LV030', 'v_sq_LV030', 'lvrt_LV030', 'm_ref_LV031', 'v_sd_LV031', 'v_sq_LV031', 'lvrt_LV031', 'm_ref_LV032', 'v_sd_LV032', 'v_sq_LV032', 'lvrt_LV032', 'm_ref_LV033', 'v_sd_LV033', 'v_sq_LV033', 'lvrt_LV033', 'm_ref_LV034', 'v_sd_LV034', 'v_sq_LV034', 'lvrt_LV034', 'm_ref_LV035', 'v_sd_LV035', 'v_sq_LV035', 'lvrt_LV035', 'm_ref_LV036', 'v_sd_LV036', 'v_sq_LV036', 'lvrt_LV036', 'm_ref_LV037', 'v_sd_LV037', 'v_sq_LV037', 'lvrt_LV037', 'm_ref_LV038', 'v_sd_LV038', 'v_sq_LV038', 'lvrt_LV038', 'm_ref_LV039', 'v_sd_LV039', 'v_sq_LV039', 'lvrt_LV039', 'm_ref_LV040', 'v_sd_LV040', 'v_sq_LV040', 'lvrt_LV040', 'm_ref_LV041', 'v_sd_LV041', 'v_sq_LV041', 'lvrt_LV041', 'm_ref_LV042', 'v_sd_LV042', 'v_sq_LV042', 'lvrt_LV042', 'm_ref_LV043', 'v_sd_LV043', 'v_sq_LV043', 'lvrt_LV043', 'm_ref_LV044', 'v_sd_LV044', 'v_sq_LV044', 'lvrt_LV044', 'm_ref_LV045', 'v_sd_LV045', 'v_sq_LV045', 'lvrt_LV045', 'm_ref_LV046', 'v_sd_LV046', 'v_sq_LV046', 'lvrt_LV046', 'm_ref_LV047', 'v_sd_LV047', 'v_sq_LV047', 'lvrt_LV047', 'm_ref_LV048', 'v_sd_LV048', 'v_sq_LV048', 'lvrt_LV048', 'm_ref_LV049', 'v_sd_LV049', 'v_sq_LV049', 'lvrt_LV049', 'm_ref_LV050', 'v_sd_LV050', 'v_sq_LV050', 'lvrt_LV050', 'm_ref_LV051', 'v_sd_LV051', 'v_sq_LV051', 'lvrt_LV051', 'm_ref_LV052', 'v_sd_LV052', 'v_sq_LV052', 'lvrt_LV052', 'm_ref_LV053', 'v_sd_LV053', 'v_sq_LV053', 'lvrt_LV053', 'm_ref_LV054', 'v_sd_LV054', 'v_sq_LV054', 'lvrt_LV054', 'm_ref_LV055', 'v_sd_LV055', 'v_sq_LV055', 'lvrt_LV055', 'm_ref_LV056', 'v_sd_LV056', 'v_sq_LV056', 'lvrt_LV056', 'm_ref_LV057', 'v_sd_LV057', 'v_sq_LV057', 'lvrt_LV057', 'm_ref_LV058', 'v_sd_LV058', 'v_sq_LV058', 'lvrt_LV058', 'm_ref_LV059', 'v_sd_LV059', 'v_sq_LV059', 'lvrt_LV059', 'm_ref_LV060', 'v_sd_LV060', 'v_sq_LV060', 'lvrt_LV060', 'm_ref_LV061', 'v_sd_LV061', 'v_sq_LV061', 'lvrt_LV061', 'm_ref_LV062', 'v_sd_LV062', 'v_sq_LV062', 'lvrt_LV062', 'm_ref_LV063', 'v_sd_LV063', 'v_sq_LV063', 'lvrt_LV063', 'm_ref_LV064', 'v_sd_LV064', 'v_sq_LV064', 'lvrt_LV064', 'm_ref_LV065', 'v_sd_LV065', 'v_sq_LV065', 'lvrt_LV065', 'm_ref_LV066', 'v_sd_LV066', 'v_sq_LV066', 'lvrt_LV066', 'm_ref_LV067', 'v_sd_LV067', 'v_sq_LV067', 'lvrt_LV067', 'm_ref_LV068', 'v_sd_LV068', 'v_sq_LV068', 'lvrt_LV068', 'm_ref_LV069', 'v_sd_LV069', 'v_sq_LV069', 'lvrt_LV069', 'm_ref_LV070', 'v_sd_LV070', 'v_sq_LV070', 'lvrt_LV070', 'm_ref_LV071', 'v_sd_LV071', 'v_sq_LV071', 'lvrt_LV071', 'm_ref_LV072', 'v_sd_LV072', 'v_sq_LV072', 'lvrt_LV072', 'm_ref_LV073', 'v_sd_LV073', 'v_sq_LV073', 'lvrt_LV073', 'm_ref_LV074', 'v_sd_LV074', 'v_sq_LV074', 'lvrt_LV074', 'm_ref_LV075', 'v_sd_LV075', 'v_sq_LV075', 'lvrt_LV075', 'm_ref_LV076', 'v_sd_LV076', 'v_sq_LV076', 'lvrt_LV076', 'm_ref_LV077', 'v_sd_LV077', 'v_sq_LV077', 'lvrt_LV077', 'm_ref_LV078', 'v_sd_LV078', 'v_sq_LV078', 'lvrt_LV078', 'm_ref_LV079', 'v_sd_LV079', 'v_sq_LV079', 'lvrt_LV079', 'm_ref_LV080', 'v_sd_LV080', 'v_sq_LV080', 'lvrt_LV080', 'm_ref_LV081', 'v_sd_LV081', 'v_sq_LV081', 'lvrt_LV081', 'm_ref_LV082', 'v_sd_LV082', 'v_sq_LV082', 'lvrt_LV082', 'm_ref_LV083', 'v_sd_LV083', 'v_sq_LV083', 'lvrt_LV083', 'm_ref_LV084', 'v_sd_LV084', 'v_sq_LV084', 'lvrt_LV084', 'm_ref_LV085', 'v_sd_LV085', 'v_sq_LV085', 'lvrt_LV085', 'm_ref_LV086', 'v_sd_LV086', 'v_sq_LV086', 'lvrt_LV086', 'm_ref_LV087', 'v_sd_LV087', 'v_sq_LV087', 'lvrt_LV087', 'm_ref_LV088', 'v_sd_LV088', 'v_sq_LV088', 'lvrt_LV088', 'm_ref_LV089', 'v_sd_LV089', 'v_sq_LV089', 'lvrt_LV089', 'm_ref_LV090', 'v_sd_LV090', 'v_sq_LV090', 'lvrt_LV090', 'm_ref_LV091', 'v_sd_LV091', 'v_sq_LV091', 'lvrt_LV091', 'm_ref_LV092', 'v_sd_LV092', 'v_sq_LV092', 'lvrt_LV092', 'm_ref_LV093', 'v_sd_LV093', 'v_sq_LV093', 'lvrt_LV093', 'm_ref_LV094', 'v_sd_LV094', 'v_sq_LV094', 'lvrt_LV094', 'm_ref_LV095', 'v_sd_LV095', 'v_sq_LV095', 'lvrt_LV095', 'm_ref_LV096', 'v_sd_LV096', 'v_sq_LV096', 'lvrt_LV096', 'm_ref_LV097', 'v_sd_LV097', 'v_sq_LV097', 'lvrt_LV097', 'm_ref_LV098', 'v_sd_LV098', 'v_sq_LV098', 'lvrt_LV098', 'm_ref_LV099', 'v_sd_LV099', 'v_sq_LV099', 'lvrt_LV099', 'm_ref_LV100', 'v_sd_LV100', 'v_sq_LV100', 'lvrt_LV100'] 
        self.x_list = ['delta_GRID', 'Domega_GRID', 'Dv_GRID', 'xi_freq'] 
        self.y_run_list = ['V_POI_MV', 'theta_POI_MV', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'V_LV001', 'theta_LV001', 'V_MV001', 'theta_MV001', 'V_LV002', 'theta_LV002', 'V_MV002', 'theta_MV002', 'V_LV003', 'theta_LV003', 'V_MV003', 'theta_MV003', 'V_LV004', 'theta_LV004', 'V_MV004', 'theta_MV004', 'V_LV005', 'theta_LV005', 'V_MV005', 'theta_MV005', 'V_LV006', 'theta_LV006', 'V_MV006', 'theta_MV006', 'V_LV007', 'theta_LV007', 'V_MV007', 'theta_MV007', 'V_LV008', 'theta_LV008', 'V_MV008', 'theta_MV008', 'V_LV009', 'theta_LV009', 'V_MV009', 'theta_MV009', 'V_LV010', 'theta_LV010', 'V_MV010', 'theta_MV010', 'V_LV011', 'theta_LV011', 'V_MV011', 'theta_MV011', 'V_LV012', 'theta_LV012', 'V_MV012', 'theta_MV012', 'V_LV013', 'theta_LV013', 'V_MV013', 'theta_MV013', 'V_LV014', 'theta_LV014', 'V_MV014', 'theta_MV014', 'V_LV015', 'theta_LV015', 'V_MV015', 'theta_MV015', 'V_LV016', 'theta_LV016', 'V_MV016', 'theta_MV016', 'V_LV017', 'theta_LV017', 'V_MV017', 'theta_MV017', 'V_LV018', 'theta_LV018', 'V_MV018', 'theta_MV018', 'V_LV019', 'theta_LV019', 'V_MV019', 'theta_MV019', 'V_LV020', 'theta_LV020', 'V_MV020', 'theta_MV020', 'V_LV021', 'theta_LV021', 'V_MV021', 'theta_MV021', 'V_LV022', 'theta_LV022', 'V_MV022', 'theta_MV022', 'V_LV023', 'theta_LV023', 'V_MV023', 'theta_MV023', 'V_LV024', 'theta_LV024', 'V_MV024', 'theta_MV024', 'V_LV025', 'theta_LV025', 'V_MV025', 'theta_MV025', 'V_LV026', 'theta_LV026', 'V_MV026', 'theta_MV026', 'V_LV027', 'theta_LV027', 'V_MV027', 'theta_MV027', 'V_LV028', 'theta_LV028', 'V_MV028', 'theta_MV028', 'V_LV029', 'theta_LV029', 'V_MV029', 'theta_MV029', 'V_LV030', 'theta_LV030', 'V_MV030', 'theta_MV030', 'V_LV031', 'theta_LV031', 'V_MV031', 'theta_MV031', 'V_LV032', 'theta_LV032', 'V_MV032', 'theta_MV032', 'V_LV033', 'theta_LV033', 'V_MV033', 'theta_MV033', 'V_LV034', 'theta_LV034', 'V_MV034', 'theta_MV034', 'V_LV035', 'theta_LV035', 'V_MV035', 'theta_MV035', 'V_LV036', 'theta_LV036', 'V_MV036', 'theta_MV036', 'V_LV037', 'theta_LV037', 'V_MV037', 'theta_MV037', 'V_LV038', 'theta_LV038', 'V_MV038', 'theta_MV038', 'V_LV039', 'theta_LV039', 'V_MV039', 'theta_MV039', 'V_LV040', 'theta_LV040', 'V_MV040', 'theta_MV040', 'V_LV041', 'theta_LV041', 'V_MV041', 'theta_MV041', 'V_LV042', 'theta_LV042', 'V_MV042', 'theta_MV042', 'V_LV043', 'theta_LV043', 'V_MV043', 'theta_MV043', 'V_LV044', 'theta_LV044', 'V_MV044', 'theta_MV044', 'V_LV045', 'theta_LV045', 'V_MV045', 'theta_MV045', 'V_LV046', 'theta_LV046', 'V_MV046', 'theta_MV046', 'V_LV047', 'theta_LV047', 'V_MV047', 'theta_MV047', 'V_LV048', 'theta_LV048', 'V_MV048', 'theta_MV048', 'V_LV049', 'theta_LV049', 'V_MV049', 'theta_MV049', 'V_LV050', 'theta_LV050', 'V_MV050', 'theta_MV050', 'V_LV051', 'theta_LV051', 'V_MV051', 'theta_MV051', 'V_LV052', 'theta_LV052', 'V_MV052', 'theta_MV052', 'V_LV053', 'theta_LV053', 'V_MV053', 'theta_MV053', 'V_LV054', 'theta_LV054', 'V_MV054', 'theta_MV054', 'V_LV055', 'theta_LV055', 'V_MV055', 'theta_MV055', 'V_LV056', 'theta_LV056', 'V_MV056', 'theta_MV056', 'V_LV057', 'theta_LV057', 'V_MV057', 'theta_MV057', 'V_LV058', 'theta_LV058', 'V_MV058', 'theta_MV058', 'V_LV059', 'theta_LV059', 'V_MV059', 'theta_MV059', 'V_LV060', 'theta_LV060', 'V_MV060', 'theta_MV060', 'V_LV061', 'theta_LV061', 'V_MV061', 'theta_MV061', 'V_LV062', 'theta_LV062', 'V_MV062', 'theta_MV062', 'V_LV063', 'theta_LV063', 'V_MV063', 'theta_MV063', 'V_LV064', 'theta_LV064', 'V_MV064', 'theta_MV064', 'V_LV065', 'theta_LV065', 'V_MV065', 'theta_MV065', 'V_LV066', 'theta_LV066', 'V_MV066', 'theta_MV066', 'V_LV067', 'theta_LV067', 'V_MV067', 'theta_MV067', 'V_LV068', 'theta_LV068', 'V_MV068', 'theta_MV068', 'V_LV069', 'theta_LV069', 'V_MV069', 'theta_MV069', 'V_LV070', 'theta_LV070', 'V_MV070', 'theta_MV070', 'V_LV071', 'theta_LV071', 'V_MV071', 'theta_MV071', 'V_LV072', 'theta_LV072', 'V_MV072', 'theta_MV072', 'V_LV073', 'theta_LV073', 'V_MV073', 'theta_MV073', 'V_LV074', 'theta_LV074', 'V_MV074', 'theta_MV074', 'V_LV075', 'theta_LV075', 'V_MV075', 'theta_MV075', 'V_LV076', 'theta_LV076', 'V_MV076', 'theta_MV076', 'V_LV077', 'theta_LV077', 'V_MV077', 'theta_MV077', 'V_LV078', 'theta_LV078', 'V_MV078', 'theta_MV078', 'V_LV079', 'theta_LV079', 'V_MV079', 'theta_MV079', 'V_LV080', 'theta_LV080', 'V_MV080', 'theta_MV080', 'V_LV081', 'theta_LV081', 'V_MV081', 'theta_MV081', 'V_LV082', 'theta_LV082', 'V_MV082', 'theta_MV082', 'V_LV083', 'theta_LV083', 'V_MV083', 'theta_MV083', 'V_LV084', 'theta_LV084', 'V_MV084', 'theta_MV084', 'V_LV085', 'theta_LV085', 'V_MV085', 'theta_MV085', 'V_LV086', 'theta_LV086', 'V_MV086', 'theta_MV086', 'V_LV087', 'theta_LV087', 'V_MV087', 'theta_MV087', 'V_LV088', 'theta_LV088', 'V_MV088', 'theta_MV088', 'V_LV089', 'theta_LV089', 'V_MV089', 'theta_MV089', 'V_LV090', 'theta_LV090', 'V_MV090', 'theta_MV090', 'V_LV091', 'theta_LV091', 'V_MV091', 'theta_MV091', 'V_LV092', 'theta_LV092', 'V_MV092', 'theta_MV092', 'V_LV093', 'theta_LV093', 'V_MV093', 'theta_MV093', 'V_LV094', 'theta_LV094', 'V_MV094', 'theta_MV094', 'V_LV095', 'theta_LV095', 'V_MV095', 'theta_MV095', 'V_LV096', 'theta_LV096', 'V_MV096', 'theta_MV096', 'V_LV097', 'theta_LV097', 'V_MV097', 'theta_MV097', 'V_LV098', 'theta_LV098', 'V_MV098', 'theta_MV098', 'V_LV099', 'theta_LV099', 'V_MV099', 'theta_MV099', 'V_LV100', 'theta_LV100', 'V_MV100', 'theta_MV100', 'omega_GRID', 'i_d_GRID', 'i_q_GRID', 'p_s_GRID', 'q_s_GRID', 'v_dc_v_LV001', 'i_sq_ref_LV001', 'i_sd_ref_LV001', 'p_s_LV001', 'q_s_LV001', 'v_dc_v_LV002', 'i_sq_ref_LV002', 'i_sd_ref_LV002', 'p_s_LV002', 'q_s_LV002', 'v_dc_v_LV003', 'i_sq_ref_LV003', 'i_sd_ref_LV003', 'p_s_LV003', 'q_s_LV003', 'v_dc_v_LV004', 'i_sq_ref_LV004', 'i_sd_ref_LV004', 'p_s_LV004', 'q_s_LV004', 'v_dc_v_LV005', 'i_sq_ref_LV005', 'i_sd_ref_LV005', 'p_s_LV005', 'q_s_LV005', 'v_dc_v_LV006', 'i_sq_ref_LV006', 'i_sd_ref_LV006', 'p_s_LV006', 'q_s_LV006', 'v_dc_v_LV007', 'i_sq_ref_LV007', 'i_sd_ref_LV007', 'p_s_LV007', 'q_s_LV007', 'v_dc_v_LV008', 'i_sq_ref_LV008', 'i_sd_ref_LV008', 'p_s_LV008', 'q_s_LV008', 'v_dc_v_LV009', 'i_sq_ref_LV009', 'i_sd_ref_LV009', 'p_s_LV009', 'q_s_LV009', 'v_dc_v_LV010', 'i_sq_ref_LV010', 'i_sd_ref_LV010', 'p_s_LV010', 'q_s_LV010', 'v_dc_v_LV011', 'i_sq_ref_LV011', 'i_sd_ref_LV011', 'p_s_LV011', 'q_s_LV011', 'v_dc_v_LV012', 'i_sq_ref_LV012', 'i_sd_ref_LV012', 'p_s_LV012', 'q_s_LV012', 'v_dc_v_LV013', 'i_sq_ref_LV013', 'i_sd_ref_LV013', 'p_s_LV013', 'q_s_LV013', 'v_dc_v_LV014', 'i_sq_ref_LV014', 'i_sd_ref_LV014', 'p_s_LV014', 'q_s_LV014', 'v_dc_v_LV015', 'i_sq_ref_LV015', 'i_sd_ref_LV015', 'p_s_LV015', 'q_s_LV015', 'v_dc_v_LV016', 'i_sq_ref_LV016', 'i_sd_ref_LV016', 'p_s_LV016', 'q_s_LV016', 'v_dc_v_LV017', 'i_sq_ref_LV017', 'i_sd_ref_LV017', 'p_s_LV017', 'q_s_LV017', 'v_dc_v_LV018', 'i_sq_ref_LV018', 'i_sd_ref_LV018', 'p_s_LV018', 'q_s_LV018', 'v_dc_v_LV019', 'i_sq_ref_LV019', 'i_sd_ref_LV019', 'p_s_LV019', 'q_s_LV019', 'v_dc_v_LV020', 'i_sq_ref_LV020', 'i_sd_ref_LV020', 'p_s_LV020', 'q_s_LV020', 'v_dc_v_LV021', 'i_sq_ref_LV021', 'i_sd_ref_LV021', 'p_s_LV021', 'q_s_LV021', 'v_dc_v_LV022', 'i_sq_ref_LV022', 'i_sd_ref_LV022', 'p_s_LV022', 'q_s_LV022', 'v_dc_v_LV023', 'i_sq_ref_LV023', 'i_sd_ref_LV023', 'p_s_LV023', 'q_s_LV023', 'v_dc_v_LV024', 'i_sq_ref_LV024', 'i_sd_ref_LV024', 'p_s_LV024', 'q_s_LV024', 'v_dc_v_LV025', 'i_sq_ref_LV025', 'i_sd_ref_LV025', 'p_s_LV025', 'q_s_LV025', 'v_dc_v_LV026', 'i_sq_ref_LV026', 'i_sd_ref_LV026', 'p_s_LV026', 'q_s_LV026', 'v_dc_v_LV027', 'i_sq_ref_LV027', 'i_sd_ref_LV027', 'p_s_LV027', 'q_s_LV027', 'v_dc_v_LV028', 'i_sq_ref_LV028', 'i_sd_ref_LV028', 'p_s_LV028', 'q_s_LV028', 'v_dc_v_LV029', 'i_sq_ref_LV029', 'i_sd_ref_LV029', 'p_s_LV029', 'q_s_LV029', 'v_dc_v_LV030', 'i_sq_ref_LV030', 'i_sd_ref_LV030', 'p_s_LV030', 'q_s_LV030', 'v_dc_v_LV031', 'i_sq_ref_LV031', 'i_sd_ref_LV031', 'p_s_LV031', 'q_s_LV031', 'v_dc_v_LV032', 'i_sq_ref_LV032', 'i_sd_ref_LV032', 'p_s_LV032', 'q_s_LV032', 'v_dc_v_LV033', 'i_sq_ref_LV033', 'i_sd_ref_LV033', 'p_s_LV033', 'q_s_LV033', 'v_dc_v_LV034', 'i_sq_ref_LV034', 'i_sd_ref_LV034', 'p_s_LV034', 'q_s_LV034', 'v_dc_v_LV035', 'i_sq_ref_LV035', 'i_sd_ref_LV035', 'p_s_LV035', 'q_s_LV035', 'v_dc_v_LV036', 'i_sq_ref_LV036', 'i_sd_ref_LV036', 'p_s_LV036', 'q_s_LV036', 'v_dc_v_LV037', 'i_sq_ref_LV037', 'i_sd_ref_LV037', 'p_s_LV037', 'q_s_LV037', 'v_dc_v_LV038', 'i_sq_ref_LV038', 'i_sd_ref_LV038', 'p_s_LV038', 'q_s_LV038', 'v_dc_v_LV039', 'i_sq_ref_LV039', 'i_sd_ref_LV039', 'p_s_LV039', 'q_s_LV039', 'v_dc_v_LV040', 'i_sq_ref_LV040', 'i_sd_ref_LV040', 'p_s_LV040', 'q_s_LV040', 'v_dc_v_LV041', 'i_sq_ref_LV041', 'i_sd_ref_LV041', 'p_s_LV041', 'q_s_LV041', 'v_dc_v_LV042', 'i_sq_ref_LV042', 'i_sd_ref_LV042', 'p_s_LV042', 'q_s_LV042', 'v_dc_v_LV043', 'i_sq_ref_LV043', 'i_sd_ref_LV043', 'p_s_LV043', 'q_s_LV043', 'v_dc_v_LV044', 'i_sq_ref_LV044', 'i_sd_ref_LV044', 'p_s_LV044', 'q_s_LV044', 'v_dc_v_LV045', 'i_sq_ref_LV045', 'i_sd_ref_LV045', 'p_s_LV045', 'q_s_LV045', 'v_dc_v_LV046', 'i_sq_ref_LV046', 'i_sd_ref_LV046', 'p_s_LV046', 'q_s_LV046', 'v_dc_v_LV047', 'i_sq_ref_LV047', 'i_sd_ref_LV047', 'p_s_LV047', 'q_s_LV047', 'v_dc_v_LV048', 'i_sq_ref_LV048', 'i_sd_ref_LV048', 'p_s_LV048', 'q_s_LV048', 'v_dc_v_LV049', 'i_sq_ref_LV049', 'i_sd_ref_LV049', 'p_s_LV049', 'q_s_LV049', 'v_dc_v_LV050', 'i_sq_ref_LV050', 'i_sd_ref_LV050', 'p_s_LV050', 'q_s_LV050', 'v_dc_v_LV051', 'i_sq_ref_LV051', 'i_sd_ref_LV051', 'p_s_LV051', 'q_s_LV051', 'v_dc_v_LV052', 'i_sq_ref_LV052', 'i_sd_ref_LV052', 'p_s_LV052', 'q_s_LV052', 'v_dc_v_LV053', 'i_sq_ref_LV053', 'i_sd_ref_LV053', 'p_s_LV053', 'q_s_LV053', 'v_dc_v_LV054', 'i_sq_ref_LV054', 'i_sd_ref_LV054', 'p_s_LV054', 'q_s_LV054', 'v_dc_v_LV055', 'i_sq_ref_LV055', 'i_sd_ref_LV055', 'p_s_LV055', 'q_s_LV055', 'v_dc_v_LV056', 'i_sq_ref_LV056', 'i_sd_ref_LV056', 'p_s_LV056', 'q_s_LV056', 'v_dc_v_LV057', 'i_sq_ref_LV057', 'i_sd_ref_LV057', 'p_s_LV057', 'q_s_LV057', 'v_dc_v_LV058', 'i_sq_ref_LV058', 'i_sd_ref_LV058', 'p_s_LV058', 'q_s_LV058', 'v_dc_v_LV059', 'i_sq_ref_LV059', 'i_sd_ref_LV059', 'p_s_LV059', 'q_s_LV059', 'v_dc_v_LV060', 'i_sq_ref_LV060', 'i_sd_ref_LV060', 'p_s_LV060', 'q_s_LV060', 'v_dc_v_LV061', 'i_sq_ref_LV061', 'i_sd_ref_LV061', 'p_s_LV061', 'q_s_LV061', 'v_dc_v_LV062', 'i_sq_ref_LV062', 'i_sd_ref_LV062', 'p_s_LV062', 'q_s_LV062', 'v_dc_v_LV063', 'i_sq_ref_LV063', 'i_sd_ref_LV063', 'p_s_LV063', 'q_s_LV063', 'v_dc_v_LV064', 'i_sq_ref_LV064', 'i_sd_ref_LV064', 'p_s_LV064', 'q_s_LV064', 'v_dc_v_LV065', 'i_sq_ref_LV065', 'i_sd_ref_LV065', 'p_s_LV065', 'q_s_LV065', 'v_dc_v_LV066', 'i_sq_ref_LV066', 'i_sd_ref_LV066', 'p_s_LV066', 'q_s_LV066', 'v_dc_v_LV067', 'i_sq_ref_LV067', 'i_sd_ref_LV067', 'p_s_LV067', 'q_s_LV067', 'v_dc_v_LV068', 'i_sq_ref_LV068', 'i_sd_ref_LV068', 'p_s_LV068', 'q_s_LV068', 'v_dc_v_LV069', 'i_sq_ref_LV069', 'i_sd_ref_LV069', 'p_s_LV069', 'q_s_LV069', 'v_dc_v_LV070', 'i_sq_ref_LV070', 'i_sd_ref_LV070', 'p_s_LV070', 'q_s_LV070', 'v_dc_v_LV071', 'i_sq_ref_LV071', 'i_sd_ref_LV071', 'p_s_LV071', 'q_s_LV071', 'v_dc_v_LV072', 'i_sq_ref_LV072', 'i_sd_ref_LV072', 'p_s_LV072', 'q_s_LV072', 'v_dc_v_LV073', 'i_sq_ref_LV073', 'i_sd_ref_LV073', 'p_s_LV073', 'q_s_LV073', 'v_dc_v_LV074', 'i_sq_ref_LV074', 'i_sd_ref_LV074', 'p_s_LV074', 'q_s_LV074', 'v_dc_v_LV075', 'i_sq_ref_LV075', 'i_sd_ref_LV075', 'p_s_LV075', 'q_s_LV075', 'v_dc_v_LV076', 'i_sq_ref_LV076', 'i_sd_ref_LV076', 'p_s_LV076', 'q_s_LV076', 'v_dc_v_LV077', 'i_sq_ref_LV077', 'i_sd_ref_LV077', 'p_s_LV077', 'q_s_LV077', 'v_dc_v_LV078', 'i_sq_ref_LV078', 'i_sd_ref_LV078', 'p_s_LV078', 'q_s_LV078', 'v_dc_v_LV079', 'i_sq_ref_LV079', 'i_sd_ref_LV079', 'p_s_LV079', 'q_s_LV079', 'v_dc_v_LV080', 'i_sq_ref_LV080', 'i_sd_ref_LV080', 'p_s_LV080', 'q_s_LV080', 'v_dc_v_LV081', 'i_sq_ref_LV081', 'i_sd_ref_LV081', 'p_s_LV081', 'q_s_LV081', 'v_dc_v_LV082', 'i_sq_ref_LV082', 'i_sd_ref_LV082', 'p_s_LV082', 'q_s_LV082', 'v_dc_v_LV083', 'i_sq_ref_LV083', 'i_sd_ref_LV083', 'p_s_LV083', 'q_s_LV083', 'v_dc_v_LV084', 'i_sq_ref_LV084', 'i_sd_ref_LV084', 'p_s_LV084', 'q_s_LV084', 'v_dc_v_LV085', 'i_sq_ref_LV085', 'i_sd_ref_LV085', 'p_s_LV085', 'q_s_LV085', 'v_dc_v_LV086', 'i_sq_ref_LV086', 'i_sd_ref_LV086', 'p_s_LV086', 'q_s_LV086', 'v_dc_v_LV087', 'i_sq_ref_LV087', 'i_sd_ref_LV087', 'p_s_LV087', 'q_s_LV087', 'v_dc_v_LV088', 'i_sq_ref_LV088', 'i_sd_ref_LV088', 'p_s_LV088', 'q_s_LV088', 'v_dc_v_LV089', 'i_sq_ref_LV089', 'i_sd_ref_LV089', 'p_s_LV089', 'q_s_LV089', 'v_dc_v_LV090', 'i_sq_ref_LV090', 'i_sd_ref_LV090', 'p_s_LV090', 'q_s_LV090', 'v_dc_v_LV091', 'i_sq_ref_LV091', 'i_sd_ref_LV091', 'p_s_LV091', 'q_s_LV091', 'v_dc_v_LV092', 'i_sq_ref_LV092', 'i_sd_ref_LV092', 'p_s_LV092', 'q_s_LV092', 'v_dc_v_LV093', 'i_sq_ref_LV093', 'i_sd_ref_LV093', 'p_s_LV093', 'q_s_LV093', 'v_dc_v_LV094', 'i_sq_ref_LV094', 'i_sd_ref_LV094', 'p_s_LV094', 'q_s_LV094', 'v_dc_v_LV095', 'i_sq_ref_LV095', 'i_sd_ref_LV095', 'p_s_LV095', 'q_s_LV095', 'v_dc_v_LV096', 'i_sq_ref_LV096', 'i_sd_ref_LV096', 'p_s_LV096', 'q_s_LV096', 'v_dc_v_LV097', 'i_sq_ref_LV097', 'i_sd_ref_LV097', 'p_s_LV097', 'q_s_LV097', 'v_dc_v_LV098', 'i_sq_ref_LV098', 'i_sd_ref_LV098', 'p_s_LV098', 'q_s_LV098', 'v_dc_v_LV099', 'i_sq_ref_LV099', 'i_sd_ref_LV099', 'p_s_LV099', 'q_s_LV099', 'v_dc_v_LV100', 'i_sq_ref_LV100', 'i_sd_ref_LV100', 'p_s_LV100', 'q_s_LV100', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_POI_MV', 'theta_POI_MV', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'V_LV001', 'theta_LV001', 'V_MV001', 'theta_MV001', 'V_LV002', 'theta_LV002', 'V_MV002', 'theta_MV002', 'V_LV003', 'theta_LV003', 'V_MV003', 'theta_MV003', 'V_LV004', 'theta_LV004', 'V_MV004', 'theta_MV004', 'V_LV005', 'theta_LV005', 'V_MV005', 'theta_MV005', 'V_LV006', 'theta_LV006', 'V_MV006', 'theta_MV006', 'V_LV007', 'theta_LV007', 'V_MV007', 'theta_MV007', 'V_LV008', 'theta_LV008', 'V_MV008', 'theta_MV008', 'V_LV009', 'theta_LV009', 'V_MV009', 'theta_MV009', 'V_LV010', 'theta_LV010', 'V_MV010', 'theta_MV010', 'V_LV011', 'theta_LV011', 'V_MV011', 'theta_MV011', 'V_LV012', 'theta_LV012', 'V_MV012', 'theta_MV012', 'V_LV013', 'theta_LV013', 'V_MV013', 'theta_MV013', 'V_LV014', 'theta_LV014', 'V_MV014', 'theta_MV014', 'V_LV015', 'theta_LV015', 'V_MV015', 'theta_MV015', 'V_LV016', 'theta_LV016', 'V_MV016', 'theta_MV016', 'V_LV017', 'theta_LV017', 'V_MV017', 'theta_MV017', 'V_LV018', 'theta_LV018', 'V_MV018', 'theta_MV018', 'V_LV019', 'theta_LV019', 'V_MV019', 'theta_MV019', 'V_LV020', 'theta_LV020', 'V_MV020', 'theta_MV020', 'V_LV021', 'theta_LV021', 'V_MV021', 'theta_MV021', 'V_LV022', 'theta_LV022', 'V_MV022', 'theta_MV022', 'V_LV023', 'theta_LV023', 'V_MV023', 'theta_MV023', 'V_LV024', 'theta_LV024', 'V_MV024', 'theta_MV024', 'V_LV025', 'theta_LV025', 'V_MV025', 'theta_MV025', 'V_LV026', 'theta_LV026', 'V_MV026', 'theta_MV026', 'V_LV027', 'theta_LV027', 'V_MV027', 'theta_MV027', 'V_LV028', 'theta_LV028', 'V_MV028', 'theta_MV028', 'V_LV029', 'theta_LV029', 'V_MV029', 'theta_MV029', 'V_LV030', 'theta_LV030', 'V_MV030', 'theta_MV030', 'V_LV031', 'theta_LV031', 'V_MV031', 'theta_MV031', 'V_LV032', 'theta_LV032', 'V_MV032', 'theta_MV032', 'V_LV033', 'theta_LV033', 'V_MV033', 'theta_MV033', 'V_LV034', 'theta_LV034', 'V_MV034', 'theta_MV034', 'V_LV035', 'theta_LV035', 'V_MV035', 'theta_MV035', 'V_LV036', 'theta_LV036', 'V_MV036', 'theta_MV036', 'V_LV037', 'theta_LV037', 'V_MV037', 'theta_MV037', 'V_LV038', 'theta_LV038', 'V_MV038', 'theta_MV038', 'V_LV039', 'theta_LV039', 'V_MV039', 'theta_MV039', 'V_LV040', 'theta_LV040', 'V_MV040', 'theta_MV040', 'V_LV041', 'theta_LV041', 'V_MV041', 'theta_MV041', 'V_LV042', 'theta_LV042', 'V_MV042', 'theta_MV042', 'V_LV043', 'theta_LV043', 'V_MV043', 'theta_MV043', 'V_LV044', 'theta_LV044', 'V_MV044', 'theta_MV044', 'V_LV045', 'theta_LV045', 'V_MV045', 'theta_MV045', 'V_LV046', 'theta_LV046', 'V_MV046', 'theta_MV046', 'V_LV047', 'theta_LV047', 'V_MV047', 'theta_MV047', 'V_LV048', 'theta_LV048', 'V_MV048', 'theta_MV048', 'V_LV049', 'theta_LV049', 'V_MV049', 'theta_MV049', 'V_LV050', 'theta_LV050', 'V_MV050', 'theta_MV050', 'V_LV051', 'theta_LV051', 'V_MV051', 'theta_MV051', 'V_LV052', 'theta_LV052', 'V_MV052', 'theta_MV052', 'V_LV053', 'theta_LV053', 'V_MV053', 'theta_MV053', 'V_LV054', 'theta_LV054', 'V_MV054', 'theta_MV054', 'V_LV055', 'theta_LV055', 'V_MV055', 'theta_MV055', 'V_LV056', 'theta_LV056', 'V_MV056', 'theta_MV056', 'V_LV057', 'theta_LV057', 'V_MV057', 'theta_MV057', 'V_LV058', 'theta_LV058', 'V_MV058', 'theta_MV058', 'V_LV059', 'theta_LV059', 'V_MV059', 'theta_MV059', 'V_LV060', 'theta_LV060', 'V_MV060', 'theta_MV060', 'V_LV061', 'theta_LV061', 'V_MV061', 'theta_MV061', 'V_LV062', 'theta_LV062', 'V_MV062', 'theta_MV062', 'V_LV063', 'theta_LV063', 'V_MV063', 'theta_MV063', 'V_LV064', 'theta_LV064', 'V_MV064', 'theta_MV064', 'V_LV065', 'theta_LV065', 'V_MV065', 'theta_MV065', 'V_LV066', 'theta_LV066', 'V_MV066', 'theta_MV066', 'V_LV067', 'theta_LV067', 'V_MV067', 'theta_MV067', 'V_LV068', 'theta_LV068', 'V_MV068', 'theta_MV068', 'V_LV069', 'theta_LV069', 'V_MV069', 'theta_MV069', 'V_LV070', 'theta_LV070', 'V_MV070', 'theta_MV070', 'V_LV071', 'theta_LV071', 'V_MV071', 'theta_MV071', 'V_LV072', 'theta_LV072', 'V_MV072', 'theta_MV072', 'V_LV073', 'theta_LV073', 'V_MV073', 'theta_MV073', 'V_LV074', 'theta_LV074', 'V_MV074', 'theta_MV074', 'V_LV075', 'theta_LV075', 'V_MV075', 'theta_MV075', 'V_LV076', 'theta_LV076', 'V_MV076', 'theta_MV076', 'V_LV077', 'theta_LV077', 'V_MV077', 'theta_MV077', 'V_LV078', 'theta_LV078', 'V_MV078', 'theta_MV078', 'V_LV079', 'theta_LV079', 'V_MV079', 'theta_MV079', 'V_LV080', 'theta_LV080', 'V_MV080', 'theta_MV080', 'V_LV081', 'theta_LV081', 'V_MV081', 'theta_MV081', 'V_LV082', 'theta_LV082', 'V_MV082', 'theta_MV082', 'V_LV083', 'theta_LV083', 'V_MV083', 'theta_MV083', 'V_LV084', 'theta_LV084', 'V_MV084', 'theta_MV084', 'V_LV085', 'theta_LV085', 'V_MV085', 'theta_MV085', 'V_LV086', 'theta_LV086', 'V_MV086', 'theta_MV086', 'V_LV087', 'theta_LV087', 'V_MV087', 'theta_MV087', 'V_LV088', 'theta_LV088', 'V_MV088', 'theta_MV088', 'V_LV089', 'theta_LV089', 'V_MV089', 'theta_MV089', 'V_LV090', 'theta_LV090', 'V_MV090', 'theta_MV090', 'V_LV091', 'theta_LV091', 'V_MV091', 'theta_MV091', 'V_LV092', 'theta_LV092', 'V_MV092', 'theta_MV092', 'V_LV093', 'theta_LV093', 'V_MV093', 'theta_MV093', 'V_LV094', 'theta_LV094', 'V_MV094', 'theta_MV094', 'V_LV095', 'theta_LV095', 'V_MV095', 'theta_MV095', 'V_LV096', 'theta_LV096', 'V_MV096', 'theta_MV096', 'V_LV097', 'theta_LV097', 'V_MV097', 'theta_MV097', 'V_LV098', 'theta_LV098', 'V_MV098', 'theta_MV098', 'V_LV099', 'theta_LV099', 'V_MV099', 'theta_MV099', 'V_LV100', 'theta_LV100', 'V_MV100', 'theta_MV100', 'omega_GRID', 'i_d_GRID', 'i_q_GRID', 'p_s_GRID', 'q_s_GRID', 'v_dc_v_LV001', 'i_sq_ref_LV001', 'i_sd_ref_LV001', 'p_s_LV001', 'q_s_LV001', 'v_dc_v_LV002', 'i_sq_ref_LV002', 'i_sd_ref_LV002', 'p_s_LV002', 'q_s_LV002', 'v_dc_v_LV003', 'i_sq_ref_LV003', 'i_sd_ref_LV003', 'p_s_LV003', 'q_s_LV003', 'v_dc_v_LV004', 'i_sq_ref_LV004', 'i_sd_ref_LV004', 'p_s_LV004', 'q_s_LV004', 'v_dc_v_LV005', 'i_sq_ref_LV005', 'i_sd_ref_LV005', 'p_s_LV005', 'q_s_LV005', 'v_dc_v_LV006', 'i_sq_ref_LV006', 'i_sd_ref_LV006', 'p_s_LV006', 'q_s_LV006', 'v_dc_v_LV007', 'i_sq_ref_LV007', 'i_sd_ref_LV007', 'p_s_LV007', 'q_s_LV007', 'v_dc_v_LV008', 'i_sq_ref_LV008', 'i_sd_ref_LV008', 'p_s_LV008', 'q_s_LV008', 'v_dc_v_LV009', 'i_sq_ref_LV009', 'i_sd_ref_LV009', 'p_s_LV009', 'q_s_LV009', 'v_dc_v_LV010', 'i_sq_ref_LV010', 'i_sd_ref_LV010', 'p_s_LV010', 'q_s_LV010', 'v_dc_v_LV011', 'i_sq_ref_LV011', 'i_sd_ref_LV011', 'p_s_LV011', 'q_s_LV011', 'v_dc_v_LV012', 'i_sq_ref_LV012', 'i_sd_ref_LV012', 'p_s_LV012', 'q_s_LV012', 'v_dc_v_LV013', 'i_sq_ref_LV013', 'i_sd_ref_LV013', 'p_s_LV013', 'q_s_LV013', 'v_dc_v_LV014', 'i_sq_ref_LV014', 'i_sd_ref_LV014', 'p_s_LV014', 'q_s_LV014', 'v_dc_v_LV015', 'i_sq_ref_LV015', 'i_sd_ref_LV015', 'p_s_LV015', 'q_s_LV015', 'v_dc_v_LV016', 'i_sq_ref_LV016', 'i_sd_ref_LV016', 'p_s_LV016', 'q_s_LV016', 'v_dc_v_LV017', 'i_sq_ref_LV017', 'i_sd_ref_LV017', 'p_s_LV017', 'q_s_LV017', 'v_dc_v_LV018', 'i_sq_ref_LV018', 'i_sd_ref_LV018', 'p_s_LV018', 'q_s_LV018', 'v_dc_v_LV019', 'i_sq_ref_LV019', 'i_sd_ref_LV019', 'p_s_LV019', 'q_s_LV019', 'v_dc_v_LV020', 'i_sq_ref_LV020', 'i_sd_ref_LV020', 'p_s_LV020', 'q_s_LV020', 'v_dc_v_LV021', 'i_sq_ref_LV021', 'i_sd_ref_LV021', 'p_s_LV021', 'q_s_LV021', 'v_dc_v_LV022', 'i_sq_ref_LV022', 'i_sd_ref_LV022', 'p_s_LV022', 'q_s_LV022', 'v_dc_v_LV023', 'i_sq_ref_LV023', 'i_sd_ref_LV023', 'p_s_LV023', 'q_s_LV023', 'v_dc_v_LV024', 'i_sq_ref_LV024', 'i_sd_ref_LV024', 'p_s_LV024', 'q_s_LV024', 'v_dc_v_LV025', 'i_sq_ref_LV025', 'i_sd_ref_LV025', 'p_s_LV025', 'q_s_LV025', 'v_dc_v_LV026', 'i_sq_ref_LV026', 'i_sd_ref_LV026', 'p_s_LV026', 'q_s_LV026', 'v_dc_v_LV027', 'i_sq_ref_LV027', 'i_sd_ref_LV027', 'p_s_LV027', 'q_s_LV027', 'v_dc_v_LV028', 'i_sq_ref_LV028', 'i_sd_ref_LV028', 'p_s_LV028', 'q_s_LV028', 'v_dc_v_LV029', 'i_sq_ref_LV029', 'i_sd_ref_LV029', 'p_s_LV029', 'q_s_LV029', 'v_dc_v_LV030', 'i_sq_ref_LV030', 'i_sd_ref_LV030', 'p_s_LV030', 'q_s_LV030', 'v_dc_v_LV031', 'i_sq_ref_LV031', 'i_sd_ref_LV031', 'p_s_LV031', 'q_s_LV031', 'v_dc_v_LV032', 'i_sq_ref_LV032', 'i_sd_ref_LV032', 'p_s_LV032', 'q_s_LV032', 'v_dc_v_LV033', 'i_sq_ref_LV033', 'i_sd_ref_LV033', 'p_s_LV033', 'q_s_LV033', 'v_dc_v_LV034', 'i_sq_ref_LV034', 'i_sd_ref_LV034', 'p_s_LV034', 'q_s_LV034', 'v_dc_v_LV035', 'i_sq_ref_LV035', 'i_sd_ref_LV035', 'p_s_LV035', 'q_s_LV035', 'v_dc_v_LV036', 'i_sq_ref_LV036', 'i_sd_ref_LV036', 'p_s_LV036', 'q_s_LV036', 'v_dc_v_LV037', 'i_sq_ref_LV037', 'i_sd_ref_LV037', 'p_s_LV037', 'q_s_LV037', 'v_dc_v_LV038', 'i_sq_ref_LV038', 'i_sd_ref_LV038', 'p_s_LV038', 'q_s_LV038', 'v_dc_v_LV039', 'i_sq_ref_LV039', 'i_sd_ref_LV039', 'p_s_LV039', 'q_s_LV039', 'v_dc_v_LV040', 'i_sq_ref_LV040', 'i_sd_ref_LV040', 'p_s_LV040', 'q_s_LV040', 'v_dc_v_LV041', 'i_sq_ref_LV041', 'i_sd_ref_LV041', 'p_s_LV041', 'q_s_LV041', 'v_dc_v_LV042', 'i_sq_ref_LV042', 'i_sd_ref_LV042', 'p_s_LV042', 'q_s_LV042', 'v_dc_v_LV043', 'i_sq_ref_LV043', 'i_sd_ref_LV043', 'p_s_LV043', 'q_s_LV043', 'v_dc_v_LV044', 'i_sq_ref_LV044', 'i_sd_ref_LV044', 'p_s_LV044', 'q_s_LV044', 'v_dc_v_LV045', 'i_sq_ref_LV045', 'i_sd_ref_LV045', 'p_s_LV045', 'q_s_LV045', 'v_dc_v_LV046', 'i_sq_ref_LV046', 'i_sd_ref_LV046', 'p_s_LV046', 'q_s_LV046', 'v_dc_v_LV047', 'i_sq_ref_LV047', 'i_sd_ref_LV047', 'p_s_LV047', 'q_s_LV047', 'v_dc_v_LV048', 'i_sq_ref_LV048', 'i_sd_ref_LV048', 'p_s_LV048', 'q_s_LV048', 'v_dc_v_LV049', 'i_sq_ref_LV049', 'i_sd_ref_LV049', 'p_s_LV049', 'q_s_LV049', 'v_dc_v_LV050', 'i_sq_ref_LV050', 'i_sd_ref_LV050', 'p_s_LV050', 'q_s_LV050', 'v_dc_v_LV051', 'i_sq_ref_LV051', 'i_sd_ref_LV051', 'p_s_LV051', 'q_s_LV051', 'v_dc_v_LV052', 'i_sq_ref_LV052', 'i_sd_ref_LV052', 'p_s_LV052', 'q_s_LV052', 'v_dc_v_LV053', 'i_sq_ref_LV053', 'i_sd_ref_LV053', 'p_s_LV053', 'q_s_LV053', 'v_dc_v_LV054', 'i_sq_ref_LV054', 'i_sd_ref_LV054', 'p_s_LV054', 'q_s_LV054', 'v_dc_v_LV055', 'i_sq_ref_LV055', 'i_sd_ref_LV055', 'p_s_LV055', 'q_s_LV055', 'v_dc_v_LV056', 'i_sq_ref_LV056', 'i_sd_ref_LV056', 'p_s_LV056', 'q_s_LV056', 'v_dc_v_LV057', 'i_sq_ref_LV057', 'i_sd_ref_LV057', 'p_s_LV057', 'q_s_LV057', 'v_dc_v_LV058', 'i_sq_ref_LV058', 'i_sd_ref_LV058', 'p_s_LV058', 'q_s_LV058', 'v_dc_v_LV059', 'i_sq_ref_LV059', 'i_sd_ref_LV059', 'p_s_LV059', 'q_s_LV059', 'v_dc_v_LV060', 'i_sq_ref_LV060', 'i_sd_ref_LV060', 'p_s_LV060', 'q_s_LV060', 'v_dc_v_LV061', 'i_sq_ref_LV061', 'i_sd_ref_LV061', 'p_s_LV061', 'q_s_LV061', 'v_dc_v_LV062', 'i_sq_ref_LV062', 'i_sd_ref_LV062', 'p_s_LV062', 'q_s_LV062', 'v_dc_v_LV063', 'i_sq_ref_LV063', 'i_sd_ref_LV063', 'p_s_LV063', 'q_s_LV063', 'v_dc_v_LV064', 'i_sq_ref_LV064', 'i_sd_ref_LV064', 'p_s_LV064', 'q_s_LV064', 'v_dc_v_LV065', 'i_sq_ref_LV065', 'i_sd_ref_LV065', 'p_s_LV065', 'q_s_LV065', 'v_dc_v_LV066', 'i_sq_ref_LV066', 'i_sd_ref_LV066', 'p_s_LV066', 'q_s_LV066', 'v_dc_v_LV067', 'i_sq_ref_LV067', 'i_sd_ref_LV067', 'p_s_LV067', 'q_s_LV067', 'v_dc_v_LV068', 'i_sq_ref_LV068', 'i_sd_ref_LV068', 'p_s_LV068', 'q_s_LV068', 'v_dc_v_LV069', 'i_sq_ref_LV069', 'i_sd_ref_LV069', 'p_s_LV069', 'q_s_LV069', 'v_dc_v_LV070', 'i_sq_ref_LV070', 'i_sd_ref_LV070', 'p_s_LV070', 'q_s_LV070', 'v_dc_v_LV071', 'i_sq_ref_LV071', 'i_sd_ref_LV071', 'p_s_LV071', 'q_s_LV071', 'v_dc_v_LV072', 'i_sq_ref_LV072', 'i_sd_ref_LV072', 'p_s_LV072', 'q_s_LV072', 'v_dc_v_LV073', 'i_sq_ref_LV073', 'i_sd_ref_LV073', 'p_s_LV073', 'q_s_LV073', 'v_dc_v_LV074', 'i_sq_ref_LV074', 'i_sd_ref_LV074', 'p_s_LV074', 'q_s_LV074', 'v_dc_v_LV075', 'i_sq_ref_LV075', 'i_sd_ref_LV075', 'p_s_LV075', 'q_s_LV075', 'v_dc_v_LV076', 'i_sq_ref_LV076', 'i_sd_ref_LV076', 'p_s_LV076', 'q_s_LV076', 'v_dc_v_LV077', 'i_sq_ref_LV077', 'i_sd_ref_LV077', 'p_s_LV077', 'q_s_LV077', 'v_dc_v_LV078', 'i_sq_ref_LV078', 'i_sd_ref_LV078', 'p_s_LV078', 'q_s_LV078', 'v_dc_v_LV079', 'i_sq_ref_LV079', 'i_sd_ref_LV079', 'p_s_LV079', 'q_s_LV079', 'v_dc_v_LV080', 'i_sq_ref_LV080', 'i_sd_ref_LV080', 'p_s_LV080', 'q_s_LV080', 'v_dc_v_LV081', 'i_sq_ref_LV081', 'i_sd_ref_LV081', 'p_s_LV081', 'q_s_LV081', 'v_dc_v_LV082', 'i_sq_ref_LV082', 'i_sd_ref_LV082', 'p_s_LV082', 'q_s_LV082', 'v_dc_v_LV083', 'i_sq_ref_LV083', 'i_sd_ref_LV083', 'p_s_LV083', 'q_s_LV083', 'v_dc_v_LV084', 'i_sq_ref_LV084', 'i_sd_ref_LV084', 'p_s_LV084', 'q_s_LV084', 'v_dc_v_LV085', 'i_sq_ref_LV085', 'i_sd_ref_LV085', 'p_s_LV085', 'q_s_LV085', 'v_dc_v_LV086', 'i_sq_ref_LV086', 'i_sd_ref_LV086', 'p_s_LV086', 'q_s_LV086', 'v_dc_v_LV087', 'i_sq_ref_LV087', 'i_sd_ref_LV087', 'p_s_LV087', 'q_s_LV087', 'v_dc_v_LV088', 'i_sq_ref_LV088', 'i_sd_ref_LV088', 'p_s_LV088', 'q_s_LV088', 'v_dc_v_LV089', 'i_sq_ref_LV089', 'i_sd_ref_LV089', 'p_s_LV089', 'q_s_LV089', 'v_dc_v_LV090', 'i_sq_ref_LV090', 'i_sd_ref_LV090', 'p_s_LV090', 'q_s_LV090', 'v_dc_v_LV091', 'i_sq_ref_LV091', 'i_sd_ref_LV091', 'p_s_LV091', 'q_s_LV091', 'v_dc_v_LV092', 'i_sq_ref_LV092', 'i_sd_ref_LV092', 'p_s_LV092', 'q_s_LV092', 'v_dc_v_LV093', 'i_sq_ref_LV093', 'i_sd_ref_LV093', 'p_s_LV093', 'q_s_LV093', 'v_dc_v_LV094', 'i_sq_ref_LV094', 'i_sd_ref_LV094', 'p_s_LV094', 'q_s_LV094', 'v_dc_v_LV095', 'i_sq_ref_LV095', 'i_sd_ref_LV095', 'p_s_LV095', 'q_s_LV095', 'v_dc_v_LV096', 'i_sq_ref_LV096', 'i_sd_ref_LV096', 'p_s_LV096', 'q_s_LV096', 'v_dc_v_LV097', 'i_sq_ref_LV097', 'i_sd_ref_LV097', 'p_s_LV097', 'q_s_LV097', 'v_dc_v_LV098', 'i_sq_ref_LV098', 'i_sd_ref_LV098', 'p_s_LV098', 'q_s_LV098', 'v_dc_v_LV099', 'i_sq_ref_LV099', 'i_sd_ref_LV099', 'p_s_LV099', 'q_s_LV099', 'v_dc_v_LV100', 'i_sq_ref_LV100', 'i_sd_ref_LV100', 'p_s_LV100', 'q_s_LV100', 'omega_coi', 'p_agc'] 
        self.xy_ini_list = self.x_list + self.y_ini_list 
        self.t = 0.0
        self.it = 0
        self.it_store = 0
        self.xy_prev = np.zeros((self.N_x+self.N_y,1))
        self.initialization_tol = 1e-6
        self.N_u = len(self.inputs_run_list) 
        self.sopt_root_method='hybr'
        self.sopt_root_jac=True
        self.u_ini_list = self.inputs_ini_list
        self.u_ini_values_list = self.inputs_ini_values_list
        self.u_run_list = self.inputs_run_list
        self.u_run_values_list = self.inputs_run_values_list
        self.N_u = len(self.u_run_list)
        self.u_ini = np.array(self.inputs_ini_values_list, dtype=np.float64)
        self.p = np.array(self.params_values_list, dtype=np.float64)
        self.xy_0 = np.zeros((self.N_x+self.N_y,),dtype=np.float64)
        self.xy = np.zeros((self.N_x+self.N_y,),dtype=np.float64)
        self.z = np.zeros((self.N_z,),dtype=np.float64)
        
        # numerical elements of jacobians computing:
        x = self.xy[:self.N_x]
        y = self.xy[self.N_x:]
        
        self.yini2urun = list(set(self.u_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.u_ini_list)))
        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store) 
        self.u_run = np.array(self.u_run_values_list,dtype=np.float64)
 
        ## jac_ini
        self.jac_ini = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        if self.sparse:
            self.sp_jac_ini_ia, self.sp_jac_ini_ja, self.sp_jac_ini_nia, self.sp_jac_ini_nja = sp_jac_ini_vectors()
            data = np.array(self.sp_jac_ini_ia,dtype=np.float64)
        #self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))
           
        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, f'./pv_100_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz(f'./{self.matrices_folder}/pv_100_sp_jac_ini_num.npz')
            
            
        self.jac_ini = self.sp_jac_ini.toarray()

        #self.J_ini_d = np.array(self.sp_jac_ini_ia)*0.0
        #self.J_ini_i = np.array(self.sp_jac_ini_ia)
        #self.J_ini_p = np.array(self.sp_jac_ini_ja)
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        if self.sparse:
            sp_jac_ini_eval(self.sp_jac_ini.data,x,y,self.u_ini,self.p,self.Dt) 
            self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 100,1e-10,'basic'       


        ## jac_run
        self.jac_run = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        if self.sparse:
            self.sp_jac_run_ia, self.sp_jac_run_ja, self.sp_jac_run_nia, self.sp_jac_run_nja = sp_jac_run_vectors()
            data = np.array(self.sp_jac_run_ia,dtype=np.float64)

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './pv_100_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz(f'./{self.matrices_folder}/pv_100_sp_jac_run_num.npz')
        self.jac_run = self.sp_jac_run.toarray()            

        if self.sparse:           
            self.J_run_d = np.array(self.sp_jac_run_ia)*0.0
            self.J_run_i = np.array(self.sp_jac_run_ia)
            self.J_run_p = np.array(self.sp_jac_run_ja)
        de_jac_run_eval(self.jac_run,x,y,self.u_run,self.p,self.Dt)

        if self.sparse:
            sp_jac_run_eval(self.J_run_d,x,y,self.u_run,self.p,self.Dt)
        
        ## jac_trap
        self.jac_trap = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))

        if self.sparse:

            self.sp_jac_trap_ia, self.sp_jac_trap_ja, self.sp_jac_trap_nia, self.sp_jac_trap_nja = sp_jac_trap_vectors()
            data = np.array(self.sp_jac_trap_ia,dtype=np.float64)
            #self.sp_jac_trap = sspa.csr_matrix((data, self.sp_jac_trap_ia, self.sp_jac_trap_ja), shape=(self.sp_jac_trap_nia,self.sp_jac_trap_nja))
        
        

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './pv_100_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz(f'./{self.matrices_folder}/pv_100_sp_jac_trap_num.npz')
            

        self.jac_trap = self.sp_jac_trap.toarray()
        
        #self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        #self.J_trap_i = np.array(self.sp_jac_trap_ia)
        #self.J_trap_p = np.array(self.sp_jac_trap_ja)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
        if self.sparse:
            sp_jac_trap_eval(self.sp_jac_trap.data,x,y,self.u_run,self.p,self.Dt)
            self.fill_factor_trap,self.drop_tol_trap,self.drop_rule_trap = 100,1e-10,'basic' 
    

        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp= 50,1e-8,1.0
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.0

        #self.sp_Fu_run = sspa.load_npz(f'./{self.matrices_folder}/pv_100_Fu_run_num.npz')
        #self.sp_Gu_run = sspa.load_npz(f'./{self.matrices_folder}/pv_100_Gu_run_num.npz')
        #self.sp_Hx_run = sspa.load_npz(f'./{self.matrices_folder}/pv_100_Hx_run_num.npz')
        #self.sp_Hy_run = sspa.load_npz(f'./{self.matrices_folder}/pv_100_Hy_run_num.npz')
        #self.sp_Hu_run = sspa.load_npz(f'./{self.matrices_folder}/pv_100_Hu_run_num.npz')        
        
        self.ss_solver = 2
        self.lsolver = 2
 
        



        
    def update(self):

        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store)
        
    def ss_ini(self):

        xy_ini,it = sstate(self.xy_0,self.u_ini,self.p,self.jac_ini,self.N_x,self.N_y)
        self.xy_ini = xy_ini
        self.N_iters = it
        
        return xy_ini
    
    # def ini(self,up_dict,xy_0={}):

    #     for item in up_dict:
    #         self.set_value(item,up_dict[item])
            
    #     self.xy_ini = self.ss_ini()
    #     self.ini2run()
    #     jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
    #     jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
    def jac_run_eval(self):
        de_jac_run_eval(self.jac_run,self.x,self.y_run,self.u_run,self.p,self.Dt)
      
    
    def run(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z
        
        t,it,it_store,xy = daesolver(t,t_end,it,it_store,xy,u,p,z,
                                  self.jac_trap,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  max_it=self.max_it,itol=self.itol,store=self.store)
        
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z
 
    def runsp(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        
        t,it,it_store,xy = daesolver_sp(t,t_end,it,it_store,xy,u,p,
                                  self.sp_jac_trap,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  max_it=50,itol=1e-8,store=1)
        
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        
    def post(self):
        
        self.Time = self.Time[:self.it_store]
        self.X = self.X[:self.it_store]
        self.Y = self.Y[:self.it_store]
        self.Z = self.Z[:self.it_store]
        
    def ini2run(self):
        
        ## y_ini to y_run
        self.y_ini = self.xy_ini[self.N_x:]
        self.y_run = np.copy(self.y_ini)
        self.u_run = np.copy(self.u_ini)
        
        ## y_ini to u_run
        for item in self.yini2urun:
            self.u_run[self.u_run_list.index(item)] = self.y_ini[self.y_ini_list.index(item)]
                
        ## u_ini to y_run
        for item in self.uini2yrun:
            self.y_run[self.y_run_list.index(item)] = self.u_ini[self.u_ini_list.index(item)]
            
        
        self.x = self.xy_ini[:self.N_x]
        self.xy[:self.N_x] = self.x
        self.xy[self.N_x:] = self.y_run
        c_h_eval(self.z,self.x,self.y_run,self.u_ini,self.p,self.Dt)
        

        
    def get_value(self,name):
        
        if name in self.inputs_run_list:
            value = self.u_run[self.inputs_run_list.index(name)]
            return value
            
        if name in self.x_list:
            idx = self.x_list.index(name)
            value = self.xy[idx]
            return value
            
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.xy[self.N_x+idy]
            return value
        
        if name in self.params_list:
            idp = self.params_list.index(name)
            value = self.p[idp]
            return value
            
        if name in self.outputs_list:
            idz = self.outputs_list.index(name)
            value = self.z[idz]
            return value

    def get_values(self,name):
        if name in self.x_list:
            values = self.X[:,self.x_list.index(name)]
        if name in self.y_run_list:
            values = self.Y[:,self.y_run_list.index(name)]
        if name in self.outputs_list:
            values = self.Z[:,self.outputs_list.index(name)]
                        
        return values

    def get_mvalue(self,names):
        '''

        Parameters
        ----------
        names : list
            list of variables names to return each value.

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        mvalue = []
        for name in names:
            mvalue += [self.get_value(name)]
                        
        return mvalue
    
    def set_value(self,name_,value):
        if name_ in self.inputs_ini_list or name_ in self.inputs_run_list:
            if name_ in self.inputs_ini_list:
                self.u_ini[self.inputs_ini_list.index(name_)] = value
            if name_ in self.inputs_run_list:
                self.u_run[self.inputs_run_list.index(name_)] = value
            return
        elif name_ in self.params_list:
            self.p[self.params_list.index(name_)] = value
            return
        else:
            print(f'Input or parameter {name_} not found.')
 
    def report_x(self,value_format='5.2f'):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')

    def report_y(self,value_format='5.2f'):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')
            
    def report_u(self,value_format='5.2f'):
        for item in self.inputs_run_list:
            print(f'{item:5s} ={self.get_value(item):{value_format}}')

    def report_z(self,value_format='5.2f'):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')

    def report_params(self,value_format='5.2f'):
        for item in self.params_list:
            print(f'{item:5s} ={self.get_value(item):{value_format}}')
            
    def ini(self,up_dict,xy_0={}):
        '''
        Find the steady state of the initialization problem:
            
               0 = f(x,y,u,p) 
               0 = g(x,y,u,p) 

        Parameters
        ----------
        up_dict : dict
            dictionary with all the parameters p and inputs u new values.
        xy_0: if scalar, all the x and y values initial guess are set to the scalar.
              if dict, the initial guesses are applied for the x and y that are in the dictionary
              if string, the initial guess considers a json file with the x and y names and their initial values

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        
        self.it = 0
        self.it_store = 0
        self.t = 0.0
    
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        if type(xy_0) == dict:
            xy_0_dict = xy_0
            self.dict2xy0(xy_0_dict)
            
        if type(xy_0) == str:
            if xy_0 == 'eval':
                N_x = self.N_x
                self.xy_0_new = np.copy(self.xy_0)*0
                xy0_eval(self.xy_0_new[:N_x],self.xy_0_new[N_x:],self.u_ini,self.p)
                self.xy_0_evaluated = np.copy(self.xy_0_new)
                self.xy_0 = np.copy(self.xy_0_new)
            else:
                self.load_xy_0(file_name = xy_0)
                
        if type(xy_0) == float or type(xy_0) == int:
            self.xy_0 = np.ones(self.N_x+self.N_y,dtype=np.float64)*xy_0

        xy_ini,it = sstate(self.xy_0,self.u_ini,self.p,
                           self.jac_ini,
                           self.N_x,self.N_y,
                           max_it=self.max_it,tol=self.itol)
        
        if it < self.max_it-1:
            
            self.xy_ini = xy_ini
            self.N_iters = it

            self.ini2run()
            
            self.ini_convergence = True
            
        if it >= self.max_it-1:
            print(f'Maximum number of iterations (max_it = {self.max_it}) reached without convergence.')
            self.ini_convergence = False
            
        return self.ini_convergence
            
        


    
    def dict2xy0(self,xy_0_dict):
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item) + self.N_x] = xy_0_dict[item]
        
    
    def save_xy_0(self,file_name = 'xy_0.json'):
        xy_0_dict = {}
        for item in self.x_list:
            xy_0_dict.update({item:self.get_value(item)})
        for item in self.y_ini_list:
            xy_0_dict.update({item:self.get_value(item)})
    
        xy_0_str = json.dumps(xy_0_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(xy_0_str)
    
    def load_xy_0(self,file_name = 'xy_0.json'):
        with open(file_name) as fobj:
            xy_0_str = fobj.read()
        xy_0_dict = json.loads(xy_0_str)
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item)+self.N_x] = xy_0_dict[item]            

    def load_params(self,data_input):
    
        if type(data_input) == str:
            json_file = data_input
            self.json_file = json_file
            self.json_data = open(json_file).read().replace("'",'"')
            data = json.loads(self.json_data)
        elif type(data_input) == dict:
            data = data_input
    
        self.data = data
        for item in self.data:
            self.set_value(item, self.data[item])

    def save_params(self,file_name = 'parameters.json'):
        params_dict = {}
        for item in self.params_list:
            params_dict.update({item:self.get_value(item)})

        params_dict_str = json.dumps(params_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(params_dict_str)

    def save_inputs_ini(self,file_name = 'inputs_ini.json'):
        inputs_ini_dict = {}
        for item in self.inputs_ini_list:
            inputs_ini_dict.update({item:self.get_value(item)})

        inputs_ini_dict_str = json.dumps(inputs_ini_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(inputs_ini_dict_str)

    def eval_preconditioner_ini(self):
    
        sp_jac_ini_eval(self.sp_jac_ini.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        csc_sp_jac_ini = sspa.csc_matrix(self.sp_jac_ini)
        P_slu = spilu(csc_sp_jac_ini,
                  fill_factor=self.fill_factor_ini,
                  drop_tol=self.drop_tol_ini,
                  drop_rule = self.drop_rule_ini)
    
        self.P_slu = P_slu
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu)   
        self.P_d = P_d
        self.P_i = P_i
        self.P_p = P_p
    
        self.perm_r = perm_r
        self.perm_c = perm_c
            
    
    def eval_preconditioner_trap(self):
    
        sp_jac_trap_eval(self.sp_jac_trap.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        #self.sp_jac_trap.data = self.J_trap_d 
        
        csc_sp_jac_trap = sspa.csc_matrix(self.sp_jac_trap)


        P_slu_trap = spilu(csc_sp_jac_trap,
                          fill_factor=self.fill_factor_trap,
                          drop_tol=self.drop_tol_trap,
                          drop_rule = self.drop_rule_trap)
    
        self.P_slu_trap = P_slu_trap
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu_trap)   
        self.P_trap_d = P_d
        self.P_trap_i = P_i
        self.P_trap_p = P_p
    
        self.perm_trap_r = perm_r
        self.perm_trap_c = perm_c
        
    def sprun(self,t_end,up_dict):
        
        for item in up_dict:
            self.set_value(item,up_dict[item])
    
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z
        self.iparams_run = np.zeros(10,dtype=np.float64)
    
        t,it,it_store,xy = spdaesolver(t,t_end,it,it_store,xy,u,p,z,
                                  self.sp_jac_trap.data,self.sp_jac_trap.indices,self.sp_jac_trap.indptr,
                                  self.P_trap_d,self.P_trap_i,self.P_trap_p,self.perm_trap_r,self.perm_trap_c,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  self.iparams_run,
                                  max_it=self.max_it,itol=self.max_it,store=self.store,
                                  lmax_it=self.lmax_it,ltol=self.ltol,ldamp=self.ldamp,mode=self.mode,
                                  lsolver = self.lsolver)
    
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z

            
    def spini(self,up_dict,xy_0={}):
    
        self.it = 0
        self.it_store = 0
        self.t = 0.0
    
        for item in up_dict:
            self.set_value(item,up_dict[item])
    
        if type(xy_0) == dict:
            xy_0_dict = xy_0
            self.dict2xy0(xy_0_dict)
    
        if type(xy_0) == str:
            if xy_0 == 'eval':
                N_x = self.N_x
                self.xy_0_new = np.copy(self.xy_0)*0
                xy0_eval(self.xy_0_new[:N_x],self.xy_0_new[N_x:],self.u_ini,self.p)
                self.xy_0_evaluated = np.copy(self.xy_0_new)
                self.xy_0 = np.copy(self.xy_0_new)
            else:
                self.load_xy_0(file_name = xy_0)

        self.xy_ini = self.spss_ini()


        if self.N_iters < self.max_it:
            
            self.ini2run()           
            self.ini_convergence = True
            
        if self.N_iters >= self.max_it:
            print(f'Maximum number of iterations (max_it = {self.max_it}) reached without convergence.')
            self.ini_convergence = False
            
        #jac_run_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        #jac_run_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
        return self.ini_convergence

        
    def spss_ini(self):
        J_d,J_i,J_p = csr2pydae(self.sp_jac_ini)
        
        xy_ini,it,iparams = spsstate(self.xy,self.u_ini,self.p,
                 self.sp_jac_ini.data,self.sp_jac_ini.indices,self.sp_jac_ini.indptr,
                 self.P_d,self.P_i,self.P_p,self.perm_r,self.perm_c,
                 self.N_x,self.N_y,
                 max_it=self.max_it,tol=self.itol,
                 lmax_it=self.lmax_it_ini,
                 ltol=self.ltol_ini,
                 ldamp=self.ldamp,solver=self.ss_solver)

 
        self.xy_ini = xy_ini
        self.N_iters = it
        self.iparams = iparams
    
        return xy_ini

    #def import_cffi(self):
        

    def eval_jac_u2z(self):

        '''

        0 =   J_run * xy + FG_u * u
        z = Hxy_run * xy + H_u * u

        xy = -1/J_run * FG_u * u
        z = -Hxy_run/J_run * FG_u * u + H_u * u
        z = (-Hxy_run/J_run * FG_u + H_u ) * u 
        '''
        
        sp_Fu_run_eval(self.sp_Fu_run.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_Gu_run_eval(self.sp_Gu_run.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_H_jacs_run_eval(self.sp_Hx_run.data,
                        self.sp_Hy_run.data,
                        self.sp_Hu_run.data,
                        self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_jac_run = self.sp_jac_run
        sp_jac_run_eval(sp_jac_run.data,
                        self.x,self.y_run,
                        self.u_run,self.p,
                        self.Dt)



        Hxy_run = sspa.bmat([[self.sp_Hx_run,self.sp_Hy_run]])
        FGu_run = sspa.bmat([[self.sp_Fu_run],[self.sp_Gu_run]])
        

        #((sspa.linalg.spsolve(s.sp_jac_ini,-Hxy_run)) @ FGu_run + sp_Hu_run )@s.u_ini

        self.jac_u2z = Hxy_run @ sspa.linalg.spsolve(self.sp_jac_run,-FGu_run) + self.sp_Hu_run  
        
        
    def step(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])

        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z

        t,it,xy = daestep(t,t_end,it,
                          xy,u,p,z,
                          self.jac_trap,
                          self.iters,
                          self.Dt,
                          self.N_x,
                          self.N_y,
                          self.N_z,
                          max_it=self.max_it,itol=self.itol,store=self.store)

        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z
           
            
    def save_run(self,file_name):
        np.savez(file_name,Time=self.Time,
             X=self.X,Y=self.Y,Z=self.Z,
             x_list = self.x_list,
             y_ini_list = self.y_ini_list,
             y_run_list = self.y_run_list,
             u_ini_list=self.u_ini_list,
             u_run_list=self.u_run_list,  
             z_list=self.outputs_list, 
            )
        
    def load_run(self,file_name):
        data = np.load(f'{file_name}.npz')
        self.Time = data['Time']
        self.X = data['X']
        self.Y = data['Y']
        self.Z = data['Z']
        self.x_list = list(data['x_list'] )
        self.y_run_list = list(data['y_run_list'] )
        self.outputs_list = list(data['z_list'] )
        
    def full_jacs_eval(self):
        N_x = self.N_x
        N_y = self.N_y
        N_xy = N_x + N_y
    
        sp_jac_run = self.sp_jac_run
        sp_Fu = self.sp_Fu_run
        sp_Gu = self.sp_Gu_run
        sp_Hx = self.sp_Hx_run
        sp_Hy = self.sp_Hy_run
        sp_Hu = self.sp_Hu_run
        
        x = self.xy[0:N_x]
        y = self.xy[N_x:]
        u = self.u_run
        p = self.p
        Dt = self.Dt
    
        sp_jac_run_eval(sp_jac_run.data,x,y,u,p,Dt)
        
        self.Fx = sp_jac_run[0:N_x,0:N_x]
        self.Fy = sp_jac_run[ 0:N_x,N_x:]
        self.Gx = sp_jac_run[ N_x:,0:N_x]
        self.Gy = sp_jac_run[ N_x:, N_x:]
        
        sp_Fu_run_eval(sp_Fu.data,x,y,u,p,Dt)
        sp_Gu_run_eval(sp_Gu.data,x,y,u,p,Dt)
        sp_H_jacs_run_eval(sp_Hx.data,sp_Hy.data,sp_Hu.data,x,y,u,p,Dt)
        
        self.Fu = sp_Fu
        self.Gu = sp_Gu
        self.Hx = sp_Hx
        self.Hy = sp_Hy
        self.Hu = sp_Hu


@numba.njit() 
def daestep(t,t_end,it,xy,u,p,z,jac_trap,iters,Dt,N_x,N_y,N_z,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    #h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = np.linalg.solve(-jac_trap,fg_i) 

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
    return t,it,xy


def daesolver_sp(t,t_end,it,it_store,xy,u,p,sp_jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 

    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    sp_jac_trap_eval_up(sp_jac_trap.data,x,y,u,p,Dt,xyup=1)
    
    if it == 0:
        f_run_eval(f,x,y,u,p)
        h_eval(h,x,y,u,p)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f,x,y,u,p)
        g_run_eval(g,x,y,u,p)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f,x,y,u,p)
            g_run_eval(g,x,y,u,p)
            sp_jac_trap_eval(sp_jac_trap.data,x,y,u,p,Dt,xyup=1)            

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = spsolve(sp_jac_trap,-fg_i) 

            x = x + Dxy_i[:N_x]
            y = y + Dxy_i[N_x:]              

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(h,x,y,u,p)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = h
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy




@numba.njit()
def sprichardson(A_d,A_i,A_p,b,P_d,P_i,P_p,perm_r,perm_c,x,iparams,damp=1.0,max_it=100,tol=1e-3):
    N_A = A_p.shape[0]-1
    f = np.zeros(N_A)
    for it in range(max_it):
        spMvmul(N_A,A_d,A_i,A_p,x,f) 
        f -= b                          # A@x-b
        x = x - damp*splu_solve(P_d,P_i,P_p,perm_r,perm_c,f)   
        if np.linalg.norm(f,2) < tol: break
    iparams[0] = it
    return x
    
@numba.njit()
def spconjgradm(A_d,A_i,A_p,b,P_d,P_i,P_p,perm_r,perm_c,x,iparams,max_it=100,tol=1e-3, damp=None):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    preconditioned conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A_d,A_i,A_p : sparse matrix 
        components in CRS form A_d = A_crs.data, A_i = A_crs.indices, A_p = A_crs.indptr.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    P_d,P_i,P_p,perm_r,perm_c: preconditioner LU matrix
        components in scipy.spilu form P_d,P_i,P_p,perm_r,perm_c = slu2pydae(M)
        with M = scipy.sparse.linalg.spilu(A_csc) 

    """  
    N   = len(b)
    Ax  = np.zeros(N)
    Ap  = np.zeros(N)
    App = np.zeros(N)
    pAp = np.zeros(N)
    z   = np.zeros(N)
    
    spMvmul(N,A_d,A_i,A_p,x,Ax)
    r = -(Ax - b)
    z = splu_solve(P_d,P_i,P_p,perm_r,perm_c,r) #z = M.solve(r)
    p = z
    zsold = 0.0
    for it in range(N):  # zsold = np.dot(np.transpose(z), z)
        zsold += z[it]*z[it]
    for i in range(max_it):
        spMvmul(N,A_d,A_i,A_p,p,App)  # #App = np.dot(A, p)
        Ap = splu_solve(P_d,P_i,P_p,perm_r,perm_c,App) #Ap = M.solve(App)
        pAp = 0.0
        for it in range(N):
            pAp += p[it]*Ap[it]

        alpha = zsold / pAp
        x = x + alpha*p
        z = z - alpha*Ap
        zz = 0.0
        for it in range(N):  # z.T@z
            zz += z[it]*z[it]
        zsnew = zz
        if np.sqrt(zsnew) < tol:
            break
            
        p = z + (zsnew/zsold)*p
        zsold = zsnew
    iparams[0] = i

    return x


@numba.njit()
def spsstate(xy,u,p,
             J_d,J_i,J_p,
             P_d,P_i,P_p,perm_r,perm_c,
             N_x,N_y,
             max_it=50,tol=1e-8,
             lmax_it=20,ltol=1e-8,ldamp=1.0, solver=2):
    
   
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    iparams = np.array([0],dtype=np.int64)    
    
    f_c_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_c_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))
    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))

    #sp_jac_ini_num_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    sp_jac_ini_up_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    
    #sp_jac_ini_eval_up(J_d,x,y,u,p,0.0)

    Dxy = np.zeros(N_x + N_y)
    for it in range(max_it):
        
        x = xy[:N_x]
        y = xy[N_x:]   
       
        sp_jac_ini_xy_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)

        
        f_ini_eval(f_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        g_ini_eval(g_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        
        #f_ini_eval(f,x,y,u,p)
        #g_ini_eval(g,x,y,u,p)
        
        fg[:N_x] = f
        fg[N_x:] = g
        
        if solver==1:
               
            Dxy = sprichardson(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
   
        if solver==2:
            
            Dxy = spconjgradm(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
            
        xy += Dxy
        #if np.max(np.abs(fg))<tol: break
        if np.linalg.norm(fg,np.inf)<tol: break

    return xy,it,iparams


    
@numba.njit() 
def daesolver(t,t_end,it,it_store,xy,u,p,z,jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    #h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = z  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = np.linalg.solve(-jac_trap,fg_i) 

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = z
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy
    
@numba.njit() 
def spdaesolver(t,t_end,it,it_store,xy,u,p,z,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,
                iparams,
                max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0,lsolver=2):

    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    z = np.zeros((N_z),dtype=np.float64)
    Dxy_i_0 = np.zeros(N_x+N_y,dtype=np.float64) 
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))
    
    #sp_jac_trap_num_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    sp_jac_trap_up_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    sp_jac_trap_xy_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = z 

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            sp_jac_trap_xy_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            #Dxy_i = np.linalg.solve(-jac_trap,fg_i) 
            if lsolver == 1:
                Dxy_i = sprichardson(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
                                     Dxy_i_0,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
            if lsolver == 2:
                Dxy_i = spconjgradm(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
                                     Dxy_i_0,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)                

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = z
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy


@cuda.jit()
def ode_solve(x,u,p,f_run,u_idxs,z_i,z,sim):

    N_i,N_j,N_x,N_z,Dt = sim

    # index of thread on GPU:
    i = cuda.grid(1)

    if i < x.size:
        for j in range(N_j):
            f_run_eval(f_run[i,:],x[i,:],u[i,u_idxs[j],:],p[i,:])
            for k in range(N_x):
              x[i,k] +=  Dt*f_run[i,k]

            # outputs in time range
            #z[i,j] = u[i,idxs[j],0]
            z[i,j] = x[i,1]
        h_eval(z_i[i,:],x[i,:],u[i,u_idxs[j],:],p[i,:])
        
def csr2pydae(A_csr):
    '''
    From scipy CSR to the three vectors:
    
    - data
    - indices
    - indptr
    
    '''
    
    A_d = A_csr.data
    A_i = A_csr.indices
    A_p = A_csr.indptr
    
    return A_d,A_i,A_p
    
def slu2pydae(P_slu):
    '''
    From SupderLU matrix to the three vectors:
    
    - data
    - indices
    - indptr
    
    and the premutation vectors:
    
    - perm_r
    - perm_c
    
    '''
    N = P_slu.shape[0]
    #P_slu_full = P_slu.L.A - sspa.eye(N,format='csr') + P_slu.U.A
    P_slu_full = P_slu.L - sspa.eye(N,format='csc') + P_slu.U
    perm_r = P_slu.perm_r
    perm_c = P_slu.perm_c
    P_csr = sspa.csr_matrix(P_slu_full)
    
    P_d = P_csr.data
    P_i = P_csr.indices
    P_p = P_csr.indptr
    
    return P_d,P_i,P_p,perm_r,perm_c

@numba.njit(cache=True)
def spMvmul(N,A_data,A_indices,A_indptr,x,y):
    '''
    y = A @ x
    
    with A in sparse CRS form
    '''
    #y = np.zeros(x.shape[0])
    for i in range(N):
        y[i] = 0.0
        for j in range(A_indptr[i],A_indptr[i + 1]):
            y[i] = y[i] + A_data[j]*x[A_indices[j]]
            
            
@numba.njit(cache=True)
def splu_solve(LU_d,LU_i,LU_p,perm_r,perm_c,b):
    N = len(b)
    y = np.zeros(N)
    x = np.zeros(N)
    z = np.zeros(N)
    bp = np.zeros(N)
    
    for i in range(N): 
        bp[perm_r[i]] = b[i]
        
    for i in range(N): 
        y[i] = bp[i]
        for j in range(LU_p[i],LU_p[i+1]):
            if LU_i[j]>i-1: break
            y[i] -= LU_d[j] * y[LU_i[j]]

    for i in range(N-1,-1,-1): #(int i = N - 1; i >= 0; i--) 
        z[i] = y[i]
        den = 0.0
        for j in range(LU_p[i],LU_p[i+1]): #(int k = i + 1; k < N; k++)
            if LU_i[j] > i:
                z[i] -= LU_d[j] * z[LU_i[j]]
            if LU_i[j] == i: den = LU_d[j]
        z[i] = z[i]/den
 
    for i in range(N):
        x[i] = z[perm_c[i]]
        
    return x



@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_ini_eval(de_jac_ini,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    de_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(de_jac_ini))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_ini_num_eval(de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_ini_up_eval( de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_ini_xy_eval( de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_ini

@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_run_eval(de_jac_run,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_run = [[Fx_run, Fy_run],
               [Gx_run, Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_run : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    de_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(de_jac_run))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_run_num_eval(de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_run_up_eval( de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_run_xy_eval( de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_run

@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_trap_eval(de_jac_trap,x,y,u,p,Dt):   
    '''
    Computes the dense full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_trap : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
 
    Returns
    -------
    
    de_jac_trap : (N, N) array_like
                  Updated matrix.    
    
    '''
        
    de_jac_trap_ptr = ffi.from_buffer(np.ascontiguousarray(de_jac_trap))
    x_c_ptr = ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr = ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr = ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr = ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_trap_num_eval(de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_trap_up_eval( de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_trap_xy_eval( de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_trap


# @numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
# def sp_jac_run_eval(sp_jac_run,x,y,u,p,Dt):   
#     '''
#     Computes the sparse full trapezoidal jacobian:
    
#     jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
#                 [             Gx_run,         Gy_run]]
                
#     for the given x,y,u,p vectors and Dt time increment.
    
#     Parameters
#     ----------
#     sp_jac_trap : (Nnz,) array_like
#                   Input data.
#     x : (N_x,) array_like
#         Vector with dynamical states.
#     y : (N_y,) array_like
#         Vector with algebraic states (run problem).
#     u : (N_u,) array_like
#         Vector with inputs (run problem). 
#     p : (N_p,) array_like
#         Vector with parameters. 
        
#     with Nnz the number of non-zeros elements in the jacobian.
 
#     Returns
#     -------
    
#     sp_jac_trap : (Nnz,) array_like
#                   Updated matrix.    
    
#     '''        
#     sp_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_run))
#     x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
#     y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
#     u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
#     p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

#     sp_jac_run_num_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
#     sp_jac_run_up_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
#     sp_jac_run_xy_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
#     return sp_jac_run

# @numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
# def sp_jac_trap_eval(sp_jac_trap,x,y,u,p,Dt):   
#     '''
#     Computes the sparse full trapezoidal jacobian:
    
#     jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
#                 [             Gx_run,         Gy_run]]
                
#     for the given x,y,u,p vectors and Dt time increment.
    
#     Parameters
#     ----------
#     sp_jac_trap : (Nnz,) array_like
#                   Input data.
#     x : (N_x,) array_like
#         Vector with dynamical states.
#     y : (N_y,) array_like
#         Vector with algebraic states (run problem).
#     u : (N_u,) array_like
#         Vector with inputs (run problem). 
#     p : (N_p,) array_like
#         Vector with parameters. 
        
#     with Nnz the number of non-zeros elements in the jacobian.
 
#     Returns
#     -------
    
#     sp_jac_trap : (Nnz,) array_like
#                   Updated matrix.    
    
#     '''        
#     sp_jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_trap))
#     x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
#     y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
#     u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
#     p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

#     sp_jac_trap_num_eval(sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
#     sp_jac_trap_up_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
#     sp_jac_trap_xy_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
#     return sp_jac_trap

# @numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
# def sp_jac_ini_eval(sp_jac_ini,x,y,u,p,Dt):   
#     '''
#     Computes the SPARSE full initialization jacobian:
    
#     jac_ini = [[Fx_ini, Fy_ini],
#                [Gx_ini, Gy_ini]]
                
#     for the given x,y,u,p vectors and Dt time increment.
    
#     Parameters
#     ----------
#     de_jac_ini : (N, N) array_like
#                   Input data.
#     x : (N_x,) array_like
#         Vector with dynamical states.
#     y : (N_y,) array_like
#         Vector with algebraic states (ini problem).
#     u : (N_u,) array_like
#         Vector with inputs (ini problem). 
#     p : (N_p,) array_like
#         Vector with parameters. 
        
#     with N = N_x+N_y
 
#     Returns
#     -------
    
#     de_jac_ini : (N, N) array_like
#                   Updated matrix.    
    
#     '''
    
#     sp_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_ini))
#     x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
#     y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
#     u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
#     p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

#     sp_jac_ini_num_eval(sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
#     sp_jac_ini_up_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
#     sp_jac_ini_xy_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
#     return sp_jac_ini


@numba.njit()
def sstate(xy,u,p,jac_ini_ss,N_x,N_y,max_it=50,tol=1e-8):
    
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]

    f_c_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_c_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))
    jac_ini_ss_ptr=ffi.from_buffer(np.ascontiguousarray(jac_ini_ss))

    #de_jac_ini_num_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    de_jac_ini_up_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)

    for it in range(max_it):
        de_jac_ini_xy_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        f_ini_eval(f_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        g_ini_eval(g_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        fg[:N_x] = f
        fg[N_x:] = g
        xy += np.linalg.solve(jac_ini_ss,-fg)
        if np.max(np.abs(fg))<tol: break

    return xy,it


@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def c_h_eval(z,x,y,u,p,Dt):   
    '''
    Computes the SPARSE full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    z_c_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    h_eval(z_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return z

#@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_Fu_run_eval(jac,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    jac_ptr=ffi.from_buffer(np.ascontiguousarray(jac))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Fu_run_up_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Fu_run_xy_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    #return jac

#@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_Gu_run_eval(jac,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    jac_ptr=ffi.from_buffer(np.ascontiguousarray(jac))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Gu_run_up_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Gu_run_xy_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    #return jac

#@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_H_jacs_run_eval(H_x,H_y,H_u,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    H_x_ptr=ffi.from_buffer(np.ascontiguousarray(H_x))
    H_y_ptr=ffi.from_buffer(np.ascontiguousarray(H_y))
    H_u_ptr=ffi.from_buffer(np.ascontiguousarray(H_u))

    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Hx_run_up_eval( H_x_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hx_run_xy_eval( H_x_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hy_run_up_eval( H_y_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hy_run_xy_eval( H_y_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hu_run_up_eval( H_u_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hu_run_xy_eval( H_u_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)





def sp_jac_ini_vectors():

    sp_jac_ini_ia = [0, 410, 915, 1, 2, 3, 915, 4, 5, 6, 7, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97, 100, 101, 104, 105, 108, 109, 112, 113, 116, 117, 120, 121, 124, 125, 128, 129, 132, 133, 136, 137, 140, 141, 144, 145, 148, 149, 152, 153, 156, 157, 160, 161, 164, 165, 168, 169, 172, 173, 176, 177, 180, 181, 184, 185, 188, 189, 192, 193, 196, 197, 200, 201, 204, 205, 208, 209, 212, 213, 216, 217, 220, 221, 224, 225, 228, 229, 232, 233, 236, 237, 240, 241, 244, 245, 248, 249, 252, 253, 256, 257, 260, 261, 264, 265, 268, 269, 272, 273, 276, 277, 280, 281, 284, 285, 288, 289, 292, 293, 296, 297, 300, 301, 304, 305, 308, 309, 312, 313, 316, 317, 320, 321, 324, 325, 328, 329, 332, 333, 336, 337, 340, 341, 344, 345, 348, 349, 352, 353, 356, 357, 360, 361, 364, 365, 368, 369, 372, 373, 376, 377, 380, 381, 384, 385, 388, 389, 392, 393, 396, 397, 400, 401, 404, 405, 408, 409, 4, 5, 6, 7, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97, 100, 101, 104, 105, 108, 109, 112, 113, 116, 117, 120, 121, 124, 125, 128, 129, 132, 133, 136, 137, 140, 141, 144, 145, 148, 149, 152, 153, 156, 157, 160, 161, 164, 165, 168, 169, 172, 173, 176, 177, 180, 181, 184, 185, 188, 189, 192, 193, 196, 197, 200, 201, 204, 205, 208, 209, 212, 213, 216, 217, 220, 221, 224, 225, 228, 229, 232, 233, 236, 237, 240, 241, 244, 245, 248, 249, 252, 253, 256, 257, 260, 261, 264, 265, 268, 269, 272, 273, 276, 277, 280, 281, 284, 285, 288, 289, 292, 293, 296, 297, 300, 301, 304, 305, 308, 309, 312, 313, 316, 317, 320, 321, 324, 325, 328, 329, 332, 333, 336, 337, 340, 341, 344, 345, 348, 349, 352, 353, 356, 357, 360, 361, 364, 365, 368, 369, 372, 373, 376, 377, 380, 381, 384, 385, 388, 389, 392, 393, 396, 397, 400, 401, 404, 405, 408, 409, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 413, 6, 7, 8, 9, 414, 10, 11, 12, 13, 418, 10, 11, 12, 13, 419, 4, 5, 10, 11, 12, 13, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 423, 14, 15, 16, 17, 424, 4, 5, 14, 15, 16, 17, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 428, 18, 19, 20, 21, 429, 4, 5, 18, 19, 20, 21, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 433, 22, 23, 24, 25, 434, 4, 5, 22, 23, 24, 25, 4, 5, 22, 23, 24, 25, 26, 27, 28, 29, 438, 26, 27, 28, 29, 439, 4, 5, 26, 27, 28, 29, 4, 5, 26, 27, 28, 29, 30, 31, 32, 33, 443, 30, 31, 32, 33, 444, 4, 5, 30, 31, 32, 33, 4, 5, 30, 31, 32, 33, 34, 35, 36, 37, 448, 34, 35, 36, 37, 449, 4, 5, 34, 35, 36, 37, 4, 5, 34, 35, 36, 37, 38, 39, 40, 41, 453, 38, 39, 40, 41, 454, 4, 5, 38, 39, 40, 41, 4, 5, 38, 39, 40, 41, 42, 43, 44, 45, 458, 42, 43, 44, 45, 459, 4, 5, 42, 43, 44, 45, 4, 5, 42, 43, 44, 45, 46, 47, 48, 49, 463, 46, 47, 48, 49, 464, 4, 5, 46, 47, 48, 49, 4, 5, 46, 47, 48, 49, 50, 51, 52, 53, 468, 50, 51, 52, 53, 469, 4, 5, 50, 51, 52, 53, 4, 5, 50, 51, 52, 53, 54, 55, 56, 57, 473, 54, 55, 56, 57, 474, 4, 5, 54, 55, 56, 57, 4, 5, 54, 55, 56, 57, 58, 59, 60, 61, 478, 58, 59, 60, 61, 479, 4, 5, 58, 59, 60, 61, 4, 5, 58, 59, 60, 61, 62, 63, 64, 65, 483, 62, 63, 64, 65, 484, 4, 5, 62, 63, 64, 65, 4, 5, 62, 63, 64, 65, 66, 67, 68, 69, 488, 66, 67, 68, 69, 489, 4, 5, 66, 67, 68, 69, 4, 5, 66, 67, 68, 69, 70, 71, 72, 73, 493, 70, 71, 72, 73, 494, 4, 5, 70, 71, 72, 73, 4, 5, 70, 71, 72, 73, 74, 75, 76, 77, 498, 74, 75, 76, 77, 499, 4, 5, 74, 75, 76, 77, 4, 5, 74, 75, 76, 77, 78, 79, 80, 81, 503, 78, 79, 80, 81, 504, 4, 5, 78, 79, 80, 81, 4, 5, 78, 79, 80, 81, 82, 83, 84, 85, 508, 82, 83, 84, 85, 509, 4, 5, 82, 83, 84, 85, 4, 5, 82, 83, 84, 85, 86, 87, 88, 89, 513, 86, 87, 88, 89, 514, 4, 5, 86, 87, 88, 89, 4, 5, 86, 87, 88, 89, 90, 91, 92, 93, 518, 90, 91, 92, 93, 519, 4, 5, 90, 91, 92, 93, 4, 5, 90, 91, 92, 93, 94, 95, 96, 97, 523, 94, 95, 96, 97, 524, 4, 5, 94, 95, 96, 97, 4, 5, 94, 95, 96, 97, 98, 99, 100, 101, 528, 98, 99, 100, 101, 529, 4, 5, 98, 99, 100, 101, 4, 5, 98, 99, 100, 101, 102, 103, 104, 105, 533, 102, 103, 104, 105, 534, 4, 5, 102, 103, 104, 105, 4, 5, 102, 103, 104, 105, 106, 107, 108, 109, 538, 106, 107, 108, 109, 539, 4, 5, 106, 107, 108, 109, 4, 5, 106, 107, 108, 109, 110, 111, 112, 113, 543, 110, 111, 112, 113, 544, 4, 5, 110, 111, 112, 113, 4, 5, 110, 111, 112, 113, 114, 115, 116, 117, 548, 114, 115, 116, 117, 549, 4, 5, 114, 115, 116, 117, 4, 5, 114, 115, 116, 117, 118, 119, 120, 121, 553, 118, 119, 120, 121, 554, 4, 5, 118, 119, 120, 121, 4, 5, 118, 119, 120, 121, 122, 123, 124, 125, 558, 122, 123, 124, 125, 559, 4, 5, 122, 123, 124, 125, 4, 5, 122, 123, 124, 125, 126, 127, 128, 129, 563, 126, 127, 128, 129, 564, 4, 5, 126, 127, 128, 129, 4, 5, 126, 127, 128, 129, 130, 131, 132, 133, 568, 130, 131, 132, 133, 569, 4, 5, 130, 131, 132, 133, 4, 5, 130, 131, 132, 133, 134, 135, 136, 137, 573, 134, 135, 136, 137, 574, 4, 5, 134, 135, 136, 137, 4, 5, 134, 135, 136, 137, 138, 139, 140, 141, 578, 138, 139, 140, 141, 579, 4, 5, 138, 139, 140, 141, 4, 5, 138, 139, 140, 141, 142, 143, 144, 145, 583, 142, 143, 144, 145, 584, 4, 5, 142, 143, 144, 145, 4, 5, 142, 143, 144, 145, 146, 147, 148, 149, 588, 146, 147, 148, 149, 589, 4, 5, 146, 147, 148, 149, 4, 5, 146, 147, 148, 149, 150, 151, 152, 153, 593, 150, 151, 152, 153, 594, 4, 5, 150, 151, 152, 153, 4, 5, 150, 151, 152, 153, 154, 155, 156, 157, 598, 154, 155, 156, 157, 599, 4, 5, 154, 155, 156, 157, 4, 5, 154, 155, 156, 157, 158, 159, 160, 161, 603, 158, 159, 160, 161, 604, 4, 5, 158, 159, 160, 161, 4, 5, 158, 159, 160, 161, 162, 163, 164, 165, 608, 162, 163, 164, 165, 609, 4, 5, 162, 163, 164, 165, 4, 5, 162, 163, 164, 165, 166, 167, 168, 169, 613, 166, 167, 168, 169, 614, 4, 5, 166, 167, 168, 169, 4, 5, 166, 167, 168, 169, 170, 171, 172, 173, 618, 170, 171, 172, 173, 619, 4, 5, 170, 171, 172, 173, 4, 5, 170, 171, 172, 173, 174, 175, 176, 177, 623, 174, 175, 176, 177, 624, 4, 5, 174, 175, 176, 177, 4, 5, 174, 175, 176, 177, 178, 179, 180, 181, 628, 178, 179, 180, 181, 629, 4, 5, 178, 179, 180, 181, 4, 5, 178, 179, 180, 181, 182, 183, 184, 185, 633, 182, 183, 184, 185, 634, 4, 5, 182, 183, 184, 185, 4, 5, 182, 183, 184, 185, 186, 187, 188, 189, 638, 186, 187, 188, 189, 639, 4, 5, 186, 187, 188, 189, 4, 5, 186, 187, 188, 189, 190, 191, 192, 193, 643, 190, 191, 192, 193, 644, 4, 5, 190, 191, 192, 193, 4, 5, 190, 191, 192, 193, 194, 195, 196, 197, 648, 194, 195, 196, 197, 649, 4, 5, 194, 195, 196, 197, 4, 5, 194, 195, 196, 197, 198, 199, 200, 201, 653, 198, 199, 200, 201, 654, 4, 5, 198, 199, 200, 201, 4, 5, 198, 199, 200, 201, 202, 203, 204, 205, 658, 202, 203, 204, 205, 659, 4, 5, 202, 203, 204, 205, 4, 5, 202, 203, 204, 205, 206, 207, 208, 209, 663, 206, 207, 208, 209, 664, 4, 5, 206, 207, 208, 209, 4, 5, 206, 207, 208, 209, 210, 211, 212, 213, 668, 210, 211, 212, 213, 669, 4, 5, 210, 211, 212, 213, 4, 5, 210, 211, 212, 213, 214, 215, 216, 217, 673, 214, 215, 216, 217, 674, 4, 5, 214, 215, 216, 217, 4, 5, 214, 215, 216, 217, 218, 219, 220, 221, 678, 218, 219, 220, 221, 679, 4, 5, 218, 219, 220, 221, 4, 5, 218, 219, 220, 221, 222, 223, 224, 225, 683, 222, 223, 224, 225, 684, 4, 5, 222, 223, 224, 225, 4, 5, 222, 223, 224, 225, 226, 227, 228, 229, 688, 226, 227, 228, 229, 689, 4, 5, 226, 227, 228, 229, 4, 5, 226, 227, 228, 229, 230, 231, 232, 233, 693, 230, 231, 232, 233, 694, 4, 5, 230, 231, 232, 233, 4, 5, 230, 231, 232, 233, 234, 235, 236, 237, 698, 234, 235, 236, 237, 699, 4, 5, 234, 235, 236, 237, 4, 5, 234, 235, 236, 237, 238, 239, 240, 241, 703, 238, 239, 240, 241, 704, 4, 5, 238, 239, 240, 241, 4, 5, 238, 239, 240, 241, 242, 243, 244, 245, 708, 242, 243, 244, 245, 709, 4, 5, 242, 243, 244, 245, 4, 5, 242, 243, 244, 245, 246, 247, 248, 249, 713, 246, 247, 248, 249, 714, 4, 5, 246, 247, 248, 249, 4, 5, 246, 247, 248, 249, 250, 251, 252, 253, 718, 250, 251, 252, 253, 719, 4, 5, 250, 251, 252, 253, 4, 5, 250, 251, 252, 253, 254, 255, 256, 257, 723, 254, 255, 256, 257, 724, 4, 5, 254, 255, 256, 257, 4, 5, 254, 255, 256, 257, 258, 259, 260, 261, 728, 258, 259, 260, 261, 729, 4, 5, 258, 259, 260, 261, 4, 5, 258, 259, 260, 261, 262, 263, 264, 265, 733, 262, 263, 264, 265, 734, 4, 5, 262, 263, 264, 265, 4, 5, 262, 263, 264, 265, 266, 267, 268, 269, 738, 266, 267, 268, 269, 739, 4, 5, 266, 267, 268, 269, 4, 5, 266, 267, 268, 269, 270, 271, 272, 273, 743, 270, 271, 272, 273, 744, 4, 5, 270, 271, 272, 273, 4, 5, 270, 271, 272, 273, 274, 275, 276, 277, 748, 274, 275, 276, 277, 749, 4, 5, 274, 275, 276, 277, 4, 5, 274, 275, 276, 277, 278, 279, 280, 281, 753, 278, 279, 280, 281, 754, 4, 5, 278, 279, 280, 281, 4, 5, 278, 279, 280, 281, 282, 283, 284, 285, 758, 282, 283, 284, 285, 759, 4, 5, 282, 283, 284, 285, 4, 5, 282, 283, 284, 285, 286, 287, 288, 289, 763, 286, 287, 288, 289, 764, 4, 5, 286, 287, 288, 289, 4, 5, 286, 287, 288, 289, 290, 291, 292, 293, 768, 290, 291, 292, 293, 769, 4, 5, 290, 291, 292, 293, 4, 5, 290, 291, 292, 293, 294, 295, 296, 297, 773, 294, 295, 296, 297, 774, 4, 5, 294, 295, 296, 297, 4, 5, 294, 295, 296, 297, 298, 299, 300, 301, 778, 298, 299, 300, 301, 779, 4, 5, 298, 299, 300, 301, 4, 5, 298, 299, 300, 301, 302, 303, 304, 305, 783, 302, 303, 304, 305, 784, 4, 5, 302, 303, 304, 305, 4, 5, 302, 303, 304, 305, 306, 307, 308, 309, 788, 306, 307, 308, 309, 789, 4, 5, 306, 307, 308, 309, 4, 5, 306, 307, 308, 309, 310, 311, 312, 313, 793, 310, 311, 312, 313, 794, 4, 5, 310, 311, 312, 313, 4, 5, 310, 311, 312, 313, 314, 315, 316, 317, 798, 314, 315, 316, 317, 799, 4, 5, 314, 315, 316, 317, 4, 5, 314, 315, 316, 317, 318, 319, 320, 321, 803, 318, 319, 320, 321, 804, 4, 5, 318, 319, 320, 321, 4, 5, 318, 319, 320, 321, 322, 323, 324, 325, 808, 322, 323, 324, 325, 809, 4, 5, 322, 323, 324, 325, 4, 5, 322, 323, 324, 325, 326, 327, 328, 329, 813, 326, 327, 328, 329, 814, 4, 5, 326, 327, 328, 329, 4, 5, 326, 327, 328, 329, 330, 331, 332, 333, 818, 330, 331, 332, 333, 819, 4, 5, 330, 331, 332, 333, 4, 5, 330, 331, 332, 333, 334, 335, 336, 337, 823, 334, 335, 336, 337, 824, 4, 5, 334, 335, 336, 337, 4, 5, 334, 335, 336, 337, 338, 339, 340, 341, 828, 338, 339, 340, 341, 829, 4, 5, 338, 339, 340, 341, 4, 5, 338, 339, 340, 341, 342, 343, 344, 345, 833, 342, 343, 344, 345, 834, 4, 5, 342, 343, 344, 345, 4, 5, 342, 343, 344, 345, 346, 347, 348, 349, 838, 346, 347, 348, 349, 839, 4, 5, 346, 347, 348, 349, 4, 5, 346, 347, 348, 349, 350, 351, 352, 353, 843, 350, 351, 352, 353, 844, 4, 5, 350, 351, 352, 353, 4, 5, 350, 351, 352, 353, 354, 355, 356, 357, 848, 354, 355, 356, 357, 849, 4, 5, 354, 355, 356, 357, 4, 5, 354, 355, 356, 357, 358, 359, 360, 361, 853, 358, 359, 360, 361, 854, 4, 5, 358, 359, 360, 361, 4, 5, 358, 359, 360, 361, 362, 363, 364, 365, 858, 362, 363, 364, 365, 859, 4, 5, 362, 363, 364, 365, 4, 5, 362, 363, 364, 365, 366, 367, 368, 369, 863, 366, 367, 368, 369, 864, 4, 5, 366, 367, 368, 369, 4, 5, 366, 367, 368, 369, 370, 371, 372, 373, 868, 370, 371, 372, 373, 869, 4, 5, 370, 371, 372, 373, 4, 5, 370, 371, 372, 373, 374, 375, 376, 377, 873, 374, 375, 376, 377, 874, 4, 5, 374, 375, 376, 377, 4, 5, 374, 375, 376, 377, 378, 379, 380, 381, 878, 378, 379, 380, 381, 879, 4, 5, 378, 379, 380, 381, 4, 5, 378, 379, 380, 381, 382, 383, 384, 385, 883, 382, 383, 384, 385, 884, 4, 5, 382, 383, 384, 385, 4, 5, 382, 383, 384, 385, 386, 387, 388, 389, 888, 386, 387, 388, 389, 889, 4, 5, 386, 387, 388, 389, 4, 5, 386, 387, 388, 389, 390, 391, 392, 393, 893, 390, 391, 392, 393, 894, 4, 5, 390, 391, 392, 393, 4, 5, 390, 391, 392, 393, 394, 395, 396, 397, 898, 394, 395, 396, 397, 899, 4, 5, 394, 395, 396, 397, 4, 5, 394, 395, 396, 397, 398, 399, 400, 401, 903, 398, 399, 400, 401, 904, 4, 5, 398, 399, 400, 401, 4, 5, 398, 399, 400, 401, 402, 403, 404, 405, 908, 402, 403, 404, 405, 909, 4, 5, 402, 403, 404, 405, 4, 5, 402, 403, 404, 405, 406, 407, 408, 409, 913, 406, 407, 408, 409, 914, 4, 5, 406, 407, 408, 409, 4, 5, 406, 407, 408, 409, 1, 410, 0, 8, 9, 411, 412, 0, 2, 8, 9, 411, 412, 0, 8, 9, 411, 412, 413, 0, 8, 9, 411, 412, 414, 415, 418, 10, 417, 10, 416, 10, 11, 416, 417, 418, 10, 11, 416, 417, 419, 420, 423, 14, 422, 14, 421, 14, 15, 421, 422, 423, 14, 15, 421, 422, 424, 425, 428, 18, 427, 18, 426, 18, 19, 426, 427, 428, 18, 19, 426, 427, 429, 430, 433, 22, 432, 22, 431, 22, 23, 431, 432, 433, 22, 23, 431, 432, 434, 435, 438, 26, 437, 26, 436, 26, 27, 436, 437, 438, 26, 27, 436, 437, 439, 440, 443, 30, 442, 30, 441, 30, 31, 441, 442, 443, 30, 31, 441, 442, 444, 445, 448, 34, 447, 34, 446, 34, 35, 446, 447, 448, 34, 35, 446, 447, 449, 450, 453, 38, 452, 38, 451, 38, 39, 451, 452, 453, 38, 39, 451, 452, 454, 455, 458, 42, 457, 42, 456, 42, 43, 456, 457, 458, 42, 43, 456, 457, 459, 460, 463, 46, 462, 46, 461, 46, 47, 461, 462, 463, 46, 47, 461, 462, 464, 465, 468, 50, 467, 50, 466, 50, 51, 466, 467, 468, 50, 51, 466, 467, 469, 470, 473, 54, 472, 54, 471, 54, 55, 471, 472, 473, 54, 55, 471, 472, 474, 475, 478, 58, 477, 58, 476, 58, 59, 476, 477, 478, 58, 59, 476, 477, 479, 480, 483, 62, 482, 62, 481, 62, 63, 481, 482, 483, 62, 63, 481, 482, 484, 485, 488, 66, 487, 66, 486, 66, 67, 486, 487, 488, 66, 67, 486, 487, 489, 490, 493, 70, 492, 70, 491, 70, 71, 491, 492, 493, 70, 71, 491, 492, 494, 495, 498, 74, 497, 74, 496, 74, 75, 496, 497, 498, 74, 75, 496, 497, 499, 500, 503, 78, 502, 78, 501, 78, 79, 501, 502, 503, 78, 79, 501, 502, 504, 505, 508, 82, 507, 82, 506, 82, 83, 506, 507, 508, 82, 83, 506, 507, 509, 510, 513, 86, 512, 86, 511, 86, 87, 511, 512, 513, 86, 87, 511, 512, 514, 515, 518, 90, 517, 90, 516, 90, 91, 516, 517, 518, 90, 91, 516, 517, 519, 520, 523, 94, 522, 94, 521, 94, 95, 521, 522, 523, 94, 95, 521, 522, 524, 525, 528, 98, 527, 98, 526, 98, 99, 526, 527, 528, 98, 99, 526, 527, 529, 530, 533, 102, 532, 102, 531, 102, 103, 531, 532, 533, 102, 103, 531, 532, 534, 535, 538, 106, 537, 106, 536, 106, 107, 536, 537, 538, 106, 107, 536, 537, 539, 540, 543, 110, 542, 110, 541, 110, 111, 541, 542, 543, 110, 111, 541, 542, 544, 545, 548, 114, 547, 114, 546, 114, 115, 546, 547, 548, 114, 115, 546, 547, 549, 550, 553, 118, 552, 118, 551, 118, 119, 551, 552, 553, 118, 119, 551, 552, 554, 555, 558, 122, 557, 122, 556, 122, 123, 556, 557, 558, 122, 123, 556, 557, 559, 560, 563, 126, 562, 126, 561, 126, 127, 561, 562, 563, 126, 127, 561, 562, 564, 565, 568, 130, 567, 130, 566, 130, 131, 566, 567, 568, 130, 131, 566, 567, 569, 570, 573, 134, 572, 134, 571, 134, 135, 571, 572, 573, 134, 135, 571, 572, 574, 575, 578, 138, 577, 138, 576, 138, 139, 576, 577, 578, 138, 139, 576, 577, 579, 580, 583, 142, 582, 142, 581, 142, 143, 581, 582, 583, 142, 143, 581, 582, 584, 585, 588, 146, 587, 146, 586, 146, 147, 586, 587, 588, 146, 147, 586, 587, 589, 590, 593, 150, 592, 150, 591, 150, 151, 591, 592, 593, 150, 151, 591, 592, 594, 595, 598, 154, 597, 154, 596, 154, 155, 596, 597, 598, 154, 155, 596, 597, 599, 600, 603, 158, 602, 158, 601, 158, 159, 601, 602, 603, 158, 159, 601, 602, 604, 605, 608, 162, 607, 162, 606, 162, 163, 606, 607, 608, 162, 163, 606, 607, 609, 610, 613, 166, 612, 166, 611, 166, 167, 611, 612, 613, 166, 167, 611, 612, 614, 615, 618, 170, 617, 170, 616, 170, 171, 616, 617, 618, 170, 171, 616, 617, 619, 620, 623, 174, 622, 174, 621, 174, 175, 621, 622, 623, 174, 175, 621, 622, 624, 625, 628, 178, 627, 178, 626, 178, 179, 626, 627, 628, 178, 179, 626, 627, 629, 630, 633, 182, 632, 182, 631, 182, 183, 631, 632, 633, 182, 183, 631, 632, 634, 635, 638, 186, 637, 186, 636, 186, 187, 636, 637, 638, 186, 187, 636, 637, 639, 640, 643, 190, 642, 190, 641, 190, 191, 641, 642, 643, 190, 191, 641, 642, 644, 645, 648, 194, 647, 194, 646, 194, 195, 646, 647, 648, 194, 195, 646, 647, 649, 650, 653, 198, 652, 198, 651, 198, 199, 651, 652, 653, 198, 199, 651, 652, 654, 655, 658, 202, 657, 202, 656, 202, 203, 656, 657, 658, 202, 203, 656, 657, 659, 660, 663, 206, 662, 206, 661, 206, 207, 661, 662, 663, 206, 207, 661, 662, 664, 665, 668, 210, 667, 210, 666, 210, 211, 666, 667, 668, 210, 211, 666, 667, 669, 670, 673, 214, 672, 214, 671, 214, 215, 671, 672, 673, 214, 215, 671, 672, 674, 675, 678, 218, 677, 218, 676, 218, 219, 676, 677, 678, 218, 219, 676, 677, 679, 680, 683, 222, 682, 222, 681, 222, 223, 681, 682, 683, 222, 223, 681, 682, 684, 685, 688, 226, 687, 226, 686, 226, 227, 686, 687, 688, 226, 227, 686, 687, 689, 690, 693, 230, 692, 230, 691, 230, 231, 691, 692, 693, 230, 231, 691, 692, 694, 695, 698, 234, 697, 234, 696, 234, 235, 696, 697, 698, 234, 235, 696, 697, 699, 700, 703, 238, 702, 238, 701, 238, 239, 701, 702, 703, 238, 239, 701, 702, 704, 705, 708, 242, 707, 242, 706, 242, 243, 706, 707, 708, 242, 243, 706, 707, 709, 710, 713, 246, 712, 246, 711, 246, 247, 711, 712, 713, 246, 247, 711, 712, 714, 715, 718, 250, 717, 250, 716, 250, 251, 716, 717, 718, 250, 251, 716, 717, 719, 720, 723, 254, 722, 254, 721, 254, 255, 721, 722, 723, 254, 255, 721, 722, 724, 725, 728, 258, 727, 258, 726, 258, 259, 726, 727, 728, 258, 259, 726, 727, 729, 730, 733, 262, 732, 262, 731, 262, 263, 731, 732, 733, 262, 263, 731, 732, 734, 735, 738, 266, 737, 266, 736, 266, 267, 736, 737, 738, 266, 267, 736, 737, 739, 740, 743, 270, 742, 270, 741, 270, 271, 741, 742, 743, 270, 271, 741, 742, 744, 745, 748, 274, 747, 274, 746, 274, 275, 746, 747, 748, 274, 275, 746, 747, 749, 750, 753, 278, 752, 278, 751, 278, 279, 751, 752, 753, 278, 279, 751, 752, 754, 755, 758, 282, 757, 282, 756, 282, 283, 756, 757, 758, 282, 283, 756, 757, 759, 760, 763, 286, 762, 286, 761, 286, 287, 761, 762, 763, 286, 287, 761, 762, 764, 765, 768, 290, 767, 290, 766, 290, 291, 766, 767, 768, 290, 291, 766, 767, 769, 770, 773, 294, 772, 294, 771, 294, 295, 771, 772, 773, 294, 295, 771, 772, 774, 775, 778, 298, 777, 298, 776, 298, 299, 776, 777, 778, 298, 299, 776, 777, 779, 780, 783, 302, 782, 302, 781, 302, 303, 781, 782, 783, 302, 303, 781, 782, 784, 785, 788, 306, 787, 306, 786, 306, 307, 786, 787, 788, 306, 307, 786, 787, 789, 790, 793, 310, 792, 310, 791, 310, 311, 791, 792, 793, 310, 311, 791, 792, 794, 795, 798, 314, 797, 314, 796, 314, 315, 796, 797, 798, 314, 315, 796, 797, 799, 800, 803, 318, 802, 318, 801, 318, 319, 801, 802, 803, 318, 319, 801, 802, 804, 805, 808, 322, 807, 322, 806, 322, 323, 806, 807, 808, 322, 323, 806, 807, 809, 810, 813, 326, 812, 326, 811, 326, 327, 811, 812, 813, 326, 327, 811, 812, 814, 815, 818, 330, 817, 330, 816, 330, 331, 816, 817, 818, 330, 331, 816, 817, 819, 820, 823, 334, 822, 334, 821, 334, 335, 821, 822, 823, 334, 335, 821, 822, 824, 825, 828, 338, 827, 338, 826, 338, 339, 826, 827, 828, 338, 339, 826, 827, 829, 830, 833, 342, 832, 342, 831, 342, 343, 831, 832, 833, 342, 343, 831, 832, 834, 835, 838, 346, 837, 346, 836, 346, 347, 836, 837, 838, 346, 347, 836, 837, 839, 840, 843, 350, 842, 350, 841, 350, 351, 841, 842, 843, 350, 351, 841, 842, 844, 845, 848, 354, 847, 354, 846, 354, 355, 846, 847, 848, 354, 355, 846, 847, 849, 850, 853, 358, 852, 358, 851, 358, 359, 851, 852, 853, 358, 359, 851, 852, 854, 855, 858, 362, 857, 362, 856, 362, 363, 856, 857, 858, 362, 363, 856, 857, 859, 860, 863, 366, 862, 366, 861, 366, 367, 861, 862, 863, 366, 367, 861, 862, 864, 865, 868, 370, 867, 370, 866, 370, 371, 866, 867, 868, 370, 371, 866, 867, 869, 870, 873, 374, 872, 374, 871, 374, 375, 871, 872, 873, 374, 375, 871, 872, 874, 875, 878, 378, 877, 378, 876, 378, 379, 876, 877, 878, 378, 379, 876, 877, 879, 880, 883, 382, 882, 382, 881, 382, 383, 881, 882, 883, 382, 383, 881, 882, 884, 885, 888, 386, 887, 386, 886, 386, 387, 886, 887, 888, 386, 387, 886, 887, 889, 890, 893, 390, 892, 390, 891, 390, 391, 891, 892, 893, 390, 391, 891, 892, 894, 895, 898, 394, 897, 394, 896, 394, 395, 896, 897, 898, 394, 395, 896, 897, 899, 900, 903, 398, 902, 398, 901, 398, 399, 901, 902, 903, 398, 399, 901, 902, 904, 905, 908, 402, 907, 402, 906, 402, 403, 906, 907, 908, 402, 403, 906, 907, 909, 910, 913, 406, 912, 406, 911, 406, 407, 911, 912, 913, 406, 407, 911, 912, 914, 410, 915, 3, 915, 916]
    sp_jac_ini_ja = [0, 3, 4, 5, 7, 211, 415, 421, 427, 432, 437, 442, 447, 453, 459, 464, 469, 475, 481, 486, 491, 497, 503, 508, 513, 519, 525, 530, 535, 541, 547, 552, 557, 563, 569, 574, 579, 585, 591, 596, 601, 607, 613, 618, 623, 629, 635, 640, 645, 651, 657, 662, 667, 673, 679, 684, 689, 695, 701, 706, 711, 717, 723, 728, 733, 739, 745, 750, 755, 761, 767, 772, 777, 783, 789, 794, 799, 805, 811, 816, 821, 827, 833, 838, 843, 849, 855, 860, 865, 871, 877, 882, 887, 893, 899, 904, 909, 915, 921, 926, 931, 937, 943, 948, 953, 959, 965, 970, 975, 981, 987, 992, 997, 1003, 1009, 1014, 1019, 1025, 1031, 1036, 1041, 1047, 1053, 1058, 1063, 1069, 1075, 1080, 1085, 1091, 1097, 1102, 1107, 1113, 1119, 1124, 1129, 1135, 1141, 1146, 1151, 1157, 1163, 1168, 1173, 1179, 1185, 1190, 1195, 1201, 1207, 1212, 1217, 1223, 1229, 1234, 1239, 1245, 1251, 1256, 1261, 1267, 1273, 1278, 1283, 1289, 1295, 1300, 1305, 1311, 1317, 1322, 1327, 1333, 1339, 1344, 1349, 1355, 1361, 1366, 1371, 1377, 1383, 1388, 1393, 1399, 1405, 1410, 1415, 1421, 1427, 1432, 1437, 1443, 1449, 1454, 1459, 1465, 1471, 1476, 1481, 1487, 1493, 1498, 1503, 1509, 1515, 1520, 1525, 1531, 1537, 1542, 1547, 1553, 1559, 1564, 1569, 1575, 1581, 1586, 1591, 1597, 1603, 1608, 1613, 1619, 1625, 1630, 1635, 1641, 1647, 1652, 1657, 1663, 1669, 1674, 1679, 1685, 1691, 1696, 1701, 1707, 1713, 1718, 1723, 1729, 1735, 1740, 1745, 1751, 1757, 1762, 1767, 1773, 1779, 1784, 1789, 1795, 1801, 1806, 1811, 1817, 1823, 1828, 1833, 1839, 1845, 1850, 1855, 1861, 1867, 1872, 1877, 1883, 1889, 1894, 1899, 1905, 1911, 1916, 1921, 1927, 1933, 1938, 1943, 1949, 1955, 1960, 1965, 1971, 1977, 1982, 1987, 1993, 1999, 2004, 2009, 2015, 2021, 2026, 2031, 2037, 2043, 2048, 2053, 2059, 2065, 2070, 2075, 2081, 2087, 2092, 2097, 2103, 2109, 2114, 2119, 2125, 2131, 2136, 2141, 2147, 2153, 2158, 2163, 2169, 2175, 2180, 2185, 2191, 2197, 2202, 2207, 2213, 2219, 2224, 2229, 2235, 2241, 2246, 2251, 2257, 2263, 2268, 2273, 2279, 2285, 2290, 2295, 2301, 2307, 2312, 2317, 2323, 2329, 2334, 2339, 2345, 2351, 2356, 2361, 2367, 2373, 2378, 2383, 2389, 2395, 2400, 2405, 2411, 2417, 2422, 2427, 2433, 2439, 2444, 2449, 2455, 2461, 2466, 2471, 2477, 2483, 2488, 2493, 2499, 2505, 2510, 2515, 2521, 2527, 2532, 2537, 2543, 2549, 2554, 2559, 2565, 2571, 2576, 2581, 2587, 2593, 2598, 2603, 2609, 2615, 2620, 2625, 2631, 2637, 2639, 2644, 2650, 2656, 2662, 2664, 2666, 2668, 2673, 2678, 2680, 2682, 2684, 2689, 2694, 2696, 2698, 2700, 2705, 2710, 2712, 2714, 2716, 2721, 2726, 2728, 2730, 2732, 2737, 2742, 2744, 2746, 2748, 2753, 2758, 2760, 2762, 2764, 2769, 2774, 2776, 2778, 2780, 2785, 2790, 2792, 2794, 2796, 2801, 2806, 2808, 2810, 2812, 2817, 2822, 2824, 2826, 2828, 2833, 2838, 2840, 2842, 2844, 2849, 2854, 2856, 2858, 2860, 2865, 2870, 2872, 2874, 2876, 2881, 2886, 2888, 2890, 2892, 2897, 2902, 2904, 2906, 2908, 2913, 2918, 2920, 2922, 2924, 2929, 2934, 2936, 2938, 2940, 2945, 2950, 2952, 2954, 2956, 2961, 2966, 2968, 2970, 2972, 2977, 2982, 2984, 2986, 2988, 2993, 2998, 3000, 3002, 3004, 3009, 3014, 3016, 3018, 3020, 3025, 3030, 3032, 3034, 3036, 3041, 3046, 3048, 3050, 3052, 3057, 3062, 3064, 3066, 3068, 3073, 3078, 3080, 3082, 3084, 3089, 3094, 3096, 3098, 3100, 3105, 3110, 3112, 3114, 3116, 3121, 3126, 3128, 3130, 3132, 3137, 3142, 3144, 3146, 3148, 3153, 3158, 3160, 3162, 3164, 3169, 3174, 3176, 3178, 3180, 3185, 3190, 3192, 3194, 3196, 3201, 3206, 3208, 3210, 3212, 3217, 3222, 3224, 3226, 3228, 3233, 3238, 3240, 3242, 3244, 3249, 3254, 3256, 3258, 3260, 3265, 3270, 3272, 3274, 3276, 3281, 3286, 3288, 3290, 3292, 3297, 3302, 3304, 3306, 3308, 3313, 3318, 3320, 3322, 3324, 3329, 3334, 3336, 3338, 3340, 3345, 3350, 3352, 3354, 3356, 3361, 3366, 3368, 3370, 3372, 3377, 3382, 3384, 3386, 3388, 3393, 3398, 3400, 3402, 3404, 3409, 3414, 3416, 3418, 3420, 3425, 3430, 3432, 3434, 3436, 3441, 3446, 3448, 3450, 3452, 3457, 3462, 3464, 3466, 3468, 3473, 3478, 3480, 3482, 3484, 3489, 3494, 3496, 3498, 3500, 3505, 3510, 3512, 3514, 3516, 3521, 3526, 3528, 3530, 3532, 3537, 3542, 3544, 3546, 3548, 3553, 3558, 3560, 3562, 3564, 3569, 3574, 3576, 3578, 3580, 3585, 3590, 3592, 3594, 3596, 3601, 3606, 3608, 3610, 3612, 3617, 3622, 3624, 3626, 3628, 3633, 3638, 3640, 3642, 3644, 3649, 3654, 3656, 3658, 3660, 3665, 3670, 3672, 3674, 3676, 3681, 3686, 3688, 3690, 3692, 3697, 3702, 3704, 3706, 3708, 3713, 3718, 3720, 3722, 3724, 3729, 3734, 3736, 3738, 3740, 3745, 3750, 3752, 3754, 3756, 3761, 3766, 3768, 3770, 3772, 3777, 3782, 3784, 3786, 3788, 3793, 3798, 3800, 3802, 3804, 3809, 3814, 3816, 3818, 3820, 3825, 3830, 3832, 3834, 3836, 3841, 3846, 3848, 3850, 3852, 3857, 3862, 3864, 3866, 3868, 3873, 3878, 3880, 3882, 3884, 3889, 3894, 3896, 3898, 3900, 3905, 3910, 3912, 3914, 3916, 3921, 3926, 3928, 3930, 3932, 3937, 3942, 3944, 3946, 3948, 3953, 3958, 3960, 3962, 3964, 3969, 3974, 3976, 3978, 3980, 3985, 3990, 3992, 3994, 3996, 4001, 4006, 4008, 4010, 4012, 4017, 4022, 4024, 4026, 4028, 4033, 4038, 4040, 4042, 4044, 4049, 4054, 4056, 4058, 4060, 4065, 4070, 4072, 4074, 4076, 4081, 4086, 4088, 4090, 4092, 4097, 4102, 4104, 4106, 4108, 4113, 4118, 4120, 4122, 4124, 4129, 4134, 4136, 4138, 4140, 4145, 4150, 4152, 4154, 4156, 4161, 4166, 4168, 4170, 4172, 4177, 4182, 4184, 4186, 4188, 4193, 4198, 4200, 4202, 4204, 4209, 4214, 4216, 4218, 4220, 4225, 4230, 4232, 4234, 4236, 4241, 4246, 4248, 4250, 4252, 4257, 4262, 4264, 4267]
    sp_jac_ini_nia = 917
    sp_jac_ini_nja = 917
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 410, 915, 1, 2, 3, 915, 4, 5, 6, 7, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97, 100, 101, 104, 105, 108, 109, 112, 113, 116, 117, 120, 121, 124, 125, 128, 129, 132, 133, 136, 137, 140, 141, 144, 145, 148, 149, 152, 153, 156, 157, 160, 161, 164, 165, 168, 169, 172, 173, 176, 177, 180, 181, 184, 185, 188, 189, 192, 193, 196, 197, 200, 201, 204, 205, 208, 209, 212, 213, 216, 217, 220, 221, 224, 225, 228, 229, 232, 233, 236, 237, 240, 241, 244, 245, 248, 249, 252, 253, 256, 257, 260, 261, 264, 265, 268, 269, 272, 273, 276, 277, 280, 281, 284, 285, 288, 289, 292, 293, 296, 297, 300, 301, 304, 305, 308, 309, 312, 313, 316, 317, 320, 321, 324, 325, 328, 329, 332, 333, 336, 337, 340, 341, 344, 345, 348, 349, 352, 353, 356, 357, 360, 361, 364, 365, 368, 369, 372, 373, 376, 377, 380, 381, 384, 385, 388, 389, 392, 393, 396, 397, 400, 401, 404, 405, 408, 409, 4, 5, 6, 7, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97, 100, 101, 104, 105, 108, 109, 112, 113, 116, 117, 120, 121, 124, 125, 128, 129, 132, 133, 136, 137, 140, 141, 144, 145, 148, 149, 152, 153, 156, 157, 160, 161, 164, 165, 168, 169, 172, 173, 176, 177, 180, 181, 184, 185, 188, 189, 192, 193, 196, 197, 200, 201, 204, 205, 208, 209, 212, 213, 216, 217, 220, 221, 224, 225, 228, 229, 232, 233, 236, 237, 240, 241, 244, 245, 248, 249, 252, 253, 256, 257, 260, 261, 264, 265, 268, 269, 272, 273, 276, 277, 280, 281, 284, 285, 288, 289, 292, 293, 296, 297, 300, 301, 304, 305, 308, 309, 312, 313, 316, 317, 320, 321, 324, 325, 328, 329, 332, 333, 336, 337, 340, 341, 344, 345, 348, 349, 352, 353, 356, 357, 360, 361, 364, 365, 368, 369, 372, 373, 376, 377, 380, 381, 384, 385, 388, 389, 392, 393, 396, 397, 400, 401, 404, 405, 408, 409, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 413, 6, 7, 8, 9, 414, 10, 11, 12, 13, 418, 10, 11, 12, 13, 419, 4, 5, 10, 11, 12, 13, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 423, 14, 15, 16, 17, 424, 4, 5, 14, 15, 16, 17, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 428, 18, 19, 20, 21, 429, 4, 5, 18, 19, 20, 21, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 433, 22, 23, 24, 25, 434, 4, 5, 22, 23, 24, 25, 4, 5, 22, 23, 24, 25, 26, 27, 28, 29, 438, 26, 27, 28, 29, 439, 4, 5, 26, 27, 28, 29, 4, 5, 26, 27, 28, 29, 30, 31, 32, 33, 443, 30, 31, 32, 33, 444, 4, 5, 30, 31, 32, 33, 4, 5, 30, 31, 32, 33, 34, 35, 36, 37, 448, 34, 35, 36, 37, 449, 4, 5, 34, 35, 36, 37, 4, 5, 34, 35, 36, 37, 38, 39, 40, 41, 453, 38, 39, 40, 41, 454, 4, 5, 38, 39, 40, 41, 4, 5, 38, 39, 40, 41, 42, 43, 44, 45, 458, 42, 43, 44, 45, 459, 4, 5, 42, 43, 44, 45, 4, 5, 42, 43, 44, 45, 46, 47, 48, 49, 463, 46, 47, 48, 49, 464, 4, 5, 46, 47, 48, 49, 4, 5, 46, 47, 48, 49, 50, 51, 52, 53, 468, 50, 51, 52, 53, 469, 4, 5, 50, 51, 52, 53, 4, 5, 50, 51, 52, 53, 54, 55, 56, 57, 473, 54, 55, 56, 57, 474, 4, 5, 54, 55, 56, 57, 4, 5, 54, 55, 56, 57, 58, 59, 60, 61, 478, 58, 59, 60, 61, 479, 4, 5, 58, 59, 60, 61, 4, 5, 58, 59, 60, 61, 62, 63, 64, 65, 483, 62, 63, 64, 65, 484, 4, 5, 62, 63, 64, 65, 4, 5, 62, 63, 64, 65, 66, 67, 68, 69, 488, 66, 67, 68, 69, 489, 4, 5, 66, 67, 68, 69, 4, 5, 66, 67, 68, 69, 70, 71, 72, 73, 493, 70, 71, 72, 73, 494, 4, 5, 70, 71, 72, 73, 4, 5, 70, 71, 72, 73, 74, 75, 76, 77, 498, 74, 75, 76, 77, 499, 4, 5, 74, 75, 76, 77, 4, 5, 74, 75, 76, 77, 78, 79, 80, 81, 503, 78, 79, 80, 81, 504, 4, 5, 78, 79, 80, 81, 4, 5, 78, 79, 80, 81, 82, 83, 84, 85, 508, 82, 83, 84, 85, 509, 4, 5, 82, 83, 84, 85, 4, 5, 82, 83, 84, 85, 86, 87, 88, 89, 513, 86, 87, 88, 89, 514, 4, 5, 86, 87, 88, 89, 4, 5, 86, 87, 88, 89, 90, 91, 92, 93, 518, 90, 91, 92, 93, 519, 4, 5, 90, 91, 92, 93, 4, 5, 90, 91, 92, 93, 94, 95, 96, 97, 523, 94, 95, 96, 97, 524, 4, 5, 94, 95, 96, 97, 4, 5, 94, 95, 96, 97, 98, 99, 100, 101, 528, 98, 99, 100, 101, 529, 4, 5, 98, 99, 100, 101, 4, 5, 98, 99, 100, 101, 102, 103, 104, 105, 533, 102, 103, 104, 105, 534, 4, 5, 102, 103, 104, 105, 4, 5, 102, 103, 104, 105, 106, 107, 108, 109, 538, 106, 107, 108, 109, 539, 4, 5, 106, 107, 108, 109, 4, 5, 106, 107, 108, 109, 110, 111, 112, 113, 543, 110, 111, 112, 113, 544, 4, 5, 110, 111, 112, 113, 4, 5, 110, 111, 112, 113, 114, 115, 116, 117, 548, 114, 115, 116, 117, 549, 4, 5, 114, 115, 116, 117, 4, 5, 114, 115, 116, 117, 118, 119, 120, 121, 553, 118, 119, 120, 121, 554, 4, 5, 118, 119, 120, 121, 4, 5, 118, 119, 120, 121, 122, 123, 124, 125, 558, 122, 123, 124, 125, 559, 4, 5, 122, 123, 124, 125, 4, 5, 122, 123, 124, 125, 126, 127, 128, 129, 563, 126, 127, 128, 129, 564, 4, 5, 126, 127, 128, 129, 4, 5, 126, 127, 128, 129, 130, 131, 132, 133, 568, 130, 131, 132, 133, 569, 4, 5, 130, 131, 132, 133, 4, 5, 130, 131, 132, 133, 134, 135, 136, 137, 573, 134, 135, 136, 137, 574, 4, 5, 134, 135, 136, 137, 4, 5, 134, 135, 136, 137, 138, 139, 140, 141, 578, 138, 139, 140, 141, 579, 4, 5, 138, 139, 140, 141, 4, 5, 138, 139, 140, 141, 142, 143, 144, 145, 583, 142, 143, 144, 145, 584, 4, 5, 142, 143, 144, 145, 4, 5, 142, 143, 144, 145, 146, 147, 148, 149, 588, 146, 147, 148, 149, 589, 4, 5, 146, 147, 148, 149, 4, 5, 146, 147, 148, 149, 150, 151, 152, 153, 593, 150, 151, 152, 153, 594, 4, 5, 150, 151, 152, 153, 4, 5, 150, 151, 152, 153, 154, 155, 156, 157, 598, 154, 155, 156, 157, 599, 4, 5, 154, 155, 156, 157, 4, 5, 154, 155, 156, 157, 158, 159, 160, 161, 603, 158, 159, 160, 161, 604, 4, 5, 158, 159, 160, 161, 4, 5, 158, 159, 160, 161, 162, 163, 164, 165, 608, 162, 163, 164, 165, 609, 4, 5, 162, 163, 164, 165, 4, 5, 162, 163, 164, 165, 166, 167, 168, 169, 613, 166, 167, 168, 169, 614, 4, 5, 166, 167, 168, 169, 4, 5, 166, 167, 168, 169, 170, 171, 172, 173, 618, 170, 171, 172, 173, 619, 4, 5, 170, 171, 172, 173, 4, 5, 170, 171, 172, 173, 174, 175, 176, 177, 623, 174, 175, 176, 177, 624, 4, 5, 174, 175, 176, 177, 4, 5, 174, 175, 176, 177, 178, 179, 180, 181, 628, 178, 179, 180, 181, 629, 4, 5, 178, 179, 180, 181, 4, 5, 178, 179, 180, 181, 182, 183, 184, 185, 633, 182, 183, 184, 185, 634, 4, 5, 182, 183, 184, 185, 4, 5, 182, 183, 184, 185, 186, 187, 188, 189, 638, 186, 187, 188, 189, 639, 4, 5, 186, 187, 188, 189, 4, 5, 186, 187, 188, 189, 190, 191, 192, 193, 643, 190, 191, 192, 193, 644, 4, 5, 190, 191, 192, 193, 4, 5, 190, 191, 192, 193, 194, 195, 196, 197, 648, 194, 195, 196, 197, 649, 4, 5, 194, 195, 196, 197, 4, 5, 194, 195, 196, 197, 198, 199, 200, 201, 653, 198, 199, 200, 201, 654, 4, 5, 198, 199, 200, 201, 4, 5, 198, 199, 200, 201, 202, 203, 204, 205, 658, 202, 203, 204, 205, 659, 4, 5, 202, 203, 204, 205, 4, 5, 202, 203, 204, 205, 206, 207, 208, 209, 663, 206, 207, 208, 209, 664, 4, 5, 206, 207, 208, 209, 4, 5, 206, 207, 208, 209, 210, 211, 212, 213, 668, 210, 211, 212, 213, 669, 4, 5, 210, 211, 212, 213, 4, 5, 210, 211, 212, 213, 214, 215, 216, 217, 673, 214, 215, 216, 217, 674, 4, 5, 214, 215, 216, 217, 4, 5, 214, 215, 216, 217, 218, 219, 220, 221, 678, 218, 219, 220, 221, 679, 4, 5, 218, 219, 220, 221, 4, 5, 218, 219, 220, 221, 222, 223, 224, 225, 683, 222, 223, 224, 225, 684, 4, 5, 222, 223, 224, 225, 4, 5, 222, 223, 224, 225, 226, 227, 228, 229, 688, 226, 227, 228, 229, 689, 4, 5, 226, 227, 228, 229, 4, 5, 226, 227, 228, 229, 230, 231, 232, 233, 693, 230, 231, 232, 233, 694, 4, 5, 230, 231, 232, 233, 4, 5, 230, 231, 232, 233, 234, 235, 236, 237, 698, 234, 235, 236, 237, 699, 4, 5, 234, 235, 236, 237, 4, 5, 234, 235, 236, 237, 238, 239, 240, 241, 703, 238, 239, 240, 241, 704, 4, 5, 238, 239, 240, 241, 4, 5, 238, 239, 240, 241, 242, 243, 244, 245, 708, 242, 243, 244, 245, 709, 4, 5, 242, 243, 244, 245, 4, 5, 242, 243, 244, 245, 246, 247, 248, 249, 713, 246, 247, 248, 249, 714, 4, 5, 246, 247, 248, 249, 4, 5, 246, 247, 248, 249, 250, 251, 252, 253, 718, 250, 251, 252, 253, 719, 4, 5, 250, 251, 252, 253, 4, 5, 250, 251, 252, 253, 254, 255, 256, 257, 723, 254, 255, 256, 257, 724, 4, 5, 254, 255, 256, 257, 4, 5, 254, 255, 256, 257, 258, 259, 260, 261, 728, 258, 259, 260, 261, 729, 4, 5, 258, 259, 260, 261, 4, 5, 258, 259, 260, 261, 262, 263, 264, 265, 733, 262, 263, 264, 265, 734, 4, 5, 262, 263, 264, 265, 4, 5, 262, 263, 264, 265, 266, 267, 268, 269, 738, 266, 267, 268, 269, 739, 4, 5, 266, 267, 268, 269, 4, 5, 266, 267, 268, 269, 270, 271, 272, 273, 743, 270, 271, 272, 273, 744, 4, 5, 270, 271, 272, 273, 4, 5, 270, 271, 272, 273, 274, 275, 276, 277, 748, 274, 275, 276, 277, 749, 4, 5, 274, 275, 276, 277, 4, 5, 274, 275, 276, 277, 278, 279, 280, 281, 753, 278, 279, 280, 281, 754, 4, 5, 278, 279, 280, 281, 4, 5, 278, 279, 280, 281, 282, 283, 284, 285, 758, 282, 283, 284, 285, 759, 4, 5, 282, 283, 284, 285, 4, 5, 282, 283, 284, 285, 286, 287, 288, 289, 763, 286, 287, 288, 289, 764, 4, 5, 286, 287, 288, 289, 4, 5, 286, 287, 288, 289, 290, 291, 292, 293, 768, 290, 291, 292, 293, 769, 4, 5, 290, 291, 292, 293, 4, 5, 290, 291, 292, 293, 294, 295, 296, 297, 773, 294, 295, 296, 297, 774, 4, 5, 294, 295, 296, 297, 4, 5, 294, 295, 296, 297, 298, 299, 300, 301, 778, 298, 299, 300, 301, 779, 4, 5, 298, 299, 300, 301, 4, 5, 298, 299, 300, 301, 302, 303, 304, 305, 783, 302, 303, 304, 305, 784, 4, 5, 302, 303, 304, 305, 4, 5, 302, 303, 304, 305, 306, 307, 308, 309, 788, 306, 307, 308, 309, 789, 4, 5, 306, 307, 308, 309, 4, 5, 306, 307, 308, 309, 310, 311, 312, 313, 793, 310, 311, 312, 313, 794, 4, 5, 310, 311, 312, 313, 4, 5, 310, 311, 312, 313, 314, 315, 316, 317, 798, 314, 315, 316, 317, 799, 4, 5, 314, 315, 316, 317, 4, 5, 314, 315, 316, 317, 318, 319, 320, 321, 803, 318, 319, 320, 321, 804, 4, 5, 318, 319, 320, 321, 4, 5, 318, 319, 320, 321, 322, 323, 324, 325, 808, 322, 323, 324, 325, 809, 4, 5, 322, 323, 324, 325, 4, 5, 322, 323, 324, 325, 326, 327, 328, 329, 813, 326, 327, 328, 329, 814, 4, 5, 326, 327, 328, 329, 4, 5, 326, 327, 328, 329, 330, 331, 332, 333, 818, 330, 331, 332, 333, 819, 4, 5, 330, 331, 332, 333, 4, 5, 330, 331, 332, 333, 334, 335, 336, 337, 823, 334, 335, 336, 337, 824, 4, 5, 334, 335, 336, 337, 4, 5, 334, 335, 336, 337, 338, 339, 340, 341, 828, 338, 339, 340, 341, 829, 4, 5, 338, 339, 340, 341, 4, 5, 338, 339, 340, 341, 342, 343, 344, 345, 833, 342, 343, 344, 345, 834, 4, 5, 342, 343, 344, 345, 4, 5, 342, 343, 344, 345, 346, 347, 348, 349, 838, 346, 347, 348, 349, 839, 4, 5, 346, 347, 348, 349, 4, 5, 346, 347, 348, 349, 350, 351, 352, 353, 843, 350, 351, 352, 353, 844, 4, 5, 350, 351, 352, 353, 4, 5, 350, 351, 352, 353, 354, 355, 356, 357, 848, 354, 355, 356, 357, 849, 4, 5, 354, 355, 356, 357, 4, 5, 354, 355, 356, 357, 358, 359, 360, 361, 853, 358, 359, 360, 361, 854, 4, 5, 358, 359, 360, 361, 4, 5, 358, 359, 360, 361, 362, 363, 364, 365, 858, 362, 363, 364, 365, 859, 4, 5, 362, 363, 364, 365, 4, 5, 362, 363, 364, 365, 366, 367, 368, 369, 863, 366, 367, 368, 369, 864, 4, 5, 366, 367, 368, 369, 4, 5, 366, 367, 368, 369, 370, 371, 372, 373, 868, 370, 371, 372, 373, 869, 4, 5, 370, 371, 372, 373, 4, 5, 370, 371, 372, 373, 374, 375, 376, 377, 873, 374, 375, 376, 377, 874, 4, 5, 374, 375, 376, 377, 4, 5, 374, 375, 376, 377, 378, 379, 380, 381, 878, 378, 379, 380, 381, 879, 4, 5, 378, 379, 380, 381, 4, 5, 378, 379, 380, 381, 382, 383, 384, 385, 883, 382, 383, 384, 385, 884, 4, 5, 382, 383, 384, 385, 4, 5, 382, 383, 384, 385, 386, 387, 388, 389, 888, 386, 387, 388, 389, 889, 4, 5, 386, 387, 388, 389, 4, 5, 386, 387, 388, 389, 390, 391, 392, 393, 893, 390, 391, 392, 393, 894, 4, 5, 390, 391, 392, 393, 4, 5, 390, 391, 392, 393, 394, 395, 396, 397, 898, 394, 395, 396, 397, 899, 4, 5, 394, 395, 396, 397, 4, 5, 394, 395, 396, 397, 398, 399, 400, 401, 903, 398, 399, 400, 401, 904, 4, 5, 398, 399, 400, 401, 4, 5, 398, 399, 400, 401, 402, 403, 404, 405, 908, 402, 403, 404, 405, 909, 4, 5, 402, 403, 404, 405, 4, 5, 402, 403, 404, 405, 406, 407, 408, 409, 913, 406, 407, 408, 409, 914, 4, 5, 406, 407, 408, 409, 4, 5, 406, 407, 408, 409, 1, 410, 0, 8, 9, 411, 412, 0, 2, 8, 9, 411, 412, 0, 8, 9, 411, 412, 413, 0, 8, 9, 411, 412, 414, 415, 418, 10, 417, 10, 416, 10, 11, 416, 417, 418, 10, 11, 416, 417, 419, 420, 423, 14, 422, 14, 421, 14, 15, 421, 422, 423, 14, 15, 421, 422, 424, 425, 428, 18, 427, 18, 426, 18, 19, 426, 427, 428, 18, 19, 426, 427, 429, 430, 433, 22, 432, 22, 431, 22, 23, 431, 432, 433, 22, 23, 431, 432, 434, 435, 438, 26, 437, 26, 436, 26, 27, 436, 437, 438, 26, 27, 436, 437, 439, 440, 443, 30, 442, 30, 441, 30, 31, 441, 442, 443, 30, 31, 441, 442, 444, 445, 448, 34, 447, 34, 446, 34, 35, 446, 447, 448, 34, 35, 446, 447, 449, 450, 453, 38, 452, 38, 451, 38, 39, 451, 452, 453, 38, 39, 451, 452, 454, 455, 458, 42, 457, 42, 456, 42, 43, 456, 457, 458, 42, 43, 456, 457, 459, 460, 463, 46, 462, 46, 461, 46, 47, 461, 462, 463, 46, 47, 461, 462, 464, 465, 468, 50, 467, 50, 466, 50, 51, 466, 467, 468, 50, 51, 466, 467, 469, 470, 473, 54, 472, 54, 471, 54, 55, 471, 472, 473, 54, 55, 471, 472, 474, 475, 478, 58, 477, 58, 476, 58, 59, 476, 477, 478, 58, 59, 476, 477, 479, 480, 483, 62, 482, 62, 481, 62, 63, 481, 482, 483, 62, 63, 481, 482, 484, 485, 488, 66, 487, 66, 486, 66, 67, 486, 487, 488, 66, 67, 486, 487, 489, 490, 493, 70, 492, 70, 491, 70, 71, 491, 492, 493, 70, 71, 491, 492, 494, 495, 498, 74, 497, 74, 496, 74, 75, 496, 497, 498, 74, 75, 496, 497, 499, 500, 503, 78, 502, 78, 501, 78, 79, 501, 502, 503, 78, 79, 501, 502, 504, 505, 508, 82, 507, 82, 506, 82, 83, 506, 507, 508, 82, 83, 506, 507, 509, 510, 513, 86, 512, 86, 511, 86, 87, 511, 512, 513, 86, 87, 511, 512, 514, 515, 518, 90, 517, 90, 516, 90, 91, 516, 517, 518, 90, 91, 516, 517, 519, 520, 523, 94, 522, 94, 521, 94, 95, 521, 522, 523, 94, 95, 521, 522, 524, 525, 528, 98, 527, 98, 526, 98, 99, 526, 527, 528, 98, 99, 526, 527, 529, 530, 533, 102, 532, 102, 531, 102, 103, 531, 532, 533, 102, 103, 531, 532, 534, 535, 538, 106, 537, 106, 536, 106, 107, 536, 537, 538, 106, 107, 536, 537, 539, 540, 543, 110, 542, 110, 541, 110, 111, 541, 542, 543, 110, 111, 541, 542, 544, 545, 548, 114, 547, 114, 546, 114, 115, 546, 547, 548, 114, 115, 546, 547, 549, 550, 553, 118, 552, 118, 551, 118, 119, 551, 552, 553, 118, 119, 551, 552, 554, 555, 558, 122, 557, 122, 556, 122, 123, 556, 557, 558, 122, 123, 556, 557, 559, 560, 563, 126, 562, 126, 561, 126, 127, 561, 562, 563, 126, 127, 561, 562, 564, 565, 568, 130, 567, 130, 566, 130, 131, 566, 567, 568, 130, 131, 566, 567, 569, 570, 573, 134, 572, 134, 571, 134, 135, 571, 572, 573, 134, 135, 571, 572, 574, 575, 578, 138, 577, 138, 576, 138, 139, 576, 577, 578, 138, 139, 576, 577, 579, 580, 583, 142, 582, 142, 581, 142, 143, 581, 582, 583, 142, 143, 581, 582, 584, 585, 588, 146, 587, 146, 586, 146, 147, 586, 587, 588, 146, 147, 586, 587, 589, 590, 593, 150, 592, 150, 591, 150, 151, 591, 592, 593, 150, 151, 591, 592, 594, 595, 598, 154, 597, 154, 596, 154, 155, 596, 597, 598, 154, 155, 596, 597, 599, 600, 603, 158, 602, 158, 601, 158, 159, 601, 602, 603, 158, 159, 601, 602, 604, 605, 608, 162, 607, 162, 606, 162, 163, 606, 607, 608, 162, 163, 606, 607, 609, 610, 613, 166, 612, 166, 611, 166, 167, 611, 612, 613, 166, 167, 611, 612, 614, 615, 618, 170, 617, 170, 616, 170, 171, 616, 617, 618, 170, 171, 616, 617, 619, 620, 623, 174, 622, 174, 621, 174, 175, 621, 622, 623, 174, 175, 621, 622, 624, 625, 628, 178, 627, 178, 626, 178, 179, 626, 627, 628, 178, 179, 626, 627, 629, 630, 633, 182, 632, 182, 631, 182, 183, 631, 632, 633, 182, 183, 631, 632, 634, 635, 638, 186, 637, 186, 636, 186, 187, 636, 637, 638, 186, 187, 636, 637, 639, 640, 643, 190, 642, 190, 641, 190, 191, 641, 642, 643, 190, 191, 641, 642, 644, 645, 648, 194, 647, 194, 646, 194, 195, 646, 647, 648, 194, 195, 646, 647, 649, 650, 653, 198, 652, 198, 651, 198, 199, 651, 652, 653, 198, 199, 651, 652, 654, 655, 658, 202, 657, 202, 656, 202, 203, 656, 657, 658, 202, 203, 656, 657, 659, 660, 663, 206, 662, 206, 661, 206, 207, 661, 662, 663, 206, 207, 661, 662, 664, 665, 668, 210, 667, 210, 666, 210, 211, 666, 667, 668, 210, 211, 666, 667, 669, 670, 673, 214, 672, 214, 671, 214, 215, 671, 672, 673, 214, 215, 671, 672, 674, 675, 678, 218, 677, 218, 676, 218, 219, 676, 677, 678, 218, 219, 676, 677, 679, 680, 683, 222, 682, 222, 681, 222, 223, 681, 682, 683, 222, 223, 681, 682, 684, 685, 688, 226, 687, 226, 686, 226, 227, 686, 687, 688, 226, 227, 686, 687, 689, 690, 693, 230, 692, 230, 691, 230, 231, 691, 692, 693, 230, 231, 691, 692, 694, 695, 698, 234, 697, 234, 696, 234, 235, 696, 697, 698, 234, 235, 696, 697, 699, 700, 703, 238, 702, 238, 701, 238, 239, 701, 702, 703, 238, 239, 701, 702, 704, 705, 708, 242, 707, 242, 706, 242, 243, 706, 707, 708, 242, 243, 706, 707, 709, 710, 713, 246, 712, 246, 711, 246, 247, 711, 712, 713, 246, 247, 711, 712, 714, 715, 718, 250, 717, 250, 716, 250, 251, 716, 717, 718, 250, 251, 716, 717, 719, 720, 723, 254, 722, 254, 721, 254, 255, 721, 722, 723, 254, 255, 721, 722, 724, 725, 728, 258, 727, 258, 726, 258, 259, 726, 727, 728, 258, 259, 726, 727, 729, 730, 733, 262, 732, 262, 731, 262, 263, 731, 732, 733, 262, 263, 731, 732, 734, 735, 738, 266, 737, 266, 736, 266, 267, 736, 737, 738, 266, 267, 736, 737, 739, 740, 743, 270, 742, 270, 741, 270, 271, 741, 742, 743, 270, 271, 741, 742, 744, 745, 748, 274, 747, 274, 746, 274, 275, 746, 747, 748, 274, 275, 746, 747, 749, 750, 753, 278, 752, 278, 751, 278, 279, 751, 752, 753, 278, 279, 751, 752, 754, 755, 758, 282, 757, 282, 756, 282, 283, 756, 757, 758, 282, 283, 756, 757, 759, 760, 763, 286, 762, 286, 761, 286, 287, 761, 762, 763, 286, 287, 761, 762, 764, 765, 768, 290, 767, 290, 766, 290, 291, 766, 767, 768, 290, 291, 766, 767, 769, 770, 773, 294, 772, 294, 771, 294, 295, 771, 772, 773, 294, 295, 771, 772, 774, 775, 778, 298, 777, 298, 776, 298, 299, 776, 777, 778, 298, 299, 776, 777, 779, 780, 783, 302, 782, 302, 781, 302, 303, 781, 782, 783, 302, 303, 781, 782, 784, 785, 788, 306, 787, 306, 786, 306, 307, 786, 787, 788, 306, 307, 786, 787, 789, 790, 793, 310, 792, 310, 791, 310, 311, 791, 792, 793, 310, 311, 791, 792, 794, 795, 798, 314, 797, 314, 796, 314, 315, 796, 797, 798, 314, 315, 796, 797, 799, 800, 803, 318, 802, 318, 801, 318, 319, 801, 802, 803, 318, 319, 801, 802, 804, 805, 808, 322, 807, 322, 806, 322, 323, 806, 807, 808, 322, 323, 806, 807, 809, 810, 813, 326, 812, 326, 811, 326, 327, 811, 812, 813, 326, 327, 811, 812, 814, 815, 818, 330, 817, 330, 816, 330, 331, 816, 817, 818, 330, 331, 816, 817, 819, 820, 823, 334, 822, 334, 821, 334, 335, 821, 822, 823, 334, 335, 821, 822, 824, 825, 828, 338, 827, 338, 826, 338, 339, 826, 827, 828, 338, 339, 826, 827, 829, 830, 833, 342, 832, 342, 831, 342, 343, 831, 832, 833, 342, 343, 831, 832, 834, 835, 838, 346, 837, 346, 836, 346, 347, 836, 837, 838, 346, 347, 836, 837, 839, 840, 843, 350, 842, 350, 841, 350, 351, 841, 842, 843, 350, 351, 841, 842, 844, 845, 848, 354, 847, 354, 846, 354, 355, 846, 847, 848, 354, 355, 846, 847, 849, 850, 853, 358, 852, 358, 851, 358, 359, 851, 852, 853, 358, 359, 851, 852, 854, 855, 858, 362, 857, 362, 856, 362, 363, 856, 857, 858, 362, 363, 856, 857, 859, 860, 863, 366, 862, 366, 861, 366, 367, 861, 862, 863, 366, 367, 861, 862, 864, 865, 868, 370, 867, 370, 866, 370, 371, 866, 867, 868, 370, 371, 866, 867, 869, 870, 873, 374, 872, 374, 871, 374, 375, 871, 872, 873, 374, 375, 871, 872, 874, 875, 878, 378, 877, 378, 876, 378, 379, 876, 877, 878, 378, 379, 876, 877, 879, 880, 883, 382, 882, 382, 881, 382, 383, 881, 882, 883, 382, 383, 881, 882, 884, 885, 888, 386, 887, 386, 886, 386, 387, 886, 887, 888, 386, 387, 886, 887, 889, 890, 893, 390, 892, 390, 891, 390, 391, 891, 892, 893, 390, 391, 891, 892, 894, 895, 898, 394, 897, 394, 896, 394, 395, 896, 897, 898, 394, 395, 896, 897, 899, 900, 903, 398, 902, 398, 901, 398, 399, 901, 902, 903, 398, 399, 901, 902, 904, 905, 908, 402, 907, 402, 906, 402, 403, 906, 907, 908, 402, 403, 906, 907, 909, 910, 913, 406, 912, 406, 911, 406, 407, 911, 912, 913, 406, 407, 911, 912, 914, 410, 915, 3, 915, 916]
    sp_jac_run_ja = [0, 3, 4, 5, 7, 211, 415, 421, 427, 432, 437, 442, 447, 453, 459, 464, 469, 475, 481, 486, 491, 497, 503, 508, 513, 519, 525, 530, 535, 541, 547, 552, 557, 563, 569, 574, 579, 585, 591, 596, 601, 607, 613, 618, 623, 629, 635, 640, 645, 651, 657, 662, 667, 673, 679, 684, 689, 695, 701, 706, 711, 717, 723, 728, 733, 739, 745, 750, 755, 761, 767, 772, 777, 783, 789, 794, 799, 805, 811, 816, 821, 827, 833, 838, 843, 849, 855, 860, 865, 871, 877, 882, 887, 893, 899, 904, 909, 915, 921, 926, 931, 937, 943, 948, 953, 959, 965, 970, 975, 981, 987, 992, 997, 1003, 1009, 1014, 1019, 1025, 1031, 1036, 1041, 1047, 1053, 1058, 1063, 1069, 1075, 1080, 1085, 1091, 1097, 1102, 1107, 1113, 1119, 1124, 1129, 1135, 1141, 1146, 1151, 1157, 1163, 1168, 1173, 1179, 1185, 1190, 1195, 1201, 1207, 1212, 1217, 1223, 1229, 1234, 1239, 1245, 1251, 1256, 1261, 1267, 1273, 1278, 1283, 1289, 1295, 1300, 1305, 1311, 1317, 1322, 1327, 1333, 1339, 1344, 1349, 1355, 1361, 1366, 1371, 1377, 1383, 1388, 1393, 1399, 1405, 1410, 1415, 1421, 1427, 1432, 1437, 1443, 1449, 1454, 1459, 1465, 1471, 1476, 1481, 1487, 1493, 1498, 1503, 1509, 1515, 1520, 1525, 1531, 1537, 1542, 1547, 1553, 1559, 1564, 1569, 1575, 1581, 1586, 1591, 1597, 1603, 1608, 1613, 1619, 1625, 1630, 1635, 1641, 1647, 1652, 1657, 1663, 1669, 1674, 1679, 1685, 1691, 1696, 1701, 1707, 1713, 1718, 1723, 1729, 1735, 1740, 1745, 1751, 1757, 1762, 1767, 1773, 1779, 1784, 1789, 1795, 1801, 1806, 1811, 1817, 1823, 1828, 1833, 1839, 1845, 1850, 1855, 1861, 1867, 1872, 1877, 1883, 1889, 1894, 1899, 1905, 1911, 1916, 1921, 1927, 1933, 1938, 1943, 1949, 1955, 1960, 1965, 1971, 1977, 1982, 1987, 1993, 1999, 2004, 2009, 2015, 2021, 2026, 2031, 2037, 2043, 2048, 2053, 2059, 2065, 2070, 2075, 2081, 2087, 2092, 2097, 2103, 2109, 2114, 2119, 2125, 2131, 2136, 2141, 2147, 2153, 2158, 2163, 2169, 2175, 2180, 2185, 2191, 2197, 2202, 2207, 2213, 2219, 2224, 2229, 2235, 2241, 2246, 2251, 2257, 2263, 2268, 2273, 2279, 2285, 2290, 2295, 2301, 2307, 2312, 2317, 2323, 2329, 2334, 2339, 2345, 2351, 2356, 2361, 2367, 2373, 2378, 2383, 2389, 2395, 2400, 2405, 2411, 2417, 2422, 2427, 2433, 2439, 2444, 2449, 2455, 2461, 2466, 2471, 2477, 2483, 2488, 2493, 2499, 2505, 2510, 2515, 2521, 2527, 2532, 2537, 2543, 2549, 2554, 2559, 2565, 2571, 2576, 2581, 2587, 2593, 2598, 2603, 2609, 2615, 2620, 2625, 2631, 2637, 2639, 2644, 2650, 2656, 2662, 2664, 2666, 2668, 2673, 2678, 2680, 2682, 2684, 2689, 2694, 2696, 2698, 2700, 2705, 2710, 2712, 2714, 2716, 2721, 2726, 2728, 2730, 2732, 2737, 2742, 2744, 2746, 2748, 2753, 2758, 2760, 2762, 2764, 2769, 2774, 2776, 2778, 2780, 2785, 2790, 2792, 2794, 2796, 2801, 2806, 2808, 2810, 2812, 2817, 2822, 2824, 2826, 2828, 2833, 2838, 2840, 2842, 2844, 2849, 2854, 2856, 2858, 2860, 2865, 2870, 2872, 2874, 2876, 2881, 2886, 2888, 2890, 2892, 2897, 2902, 2904, 2906, 2908, 2913, 2918, 2920, 2922, 2924, 2929, 2934, 2936, 2938, 2940, 2945, 2950, 2952, 2954, 2956, 2961, 2966, 2968, 2970, 2972, 2977, 2982, 2984, 2986, 2988, 2993, 2998, 3000, 3002, 3004, 3009, 3014, 3016, 3018, 3020, 3025, 3030, 3032, 3034, 3036, 3041, 3046, 3048, 3050, 3052, 3057, 3062, 3064, 3066, 3068, 3073, 3078, 3080, 3082, 3084, 3089, 3094, 3096, 3098, 3100, 3105, 3110, 3112, 3114, 3116, 3121, 3126, 3128, 3130, 3132, 3137, 3142, 3144, 3146, 3148, 3153, 3158, 3160, 3162, 3164, 3169, 3174, 3176, 3178, 3180, 3185, 3190, 3192, 3194, 3196, 3201, 3206, 3208, 3210, 3212, 3217, 3222, 3224, 3226, 3228, 3233, 3238, 3240, 3242, 3244, 3249, 3254, 3256, 3258, 3260, 3265, 3270, 3272, 3274, 3276, 3281, 3286, 3288, 3290, 3292, 3297, 3302, 3304, 3306, 3308, 3313, 3318, 3320, 3322, 3324, 3329, 3334, 3336, 3338, 3340, 3345, 3350, 3352, 3354, 3356, 3361, 3366, 3368, 3370, 3372, 3377, 3382, 3384, 3386, 3388, 3393, 3398, 3400, 3402, 3404, 3409, 3414, 3416, 3418, 3420, 3425, 3430, 3432, 3434, 3436, 3441, 3446, 3448, 3450, 3452, 3457, 3462, 3464, 3466, 3468, 3473, 3478, 3480, 3482, 3484, 3489, 3494, 3496, 3498, 3500, 3505, 3510, 3512, 3514, 3516, 3521, 3526, 3528, 3530, 3532, 3537, 3542, 3544, 3546, 3548, 3553, 3558, 3560, 3562, 3564, 3569, 3574, 3576, 3578, 3580, 3585, 3590, 3592, 3594, 3596, 3601, 3606, 3608, 3610, 3612, 3617, 3622, 3624, 3626, 3628, 3633, 3638, 3640, 3642, 3644, 3649, 3654, 3656, 3658, 3660, 3665, 3670, 3672, 3674, 3676, 3681, 3686, 3688, 3690, 3692, 3697, 3702, 3704, 3706, 3708, 3713, 3718, 3720, 3722, 3724, 3729, 3734, 3736, 3738, 3740, 3745, 3750, 3752, 3754, 3756, 3761, 3766, 3768, 3770, 3772, 3777, 3782, 3784, 3786, 3788, 3793, 3798, 3800, 3802, 3804, 3809, 3814, 3816, 3818, 3820, 3825, 3830, 3832, 3834, 3836, 3841, 3846, 3848, 3850, 3852, 3857, 3862, 3864, 3866, 3868, 3873, 3878, 3880, 3882, 3884, 3889, 3894, 3896, 3898, 3900, 3905, 3910, 3912, 3914, 3916, 3921, 3926, 3928, 3930, 3932, 3937, 3942, 3944, 3946, 3948, 3953, 3958, 3960, 3962, 3964, 3969, 3974, 3976, 3978, 3980, 3985, 3990, 3992, 3994, 3996, 4001, 4006, 4008, 4010, 4012, 4017, 4022, 4024, 4026, 4028, 4033, 4038, 4040, 4042, 4044, 4049, 4054, 4056, 4058, 4060, 4065, 4070, 4072, 4074, 4076, 4081, 4086, 4088, 4090, 4092, 4097, 4102, 4104, 4106, 4108, 4113, 4118, 4120, 4122, 4124, 4129, 4134, 4136, 4138, 4140, 4145, 4150, 4152, 4154, 4156, 4161, 4166, 4168, 4170, 4172, 4177, 4182, 4184, 4186, 4188, 4193, 4198, 4200, 4202, 4204, 4209, 4214, 4216, 4218, 4220, 4225, 4230, 4232, 4234, 4236, 4241, 4246, 4248, 4250, 4252, 4257, 4262, 4264, 4267]
    sp_jac_run_nia = 917
    sp_jac_run_nja = 917
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 410, 915, 1, 2, 3, 915, 4, 5, 6, 7, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97, 100, 101, 104, 105, 108, 109, 112, 113, 116, 117, 120, 121, 124, 125, 128, 129, 132, 133, 136, 137, 140, 141, 144, 145, 148, 149, 152, 153, 156, 157, 160, 161, 164, 165, 168, 169, 172, 173, 176, 177, 180, 181, 184, 185, 188, 189, 192, 193, 196, 197, 200, 201, 204, 205, 208, 209, 212, 213, 216, 217, 220, 221, 224, 225, 228, 229, 232, 233, 236, 237, 240, 241, 244, 245, 248, 249, 252, 253, 256, 257, 260, 261, 264, 265, 268, 269, 272, 273, 276, 277, 280, 281, 284, 285, 288, 289, 292, 293, 296, 297, 300, 301, 304, 305, 308, 309, 312, 313, 316, 317, 320, 321, 324, 325, 328, 329, 332, 333, 336, 337, 340, 341, 344, 345, 348, 349, 352, 353, 356, 357, 360, 361, 364, 365, 368, 369, 372, 373, 376, 377, 380, 381, 384, 385, 388, 389, 392, 393, 396, 397, 400, 401, 404, 405, 408, 409, 4, 5, 6, 7, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37, 40, 41, 44, 45, 48, 49, 52, 53, 56, 57, 60, 61, 64, 65, 68, 69, 72, 73, 76, 77, 80, 81, 84, 85, 88, 89, 92, 93, 96, 97, 100, 101, 104, 105, 108, 109, 112, 113, 116, 117, 120, 121, 124, 125, 128, 129, 132, 133, 136, 137, 140, 141, 144, 145, 148, 149, 152, 153, 156, 157, 160, 161, 164, 165, 168, 169, 172, 173, 176, 177, 180, 181, 184, 185, 188, 189, 192, 193, 196, 197, 200, 201, 204, 205, 208, 209, 212, 213, 216, 217, 220, 221, 224, 225, 228, 229, 232, 233, 236, 237, 240, 241, 244, 245, 248, 249, 252, 253, 256, 257, 260, 261, 264, 265, 268, 269, 272, 273, 276, 277, 280, 281, 284, 285, 288, 289, 292, 293, 296, 297, 300, 301, 304, 305, 308, 309, 312, 313, 316, 317, 320, 321, 324, 325, 328, 329, 332, 333, 336, 337, 340, 341, 344, 345, 348, 349, 352, 353, 356, 357, 360, 361, 364, 365, 368, 369, 372, 373, 376, 377, 380, 381, 384, 385, 388, 389, 392, 393, 396, 397, 400, 401, 404, 405, 408, 409, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 6, 7, 8, 9, 413, 6, 7, 8, 9, 414, 10, 11, 12, 13, 418, 10, 11, 12, 13, 419, 4, 5, 10, 11, 12, 13, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 423, 14, 15, 16, 17, 424, 4, 5, 14, 15, 16, 17, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 428, 18, 19, 20, 21, 429, 4, 5, 18, 19, 20, 21, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 433, 22, 23, 24, 25, 434, 4, 5, 22, 23, 24, 25, 4, 5, 22, 23, 24, 25, 26, 27, 28, 29, 438, 26, 27, 28, 29, 439, 4, 5, 26, 27, 28, 29, 4, 5, 26, 27, 28, 29, 30, 31, 32, 33, 443, 30, 31, 32, 33, 444, 4, 5, 30, 31, 32, 33, 4, 5, 30, 31, 32, 33, 34, 35, 36, 37, 448, 34, 35, 36, 37, 449, 4, 5, 34, 35, 36, 37, 4, 5, 34, 35, 36, 37, 38, 39, 40, 41, 453, 38, 39, 40, 41, 454, 4, 5, 38, 39, 40, 41, 4, 5, 38, 39, 40, 41, 42, 43, 44, 45, 458, 42, 43, 44, 45, 459, 4, 5, 42, 43, 44, 45, 4, 5, 42, 43, 44, 45, 46, 47, 48, 49, 463, 46, 47, 48, 49, 464, 4, 5, 46, 47, 48, 49, 4, 5, 46, 47, 48, 49, 50, 51, 52, 53, 468, 50, 51, 52, 53, 469, 4, 5, 50, 51, 52, 53, 4, 5, 50, 51, 52, 53, 54, 55, 56, 57, 473, 54, 55, 56, 57, 474, 4, 5, 54, 55, 56, 57, 4, 5, 54, 55, 56, 57, 58, 59, 60, 61, 478, 58, 59, 60, 61, 479, 4, 5, 58, 59, 60, 61, 4, 5, 58, 59, 60, 61, 62, 63, 64, 65, 483, 62, 63, 64, 65, 484, 4, 5, 62, 63, 64, 65, 4, 5, 62, 63, 64, 65, 66, 67, 68, 69, 488, 66, 67, 68, 69, 489, 4, 5, 66, 67, 68, 69, 4, 5, 66, 67, 68, 69, 70, 71, 72, 73, 493, 70, 71, 72, 73, 494, 4, 5, 70, 71, 72, 73, 4, 5, 70, 71, 72, 73, 74, 75, 76, 77, 498, 74, 75, 76, 77, 499, 4, 5, 74, 75, 76, 77, 4, 5, 74, 75, 76, 77, 78, 79, 80, 81, 503, 78, 79, 80, 81, 504, 4, 5, 78, 79, 80, 81, 4, 5, 78, 79, 80, 81, 82, 83, 84, 85, 508, 82, 83, 84, 85, 509, 4, 5, 82, 83, 84, 85, 4, 5, 82, 83, 84, 85, 86, 87, 88, 89, 513, 86, 87, 88, 89, 514, 4, 5, 86, 87, 88, 89, 4, 5, 86, 87, 88, 89, 90, 91, 92, 93, 518, 90, 91, 92, 93, 519, 4, 5, 90, 91, 92, 93, 4, 5, 90, 91, 92, 93, 94, 95, 96, 97, 523, 94, 95, 96, 97, 524, 4, 5, 94, 95, 96, 97, 4, 5, 94, 95, 96, 97, 98, 99, 100, 101, 528, 98, 99, 100, 101, 529, 4, 5, 98, 99, 100, 101, 4, 5, 98, 99, 100, 101, 102, 103, 104, 105, 533, 102, 103, 104, 105, 534, 4, 5, 102, 103, 104, 105, 4, 5, 102, 103, 104, 105, 106, 107, 108, 109, 538, 106, 107, 108, 109, 539, 4, 5, 106, 107, 108, 109, 4, 5, 106, 107, 108, 109, 110, 111, 112, 113, 543, 110, 111, 112, 113, 544, 4, 5, 110, 111, 112, 113, 4, 5, 110, 111, 112, 113, 114, 115, 116, 117, 548, 114, 115, 116, 117, 549, 4, 5, 114, 115, 116, 117, 4, 5, 114, 115, 116, 117, 118, 119, 120, 121, 553, 118, 119, 120, 121, 554, 4, 5, 118, 119, 120, 121, 4, 5, 118, 119, 120, 121, 122, 123, 124, 125, 558, 122, 123, 124, 125, 559, 4, 5, 122, 123, 124, 125, 4, 5, 122, 123, 124, 125, 126, 127, 128, 129, 563, 126, 127, 128, 129, 564, 4, 5, 126, 127, 128, 129, 4, 5, 126, 127, 128, 129, 130, 131, 132, 133, 568, 130, 131, 132, 133, 569, 4, 5, 130, 131, 132, 133, 4, 5, 130, 131, 132, 133, 134, 135, 136, 137, 573, 134, 135, 136, 137, 574, 4, 5, 134, 135, 136, 137, 4, 5, 134, 135, 136, 137, 138, 139, 140, 141, 578, 138, 139, 140, 141, 579, 4, 5, 138, 139, 140, 141, 4, 5, 138, 139, 140, 141, 142, 143, 144, 145, 583, 142, 143, 144, 145, 584, 4, 5, 142, 143, 144, 145, 4, 5, 142, 143, 144, 145, 146, 147, 148, 149, 588, 146, 147, 148, 149, 589, 4, 5, 146, 147, 148, 149, 4, 5, 146, 147, 148, 149, 150, 151, 152, 153, 593, 150, 151, 152, 153, 594, 4, 5, 150, 151, 152, 153, 4, 5, 150, 151, 152, 153, 154, 155, 156, 157, 598, 154, 155, 156, 157, 599, 4, 5, 154, 155, 156, 157, 4, 5, 154, 155, 156, 157, 158, 159, 160, 161, 603, 158, 159, 160, 161, 604, 4, 5, 158, 159, 160, 161, 4, 5, 158, 159, 160, 161, 162, 163, 164, 165, 608, 162, 163, 164, 165, 609, 4, 5, 162, 163, 164, 165, 4, 5, 162, 163, 164, 165, 166, 167, 168, 169, 613, 166, 167, 168, 169, 614, 4, 5, 166, 167, 168, 169, 4, 5, 166, 167, 168, 169, 170, 171, 172, 173, 618, 170, 171, 172, 173, 619, 4, 5, 170, 171, 172, 173, 4, 5, 170, 171, 172, 173, 174, 175, 176, 177, 623, 174, 175, 176, 177, 624, 4, 5, 174, 175, 176, 177, 4, 5, 174, 175, 176, 177, 178, 179, 180, 181, 628, 178, 179, 180, 181, 629, 4, 5, 178, 179, 180, 181, 4, 5, 178, 179, 180, 181, 182, 183, 184, 185, 633, 182, 183, 184, 185, 634, 4, 5, 182, 183, 184, 185, 4, 5, 182, 183, 184, 185, 186, 187, 188, 189, 638, 186, 187, 188, 189, 639, 4, 5, 186, 187, 188, 189, 4, 5, 186, 187, 188, 189, 190, 191, 192, 193, 643, 190, 191, 192, 193, 644, 4, 5, 190, 191, 192, 193, 4, 5, 190, 191, 192, 193, 194, 195, 196, 197, 648, 194, 195, 196, 197, 649, 4, 5, 194, 195, 196, 197, 4, 5, 194, 195, 196, 197, 198, 199, 200, 201, 653, 198, 199, 200, 201, 654, 4, 5, 198, 199, 200, 201, 4, 5, 198, 199, 200, 201, 202, 203, 204, 205, 658, 202, 203, 204, 205, 659, 4, 5, 202, 203, 204, 205, 4, 5, 202, 203, 204, 205, 206, 207, 208, 209, 663, 206, 207, 208, 209, 664, 4, 5, 206, 207, 208, 209, 4, 5, 206, 207, 208, 209, 210, 211, 212, 213, 668, 210, 211, 212, 213, 669, 4, 5, 210, 211, 212, 213, 4, 5, 210, 211, 212, 213, 214, 215, 216, 217, 673, 214, 215, 216, 217, 674, 4, 5, 214, 215, 216, 217, 4, 5, 214, 215, 216, 217, 218, 219, 220, 221, 678, 218, 219, 220, 221, 679, 4, 5, 218, 219, 220, 221, 4, 5, 218, 219, 220, 221, 222, 223, 224, 225, 683, 222, 223, 224, 225, 684, 4, 5, 222, 223, 224, 225, 4, 5, 222, 223, 224, 225, 226, 227, 228, 229, 688, 226, 227, 228, 229, 689, 4, 5, 226, 227, 228, 229, 4, 5, 226, 227, 228, 229, 230, 231, 232, 233, 693, 230, 231, 232, 233, 694, 4, 5, 230, 231, 232, 233, 4, 5, 230, 231, 232, 233, 234, 235, 236, 237, 698, 234, 235, 236, 237, 699, 4, 5, 234, 235, 236, 237, 4, 5, 234, 235, 236, 237, 238, 239, 240, 241, 703, 238, 239, 240, 241, 704, 4, 5, 238, 239, 240, 241, 4, 5, 238, 239, 240, 241, 242, 243, 244, 245, 708, 242, 243, 244, 245, 709, 4, 5, 242, 243, 244, 245, 4, 5, 242, 243, 244, 245, 246, 247, 248, 249, 713, 246, 247, 248, 249, 714, 4, 5, 246, 247, 248, 249, 4, 5, 246, 247, 248, 249, 250, 251, 252, 253, 718, 250, 251, 252, 253, 719, 4, 5, 250, 251, 252, 253, 4, 5, 250, 251, 252, 253, 254, 255, 256, 257, 723, 254, 255, 256, 257, 724, 4, 5, 254, 255, 256, 257, 4, 5, 254, 255, 256, 257, 258, 259, 260, 261, 728, 258, 259, 260, 261, 729, 4, 5, 258, 259, 260, 261, 4, 5, 258, 259, 260, 261, 262, 263, 264, 265, 733, 262, 263, 264, 265, 734, 4, 5, 262, 263, 264, 265, 4, 5, 262, 263, 264, 265, 266, 267, 268, 269, 738, 266, 267, 268, 269, 739, 4, 5, 266, 267, 268, 269, 4, 5, 266, 267, 268, 269, 270, 271, 272, 273, 743, 270, 271, 272, 273, 744, 4, 5, 270, 271, 272, 273, 4, 5, 270, 271, 272, 273, 274, 275, 276, 277, 748, 274, 275, 276, 277, 749, 4, 5, 274, 275, 276, 277, 4, 5, 274, 275, 276, 277, 278, 279, 280, 281, 753, 278, 279, 280, 281, 754, 4, 5, 278, 279, 280, 281, 4, 5, 278, 279, 280, 281, 282, 283, 284, 285, 758, 282, 283, 284, 285, 759, 4, 5, 282, 283, 284, 285, 4, 5, 282, 283, 284, 285, 286, 287, 288, 289, 763, 286, 287, 288, 289, 764, 4, 5, 286, 287, 288, 289, 4, 5, 286, 287, 288, 289, 290, 291, 292, 293, 768, 290, 291, 292, 293, 769, 4, 5, 290, 291, 292, 293, 4, 5, 290, 291, 292, 293, 294, 295, 296, 297, 773, 294, 295, 296, 297, 774, 4, 5, 294, 295, 296, 297, 4, 5, 294, 295, 296, 297, 298, 299, 300, 301, 778, 298, 299, 300, 301, 779, 4, 5, 298, 299, 300, 301, 4, 5, 298, 299, 300, 301, 302, 303, 304, 305, 783, 302, 303, 304, 305, 784, 4, 5, 302, 303, 304, 305, 4, 5, 302, 303, 304, 305, 306, 307, 308, 309, 788, 306, 307, 308, 309, 789, 4, 5, 306, 307, 308, 309, 4, 5, 306, 307, 308, 309, 310, 311, 312, 313, 793, 310, 311, 312, 313, 794, 4, 5, 310, 311, 312, 313, 4, 5, 310, 311, 312, 313, 314, 315, 316, 317, 798, 314, 315, 316, 317, 799, 4, 5, 314, 315, 316, 317, 4, 5, 314, 315, 316, 317, 318, 319, 320, 321, 803, 318, 319, 320, 321, 804, 4, 5, 318, 319, 320, 321, 4, 5, 318, 319, 320, 321, 322, 323, 324, 325, 808, 322, 323, 324, 325, 809, 4, 5, 322, 323, 324, 325, 4, 5, 322, 323, 324, 325, 326, 327, 328, 329, 813, 326, 327, 328, 329, 814, 4, 5, 326, 327, 328, 329, 4, 5, 326, 327, 328, 329, 330, 331, 332, 333, 818, 330, 331, 332, 333, 819, 4, 5, 330, 331, 332, 333, 4, 5, 330, 331, 332, 333, 334, 335, 336, 337, 823, 334, 335, 336, 337, 824, 4, 5, 334, 335, 336, 337, 4, 5, 334, 335, 336, 337, 338, 339, 340, 341, 828, 338, 339, 340, 341, 829, 4, 5, 338, 339, 340, 341, 4, 5, 338, 339, 340, 341, 342, 343, 344, 345, 833, 342, 343, 344, 345, 834, 4, 5, 342, 343, 344, 345, 4, 5, 342, 343, 344, 345, 346, 347, 348, 349, 838, 346, 347, 348, 349, 839, 4, 5, 346, 347, 348, 349, 4, 5, 346, 347, 348, 349, 350, 351, 352, 353, 843, 350, 351, 352, 353, 844, 4, 5, 350, 351, 352, 353, 4, 5, 350, 351, 352, 353, 354, 355, 356, 357, 848, 354, 355, 356, 357, 849, 4, 5, 354, 355, 356, 357, 4, 5, 354, 355, 356, 357, 358, 359, 360, 361, 853, 358, 359, 360, 361, 854, 4, 5, 358, 359, 360, 361, 4, 5, 358, 359, 360, 361, 362, 363, 364, 365, 858, 362, 363, 364, 365, 859, 4, 5, 362, 363, 364, 365, 4, 5, 362, 363, 364, 365, 366, 367, 368, 369, 863, 366, 367, 368, 369, 864, 4, 5, 366, 367, 368, 369, 4, 5, 366, 367, 368, 369, 370, 371, 372, 373, 868, 370, 371, 372, 373, 869, 4, 5, 370, 371, 372, 373, 4, 5, 370, 371, 372, 373, 374, 375, 376, 377, 873, 374, 375, 376, 377, 874, 4, 5, 374, 375, 376, 377, 4, 5, 374, 375, 376, 377, 378, 379, 380, 381, 878, 378, 379, 380, 381, 879, 4, 5, 378, 379, 380, 381, 4, 5, 378, 379, 380, 381, 382, 383, 384, 385, 883, 382, 383, 384, 385, 884, 4, 5, 382, 383, 384, 385, 4, 5, 382, 383, 384, 385, 386, 387, 388, 389, 888, 386, 387, 388, 389, 889, 4, 5, 386, 387, 388, 389, 4, 5, 386, 387, 388, 389, 390, 391, 392, 393, 893, 390, 391, 392, 393, 894, 4, 5, 390, 391, 392, 393, 4, 5, 390, 391, 392, 393, 394, 395, 396, 397, 898, 394, 395, 396, 397, 899, 4, 5, 394, 395, 396, 397, 4, 5, 394, 395, 396, 397, 398, 399, 400, 401, 903, 398, 399, 400, 401, 904, 4, 5, 398, 399, 400, 401, 4, 5, 398, 399, 400, 401, 402, 403, 404, 405, 908, 402, 403, 404, 405, 909, 4, 5, 402, 403, 404, 405, 4, 5, 402, 403, 404, 405, 406, 407, 408, 409, 913, 406, 407, 408, 409, 914, 4, 5, 406, 407, 408, 409, 4, 5, 406, 407, 408, 409, 1, 410, 0, 8, 9, 411, 412, 0, 2, 8, 9, 411, 412, 0, 8, 9, 411, 412, 413, 0, 8, 9, 411, 412, 414, 415, 418, 10, 417, 10, 416, 10, 11, 416, 417, 418, 10, 11, 416, 417, 419, 420, 423, 14, 422, 14, 421, 14, 15, 421, 422, 423, 14, 15, 421, 422, 424, 425, 428, 18, 427, 18, 426, 18, 19, 426, 427, 428, 18, 19, 426, 427, 429, 430, 433, 22, 432, 22, 431, 22, 23, 431, 432, 433, 22, 23, 431, 432, 434, 435, 438, 26, 437, 26, 436, 26, 27, 436, 437, 438, 26, 27, 436, 437, 439, 440, 443, 30, 442, 30, 441, 30, 31, 441, 442, 443, 30, 31, 441, 442, 444, 445, 448, 34, 447, 34, 446, 34, 35, 446, 447, 448, 34, 35, 446, 447, 449, 450, 453, 38, 452, 38, 451, 38, 39, 451, 452, 453, 38, 39, 451, 452, 454, 455, 458, 42, 457, 42, 456, 42, 43, 456, 457, 458, 42, 43, 456, 457, 459, 460, 463, 46, 462, 46, 461, 46, 47, 461, 462, 463, 46, 47, 461, 462, 464, 465, 468, 50, 467, 50, 466, 50, 51, 466, 467, 468, 50, 51, 466, 467, 469, 470, 473, 54, 472, 54, 471, 54, 55, 471, 472, 473, 54, 55, 471, 472, 474, 475, 478, 58, 477, 58, 476, 58, 59, 476, 477, 478, 58, 59, 476, 477, 479, 480, 483, 62, 482, 62, 481, 62, 63, 481, 482, 483, 62, 63, 481, 482, 484, 485, 488, 66, 487, 66, 486, 66, 67, 486, 487, 488, 66, 67, 486, 487, 489, 490, 493, 70, 492, 70, 491, 70, 71, 491, 492, 493, 70, 71, 491, 492, 494, 495, 498, 74, 497, 74, 496, 74, 75, 496, 497, 498, 74, 75, 496, 497, 499, 500, 503, 78, 502, 78, 501, 78, 79, 501, 502, 503, 78, 79, 501, 502, 504, 505, 508, 82, 507, 82, 506, 82, 83, 506, 507, 508, 82, 83, 506, 507, 509, 510, 513, 86, 512, 86, 511, 86, 87, 511, 512, 513, 86, 87, 511, 512, 514, 515, 518, 90, 517, 90, 516, 90, 91, 516, 517, 518, 90, 91, 516, 517, 519, 520, 523, 94, 522, 94, 521, 94, 95, 521, 522, 523, 94, 95, 521, 522, 524, 525, 528, 98, 527, 98, 526, 98, 99, 526, 527, 528, 98, 99, 526, 527, 529, 530, 533, 102, 532, 102, 531, 102, 103, 531, 532, 533, 102, 103, 531, 532, 534, 535, 538, 106, 537, 106, 536, 106, 107, 536, 537, 538, 106, 107, 536, 537, 539, 540, 543, 110, 542, 110, 541, 110, 111, 541, 542, 543, 110, 111, 541, 542, 544, 545, 548, 114, 547, 114, 546, 114, 115, 546, 547, 548, 114, 115, 546, 547, 549, 550, 553, 118, 552, 118, 551, 118, 119, 551, 552, 553, 118, 119, 551, 552, 554, 555, 558, 122, 557, 122, 556, 122, 123, 556, 557, 558, 122, 123, 556, 557, 559, 560, 563, 126, 562, 126, 561, 126, 127, 561, 562, 563, 126, 127, 561, 562, 564, 565, 568, 130, 567, 130, 566, 130, 131, 566, 567, 568, 130, 131, 566, 567, 569, 570, 573, 134, 572, 134, 571, 134, 135, 571, 572, 573, 134, 135, 571, 572, 574, 575, 578, 138, 577, 138, 576, 138, 139, 576, 577, 578, 138, 139, 576, 577, 579, 580, 583, 142, 582, 142, 581, 142, 143, 581, 582, 583, 142, 143, 581, 582, 584, 585, 588, 146, 587, 146, 586, 146, 147, 586, 587, 588, 146, 147, 586, 587, 589, 590, 593, 150, 592, 150, 591, 150, 151, 591, 592, 593, 150, 151, 591, 592, 594, 595, 598, 154, 597, 154, 596, 154, 155, 596, 597, 598, 154, 155, 596, 597, 599, 600, 603, 158, 602, 158, 601, 158, 159, 601, 602, 603, 158, 159, 601, 602, 604, 605, 608, 162, 607, 162, 606, 162, 163, 606, 607, 608, 162, 163, 606, 607, 609, 610, 613, 166, 612, 166, 611, 166, 167, 611, 612, 613, 166, 167, 611, 612, 614, 615, 618, 170, 617, 170, 616, 170, 171, 616, 617, 618, 170, 171, 616, 617, 619, 620, 623, 174, 622, 174, 621, 174, 175, 621, 622, 623, 174, 175, 621, 622, 624, 625, 628, 178, 627, 178, 626, 178, 179, 626, 627, 628, 178, 179, 626, 627, 629, 630, 633, 182, 632, 182, 631, 182, 183, 631, 632, 633, 182, 183, 631, 632, 634, 635, 638, 186, 637, 186, 636, 186, 187, 636, 637, 638, 186, 187, 636, 637, 639, 640, 643, 190, 642, 190, 641, 190, 191, 641, 642, 643, 190, 191, 641, 642, 644, 645, 648, 194, 647, 194, 646, 194, 195, 646, 647, 648, 194, 195, 646, 647, 649, 650, 653, 198, 652, 198, 651, 198, 199, 651, 652, 653, 198, 199, 651, 652, 654, 655, 658, 202, 657, 202, 656, 202, 203, 656, 657, 658, 202, 203, 656, 657, 659, 660, 663, 206, 662, 206, 661, 206, 207, 661, 662, 663, 206, 207, 661, 662, 664, 665, 668, 210, 667, 210, 666, 210, 211, 666, 667, 668, 210, 211, 666, 667, 669, 670, 673, 214, 672, 214, 671, 214, 215, 671, 672, 673, 214, 215, 671, 672, 674, 675, 678, 218, 677, 218, 676, 218, 219, 676, 677, 678, 218, 219, 676, 677, 679, 680, 683, 222, 682, 222, 681, 222, 223, 681, 682, 683, 222, 223, 681, 682, 684, 685, 688, 226, 687, 226, 686, 226, 227, 686, 687, 688, 226, 227, 686, 687, 689, 690, 693, 230, 692, 230, 691, 230, 231, 691, 692, 693, 230, 231, 691, 692, 694, 695, 698, 234, 697, 234, 696, 234, 235, 696, 697, 698, 234, 235, 696, 697, 699, 700, 703, 238, 702, 238, 701, 238, 239, 701, 702, 703, 238, 239, 701, 702, 704, 705, 708, 242, 707, 242, 706, 242, 243, 706, 707, 708, 242, 243, 706, 707, 709, 710, 713, 246, 712, 246, 711, 246, 247, 711, 712, 713, 246, 247, 711, 712, 714, 715, 718, 250, 717, 250, 716, 250, 251, 716, 717, 718, 250, 251, 716, 717, 719, 720, 723, 254, 722, 254, 721, 254, 255, 721, 722, 723, 254, 255, 721, 722, 724, 725, 728, 258, 727, 258, 726, 258, 259, 726, 727, 728, 258, 259, 726, 727, 729, 730, 733, 262, 732, 262, 731, 262, 263, 731, 732, 733, 262, 263, 731, 732, 734, 735, 738, 266, 737, 266, 736, 266, 267, 736, 737, 738, 266, 267, 736, 737, 739, 740, 743, 270, 742, 270, 741, 270, 271, 741, 742, 743, 270, 271, 741, 742, 744, 745, 748, 274, 747, 274, 746, 274, 275, 746, 747, 748, 274, 275, 746, 747, 749, 750, 753, 278, 752, 278, 751, 278, 279, 751, 752, 753, 278, 279, 751, 752, 754, 755, 758, 282, 757, 282, 756, 282, 283, 756, 757, 758, 282, 283, 756, 757, 759, 760, 763, 286, 762, 286, 761, 286, 287, 761, 762, 763, 286, 287, 761, 762, 764, 765, 768, 290, 767, 290, 766, 290, 291, 766, 767, 768, 290, 291, 766, 767, 769, 770, 773, 294, 772, 294, 771, 294, 295, 771, 772, 773, 294, 295, 771, 772, 774, 775, 778, 298, 777, 298, 776, 298, 299, 776, 777, 778, 298, 299, 776, 777, 779, 780, 783, 302, 782, 302, 781, 302, 303, 781, 782, 783, 302, 303, 781, 782, 784, 785, 788, 306, 787, 306, 786, 306, 307, 786, 787, 788, 306, 307, 786, 787, 789, 790, 793, 310, 792, 310, 791, 310, 311, 791, 792, 793, 310, 311, 791, 792, 794, 795, 798, 314, 797, 314, 796, 314, 315, 796, 797, 798, 314, 315, 796, 797, 799, 800, 803, 318, 802, 318, 801, 318, 319, 801, 802, 803, 318, 319, 801, 802, 804, 805, 808, 322, 807, 322, 806, 322, 323, 806, 807, 808, 322, 323, 806, 807, 809, 810, 813, 326, 812, 326, 811, 326, 327, 811, 812, 813, 326, 327, 811, 812, 814, 815, 818, 330, 817, 330, 816, 330, 331, 816, 817, 818, 330, 331, 816, 817, 819, 820, 823, 334, 822, 334, 821, 334, 335, 821, 822, 823, 334, 335, 821, 822, 824, 825, 828, 338, 827, 338, 826, 338, 339, 826, 827, 828, 338, 339, 826, 827, 829, 830, 833, 342, 832, 342, 831, 342, 343, 831, 832, 833, 342, 343, 831, 832, 834, 835, 838, 346, 837, 346, 836, 346, 347, 836, 837, 838, 346, 347, 836, 837, 839, 840, 843, 350, 842, 350, 841, 350, 351, 841, 842, 843, 350, 351, 841, 842, 844, 845, 848, 354, 847, 354, 846, 354, 355, 846, 847, 848, 354, 355, 846, 847, 849, 850, 853, 358, 852, 358, 851, 358, 359, 851, 852, 853, 358, 359, 851, 852, 854, 855, 858, 362, 857, 362, 856, 362, 363, 856, 857, 858, 362, 363, 856, 857, 859, 860, 863, 366, 862, 366, 861, 366, 367, 861, 862, 863, 366, 367, 861, 862, 864, 865, 868, 370, 867, 370, 866, 370, 371, 866, 867, 868, 370, 371, 866, 867, 869, 870, 873, 374, 872, 374, 871, 374, 375, 871, 872, 873, 374, 375, 871, 872, 874, 875, 878, 378, 877, 378, 876, 378, 379, 876, 877, 878, 378, 379, 876, 877, 879, 880, 883, 382, 882, 382, 881, 382, 383, 881, 882, 883, 382, 383, 881, 882, 884, 885, 888, 386, 887, 386, 886, 386, 387, 886, 887, 888, 386, 387, 886, 887, 889, 890, 893, 390, 892, 390, 891, 390, 391, 891, 892, 893, 390, 391, 891, 892, 894, 895, 898, 394, 897, 394, 896, 394, 395, 896, 897, 898, 394, 395, 896, 897, 899, 900, 903, 398, 902, 398, 901, 398, 399, 901, 902, 903, 398, 399, 901, 902, 904, 905, 908, 402, 907, 402, 906, 402, 403, 906, 907, 908, 402, 403, 906, 907, 909, 910, 913, 406, 912, 406, 911, 406, 407, 911, 912, 913, 406, 407, 911, 912, 914, 410, 915, 3, 915, 916]
    sp_jac_trap_ja = [0, 3, 4, 5, 7, 211, 415, 421, 427, 432, 437, 442, 447, 453, 459, 464, 469, 475, 481, 486, 491, 497, 503, 508, 513, 519, 525, 530, 535, 541, 547, 552, 557, 563, 569, 574, 579, 585, 591, 596, 601, 607, 613, 618, 623, 629, 635, 640, 645, 651, 657, 662, 667, 673, 679, 684, 689, 695, 701, 706, 711, 717, 723, 728, 733, 739, 745, 750, 755, 761, 767, 772, 777, 783, 789, 794, 799, 805, 811, 816, 821, 827, 833, 838, 843, 849, 855, 860, 865, 871, 877, 882, 887, 893, 899, 904, 909, 915, 921, 926, 931, 937, 943, 948, 953, 959, 965, 970, 975, 981, 987, 992, 997, 1003, 1009, 1014, 1019, 1025, 1031, 1036, 1041, 1047, 1053, 1058, 1063, 1069, 1075, 1080, 1085, 1091, 1097, 1102, 1107, 1113, 1119, 1124, 1129, 1135, 1141, 1146, 1151, 1157, 1163, 1168, 1173, 1179, 1185, 1190, 1195, 1201, 1207, 1212, 1217, 1223, 1229, 1234, 1239, 1245, 1251, 1256, 1261, 1267, 1273, 1278, 1283, 1289, 1295, 1300, 1305, 1311, 1317, 1322, 1327, 1333, 1339, 1344, 1349, 1355, 1361, 1366, 1371, 1377, 1383, 1388, 1393, 1399, 1405, 1410, 1415, 1421, 1427, 1432, 1437, 1443, 1449, 1454, 1459, 1465, 1471, 1476, 1481, 1487, 1493, 1498, 1503, 1509, 1515, 1520, 1525, 1531, 1537, 1542, 1547, 1553, 1559, 1564, 1569, 1575, 1581, 1586, 1591, 1597, 1603, 1608, 1613, 1619, 1625, 1630, 1635, 1641, 1647, 1652, 1657, 1663, 1669, 1674, 1679, 1685, 1691, 1696, 1701, 1707, 1713, 1718, 1723, 1729, 1735, 1740, 1745, 1751, 1757, 1762, 1767, 1773, 1779, 1784, 1789, 1795, 1801, 1806, 1811, 1817, 1823, 1828, 1833, 1839, 1845, 1850, 1855, 1861, 1867, 1872, 1877, 1883, 1889, 1894, 1899, 1905, 1911, 1916, 1921, 1927, 1933, 1938, 1943, 1949, 1955, 1960, 1965, 1971, 1977, 1982, 1987, 1993, 1999, 2004, 2009, 2015, 2021, 2026, 2031, 2037, 2043, 2048, 2053, 2059, 2065, 2070, 2075, 2081, 2087, 2092, 2097, 2103, 2109, 2114, 2119, 2125, 2131, 2136, 2141, 2147, 2153, 2158, 2163, 2169, 2175, 2180, 2185, 2191, 2197, 2202, 2207, 2213, 2219, 2224, 2229, 2235, 2241, 2246, 2251, 2257, 2263, 2268, 2273, 2279, 2285, 2290, 2295, 2301, 2307, 2312, 2317, 2323, 2329, 2334, 2339, 2345, 2351, 2356, 2361, 2367, 2373, 2378, 2383, 2389, 2395, 2400, 2405, 2411, 2417, 2422, 2427, 2433, 2439, 2444, 2449, 2455, 2461, 2466, 2471, 2477, 2483, 2488, 2493, 2499, 2505, 2510, 2515, 2521, 2527, 2532, 2537, 2543, 2549, 2554, 2559, 2565, 2571, 2576, 2581, 2587, 2593, 2598, 2603, 2609, 2615, 2620, 2625, 2631, 2637, 2639, 2644, 2650, 2656, 2662, 2664, 2666, 2668, 2673, 2678, 2680, 2682, 2684, 2689, 2694, 2696, 2698, 2700, 2705, 2710, 2712, 2714, 2716, 2721, 2726, 2728, 2730, 2732, 2737, 2742, 2744, 2746, 2748, 2753, 2758, 2760, 2762, 2764, 2769, 2774, 2776, 2778, 2780, 2785, 2790, 2792, 2794, 2796, 2801, 2806, 2808, 2810, 2812, 2817, 2822, 2824, 2826, 2828, 2833, 2838, 2840, 2842, 2844, 2849, 2854, 2856, 2858, 2860, 2865, 2870, 2872, 2874, 2876, 2881, 2886, 2888, 2890, 2892, 2897, 2902, 2904, 2906, 2908, 2913, 2918, 2920, 2922, 2924, 2929, 2934, 2936, 2938, 2940, 2945, 2950, 2952, 2954, 2956, 2961, 2966, 2968, 2970, 2972, 2977, 2982, 2984, 2986, 2988, 2993, 2998, 3000, 3002, 3004, 3009, 3014, 3016, 3018, 3020, 3025, 3030, 3032, 3034, 3036, 3041, 3046, 3048, 3050, 3052, 3057, 3062, 3064, 3066, 3068, 3073, 3078, 3080, 3082, 3084, 3089, 3094, 3096, 3098, 3100, 3105, 3110, 3112, 3114, 3116, 3121, 3126, 3128, 3130, 3132, 3137, 3142, 3144, 3146, 3148, 3153, 3158, 3160, 3162, 3164, 3169, 3174, 3176, 3178, 3180, 3185, 3190, 3192, 3194, 3196, 3201, 3206, 3208, 3210, 3212, 3217, 3222, 3224, 3226, 3228, 3233, 3238, 3240, 3242, 3244, 3249, 3254, 3256, 3258, 3260, 3265, 3270, 3272, 3274, 3276, 3281, 3286, 3288, 3290, 3292, 3297, 3302, 3304, 3306, 3308, 3313, 3318, 3320, 3322, 3324, 3329, 3334, 3336, 3338, 3340, 3345, 3350, 3352, 3354, 3356, 3361, 3366, 3368, 3370, 3372, 3377, 3382, 3384, 3386, 3388, 3393, 3398, 3400, 3402, 3404, 3409, 3414, 3416, 3418, 3420, 3425, 3430, 3432, 3434, 3436, 3441, 3446, 3448, 3450, 3452, 3457, 3462, 3464, 3466, 3468, 3473, 3478, 3480, 3482, 3484, 3489, 3494, 3496, 3498, 3500, 3505, 3510, 3512, 3514, 3516, 3521, 3526, 3528, 3530, 3532, 3537, 3542, 3544, 3546, 3548, 3553, 3558, 3560, 3562, 3564, 3569, 3574, 3576, 3578, 3580, 3585, 3590, 3592, 3594, 3596, 3601, 3606, 3608, 3610, 3612, 3617, 3622, 3624, 3626, 3628, 3633, 3638, 3640, 3642, 3644, 3649, 3654, 3656, 3658, 3660, 3665, 3670, 3672, 3674, 3676, 3681, 3686, 3688, 3690, 3692, 3697, 3702, 3704, 3706, 3708, 3713, 3718, 3720, 3722, 3724, 3729, 3734, 3736, 3738, 3740, 3745, 3750, 3752, 3754, 3756, 3761, 3766, 3768, 3770, 3772, 3777, 3782, 3784, 3786, 3788, 3793, 3798, 3800, 3802, 3804, 3809, 3814, 3816, 3818, 3820, 3825, 3830, 3832, 3834, 3836, 3841, 3846, 3848, 3850, 3852, 3857, 3862, 3864, 3866, 3868, 3873, 3878, 3880, 3882, 3884, 3889, 3894, 3896, 3898, 3900, 3905, 3910, 3912, 3914, 3916, 3921, 3926, 3928, 3930, 3932, 3937, 3942, 3944, 3946, 3948, 3953, 3958, 3960, 3962, 3964, 3969, 3974, 3976, 3978, 3980, 3985, 3990, 3992, 3994, 3996, 4001, 4006, 4008, 4010, 4012, 4017, 4022, 4024, 4026, 4028, 4033, 4038, 4040, 4042, 4044, 4049, 4054, 4056, 4058, 4060, 4065, 4070, 4072, 4074, 4076, 4081, 4086, 4088, 4090, 4092, 4097, 4102, 4104, 4106, 4108, 4113, 4118, 4120, 4122, 4124, 4129, 4134, 4136, 4138, 4140, 4145, 4150, 4152, 4154, 4156, 4161, 4166, 4168, 4170, 4172, 4177, 4182, 4184, 4186, 4188, 4193, 4198, 4200, 4202, 4204, 4209, 4214, 4216, 4218, 4220, 4225, 4230, 4232, 4234, 4236, 4241, 4246, 4248, 4250, 4252, 4257, 4262, 4264, 4267]
    sp_jac_trap_nia = 917
    sp_jac_trap_nja = 917
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
