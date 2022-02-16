import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import cigre_eu_lv_com_cffi as jacs

cffi_support.register_module(jacs)
f_ini_eval = jacs.lib.f_ini_eval
g_ini_eval = jacs.lib.g_ini_eval
f_run_eval = jacs.lib.f_run_eval
g_run_eval = jacs.lib.g_run_eval
h_eval  = jacs.lib.h_eval

de_jac_ini_xy_eval = jacs.lib.de_jac_ini_xy_eval
de_jac_ini_up_eval = jacs.lib.de_jac_ini_up_eval
de_jac_ini_num_eval = jacs.lib.de_jac_ini_num_eval

sp_jac_ini_xy_eval = jacs.lib.sp_jac_ini_xy_eval
sp_jac_ini_up_eval = jacs.lib.sp_jac_ini_up_eval
sp_jac_ini_num_eval = jacs.lib.sp_jac_ini_num_eval

de_jac_run_xy_eval = jacs.lib.de_jac_run_xy_eval
de_jac_run_up_eval = jacs.lib.de_jac_run_up_eval
de_jac_run_num_eval = jacs.lib.de_jac_run_num_eval

sp_jac_run_xy_eval = jacs.lib.sp_jac_run_xy_eval
sp_jac_run_up_eval = jacs.lib.sp_jac_run_up_eval
sp_jac_run_num_eval = jacs.lib.sp_jac_run_num_eval

de_jac_trap_xy_eval= jacs.lib.de_jac_trap_xy_eval            
de_jac_trap_up_eval= jacs.lib.de_jac_trap_up_eval        
de_jac_trap_num_eval= jacs.lib.de_jac_trap_num_eval

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


class cigre_eu_lv_com_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 1
        self.N_y = 224 
        self.N_z = 249 
        self.N_store = 10000 
        self.params_list = [] 
        self.params_values_list  = [] 
        self.inputs_ini_list = ['v_C00_a_r', 'v_C00_a_i', 'v_C00_b_r', 'v_C00_b_i', 'v_C00_c_r', 'v_C00_c_i', 'i_C01_a_r', 'i_C01_a_i', 'i_C01_b_r', 'i_C01_b_i', 'i_C01_c_r', 'i_C01_c_i', 'i_C01_n_r', 'i_C01_n_i', 'i_C12_a_r', 'i_C12_a_i', 'i_C12_b_r', 'i_C12_b_i', 'i_C12_c_r', 'i_C12_c_i', 'i_C12_n_r', 'i_C12_n_i', 'i_C13_a_r', 'i_C13_a_i', 'i_C13_b_r', 'i_C13_b_i', 'i_C13_c_r', 'i_C13_c_i', 'i_C13_n_r', 'i_C13_n_i', 'i_C14_a_r', 'i_C14_a_i', 'i_C14_b_r', 'i_C14_b_i', 'i_C14_c_r', 'i_C14_c_i', 'i_C14_n_r', 'i_C14_n_i', 'i_C17_a_r', 'i_C17_a_i', 'i_C17_b_r', 'i_C17_b_i', 'i_C17_c_r', 'i_C17_c_i', 'i_C17_n_r', 'i_C17_n_i', 'i_C18_a_r', 'i_C18_a_i', 'i_C18_b_r', 'i_C18_b_i', 'i_C18_c_r', 'i_C18_c_i', 'i_C18_n_r', 'i_C18_n_i', 'i_C19_a_r', 'i_C19_a_i', 'i_C19_b_r', 'i_C19_b_i', 'i_C19_c_r', 'i_C19_c_i', 'i_C19_n_r', 'i_C19_n_i', 'i_C20_a_r', 'i_C20_a_i', 'i_C20_b_r', 'i_C20_b_i', 'i_C20_c_r', 'i_C20_c_i', 'i_C20_n_r', 'i_C20_n_i', 'i_C02_a_r', 'i_C02_a_i', 'i_C02_b_r', 'i_C02_b_i', 'i_C02_c_r', 'i_C02_c_i', 'i_C02_n_r', 'i_C02_n_i', 'i_C03_a_r', 'i_C03_a_i', 'i_C03_b_r', 'i_C03_b_i', 'i_C03_c_r', 'i_C03_c_i', 'i_C03_n_r', 'i_C03_n_i', 'i_C04_a_r', 'i_C04_a_i', 'i_C04_b_r', 'i_C04_b_i', 'i_C04_c_r', 'i_C04_c_i', 'i_C04_n_r', 'i_C04_n_i', 'i_C05_a_r', 'i_C05_a_i', 'i_C05_b_r', 'i_C05_b_i', 'i_C05_c_r', 'i_C05_c_i', 'i_C05_n_r', 'i_C05_n_i', 'i_C06_a_r', 'i_C06_a_i', 'i_C06_b_r', 'i_C06_b_i', 'i_C06_c_r', 'i_C06_c_i', 'i_C06_n_r', 'i_C06_n_i', 'i_C07_a_r', 'i_C07_a_i', 'i_C07_b_r', 'i_C07_b_i', 'i_C07_c_r', 'i_C07_c_i', 'i_C07_n_r', 'i_C07_n_i', 'i_C08_a_r', 'i_C08_a_i', 'i_C08_b_r', 'i_C08_b_i', 'i_C08_c_r', 'i_C08_c_i', 'i_C08_n_r', 'i_C08_n_i', 'i_C09_a_r', 'i_C09_a_i', 'i_C09_b_r', 'i_C09_b_i', 'i_C09_c_r', 'i_C09_c_i', 'i_C09_n_r', 'i_C09_n_i', 'i_C10_a_r', 'i_C10_a_i', 'i_C10_b_r', 'i_C10_b_i', 'i_C10_c_r', 'i_C10_c_i', 'i_C10_n_r', 'i_C10_n_i', 'i_C11_a_r', 'i_C11_a_i', 'i_C11_b_r', 'i_C11_b_i', 'i_C11_c_r', 'i_C11_c_i', 'i_C11_n_r', 'i_C11_n_i', 'i_C15_a_r', 'i_C15_a_i', 'i_C15_b_r', 'i_C15_b_i', 'i_C15_c_r', 'i_C15_c_i', 'i_C15_n_r', 'i_C15_n_i', 'i_C16_a_r', 'i_C16_a_i', 'i_C16_b_r', 'i_C16_b_i', 'i_C16_c_r', 'i_C16_c_i', 'i_C16_n_r', 'i_C16_n_i', 'p_load_C01_a', 'q_load_C01_a', 'p_load_C01_b', 'q_load_C01_b', 'p_load_C01_c', 'q_load_C01_c', 'p_load_C12_a', 'q_load_C12_a', 'p_load_C12_b', 'q_load_C12_b', 'p_load_C12_c', 'q_load_C12_c', 'p_load_C13_a', 'q_load_C13_a', 'p_load_C13_b', 'q_load_C13_b', 'p_load_C13_c', 'q_load_C13_c', 'p_load_C14_a', 'q_load_C14_a', 'p_load_C14_b', 'q_load_C14_b', 'p_load_C14_c', 'q_load_C14_c', 'p_load_C17_a', 'q_load_C17_a', 'p_load_C17_b', 'q_load_C17_b', 'p_load_C17_c', 'q_load_C17_c', 'p_load_C18_a', 'q_load_C18_a', 'p_load_C18_b', 'q_load_C18_b', 'p_load_C18_c', 'q_load_C18_c', 'p_load_C19_a', 'q_load_C19_a', 'p_load_C19_b', 'q_load_C19_b', 'p_load_C19_c', 'q_load_C19_c', 'p_load_C20_a', 'q_load_C20_a', 'p_load_C20_b', 'q_load_C20_b', 'p_load_C20_c', 'q_load_C20_c', 'u_dummy'] 
        self.inputs_ini_values_list  = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35999.99999927632, 17435.59577397324, 36000.00000015082, 17435.5957743769, 36000.00000051905, 17435.595774156773, 5999.999999179263, 2905.9326295761994, 6000.000000078992, 2905.93262862606, 6000.000000709402, 2905.9326288557913, 5999.999999179263, 2905.9326295761994, 6000.000000078992, 2905.93262862606, 6000.000000709402, 2905.9326288557913, 7499.999999117883, 3632.415786829063, 7500.000000094685, 3632.4157858770086, 7500.000000752402, 3632.4157861176946, 7499.999998756757, 3632.415787119777, 7500.000000141894, 3632.4157856514057, 7500.000001050942, 3632.415786043513, 2399.9999996792817, 1162.3730517916867, 2400.000000045395, 1162.3730514632152, 2400.000000261788, 1162.3730515649215, 4799.999999379609, 2324.7461035279844, 4800.000000101085, 2324.74610294963, 4800.000000490497, 2324.746103156994, 2399.9999996960905, 1162.3730517514823, 2400.0000000526043, 1162.373051481001, 2400.0000002371744, 1162.3730515841944, 1.0] 
        self.inputs_run_list = ['v_C00_a_r', 'v_C00_a_i', 'v_C00_b_r', 'v_C00_b_i', 'v_C00_c_r', 'v_C00_c_i', 'i_C01_a_r', 'i_C01_a_i', 'i_C01_b_r', 'i_C01_b_i', 'i_C01_c_r', 'i_C01_c_i', 'i_C01_n_r', 'i_C01_n_i', 'i_C12_a_r', 'i_C12_a_i', 'i_C12_b_r', 'i_C12_b_i', 'i_C12_c_r', 'i_C12_c_i', 'i_C12_n_r', 'i_C12_n_i', 'i_C13_a_r', 'i_C13_a_i', 'i_C13_b_r', 'i_C13_b_i', 'i_C13_c_r', 'i_C13_c_i', 'i_C13_n_r', 'i_C13_n_i', 'i_C14_a_r', 'i_C14_a_i', 'i_C14_b_r', 'i_C14_b_i', 'i_C14_c_r', 'i_C14_c_i', 'i_C14_n_r', 'i_C14_n_i', 'i_C17_a_r', 'i_C17_a_i', 'i_C17_b_r', 'i_C17_b_i', 'i_C17_c_r', 'i_C17_c_i', 'i_C17_n_r', 'i_C17_n_i', 'i_C18_a_r', 'i_C18_a_i', 'i_C18_b_r', 'i_C18_b_i', 'i_C18_c_r', 'i_C18_c_i', 'i_C18_n_r', 'i_C18_n_i', 'i_C19_a_r', 'i_C19_a_i', 'i_C19_b_r', 'i_C19_b_i', 'i_C19_c_r', 'i_C19_c_i', 'i_C19_n_r', 'i_C19_n_i', 'i_C20_a_r', 'i_C20_a_i', 'i_C20_b_r', 'i_C20_b_i', 'i_C20_c_r', 'i_C20_c_i', 'i_C20_n_r', 'i_C20_n_i', 'i_C02_a_r', 'i_C02_a_i', 'i_C02_b_r', 'i_C02_b_i', 'i_C02_c_r', 'i_C02_c_i', 'i_C02_n_r', 'i_C02_n_i', 'i_C03_a_r', 'i_C03_a_i', 'i_C03_b_r', 'i_C03_b_i', 'i_C03_c_r', 'i_C03_c_i', 'i_C03_n_r', 'i_C03_n_i', 'i_C04_a_r', 'i_C04_a_i', 'i_C04_b_r', 'i_C04_b_i', 'i_C04_c_r', 'i_C04_c_i', 'i_C04_n_r', 'i_C04_n_i', 'i_C05_a_r', 'i_C05_a_i', 'i_C05_b_r', 'i_C05_b_i', 'i_C05_c_r', 'i_C05_c_i', 'i_C05_n_r', 'i_C05_n_i', 'i_C06_a_r', 'i_C06_a_i', 'i_C06_b_r', 'i_C06_b_i', 'i_C06_c_r', 'i_C06_c_i', 'i_C06_n_r', 'i_C06_n_i', 'i_C07_a_r', 'i_C07_a_i', 'i_C07_b_r', 'i_C07_b_i', 'i_C07_c_r', 'i_C07_c_i', 'i_C07_n_r', 'i_C07_n_i', 'i_C08_a_r', 'i_C08_a_i', 'i_C08_b_r', 'i_C08_b_i', 'i_C08_c_r', 'i_C08_c_i', 'i_C08_n_r', 'i_C08_n_i', 'i_C09_a_r', 'i_C09_a_i', 'i_C09_b_r', 'i_C09_b_i', 'i_C09_c_r', 'i_C09_c_i', 'i_C09_n_r', 'i_C09_n_i', 'i_C10_a_r', 'i_C10_a_i', 'i_C10_b_r', 'i_C10_b_i', 'i_C10_c_r', 'i_C10_c_i', 'i_C10_n_r', 'i_C10_n_i', 'i_C11_a_r', 'i_C11_a_i', 'i_C11_b_r', 'i_C11_b_i', 'i_C11_c_r', 'i_C11_c_i', 'i_C11_n_r', 'i_C11_n_i', 'i_C15_a_r', 'i_C15_a_i', 'i_C15_b_r', 'i_C15_b_i', 'i_C15_c_r', 'i_C15_c_i', 'i_C15_n_r', 'i_C15_n_i', 'i_C16_a_r', 'i_C16_a_i', 'i_C16_b_r', 'i_C16_b_i', 'i_C16_c_r', 'i_C16_c_i', 'i_C16_n_r', 'i_C16_n_i', 'p_load_C01_a', 'q_load_C01_a', 'p_load_C01_b', 'q_load_C01_b', 'p_load_C01_c', 'q_load_C01_c', 'p_load_C12_a', 'q_load_C12_a', 'p_load_C12_b', 'q_load_C12_b', 'p_load_C12_c', 'q_load_C12_c', 'p_load_C13_a', 'q_load_C13_a', 'p_load_C13_b', 'q_load_C13_b', 'p_load_C13_c', 'q_load_C13_c', 'p_load_C14_a', 'q_load_C14_a', 'p_load_C14_b', 'q_load_C14_b', 'p_load_C14_c', 'q_load_C14_c', 'p_load_C17_a', 'q_load_C17_a', 'p_load_C17_b', 'q_load_C17_b', 'p_load_C17_c', 'q_load_C17_c', 'p_load_C18_a', 'q_load_C18_a', 'p_load_C18_b', 'q_load_C18_b', 'p_load_C18_c', 'q_load_C18_c', 'p_load_C19_a', 'q_load_C19_a', 'p_load_C19_b', 'q_load_C19_b', 'p_load_C19_c', 'q_load_C19_c', 'p_load_C20_a', 'q_load_C20_a', 'p_load_C20_b', 'q_load_C20_b', 'p_load_C20_c', 'q_load_C20_c', 'u_dummy'] 
        self.inputs_run_values_list = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35999.99999927632, 17435.59577397324, 36000.00000015082, 17435.5957743769, 36000.00000051905, 17435.595774156773, 5999.999999179263, 2905.9326295761994, 6000.000000078992, 2905.93262862606, 6000.000000709402, 2905.9326288557913, 5999.999999179263, 2905.9326295761994, 6000.000000078992, 2905.93262862606, 6000.000000709402, 2905.9326288557913, 7499.999999117883, 3632.415786829063, 7500.000000094685, 3632.4157858770086, 7500.000000752402, 3632.4157861176946, 7499.999998756757, 3632.415787119777, 7500.000000141894, 3632.4157856514057, 7500.000001050942, 3632.415786043513, 2399.9999996792817, 1162.3730517916867, 2400.000000045395, 1162.3730514632152, 2400.000000261788, 1162.3730515649215, 4799.999999379609, 2324.7461035279844, 4800.000000101085, 2324.74610294963, 4800.000000490497, 2324.746103156994, 2399.9999996960905, 1162.3730517514823, 2400.0000000526043, 1162.373051481001, 2400.0000002371744, 1162.3730515841944, 1.0] 
        self.outputs_list = ['i_t_C00_C01_1_a_r', 'i_t_C00_C01_1_a_i', 'i_t_C00_C01_1_b_r', 'i_t_C00_C01_1_b_i', 'i_t_C00_C01_1_c_r', 'i_t_C00_C01_1_c_i', 'i_t_C00_C01_2_a_r', 'i_t_C00_C01_2_a_i', 'i_t_C00_C01_2_b_r', 'i_t_C00_C01_2_b_i', 'i_t_C00_C01_2_c_r', 'i_t_C00_C01_2_c_i', 'i_t_C00_C01_2_n_r', 'i_t_C00_C01_2_n_i', 'i_l_C01_C02_a_r', 'i_l_C01_C02_a_i', 'i_l_C01_C02_b_r', 'i_l_C01_C02_b_i', 'i_l_C01_C02_c_r', 'i_l_C01_C02_c_i', 'i_l_C01_C02_n_r', 'i_l_C01_C02_n_i', 'i_l_C02_C03_a_r', 'i_l_C02_C03_a_i', 'i_l_C02_C03_b_r', 'i_l_C02_C03_b_i', 'i_l_C02_C03_c_r', 'i_l_C02_C03_c_i', 'i_l_C02_C03_n_r', 'i_l_C02_C03_n_i', 'i_l_C03_C04_a_r', 'i_l_C03_C04_a_i', 'i_l_C03_C04_b_r', 'i_l_C03_C04_b_i', 'i_l_C03_C04_c_r', 'i_l_C03_C04_c_i', 'i_l_C03_C04_n_r', 'i_l_C03_C04_n_i', 'i_l_C04_C05_a_r', 'i_l_C04_C05_a_i', 'i_l_C04_C05_b_r', 'i_l_C04_C05_b_i', 'i_l_C04_C05_c_r', 'i_l_C04_C05_c_i', 'i_l_C04_C05_n_r', 'i_l_C04_C05_n_i', 'i_l_C05_C06_a_r', 'i_l_C05_C06_a_i', 'i_l_C05_C06_b_r', 'i_l_C05_C06_b_i', 'i_l_C05_C06_c_r', 'i_l_C05_C06_c_i', 'i_l_C05_C06_n_r', 'i_l_C05_C06_n_i', 'i_l_C06_C07_a_r', 'i_l_C06_C07_a_i', 'i_l_C06_C07_b_r', 'i_l_C06_C07_b_i', 'i_l_C06_C07_c_r', 'i_l_C06_C07_c_i', 'i_l_C06_C07_n_r', 'i_l_C06_C07_n_i', 'i_l_C07_C08_a_r', 'i_l_C07_C08_a_i', 'i_l_C07_C08_b_r', 'i_l_C07_C08_b_i', 'i_l_C07_C08_c_r', 'i_l_C07_C08_c_i', 'i_l_C07_C08_n_r', 'i_l_C07_C08_n_i', 'i_l_C08_C09_a_r', 'i_l_C08_C09_a_i', 'i_l_C08_C09_b_r', 'i_l_C08_C09_b_i', 'i_l_C08_C09_c_r', 'i_l_C08_C09_c_i', 'i_l_C08_C09_n_r', 'i_l_C08_C09_n_i', 'i_l_C03_C10_a_r', 'i_l_C03_C10_a_i', 'i_l_C03_C10_b_r', 'i_l_C03_C10_b_i', 'i_l_C03_C10_c_r', 'i_l_C03_C10_c_i', 'i_l_C03_C10_n_r', 'i_l_C03_C10_n_i', 'i_l_C10_C11_a_r', 'i_l_C10_C11_a_i', 'i_l_C10_C11_b_r', 'i_l_C10_C11_b_i', 'i_l_C10_C11_c_r', 'i_l_C10_C11_c_i', 'i_l_C10_C11_n_r', 'i_l_C10_C11_n_i', 'i_l_C11_C12_a_r', 'i_l_C11_C12_a_i', 'i_l_C11_C12_b_r', 'i_l_C11_C12_b_i', 'i_l_C11_C12_c_r', 'i_l_C11_C12_c_i', 'i_l_C11_C12_n_r', 'i_l_C11_C12_n_i', 'i_l_C11_C13_a_r', 'i_l_C11_C13_a_i', 'i_l_C11_C13_b_r', 'i_l_C11_C13_b_i', 'i_l_C11_C13_c_r', 'i_l_C11_C13_c_i', 'i_l_C11_C13_n_r', 'i_l_C11_C13_n_i', 'i_l_C10_C14_a_r', 'i_l_C10_C14_a_i', 'i_l_C10_C14_b_r', 'i_l_C10_C14_b_i', 'i_l_C10_C14_c_r', 'i_l_C10_C14_c_i', 'i_l_C10_C14_n_r', 'i_l_C10_C14_n_i', 'i_l_C05_C15_a_r', 'i_l_C05_C15_a_i', 'i_l_C05_C15_b_r', 'i_l_C05_C15_b_i', 'i_l_C05_C15_c_r', 'i_l_C05_C15_c_i', 'i_l_C05_C15_n_r', 'i_l_C05_C15_n_i', 'i_l_C15_C16_a_r', 'i_l_C15_C16_a_i', 'i_l_C15_C16_b_r', 'i_l_C15_C16_b_i', 'i_l_C15_C16_c_r', 'i_l_C15_C16_c_i', 'i_l_C15_C16_n_r', 'i_l_C15_C16_n_i', 'i_l_C15_C18_a_r', 'i_l_C15_C18_a_i', 'i_l_C15_C18_b_r', 'i_l_C15_C18_b_i', 'i_l_C15_C18_c_r', 'i_l_C15_C18_c_i', 'i_l_C15_C18_n_r', 'i_l_C15_C18_n_i', 'i_l_C16_C17_a_r', 'i_l_C16_C17_a_i', 'i_l_C16_C17_b_r', 'i_l_C16_C17_b_i', 'i_l_C16_C17_c_r', 'i_l_C16_C17_c_i', 'i_l_C16_C17_n_r', 'i_l_C16_C17_n_i', 'i_l_C08_C19_a_r', 'i_l_C08_C19_a_i', 'i_l_C08_C19_b_r', 'i_l_C08_C19_b_i', 'i_l_C08_C19_c_r', 'i_l_C08_C19_c_i', 'i_l_C08_C19_n_r', 'i_l_C08_C19_n_i', 'i_l_C09_C20_a_r', 'i_l_C09_C20_a_i', 'i_l_C09_C20_b_r', 'i_l_C09_C20_b_i', 'i_l_C09_C20_c_r', 'i_l_C09_C20_c_i', 'i_l_C09_C20_n_r', 'i_l_C09_C20_n_i', 'v_C00_a_m', 'v_C00_b_m', 'v_C00_c_m', 'v_C01_a_m', 'v_C01_b_m', 'v_C01_c_m', 'v_C01_n_m', 'v_C12_a_m', 'v_C12_b_m', 'v_C12_c_m', 'v_C12_n_m', 'v_C13_a_m', 'v_C13_b_m', 'v_C13_c_m', 'v_C13_n_m', 'v_C14_a_m', 'v_C14_b_m', 'v_C14_c_m', 'v_C14_n_m', 'v_C17_a_m', 'v_C17_b_m', 'v_C17_c_m', 'v_C17_n_m', 'v_C18_a_m', 'v_C18_b_m', 'v_C18_c_m', 'v_C18_n_m', 'v_C19_a_m', 'v_C19_b_m', 'v_C19_c_m', 'v_C19_n_m', 'v_C20_a_m', 'v_C20_b_m', 'v_C20_c_m', 'v_C20_n_m', 'v_C02_a_m', 'v_C02_b_m', 'v_C02_c_m', 'v_C02_n_m', 'v_C03_a_m', 'v_C03_b_m', 'v_C03_c_m', 'v_C03_n_m', 'v_C04_a_m', 'v_C04_b_m', 'v_C04_c_m', 'v_C04_n_m', 'v_C05_a_m', 'v_C05_b_m', 'v_C05_c_m', 'v_C05_n_m', 'v_C06_a_m', 'v_C06_b_m', 'v_C06_c_m', 'v_C06_n_m', 'v_C07_a_m', 'v_C07_b_m', 'v_C07_c_m', 'v_C07_n_m', 'v_C08_a_m', 'v_C08_b_m', 'v_C08_c_m', 'v_C08_n_m', 'v_C09_a_m', 'v_C09_b_m', 'v_C09_c_m', 'v_C09_n_m', 'v_C10_a_m', 'v_C10_b_m', 'v_C10_c_m', 'v_C10_n_m', 'v_C11_a_m', 'v_C11_b_m', 'v_C11_c_m', 'v_C11_n_m', 'v_C15_a_m', 'v_C15_b_m', 'v_C15_c_m', 'v_C15_n_m', 'v_C16_a_m', 'v_C16_b_m', 'v_C16_c_m', 'v_C16_n_m'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_C01_a_r', 'v_C01_a_i', 'v_C01_b_r', 'v_C01_b_i', 'v_C01_c_r', 'v_C01_c_i', 'v_C01_n_r', 'v_C01_n_i', 'v_C12_a_r', 'v_C12_a_i', 'v_C12_b_r', 'v_C12_b_i', 'v_C12_c_r', 'v_C12_c_i', 'v_C12_n_r', 'v_C12_n_i', 'v_C13_a_r', 'v_C13_a_i', 'v_C13_b_r', 'v_C13_b_i', 'v_C13_c_r', 'v_C13_c_i', 'v_C13_n_r', 'v_C13_n_i', 'v_C14_a_r', 'v_C14_a_i', 'v_C14_b_r', 'v_C14_b_i', 'v_C14_c_r', 'v_C14_c_i', 'v_C14_n_r', 'v_C14_n_i', 'v_C17_a_r', 'v_C17_a_i', 'v_C17_b_r', 'v_C17_b_i', 'v_C17_c_r', 'v_C17_c_i', 'v_C17_n_r', 'v_C17_n_i', 'v_C18_a_r', 'v_C18_a_i', 'v_C18_b_r', 'v_C18_b_i', 'v_C18_c_r', 'v_C18_c_i', 'v_C18_n_r', 'v_C18_n_i', 'v_C19_a_r', 'v_C19_a_i', 'v_C19_b_r', 'v_C19_b_i', 'v_C19_c_r', 'v_C19_c_i', 'v_C19_n_r', 'v_C19_n_i', 'v_C20_a_r', 'v_C20_a_i', 'v_C20_b_r', 'v_C20_b_i', 'v_C20_c_r', 'v_C20_c_i', 'v_C20_n_r', 'v_C20_n_i', 'v_C02_a_r', 'v_C02_a_i', 'v_C02_b_r', 'v_C02_b_i', 'v_C02_c_r', 'v_C02_c_i', 'v_C02_n_r', 'v_C02_n_i', 'v_C03_a_r', 'v_C03_a_i', 'v_C03_b_r', 'v_C03_b_i', 'v_C03_c_r', 'v_C03_c_i', 'v_C03_n_r', 'v_C03_n_i', 'v_C04_a_r', 'v_C04_a_i', 'v_C04_b_r', 'v_C04_b_i', 'v_C04_c_r', 'v_C04_c_i', 'v_C04_n_r', 'v_C04_n_i', 'v_C05_a_r', 'v_C05_a_i', 'v_C05_b_r', 'v_C05_b_i', 'v_C05_c_r', 'v_C05_c_i', 'v_C05_n_r', 'v_C05_n_i', 'v_C06_a_r', 'v_C06_a_i', 'v_C06_b_r', 'v_C06_b_i', 'v_C06_c_r', 'v_C06_c_i', 'v_C06_n_r', 'v_C06_n_i', 'v_C07_a_r', 'v_C07_a_i', 'v_C07_b_r', 'v_C07_b_i', 'v_C07_c_r', 'v_C07_c_i', 'v_C07_n_r', 'v_C07_n_i', 'v_C08_a_r', 'v_C08_a_i', 'v_C08_b_r', 'v_C08_b_i', 'v_C08_c_r', 'v_C08_c_i', 'v_C08_n_r', 'v_C08_n_i', 'v_C09_a_r', 'v_C09_a_i', 'v_C09_b_r', 'v_C09_b_i', 'v_C09_c_r', 'v_C09_c_i', 'v_C09_n_r', 'v_C09_n_i', 'v_C10_a_r', 'v_C10_a_i', 'v_C10_b_r', 'v_C10_b_i', 'v_C10_c_r', 'v_C10_c_i', 'v_C10_n_r', 'v_C10_n_i', 'v_C11_a_r', 'v_C11_a_i', 'v_C11_b_r', 'v_C11_b_i', 'v_C11_c_r', 'v_C11_c_i', 'v_C11_n_r', 'v_C11_n_i', 'v_C15_a_r', 'v_C15_a_i', 'v_C15_b_r', 'v_C15_b_i', 'v_C15_c_r', 'v_C15_c_i', 'v_C15_n_r', 'v_C15_n_i', 'v_C16_a_r', 'v_C16_a_i', 'v_C16_b_r', 'v_C16_b_i', 'v_C16_c_r', 'v_C16_c_i', 'v_C16_n_r', 'v_C16_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_C01_a_r', 'v_C01_a_i', 'v_C01_b_r', 'v_C01_b_i', 'v_C01_c_r', 'v_C01_c_i', 'v_C01_n_r', 'v_C01_n_i', 'v_C12_a_r', 'v_C12_a_i', 'v_C12_b_r', 'v_C12_b_i', 'v_C12_c_r', 'v_C12_c_i', 'v_C12_n_r', 'v_C12_n_i', 'v_C13_a_r', 'v_C13_a_i', 'v_C13_b_r', 'v_C13_b_i', 'v_C13_c_r', 'v_C13_c_i', 'v_C13_n_r', 'v_C13_n_i', 'v_C14_a_r', 'v_C14_a_i', 'v_C14_b_r', 'v_C14_b_i', 'v_C14_c_r', 'v_C14_c_i', 'v_C14_n_r', 'v_C14_n_i', 'v_C17_a_r', 'v_C17_a_i', 'v_C17_b_r', 'v_C17_b_i', 'v_C17_c_r', 'v_C17_c_i', 'v_C17_n_r', 'v_C17_n_i', 'v_C18_a_r', 'v_C18_a_i', 'v_C18_b_r', 'v_C18_b_i', 'v_C18_c_r', 'v_C18_c_i', 'v_C18_n_r', 'v_C18_n_i', 'v_C19_a_r', 'v_C19_a_i', 'v_C19_b_r', 'v_C19_b_i', 'v_C19_c_r', 'v_C19_c_i', 'v_C19_n_r', 'v_C19_n_i', 'v_C20_a_r', 'v_C20_a_i', 'v_C20_b_r', 'v_C20_b_i', 'v_C20_c_r', 'v_C20_c_i', 'v_C20_n_r', 'v_C20_n_i', 'v_C02_a_r', 'v_C02_a_i', 'v_C02_b_r', 'v_C02_b_i', 'v_C02_c_r', 'v_C02_c_i', 'v_C02_n_r', 'v_C02_n_i', 'v_C03_a_r', 'v_C03_a_i', 'v_C03_b_r', 'v_C03_b_i', 'v_C03_c_r', 'v_C03_c_i', 'v_C03_n_r', 'v_C03_n_i', 'v_C04_a_r', 'v_C04_a_i', 'v_C04_b_r', 'v_C04_b_i', 'v_C04_c_r', 'v_C04_c_i', 'v_C04_n_r', 'v_C04_n_i', 'v_C05_a_r', 'v_C05_a_i', 'v_C05_b_r', 'v_C05_b_i', 'v_C05_c_r', 'v_C05_c_i', 'v_C05_n_r', 'v_C05_n_i', 'v_C06_a_r', 'v_C06_a_i', 'v_C06_b_r', 'v_C06_b_i', 'v_C06_c_r', 'v_C06_c_i', 'v_C06_n_r', 'v_C06_n_i', 'v_C07_a_r', 'v_C07_a_i', 'v_C07_b_r', 'v_C07_b_i', 'v_C07_c_r', 'v_C07_c_i', 'v_C07_n_r', 'v_C07_n_i', 'v_C08_a_r', 'v_C08_a_i', 'v_C08_b_r', 'v_C08_b_i', 'v_C08_c_r', 'v_C08_c_i', 'v_C08_n_r', 'v_C08_n_i', 'v_C09_a_r', 'v_C09_a_i', 'v_C09_b_r', 'v_C09_b_i', 'v_C09_c_r', 'v_C09_c_i', 'v_C09_n_r', 'v_C09_n_i', 'v_C10_a_r', 'v_C10_a_i', 'v_C10_b_r', 'v_C10_b_i', 'v_C10_c_r', 'v_C10_c_i', 'v_C10_n_r', 'v_C10_n_i', 'v_C11_a_r', 'v_C11_a_i', 'v_C11_b_r', 'v_C11_b_i', 'v_C11_c_r', 'v_C11_c_i', 'v_C11_n_r', 'v_C11_n_i', 'v_C15_a_r', 'v_C15_a_i', 'v_C15_b_r', 'v_C15_b_i', 'v_C15_c_r', 'v_C15_c_i', 'v_C15_n_r', 'v_C15_n_i', 'v_C16_a_r', 'v_C16_a_i', 'v_C16_b_r', 'v_C16_b_i', 'v_C16_c_r', 'v_C16_c_i', 'v_C16_n_r', 'v_C16_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i'] 
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
        self.u_ini = np.array(self.inputs_ini_values_list)
        self.p = np.array(self.params_values_list)
        self.xy_0 = np.zeros((self.N_x+self.N_y,))
        self.xy = np.zeros((self.N_x+self.N_y,))
        self.z = np.zeros((self.N_z,))
        
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
        self.sp_jac_ini_ia, self.sp_jac_ini_ja, self.sp_jac_ini_nia, self.sp_jac_ini_nja = sp_jac_ini_vectors()
        data = np.array(self.sp_jac_ini_ia,dtype=np.float64)
        #self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))
        self.sp_jac_ini = sspa.load_npz('cigre_eu_lv_com_sp_jac_ini_num.npz')
        self.jac_ini = self.sp_jac_ini.toarray()

        self.J_ini_d = np.array(self.sp_jac_ini_ia)*0.0
        self.J_ini_i = np.array(self.sp_jac_ini_ia)
        self.J_ini_p = np.array(self.sp_jac_ini_ja)
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_ini_eval(self.J_ini_d,x,y,self.u_ini,self.p,self.Dt) 
        self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 100,1e-10,'basic'       


        ## jac_run
        self.jac_run = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_run_ia, self.sp_jac_run_ja, self.sp_jac_run_nia, self.sp_jac_run_nja = sp_jac_run_vectors()
        data = np.array(self.sp_jac_run_ia,dtype=np.float64)
        #self.sp_jac_run = sspa.csr_matrix((data, self.sp_jac_run_ia, self.sp_jac_run_ja), shape=(self.sp_jac_run_nia,self.sp_jac_run_nja))
        self.sp_jac_run = sspa.load_npz('cigre_eu_lv_com_sp_jac_run_num.npz')
        self.jac_run = self.sp_jac_run.toarray()

        self.J_run_d = np.array(self.sp_jac_run_ia)*0.0
        self.J_run_i = np.array(self.sp_jac_run_ia)
        self.J_run_p = np.array(self.sp_jac_run_ja)
        de_jac_run_eval(self.jac_run,x,y,self.u_run,self.p,self.Dt)
        sp_jac_run_eval(self.J_run_d,x,y,self.u_run,self.p,self.Dt)
        
        ## jac_trap
        self.jac_trap = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_trap_ia, self.sp_jac_trap_ja, self.sp_jac_trap_nia, self.sp_jac_trap_nja = sp_jac_trap_vectors()
        data = np.array(self.sp_jac_trap_ia,dtype=np.float64)
        #self.sp_jac_trap = sspa.csr_matrix((data, self.sp_jac_trap_ia, self.sp_jac_trap_ja), shape=(self.sp_jac_trap_nia,self.sp_jac_trap_nja))
        self.sp_jac_trap = sspa.load_npz('cigre_eu_lv_com_sp_jac_trap_num.npz')
        self.jac_trap = self.sp_jac_trap.toarray()
        
        self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        self.J_trap_i = np.array(self.sp_jac_trap_ia)
        self.J_trap_p = np.array(self.sp_jac_trap_ja)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
        sp_jac_trap_eval(self.J_trap_d,x,y,self.u_run,self.p,self.Dt)
        self.fill_factor_trap,self.drop_tol_trap,self.drop_rule_trap = 100,1e-10,'basic' 
   

        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp=50,1e-8,1.0
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.0

        
 
        



        
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
        
        t,it,it_store,xy = daesolver(t,t_end,it,it_store,xy,u,p,
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
        
        if it < self.max_it:
            
            self.xy_ini = xy_ini
            self.N_iters = it

            self.ini2run()
            
            self.ini_convergence = True
            
        if it >= self.max_it:
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
            self.struct[0][item] = self.data[item]
            if item in self.params_list:
                self.params_values_list[self.params_list.index(item)] = self.data[item]
            elif item in self.inputs_ini_list:
                self.inputs_ini_values_list[self.inputs_ini_list.index(item)] = self.data[item]
            elif item in self.inputs_run_list:
                self.inputs_run_values_list[self.inputs_run_list.index(item)] = self.data[item]
            else: 
                print(f'parameter or input {item} not found')

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
    
        sp_jac_trap_eval(self.J_trap_d,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        self.sp_jac_trap.data = self.J_trap_d 
        
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
        self.iparams_run = np.zeros(10,dtype=np.float64)
    
        t,it,it_store,xy = spdaesolver(t,t_end,it,it_store,xy,u,p,
                                  self.jac_trap,
                                  self.J_trap_d,self.J_trap_i,self.J_trap_p,
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
                                  lmax_it=self.lmax_it,ltol=self.ltol,ldamp=self.ldamp,mode=self.mode)
    
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
            
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
            
        #jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        #jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
        return self.ini_convergence

        
    def spss_ini(self):
        J_d,J_i,J_p = csr2pydae(self.sp_jac_ini)
        
        xy_ini,it,iparams = spsstate(self.xy,self.u_ini,self.p,
                 J_d,J_i,J_p,
                 self.P_d,self.P_i,self.P_p,self.perm_r,self.perm_c,
                 self.N_x,self.N_y,
                 max_it=self.max_it,tol=self.itol,
                 lmax_it=self.lmax_it_ini,
                 ltol=self.ltol_ini,
                 ldamp=self.ldamp)

 
        self.xy_ini = xy_ini
        self.N_iters = it
        self.iparams = iparams
    
        return xy_ini

    #def import_cffi(self):
        

        

           
            



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
def spsstate(xy,u,p,
             J_d,J_i,J_p,
             P_d,P_i,P_p,perm_r,perm_c,
             N_x,N_y,
             max_it=50,tol=1e-8,
             lmax_it=20,ltol=1e-8,ldamp=1.0):
    
   
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

    sp_jac_ini_num_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
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
               
        Dxy = sprichardson(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
   
        xy += Dxy
        #if np.max(np.abs(fg))<tol: break
        if np.linalg.norm(fg,np.inf)<tol: break

    return xy,it,iparams


    
@numba.njit() 
def daesolver(t,t_end,it,it_store,xy,u,p,jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    h_ptr=ffi.from_buffer(np.ascontiguousarray(h))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

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
                
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
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
def spdaesolver(t,t_end,it,it_store,xy,u,p,jac_trap,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,
                iparams,
                max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0):

    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    Dxy_i_0 = np.zeros(N_x+N_y,dtype=np.float64) 
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    h_ptr=ffi.from_buffer(np.ascontiguousarray(h))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))
    
    sp_jac_trap_num_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    sp_jac_trap_up_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    sp_jac_trap_xy_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

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
            Dxy_i = sprichardson(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
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
                
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
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


@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_run_eval(sp_jac_run,x,y,u,p,Dt):   
    '''
    Computes the sparse full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    sp_jac_trap : (Nnz,) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with Nnz the number of non-zeros elements in the jacobian.
 
    Returns
    -------
    
    sp_jac_trap : (Nnz,) array_like
                  Updated matrix.    
    
    '''        
    sp_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_run))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_run_num_eval(sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_run_up_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_run_xy_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_run

@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_trap_eval(sp_jac_trap,x,y,u,p,Dt):   
    '''
    Computes the sparse full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    sp_jac_trap : (Nnz,) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with Nnz the number of non-zeros elements in the jacobian.
 
    Returns
    -------
    
    sp_jac_trap : (Nnz,) array_like
                  Updated matrix.    
    
    '''        
    sp_jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_trap))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_trap_num_eval(sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_trap_up_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_trap_xy_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_trap

@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_ini_eval(sp_jac_ini,x,y,u,p,Dt):   
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
    
    sp_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_ini))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_ini_num_eval(sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_ini_up_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_ini_xy_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_ini


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

    de_jac_ini_num_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
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

def sp_jac_ini_vectors():

    sp_jac_ini_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 161, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 162, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 163, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 164, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 165, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 166, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 167, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 168, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 169, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 170, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 171, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 172, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 173, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 174, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 175, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 176, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 177, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 178, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 179, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 180, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 181, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 182, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 183, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 184, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 185, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 186, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 187, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 188, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 189, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 190, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 191, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 192, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 193, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 194, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 195, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 196, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 197, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 198, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 199, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 200, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 201, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 202, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 203, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 204, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 205, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 206, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 207, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 208, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 209, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 210, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 211, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 212, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 213, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 214, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 215, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 216, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 217, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 218, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 219, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 220, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 221, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 222, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 223, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 224, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 1, 2, 7, 8, 161, 162, 3, 4, 7, 8, 163, 164, 5, 6, 7, 8, 165, 166, 1, 2, 7, 8, 161, 162, 3, 4, 7, 8, 163, 164, 5, 6, 7, 8, 165, 166, 161, 163, 165, 167, 162, 164, 166, 168, 9, 10, 15, 16, 169, 170, 11, 12, 15, 16, 171, 172, 13, 14, 15, 16, 173, 174, 9, 10, 15, 16, 169, 170, 11, 12, 15, 16, 171, 172, 13, 14, 15, 16, 173, 174, 169, 171, 173, 175, 170, 172, 174, 176, 17, 18, 23, 24, 177, 178, 19, 20, 23, 24, 179, 180, 21, 22, 23, 24, 181, 182, 17, 18, 23, 24, 177, 178, 19, 20, 23, 24, 179, 180, 21, 22, 23, 24, 181, 182, 177, 179, 181, 183, 178, 180, 182, 184, 25, 26, 31, 32, 185, 186, 27, 28, 31, 32, 187, 188, 29, 30, 31, 32, 189, 190, 25, 26, 31, 32, 185, 186, 27, 28, 31, 32, 187, 188, 29, 30, 31, 32, 189, 190, 185, 187, 189, 191, 186, 188, 190, 192, 33, 34, 39, 40, 193, 194, 35, 36, 39, 40, 195, 196, 37, 38, 39, 40, 197, 198, 33, 34, 39, 40, 193, 194, 35, 36, 39, 40, 195, 196, 37, 38, 39, 40, 197, 198, 193, 195, 197, 199, 194, 196, 198, 200, 41, 42, 47, 48, 201, 202, 43, 44, 47, 48, 203, 204, 45, 46, 47, 48, 205, 206, 41, 42, 47, 48, 201, 202, 43, 44, 47, 48, 203, 204, 45, 46, 47, 48, 205, 206, 201, 203, 205, 207, 202, 204, 206, 208, 49, 50, 55, 56, 209, 210, 51, 52, 55, 56, 211, 212, 53, 54, 55, 56, 213, 214, 49, 50, 55, 56, 209, 210, 51, 52, 55, 56, 211, 212, 53, 54, 55, 56, 213, 214, 209, 211, 213, 215, 210, 212, 214, 216, 57, 58, 63, 64, 217, 218, 59, 60, 63, 64, 219, 220, 61, 62, 63, 64, 221, 222, 57, 58, 63, 64, 217, 218, 59, 60, 63, 64, 219, 220, 61, 62, 63, 64, 221, 222, 217, 219, 221, 223, 218, 220, 222, 224]
    sp_jac_ini_ja = [0, 1, 18, 35, 52, 69, 86, 103, 120, 137, 154, 171, 188, 205, 222, 239, 256, 273, 290, 307, 324, 341, 358, 375, 392, 409, 426, 443, 460, 477, 494, 511, 528, 545, 562, 579, 596, 613, 630, 647, 664, 681, 698, 715, 732, 749, 766, 783, 800, 817, 834, 851, 868, 885, 902, 919, 936, 953, 970, 987, 1004, 1021, 1038, 1055, 1072, 1089, 1113, 1137, 1161, 1185, 1209, 1233, 1257, 1281, 1313, 1345, 1377, 1409, 1441, 1473, 1505, 1537, 1561, 1585, 1609, 1633, 1657, 1681, 1705, 1729, 1761, 1793, 1825, 1857, 1889, 1921, 1953, 1985, 2009, 2033, 2057, 2081, 2105, 2129, 2153, 2177, 2201, 2225, 2249, 2273, 2297, 2321, 2345, 2369, 2401, 2433, 2465, 2497, 2529, 2561, 2593, 2625, 2649, 2673, 2697, 2721, 2745, 2769, 2793, 2817, 2849, 2881, 2913, 2945, 2977, 3009, 3041, 3073, 3105, 3137, 3169, 3201, 3233, 3265, 3297, 3329, 3361, 3393, 3425, 3457, 3489, 3521, 3553, 3585, 3609, 3633, 3657, 3681, 3705, 3729, 3753, 3777, 3783, 3789, 3795, 3801, 3807, 3813, 3817, 3821, 3827, 3833, 3839, 3845, 3851, 3857, 3861, 3865, 3871, 3877, 3883, 3889, 3895, 3901, 3905, 3909, 3915, 3921, 3927, 3933, 3939, 3945, 3949, 3953, 3959, 3965, 3971, 3977, 3983, 3989, 3993, 3997, 4003, 4009, 4015, 4021, 4027, 4033, 4037, 4041, 4047, 4053, 4059, 4065, 4071, 4077, 4081, 4085, 4091, 4097, 4103, 4109, 4115, 4121, 4125, 4129]
    sp_jac_ini_nia = 225
    sp_jac_ini_nja = 225
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 161, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 162, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 163, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 164, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 165, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 166, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 167, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 168, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 169, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 170, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 171, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 172, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 173, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 174, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 175, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 176, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 177, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 178, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 179, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 180, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 181, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 182, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 183, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 184, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 185, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 186, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 187, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 188, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 189, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 190, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 191, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 192, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 193, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 194, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 195, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 196, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 197, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 198, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 199, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 200, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 201, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 202, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 203, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 204, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 205, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 206, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 207, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 208, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 209, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 210, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 211, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 212, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 213, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 214, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 215, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 216, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 217, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 218, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 219, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 220, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 221, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 222, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 223, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 224, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 1, 2, 7, 8, 161, 162, 3, 4, 7, 8, 163, 164, 5, 6, 7, 8, 165, 166, 1, 2, 7, 8, 161, 162, 3, 4, 7, 8, 163, 164, 5, 6, 7, 8, 165, 166, 161, 163, 165, 167, 162, 164, 166, 168, 9, 10, 15, 16, 169, 170, 11, 12, 15, 16, 171, 172, 13, 14, 15, 16, 173, 174, 9, 10, 15, 16, 169, 170, 11, 12, 15, 16, 171, 172, 13, 14, 15, 16, 173, 174, 169, 171, 173, 175, 170, 172, 174, 176, 17, 18, 23, 24, 177, 178, 19, 20, 23, 24, 179, 180, 21, 22, 23, 24, 181, 182, 17, 18, 23, 24, 177, 178, 19, 20, 23, 24, 179, 180, 21, 22, 23, 24, 181, 182, 177, 179, 181, 183, 178, 180, 182, 184, 25, 26, 31, 32, 185, 186, 27, 28, 31, 32, 187, 188, 29, 30, 31, 32, 189, 190, 25, 26, 31, 32, 185, 186, 27, 28, 31, 32, 187, 188, 29, 30, 31, 32, 189, 190, 185, 187, 189, 191, 186, 188, 190, 192, 33, 34, 39, 40, 193, 194, 35, 36, 39, 40, 195, 196, 37, 38, 39, 40, 197, 198, 33, 34, 39, 40, 193, 194, 35, 36, 39, 40, 195, 196, 37, 38, 39, 40, 197, 198, 193, 195, 197, 199, 194, 196, 198, 200, 41, 42, 47, 48, 201, 202, 43, 44, 47, 48, 203, 204, 45, 46, 47, 48, 205, 206, 41, 42, 47, 48, 201, 202, 43, 44, 47, 48, 203, 204, 45, 46, 47, 48, 205, 206, 201, 203, 205, 207, 202, 204, 206, 208, 49, 50, 55, 56, 209, 210, 51, 52, 55, 56, 211, 212, 53, 54, 55, 56, 213, 214, 49, 50, 55, 56, 209, 210, 51, 52, 55, 56, 211, 212, 53, 54, 55, 56, 213, 214, 209, 211, 213, 215, 210, 212, 214, 216, 57, 58, 63, 64, 217, 218, 59, 60, 63, 64, 219, 220, 61, 62, 63, 64, 221, 222, 57, 58, 63, 64, 217, 218, 59, 60, 63, 64, 219, 220, 61, 62, 63, 64, 221, 222, 217, 219, 221, 223, 218, 220, 222, 224]
    sp_jac_run_ja = [0, 1, 18, 35, 52, 69, 86, 103, 120, 137, 154, 171, 188, 205, 222, 239, 256, 273, 290, 307, 324, 341, 358, 375, 392, 409, 426, 443, 460, 477, 494, 511, 528, 545, 562, 579, 596, 613, 630, 647, 664, 681, 698, 715, 732, 749, 766, 783, 800, 817, 834, 851, 868, 885, 902, 919, 936, 953, 970, 987, 1004, 1021, 1038, 1055, 1072, 1089, 1113, 1137, 1161, 1185, 1209, 1233, 1257, 1281, 1313, 1345, 1377, 1409, 1441, 1473, 1505, 1537, 1561, 1585, 1609, 1633, 1657, 1681, 1705, 1729, 1761, 1793, 1825, 1857, 1889, 1921, 1953, 1985, 2009, 2033, 2057, 2081, 2105, 2129, 2153, 2177, 2201, 2225, 2249, 2273, 2297, 2321, 2345, 2369, 2401, 2433, 2465, 2497, 2529, 2561, 2593, 2625, 2649, 2673, 2697, 2721, 2745, 2769, 2793, 2817, 2849, 2881, 2913, 2945, 2977, 3009, 3041, 3073, 3105, 3137, 3169, 3201, 3233, 3265, 3297, 3329, 3361, 3393, 3425, 3457, 3489, 3521, 3553, 3585, 3609, 3633, 3657, 3681, 3705, 3729, 3753, 3777, 3783, 3789, 3795, 3801, 3807, 3813, 3817, 3821, 3827, 3833, 3839, 3845, 3851, 3857, 3861, 3865, 3871, 3877, 3883, 3889, 3895, 3901, 3905, 3909, 3915, 3921, 3927, 3933, 3939, 3945, 3949, 3953, 3959, 3965, 3971, 3977, 3983, 3989, 3993, 3997, 4003, 4009, 4015, 4021, 4027, 4033, 4037, 4041, 4047, 4053, 4059, 4065, 4071, 4077, 4081, 4085, 4091, 4097, 4103, 4109, 4115, 4121, 4125, 4129]
    sp_jac_run_nia = 225
    sp_jac_run_nja = 225
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 161, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 162, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 163, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 164, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 165, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 166, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 167, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 168, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 169, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 170, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 171, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 172, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 173, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 174, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 175, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 176, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 177, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 178, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 179, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 180, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 181, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 182, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 183, 17, 18, 19, 20, 21, 22, 23, 24, 137, 138, 139, 140, 141, 142, 143, 144, 184, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 185, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 186, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 187, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 188, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 189, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 190, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 191, 25, 26, 27, 28, 29, 30, 31, 32, 129, 130, 131, 132, 133, 134, 135, 136, 192, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 193, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 194, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 195, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 196, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 197, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 198, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 199, 33, 34, 35, 36, 37, 38, 39, 40, 153, 154, 155, 156, 157, 158, 159, 160, 200, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 201, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 202, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 203, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 204, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 205, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 206, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 207, 41, 42, 43, 44, 45, 46, 47, 48, 145, 146, 147, 148, 149, 150, 151, 152, 208, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 209, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 210, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 211, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 212, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 213, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 214, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 215, 49, 50, 51, 52, 53, 54, 55, 56, 113, 114, 115, 116, 117, 118, 119, 120, 216, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 217, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 218, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 219, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 220, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 221, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 222, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 223, 57, 58, 59, 60, 61, 62, 63, 64, 121, 122, 123, 124, 125, 126, 127, 128, 224, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 129, 130, 131, 132, 133, 134, 135, 136, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 145, 146, 147, 148, 149, 150, 151, 152, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 25, 26, 27, 28, 29, 30, 31, 32, 73, 74, 75, 76, 77, 78, 79, 80, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 41, 42, 43, 44, 45, 46, 47, 48, 89, 90, 91, 92, 93, 94, 95, 96, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 33, 34, 35, 36, 37, 38, 39, 40, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 1, 2, 7, 8, 161, 162, 3, 4, 7, 8, 163, 164, 5, 6, 7, 8, 165, 166, 1, 2, 7, 8, 161, 162, 3, 4, 7, 8, 163, 164, 5, 6, 7, 8, 165, 166, 161, 163, 165, 167, 162, 164, 166, 168, 9, 10, 15, 16, 169, 170, 11, 12, 15, 16, 171, 172, 13, 14, 15, 16, 173, 174, 9, 10, 15, 16, 169, 170, 11, 12, 15, 16, 171, 172, 13, 14, 15, 16, 173, 174, 169, 171, 173, 175, 170, 172, 174, 176, 17, 18, 23, 24, 177, 178, 19, 20, 23, 24, 179, 180, 21, 22, 23, 24, 181, 182, 17, 18, 23, 24, 177, 178, 19, 20, 23, 24, 179, 180, 21, 22, 23, 24, 181, 182, 177, 179, 181, 183, 178, 180, 182, 184, 25, 26, 31, 32, 185, 186, 27, 28, 31, 32, 187, 188, 29, 30, 31, 32, 189, 190, 25, 26, 31, 32, 185, 186, 27, 28, 31, 32, 187, 188, 29, 30, 31, 32, 189, 190, 185, 187, 189, 191, 186, 188, 190, 192, 33, 34, 39, 40, 193, 194, 35, 36, 39, 40, 195, 196, 37, 38, 39, 40, 197, 198, 33, 34, 39, 40, 193, 194, 35, 36, 39, 40, 195, 196, 37, 38, 39, 40, 197, 198, 193, 195, 197, 199, 194, 196, 198, 200, 41, 42, 47, 48, 201, 202, 43, 44, 47, 48, 203, 204, 45, 46, 47, 48, 205, 206, 41, 42, 47, 48, 201, 202, 43, 44, 47, 48, 203, 204, 45, 46, 47, 48, 205, 206, 201, 203, 205, 207, 202, 204, 206, 208, 49, 50, 55, 56, 209, 210, 51, 52, 55, 56, 211, 212, 53, 54, 55, 56, 213, 214, 49, 50, 55, 56, 209, 210, 51, 52, 55, 56, 211, 212, 53, 54, 55, 56, 213, 214, 209, 211, 213, 215, 210, 212, 214, 216, 57, 58, 63, 64, 217, 218, 59, 60, 63, 64, 219, 220, 61, 62, 63, 64, 221, 222, 57, 58, 63, 64, 217, 218, 59, 60, 63, 64, 219, 220, 61, 62, 63, 64, 221, 222, 217, 219, 221, 223, 218, 220, 222, 224]
    sp_jac_trap_ja = [0, 1, 18, 35, 52, 69, 86, 103, 120, 137, 154, 171, 188, 205, 222, 239, 256, 273, 290, 307, 324, 341, 358, 375, 392, 409, 426, 443, 460, 477, 494, 511, 528, 545, 562, 579, 596, 613, 630, 647, 664, 681, 698, 715, 732, 749, 766, 783, 800, 817, 834, 851, 868, 885, 902, 919, 936, 953, 970, 987, 1004, 1021, 1038, 1055, 1072, 1089, 1113, 1137, 1161, 1185, 1209, 1233, 1257, 1281, 1313, 1345, 1377, 1409, 1441, 1473, 1505, 1537, 1561, 1585, 1609, 1633, 1657, 1681, 1705, 1729, 1761, 1793, 1825, 1857, 1889, 1921, 1953, 1985, 2009, 2033, 2057, 2081, 2105, 2129, 2153, 2177, 2201, 2225, 2249, 2273, 2297, 2321, 2345, 2369, 2401, 2433, 2465, 2497, 2529, 2561, 2593, 2625, 2649, 2673, 2697, 2721, 2745, 2769, 2793, 2817, 2849, 2881, 2913, 2945, 2977, 3009, 3041, 3073, 3105, 3137, 3169, 3201, 3233, 3265, 3297, 3329, 3361, 3393, 3425, 3457, 3489, 3521, 3553, 3585, 3609, 3633, 3657, 3681, 3705, 3729, 3753, 3777, 3783, 3789, 3795, 3801, 3807, 3813, 3817, 3821, 3827, 3833, 3839, 3845, 3851, 3857, 3861, 3865, 3871, 3877, 3883, 3889, 3895, 3901, 3905, 3909, 3915, 3921, 3927, 3933, 3939, 3945, 3949, 3953, 3959, 3965, 3971, 3977, 3983, 3989, 3993, 3997, 4003, 4009, 4015, 4021, 4027, 4033, 4037, 4041, 4047, 4053, 4059, 4065, 4071, 4077, 4081, 4085, 4091, 4097, 4103, 4109, 4115, 4121, 4125, 4129]
    sp_jac_trap_nia = 225
    sp_jac_trap_nja = 225
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
