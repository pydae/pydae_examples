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
import os

dae_file_mode = 'local'

ffi = cffi.FFI()

if dae_file_mode == 'local':
    import lab_lv_cffi as jacs
if dae_file_mode == 'enviroment':
    import envus.no_enviroment.lab_lv_cffi as jacs
if dae_file_mode == 'colab':
    import lab_lv_cffi as jacs
if dae_file_mode == 'testing':
    from pydae.temp import lab_lv_cffi as jacs
    
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

sp_Fu_run_up_eval = jacs.lib.sp_Fu_run_up_eval
sp_Gu_run_up_eval = jacs.lib.sp_Gu_run_up_eval
sp_Hx_run_up_eval = jacs.lib.sp_Hx_run_up_eval
sp_Hy_run_up_eval = jacs.lib.sp_Hy_run_up_eval
sp_Hu_run_up_eval = jacs.lib.sp_Hu_run_up_eval
sp_Fu_run_xy_eval = jacs.lib.sp_Fu_run_xy_eval
sp_Gu_run_xy_eval = jacs.lib.sp_Gu_run_xy_eval
sp_Hx_run_xy_eval = jacs.lib.sp_Hx_run_xy_eval
sp_Hy_run_xy_eval = jacs.lib.sp_Hy_run_xy_eval
sp_Hu_run_xy_eval = jacs.lib.sp_Hu_run_xy_eval



import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class model: 

    def __init__(self,matrices_folder='./build'): 
        
        self.matrices_folder = matrices_folder
        
        self.dae_file_mode = 'local'
        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 1
        self.N_y = 142 
        self.N_z = 116 
        self.N_store = 100000 
        self.params_list = ['g_N02_0_N03_0_0', 'b_N02_0_N03_0_0', 'g_N02_1_N03_1_1', 'b_N02_1_N03_1_1', 'g_N02_2_N03_2_2', 'b_N02_2_N03_2_2', 'g_N02_3_N03_3_3', 'b_N02_3_N03_3_3', 'g_N03_0_N04_0_0', 'b_N03_0_N04_0_0', 'g_N03_1_N04_1_1', 'b_N03_1_N04_1_1', 'g_N03_2_N04_2_2', 'b_N03_2_N04_2_2', 'g_N03_3_N04_3_3', 'b_N03_3_N04_3_3', 'g_N04_0_N05_0_0', 'b_N04_0_N05_0_0', 'g_N04_1_N05_1_1', 'b_N04_1_N05_1_1', 'g_N04_2_N05_2_2', 'b_N04_2_N05_2_2', 'g_N04_3_N05_3_3', 'b_N04_3_N05_3_3', 'g_N05_0_N06_0_0', 'b_N05_0_N06_0_0', 'g_N05_1_N06_1_1', 'b_N05_1_N06_1_1', 'g_N05_2_N06_2_2', 'b_N05_2_N06_2_2', 'g_N05_3_N06_3_3', 'b_N05_3_N06_3_3', 'g_N06_0_N07_0_0', 'b_N06_0_N07_0_0', 'g_N06_1_N07_1_1', 'b_N06_1_N07_1_1', 'g_N06_2_N07_2_2', 'b_N06_2_N07_2_2', 'g_N06_3_N07_3_3', 'b_N06_3_N07_3_3', 'g_N07_0_N08_0_0', 'b_N07_0_N08_0_0', 'g_N07_1_N08_1_1', 'b_N07_1_N08_1_1', 'g_N07_2_N08_2_2', 'b_N07_2_N08_2_2', 'g_N07_3_N08_3_3', 'b_N07_3_N08_3_3', 'g_N06_0_N09_0_0', 'b_N06_0_N09_0_0', 'g_N06_1_N09_1_1', 'b_N06_1_N09_1_1', 'g_N06_2_N09_2_2', 'b_N06_2_N09_2_2', 'g_N06_3_N09_3_3', 'b_N06_3_N09_3_3', 'g_N05_0_N10_0_0', 'b_N05_0_N10_0_0', 'g_N05_1_N10_1_1', 'b_N05_1_N10_1_1', 'g_N05_2_N10_2_2', 'b_N05_2_N10_2_2', 'g_N05_3_N10_3_3', 'b_N05_3_N10_3_3', 'g_N04_0_N11_0_0', 'b_N04_0_N11_0_0', 'g_N04_1_N11_1_1', 'b_N04_1_N11_1_1', 'g_N04_2_N11_2_2', 'b_N04_2_N11_2_2', 'g_N04_3_N11_3_3', 'b_N04_3_N11_3_3', 'g_N03_0_N12_0_0', 'b_N03_0_N12_0_0', 'g_N03_1_N12_1_1', 'b_N03_1_N12_1_1', 'g_N03_2_N12_2_2', 'b_N03_2_N12_2_2', 'g_N03_3_N12_3_3', 'b_N03_3_N12_3_3', 'g_shunt_N02_3', 'b_shunt_N02_3', 'K_abc_N07', 'K_abc_N08', 'K_abc_N09', 'K_abc_N10', 'K_abc_N11', 'X_N01_s', 'R_N01_s', 'K_p_agc', 'K_i_agc', 'K_xif'] 
        self.params_values_list  = [7.076134443832843, -4.238876690872944, 7.076134443832843, -4.238876690872944, 7.076134443832843, -4.238876690872944, 7.076134443832843, -4.238876690872944, 15.198364865573085, -9.826529007913633, 15.198364865573085, -9.826529007913633, 15.198364865573085, -9.826529007913633, 15.198364865573085, -9.826529007913633, 7.914286518276725, -5.029968764949208, 7.914286518276725, -5.029968764949208, 7.914286518276725, -5.029968764949208, 7.914286518276725, -5.029968764949208, 4.35396136636231, -3.4370140256019086, 4.35396136636231, -3.4370140256019086, 4.35396136636231, -3.4370140256019086, 4.35396136636231, -3.4370140256019086, 13.662654369235579, -11.22401125527591, 13.662654369235579, -11.22401125527591, 13.662654369235579, -11.22401125527591, 13.662654369235579, -11.22401125527591, 5.449213452259174, -0.8453578904732424, 5.449213452259174, -0.8453578904732424, 5.449213452259174, -0.8453578904732424, 5.449213452259174, -0.8453578904732424, 5.539499789105337, -0.746723321825254, 5.539499789105337, -0.746723321825254, 5.539499789105337, -0.746723321825254, 5.539499789105337, -0.746723321825254, 5.728554494749158, -0.7131533809635368, 5.728554494749158, -0.7131533809635368, 5.728554494749158, -0.7131533809635368, 5.728554494749158, -0.7131533809635368, 1.8901241470398842, -0.28997261532262103, 1.8901241470398842, -0.28997261532262103, 1.8901241470398842, -0.28997261532262103, 1.8901241470398842, -0.28997261532262103, 5.587756576194438, -0.773248774496155, 5.587756576194438, -0.773248774496155, 5.587756576194438, -0.773248774496155, 5.587756576194438, -0.773248774496155, 0.3333333333333333, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.01, 0.01, 0.01, 0.01] 
        self.inputs_ini_list = ['p_load_N07_a', 'q_load_N07_a', 'g_load_N07_a', 'b_load_N07_a', 'p_load_N07_b', 'q_load_N07_b', 'g_load_N07_b', 'b_load_N07_b', 'p_load_N07_c', 'q_load_N07_c', 'g_load_N07_c', 'b_load_N07_c', 'p_load_N08_a', 'q_load_N08_a', 'g_load_N08_a', 'b_load_N08_a', 'p_load_N08_b', 'q_load_N08_b', 'g_load_N08_b', 'b_load_N08_b', 'p_load_N08_c', 'q_load_N08_c', 'g_load_N08_c', 'b_load_N08_c', 'p_load_N09_a', 'q_load_N09_a', 'g_load_N09_a', 'b_load_N09_a', 'p_load_N09_b', 'q_load_N09_b', 'g_load_N09_b', 'b_load_N09_b', 'p_load_N09_c', 'q_load_N09_c', 'g_load_N09_c', 'b_load_N09_c', 'p_load_N10_a', 'q_load_N10_a', 'g_load_N10_a', 'b_load_N10_a', 'p_load_N10_b', 'q_load_N10_b', 'g_load_N10_b', 'b_load_N10_b', 'p_load_N10_c', 'q_load_N10_c', 'g_load_N10_c', 'b_load_N10_c', 'p_load_N11_a', 'q_load_N11_a', 'g_load_N11_a', 'b_load_N11_a', 'p_load_N11_b', 'q_load_N11_b', 'g_load_N11_b', 'b_load_N11_b', 'p_load_N11_c', 'q_load_N11_c', 'g_load_N11_c', 'b_load_N11_c', 'e_ao_m_N01', 'e_bo_m_N01', 'e_co_m_N01', 'phi_N01', 'u_freq'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11547.005383792515, 11547.005383792515, 11547.005383792515, 0.0, 0.0] 
        self.inputs_run_list = ['p_load_N07_a', 'q_load_N07_a', 'g_load_N07_a', 'b_load_N07_a', 'p_load_N07_b', 'q_load_N07_b', 'g_load_N07_b', 'b_load_N07_b', 'p_load_N07_c', 'q_load_N07_c', 'g_load_N07_c', 'b_load_N07_c', 'p_load_N08_a', 'q_load_N08_a', 'g_load_N08_a', 'b_load_N08_a', 'p_load_N08_b', 'q_load_N08_b', 'g_load_N08_b', 'b_load_N08_b', 'p_load_N08_c', 'q_load_N08_c', 'g_load_N08_c', 'b_load_N08_c', 'p_load_N09_a', 'q_load_N09_a', 'g_load_N09_a', 'b_load_N09_a', 'p_load_N09_b', 'q_load_N09_b', 'g_load_N09_b', 'b_load_N09_b', 'p_load_N09_c', 'q_load_N09_c', 'g_load_N09_c', 'b_load_N09_c', 'p_load_N10_a', 'q_load_N10_a', 'g_load_N10_a', 'b_load_N10_a', 'p_load_N10_b', 'q_load_N10_b', 'g_load_N10_b', 'b_load_N10_b', 'p_load_N10_c', 'q_load_N10_c', 'g_load_N10_c', 'b_load_N10_c', 'p_load_N11_a', 'q_load_N11_a', 'g_load_N11_a', 'b_load_N11_a', 'p_load_N11_b', 'q_load_N11_b', 'g_load_N11_b', 'b_load_N11_b', 'p_load_N11_c', 'q_load_N11_c', 'g_load_N11_c', 'b_load_N11_c', 'e_ao_m_N01', 'e_bo_m_N01', 'e_co_m_N01', 'phi_N01', 'u_freq'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11547.005383792515, 11547.005383792515, 11547.005383792515, 0.0, 0.0] 
        self.outputs_list = ['i_l_N02_0_N03_0_r', 'i_l_N02_0_N03_0_i', 'i_l_N02_1_N03_1_r', 'i_l_N02_1_N03_1_i', 'i_l_N02_2_N03_2_r', 'i_l_N02_2_N03_2_i', 'i_l_N02_3_N03_3_r', 'i_l_N02_3_N03_3_i', 'i_l_N03_0_N04_0_r', 'i_l_N03_0_N04_0_i', 'i_l_N03_1_N04_1_r', 'i_l_N03_1_N04_1_i', 'i_l_N03_2_N04_2_r', 'i_l_N03_2_N04_2_i', 'i_l_N03_3_N04_3_r', 'i_l_N03_3_N04_3_i', 'i_l_N04_0_N05_0_r', 'i_l_N04_0_N05_0_i', 'i_l_N04_1_N05_1_r', 'i_l_N04_1_N05_1_i', 'i_l_N04_2_N05_2_r', 'i_l_N04_2_N05_2_i', 'i_l_N04_3_N05_3_r', 'i_l_N04_3_N05_3_i', 'i_l_N05_0_N06_0_r', 'i_l_N05_0_N06_0_i', 'i_l_N05_1_N06_1_r', 'i_l_N05_1_N06_1_i', 'i_l_N05_2_N06_2_r', 'i_l_N05_2_N06_2_i', 'i_l_N05_3_N06_3_r', 'i_l_N05_3_N06_3_i', 'i_l_N06_0_N07_0_r', 'i_l_N06_0_N07_0_i', 'i_l_N06_1_N07_1_r', 'i_l_N06_1_N07_1_i', 'i_l_N06_2_N07_2_r', 'i_l_N06_2_N07_2_i', 'i_l_N06_3_N07_3_r', 'i_l_N06_3_N07_3_i', 'i_l_N07_0_N08_0_r', 'i_l_N07_0_N08_0_i', 'i_l_N07_1_N08_1_r', 'i_l_N07_1_N08_1_i', 'i_l_N07_2_N08_2_r', 'i_l_N07_2_N08_2_i', 'i_l_N07_3_N08_3_r', 'i_l_N07_3_N08_3_i', 'i_l_N06_0_N09_0_r', 'i_l_N06_0_N09_0_i', 'i_l_N06_1_N09_1_r', 'i_l_N06_1_N09_1_i', 'i_l_N06_2_N09_2_r', 'i_l_N06_2_N09_2_i', 'i_l_N06_3_N09_3_r', 'i_l_N06_3_N09_3_i', 'i_l_N05_0_N10_0_r', 'i_l_N05_0_N10_0_i', 'i_l_N05_1_N10_1_r', 'i_l_N05_1_N10_1_i', 'i_l_N05_2_N10_2_r', 'i_l_N05_2_N10_2_i', 'i_l_N05_3_N10_3_r', 'i_l_N05_3_N10_3_i', 'i_l_N04_0_N11_0_r', 'i_l_N04_0_N11_0_i', 'i_l_N04_1_N11_1_r', 'i_l_N04_1_N11_1_i', 'i_l_N04_2_N11_2_r', 'i_l_N04_2_N11_2_i', 'i_l_N04_3_N11_3_r', 'i_l_N04_3_N11_3_i', 'i_l_N03_0_N12_0_r', 'i_l_N03_0_N12_0_i', 'i_l_N03_1_N12_1_r', 'i_l_N03_1_N12_1_i', 'i_l_N03_2_N12_2_r', 'i_l_N03_2_N12_2_i', 'i_l_N03_3_N12_3_r', 'i_l_N03_3_N12_3_i', 'i_t_N01_N02_1_0_r', 'i_t_N01_N02_1_0_i', 'i_t_N01_N02_1_1_r', 'i_t_N01_N02_1_1_i', 'i_t_N01_N02_1_2_r', 'i_t_N01_N02_1_2_i', 'i_t_N01_N02_2_0_r', 'i_t_N01_N02_2_0_i', 'i_t_N01_N02_2_1_r', 'i_t_N01_N02_2_1_i', 'i_t_N01_N02_2_2_r', 'i_t_N01_N02_2_2_i', 'i_t_N01_N02_2_3_r', 'i_t_N01_N02_2_3_i', 'v_anm_N07', 'v_bnm_N07', 'v_cnm_N07', 'v_anm_N08', 'v_bnm_N08', 'v_cnm_N08', 'v_anm_N09', 'v_bnm_N09', 'v_cnm_N09', 'v_anm_N10', 'v_bnm_N10', 'v_cnm_N10', 'v_anm_N11', 'v_bnm_N11', 'v_cnm_N11', 'i_vsc_N01_a_m', 'i_vsc_N01_b_m', 'i_vsc_N01_c_m', 'p_N01', 'q_N01', 'xi_freq', 'u_freq'] 
        self.x_list = ['xi_freq'] 
        self.y_run_list = ['V_N01_0_r', 'V_N01_0_i', 'V_N01_1_r', 'V_N01_1_i', 'V_N01_2_r', 'V_N01_2_i', 'V_N02_0_r', 'V_N02_0_i', 'V_N02_1_r', 'V_N02_1_i', 'V_N02_2_r', 'V_N02_2_i', 'V_N02_3_r', 'V_N02_3_i', 'V_N03_0_r', 'V_N03_0_i', 'V_N03_1_r', 'V_N03_1_i', 'V_N03_2_r', 'V_N03_2_i', 'V_N03_3_r', 'V_N03_3_i', 'V_N04_0_r', 'V_N04_0_i', 'V_N04_1_r', 'V_N04_1_i', 'V_N04_2_r', 'V_N04_2_i', 'V_N04_3_r', 'V_N04_3_i', 'V_N05_0_r', 'V_N05_0_i', 'V_N05_1_r', 'V_N05_1_i', 'V_N05_2_r', 'V_N05_2_i', 'V_N05_3_r', 'V_N05_3_i', 'V_N06_0_r', 'V_N06_0_i', 'V_N06_1_r', 'V_N06_1_i', 'V_N06_2_r', 'V_N06_2_i', 'V_N06_3_r', 'V_N06_3_i', 'V_N07_0_r', 'V_N07_0_i', 'V_N07_1_r', 'V_N07_1_i', 'V_N07_2_r', 'V_N07_2_i', 'V_N07_3_r', 'V_N07_3_i', 'V_N08_0_r', 'V_N08_0_i', 'V_N08_1_r', 'V_N08_1_i', 'V_N08_2_r', 'V_N08_2_i', 'V_N08_3_r', 'V_N08_3_i', 'V_N09_0_r', 'V_N09_0_i', 'V_N09_1_r', 'V_N09_1_i', 'V_N09_2_r', 'V_N09_2_i', 'V_N09_3_r', 'V_N09_3_i', 'V_N10_0_r', 'V_N10_0_i', 'V_N10_1_r', 'V_N10_1_i', 'V_N10_2_r', 'V_N10_2_i', 'V_N10_3_r', 'V_N10_3_i', 'V_N11_0_r', 'V_N11_0_i', 'V_N11_1_r', 'V_N11_1_i', 'V_N11_2_r', 'V_N11_2_i', 'V_N11_3_r', 'V_N11_3_i', 'V_N12_0_r', 'V_N12_0_i', 'V_N12_1_r', 'V_N12_1_i', 'V_N12_2_r', 'V_N12_2_i', 'V_N12_3_r', 'V_N12_3_i', 'i_load_N07_a_r', 'i_load_N07_a_i', 'i_load_N07_b_r', 'i_load_N07_b_i', 'i_load_N07_c_r', 'i_load_N07_c_i', 'i_load_N07_n_r', 'i_load_N07_n_i', 'i_load_N08_a_r', 'i_load_N08_a_i', 'i_load_N08_b_r', 'i_load_N08_b_i', 'i_load_N08_c_r', 'i_load_N08_c_i', 'i_load_N08_n_r', 'i_load_N08_n_i', 'i_load_N09_a_r', 'i_load_N09_a_i', 'i_load_N09_b_r', 'i_load_N09_b_i', 'i_load_N09_c_r', 'i_load_N09_c_i', 'i_load_N09_n_r', 'i_load_N09_n_i', 'i_load_N10_a_r', 'i_load_N10_a_i', 'i_load_N10_b_r', 'i_load_N10_b_i', 'i_load_N10_c_r', 'i_load_N10_c_i', 'i_load_N10_n_r', 'i_load_N10_n_i', 'i_load_N11_a_r', 'i_load_N11_a_i', 'i_load_N11_b_r', 'i_load_N11_b_i', 'i_load_N11_c_r', 'i_load_N11_c_i', 'i_load_N11_n_r', 'i_load_N11_n_i', 'i_vsc_N01_a_r', 'i_vsc_N01_b_r', 'i_vsc_N01_c_r', 'i_vsc_N01_a_i', 'i_vsc_N01_b_i', 'i_vsc_N01_c_i', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_N01_0_r', 'V_N01_0_i', 'V_N01_1_r', 'V_N01_1_i', 'V_N01_2_r', 'V_N01_2_i', 'V_N02_0_r', 'V_N02_0_i', 'V_N02_1_r', 'V_N02_1_i', 'V_N02_2_r', 'V_N02_2_i', 'V_N02_3_r', 'V_N02_3_i', 'V_N03_0_r', 'V_N03_0_i', 'V_N03_1_r', 'V_N03_1_i', 'V_N03_2_r', 'V_N03_2_i', 'V_N03_3_r', 'V_N03_3_i', 'V_N04_0_r', 'V_N04_0_i', 'V_N04_1_r', 'V_N04_1_i', 'V_N04_2_r', 'V_N04_2_i', 'V_N04_3_r', 'V_N04_3_i', 'V_N05_0_r', 'V_N05_0_i', 'V_N05_1_r', 'V_N05_1_i', 'V_N05_2_r', 'V_N05_2_i', 'V_N05_3_r', 'V_N05_3_i', 'V_N06_0_r', 'V_N06_0_i', 'V_N06_1_r', 'V_N06_1_i', 'V_N06_2_r', 'V_N06_2_i', 'V_N06_3_r', 'V_N06_3_i', 'V_N07_0_r', 'V_N07_0_i', 'V_N07_1_r', 'V_N07_1_i', 'V_N07_2_r', 'V_N07_2_i', 'V_N07_3_r', 'V_N07_3_i', 'V_N08_0_r', 'V_N08_0_i', 'V_N08_1_r', 'V_N08_1_i', 'V_N08_2_r', 'V_N08_2_i', 'V_N08_3_r', 'V_N08_3_i', 'V_N09_0_r', 'V_N09_0_i', 'V_N09_1_r', 'V_N09_1_i', 'V_N09_2_r', 'V_N09_2_i', 'V_N09_3_r', 'V_N09_3_i', 'V_N10_0_r', 'V_N10_0_i', 'V_N10_1_r', 'V_N10_1_i', 'V_N10_2_r', 'V_N10_2_i', 'V_N10_3_r', 'V_N10_3_i', 'V_N11_0_r', 'V_N11_0_i', 'V_N11_1_r', 'V_N11_1_i', 'V_N11_2_r', 'V_N11_2_i', 'V_N11_3_r', 'V_N11_3_i', 'V_N12_0_r', 'V_N12_0_i', 'V_N12_1_r', 'V_N12_1_i', 'V_N12_2_r', 'V_N12_2_i', 'V_N12_3_r', 'V_N12_3_i', 'i_load_N07_a_r', 'i_load_N07_a_i', 'i_load_N07_b_r', 'i_load_N07_b_i', 'i_load_N07_c_r', 'i_load_N07_c_i', 'i_load_N07_n_r', 'i_load_N07_n_i', 'i_load_N08_a_r', 'i_load_N08_a_i', 'i_load_N08_b_r', 'i_load_N08_b_i', 'i_load_N08_c_r', 'i_load_N08_c_i', 'i_load_N08_n_r', 'i_load_N08_n_i', 'i_load_N09_a_r', 'i_load_N09_a_i', 'i_load_N09_b_r', 'i_load_N09_b_i', 'i_load_N09_c_r', 'i_load_N09_c_i', 'i_load_N09_n_r', 'i_load_N09_n_i', 'i_load_N10_a_r', 'i_load_N10_a_i', 'i_load_N10_b_r', 'i_load_N10_b_i', 'i_load_N10_c_r', 'i_load_N10_c_i', 'i_load_N10_n_r', 'i_load_N10_n_i', 'i_load_N11_a_r', 'i_load_N11_a_i', 'i_load_N11_b_r', 'i_load_N11_b_i', 'i_load_N11_c_r', 'i_load_N11_c_i', 'i_load_N11_n_r', 'i_load_N11_n_i', 'i_vsc_N01_a_r', 'i_vsc_N01_b_r', 'i_vsc_N01_c_r', 'i_vsc_N01_a_i', 'i_vsc_N01_b_i', 'i_vsc_N01_c_i', 'omega_coi', 'p_agc'] 
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
        self.u_ini = np.array(self.inputs_ini_values_list,dtype=np.float64)
        self.p = np.array(self.params_values_list,dtype=np.float64)
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
        self.sp_jac_ini_ia, self.sp_jac_ini_ja, self.sp_jac_ini_nia, self.sp_jac_ini_nja = sp_jac_ini_vectors()
        data = np.array(self.sp_jac_ini_ia,dtype=np.float64)
        #self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))
           
        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, f'./lab_lv_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz(f'{self.matrices_folder}/lab_lv_sp_jac_ini_num.npz')
            
            
        self.jac_ini = self.sp_jac_ini.toarray()

        #self.J_ini_d = np.array(self.sp_jac_ini_ia)*0.0
        #self.J_ini_i = np.array(self.sp_jac_ini_ia)
        #self.J_ini_p = np.array(self.sp_jac_ini_ja)
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_ini_eval(self.sp_jac_ini.data,x,y,self.u_ini,self.p,self.Dt) 
        self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 100,1e-10,'basic'       


        ## jac_run
        self.jac_run = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_run_ia, self.sp_jac_run_ja, self.sp_jac_run_nia, self.sp_jac_run_nja = sp_jac_run_vectors()
        data = np.array(self.sp_jac_run_ia,dtype=np.float64)

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './lab_lv_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz(f'{self.matrices_folder}/lab_lv_sp_jac_run_num.npz')
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
       
    

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './lab_lv_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz(f'{self.matrices_folder}/lab_lv_sp_jac_trap_num.npz')
            

        self.jac_trap = self.sp_jac_trap.toarray()
        
        #self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        #self.J_trap_i = np.array(self.sp_jac_trap_ia)
        #self.J_trap_p = np.array(self.sp_jac_trap_ja)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
        sp_jac_trap_eval(self.sp_jac_trap.data,x,y,self.u_run,self.p,self.Dt)
        self.fill_factor_trap,self.drop_tol_trap,self.drop_rule_trap = 100,1e-10,'basic' 
   

        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp= 50,1e-8,1.0
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.0

        self.sp_Fu_run = sspa.load_npz(f'{self.matrices_folder}/lab_lv_Fu_run_num.npz')
        self.sp_Gu_run = sspa.load_npz(f'{self.matrices_folder}/lab_lv_Gu_run_num.npz')
        self.sp_Hx_run = sspa.load_npz(f'{self.matrices_folder}/lab_lv_Hx_run_num.npz')
        self.sp_Hy_run = sspa.load_npz(f'{self.matrices_folder}/lab_lv_Hy_run_num.npz')
        self.sp_Hu_run = sspa.load_npz(f'{self.matrices_folder}/lab_lv_Hu_run_num.npz')        
        
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

    sp_jac_run_num_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
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

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
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

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
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

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
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

    sp_jac_ini_ia = [0, 141, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 135, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 138, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 136, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 139, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 137, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 140, 1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 13, 14, 17, 18, 1, 2, 3, 4, 9, 10, 13, 14, 17, 18, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 7, 8, 15, 16, 23, 24, 87, 88, 7, 8, 15, 16, 23, 24, 87, 88, 9, 10, 17, 18, 25, 26, 89, 90, 9, 10, 17, 18, 25, 26, 89, 90, 11, 12, 19, 20, 27, 28, 91, 92, 11, 12, 19, 20, 27, 28, 91, 92, 13, 14, 21, 22, 29, 30, 93, 94, 13, 14, 21, 22, 29, 30, 93, 94, 15, 16, 23, 24, 31, 32, 79, 80, 15, 16, 23, 24, 31, 32, 79, 80, 17, 18, 25, 26, 33, 34, 81, 82, 17, 18, 25, 26, 33, 34, 81, 82, 19, 20, 27, 28, 35, 36, 83, 84, 19, 20, 27, 28, 35, 36, 83, 84, 21, 22, 29, 30, 37, 38, 85, 86, 21, 22, 29, 30, 37, 38, 85, 86, 23, 24, 31, 32, 39, 40, 71, 72, 23, 24, 31, 32, 39, 40, 71, 72, 25, 26, 33, 34, 41, 42, 73, 74, 25, 26, 33, 34, 41, 42, 73, 74, 27, 28, 35, 36, 43, 44, 75, 76, 27, 28, 35, 36, 43, 44, 75, 76, 29, 30, 37, 38, 45, 46, 77, 78, 29, 30, 37, 38, 45, 46, 77, 78, 31, 32, 39, 40, 47, 48, 63, 64, 31, 32, 39, 40, 47, 48, 63, 64, 33, 34, 41, 42, 49, 50, 65, 66, 33, 34, 41, 42, 49, 50, 65, 66, 35, 36, 43, 44, 51, 52, 67, 68, 35, 36, 43, 44, 51, 52, 67, 68, 37, 38, 45, 46, 53, 54, 69, 70, 37, 38, 45, 46, 53, 54, 69, 70, 39, 40, 47, 48, 55, 56, 95, 39, 40, 47, 48, 55, 56, 96, 41, 42, 49, 50, 57, 58, 97, 41, 42, 49, 50, 57, 58, 98, 43, 44, 51, 52, 59, 60, 99, 43, 44, 51, 52, 59, 60, 100, 45, 46, 53, 54, 61, 62, 101, 45, 46, 53, 54, 61, 62, 102, 47, 48, 55, 56, 103, 47, 48, 55, 56, 104, 49, 50, 57, 58, 105, 49, 50, 57, 58, 106, 51, 52, 59, 60, 107, 51, 52, 59, 60, 108, 53, 54, 61, 62, 109, 53, 54, 61, 62, 110, 39, 40, 63, 64, 111, 39, 40, 63, 64, 112, 41, 42, 65, 66, 113, 41, 42, 65, 66, 114, 43, 44, 67, 68, 115, 43, 44, 67, 68, 116, 45, 46, 69, 70, 117, 45, 46, 69, 70, 118, 31, 32, 71, 72, 119, 31, 32, 71, 72, 120, 33, 34, 73, 74, 121, 33, 34, 73, 74, 122, 35, 36, 75, 76, 123, 35, 36, 75, 76, 124, 37, 38, 77, 78, 125, 37, 38, 77, 78, 126, 23, 24, 79, 80, 127, 23, 24, 79, 80, 128, 25, 26, 81, 82, 129, 25, 26, 81, 82, 130, 27, 28, 83, 84, 131, 27, 28, 83, 84, 132, 29, 30, 85, 86, 133, 29, 30, 85, 86, 134, 15, 16, 87, 88, 15, 16, 87, 88, 17, 18, 89, 90, 17, 18, 89, 90, 19, 20, 91, 92, 19, 20, 91, 92, 21, 22, 93, 94, 21, 22, 93, 94, 47, 48, 53, 54, 95, 96, 49, 50, 53, 54, 97, 98, 51, 52, 53, 54, 99, 100, 47, 48, 53, 54, 95, 96, 49, 50, 53, 54, 97, 98, 51, 52, 53, 54, 99, 100, 95, 97, 99, 101, 96, 98, 100, 102, 55, 56, 61, 62, 103, 104, 57, 58, 61, 62, 105, 106, 59, 60, 61, 62, 107, 108, 55, 56, 61, 62, 103, 104, 57, 58, 61, 62, 105, 106, 59, 60, 61, 62, 107, 108, 103, 105, 107, 109, 104, 106, 108, 110, 63, 64, 69, 70, 111, 112, 65, 66, 69, 70, 113, 114, 67, 68, 69, 70, 115, 116, 63, 64, 69, 70, 111, 112, 65, 66, 69, 70, 113, 114, 67, 68, 69, 70, 115, 116, 111, 113, 115, 117, 112, 114, 116, 118, 71, 72, 77, 78, 119, 120, 73, 74, 77, 78, 121, 122, 75, 76, 77, 78, 123, 124, 71, 72, 77, 78, 119, 120, 73, 74, 77, 78, 121, 122, 75, 76, 77, 78, 123, 124, 119, 121, 123, 125, 120, 122, 124, 126, 79, 80, 85, 86, 127, 128, 81, 82, 85, 86, 129, 130, 83, 84, 85, 86, 131, 132, 79, 80, 85, 86, 127, 128, 81, 82, 85, 86, 129, 130, 83, 84, 85, 86, 131, 132, 127, 129, 131, 133, 128, 130, 132, 134, 1, 135, 138, 3, 136, 139, 5, 137, 140, 2, 135, 138, 4, 136, 139, 6, 137, 140, 141, 0, 141, 142]
    sp_jac_ini_ja = [0, 2, 13, 24, 35, 46, 57, 68, 78, 88, 98, 108, 118, 128, 138, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252, 260, 268, 276, 284, 292, 300, 308, 316, 324, 332, 340, 348, 356, 364, 372, 380, 388, 396, 404, 411, 418, 425, 432, 439, 446, 453, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 624, 628, 632, 636, 640, 644, 648, 652, 658, 664, 670, 676, 682, 688, 692, 696, 702, 708, 714, 720, 726, 732, 736, 740, 746, 752, 758, 764, 770, 776, 780, 784, 790, 796, 802, 808, 814, 820, 824, 828, 834, 840, 846, 852, 858, 864, 868, 872, 875, 878, 881, 884, 887, 890, 891, 894]
    sp_jac_ini_nia = 143
    sp_jac_ini_nja = 143
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 141, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 135, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 138, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 136, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 139, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 137, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 140, 1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 13, 14, 17, 18, 1, 2, 3, 4, 9, 10, 13, 14, 17, 18, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 7, 8, 15, 16, 23, 24, 87, 88, 7, 8, 15, 16, 23, 24, 87, 88, 9, 10, 17, 18, 25, 26, 89, 90, 9, 10, 17, 18, 25, 26, 89, 90, 11, 12, 19, 20, 27, 28, 91, 92, 11, 12, 19, 20, 27, 28, 91, 92, 13, 14, 21, 22, 29, 30, 93, 94, 13, 14, 21, 22, 29, 30, 93, 94, 15, 16, 23, 24, 31, 32, 79, 80, 15, 16, 23, 24, 31, 32, 79, 80, 17, 18, 25, 26, 33, 34, 81, 82, 17, 18, 25, 26, 33, 34, 81, 82, 19, 20, 27, 28, 35, 36, 83, 84, 19, 20, 27, 28, 35, 36, 83, 84, 21, 22, 29, 30, 37, 38, 85, 86, 21, 22, 29, 30, 37, 38, 85, 86, 23, 24, 31, 32, 39, 40, 71, 72, 23, 24, 31, 32, 39, 40, 71, 72, 25, 26, 33, 34, 41, 42, 73, 74, 25, 26, 33, 34, 41, 42, 73, 74, 27, 28, 35, 36, 43, 44, 75, 76, 27, 28, 35, 36, 43, 44, 75, 76, 29, 30, 37, 38, 45, 46, 77, 78, 29, 30, 37, 38, 45, 46, 77, 78, 31, 32, 39, 40, 47, 48, 63, 64, 31, 32, 39, 40, 47, 48, 63, 64, 33, 34, 41, 42, 49, 50, 65, 66, 33, 34, 41, 42, 49, 50, 65, 66, 35, 36, 43, 44, 51, 52, 67, 68, 35, 36, 43, 44, 51, 52, 67, 68, 37, 38, 45, 46, 53, 54, 69, 70, 37, 38, 45, 46, 53, 54, 69, 70, 39, 40, 47, 48, 55, 56, 95, 39, 40, 47, 48, 55, 56, 96, 41, 42, 49, 50, 57, 58, 97, 41, 42, 49, 50, 57, 58, 98, 43, 44, 51, 52, 59, 60, 99, 43, 44, 51, 52, 59, 60, 100, 45, 46, 53, 54, 61, 62, 101, 45, 46, 53, 54, 61, 62, 102, 47, 48, 55, 56, 103, 47, 48, 55, 56, 104, 49, 50, 57, 58, 105, 49, 50, 57, 58, 106, 51, 52, 59, 60, 107, 51, 52, 59, 60, 108, 53, 54, 61, 62, 109, 53, 54, 61, 62, 110, 39, 40, 63, 64, 111, 39, 40, 63, 64, 112, 41, 42, 65, 66, 113, 41, 42, 65, 66, 114, 43, 44, 67, 68, 115, 43, 44, 67, 68, 116, 45, 46, 69, 70, 117, 45, 46, 69, 70, 118, 31, 32, 71, 72, 119, 31, 32, 71, 72, 120, 33, 34, 73, 74, 121, 33, 34, 73, 74, 122, 35, 36, 75, 76, 123, 35, 36, 75, 76, 124, 37, 38, 77, 78, 125, 37, 38, 77, 78, 126, 23, 24, 79, 80, 127, 23, 24, 79, 80, 128, 25, 26, 81, 82, 129, 25, 26, 81, 82, 130, 27, 28, 83, 84, 131, 27, 28, 83, 84, 132, 29, 30, 85, 86, 133, 29, 30, 85, 86, 134, 15, 16, 87, 88, 15, 16, 87, 88, 17, 18, 89, 90, 17, 18, 89, 90, 19, 20, 91, 92, 19, 20, 91, 92, 21, 22, 93, 94, 21, 22, 93, 94, 47, 48, 53, 54, 95, 96, 49, 50, 53, 54, 97, 98, 51, 52, 53, 54, 99, 100, 47, 48, 53, 54, 95, 96, 49, 50, 53, 54, 97, 98, 51, 52, 53, 54, 99, 100, 95, 97, 99, 101, 96, 98, 100, 102, 55, 56, 61, 62, 103, 104, 57, 58, 61, 62, 105, 106, 59, 60, 61, 62, 107, 108, 55, 56, 61, 62, 103, 104, 57, 58, 61, 62, 105, 106, 59, 60, 61, 62, 107, 108, 103, 105, 107, 109, 104, 106, 108, 110, 63, 64, 69, 70, 111, 112, 65, 66, 69, 70, 113, 114, 67, 68, 69, 70, 115, 116, 63, 64, 69, 70, 111, 112, 65, 66, 69, 70, 113, 114, 67, 68, 69, 70, 115, 116, 111, 113, 115, 117, 112, 114, 116, 118, 71, 72, 77, 78, 119, 120, 73, 74, 77, 78, 121, 122, 75, 76, 77, 78, 123, 124, 71, 72, 77, 78, 119, 120, 73, 74, 77, 78, 121, 122, 75, 76, 77, 78, 123, 124, 119, 121, 123, 125, 120, 122, 124, 126, 79, 80, 85, 86, 127, 128, 81, 82, 85, 86, 129, 130, 83, 84, 85, 86, 131, 132, 79, 80, 85, 86, 127, 128, 81, 82, 85, 86, 129, 130, 83, 84, 85, 86, 131, 132, 127, 129, 131, 133, 128, 130, 132, 134, 1, 135, 138, 3, 136, 139, 5, 137, 140, 2, 135, 138, 4, 136, 139, 6, 137, 140, 141, 0, 141, 142]
    sp_jac_run_ja = [0, 2, 13, 24, 35, 46, 57, 68, 78, 88, 98, 108, 118, 128, 138, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252, 260, 268, 276, 284, 292, 300, 308, 316, 324, 332, 340, 348, 356, 364, 372, 380, 388, 396, 404, 411, 418, 425, 432, 439, 446, 453, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 624, 628, 632, 636, 640, 644, 648, 652, 658, 664, 670, 676, 682, 688, 692, 696, 702, 708, 714, 720, 726, 732, 736, 740, 746, 752, 758, 764, 770, 776, 780, 784, 790, 796, 802, 808, 814, 820, 824, 828, 834, 840, 846, 852, 858, 864, 868, 872, 875, 878, 881, 884, 887, 890, 891, 894]
    sp_jac_run_nia = 143
    sp_jac_run_nja = 143
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 141, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 135, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 138, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 136, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 139, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 137, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 140, 1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 1, 2, 5, 6, 7, 8, 13, 14, 15, 16, 1, 2, 3, 4, 9, 10, 13, 14, 17, 18, 1, 2, 3, 4, 9, 10, 13, 14, 17, 18, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 3, 4, 5, 6, 11, 12, 13, 14, 19, 20, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 7, 8, 15, 16, 23, 24, 87, 88, 7, 8, 15, 16, 23, 24, 87, 88, 9, 10, 17, 18, 25, 26, 89, 90, 9, 10, 17, 18, 25, 26, 89, 90, 11, 12, 19, 20, 27, 28, 91, 92, 11, 12, 19, 20, 27, 28, 91, 92, 13, 14, 21, 22, 29, 30, 93, 94, 13, 14, 21, 22, 29, 30, 93, 94, 15, 16, 23, 24, 31, 32, 79, 80, 15, 16, 23, 24, 31, 32, 79, 80, 17, 18, 25, 26, 33, 34, 81, 82, 17, 18, 25, 26, 33, 34, 81, 82, 19, 20, 27, 28, 35, 36, 83, 84, 19, 20, 27, 28, 35, 36, 83, 84, 21, 22, 29, 30, 37, 38, 85, 86, 21, 22, 29, 30, 37, 38, 85, 86, 23, 24, 31, 32, 39, 40, 71, 72, 23, 24, 31, 32, 39, 40, 71, 72, 25, 26, 33, 34, 41, 42, 73, 74, 25, 26, 33, 34, 41, 42, 73, 74, 27, 28, 35, 36, 43, 44, 75, 76, 27, 28, 35, 36, 43, 44, 75, 76, 29, 30, 37, 38, 45, 46, 77, 78, 29, 30, 37, 38, 45, 46, 77, 78, 31, 32, 39, 40, 47, 48, 63, 64, 31, 32, 39, 40, 47, 48, 63, 64, 33, 34, 41, 42, 49, 50, 65, 66, 33, 34, 41, 42, 49, 50, 65, 66, 35, 36, 43, 44, 51, 52, 67, 68, 35, 36, 43, 44, 51, 52, 67, 68, 37, 38, 45, 46, 53, 54, 69, 70, 37, 38, 45, 46, 53, 54, 69, 70, 39, 40, 47, 48, 55, 56, 95, 39, 40, 47, 48, 55, 56, 96, 41, 42, 49, 50, 57, 58, 97, 41, 42, 49, 50, 57, 58, 98, 43, 44, 51, 52, 59, 60, 99, 43, 44, 51, 52, 59, 60, 100, 45, 46, 53, 54, 61, 62, 101, 45, 46, 53, 54, 61, 62, 102, 47, 48, 55, 56, 103, 47, 48, 55, 56, 104, 49, 50, 57, 58, 105, 49, 50, 57, 58, 106, 51, 52, 59, 60, 107, 51, 52, 59, 60, 108, 53, 54, 61, 62, 109, 53, 54, 61, 62, 110, 39, 40, 63, 64, 111, 39, 40, 63, 64, 112, 41, 42, 65, 66, 113, 41, 42, 65, 66, 114, 43, 44, 67, 68, 115, 43, 44, 67, 68, 116, 45, 46, 69, 70, 117, 45, 46, 69, 70, 118, 31, 32, 71, 72, 119, 31, 32, 71, 72, 120, 33, 34, 73, 74, 121, 33, 34, 73, 74, 122, 35, 36, 75, 76, 123, 35, 36, 75, 76, 124, 37, 38, 77, 78, 125, 37, 38, 77, 78, 126, 23, 24, 79, 80, 127, 23, 24, 79, 80, 128, 25, 26, 81, 82, 129, 25, 26, 81, 82, 130, 27, 28, 83, 84, 131, 27, 28, 83, 84, 132, 29, 30, 85, 86, 133, 29, 30, 85, 86, 134, 15, 16, 87, 88, 15, 16, 87, 88, 17, 18, 89, 90, 17, 18, 89, 90, 19, 20, 91, 92, 19, 20, 91, 92, 21, 22, 93, 94, 21, 22, 93, 94, 47, 48, 53, 54, 95, 96, 49, 50, 53, 54, 97, 98, 51, 52, 53, 54, 99, 100, 47, 48, 53, 54, 95, 96, 49, 50, 53, 54, 97, 98, 51, 52, 53, 54, 99, 100, 95, 97, 99, 101, 96, 98, 100, 102, 55, 56, 61, 62, 103, 104, 57, 58, 61, 62, 105, 106, 59, 60, 61, 62, 107, 108, 55, 56, 61, 62, 103, 104, 57, 58, 61, 62, 105, 106, 59, 60, 61, 62, 107, 108, 103, 105, 107, 109, 104, 106, 108, 110, 63, 64, 69, 70, 111, 112, 65, 66, 69, 70, 113, 114, 67, 68, 69, 70, 115, 116, 63, 64, 69, 70, 111, 112, 65, 66, 69, 70, 113, 114, 67, 68, 69, 70, 115, 116, 111, 113, 115, 117, 112, 114, 116, 118, 71, 72, 77, 78, 119, 120, 73, 74, 77, 78, 121, 122, 75, 76, 77, 78, 123, 124, 71, 72, 77, 78, 119, 120, 73, 74, 77, 78, 121, 122, 75, 76, 77, 78, 123, 124, 119, 121, 123, 125, 120, 122, 124, 126, 79, 80, 85, 86, 127, 128, 81, 82, 85, 86, 129, 130, 83, 84, 85, 86, 131, 132, 79, 80, 85, 86, 127, 128, 81, 82, 85, 86, 129, 130, 83, 84, 85, 86, 131, 132, 127, 129, 131, 133, 128, 130, 132, 134, 1, 135, 138, 3, 136, 139, 5, 137, 140, 2, 135, 138, 4, 136, 139, 6, 137, 140, 141, 0, 141, 142]
    sp_jac_trap_ja = [0, 2, 13, 24, 35, 46, 57, 68, 78, 88, 98, 108, 118, 128, 138, 148, 156, 164, 172, 180, 188, 196, 204, 212, 220, 228, 236, 244, 252, 260, 268, 276, 284, 292, 300, 308, 316, 324, 332, 340, 348, 356, 364, 372, 380, 388, 396, 404, 411, 418, 425, 432, 439, 446, 453, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515, 520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580, 585, 590, 595, 600, 605, 610, 615, 620, 624, 628, 632, 636, 640, 644, 648, 652, 658, 664, 670, 676, 682, 688, 692, 696, 702, 708, 714, 720, 726, 732, 736, 740, 746, 752, 758, 764, 770, 776, 780, 784, 790, 796, 802, 808, 814, 820, 824, 828, 834, 840, 846, 852, 858, 864, 868, 872, 875, 878, 881, 884, 887, 890, 891, 894]
    sp_jac_trap_nia = 143
    sp_jac_trap_nja = 143
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
