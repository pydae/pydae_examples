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
    import newengland_ini_cffi as jacs_ini
    import newengland_run_cffi as jacs_run
    import newengland_trap_cffi as jacs_trap

if dae_file_mode == 'enviroment':
    import envus.no_enviroment.newengland_cffi as jacs
if dae_file_mode == 'colab':
    import newengland_cffi as jacs
    
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
        self.N_x = 170
        self.N_y = 226 
        self.N_z = 127 
        self.N_store = 100000 
        self.params_list = ['S_base', 'g_01_02', 'b_01_02', 'bs_01_02', 'g_01_39', 'b_01_39', 'bs_01_39', 'g_02_03', 'b_02_03', 'bs_02_03', 'g_02_25', 'b_02_25', 'bs_02_25', 'g_03_04', 'b_03_04', 'bs_03_04', 'g_03_18', 'b_03_18', 'bs_03_18', 'g_04_05', 'b_04_05', 'bs_04_05', 'g_04_14', 'b_04_14', 'bs_04_14', 'g_05_06', 'b_05_06', 'bs_05_06', 'g_05_08', 'b_05_08', 'bs_05_08', 'g_06_07', 'b_06_07', 'bs_06_07', 'g_06_11', 'b_06_11', 'bs_06_11', 'g_07_08', 'b_07_08', 'bs_07_08', 'g_08_09', 'b_08_09', 'bs_08_09', 'g_09_39', 'b_09_39', 'bs_09_39', 'g_10_11', 'b_10_11', 'bs_10_11', 'g_10_13', 'b_10_13', 'bs_10_13', 'g_13_14', 'b_13_14', 'bs_13_14', 'g_14_15', 'b_14_15', 'bs_14_15', 'g_15_16', 'b_15_16', 'bs_15_16', 'g_16_17', 'b_16_17', 'bs_16_17', 'g_16_19', 'b_16_19', 'bs_16_19', 'g_16_21', 'b_16_21', 'bs_16_21', 'g_16_24', 'b_16_24', 'bs_16_24', 'g_17_18', 'b_17_18', 'bs_17_18', 'g_17_27', 'b_17_27', 'bs_17_27', 'g_21_22', 'b_21_22', 'bs_21_22', 'g_22_23', 'b_22_23', 'bs_22_23', 'g_23_24', 'b_23_24', 'bs_23_24', 'g_25_26', 'b_25_26', 'bs_25_26', 'g_26_27', 'b_26_27', 'bs_26_27', 'g_26_28', 'b_26_28', 'bs_26_28', 'g_26_29', 'b_26_29', 'bs_26_29', 'g_28_29', 'b_28_29', 'bs_28_29', 'g_cc_12_11', 'b_cc_12_11', 'tap_12_11', 'ang_12_11', 'g_cc_12_13', 'b_cc_12_13', 'tap_12_13', 'ang_12_13', 'g_cc_06_31', 'b_cc_06_31', 'tap_06_31', 'ang_06_31', 'g_cc_10_32', 'b_cc_10_32', 'tap_10_32', 'ang_10_32', 'g_cc_19_33', 'b_cc_19_33', 'tap_19_33', 'ang_19_33', 'g_cc_20_34', 'b_cc_20_34', 'tap_20_34', 'ang_20_34', 'g_cc_22_35', 'b_cc_22_35', 'tap_22_35', 'ang_22_35', 'g_cc_23_36', 'b_cc_23_36', 'tap_23_36', 'ang_23_36', 'g_cc_25_37', 'b_cc_25_37', 'tap_25_37', 'ang_25_37', 'g_cc_02_30', 'b_cc_02_30', 'tap_02_30', 'ang_02_30', 'g_cc_29_38', 'b_cc_29_38', 'tap_29_38', 'ang_29_38', 'g_cc_19_20', 'b_cc_19_20', 'tap_19_20', 'ang_19_20', 'g_shunt_16', 'b_shunt_16', 'U_01_n', 'U_02_n', 'U_03_n', 'U_04_n', 'U_05_n', 'U_06_n', 'U_07_n', 'U_08_n', 'U_09_n', 'U_10_n', 'U_11_n', 'U_12_n', 'U_13_n', 'U_14_n', 'U_15_n', 'U_16_n', 'U_17_n', 'U_18_n', 'U_19_n', 'U_20_n', 'U_21_n', 'U_22_n', 'U_23_n', 'U_24_n', 'U_25_n', 'U_26_n', 'U_27_n', 'U_28_n', 'U_29_n', 'U_30_n', 'U_31_n', 'U_32_n', 'U_33_n', 'U_34_n', 'U_35_n', 'U_36_n', 'U_37_n', 'U_38_n', 'U_39_n', 'Omega_b_G10', 'S_n_G10', 'H_G10', 'T1d0_G10', 'T1q0_G10', 'X_d_G10', 'X_q_G10', 'X1d_G10', 'X1q_G10', 'D_G10', 'R_a_G10', 'K_delta_G10', 'K_sec_G10', 'K_a_G10', 'K_ai_G10', 'T_r_G10', 'T_c_G10', 'T_b_G10', 'V_f_max_G10', 'V_f_min_G10', 'Droop_G10', 'T_gov_1_G10', 'T_gov_2_G10', 'T_gov_3_G10', 'D_t_G10', 'omega_ref_G10', 'T_wo_pss_G10', 'T_1_pss_G10', 'T_2_pss_G10', 'T_3_pss_G10', 'T_4_pss_G10', 'K_stab_G10', 'V_lim_pss_G10', 'Omega_b_G02', 'S_n_G02', 'H_G02', 'T1d0_G02', 'T1q0_G02', 'X_d_G02', 'X_q_G02', 'X1d_G02', 'X1q_G02', 'D_G02', 'R_a_G02', 'K_delta_G02', 'K_sec_G02', 'K_a_G02', 'K_ai_G02', 'T_r_G02', 'T_c_G02', 'T_b_G02', 'V_f_max_G02', 'V_f_min_G02', 'Droop_G02', 'T_gov_1_G02', 'T_gov_2_G02', 'T_gov_3_G02', 'D_t_G02', 'omega_ref_G02', 'T_wo_pss_G02', 'T_1_pss_G02', 'T_2_pss_G02', 'T_3_pss_G02', 'T_4_pss_G02', 'K_stab_G02', 'V_lim_pss_G02', 'Omega_b_G03', 'S_n_G03', 'H_G03', 'T1d0_G03', 'T1q0_G03', 'X_d_G03', 'X_q_G03', 'X1d_G03', 'X1q_G03', 'D_G03', 'R_a_G03', 'K_delta_G03', 'K_sec_G03', 'K_a_G03', 'K_ai_G03', 'T_r_G03', 'T_c_G03', 'T_b_G03', 'V_f_max_G03', 'V_f_min_G03', 'Droop_G03', 'T_gov_1_G03', 'T_gov_2_G03', 'T_gov_3_G03', 'D_t_G03', 'omega_ref_G03', 'T_wo_pss_G03', 'T_1_pss_G03', 'T_2_pss_G03', 'T_3_pss_G03', 'T_4_pss_G03', 'K_stab_G03', 'V_lim_pss_G03', 'Omega_b_G04', 'S_n_G04', 'H_G04', 'T1d0_G04', 'T1q0_G04', 'X_d_G04', 'X_q_G04', 'X1d_G04', 'X1q_G04', 'D_G04', 'R_a_G04', 'K_delta_G04', 'K_sec_G04', 'K_a_G04', 'K_ai_G04', 'T_r_G04', 'T_c_G04', 'T_b_G04', 'V_f_max_G04', 'V_f_min_G04', 'Droop_G04', 'T_gov_1_G04', 'T_gov_2_G04', 'T_gov_3_G04', 'D_t_G04', 'omega_ref_G04', 'T_wo_pss_G04', 'T_1_pss_G04', 'T_2_pss_G04', 'T_3_pss_G04', 'T_4_pss_G04', 'K_stab_G04', 'V_lim_pss_G04', 'Omega_b_G05', 'S_n_G05', 'H_G05', 'T1d0_G05', 'T1q0_G05', 'X_d_G05', 'X_q_G05', 'X1d_G05', 'X1q_G05', 'D_G05', 'R_a_G05', 'K_delta_G05', 'K_sec_G05', 'K_a_G05', 'K_ai_G05', 'T_r_G05', 'T_c_G05', 'T_b_G05', 'V_f_max_G05', 'V_f_min_G05', 'Droop_G05', 'T_gov_1_G05', 'T_gov_2_G05', 'T_gov_3_G05', 'D_t_G05', 'omega_ref_G05', 'T_wo_pss_G05', 'T_1_pss_G05', 'T_2_pss_G05', 'T_3_pss_G05', 'T_4_pss_G05', 'K_stab_G05', 'V_lim_pss_G05', 'Omega_b_G06', 'S_n_G06', 'H_G06', 'T1d0_G06', 'T1q0_G06', 'X_d_G06', 'X_q_G06', 'X1d_G06', 'X1q_G06', 'D_G06', 'R_a_G06', 'K_delta_G06', 'K_sec_G06', 'K_a_G06', 'K_ai_G06', 'T_r_G06', 'T_c_G06', 'T_b_G06', 'V_f_max_G06', 'V_f_min_G06', 'Droop_G06', 'T_gov_1_G06', 'T_gov_2_G06', 'T_gov_3_G06', 'D_t_G06', 'omega_ref_G06', 'T_wo_pss_G06', 'T_1_pss_G06', 'T_2_pss_G06', 'T_3_pss_G06', 'T_4_pss_G06', 'K_stab_G06', 'V_lim_pss_G06', 'Omega_b_G07', 'S_n_G07', 'H_G07', 'T1d0_G07', 'T1q0_G07', 'X_d_G07', 'X_q_G07', 'X1d_G07', 'X1q_G07', 'D_G07', 'R_a_G07', 'K_delta_G07', 'K_sec_G07', 'K_a_G07', 'K_ai_G07', 'T_r_G07', 'T_c_G07', 'T_b_G07', 'V_f_max_G07', 'V_f_min_G07', 'Droop_G07', 'T_gov_1_G07', 'T_gov_2_G07', 'T_gov_3_G07', 'D_t_G07', 'omega_ref_G07', 'T_wo_pss_G07', 'T_1_pss_G07', 'T_2_pss_G07', 'T_3_pss_G07', 'T_4_pss_G07', 'K_stab_G07', 'V_lim_pss_G07', 'Omega_b_G08', 'S_n_G08', 'H_G08', 'T1d0_G08', 'T1q0_G08', 'X_d_G08', 'X_q_G08', 'X1d_G08', 'X1q_G08', 'D_G08', 'R_a_G08', 'K_delta_G08', 'K_sec_G08', 'K_a_G08', 'K_ai_G08', 'T_r_G08', 'T_c_G08', 'T_b_G08', 'V_f_max_G08', 'V_f_min_G08', 'Droop_G08', 'T_gov_1_G08', 'T_gov_2_G08', 'T_gov_3_G08', 'D_t_G08', 'omega_ref_G08', 'T_wo_pss_G08', 'T_1_pss_G08', 'T_2_pss_G08', 'T_3_pss_G08', 'T_4_pss_G08', 'K_stab_G08', 'V_lim_pss_G08', 'Omega_b_G09', 'S_n_G09', 'H_G09', 'T1d0_G09', 'T1q0_G09', 'X_d_G09', 'X_q_G09', 'X1d_G09', 'X1q_G09', 'D_G09', 'R_a_G09', 'K_delta_G09', 'K_sec_G09', 'K_a_G09', 'K_ai_G09', 'T_r_G09', 'T_c_G09', 'T_b_G09', 'V_f_max_G09', 'V_f_min_G09', 'Droop_G09', 'T_gov_1_G09', 'T_gov_2_G09', 'T_gov_3_G09', 'D_t_G09', 'omega_ref_G09', 'T_wo_pss_G09', 'T_1_pss_G09', 'T_2_pss_G09', 'T_3_pss_G09', 'T_4_pss_G09', 'K_stab_G09', 'V_lim_pss_G09', 'Omega_b_G01', 'S_n_G01', 'H_G01', 'T1d0_G01', 'T1q0_G01', 'X_d_G01', 'X_q_G01', 'X1d_G01', 'X1q_G01', 'D_G01', 'R_a_G01', 'K_delta_G01', 'K_sec_G01', 'K_a_G01', 'K_ai_G01', 'T_r_G01', 'T_c_G01', 'T_b_G01', 'V_f_max_G01', 'V_f_min_G01', 'Droop_G01', 'T_gov_1_G01', 'T_gov_2_G01', 'T_gov_3_G01', 'D_t_G01', 'omega_ref_G01', 'T_wo_pss_G01', 'T_1_pss_G01', 'T_2_pss_G01', 'T_3_pss_G01', 'T_4_pss_G01', 'K_stab_G01', 'V_lim_pss_G01', 'T_pz_03', 'T_qz_03', 'T_pz_04', 'T_qz_04', 'T_pz_07', 'T_qz_07', 'T_pz_08', 'T_qz_08', 'T_pz_12', 'T_qz_12', 'T_pz_15', 'T_qz_15', 'T_pz_16', 'T_qz_16', 'T_pz_18', 'T_qz_18', 'T_pz_20', 'T_qz_20', 'T_pz_21', 'T_qz_21', 'T_pz_23', 'T_qz_23', 'T_pz_24', 'T_qz_24', 'T_pz_25', 'T_qz_25', 'T_pz_26', 'T_qz_26', 'T_pz_27', 'T_qz_27', 'T_pz_28', 'T_qz_28', 'T_pz_29', 'T_qz_29', 'T_pz_31', 'T_qz_31', 'T_pz_39', 'T_qz_39', 'RampDown_16', 'RampUp_16', 'K_fault_16', 'K_p_agc', 'K_i_agc', 'K_xif'] 
        self.params_values_list  = [100000000, 2.0570568805614005, -24.155725083163873, 0.6987, 1.5974440894568687, -39.93610223642172, 0.75, 5.659555942533739, -65.73791902481499, 0.2572, 56.92908262849707, -69.94144437215354, 0.146, 2.8547586630945583, -46.77412271070315, 0.2214, 6.17630544637844, -74.6771476698484, 0.2138, 4.863813229571985, -77.82101167315176, 0.1342, 4.788985333732416, -77.2223885064352, 0.1382, 29.41176470588236, -382.3529411764706, 0.0434, 6.34517766497462, -88.83248730964468, 0.1476, 7.058823529411764, -108.23529411764704, 0.113, 10.335154289089028, -121.06895024361432, 0.1389, 18.761726078799253, -215.75984990619136, 0.078, 1.7384994482153926, -27.438056508790762, 0.3804, 1.5974440894568687, -39.93610223642172, 1.2, 21.447721179624665, -230.56300268096516, 0.0729, 21.447721179624665, -230.56300268096516, 0.0729, 8.753160863645206, -98.22991635868509, 0.1723, 3.7964271402358, -45.76803830173159, 0.366, 10.093080632499719, -105.41661993944152, 0.171, 8.782936010037641, -111.66875784190715, 0.1342, 4.179619132206578, -50.93910817376767, 0.304, 4.3742140084203625, -73.81486139209362, 0.2548, 8.595988538681947, -169.05444126074497, 0.068, 10.335154289089028, -121.06895024361432, 0.1319, 4.319223868695595, -57.478902252641376, 0.3216, 4.068348250610252, -71.19609438567942, 0.2565, 6.485084306095979, -103.76134889753567, 0.1846, 1.7888505821895528, -28.458986534833794, 0.361, 3.037407572636755, -30.658832686302244, 0.513, 6.420545746388443, -67.41573033707866, 0.2396, 1.8982452267961594, -20.92484273259022, 0.7802, 1.4471633060318785, -15.868018706489893, 1.029, 6.087750576162108, -65.66073835717702, 0.249, 0.8444118407650373, -22.95744692079945, 1.006, 0.0, 0.8444118407650373, -22.95744692079945, 1.006, 0.0, 0.0, -39.99999999999999, 1.07, 0.0, 0.0, -50.0, 1.07, 0.0, 3.463117795478157, -70.25181813684263, 1.07, 0.0, 2.770850651149903, -55.41701302299806, 1.009, 0.0, 0.0, -69.93006993006992, 1.025, 0.0, 0.6755935088975665, -36.75228688402762, 1, 0.0, 1.1139992573338284, -43.07463795024137, 1.025, 0.0, 0.0, -55.248618784530386, 1.025, 0.0, 3.2786885245901645, -63.934426229508205, 1.025, 0.0, 3.666265123343634, -72.27779814591736, 1.06, 0.0, 0.0, -1e-06, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 400000, 376.99111843077515, 312500000, 13.44, 10.2, 0.1, 0.3125, 0.2156, 0.0969, 0.025, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 1, 0.05, 3, 0.5, 1, 0.1, 376.99111843077515, 651000000, 4.65, 6.56, 1.5, 1.9205, 1.8359, 0.4538, 1.1067, 0, 0, 0.01, 1, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 5, 0.4, 1, 0.1, 1, 0.1, 376.99111843077515, 812500000, 4.41, 5.7, 1.5, 2.0272, 1.9256, 0.4314, 0.7118, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 3, 0.2, 2, 0.2, 1, 0.1, 376.99111843077515, 790000000, 3.62, 5.69, 1.5, 2.0698, 2.0382, 0.3444, 1.3114, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 1, 0.1, 1, 0.3, 1, 0.1, 376.99111843077515, 635000000, 4.09, 5.4, 0.44, 4.2545, 3.937, 0.8382, 1.0541, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 1.5, 0.2, 1, 0.1, 1, 0.1, 376.99111843077515, 812500000, 4.28, 7.3, 0.4, 2.0638, 1.9581, 0.4062, 0.6614, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 0.5, 0.1, 0.5, 0.05, 1, 0.1, 376.99111843077515, 700000000, 3.77, 5.66, 1.5, 2.065, 2.044, 0.343, 1.302, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 0.2, 0.02, 0.5, 0.1, 1, 0.1, 376.99111843077515, 675000000, 3.6, 6.7, 0.41, 1.9575, 1.89, 0.3848, 0.6149, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 1, 0.2, 1, 0.1, 1, 0.1, 376.99111843077515, 1037500000, 3.33, 4.79, 1.96, 2.185, 2.1269, 0.5914, 0.609, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 1, 0.5, 2, 0.1, 1, 0.1, 376.99111843077515, 1250000000, 40, 7, 0.7, 0.25, 0.2375, 0.075, 0.1, 0, 0, 0, 0, 200, 1e-06, 0.01, 1, 10, 10.0, -10.0, 0.05, 1, 1, 1, 0, 1.0, 10, 5, 0.6, 3, 0.5, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -20000, 10000, 10000, 0.01, 0.01, 0.0] 
        self.inputs_ini_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'P_13', 'Q_13', 'P_14', 'Q_14', 'P_15', 'Q_15', 'P_16', 'Q_16', 'P_17', 'Q_17', 'P_18', 'Q_18', 'P_19', 'Q_19', 'P_20', 'Q_20', 'P_21', 'Q_21', 'P_22', 'Q_22', 'P_23', 'Q_23', 'P_24', 'Q_24', 'P_25', 'Q_25', 'P_26', 'Q_26', 'P_27', 'Q_27', 'P_28', 'Q_28', 'P_29', 'Q_29', 'P_30', 'Q_30', 'P_31', 'Q_31', 'P_32', 'Q_32', 'P_33', 'Q_33', 'P_34', 'Q_34', 'P_35', 'Q_35', 'P_36', 'Q_36', 'P_37', 'Q_37', 'P_38', 'Q_38', 'P_39', 'Q_39', 'v_ref_G10', 'p_c_G10', 'v_ref_G02', 'p_c_G02', 'v_ref_G03', 'p_c_G03', 'v_ref_G04', 'p_c_G04', 'v_ref_G05', 'p_c_G05', 'v_ref_G06', 'p_c_G06', 'v_ref_G07', 'p_c_G07', 'v_ref_G08', 'p_c_G08', 'v_ref_G09', 'p_c_G09', 'v_ref_G01', 'p_c_G01', 'p_z_03', 'q_z_03', 'p_i_03', 'q_i_03', 'p_p_03', 'q_p_03', 'p_z_04', 'q_z_04', 'p_i_04', 'q_i_04', 'p_p_04', 'q_p_04', 'p_z_07', 'q_z_07', 'p_i_07', 'q_i_07', 'p_p_07', 'q_p_07', 'p_z_08', 'q_z_08', 'p_i_08', 'q_i_08', 'p_p_08', 'q_p_08', 'p_z_12', 'q_z_12', 'p_i_12', 'q_i_12', 'p_p_12', 'q_p_12', 'p_z_15', 'q_z_15', 'p_i_15', 'q_i_15', 'p_p_15', 'q_p_15', 'p_z_16', 'q_z_16', 'p_i_16', 'q_i_16', 'p_p_16', 'q_p_16', 'p_z_18', 'q_z_18', 'p_i_18', 'q_i_18', 'p_p_18', 'q_p_18', 'p_z_20', 'q_z_20', 'p_i_20', 'q_i_20', 'p_p_20', 'q_p_20', 'p_z_21', 'q_z_21', 'p_i_21', 'q_i_21', 'p_p_21', 'q_p_21', 'p_z_23', 'q_z_23', 'p_i_23', 'q_i_23', 'p_p_23', 'q_p_23', 'p_z_24', 'q_z_24', 'p_i_24', 'q_i_24', 'p_p_24', 'q_p_24', 'p_z_25', 'q_z_25', 'p_i_25', 'q_i_25', 'p_p_25', 'q_p_25', 'p_z_26', 'q_z_26', 'p_i_26', 'q_i_26', 'p_p_26', 'q_p_26', 'p_z_27', 'q_z_27', 'p_i_27', 'q_i_27', 'p_p_27', 'q_p_27', 'p_z_28', 'q_z_28', 'p_i_28', 'q_i_28', 'p_p_28', 'q_p_28', 'p_z_29', 'q_z_29', 'p_i_29', 'q_i_29', 'p_p_29', 'q_p_29', 'p_z_31', 'q_z_31', 'p_i_31', 'q_i_31', 'p_p_31', 'q_p_31', 'p_z_39', 'q_z_39', 'p_i_39', 'q_i_39', 'p_p_39', 'q_p_39', 'fault_b_16', 'fault_g_ref_16'] 
        self.inputs_ini_values_list  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0475, 0, 0.982, 0, 0.9831, 0, 0.9972, 0, 1.0123, 0, 1.0493, 0, 1.0635, 0, 1.0278, 0, 1.0265, 0, 1.03, 0, 0.0, 0.0, 0.0, 0.0, 3.22, 0.024, 0.0, 0.0, 0.0, 0.0, 5.0, 1.84, 0.0, 0.0, 0.0, 0.0, 2.338, 0.84, 0.0, 0.0, 0.0, 0.0, 5.22, 1.76, 0.0, 0.0, 0.0, 0.0, 0.075, 0.88, 0.0, 0.0, 0.0, 0.0, 3.2, 1.53, 0.0, 0.0, 0.0, 0.0, 3.29, 0.32299999999999995, 0.0, 0.0, 0.0, 0.0, 1.58, 0.3, 0.0, 0.0, 0.0, 0.0, 6.28, 1.03, 0.0, 0.0, 0.0, 0.0, 2.74, 1.15, 0.0, 0.0, 0.0, 0.0, 2.475, 0.846, 0.0, 0.0, 0.0, -0.0, 3.086, -0.92, 0.0, 0.0, 0.0, 0.0, 2.24, 0.472, 0.0, 0.0, 0.0, 0.0, 1.39, 0.17, 0.0, 0.0, 0.0, 0.0, 2.81, 0.755, 0.0, 0.0, 0.0, 0.0, 2.06, 0.276, 0.0, 0.0, 0.0, 0.0, 2.835, 0.269, 0.0, 0.0, 0.0, 0.0, 0.092, 0.046, 0.0, 0.0, 0.0, 0.0, 11.04, 2.5, 0.0, 0.0] 
        self.inputs_run_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'P_13', 'Q_13', 'P_14', 'Q_14', 'P_15', 'Q_15', 'P_16', 'Q_16', 'P_17', 'Q_17', 'P_18', 'Q_18', 'P_19', 'Q_19', 'P_20', 'Q_20', 'P_21', 'Q_21', 'P_22', 'Q_22', 'P_23', 'Q_23', 'P_24', 'Q_24', 'P_25', 'Q_25', 'P_26', 'Q_26', 'P_27', 'Q_27', 'P_28', 'Q_28', 'P_29', 'Q_29', 'P_30', 'Q_30', 'P_31', 'Q_31', 'P_32', 'Q_32', 'P_33', 'Q_33', 'P_34', 'Q_34', 'P_35', 'Q_35', 'P_36', 'Q_36', 'P_37', 'Q_37', 'P_38', 'Q_38', 'P_39', 'Q_39', 'v_ref_G10', 'p_c_G10', 'v_ref_G02', 'p_c_G02', 'v_ref_G03', 'p_c_G03', 'v_ref_G04', 'p_c_G04', 'v_ref_G05', 'p_c_G05', 'v_ref_G06', 'p_c_G06', 'v_ref_G07', 'p_c_G07', 'v_ref_G08', 'p_c_G08', 'v_ref_G09', 'p_c_G09', 'v_ref_G01', 'p_c_G01', 'g_load_03', 'b_load_03', 'i_p_03', 'i_q_03', 'p_p_03', 'q_p_03', 'g_load_04', 'b_load_04', 'i_p_04', 'i_q_04', 'p_p_04', 'q_p_04', 'g_load_07', 'b_load_07', 'i_p_07', 'i_q_07', 'p_p_07', 'q_p_07', 'g_load_08', 'b_load_08', 'i_p_08', 'i_q_08', 'p_p_08', 'q_p_08', 'g_load_12', 'b_load_12', 'i_p_12', 'i_q_12', 'p_p_12', 'q_p_12', 'g_load_15', 'b_load_15', 'i_p_15', 'i_q_15', 'p_p_15', 'q_p_15', 'g_load_16', 'b_load_16', 'i_p_16', 'i_q_16', 'p_p_16', 'q_p_16', 'g_load_18', 'b_load_18', 'i_p_18', 'i_q_18', 'p_p_18', 'q_p_18', 'g_load_20', 'b_load_20', 'i_p_20', 'i_q_20', 'p_p_20', 'q_p_20', 'g_load_21', 'b_load_21', 'i_p_21', 'i_q_21', 'p_p_21', 'q_p_21', 'g_load_23', 'b_load_23', 'i_p_23', 'i_q_23', 'p_p_23', 'q_p_23', 'g_load_24', 'b_load_24', 'i_p_24', 'i_q_24', 'p_p_24', 'q_p_24', 'g_load_25', 'b_load_25', 'i_p_25', 'i_q_25', 'p_p_25', 'q_p_25', 'g_load_26', 'b_load_26', 'i_p_26', 'i_q_26', 'p_p_26', 'q_p_26', 'g_load_27', 'b_load_27', 'i_p_27', 'i_q_27', 'p_p_27', 'q_p_27', 'g_load_28', 'b_load_28', 'i_p_28', 'i_q_28', 'p_p_28', 'q_p_28', 'g_load_29', 'b_load_29', 'i_p_29', 'i_q_29', 'p_p_29', 'q_p_29', 'g_load_31', 'b_load_31', 'i_p_31', 'i_q_31', 'p_p_31', 'q_p_31', 'g_load_39', 'b_load_39', 'i_p_39', 'i_q_39', 'p_p_39', 'q_p_39', 'fault_b_16', 'fault_g_ref_16'] 
        self.inputs_run_values_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0475, 0, 0.982, 0, 0.9831, 0, 0.9972, 0, 1.0123, 0, 1.0493, 0, 1.0635, 0, 1.0278, 0, 1.0265, 0, 1.03, 0, 0.0, 0.0, 0.0, 0.0, 3.22, 0.024, 0.0, 0.0, 0.0, 0.0, 5.0, 1.84, 0.0, 0.0, 0.0, 0.0, 2.338, 0.84, 0.0, 0.0, 0.0, 0.0, 5.22, 1.76, 0.0, 0.0, 0.0, 0.0, 0.075, 0.88, 0.0, 0.0, 0.0, 0.0, 3.2, 1.53, 0.0, 0.0, 0.0, 0.0, 3.29, 0.32299999999999995, 0.0, 0.0, 0.0, 0.0, 1.58, 0.3, 0.0, 0.0, 0.0, 0.0, 6.28, 1.03, 0.0, 0.0, 0.0, 0.0, 2.74, 1.15, 0.0, 0.0, 0.0, 0.0, 2.475, 0.846, 0.0, 0.0, 0.0, 0.0, 3.086, -0.92, 0.0, 0.0, 0.0, 0.0, 2.24, 0.472, 0.0, 0.0, 0.0, 0.0, 1.39, 0.17, 0.0, 0.0, 0.0, 0.0, 2.81, 0.755, 0.0, 0.0, 0.0, 0.0, 2.06, 0.276, 0.0, 0.0, 0.0, 0.0, 2.835, 0.269, 0.0, 0.0, 0.0, 0.0, 0.092, 0.046, 0.0, 0.0, 0.0, 0.0, 11.04, 2.5, 0.0, 0.0] 
        self.outputs_list = ['V_01', 'V_02', 'V_03', 'V_04', 'V_05', 'V_06', 'V_07', 'V_08', 'V_09', 'V_10', 'V_11', 'V_12', 'V_13', 'V_14', 'V_15', 'V_16', 'V_17', 'V_18', 'V_19', 'V_20', 'V_21', 'V_22', 'V_23', 'V_24', 'V_25', 'V_26', 'V_27', 'V_28', 'V_29', 'V_30', 'V_31', 'V_32', 'V_33', 'V_34', 'V_35', 'V_36', 'V_37', 'V_38', 'V_39', 'p_e_G10', 'v_f_G10', 'p_m_G10', 'v_ref_G10', 'p_c_G10', 'p_e_G02', 'v_f_G02', 'p_m_G02', 'v_ref_G02', 'p_c_G02', 'p_e_G03', 'v_f_G03', 'p_m_G03', 'v_ref_G03', 'p_c_G03', 'p_e_G04', 'v_f_G04', 'p_m_G04', 'v_ref_G04', 'p_c_G04', 'p_e_G05', 'v_f_G05', 'p_m_G05', 'v_ref_G05', 'p_c_G05', 'p_e_G06', 'v_f_G06', 'p_m_G06', 'v_ref_G06', 'p_c_G06', 'p_e_G07', 'v_f_G07', 'p_m_G07', 'v_ref_G07', 'p_c_G07', 'p_e_G08', 'v_f_G08', 'p_m_G08', 'v_ref_G08', 'p_c_G08', 'p_e_G09', 'v_f_G09', 'p_m_G09', 'v_ref_G09', 'p_c_G09', 'p_e_G01', 'v_f_G01', 'p_m_G01', 'v_ref_G01', 'p_c_G01', 'p_load_03', 'q_load_03', 'p_load_04', 'q_load_04', 'p_load_07', 'q_load_07', 'p_load_08', 'q_load_08', 'p_load_12', 'q_load_12', 'p_load_15', 'q_load_15', 'p_load_16', 'q_load_16', 'p_load_18', 'q_load_18', 'p_load_20', 'q_load_20', 'p_load_21', 'q_load_21', 'p_load_23', 'q_load_23', 'p_load_24', 'q_load_24', 'p_load_25', 'q_load_25', 'p_load_26', 'q_load_26', 'p_load_27', 'q_load_27', 'p_load_28', 'q_load_28', 'p_load_29', 'q_load_29', 'p_load_31', 'q_load_31', 'p_load_39', 'q_load_39'] 
        self.x_list = ['delta_G10', 'omega_G10', 'e1q_G10', 'e1d_G10', 'v_r_G10', 'x_cb_G10', 'xi_v_G10', 'v_f_G10', 'x_gov_1_G10', 'x_gov_2_G10', 'x_wo_pss_G10', 'x_12_pss_G10', 'x_34_pss_G10', 'delta_G02', 'omega_G02', 'e1q_G02', 'e1d_G02', 'v_r_G02', 'x_cb_G02', 'xi_v_G02', 'v_f_G02', 'x_gov_1_G02', 'x_gov_2_G02', 'x_wo_pss_G02', 'x_12_pss_G02', 'x_34_pss_G02', 'delta_G03', 'omega_G03', 'e1q_G03', 'e1d_G03', 'v_r_G03', 'x_cb_G03', 'xi_v_G03', 'v_f_G03', 'x_gov_1_G03', 'x_gov_2_G03', 'x_wo_pss_G03', 'x_12_pss_G03', 'x_34_pss_G03', 'delta_G04', 'omega_G04', 'e1q_G04', 'e1d_G04', 'v_r_G04', 'x_cb_G04', 'xi_v_G04', 'v_f_G04', 'x_gov_1_G04', 'x_gov_2_G04', 'x_wo_pss_G04', 'x_12_pss_G04', 'x_34_pss_G04', 'delta_G05', 'omega_G05', 'e1q_G05', 'e1d_G05', 'v_r_G05', 'x_cb_G05', 'xi_v_G05', 'v_f_G05', 'x_gov_1_G05', 'x_gov_2_G05', 'x_wo_pss_G05', 'x_12_pss_G05', 'x_34_pss_G05', 'delta_G06', 'omega_G06', 'e1q_G06', 'e1d_G06', 'v_r_G06', 'x_cb_G06', 'xi_v_G06', 'v_f_G06', 'x_gov_1_G06', 'x_gov_2_G06', 'x_wo_pss_G06', 'x_12_pss_G06', 'x_34_pss_G06', 'delta_G07', 'omega_G07', 'e1q_G07', 'e1d_G07', 'v_r_G07', 'x_cb_G07', 'xi_v_G07', 'v_f_G07', 'x_gov_1_G07', 'x_gov_2_G07', 'x_wo_pss_G07', 'x_12_pss_G07', 'x_34_pss_G07', 'delta_G08', 'omega_G08', 'e1q_G08', 'e1d_G08', 'v_r_G08', 'x_cb_G08', 'xi_v_G08', 'v_f_G08', 'x_gov_1_G08', 'x_gov_2_G08', 'x_wo_pss_G08', 'x_12_pss_G08', 'x_34_pss_G08', 'delta_G09', 'omega_G09', 'e1q_G09', 'e1d_G09', 'v_r_G09', 'x_cb_G09', 'xi_v_G09', 'v_f_G09', 'x_gov_1_G09', 'x_gov_2_G09', 'x_wo_pss_G09', 'x_12_pss_G09', 'x_34_pss_G09', 'delta_G01', 'omega_G01', 'e1q_G01', 'e1d_G01', 'v_r_G01', 'x_cb_G01', 'xi_v_G01', 'v_f_G01', 'x_gov_1_G01', 'x_gov_2_G01', 'x_wo_pss_G01', 'x_12_pss_G01', 'x_34_pss_G01', 'p_z_f_03', 'q_z_f_03', 'p_z_f_04', 'q_z_f_04', 'p_z_f_07', 'q_z_f_07', 'p_z_f_08', 'q_z_f_08', 'p_z_f_12', 'q_z_f_12', 'p_z_f_15', 'q_z_f_15', 'p_z_f_16', 'q_z_f_16', 'p_z_f_18', 'q_z_f_18', 'p_z_f_20', 'q_z_f_20', 'p_z_f_21', 'q_z_f_21', 'p_z_f_23', 'q_z_f_23', 'p_z_f_24', 'q_z_f_24', 'p_z_f_25', 'q_z_f_25', 'p_z_f_26', 'q_z_f_26', 'p_z_f_27', 'q_z_f_27', 'p_z_f_28', 'q_z_f_28', 'p_z_f_29', 'q_z_f_29', 'p_z_f_31', 'q_z_f_31', 'p_z_f_39', 'q_z_f_39', 'fault_g_16', 'xi_freq'] 
        self.y_run_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'V_13', 'theta_13', 'V_14', 'theta_14', 'V_15', 'theta_15', 'V_16', 'theta_16', 'V_17', 'theta_17', 'V_18', 'theta_18', 'V_19', 'theta_19', 'V_20', 'theta_20', 'V_21', 'theta_21', 'V_22', 'theta_22', 'V_23', 'theta_23', 'V_24', 'theta_24', 'V_25', 'theta_25', 'V_26', 'theta_26', 'V_27', 'theta_27', 'V_28', 'theta_28', 'V_29', 'theta_29', 'V_30', 'theta_30', 'V_31', 'theta_31', 'V_32', 'theta_32', 'V_33', 'theta_33', 'V_34', 'theta_34', 'V_35', 'theta_35', 'V_36', 'theta_36', 'V_37', 'theta_37', 'V_38', 'theta_38', 'V_39', 'theta_39', 'i_d_G10', 'i_q_G10', 'p_g_G10', 'q_g_G10', 'p_m_ref_G10', 'p_m_G10', 'v_pss_G10', 'i_d_G02', 'i_q_G02', 'p_g_G02', 'q_g_G02', 'p_m_ref_G02', 'p_m_G02', 'v_pss_G02', 'i_d_G03', 'i_q_G03', 'p_g_G03', 'q_g_G03', 'p_m_ref_G03', 'p_m_G03', 'v_pss_G03', 'i_d_G04', 'i_q_G04', 'p_g_G04', 'q_g_G04', 'p_m_ref_G04', 'p_m_G04', 'v_pss_G04', 'i_d_G05', 'i_q_G05', 'p_g_G05', 'q_g_G05', 'p_m_ref_G05', 'p_m_G05', 'v_pss_G05', 'i_d_G06', 'i_q_G06', 'p_g_G06', 'q_g_G06', 'p_m_ref_G06', 'p_m_G06', 'v_pss_G06', 'i_d_G07', 'i_q_G07', 'p_g_G07', 'q_g_G07', 'p_m_ref_G07', 'p_m_G07', 'v_pss_G07', 'i_d_G08', 'i_q_G08', 'p_g_G08', 'q_g_G08', 'p_m_ref_G08', 'p_m_G08', 'v_pss_G08', 'i_d_G09', 'i_q_G09', 'p_g_G09', 'q_g_G09', 'p_m_ref_G09', 'p_m_G09', 'v_pss_G09', 'i_d_G01', 'i_q_G01', 'p_g_G01', 'q_g_G01', 'p_m_ref_G01', 'p_m_G01', 'v_pss_G01', 'p_i_03', 'q_i_03', 'p_z_03', 'q_z_03', 'p_i_04', 'q_i_04', 'p_z_04', 'q_z_04', 'p_i_07', 'q_i_07', 'p_z_07', 'q_z_07', 'p_i_08', 'q_i_08', 'p_z_08', 'q_z_08', 'p_i_12', 'q_i_12', 'p_z_12', 'q_z_12', 'p_i_15', 'q_i_15', 'p_z_15', 'q_z_15', 'p_i_16', 'q_i_16', 'p_z_16', 'q_z_16', 'p_i_18', 'q_i_18', 'p_z_18', 'q_z_18', 'p_i_20', 'q_i_20', 'p_z_20', 'q_z_20', 'p_i_21', 'q_i_21', 'p_z_21', 'q_z_21', 'p_i_23', 'q_i_23', 'p_z_23', 'q_z_23', 'p_i_24', 'q_i_24', 'p_z_24', 'q_z_24', 'p_i_25', 'q_i_25', 'p_z_25', 'q_z_25', 'p_i_26', 'q_i_26', 'p_z_26', 'q_z_26', 'p_i_27', 'q_i_27', 'p_z_27', 'q_z_27', 'p_i_28', 'q_i_28', 'p_z_28', 'q_z_28', 'p_i_29', 'q_i_29', 'p_z_29', 'q_z_29', 'p_i_31', 'q_i_31', 'p_z_31', 'q_z_31', 'p_i_39', 'q_i_39', 'p_z_39', 'q_z_39', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'V_13', 'theta_13', 'V_14', 'theta_14', 'V_15', 'theta_15', 'V_16', 'theta_16', 'V_17', 'theta_17', 'V_18', 'theta_18', 'V_19', 'theta_19', 'V_20', 'theta_20', 'V_21', 'theta_21', 'V_22', 'theta_22', 'V_23', 'theta_23', 'V_24', 'theta_24', 'V_25', 'theta_25', 'V_26', 'theta_26', 'V_27', 'theta_27', 'V_28', 'theta_28', 'V_29', 'theta_29', 'V_30', 'theta_30', 'V_31', 'theta_31', 'V_32', 'theta_32', 'V_33', 'theta_33', 'V_34', 'theta_34', 'V_35', 'theta_35', 'V_36', 'theta_36', 'V_37', 'theta_37', 'V_38', 'theta_38', 'V_39', 'theta_39', 'i_d_G10', 'i_q_G10', 'p_g_G10', 'q_g_G10', 'p_m_ref_G10', 'p_m_G10', 'v_pss_G10', 'i_d_G02', 'i_q_G02', 'p_g_G02', 'q_g_G02', 'p_m_ref_G02', 'p_m_G02', 'v_pss_G02', 'i_d_G03', 'i_q_G03', 'p_g_G03', 'q_g_G03', 'p_m_ref_G03', 'p_m_G03', 'v_pss_G03', 'i_d_G04', 'i_q_G04', 'p_g_G04', 'q_g_G04', 'p_m_ref_G04', 'p_m_G04', 'v_pss_G04', 'i_d_G05', 'i_q_G05', 'p_g_G05', 'q_g_G05', 'p_m_ref_G05', 'p_m_G05', 'v_pss_G05', 'i_d_G06', 'i_q_G06', 'p_g_G06', 'q_g_G06', 'p_m_ref_G06', 'p_m_G06', 'v_pss_G06', 'i_d_G07', 'i_q_G07', 'p_g_G07', 'q_g_G07', 'p_m_ref_G07', 'p_m_G07', 'v_pss_G07', 'i_d_G08', 'i_q_G08', 'p_g_G08', 'q_g_G08', 'p_m_ref_G08', 'p_m_G08', 'v_pss_G08', 'i_d_G09', 'i_q_G09', 'p_g_G09', 'q_g_G09', 'p_m_ref_G09', 'p_m_G09', 'v_pss_G09', 'i_d_G01', 'i_q_G01', 'p_g_G01', 'q_g_G01', 'p_m_ref_G01', 'p_m_G01', 'v_pss_G01', 'i_p_03', 'i_q_03', 'g_load_03', 'b_load_03', 'i_p_04', 'i_q_04', 'g_load_04', 'b_load_04', 'i_p_07', 'i_q_07', 'g_load_07', 'b_load_07', 'i_p_08', 'i_q_08', 'g_load_08', 'b_load_08', 'i_p_12', 'i_q_12', 'g_load_12', 'b_load_12', 'i_p_15', 'i_q_15', 'g_load_15', 'b_load_15', 'i_p_16', 'i_q_16', 'g_load_16', 'b_load_16', 'i_p_18', 'i_q_18', 'g_load_18', 'b_load_18', 'i_p_20', 'i_q_20', 'g_load_20', 'b_load_20', 'i_p_21', 'i_q_21', 'g_load_21', 'b_load_21', 'i_p_23', 'i_q_23', 'g_load_23', 'b_load_23', 'i_p_24', 'i_q_24', 'g_load_24', 'b_load_24', 'i_p_25', 'i_q_25', 'g_load_25', 'b_load_25', 'i_p_26', 'i_q_26', 'g_load_26', 'b_load_26', 'i_p_27', 'i_q_27', 'g_load_27', 'b_load_27', 'i_p_28', 'i_q_28', 'g_load_28', 'b_load_28', 'i_p_29', 'i_q_29', 'g_load_29', 'b_load_29', 'i_p_31', 'i_q_31', 'g_load_31', 'b_load_31', 'i_p_39', 'i_q_39', 'g_load_39', 'b_load_39', 'omega_coi', 'p_agc'] 
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
            fobj = BytesIO(pkgutil.get_data(__name__, f'./newengland_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz(f'./{self.matrices_folder}/newengland_sp_jac_ini_num.npz')
            
            
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
            fobj = BytesIO(pkgutil.get_data(__name__, './newengland_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz(f'./{self.matrices_folder}/newengland_sp_jac_run_num.npz')
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
            fobj = BytesIO(pkgutil.get_data(__name__, './newengland_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz(f'./{self.matrices_folder}/newengland_sp_jac_trap_num.npz')
            

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

        #self.sp_Fu_run = sspa.load_npz(f'./{self.matrices_folder}/newengland_Fu_run_num.npz')
        #self.sp_Gu_run = sspa.load_npz(f'./{self.matrices_folder}/newengland_Gu_run_num.npz')
        #self.sp_Hx_run = sspa.load_npz(f'./{self.matrices_folder}/newengland_Hx_run_num.npz')
        #self.sp_Hy_run = sspa.load_npz(f'./{self.matrices_folder}/newengland_Hy_run_num.npz')
        #self.sp_Hu_run = sspa.load_npz(f'./{self.matrices_folder}/newengland_Hu_run_num.npz')        
        
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
        
        ## y_ini to u_run
        for item in self.yini2urun:
            self.u_run[self.u_run_list.index(item)] = self.y_ini[self.y_ini_list.index(item)]
                
        ## u_ini to y_run
        for item in self.uini2yrun:
            self.y_run[self.y_run_list.index(item)] = self.u_ini[self.u_ini_list.index(item)]
            
        
        self.x = self.xy_ini[:self.N_x]
        self.xy[:self.N_x] = self.x
        self.xy[self.N_x:] = self.y_run
        c_h_eval(self.z,self.x,self.y_run,self.u_run,self.p,self.Dt)
        

        
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

    sp_jac_ini_ia = [0, 1, 394, 0, 1, 228, 229, 248, 249, 253, 394, 2, 7, 248, 3, 249, 4, 228, 5, 6, 228, 254, 228, 5, 6, 7, 228, 254, 8, 252, 8, 9, 1, 10, 1, 10, 11, 1, 10, 11, 12, 13, 14, 394, 13, 14, 230, 231, 255, 256, 260, 394, 15, 20, 255, 16, 256, 17, 230, 18, 19, 230, 261, 230, 18, 19, 20, 230, 261, 21, 259, 21, 22, 14, 23, 14, 23, 24, 14, 23, 24, 25, 26, 27, 394, 26, 27, 232, 233, 262, 263, 267, 394, 28, 33, 262, 29, 263, 30, 232, 31, 32, 232, 268, 232, 31, 32, 33, 232, 268, 34, 266, 34, 35, 27, 36, 27, 36, 37, 27, 36, 37, 38, 39, 40, 394, 39, 40, 234, 235, 269, 270, 274, 394, 41, 46, 269, 42, 270, 43, 234, 44, 45, 234, 275, 234, 44, 45, 46, 234, 275, 47, 273, 47, 48, 40, 49, 40, 49, 50, 40, 49, 50, 51, 52, 53, 394, 52, 53, 236, 237, 276, 277, 281, 394, 54, 59, 276, 55, 277, 56, 236, 57, 58, 236, 282, 236, 57, 58, 59, 236, 282, 60, 280, 60, 61, 53, 62, 53, 62, 63, 53, 62, 63, 64, 65, 66, 394, 65, 66, 238, 239, 283, 284, 288, 394, 67, 72, 283, 68, 284, 69, 238, 70, 71, 238, 289, 238, 70, 71, 72, 238, 289, 73, 287, 73, 74, 66, 75, 66, 75, 76, 66, 75, 76, 77, 78, 79, 394, 78, 79, 240, 241, 290, 291, 295, 394, 80, 85, 290, 81, 291, 82, 240, 83, 84, 240, 296, 240, 83, 84, 85, 240, 296, 86, 294, 86, 87, 79, 88, 79, 88, 89, 79, 88, 89, 90, 91, 92, 394, 91, 92, 242, 243, 297, 298, 302, 394, 93, 98, 297, 94, 298, 95, 242, 96, 97, 242, 303, 242, 96, 97, 98, 242, 303, 99, 301, 99, 100, 92, 101, 92, 101, 102, 92, 101, 102, 103, 104, 105, 394, 104, 105, 244, 245, 304, 305, 309, 394, 106, 111, 304, 107, 305, 108, 244, 109, 110, 244, 310, 244, 109, 110, 111, 244, 310, 112, 308, 112, 113, 105, 114, 105, 114, 115, 105, 114, 115, 116, 117, 118, 394, 117, 118, 246, 247, 311, 312, 316, 394, 119, 124, 311, 120, 312, 121, 246, 122, 123, 246, 317, 246, 122, 123, 124, 246, 317, 125, 315, 125, 126, 118, 127, 118, 127, 128, 118, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 394, 170, 171, 172, 173, 246, 247, 170, 171, 172, 173, 246, 247, 170, 171, 172, 173, 174, 175, 218, 219, 228, 229, 170, 171, 172, 173, 174, 175, 218, 219, 228, 229, 130, 172, 173, 174, 175, 176, 177, 204, 205, 131, 172, 173, 174, 175, 176, 177, 204, 205, 132, 174, 175, 176, 177, 178, 179, 196, 197, 133, 174, 175, 176, 177, 178, 179, 196, 197, 176, 177, 178, 179, 180, 181, 184, 185, 176, 177, 178, 179, 180, 181, 184, 185, 178, 179, 180, 181, 182, 183, 190, 191, 230, 231, 178, 179, 180, 181, 182, 183, 190, 191, 230, 231, 134, 180, 181, 182, 183, 184, 185, 135, 180, 181, 182, 183, 184, 185, 136, 178, 179, 182, 183, 184, 185, 186, 187, 137, 178, 179, 182, 183, 184, 185, 186, 187, 184, 185, 186, 187, 246, 247, 184, 185, 186, 187, 246, 247, 188, 189, 190, 191, 194, 195, 232, 233, 188, 189, 190, 191, 194, 195, 232, 233, 180, 181, 188, 189, 190, 191, 192, 193, 180, 181, 188, 189, 190, 191, 192, 193, 138, 190, 191, 192, 193, 194, 195, 139, 190, 191, 192, 193, 194, 195, 188, 189, 192, 193, 194, 195, 196, 197, 188, 189, 192, 193, 194, 195, 196, 197, 176, 177, 194, 195, 196, 197, 198, 199, 176, 177, 194, 195, 196, 197, 198, 199, 140, 196, 197, 198, 199, 200, 201, 141, 196, 197, 198, 199, 200, 201, 142, 168, 198, 199, 200, 201, 202, 203, 206, 207, 210, 211, 216, 217, 143, 198, 199, 200, 201, 202, 203, 206, 207, 210, 211, 216, 217, 200, 201, 202, 203, 204, 205, 222, 223, 200, 201, 202, 203, 204, 205, 222, 223, 144, 174, 175, 202, 203, 204, 205, 145, 174, 175, 202, 203, 204, 205, 200, 201, 206, 207, 208, 209, 234, 235, 200, 201, 206, 207, 208, 209, 234, 235, 146, 206, 207, 208, 209, 236, 237, 147, 206, 207, 208, 209, 236, 237, 148, 200, 201, 210, 211, 212, 213, 149, 200, 201, 210, 211, 212, 213, 210, 211, 212, 213, 214, 215, 238, 239, 210, 211, 212, 213, 214, 215, 238, 239, 150, 212, 213, 214, 215, 216, 217, 240, 241, 151, 212, 213, 214, 215, 216, 217, 240, 241, 152, 200, 201, 214, 215, 216, 217, 153, 200, 201, 214, 215, 216, 217, 154, 172, 173, 218, 219, 220, 221, 242, 243, 155, 172, 173, 218, 219, 220, 221, 242, 243, 156, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 157, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 158, 202, 203, 220, 221, 222, 223, 159, 202, 203, 220, 221, 222, 223, 160, 220, 221, 224, 225, 226, 227, 161, 220, 221, 224, 225, 226, 227, 162, 220, 221, 224, 225, 226, 227, 244, 245, 163, 220, 221, 224, 225, 226, 227, 244, 245, 172, 173, 228, 229, 250, 172, 173, 228, 229, 251, 164, 180, 181, 230, 231, 257, 165, 180, 181, 230, 231, 258, 188, 189, 232, 233, 264, 188, 189, 232, 233, 265, 206, 207, 234, 235, 271, 206, 207, 234, 235, 272, 208, 209, 236, 237, 278, 208, 209, 236, 237, 279, 212, 213, 238, 239, 285, 212, 213, 238, 239, 286, 214, 215, 240, 241, 292, 214, 215, 240, 241, 293, 218, 219, 242, 243, 299, 218, 219, 242, 243, 300, 226, 227, 244, 245, 306, 226, 227, 244, 245, 307, 166, 170, 171, 186, 187, 246, 247, 313, 167, 170, 171, 186, 187, 246, 247, 314, 0, 2, 228, 229, 248, 249, 0, 3, 228, 229, 248, 249, 0, 228, 229, 248, 249, 250, 0, 228, 229, 248, 249, 251, 1, 252, 395, 1, 8, 9, 253, 1, 10, 11, 12, 254, 13, 15, 230, 231, 255, 256, 13, 16, 230, 231, 255, 256, 13, 230, 231, 255, 256, 257, 13, 230, 231, 255, 256, 258, 14, 259, 395, 14, 21, 22, 260, 14, 23, 24, 25, 261, 26, 28, 232, 233, 262, 263, 26, 29, 232, 233, 262, 263, 26, 232, 233, 262, 263, 264, 26, 232, 233, 262, 263, 265, 27, 266, 395, 27, 34, 35, 267, 27, 36, 37, 38, 268, 39, 41, 234, 235, 269, 270, 39, 42, 234, 235, 269, 270, 39, 234, 235, 269, 270, 271, 39, 234, 235, 269, 270, 272, 40, 273, 395, 40, 47, 48, 274, 40, 49, 50, 51, 275, 52, 54, 236, 237, 276, 277, 52, 55, 236, 237, 276, 277, 52, 236, 237, 276, 277, 278, 52, 236, 237, 276, 277, 279, 53, 280, 395, 53, 60, 61, 281, 53, 62, 63, 64, 282, 65, 67, 238, 239, 283, 284, 65, 68, 238, 239, 283, 284, 65, 238, 239, 283, 284, 285, 65, 238, 239, 283, 284, 286, 66, 287, 395, 66, 73, 74, 288, 66, 75, 76, 77, 289, 78, 80, 240, 241, 290, 291, 78, 81, 240, 241, 290, 291, 78, 240, 241, 290, 291, 292, 78, 240, 241, 290, 291, 293, 79, 294, 395, 79, 86, 87, 295, 79, 88, 89, 90, 296, 91, 93, 242, 243, 297, 298, 91, 94, 242, 243, 297, 298, 91, 242, 243, 297, 298, 299, 91, 242, 243, 297, 298, 300, 92, 301, 395, 92, 99, 100, 302, 92, 101, 102, 103, 303, 104, 106, 244, 245, 304, 305, 104, 107, 244, 245, 304, 305, 104, 244, 245, 304, 305, 306, 104, 244, 245, 304, 305, 307, 105, 308, 395, 105, 112, 113, 309, 105, 114, 115, 116, 310, 117, 119, 246, 247, 311, 312, 117, 120, 246, 247, 311, 312, 117, 246, 247, 311, 312, 313, 117, 246, 247, 311, 312, 314, 118, 315, 395, 118, 125, 126, 316, 118, 127, 128, 129, 317, 174, 318, 174, 319, 174, 320, 174, 321, 176, 322, 176, 323, 176, 324, 176, 325, 182, 326, 182, 327, 182, 328, 182, 329, 184, 330, 184, 331, 184, 332, 184, 333, 192, 334, 192, 335, 192, 336, 192, 337, 198, 338, 198, 339, 198, 340, 198, 341, 200, 342, 200, 343, 200, 344, 200, 345, 204, 346, 204, 347, 204, 348, 204, 349, 208, 350, 208, 351, 208, 352, 208, 353, 210, 354, 210, 355, 210, 356, 210, 357, 214, 358, 214, 359, 214, 360, 214, 361, 216, 362, 216, 363, 216, 364, 216, 365, 218, 366, 218, 367, 218, 368, 218, 369, 220, 370, 220, 371, 220, 372, 220, 373, 222, 374, 222, 375, 222, 376, 222, 377, 224, 378, 224, 379, 224, 380, 224, 381, 226, 382, 226, 383, 226, 384, 226, 385, 230, 386, 230, 387, 230, 388, 230, 389, 246, 390, 246, 391, 246, 392, 246, 393, 1, 14, 27, 40, 53, 66, 79, 92, 105, 118, 394, 169, 394, 395]
    sp_jac_ini_ja = [0, 3, 11, 14, 16, 18, 22, 23, 28, 30, 32, 34, 37, 41, 44, 52, 55, 57, 59, 63, 64, 69, 71, 73, 75, 78, 82, 85, 93, 96, 98, 100, 104, 105, 110, 112, 114, 116, 119, 123, 126, 134, 137, 139, 141, 145, 146, 151, 153, 155, 157, 160, 164, 167, 175, 178, 180, 182, 186, 187, 192, 194, 196, 198, 201, 205, 208, 216, 219, 221, 223, 227, 228, 233, 235, 237, 239, 242, 246, 249, 257, 260, 262, 264, 268, 269, 274, 276, 278, 280, 283, 287, 290, 298, 301, 303, 305, 309, 310, 315, 317, 319, 321, 324, 328, 331, 339, 342, 344, 346, 350, 351, 356, 358, 360, 362, 365, 369, 372, 380, 383, 385, 387, 391, 392, 397, 399, 401, 403, 406, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 451, 457, 463, 473, 483, 492, 501, 510, 519, 527, 535, 545, 555, 562, 569, 578, 587, 593, 599, 607, 615, 623, 631, 638, 645, 653, 661, 669, 677, 684, 691, 705, 718, 726, 734, 741, 748, 756, 764, 771, 778, 785, 792, 800, 808, 817, 826, 833, 840, 849, 858, 869, 880, 887, 894, 901, 908, 917, 926, 931, 936, 942, 948, 953, 958, 963, 968, 973, 978, 983, 988, 993, 998, 1003, 1008, 1013, 1018, 1026, 1034, 1040, 1046, 1052, 1058, 1061, 1065, 1070, 1076, 1082, 1088, 1094, 1097, 1101, 1106, 1112, 1118, 1124, 1130, 1133, 1137, 1142, 1148, 1154, 1160, 1166, 1169, 1173, 1178, 1184, 1190, 1196, 1202, 1205, 1209, 1214, 1220, 1226, 1232, 1238, 1241, 1245, 1250, 1256, 1262, 1268, 1274, 1277, 1281, 1286, 1292, 1298, 1304, 1310, 1313, 1317, 1322, 1328, 1334, 1340, 1346, 1349, 1353, 1358, 1364, 1370, 1376, 1382, 1385, 1389, 1394, 1396, 1398, 1400, 1402, 1404, 1406, 1408, 1410, 1412, 1414, 1416, 1418, 1420, 1422, 1424, 1426, 1428, 1430, 1432, 1434, 1436, 1438, 1440, 1442, 1444, 1446, 1448, 1450, 1452, 1454, 1456, 1458, 1460, 1462, 1464, 1466, 1468, 1470, 1472, 1474, 1476, 1478, 1480, 1482, 1484, 1486, 1488, 1490, 1492, 1494, 1496, 1498, 1500, 1502, 1504, 1506, 1508, 1510, 1512, 1514, 1516, 1518, 1520, 1522, 1524, 1526, 1528, 1530, 1532, 1534, 1536, 1538, 1540, 1542, 1544, 1546, 1557, 1560]
    sp_jac_ini_nia = 396
    sp_jac_ini_nja = 396
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 1, 394, 0, 1, 228, 229, 248, 249, 253, 394, 2, 7, 248, 3, 249, 4, 228, 5, 6, 228, 254, 228, 5, 6, 7, 228, 254, 8, 252, 8, 9, 1, 10, 1, 10, 11, 1, 10, 11, 12, 13, 14, 394, 13, 14, 230, 231, 255, 256, 260, 394, 15, 20, 255, 16, 256, 17, 230, 18, 19, 230, 261, 230, 18, 19, 20, 230, 261, 21, 259, 21, 22, 14, 23, 14, 23, 24, 14, 23, 24, 25, 26, 27, 394, 26, 27, 232, 233, 262, 263, 267, 394, 28, 33, 262, 29, 263, 30, 232, 31, 32, 232, 268, 232, 31, 32, 33, 232, 268, 34, 266, 34, 35, 27, 36, 27, 36, 37, 27, 36, 37, 38, 39, 40, 394, 39, 40, 234, 235, 269, 270, 274, 394, 41, 46, 269, 42, 270, 43, 234, 44, 45, 234, 275, 234, 44, 45, 46, 234, 275, 47, 273, 47, 48, 40, 49, 40, 49, 50, 40, 49, 50, 51, 52, 53, 394, 52, 53, 236, 237, 276, 277, 281, 394, 54, 59, 276, 55, 277, 56, 236, 57, 58, 236, 282, 236, 57, 58, 59, 236, 282, 60, 280, 60, 61, 53, 62, 53, 62, 63, 53, 62, 63, 64, 65, 66, 394, 65, 66, 238, 239, 283, 284, 288, 394, 67, 72, 283, 68, 284, 69, 238, 70, 71, 238, 289, 238, 70, 71, 72, 238, 289, 73, 287, 73, 74, 66, 75, 66, 75, 76, 66, 75, 76, 77, 78, 79, 394, 78, 79, 240, 241, 290, 291, 295, 394, 80, 85, 290, 81, 291, 82, 240, 83, 84, 240, 296, 240, 83, 84, 85, 240, 296, 86, 294, 86, 87, 79, 88, 79, 88, 89, 79, 88, 89, 90, 91, 92, 394, 91, 92, 242, 243, 297, 298, 302, 394, 93, 98, 297, 94, 298, 95, 242, 96, 97, 242, 303, 242, 96, 97, 98, 242, 303, 99, 301, 99, 100, 92, 101, 92, 101, 102, 92, 101, 102, 103, 104, 105, 394, 104, 105, 244, 245, 304, 305, 309, 394, 106, 111, 304, 107, 305, 108, 244, 109, 110, 244, 310, 244, 109, 110, 111, 244, 310, 112, 308, 112, 113, 105, 114, 105, 114, 115, 105, 114, 115, 116, 117, 118, 394, 117, 118, 246, 247, 311, 312, 316, 394, 119, 124, 311, 120, 312, 121, 246, 122, 123, 246, 317, 246, 122, 123, 124, 246, 317, 125, 315, 125, 126, 118, 127, 118, 127, 128, 118, 127, 128, 129, 130, 320, 131, 321, 132, 324, 133, 325, 134, 328, 135, 329, 136, 332, 137, 333, 138, 336, 139, 337, 140, 340, 141, 341, 142, 344, 143, 345, 144, 348, 145, 349, 146, 352, 147, 353, 148, 356, 149, 357, 150, 360, 151, 361, 152, 364, 153, 365, 154, 368, 155, 369, 156, 372, 157, 373, 158, 376, 159, 377, 160, 380, 161, 381, 162, 384, 163, 385, 164, 388, 165, 389, 166, 392, 167, 393, 168, 169, 394, 170, 171, 172, 173, 246, 247, 170, 171, 172, 173, 246, 247, 170, 171, 172, 173, 174, 175, 218, 219, 228, 229, 170, 171, 172, 173, 174, 175, 218, 219, 228, 229, 130, 172, 173, 174, 175, 176, 177, 204, 205, 318, 131, 172, 173, 174, 175, 176, 177, 204, 205, 319, 132, 174, 175, 176, 177, 178, 179, 196, 197, 322, 133, 174, 175, 176, 177, 178, 179, 196, 197, 323, 176, 177, 178, 179, 180, 181, 184, 185, 176, 177, 178, 179, 180, 181, 184, 185, 178, 179, 180, 181, 182, 183, 190, 191, 230, 231, 178, 179, 180, 181, 182, 183, 190, 191, 230, 231, 134, 180, 181, 182, 183, 184, 185, 326, 135, 180, 181, 182, 183, 184, 185, 327, 136, 178, 179, 182, 183, 184, 185, 186, 187, 330, 137, 178, 179, 182, 183, 184, 185, 186, 187, 331, 184, 185, 186, 187, 246, 247, 184, 185, 186, 187, 246, 247, 188, 189, 190, 191, 194, 195, 232, 233, 188, 189, 190, 191, 194, 195, 232, 233, 180, 181, 188, 189, 190, 191, 192, 193, 180, 181, 188, 189, 190, 191, 192, 193, 138, 190, 191, 192, 193, 194, 195, 334, 139, 190, 191, 192, 193, 194, 195, 335, 188, 189, 192, 193, 194, 195, 196, 197, 188, 189, 192, 193, 194, 195, 196, 197, 176, 177, 194, 195, 196, 197, 198, 199, 176, 177, 194, 195, 196, 197, 198, 199, 140, 196, 197, 198, 199, 200, 201, 338, 141, 196, 197, 198, 199, 200, 201, 339, 142, 168, 198, 199, 200, 201, 202, 203, 206, 207, 210, 211, 216, 217, 342, 143, 198, 199, 200, 201, 202, 203, 206, 207, 210, 211, 216, 217, 343, 200, 201, 202, 203, 204, 205, 222, 223, 200, 201, 202, 203, 204, 205, 222, 223, 144, 174, 175, 202, 203, 204, 205, 346, 145, 174, 175, 202, 203, 204, 205, 347, 200, 201, 206, 207, 208, 209, 234, 235, 200, 201, 206, 207, 208, 209, 234, 235, 146, 206, 207, 208, 209, 236, 237, 350, 147, 206, 207, 208, 209, 236, 237, 351, 148, 200, 201, 210, 211, 212, 213, 354, 149, 200, 201, 210, 211, 212, 213, 355, 210, 211, 212, 213, 214, 215, 238, 239, 210, 211, 212, 213, 214, 215, 238, 239, 150, 212, 213, 214, 215, 216, 217, 240, 241, 358, 151, 212, 213, 214, 215, 216, 217, 240, 241, 359, 152, 200, 201, 214, 215, 216, 217, 362, 153, 200, 201, 214, 215, 216, 217, 363, 154, 172, 173, 218, 219, 220, 221, 242, 243, 366, 155, 172, 173, 218, 219, 220, 221, 242, 243, 367, 156, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 370, 157, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 371, 158, 202, 203, 220, 221, 222, 223, 374, 159, 202, 203, 220, 221, 222, 223, 375, 160, 220, 221, 224, 225, 226, 227, 378, 161, 220, 221, 224, 225, 226, 227, 379, 162, 220, 221, 224, 225, 226, 227, 244, 245, 382, 163, 220, 221, 224, 225, 226, 227, 244, 245, 383, 172, 173, 228, 229, 250, 172, 173, 228, 229, 251, 164, 180, 181, 230, 231, 257, 386, 165, 180, 181, 230, 231, 258, 387, 188, 189, 232, 233, 264, 188, 189, 232, 233, 265, 206, 207, 234, 235, 271, 206, 207, 234, 235, 272, 208, 209, 236, 237, 278, 208, 209, 236, 237, 279, 212, 213, 238, 239, 285, 212, 213, 238, 239, 286, 214, 215, 240, 241, 292, 214, 215, 240, 241, 293, 218, 219, 242, 243, 299, 218, 219, 242, 243, 300, 226, 227, 244, 245, 306, 226, 227, 244, 245, 307, 166, 170, 171, 186, 187, 246, 247, 313, 390, 167, 170, 171, 186, 187, 246, 247, 314, 391, 0, 2, 228, 229, 248, 249, 0, 3, 228, 229, 248, 249, 0, 228, 229, 248, 249, 250, 0, 228, 229, 248, 249, 251, 1, 252, 395, 1, 8, 9, 253, 1, 10, 11, 12, 254, 13, 15, 230, 231, 255, 256, 13, 16, 230, 231, 255, 256, 13, 230, 231, 255, 256, 257, 13, 230, 231, 255, 256, 258, 14, 259, 395, 14, 21, 22, 260, 14, 23, 24, 25, 261, 26, 28, 232, 233, 262, 263, 26, 29, 232, 233, 262, 263, 26, 232, 233, 262, 263, 264, 26, 232, 233, 262, 263, 265, 27, 266, 395, 27, 34, 35, 267, 27, 36, 37, 38, 268, 39, 41, 234, 235, 269, 270, 39, 42, 234, 235, 269, 270, 39, 234, 235, 269, 270, 271, 39, 234, 235, 269, 270, 272, 40, 273, 395, 40, 47, 48, 274, 40, 49, 50, 51, 275, 52, 54, 236, 237, 276, 277, 52, 55, 236, 237, 276, 277, 52, 236, 237, 276, 277, 278, 52, 236, 237, 276, 277, 279, 53, 280, 395, 53, 60, 61, 281, 53, 62, 63, 64, 282, 65, 67, 238, 239, 283, 284, 65, 68, 238, 239, 283, 284, 65, 238, 239, 283, 284, 285, 65, 238, 239, 283, 284, 286, 66, 287, 395, 66, 73, 74, 288, 66, 75, 76, 77, 289, 78, 80, 240, 241, 290, 291, 78, 81, 240, 241, 290, 291, 78, 240, 241, 290, 291, 292, 78, 240, 241, 290, 291, 293, 79, 294, 395, 79, 86, 87, 295, 79, 88, 89, 90, 296, 91, 93, 242, 243, 297, 298, 91, 94, 242, 243, 297, 298, 91, 242, 243, 297, 298, 299, 91, 242, 243, 297, 298, 300, 92, 301, 395, 92, 99, 100, 302, 92, 101, 102, 103, 303, 104, 106, 244, 245, 304, 305, 104, 107, 244, 245, 304, 305, 104, 244, 245, 304, 305, 306, 104, 244, 245, 304, 305, 307, 105, 308, 395, 105, 112, 113, 309, 105, 114, 115, 116, 310, 117, 119, 246, 247, 311, 312, 117, 120, 246, 247, 311, 312, 117, 246, 247, 311, 312, 313, 117, 246, 247, 311, 312, 314, 118, 315, 395, 118, 125, 126, 316, 118, 127, 128, 129, 317, 174, 318, 174, 319, 174, 320, 174, 321, 176, 322, 176, 323, 176, 324, 176, 325, 182, 326, 182, 327, 182, 328, 182, 329, 184, 330, 184, 331, 184, 332, 184, 333, 192, 334, 192, 335, 192, 336, 192, 337, 198, 338, 198, 339, 198, 340, 198, 341, 200, 342, 200, 343, 200, 344, 200, 345, 204, 346, 204, 347, 204, 348, 204, 349, 208, 350, 208, 351, 208, 352, 208, 353, 210, 354, 210, 355, 210, 356, 210, 357, 214, 358, 214, 359, 214, 360, 214, 361, 216, 362, 216, 363, 216, 364, 216, 365, 218, 366, 218, 367, 218, 368, 218, 369, 220, 370, 220, 371, 220, 372, 220, 373, 222, 374, 222, 375, 222, 376, 222, 377, 224, 378, 224, 379, 224, 380, 224, 381, 226, 382, 226, 383, 226, 384, 226, 385, 230, 386, 230, 387, 230, 388, 230, 389, 246, 390, 246, 391, 246, 392, 246, 393, 1, 14, 27, 40, 53, 66, 79, 92, 105, 118, 394, 169, 394, 395]
    sp_jac_run_ja = [0, 3, 11, 14, 16, 18, 22, 23, 28, 30, 32, 34, 37, 41, 44, 52, 55, 57, 59, 63, 64, 69, 71, 73, 75, 78, 82, 85, 93, 96, 98, 100, 104, 105, 110, 112, 114, 116, 119, 123, 126, 134, 137, 139, 141, 145, 146, 151, 153, 155, 157, 160, 164, 167, 175, 178, 180, 182, 186, 187, 192, 194, 196, 198, 201, 205, 208, 216, 219, 221, 223, 227, 228, 233, 235, 237, 239, 242, 246, 249, 257, 260, 262, 264, 268, 269, 274, 276, 278, 280, 283, 287, 290, 298, 301, 303, 305, 309, 310, 315, 317, 319, 321, 324, 328, 331, 339, 342, 344, 346, 350, 351, 356, 358, 360, 362, 365, 369, 372, 380, 383, 385, 387, 391, 392, 397, 399, 401, 403, 406, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 487, 489, 495, 501, 511, 521, 531, 541, 551, 561, 569, 577, 587, 597, 605, 613, 623, 633, 639, 645, 653, 661, 669, 677, 685, 693, 701, 709, 717, 725, 733, 741, 756, 770, 778, 786, 794, 802, 810, 818, 826, 834, 842, 850, 858, 866, 876, 886, 894, 902, 912, 922, 934, 946, 954, 962, 970, 978, 988, 998, 1003, 1008, 1015, 1022, 1027, 1032, 1037, 1042, 1047, 1052, 1057, 1062, 1067, 1072, 1077, 1082, 1087, 1092, 1101, 1110, 1116, 1122, 1128, 1134, 1137, 1141, 1146, 1152, 1158, 1164, 1170, 1173, 1177, 1182, 1188, 1194, 1200, 1206, 1209, 1213, 1218, 1224, 1230, 1236, 1242, 1245, 1249, 1254, 1260, 1266, 1272, 1278, 1281, 1285, 1290, 1296, 1302, 1308, 1314, 1317, 1321, 1326, 1332, 1338, 1344, 1350, 1353, 1357, 1362, 1368, 1374, 1380, 1386, 1389, 1393, 1398, 1404, 1410, 1416, 1422, 1425, 1429, 1434, 1440, 1446, 1452, 1458, 1461, 1465, 1470, 1472, 1474, 1476, 1478, 1480, 1482, 1484, 1486, 1488, 1490, 1492, 1494, 1496, 1498, 1500, 1502, 1504, 1506, 1508, 1510, 1512, 1514, 1516, 1518, 1520, 1522, 1524, 1526, 1528, 1530, 1532, 1534, 1536, 1538, 1540, 1542, 1544, 1546, 1548, 1550, 1552, 1554, 1556, 1558, 1560, 1562, 1564, 1566, 1568, 1570, 1572, 1574, 1576, 1578, 1580, 1582, 1584, 1586, 1588, 1590, 1592, 1594, 1596, 1598, 1600, 1602, 1604, 1606, 1608, 1610, 1612, 1614, 1616, 1618, 1620, 1622, 1633, 1636]
    sp_jac_run_nia = 396
    sp_jac_run_nja = 396
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 394, 0, 1, 228, 229, 248, 249, 253, 394, 2, 7, 248, 3, 249, 4, 228, 5, 6, 228, 254, 6, 228, 5, 6, 7, 228, 254, 8, 252, 8, 9, 1, 10, 1, 10, 11, 1, 10, 11, 12, 13, 14, 394, 13, 14, 230, 231, 255, 256, 260, 394, 15, 20, 255, 16, 256, 17, 230, 18, 19, 230, 261, 19, 230, 18, 19, 20, 230, 261, 21, 259, 21, 22, 14, 23, 14, 23, 24, 14, 23, 24, 25, 26, 27, 394, 26, 27, 232, 233, 262, 263, 267, 394, 28, 33, 262, 29, 263, 30, 232, 31, 32, 232, 268, 32, 232, 31, 32, 33, 232, 268, 34, 266, 34, 35, 27, 36, 27, 36, 37, 27, 36, 37, 38, 39, 40, 394, 39, 40, 234, 235, 269, 270, 274, 394, 41, 46, 269, 42, 270, 43, 234, 44, 45, 234, 275, 45, 234, 44, 45, 46, 234, 275, 47, 273, 47, 48, 40, 49, 40, 49, 50, 40, 49, 50, 51, 52, 53, 394, 52, 53, 236, 237, 276, 277, 281, 394, 54, 59, 276, 55, 277, 56, 236, 57, 58, 236, 282, 58, 236, 57, 58, 59, 236, 282, 60, 280, 60, 61, 53, 62, 53, 62, 63, 53, 62, 63, 64, 65, 66, 394, 65, 66, 238, 239, 283, 284, 288, 394, 67, 72, 283, 68, 284, 69, 238, 70, 71, 238, 289, 71, 238, 70, 71, 72, 238, 289, 73, 287, 73, 74, 66, 75, 66, 75, 76, 66, 75, 76, 77, 78, 79, 394, 78, 79, 240, 241, 290, 291, 295, 394, 80, 85, 290, 81, 291, 82, 240, 83, 84, 240, 296, 84, 240, 83, 84, 85, 240, 296, 86, 294, 86, 87, 79, 88, 79, 88, 89, 79, 88, 89, 90, 91, 92, 394, 91, 92, 242, 243, 297, 298, 302, 394, 93, 98, 297, 94, 298, 95, 242, 96, 97, 242, 303, 97, 242, 96, 97, 98, 242, 303, 99, 301, 99, 100, 92, 101, 92, 101, 102, 92, 101, 102, 103, 104, 105, 394, 104, 105, 244, 245, 304, 305, 309, 394, 106, 111, 304, 107, 305, 108, 244, 109, 110, 244, 310, 110, 244, 109, 110, 111, 244, 310, 112, 308, 112, 113, 105, 114, 105, 114, 115, 105, 114, 115, 116, 117, 118, 394, 117, 118, 246, 247, 311, 312, 316, 394, 119, 124, 311, 120, 312, 121, 246, 122, 123, 246, 317, 123, 246, 122, 123, 124, 246, 317, 125, 315, 125, 126, 118, 127, 118, 127, 128, 118, 127, 128, 129, 130, 320, 131, 321, 132, 324, 133, 325, 134, 328, 135, 329, 136, 332, 137, 333, 138, 336, 139, 337, 140, 340, 141, 341, 142, 344, 143, 345, 144, 348, 145, 349, 146, 352, 147, 353, 148, 356, 149, 357, 150, 360, 151, 361, 152, 364, 153, 365, 154, 368, 155, 369, 156, 372, 157, 373, 158, 376, 159, 377, 160, 380, 161, 381, 162, 384, 163, 385, 164, 388, 165, 389, 166, 392, 167, 393, 168, 169, 394, 170, 171, 172, 173, 246, 247, 170, 171, 172, 173, 246, 247, 170, 171, 172, 173, 174, 175, 218, 219, 228, 229, 170, 171, 172, 173, 174, 175, 218, 219, 228, 229, 130, 172, 173, 174, 175, 176, 177, 204, 205, 318, 131, 172, 173, 174, 175, 176, 177, 204, 205, 319, 132, 174, 175, 176, 177, 178, 179, 196, 197, 322, 133, 174, 175, 176, 177, 178, 179, 196, 197, 323, 176, 177, 178, 179, 180, 181, 184, 185, 176, 177, 178, 179, 180, 181, 184, 185, 178, 179, 180, 181, 182, 183, 190, 191, 230, 231, 178, 179, 180, 181, 182, 183, 190, 191, 230, 231, 134, 180, 181, 182, 183, 184, 185, 326, 135, 180, 181, 182, 183, 184, 185, 327, 136, 178, 179, 182, 183, 184, 185, 186, 187, 330, 137, 178, 179, 182, 183, 184, 185, 186, 187, 331, 184, 185, 186, 187, 246, 247, 184, 185, 186, 187, 246, 247, 188, 189, 190, 191, 194, 195, 232, 233, 188, 189, 190, 191, 194, 195, 232, 233, 180, 181, 188, 189, 190, 191, 192, 193, 180, 181, 188, 189, 190, 191, 192, 193, 138, 190, 191, 192, 193, 194, 195, 334, 139, 190, 191, 192, 193, 194, 195, 335, 188, 189, 192, 193, 194, 195, 196, 197, 188, 189, 192, 193, 194, 195, 196, 197, 176, 177, 194, 195, 196, 197, 198, 199, 176, 177, 194, 195, 196, 197, 198, 199, 140, 196, 197, 198, 199, 200, 201, 338, 141, 196, 197, 198, 199, 200, 201, 339, 142, 168, 198, 199, 200, 201, 202, 203, 206, 207, 210, 211, 216, 217, 342, 143, 198, 199, 200, 201, 202, 203, 206, 207, 210, 211, 216, 217, 343, 200, 201, 202, 203, 204, 205, 222, 223, 200, 201, 202, 203, 204, 205, 222, 223, 144, 174, 175, 202, 203, 204, 205, 346, 145, 174, 175, 202, 203, 204, 205, 347, 200, 201, 206, 207, 208, 209, 234, 235, 200, 201, 206, 207, 208, 209, 234, 235, 146, 206, 207, 208, 209, 236, 237, 350, 147, 206, 207, 208, 209, 236, 237, 351, 148, 200, 201, 210, 211, 212, 213, 354, 149, 200, 201, 210, 211, 212, 213, 355, 210, 211, 212, 213, 214, 215, 238, 239, 210, 211, 212, 213, 214, 215, 238, 239, 150, 212, 213, 214, 215, 216, 217, 240, 241, 358, 151, 212, 213, 214, 215, 216, 217, 240, 241, 359, 152, 200, 201, 214, 215, 216, 217, 362, 153, 200, 201, 214, 215, 216, 217, 363, 154, 172, 173, 218, 219, 220, 221, 242, 243, 366, 155, 172, 173, 218, 219, 220, 221, 242, 243, 367, 156, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 370, 157, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 371, 158, 202, 203, 220, 221, 222, 223, 374, 159, 202, 203, 220, 221, 222, 223, 375, 160, 220, 221, 224, 225, 226, 227, 378, 161, 220, 221, 224, 225, 226, 227, 379, 162, 220, 221, 224, 225, 226, 227, 244, 245, 382, 163, 220, 221, 224, 225, 226, 227, 244, 245, 383, 172, 173, 228, 229, 250, 172, 173, 228, 229, 251, 164, 180, 181, 230, 231, 257, 386, 165, 180, 181, 230, 231, 258, 387, 188, 189, 232, 233, 264, 188, 189, 232, 233, 265, 206, 207, 234, 235, 271, 206, 207, 234, 235, 272, 208, 209, 236, 237, 278, 208, 209, 236, 237, 279, 212, 213, 238, 239, 285, 212, 213, 238, 239, 286, 214, 215, 240, 241, 292, 214, 215, 240, 241, 293, 218, 219, 242, 243, 299, 218, 219, 242, 243, 300, 226, 227, 244, 245, 306, 226, 227, 244, 245, 307, 166, 170, 171, 186, 187, 246, 247, 313, 390, 167, 170, 171, 186, 187, 246, 247, 314, 391, 0, 2, 228, 229, 248, 249, 0, 3, 228, 229, 248, 249, 0, 228, 229, 248, 249, 250, 0, 228, 229, 248, 249, 251, 1, 252, 395, 1, 8, 9, 253, 1, 10, 11, 12, 254, 13, 15, 230, 231, 255, 256, 13, 16, 230, 231, 255, 256, 13, 230, 231, 255, 256, 257, 13, 230, 231, 255, 256, 258, 14, 259, 395, 14, 21, 22, 260, 14, 23, 24, 25, 261, 26, 28, 232, 233, 262, 263, 26, 29, 232, 233, 262, 263, 26, 232, 233, 262, 263, 264, 26, 232, 233, 262, 263, 265, 27, 266, 395, 27, 34, 35, 267, 27, 36, 37, 38, 268, 39, 41, 234, 235, 269, 270, 39, 42, 234, 235, 269, 270, 39, 234, 235, 269, 270, 271, 39, 234, 235, 269, 270, 272, 40, 273, 395, 40, 47, 48, 274, 40, 49, 50, 51, 275, 52, 54, 236, 237, 276, 277, 52, 55, 236, 237, 276, 277, 52, 236, 237, 276, 277, 278, 52, 236, 237, 276, 277, 279, 53, 280, 395, 53, 60, 61, 281, 53, 62, 63, 64, 282, 65, 67, 238, 239, 283, 284, 65, 68, 238, 239, 283, 284, 65, 238, 239, 283, 284, 285, 65, 238, 239, 283, 284, 286, 66, 287, 395, 66, 73, 74, 288, 66, 75, 76, 77, 289, 78, 80, 240, 241, 290, 291, 78, 81, 240, 241, 290, 291, 78, 240, 241, 290, 291, 292, 78, 240, 241, 290, 291, 293, 79, 294, 395, 79, 86, 87, 295, 79, 88, 89, 90, 296, 91, 93, 242, 243, 297, 298, 91, 94, 242, 243, 297, 298, 91, 242, 243, 297, 298, 299, 91, 242, 243, 297, 298, 300, 92, 301, 395, 92, 99, 100, 302, 92, 101, 102, 103, 303, 104, 106, 244, 245, 304, 305, 104, 107, 244, 245, 304, 305, 104, 244, 245, 304, 305, 306, 104, 244, 245, 304, 305, 307, 105, 308, 395, 105, 112, 113, 309, 105, 114, 115, 116, 310, 117, 119, 246, 247, 311, 312, 117, 120, 246, 247, 311, 312, 117, 246, 247, 311, 312, 313, 117, 246, 247, 311, 312, 314, 118, 315, 395, 118, 125, 126, 316, 118, 127, 128, 129, 317, 174, 318, 174, 319, 174, 320, 174, 321, 176, 322, 176, 323, 176, 324, 176, 325, 182, 326, 182, 327, 182, 328, 182, 329, 184, 330, 184, 331, 184, 332, 184, 333, 192, 334, 192, 335, 192, 336, 192, 337, 198, 338, 198, 339, 198, 340, 198, 341, 200, 342, 200, 343, 200, 344, 200, 345, 204, 346, 204, 347, 204, 348, 204, 349, 208, 350, 208, 351, 208, 352, 208, 353, 210, 354, 210, 355, 210, 356, 210, 357, 214, 358, 214, 359, 214, 360, 214, 361, 216, 362, 216, 363, 216, 364, 216, 365, 218, 366, 218, 367, 218, 368, 218, 369, 220, 370, 220, 371, 220, 372, 220, 373, 222, 374, 222, 375, 222, 376, 222, 377, 224, 378, 224, 379, 224, 380, 224, 381, 226, 382, 226, 383, 226, 384, 226, 385, 230, 386, 230, 387, 230, 388, 230, 389, 246, 390, 246, 391, 246, 392, 246, 393, 1, 14, 27, 40, 53, 66, 79, 92, 105, 118, 394, 169, 394, 395]
    sp_jac_trap_ja = [0, 3, 11, 14, 16, 18, 22, 24, 29, 31, 33, 35, 38, 42, 45, 53, 56, 58, 60, 64, 66, 71, 73, 75, 77, 80, 84, 87, 95, 98, 100, 102, 106, 108, 113, 115, 117, 119, 122, 126, 129, 137, 140, 142, 144, 148, 150, 155, 157, 159, 161, 164, 168, 171, 179, 182, 184, 186, 190, 192, 197, 199, 201, 203, 206, 210, 213, 221, 224, 226, 228, 232, 234, 239, 241, 243, 245, 248, 252, 255, 263, 266, 268, 270, 274, 276, 281, 283, 285, 287, 290, 294, 297, 305, 308, 310, 312, 316, 318, 323, 325, 327, 329, 332, 336, 339, 347, 350, 352, 354, 358, 360, 365, 367, 369, 371, 374, 378, 381, 389, 392, 394, 396, 400, 402, 407, 409, 411, 413, 416, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492, 494, 496, 497, 499, 505, 511, 521, 531, 541, 551, 561, 571, 579, 587, 597, 607, 615, 623, 633, 643, 649, 655, 663, 671, 679, 687, 695, 703, 711, 719, 727, 735, 743, 751, 766, 780, 788, 796, 804, 812, 820, 828, 836, 844, 852, 860, 868, 876, 886, 896, 904, 912, 922, 932, 944, 956, 964, 972, 980, 988, 998, 1008, 1013, 1018, 1025, 1032, 1037, 1042, 1047, 1052, 1057, 1062, 1067, 1072, 1077, 1082, 1087, 1092, 1097, 1102, 1111, 1120, 1126, 1132, 1138, 1144, 1147, 1151, 1156, 1162, 1168, 1174, 1180, 1183, 1187, 1192, 1198, 1204, 1210, 1216, 1219, 1223, 1228, 1234, 1240, 1246, 1252, 1255, 1259, 1264, 1270, 1276, 1282, 1288, 1291, 1295, 1300, 1306, 1312, 1318, 1324, 1327, 1331, 1336, 1342, 1348, 1354, 1360, 1363, 1367, 1372, 1378, 1384, 1390, 1396, 1399, 1403, 1408, 1414, 1420, 1426, 1432, 1435, 1439, 1444, 1450, 1456, 1462, 1468, 1471, 1475, 1480, 1482, 1484, 1486, 1488, 1490, 1492, 1494, 1496, 1498, 1500, 1502, 1504, 1506, 1508, 1510, 1512, 1514, 1516, 1518, 1520, 1522, 1524, 1526, 1528, 1530, 1532, 1534, 1536, 1538, 1540, 1542, 1544, 1546, 1548, 1550, 1552, 1554, 1556, 1558, 1560, 1562, 1564, 1566, 1568, 1570, 1572, 1574, 1576, 1578, 1580, 1582, 1584, 1586, 1588, 1590, 1592, 1594, 1596, 1598, 1600, 1602, 1604, 1606, 1608, 1610, 1612, 1614, 1616, 1618, 1620, 1622, 1624, 1626, 1628, 1630, 1632, 1643, 1646]
    sp_jac_trap_nia = 396
    sp_jac_trap_nja = 396
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
