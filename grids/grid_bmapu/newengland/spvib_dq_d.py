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
    import spvib_dq_d_ini_cffi as jacs_ini
    import spvib_dq_d_run_cffi as jacs_run
    import spvib_dq_d_trap_cffi as jacs_trap

if dae_file_mode == 'enviroment':
    import envus.no_enviroment.spvib_dq_d_cffi as jacs
if dae_file_mode == 'colab':
    import spvib_dq_d_cffi as jacs
    
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
        self.N_x = 111
        self.N_y = 170 
        self.N_z = 99 
        self.N_store = 100000 
        self.params_list = ['S_base', 'g_01_02', 'b_01_02', 'bs_01_02', 'g_01_39', 'b_01_39', 'bs_01_39', 'g_02_03', 'b_02_03', 'bs_02_03', 'g_02_25', 'b_02_25', 'bs_02_25', 'g_03_04', 'b_03_04', 'bs_03_04', 'g_03_18', 'b_03_18', 'bs_03_18', 'g_04_05', 'b_04_05', 'bs_04_05', 'g_04_14', 'b_04_14', 'bs_04_14', 'g_05_06', 'b_05_06', 'bs_05_06', 'g_05_08', 'b_05_08', 'bs_05_08', 'g_06_07', 'b_06_07', 'bs_06_07', 'g_06_11', 'b_06_11', 'bs_06_11', 'g_07_08', 'b_07_08', 'bs_07_08', 'g_08_09', 'b_08_09', 'bs_08_09', 'g_09_39', 'b_09_39', 'bs_09_39', 'g_10_11', 'b_10_11', 'bs_10_11', 'g_10_13', 'b_10_13', 'bs_10_13', 'g_13_14', 'b_13_14', 'bs_13_14', 'g_14_15', 'b_14_15', 'bs_14_15', 'g_15_16', 'b_15_16', 'bs_15_16', 'g_16_17', 'b_16_17', 'bs_16_17', 'g_16_19', 'b_16_19', 'bs_16_19', 'g_16_21', 'b_16_21', 'bs_16_21', 'g_16_24', 'b_16_24', 'bs_16_24', 'g_17_18', 'b_17_18', 'bs_17_18', 'g_17_27', 'b_17_27', 'bs_17_27', 'g_21_22', 'b_21_22', 'bs_21_22', 'g_22_23', 'b_22_23', 'bs_22_23', 'g_23_24', 'b_23_24', 'bs_23_24', 'g_25_26', 'b_25_26', 'bs_25_26', 'g_26_27', 'b_26_27', 'bs_26_27', 'g_26_28', 'b_26_28', 'bs_26_28', 'g_26_29', 'b_26_29', 'bs_26_29', 'g_28_29', 'b_28_29', 'bs_28_29', 'g_12_11', 'b_12_11', 'bs_12_11', 'g_12_13', 'b_12_13', 'bs_12_13', 'g_06_31', 'b_06_31', 'bs_06_31', 'g_10_32', 'b_10_32', 'bs_10_32', 'g_19_33', 'b_19_33', 'bs_19_33', 'g_20_34', 'b_20_34', 'bs_20_34', 'g_22_35', 'b_22_35', 'bs_22_35', 'g_23_36', 'b_23_36', 'bs_23_36', 'g_25_37', 'b_25_37', 'bs_25_37', 'g_02_30', 'b_02_30', 'bs_02_30', 'g_29_38', 'b_29_38', 'bs_29_38', 'g_19_20', 'b_19_20', 'bs_19_20', 'U_01_n', 'U_02_n', 'U_03_n', 'U_04_n', 'U_05_n', 'U_06_n', 'U_07_n', 'U_08_n', 'U_09_n', 'U_10_n', 'U_11_n', 'U_12_n', 'U_13_n', 'U_14_n', 'U_15_n', 'U_16_n', 'U_17_n', 'U_18_n', 'U_19_n', 'U_20_n', 'U_21_n', 'U_22_n', 'U_23_n', 'U_24_n', 'U_25_n', 'U_26_n', 'U_27_n', 'U_28_n', 'U_29_n', 'U_30_n', 'U_31_n', 'U_32_n', 'U_33_n', 'U_34_n', 'U_35_n', 'U_36_n', 'U_37_n', 'U_38_n', 'U_39_n', 'S_n_30', 'Omega_b_30', 'H_30', 'T1d0_30', 'T1q0_30', 'X_d_30', 'X_q_30', 'X1d_30', 'X1q_30', 'D_30', 'R_a_30', 'K_delta_30', 'K_sec_30', 'K_a_30', 'K_ai_30', 'T_a_30', 'T_b_30', 'T_e_30', 'E_min_30', 'E_max_30', 'Droop_30', 'T_gov_1_30', 'T_gov_2_30', 'T_gov_3_30', 'D_t_30', 'omega_ref_30', 'T_wo_30', 'T_1_30', 'T_2_30', 'K_stab_30', 'V_lim_30', 'S_n_31', 'Omega_b_31', 'H_31', 'T1d0_31', 'T1q0_31', 'X_d_31', 'X_q_31', 'X1d_31', 'X1q_31', 'D_31', 'R_a_31', 'K_delta_31', 'K_sec_31', 'K_a_31', 'K_ai_31', 'T_a_31', 'T_b_31', 'T_e_31', 'E_min_31', 'E_max_31', 'Droop_31', 'T_gov_1_31', 'T_gov_2_31', 'T_gov_3_31', 'D_t_31', 'omega_ref_31', 'T_wo_31', 'T_1_31', 'T_2_31', 'K_stab_31', 'V_lim_31', 'S_n_32', 'Omega_b_32', 'H_32', 'T1d0_32', 'T1q0_32', 'X_d_32', 'X_q_32', 'X1d_32', 'X1q_32', 'D_32', 'R_a_32', 'K_delta_32', 'K_sec_32', 'K_a_32', 'K_ai_32', 'T_a_32', 'T_b_32', 'T_e_32', 'E_min_32', 'E_max_32', 'Droop_32', 'T_gov_1_32', 'T_gov_2_32', 'T_gov_3_32', 'D_t_32', 'omega_ref_32', 'T_wo_32', 'T_1_32', 'T_2_32', 'K_stab_32', 'V_lim_32', 'S_n_33', 'Omega_b_33', 'H_33', 'T1d0_33', 'T1q0_33', 'X_d_33', 'X_q_33', 'X1d_33', 'X1q_33', 'D_33', 'R_a_33', 'K_delta_33', 'K_sec_33', 'K_a_33', 'K_ai_33', 'T_a_33', 'T_b_33', 'T_e_33', 'E_min_33', 'E_max_33', 'Droop_33', 'T_gov_1_33', 'T_gov_2_33', 'T_gov_3_33', 'D_t_33', 'omega_ref_33', 'T_wo_33', 'T_1_33', 'T_2_33', 'K_stab_33', 'V_lim_33', 'S_n_34', 'Omega_b_34', 'H_34', 'T1d0_34', 'T1q0_34', 'X_d_34', 'X_q_34', 'X1d_34', 'X1q_34', 'D_34', 'R_a_34', 'K_delta_34', 'K_sec_34', 'K_a_34', 'K_ai_34', 'T_a_34', 'T_b_34', 'T_e_34', 'E_min_34', 'E_max_34', 'Droop_34', 'T_gov_1_34', 'T_gov_2_34', 'T_gov_3_34', 'D_t_34', 'omega_ref_34', 'T_wo_34', 'T_1_34', 'T_2_34', 'K_stab_34', 'V_lim_34', 'S_n_35', 'Omega_b_35', 'H_35', 'T1d0_35', 'T1q0_35', 'X_d_35', 'X_q_35', 'X1d_35', 'X1q_35', 'D_35', 'R_a_35', 'K_delta_35', 'K_sec_35', 'K_a_35', 'K_ai_35', 'T_a_35', 'T_b_35', 'T_e_35', 'E_min_35', 'E_max_35', 'Droop_35', 'T_gov_1_35', 'T_gov_2_35', 'T_gov_3_35', 'D_t_35', 'omega_ref_35', 'T_wo_35', 'T_1_35', 'T_2_35', 'K_stab_35', 'V_lim_35', 'S_n_36', 'Omega_b_36', 'H_36', 'T1d0_36', 'T1q0_36', 'X_d_36', 'X_q_36', 'X1d_36', 'X1q_36', 'D_36', 'R_a_36', 'K_delta_36', 'K_sec_36', 'K_a_36', 'K_ai_36', 'T_a_36', 'T_b_36', 'T_e_36', 'E_min_36', 'E_max_36', 'Droop_36', 'T_gov_1_36', 'T_gov_2_36', 'T_gov_3_36', 'D_t_36', 'omega_ref_36', 'T_wo_36', 'T_1_36', 'T_2_36', 'K_stab_36', 'V_lim_36', 'S_n_37', 'Omega_b_37', 'H_37', 'T1d0_37', 'T1q0_37', 'X_d_37', 'X_q_37', 'X1d_37', 'X1q_37', 'D_37', 'R_a_37', 'K_delta_37', 'K_sec_37', 'K_a_37', 'K_ai_37', 'T_a_37', 'T_b_37', 'T_e_37', 'E_min_37', 'E_max_37', 'Droop_37', 'T_gov_1_37', 'T_gov_2_37', 'T_gov_3_37', 'D_t_37', 'omega_ref_37', 'T_wo_37', 'T_1_37', 'T_2_37', 'K_stab_37', 'V_lim_37', 'S_n_38', 'Omega_b_38', 'H_38', 'T1d0_38', 'T1q0_38', 'X_d_38', 'X_q_38', 'X1d_38', 'X1q_38', 'D_38', 'R_a_38', 'K_delta_38', 'K_sec_38', 'K_a_38', 'K_ai_38', 'T_a_38', 'T_b_38', 'T_e_38', 'E_min_38', 'E_max_38', 'Droop_38', 'T_gov_1_38', 'T_gov_2_38', 'T_gov_3_38', 'D_t_38', 'omega_ref_38', 'T_wo_38', 'T_1_38', 'T_2_38', 'K_stab_38', 'V_lim_38', 'S_n_39', 'Omega_b_39', 'H_39', 'T1d0_39', 'T1q0_39', 'X_d_39', 'X_q_39', 'X1d_39', 'X1q_39', 'D_39', 'R_a_39', 'K_delta_39', 'K_sec_39', 'K_a_39', 'K_ai_39', 'T_a_39', 'T_b_39', 'T_e_39', 'E_min_39', 'E_max_39', 'Droop_39', 'T_gov_1_39', 'T_gov_2_39', 'T_gov_3_39', 'D_t_39', 'omega_ref_39', 'T_wo_39', 'T_1_39', 'T_2_39', 'K_stab_39', 'V_lim_39', 'K_p_agc', 'K_i_agc', 'K_xif'] 
        self.params_values_list  = [100000000.0, 2.0570568805614005, -24.155725083163873, 0.0, 1.5974440894568687, -39.93610223642172, 0.0, 5.659555942533739, -65.73791902481499, 0.0, 56.92908262849707, -69.94144437215354, 0.0, 2.8547586630945583, -46.77412271070315, 0.0, 6.17630544637844, -74.6771476698484, 0.0, 4.863813229571985, -77.82101167315176, 0.0, 4.788985333732416, -77.2223885064352, 0.0, 29.41176470588236, -382.3529411764706, 0.0, 6.34517766497462, -88.83248730964468, 0.0, 7.058823529411764, -108.23529411764704, 0.0, 10.335154289089028, -121.06895024361432, 0.0, 18.761726078799253, -215.75984990619136, 0.0, 1.7384994482153926, -27.438056508790762, 0.0, 1.5974440894568687, -39.93610223642172, 0.0, 21.447721179624665, -230.56300268096516, 0.0, 21.447721179624665, -230.56300268096516, 0.0, 8.753160863645206, -98.22991635868509, 0.0, 3.7964271402358, -45.76803830173159, 0.0, 10.093080632499719, -105.41661993944152, 0.0, 8.782936010037641, -111.66875784190715, 0.0, 4.179619132206578, -50.93910817376767, 0.0, 4.3742140084203625, -73.81486139209362, 0.0, 8.595988538681947, -169.05444126074497, 0.0, 10.335154289089028, -121.06895024361432, 0.0, 4.319223868695595, -57.478902252641376, 0.0, 4.068348250610252, -71.19609438567942, 0.0, 6.485084306095979, -103.76134889753567, 0.0, 1.7888505821895528, -28.458986534833794, 0.0, 3.037407572636755, -30.658832686302244, 0.0, 6.420545746388443, -67.41573033707866, 0.0, 1.8982452267961594, -20.92484273259022, 0.0, 1.4471633060318785, -15.868018706489893, 0.0, 6.087750576162108, -65.66073835717702, 0.0, 0.8444118407650373, -22.95744692079945, 0.0, 0.8444118407650373, -22.95744692079945, 0.0, 0.0, -39.99999999999999, 0.0, 0.0, -50.0, 0.0, 3.463117795478157, -70.25181813684263, 0.0, 2.770850651149903, -55.41701302299806, 0.0, 0.0, -69.93006993006992, 0.0, 0.6755935088975665, -36.75228688402762, 0.0, 1.1139992573338284, -43.07463795024137, 0.0, 0.0, -55.248618784530386, 0.0, 3.2786885245901645, -63.934426229508205, 0.0, 3.666265123343634, -72.27779814591736, 0.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 900000000.0, 314.1592653589793, 6.5, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.001, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.5, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 1.0, 100.0, 1e-06, 0.1, 0.1, 0.1, -10.0, 10.0, 0.05, 1.0, 1.0, 1.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 0.01, 0.01, 0.0] 
        self.inputs_ini_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'P_13', 'Q_13', 'P_14', 'Q_14', 'P_15', 'Q_15', 'P_16', 'Q_16', 'P_17', 'Q_17', 'P_18', 'Q_18', 'P_19', 'Q_19', 'P_20', 'Q_20', 'P_21', 'Q_21', 'P_22', 'Q_22', 'P_23', 'Q_23', 'P_24', 'Q_24', 'P_25', 'Q_25', 'P_26', 'Q_26', 'P_27', 'Q_27', 'P_28', 'Q_28', 'P_29', 'Q_29', 'P_30', 'Q_30', 'P_31', 'Q_31', 'P_32', 'Q_32', 'P_33', 'Q_33', 'P_34', 'Q_34', 'P_35', 'Q_35', 'P_36', 'Q_36', 'P_37', 'Q_37', 'P_38', 'Q_38', 'P_39', 'Q_39', 'v_ref_30', 'p_c_30', 'v_ref_31', 'p_c_31', 'v_ref_32', 'p_c_32', 'v_ref_33', 'p_c_33', 'v_ref_34', 'p_c_34', 'v_ref_35', 'p_c_35', 'v_ref_36', 'p_c_36', 'v_ref_37', 'p_c_37', 'v_ref_38', 'p_c_38', 'v_ref_39', 'p_c_39'] 
        self.inputs_ini_values_list  = [0.0, -0.0, 0.0, -0.0, -322000000.0, -2400000.0, -500000000.0, -184000000.0, 0.0, -0.0, 0.0, -0.0, -233800000.0, -84000000.0, -522000000.0, -176000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7500000.0, -88000000.0, 0.0, 0.0, 0.0, 0.0, -320000000.0, -153000000.0, -329000000.0, -32300000.0, 0.0, 0.0, -158000000.0, -30000000.0, 0.0, 0.0, -628000000.0, -103000000.0, -274000000.0, -115000000.0, 0.0, 0.0, -247500000.0, -84600000.0, -308600000.0, 92000000.0, -224000000.0, -47200000.0, -139000000.0, -17000000.0, -281000000.0, -75500000.0, -206000000.0, -27600000.0, -283500000.0, -26900000.0, -47500.0, 0.0, -982000.0, -9200000.0, -983100.0, 0.0, -997200.0, 0.0, -12300.0, 0.0, -49300.0, 0.0, -63500.0, 0.0, -27800.0, 0.0, -26500.0, 0.0, -30000.0, -1104000000.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] 
        self.inputs_run_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'P_13', 'Q_13', 'P_14', 'Q_14', 'P_15', 'Q_15', 'P_16', 'Q_16', 'P_17', 'Q_17', 'P_18', 'Q_18', 'P_19', 'Q_19', 'P_20', 'Q_20', 'P_21', 'Q_21', 'P_22', 'Q_22', 'P_23', 'Q_23', 'P_24', 'Q_24', 'P_25', 'Q_25', 'P_26', 'Q_26', 'P_27', 'Q_27', 'P_28', 'Q_28', 'P_29', 'Q_29', 'P_30', 'Q_30', 'P_31', 'Q_31', 'P_32', 'Q_32', 'P_33', 'Q_33', 'P_34', 'Q_34', 'P_35', 'Q_35', 'P_36', 'Q_36', 'P_37', 'Q_37', 'P_38', 'Q_38', 'P_39', 'Q_39', 'v_ref_30', 'p_c_30', 'v_ref_31', 'p_c_31', 'v_ref_32', 'p_c_32', 'v_ref_33', 'p_c_33', 'v_ref_34', 'p_c_34', 'v_ref_35', 'p_c_35', 'v_ref_36', 'p_c_36', 'v_ref_37', 'p_c_37', 'v_ref_38', 'p_c_38', 'v_ref_39', 'p_c_39'] 
        self.inputs_run_values_list = [0.0, -0.0, 0.0, -0.0, -322000000.0, -2400000.0, -500000000.0, -184000000.0, 0.0, -0.0, 0.0, -0.0, -233800000.0, -84000000.0, -522000000.0, -176000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7500000.0, -88000000.0, 0.0, 0.0, 0.0, 0.0, -320000000.0, -153000000.0, -329000000.0, -32300000.0, 0.0, 0.0, -158000000.0, -30000000.0, 0.0, 0.0, -628000000.0, -103000000.0, -274000000.0, -115000000.0, 0.0, 0.0, -247500000.0, -84600000.0, -308600000.0, 92000000.0, -224000000.0, -47200000.0, -139000000.0, -17000000.0, -281000000.0, -75500000.0, -206000000.0, -27600000.0, -283500000.0, -26900000.0, -47500.0, 0.0, -982000.0, -9200000.0, -983100.0, 0.0, -997200.0, 0.0, -12300.0, 0.0, -49300.0, 0.0, -63500.0, 0.0, -27800.0, 0.0, -26500.0, 0.0, -30000.0, -1104000000.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0] 
        self.outputs_list = ['V_01', 'V_02', 'V_03', 'V_04', 'V_05', 'V_06', 'V_07', 'V_08', 'V_09', 'V_10', 'V_11', 'V_12', 'V_13', 'V_14', 'V_15', 'V_16', 'V_17', 'V_18', 'V_19', 'V_20', 'V_21', 'V_22', 'V_23', 'V_24', 'V_25', 'V_26', 'V_27', 'V_28', 'V_29', 'V_30', 'V_31', 'V_32', 'V_33', 'V_34', 'V_35', 'V_36', 'V_37', 'V_38', 'V_39', 'p_e_30', 'v_f_30', 'p_m_30', 'v_pss_30', 'v_ref_30', 'p_c_30', 'p_e_31', 'v_f_31', 'p_m_31', 'v_pss_31', 'v_ref_31', 'p_c_31', 'p_e_32', 'v_f_32', 'p_m_32', 'v_pss_32', 'v_ref_32', 'p_c_32', 'p_e_33', 'v_f_33', 'p_m_33', 'v_pss_33', 'v_ref_33', 'p_c_33', 'p_e_34', 'v_f_34', 'p_m_34', 'v_pss_34', 'v_ref_34', 'p_c_34', 'p_e_35', 'v_f_35', 'p_m_35', 'v_pss_35', 'v_ref_35', 'p_c_35', 'p_e_36', 'v_f_36', 'p_m_36', 'v_pss_36', 'v_ref_36', 'p_c_36', 'p_e_37', 'v_f_37', 'p_m_37', 'v_pss_37', 'v_ref_37', 'p_c_37', 'p_e_38', 'v_f_38', 'p_m_38', 'v_pss_38', 'v_ref_38', 'p_c_38', 'p_e_39', 'v_f_39', 'p_m_39', 'v_pss_39', 'v_ref_39', 'p_c_39'] 
        self.x_list = ['delta_30', 'omega_30', 'e1q_30', 'e1d_30', 'x_ab_30', 'x_e_30', 'xi_v_30', 'x_gov_1_30', 'x_gov_2_30', 'x_wo_30', 'x_lead_30', 'delta_31', 'omega_31', 'e1q_31', 'e1d_31', 'x_ab_31', 'x_e_31', 'xi_v_31', 'x_gov_1_31', 'x_gov_2_31', 'x_wo_31', 'x_lead_31', 'delta_32', 'omega_32', 'e1q_32', 'e1d_32', 'x_ab_32', 'x_e_32', 'xi_v_32', 'x_gov_1_32', 'x_gov_2_32', 'x_wo_32', 'x_lead_32', 'delta_33', 'omega_33', 'e1q_33', 'e1d_33', 'x_ab_33', 'x_e_33', 'xi_v_33', 'x_gov_1_33', 'x_gov_2_33', 'x_wo_33', 'x_lead_33', 'delta_34', 'omega_34', 'e1q_34', 'e1d_34', 'x_ab_34', 'x_e_34', 'xi_v_34', 'x_gov_1_34', 'x_gov_2_34', 'x_wo_34', 'x_lead_34', 'delta_35', 'omega_35', 'e1q_35', 'e1d_35', 'x_ab_35', 'x_e_35', 'xi_v_35', 'x_gov_1_35', 'x_gov_2_35', 'x_wo_35', 'x_lead_35', 'delta_36', 'omega_36', 'e1q_36', 'e1d_36', 'x_ab_36', 'x_e_36', 'xi_v_36', 'x_gov_1_36', 'x_gov_2_36', 'x_wo_36', 'x_lead_36', 'delta_37', 'omega_37', 'e1q_37', 'e1d_37', 'x_ab_37', 'x_e_37', 'xi_v_37', 'x_gov_1_37', 'x_gov_2_37', 'x_wo_37', 'x_lead_37', 'delta_38', 'omega_38', 'e1q_38', 'e1d_38', 'x_ab_38', 'x_e_38', 'xi_v_38', 'x_gov_1_38', 'x_gov_2_38', 'x_wo_38', 'x_lead_38', 'delta_39', 'omega_39', 'e1q_39', 'e1d_39', 'x_ab_39', 'x_e_39', 'xi_v_39', 'x_gov_1_39', 'x_gov_2_39', 'x_wo_39', 'x_lead_39', 'xi_freq'] 
        self.y_run_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'V_13', 'theta_13', 'V_14', 'theta_14', 'V_15', 'theta_15', 'V_16', 'theta_16', 'V_17', 'theta_17', 'V_18', 'theta_18', 'V_19', 'theta_19', 'V_20', 'theta_20', 'V_21', 'theta_21', 'V_22', 'theta_22', 'V_23', 'theta_23', 'V_24', 'theta_24', 'V_25', 'theta_25', 'V_26', 'theta_26', 'V_27', 'theta_27', 'V_28', 'theta_28', 'V_29', 'theta_29', 'V_30', 'theta_30', 'V_31', 'theta_31', 'V_32', 'theta_32', 'V_33', 'theta_33', 'V_34', 'theta_34', 'V_35', 'theta_35', 'V_36', 'theta_36', 'V_37', 'theta_37', 'V_38', 'theta_38', 'V_39', 'theta_39', 'i_d_30', 'i_q_30', 'p_g_30', 'q_g_30', 'v_f_30', 'p_m_ref_30', 'p_m_30', 'z_wo_30', 'v_pss_30', 'i_d_31', 'i_q_31', 'p_g_31', 'q_g_31', 'v_f_31', 'p_m_ref_31', 'p_m_31', 'z_wo_31', 'v_pss_31', 'i_d_32', 'i_q_32', 'p_g_32', 'q_g_32', 'v_f_32', 'p_m_ref_32', 'p_m_32', 'z_wo_32', 'v_pss_32', 'i_d_33', 'i_q_33', 'p_g_33', 'q_g_33', 'v_f_33', 'p_m_ref_33', 'p_m_33', 'z_wo_33', 'v_pss_33', 'i_d_34', 'i_q_34', 'p_g_34', 'q_g_34', 'v_f_34', 'p_m_ref_34', 'p_m_34', 'z_wo_34', 'v_pss_34', 'i_d_35', 'i_q_35', 'p_g_35', 'q_g_35', 'v_f_35', 'p_m_ref_35', 'p_m_35', 'z_wo_35', 'v_pss_35', 'i_d_36', 'i_q_36', 'p_g_36', 'q_g_36', 'v_f_36', 'p_m_ref_36', 'p_m_36', 'z_wo_36', 'v_pss_36', 'i_d_37', 'i_q_37', 'p_g_37', 'q_g_37', 'v_f_37', 'p_m_ref_37', 'p_m_37', 'z_wo_37', 'v_pss_37', 'i_d_38', 'i_q_38', 'p_g_38', 'q_g_38', 'v_f_38', 'p_m_ref_38', 'p_m_38', 'z_wo_38', 'v_pss_38', 'i_d_39', 'i_q_39', 'p_g_39', 'q_g_39', 'v_f_39', 'p_m_ref_39', 'p_m_39', 'z_wo_39', 'v_pss_39', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'V_13', 'theta_13', 'V_14', 'theta_14', 'V_15', 'theta_15', 'V_16', 'theta_16', 'V_17', 'theta_17', 'V_18', 'theta_18', 'V_19', 'theta_19', 'V_20', 'theta_20', 'V_21', 'theta_21', 'V_22', 'theta_22', 'V_23', 'theta_23', 'V_24', 'theta_24', 'V_25', 'theta_25', 'V_26', 'theta_26', 'V_27', 'theta_27', 'V_28', 'theta_28', 'V_29', 'theta_29', 'V_30', 'theta_30', 'V_31', 'theta_31', 'V_32', 'theta_32', 'V_33', 'theta_33', 'V_34', 'theta_34', 'V_35', 'theta_35', 'V_36', 'theta_36', 'V_37', 'theta_37', 'V_38', 'theta_38', 'V_39', 'theta_39', 'i_d_30', 'i_q_30', 'p_g_30', 'q_g_30', 'v_f_30', 'p_m_ref_30', 'p_m_30', 'z_wo_30', 'v_pss_30', 'i_d_31', 'i_q_31', 'p_g_31', 'q_g_31', 'v_f_31', 'p_m_ref_31', 'p_m_31', 'z_wo_31', 'v_pss_31', 'i_d_32', 'i_q_32', 'p_g_32', 'q_g_32', 'v_f_32', 'p_m_ref_32', 'p_m_32', 'z_wo_32', 'v_pss_32', 'i_d_33', 'i_q_33', 'p_g_33', 'q_g_33', 'v_f_33', 'p_m_ref_33', 'p_m_33', 'z_wo_33', 'v_pss_33', 'i_d_34', 'i_q_34', 'p_g_34', 'q_g_34', 'v_f_34', 'p_m_ref_34', 'p_m_34', 'z_wo_34', 'v_pss_34', 'i_d_35', 'i_q_35', 'p_g_35', 'q_g_35', 'v_f_35', 'p_m_ref_35', 'p_m_35', 'z_wo_35', 'v_pss_35', 'i_d_36', 'i_q_36', 'p_g_36', 'q_g_36', 'v_f_36', 'p_m_ref_36', 'p_m_36', 'z_wo_36', 'v_pss_36', 'i_d_37', 'i_q_37', 'p_g_37', 'q_g_37', 'v_f_37', 'p_m_ref_37', 'p_m_37', 'z_wo_37', 'v_pss_37', 'i_d_38', 'i_q_38', 'p_g_38', 'q_g_38', 'v_f_38', 'p_m_ref_38', 'p_m_38', 'z_wo_38', 'v_pss_38', 'i_d_39', 'i_q_39', 'p_g_39', 'q_g_39', 'v_f_39', 'p_m_ref_39', 'p_m_39', 'z_wo_39', 'v_pss_39', 'omega_coi', 'p_agc'] 
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
            fobj = BytesIO(pkgutil.get_data(__name__, f'./spvib_dq_d_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz(f'./{self.matrices_folder}/spvib_dq_d_sp_jac_ini_num.npz')
            
            
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
            fobj = BytesIO(pkgutil.get_data(__name__, './spvib_dq_d_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz(f'./{self.matrices_folder}/spvib_dq_d_sp_jac_run_num.npz')
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
            fobj = BytesIO(pkgutil.get_data(__name__, './spvib_dq_d_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz(f'./{self.matrices_folder}/spvib_dq_d_sp_jac_trap_num.npz')
            

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

        #self.sp_Fu_run = sspa.load_npz(f'./{self.matrices_folder}/spvib_dq_d_Fu_run_num.npz')
        #self.sp_Gu_run = sspa.load_npz(f'./{self.matrices_folder}/spvib_dq_d_Gu_run_num.npz')
        #self.sp_Hx_run = sspa.load_npz(f'./{self.matrices_folder}/spvib_dq_d_Hx_run_num.npz')
        #self.sp_Hy_run = sspa.load_npz(f'./{self.matrices_folder}/spvib_dq_d_Hy_run_num.npz')
        #self.sp_Hu_run = sspa.load_npz(f'./{self.matrices_folder}/spvib_dq_d_Hu_run_num.npz')        
        
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

    sp_jac_ini_ia = [0, 1, 279, 0, 1, 169, 170, 189, 190, 195, 279, 2, 189, 193, 3, 190, 4, 6, 169, 197, 4, 5, 6, 169, 197, 169, 7, 194, 7, 8, 1, 9, 10, 196, 11, 12, 279, 11, 12, 171, 172, 198, 199, 204, 279, 13, 198, 202, 14, 199, 15, 17, 171, 206, 15, 16, 17, 171, 206, 171, 18, 203, 18, 19, 12, 20, 21, 205, 22, 23, 279, 22, 23, 173, 174, 207, 208, 213, 279, 24, 207, 211, 25, 208, 26, 28, 173, 215, 26, 27, 28, 173, 215, 173, 29, 212, 29, 30, 23, 31, 32, 214, 33, 34, 279, 33, 34, 175, 176, 216, 217, 222, 279, 35, 216, 220, 36, 217, 37, 39, 175, 224, 37, 38, 39, 175, 224, 175, 40, 221, 40, 41, 34, 42, 43, 223, 44, 45, 279, 44, 45, 177, 178, 225, 226, 231, 279, 46, 225, 229, 47, 226, 48, 50, 177, 233, 48, 49, 50, 177, 233, 177, 51, 230, 51, 52, 45, 53, 54, 232, 55, 56, 279, 55, 56, 179, 180, 234, 235, 240, 279, 57, 234, 238, 58, 235, 59, 61, 179, 242, 59, 60, 61, 179, 242, 179, 62, 239, 62, 63, 56, 64, 65, 241, 66, 67, 279, 66, 67, 181, 182, 243, 244, 249, 279, 68, 243, 247, 69, 244, 70, 72, 181, 251, 70, 71, 72, 181, 251, 181, 73, 248, 73, 74, 67, 75, 76, 250, 77, 78, 279, 77, 78, 183, 184, 252, 253, 258, 279, 79, 252, 256, 80, 253, 81, 83, 183, 260, 81, 82, 83, 183, 260, 183, 84, 257, 84, 85, 78, 86, 87, 259, 88, 89, 279, 88, 89, 185, 186, 261, 262, 267, 279, 90, 261, 265, 91, 262, 92, 94, 185, 269, 92, 93, 94, 185, 269, 185, 95, 266, 95, 96, 89, 97, 98, 268, 99, 100, 279, 99, 100, 187, 188, 270, 271, 276, 279, 101, 270, 274, 102, 271, 103, 105, 187, 278, 103, 104, 105, 187, 278, 187, 106, 275, 106, 107, 100, 108, 109, 277, 110, 279, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 113, 114, 115, 116, 117, 118, 145, 146, 113, 114, 115, 116, 117, 118, 145, 146, 115, 116, 117, 118, 119, 120, 137, 138, 115, 116, 117, 118, 119, 120, 137, 138, 117, 118, 119, 120, 121, 122, 125, 126, 117, 118, 119, 120, 121, 122, 125, 126, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 121, 122, 123, 124, 125, 126, 121, 122, 123, 124, 125, 126, 119, 120, 123, 124, 125, 126, 127, 128, 119, 120, 123, 124, 125, 126, 127, 128, 125, 126, 127, 128, 187, 188, 125, 126, 127, 128, 187, 188, 129, 130, 131, 132, 135, 136, 173, 174, 129, 130, 131, 132, 135, 136, 173, 174, 121, 122, 129, 130, 131, 132, 133, 134, 121, 122, 129, 130, 131, 132, 133, 134, 131, 132, 133, 134, 135, 136, 131, 132, 133, 134, 135, 136, 129, 130, 133, 134, 135, 136, 137, 138, 129, 130, 133, 134, 135, 136, 137, 138, 117, 118, 135, 136, 137, 138, 139, 140, 117, 118, 135, 136, 137, 138, 139, 140, 137, 138, 139, 140, 141, 142, 137, 138, 139, 140, 141, 142, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 141, 142, 143, 144, 145, 146, 163, 164, 141, 142, 143, 144, 145, 146, 163, 164, 115, 116, 143, 144, 145, 146, 115, 116, 143, 144, 145, 146, 141, 142, 147, 148, 149, 150, 175, 176, 141, 142, 147, 148, 149, 150, 175, 176, 147, 148, 149, 150, 177, 178, 147, 148, 149, 150, 177, 178, 141, 142, 151, 152, 153, 154, 141, 142, 151, 152, 153, 154, 151, 152, 153, 154, 155, 156, 179, 180, 151, 152, 153, 154, 155, 156, 179, 180, 153, 154, 155, 156, 157, 158, 181, 182, 153, 154, 155, 156, 157, 158, 181, 182, 141, 142, 155, 156, 157, 158, 141, 142, 155, 156, 157, 158, 113, 114, 159, 160, 161, 162, 183, 184, 113, 114, 159, 160, 161, 162, 183, 184, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 143, 144, 161, 162, 163, 164, 143, 144, 161, 162, 163, 164, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 185, 186, 161, 162, 165, 166, 167, 168, 185, 186, 113, 114, 169, 170, 191, 113, 114, 169, 170, 192, 121, 122, 171, 172, 200, 121, 122, 171, 172, 201, 129, 130, 173, 174, 209, 129, 130, 173, 174, 210, 147, 148, 175, 176, 218, 147, 148, 175, 176, 219, 149, 150, 177, 178, 227, 149, 150, 177, 178, 228, 153, 154, 179, 180, 236, 153, 154, 179, 180, 237, 155, 156, 181, 182, 245, 155, 156, 181, 182, 246, 159, 160, 183, 184, 254, 159, 160, 183, 184, 255, 167, 168, 185, 186, 263, 167, 168, 185, 186, 264, 111, 112, 127, 128, 187, 188, 272, 111, 112, 127, 128, 187, 188, 273, 0, 2, 169, 170, 189, 190, 0, 3, 169, 170, 189, 190, 0, 169, 170, 189, 190, 191, 0, 169, 170, 189, 190, 192, 5, 193, 1, 194, 280, 1, 7, 8, 195, 1, 9, 196, 10, 196, 197, 11, 13, 171, 172, 198, 199, 11, 14, 171, 172, 198, 199, 11, 171, 172, 198, 199, 200, 11, 171, 172, 198, 199, 201, 16, 202, 12, 203, 280, 12, 18, 19, 204, 12, 20, 205, 21, 205, 206, 22, 24, 173, 174, 207, 208, 22, 25, 173, 174, 207, 208, 22, 173, 174, 207, 208, 209, 22, 173, 174, 207, 208, 210, 27, 211, 23, 212, 280, 23, 29, 30, 213, 23, 31, 214, 32, 214, 215, 33, 35, 175, 176, 216, 217, 33, 36, 175, 176, 216, 217, 33, 175, 176, 216, 217, 218, 33, 175, 176, 216, 217, 219, 38, 220, 34, 221, 280, 34, 40, 41, 222, 34, 42, 223, 43, 223, 224, 44, 46, 177, 178, 225, 226, 44, 47, 177, 178, 225, 226, 44, 177, 178, 225, 226, 227, 44, 177, 178, 225, 226, 228, 49, 229, 45, 230, 280, 45, 51, 52, 231, 45, 53, 232, 54, 232, 233, 55, 57, 179, 180, 234, 235, 55, 58, 179, 180, 234, 235, 55, 179, 180, 234, 235, 236, 55, 179, 180, 234, 235, 237, 60, 238, 56, 239, 280, 56, 62, 63, 240, 56, 64, 241, 65, 241, 242, 66, 68, 181, 182, 243, 244, 66, 69, 181, 182, 243, 244, 66, 181, 182, 243, 244, 245, 66, 181, 182, 243, 244, 246, 71, 247, 67, 248, 280, 67, 73, 74, 249, 67, 75, 250, 76, 250, 251, 77, 79, 183, 184, 252, 253, 77, 80, 183, 184, 252, 253, 77, 183, 184, 252, 253, 254, 77, 183, 184, 252, 253, 255, 82, 256, 78, 257, 280, 78, 84, 85, 258, 78, 86, 259, 87, 259, 260, 88, 90, 185, 186, 261, 262, 88, 91, 185, 186, 261, 262, 88, 185, 186, 261, 262, 263, 88, 185, 186, 261, 262, 264, 93, 265, 89, 266, 280, 89, 95, 96, 267, 89, 97, 268, 98, 268, 269, 99, 101, 187, 188, 270, 271, 99, 102, 187, 188, 270, 271, 99, 187, 188, 270, 271, 272, 99, 187, 188, 270, 271, 273, 104, 274, 100, 275, 280, 100, 106, 107, 276, 100, 108, 277, 109, 277, 278, 1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 279, 110, 279, 280]
    sp_jac_ini_ja = [0, 3, 11, 14, 16, 20, 25, 26, 28, 30, 32, 34, 37, 45, 48, 50, 54, 59, 60, 62, 64, 66, 68, 71, 79, 82, 84, 88, 93, 94, 96, 98, 100, 102, 105, 113, 116, 118, 122, 127, 128, 130, 132, 134, 136, 139, 147, 150, 152, 156, 161, 162, 164, 166, 168, 170, 173, 181, 184, 186, 190, 195, 196, 198, 200, 202, 204, 207, 215, 218, 220, 224, 229, 230, 232, 234, 236, 238, 241, 249, 252, 254, 258, 263, 264, 266, 268, 270, 272, 275, 283, 286, 288, 292, 297, 298, 300, 302, 304, 306, 309, 317, 320, 322, 326, 331, 332, 334, 336, 338, 340, 342, 348, 354, 364, 374, 382, 390, 398, 406, 414, 422, 432, 442, 448, 454, 462, 470, 476, 482, 490, 498, 506, 514, 520, 526, 534, 542, 550, 558, 564, 570, 582, 594, 602, 610, 616, 622, 630, 638, 644, 650, 656, 662, 670, 678, 686, 694, 700, 706, 714, 722, 732, 742, 748, 754, 760, 766, 774, 782, 787, 792, 797, 802, 807, 812, 817, 822, 827, 832, 837, 842, 847, 852, 857, 862, 867, 872, 879, 886, 892, 898, 904, 910, 912, 915, 919, 922, 925, 931, 937, 943, 949, 951, 954, 958, 961, 964, 970, 976, 982, 988, 990, 993, 997, 1000, 1003, 1009, 1015, 1021, 1027, 1029, 1032, 1036, 1039, 1042, 1048, 1054, 1060, 1066, 1068, 1071, 1075, 1078, 1081, 1087, 1093, 1099, 1105, 1107, 1110, 1114, 1117, 1120, 1126, 1132, 1138, 1144, 1146, 1149, 1153, 1156, 1159, 1165, 1171, 1177, 1183, 1185, 1188, 1192, 1195, 1198, 1204, 1210, 1216, 1222, 1224, 1227, 1231, 1234, 1237, 1243, 1249, 1255, 1261, 1263, 1266, 1270, 1273, 1276, 1287, 1290]
    sp_jac_ini_nia = 281
    sp_jac_ini_nja = 281
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 1, 279, 0, 1, 169, 170, 189, 190, 195, 279, 2, 189, 193, 3, 190, 4, 6, 169, 197, 4, 5, 6, 169, 197, 169, 7, 194, 7, 8, 1, 9, 10, 196, 11, 12, 279, 11, 12, 171, 172, 198, 199, 204, 279, 13, 198, 202, 14, 199, 15, 17, 171, 206, 15, 16, 17, 171, 206, 171, 18, 203, 18, 19, 12, 20, 21, 205, 22, 23, 279, 22, 23, 173, 174, 207, 208, 213, 279, 24, 207, 211, 25, 208, 26, 28, 173, 215, 26, 27, 28, 173, 215, 173, 29, 212, 29, 30, 23, 31, 32, 214, 33, 34, 279, 33, 34, 175, 176, 216, 217, 222, 279, 35, 216, 220, 36, 217, 37, 39, 175, 224, 37, 38, 39, 175, 224, 175, 40, 221, 40, 41, 34, 42, 43, 223, 44, 45, 279, 44, 45, 177, 178, 225, 226, 231, 279, 46, 225, 229, 47, 226, 48, 50, 177, 233, 48, 49, 50, 177, 233, 177, 51, 230, 51, 52, 45, 53, 54, 232, 55, 56, 279, 55, 56, 179, 180, 234, 235, 240, 279, 57, 234, 238, 58, 235, 59, 61, 179, 242, 59, 60, 61, 179, 242, 179, 62, 239, 62, 63, 56, 64, 65, 241, 66, 67, 279, 66, 67, 181, 182, 243, 244, 249, 279, 68, 243, 247, 69, 244, 70, 72, 181, 251, 70, 71, 72, 181, 251, 181, 73, 248, 73, 74, 67, 75, 76, 250, 77, 78, 279, 77, 78, 183, 184, 252, 253, 258, 279, 79, 252, 256, 80, 253, 81, 83, 183, 260, 81, 82, 83, 183, 260, 183, 84, 257, 84, 85, 78, 86, 87, 259, 88, 89, 279, 88, 89, 185, 186, 261, 262, 267, 279, 90, 261, 265, 91, 262, 92, 94, 185, 269, 92, 93, 94, 185, 269, 185, 95, 266, 95, 96, 89, 97, 98, 268, 99, 100, 279, 99, 100, 187, 188, 270, 271, 276, 279, 101, 270, 274, 102, 271, 103, 105, 187, 278, 103, 104, 105, 187, 278, 187, 106, 275, 106, 107, 100, 108, 109, 277, 110, 279, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 113, 114, 115, 116, 117, 118, 145, 146, 113, 114, 115, 116, 117, 118, 145, 146, 115, 116, 117, 118, 119, 120, 137, 138, 115, 116, 117, 118, 119, 120, 137, 138, 117, 118, 119, 120, 121, 122, 125, 126, 117, 118, 119, 120, 121, 122, 125, 126, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 121, 122, 123, 124, 125, 126, 121, 122, 123, 124, 125, 126, 119, 120, 123, 124, 125, 126, 127, 128, 119, 120, 123, 124, 125, 126, 127, 128, 125, 126, 127, 128, 187, 188, 125, 126, 127, 128, 187, 188, 129, 130, 131, 132, 135, 136, 173, 174, 129, 130, 131, 132, 135, 136, 173, 174, 121, 122, 129, 130, 131, 132, 133, 134, 121, 122, 129, 130, 131, 132, 133, 134, 131, 132, 133, 134, 135, 136, 131, 132, 133, 134, 135, 136, 129, 130, 133, 134, 135, 136, 137, 138, 129, 130, 133, 134, 135, 136, 137, 138, 117, 118, 135, 136, 137, 138, 139, 140, 117, 118, 135, 136, 137, 138, 139, 140, 137, 138, 139, 140, 141, 142, 137, 138, 139, 140, 141, 142, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 141, 142, 143, 144, 145, 146, 163, 164, 141, 142, 143, 144, 145, 146, 163, 164, 115, 116, 143, 144, 145, 146, 115, 116, 143, 144, 145, 146, 141, 142, 147, 148, 149, 150, 175, 176, 141, 142, 147, 148, 149, 150, 175, 176, 147, 148, 149, 150, 177, 178, 147, 148, 149, 150, 177, 178, 141, 142, 151, 152, 153, 154, 141, 142, 151, 152, 153, 154, 151, 152, 153, 154, 155, 156, 179, 180, 151, 152, 153, 154, 155, 156, 179, 180, 153, 154, 155, 156, 157, 158, 181, 182, 153, 154, 155, 156, 157, 158, 181, 182, 141, 142, 155, 156, 157, 158, 141, 142, 155, 156, 157, 158, 113, 114, 159, 160, 161, 162, 183, 184, 113, 114, 159, 160, 161, 162, 183, 184, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 143, 144, 161, 162, 163, 164, 143, 144, 161, 162, 163, 164, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 185, 186, 161, 162, 165, 166, 167, 168, 185, 186, 113, 114, 169, 170, 191, 113, 114, 169, 170, 192, 121, 122, 171, 172, 200, 121, 122, 171, 172, 201, 129, 130, 173, 174, 209, 129, 130, 173, 174, 210, 147, 148, 175, 176, 218, 147, 148, 175, 176, 219, 149, 150, 177, 178, 227, 149, 150, 177, 178, 228, 153, 154, 179, 180, 236, 153, 154, 179, 180, 237, 155, 156, 181, 182, 245, 155, 156, 181, 182, 246, 159, 160, 183, 184, 254, 159, 160, 183, 184, 255, 167, 168, 185, 186, 263, 167, 168, 185, 186, 264, 111, 112, 127, 128, 187, 188, 272, 111, 112, 127, 128, 187, 188, 273, 0, 2, 169, 170, 189, 190, 0, 3, 169, 170, 189, 190, 0, 169, 170, 189, 190, 191, 0, 169, 170, 189, 190, 192, 5, 193, 1, 194, 280, 1, 7, 8, 195, 1, 9, 196, 10, 196, 197, 11, 13, 171, 172, 198, 199, 11, 14, 171, 172, 198, 199, 11, 171, 172, 198, 199, 200, 11, 171, 172, 198, 199, 201, 16, 202, 12, 203, 280, 12, 18, 19, 204, 12, 20, 205, 21, 205, 206, 22, 24, 173, 174, 207, 208, 22, 25, 173, 174, 207, 208, 22, 173, 174, 207, 208, 209, 22, 173, 174, 207, 208, 210, 27, 211, 23, 212, 280, 23, 29, 30, 213, 23, 31, 214, 32, 214, 215, 33, 35, 175, 176, 216, 217, 33, 36, 175, 176, 216, 217, 33, 175, 176, 216, 217, 218, 33, 175, 176, 216, 217, 219, 38, 220, 34, 221, 280, 34, 40, 41, 222, 34, 42, 223, 43, 223, 224, 44, 46, 177, 178, 225, 226, 44, 47, 177, 178, 225, 226, 44, 177, 178, 225, 226, 227, 44, 177, 178, 225, 226, 228, 49, 229, 45, 230, 280, 45, 51, 52, 231, 45, 53, 232, 54, 232, 233, 55, 57, 179, 180, 234, 235, 55, 58, 179, 180, 234, 235, 55, 179, 180, 234, 235, 236, 55, 179, 180, 234, 235, 237, 60, 238, 56, 239, 280, 56, 62, 63, 240, 56, 64, 241, 65, 241, 242, 66, 68, 181, 182, 243, 244, 66, 69, 181, 182, 243, 244, 66, 181, 182, 243, 244, 245, 66, 181, 182, 243, 244, 246, 71, 247, 67, 248, 280, 67, 73, 74, 249, 67, 75, 250, 76, 250, 251, 77, 79, 183, 184, 252, 253, 77, 80, 183, 184, 252, 253, 77, 183, 184, 252, 253, 254, 77, 183, 184, 252, 253, 255, 82, 256, 78, 257, 280, 78, 84, 85, 258, 78, 86, 259, 87, 259, 260, 88, 90, 185, 186, 261, 262, 88, 91, 185, 186, 261, 262, 88, 185, 186, 261, 262, 263, 88, 185, 186, 261, 262, 264, 93, 265, 89, 266, 280, 89, 95, 96, 267, 89, 97, 268, 98, 268, 269, 99, 101, 187, 188, 270, 271, 99, 102, 187, 188, 270, 271, 99, 187, 188, 270, 271, 272, 99, 187, 188, 270, 271, 273, 104, 274, 100, 275, 280, 100, 106, 107, 276, 100, 108, 277, 109, 277, 278, 1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 279, 110, 279, 280]
    sp_jac_run_ja = [0, 3, 11, 14, 16, 20, 25, 26, 28, 30, 32, 34, 37, 45, 48, 50, 54, 59, 60, 62, 64, 66, 68, 71, 79, 82, 84, 88, 93, 94, 96, 98, 100, 102, 105, 113, 116, 118, 122, 127, 128, 130, 132, 134, 136, 139, 147, 150, 152, 156, 161, 162, 164, 166, 168, 170, 173, 181, 184, 186, 190, 195, 196, 198, 200, 202, 204, 207, 215, 218, 220, 224, 229, 230, 232, 234, 236, 238, 241, 249, 252, 254, 258, 263, 264, 266, 268, 270, 272, 275, 283, 286, 288, 292, 297, 298, 300, 302, 304, 306, 309, 317, 320, 322, 326, 331, 332, 334, 336, 338, 340, 342, 348, 354, 364, 374, 382, 390, 398, 406, 414, 422, 432, 442, 448, 454, 462, 470, 476, 482, 490, 498, 506, 514, 520, 526, 534, 542, 550, 558, 564, 570, 582, 594, 602, 610, 616, 622, 630, 638, 644, 650, 656, 662, 670, 678, 686, 694, 700, 706, 714, 722, 732, 742, 748, 754, 760, 766, 774, 782, 787, 792, 797, 802, 807, 812, 817, 822, 827, 832, 837, 842, 847, 852, 857, 862, 867, 872, 879, 886, 892, 898, 904, 910, 912, 915, 919, 922, 925, 931, 937, 943, 949, 951, 954, 958, 961, 964, 970, 976, 982, 988, 990, 993, 997, 1000, 1003, 1009, 1015, 1021, 1027, 1029, 1032, 1036, 1039, 1042, 1048, 1054, 1060, 1066, 1068, 1071, 1075, 1078, 1081, 1087, 1093, 1099, 1105, 1107, 1110, 1114, 1117, 1120, 1126, 1132, 1138, 1144, 1146, 1149, 1153, 1156, 1159, 1165, 1171, 1177, 1183, 1185, 1188, 1192, 1195, 1198, 1204, 1210, 1216, 1222, 1224, 1227, 1231, 1234, 1237, 1243, 1249, 1255, 1261, 1263, 1266, 1270, 1273, 1276, 1287, 1290]
    sp_jac_run_nia = 281
    sp_jac_run_nja = 281
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 279, 0, 1, 169, 170, 189, 190, 195, 279, 2, 189, 193, 3, 190, 4, 6, 169, 197, 4, 5, 6, 169, 197, 6, 169, 7, 194, 7, 8, 1, 9, 10, 196, 11, 12, 279, 11, 12, 171, 172, 198, 199, 204, 279, 13, 198, 202, 14, 199, 15, 17, 171, 206, 15, 16, 17, 171, 206, 17, 171, 18, 203, 18, 19, 12, 20, 21, 205, 22, 23, 279, 22, 23, 173, 174, 207, 208, 213, 279, 24, 207, 211, 25, 208, 26, 28, 173, 215, 26, 27, 28, 173, 215, 28, 173, 29, 212, 29, 30, 23, 31, 32, 214, 33, 34, 279, 33, 34, 175, 176, 216, 217, 222, 279, 35, 216, 220, 36, 217, 37, 39, 175, 224, 37, 38, 39, 175, 224, 39, 175, 40, 221, 40, 41, 34, 42, 43, 223, 44, 45, 279, 44, 45, 177, 178, 225, 226, 231, 279, 46, 225, 229, 47, 226, 48, 50, 177, 233, 48, 49, 50, 177, 233, 50, 177, 51, 230, 51, 52, 45, 53, 54, 232, 55, 56, 279, 55, 56, 179, 180, 234, 235, 240, 279, 57, 234, 238, 58, 235, 59, 61, 179, 242, 59, 60, 61, 179, 242, 61, 179, 62, 239, 62, 63, 56, 64, 65, 241, 66, 67, 279, 66, 67, 181, 182, 243, 244, 249, 279, 68, 243, 247, 69, 244, 70, 72, 181, 251, 70, 71, 72, 181, 251, 72, 181, 73, 248, 73, 74, 67, 75, 76, 250, 77, 78, 279, 77, 78, 183, 184, 252, 253, 258, 279, 79, 252, 256, 80, 253, 81, 83, 183, 260, 81, 82, 83, 183, 260, 83, 183, 84, 257, 84, 85, 78, 86, 87, 259, 88, 89, 279, 88, 89, 185, 186, 261, 262, 267, 279, 90, 261, 265, 91, 262, 92, 94, 185, 269, 92, 93, 94, 185, 269, 94, 185, 95, 266, 95, 96, 89, 97, 98, 268, 99, 100, 279, 99, 100, 187, 188, 270, 271, 276, 279, 101, 270, 274, 102, 271, 103, 105, 187, 278, 103, 104, 105, 187, 278, 105, 187, 106, 275, 106, 107, 100, 108, 109, 277, 110, 279, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 113, 114, 115, 116, 117, 118, 145, 146, 113, 114, 115, 116, 117, 118, 145, 146, 115, 116, 117, 118, 119, 120, 137, 138, 115, 116, 117, 118, 119, 120, 137, 138, 117, 118, 119, 120, 121, 122, 125, 126, 117, 118, 119, 120, 121, 122, 125, 126, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 121, 122, 123, 124, 125, 126, 121, 122, 123, 124, 125, 126, 119, 120, 123, 124, 125, 126, 127, 128, 119, 120, 123, 124, 125, 126, 127, 128, 125, 126, 127, 128, 187, 188, 125, 126, 127, 128, 187, 188, 129, 130, 131, 132, 135, 136, 173, 174, 129, 130, 131, 132, 135, 136, 173, 174, 121, 122, 129, 130, 131, 132, 133, 134, 121, 122, 129, 130, 131, 132, 133, 134, 131, 132, 133, 134, 135, 136, 131, 132, 133, 134, 135, 136, 129, 130, 133, 134, 135, 136, 137, 138, 129, 130, 133, 134, 135, 136, 137, 138, 117, 118, 135, 136, 137, 138, 139, 140, 117, 118, 135, 136, 137, 138, 139, 140, 137, 138, 139, 140, 141, 142, 137, 138, 139, 140, 141, 142, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 141, 142, 143, 144, 145, 146, 163, 164, 141, 142, 143, 144, 145, 146, 163, 164, 115, 116, 143, 144, 145, 146, 115, 116, 143, 144, 145, 146, 141, 142, 147, 148, 149, 150, 175, 176, 141, 142, 147, 148, 149, 150, 175, 176, 147, 148, 149, 150, 177, 178, 147, 148, 149, 150, 177, 178, 141, 142, 151, 152, 153, 154, 141, 142, 151, 152, 153, 154, 151, 152, 153, 154, 155, 156, 179, 180, 151, 152, 153, 154, 155, 156, 179, 180, 153, 154, 155, 156, 157, 158, 181, 182, 153, 154, 155, 156, 157, 158, 181, 182, 141, 142, 155, 156, 157, 158, 141, 142, 155, 156, 157, 158, 113, 114, 159, 160, 161, 162, 183, 184, 113, 114, 159, 160, 161, 162, 183, 184, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 143, 144, 161, 162, 163, 164, 143, 144, 161, 162, 163, 164, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 185, 186, 161, 162, 165, 166, 167, 168, 185, 186, 113, 114, 169, 170, 191, 113, 114, 169, 170, 192, 121, 122, 171, 172, 200, 121, 122, 171, 172, 201, 129, 130, 173, 174, 209, 129, 130, 173, 174, 210, 147, 148, 175, 176, 218, 147, 148, 175, 176, 219, 149, 150, 177, 178, 227, 149, 150, 177, 178, 228, 153, 154, 179, 180, 236, 153, 154, 179, 180, 237, 155, 156, 181, 182, 245, 155, 156, 181, 182, 246, 159, 160, 183, 184, 254, 159, 160, 183, 184, 255, 167, 168, 185, 186, 263, 167, 168, 185, 186, 264, 111, 112, 127, 128, 187, 188, 272, 111, 112, 127, 128, 187, 188, 273, 0, 2, 169, 170, 189, 190, 0, 3, 169, 170, 189, 190, 0, 169, 170, 189, 190, 191, 0, 169, 170, 189, 190, 192, 5, 193, 1, 194, 280, 1, 7, 8, 195, 1, 9, 196, 10, 196, 197, 11, 13, 171, 172, 198, 199, 11, 14, 171, 172, 198, 199, 11, 171, 172, 198, 199, 200, 11, 171, 172, 198, 199, 201, 16, 202, 12, 203, 280, 12, 18, 19, 204, 12, 20, 205, 21, 205, 206, 22, 24, 173, 174, 207, 208, 22, 25, 173, 174, 207, 208, 22, 173, 174, 207, 208, 209, 22, 173, 174, 207, 208, 210, 27, 211, 23, 212, 280, 23, 29, 30, 213, 23, 31, 214, 32, 214, 215, 33, 35, 175, 176, 216, 217, 33, 36, 175, 176, 216, 217, 33, 175, 176, 216, 217, 218, 33, 175, 176, 216, 217, 219, 38, 220, 34, 221, 280, 34, 40, 41, 222, 34, 42, 223, 43, 223, 224, 44, 46, 177, 178, 225, 226, 44, 47, 177, 178, 225, 226, 44, 177, 178, 225, 226, 227, 44, 177, 178, 225, 226, 228, 49, 229, 45, 230, 280, 45, 51, 52, 231, 45, 53, 232, 54, 232, 233, 55, 57, 179, 180, 234, 235, 55, 58, 179, 180, 234, 235, 55, 179, 180, 234, 235, 236, 55, 179, 180, 234, 235, 237, 60, 238, 56, 239, 280, 56, 62, 63, 240, 56, 64, 241, 65, 241, 242, 66, 68, 181, 182, 243, 244, 66, 69, 181, 182, 243, 244, 66, 181, 182, 243, 244, 245, 66, 181, 182, 243, 244, 246, 71, 247, 67, 248, 280, 67, 73, 74, 249, 67, 75, 250, 76, 250, 251, 77, 79, 183, 184, 252, 253, 77, 80, 183, 184, 252, 253, 77, 183, 184, 252, 253, 254, 77, 183, 184, 252, 253, 255, 82, 256, 78, 257, 280, 78, 84, 85, 258, 78, 86, 259, 87, 259, 260, 88, 90, 185, 186, 261, 262, 88, 91, 185, 186, 261, 262, 88, 185, 186, 261, 262, 263, 88, 185, 186, 261, 262, 264, 93, 265, 89, 266, 280, 89, 95, 96, 267, 89, 97, 268, 98, 268, 269, 99, 101, 187, 188, 270, 271, 99, 102, 187, 188, 270, 271, 99, 187, 188, 270, 271, 272, 99, 187, 188, 270, 271, 273, 104, 274, 100, 275, 280, 100, 106, 107, 276, 100, 108, 277, 109, 277, 278, 1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 279, 110, 279, 280]
    sp_jac_trap_ja = [0, 3, 11, 14, 16, 20, 25, 27, 29, 31, 33, 35, 38, 46, 49, 51, 55, 60, 62, 64, 66, 68, 70, 73, 81, 84, 86, 90, 95, 97, 99, 101, 103, 105, 108, 116, 119, 121, 125, 130, 132, 134, 136, 138, 140, 143, 151, 154, 156, 160, 165, 167, 169, 171, 173, 175, 178, 186, 189, 191, 195, 200, 202, 204, 206, 208, 210, 213, 221, 224, 226, 230, 235, 237, 239, 241, 243, 245, 248, 256, 259, 261, 265, 270, 272, 274, 276, 278, 280, 283, 291, 294, 296, 300, 305, 307, 309, 311, 313, 315, 318, 326, 329, 331, 335, 340, 342, 344, 346, 348, 350, 352, 358, 364, 374, 384, 392, 400, 408, 416, 424, 432, 442, 452, 458, 464, 472, 480, 486, 492, 500, 508, 516, 524, 530, 536, 544, 552, 560, 568, 574, 580, 592, 604, 612, 620, 626, 632, 640, 648, 654, 660, 666, 672, 680, 688, 696, 704, 710, 716, 724, 732, 742, 752, 758, 764, 770, 776, 784, 792, 797, 802, 807, 812, 817, 822, 827, 832, 837, 842, 847, 852, 857, 862, 867, 872, 877, 882, 889, 896, 902, 908, 914, 920, 922, 925, 929, 932, 935, 941, 947, 953, 959, 961, 964, 968, 971, 974, 980, 986, 992, 998, 1000, 1003, 1007, 1010, 1013, 1019, 1025, 1031, 1037, 1039, 1042, 1046, 1049, 1052, 1058, 1064, 1070, 1076, 1078, 1081, 1085, 1088, 1091, 1097, 1103, 1109, 1115, 1117, 1120, 1124, 1127, 1130, 1136, 1142, 1148, 1154, 1156, 1159, 1163, 1166, 1169, 1175, 1181, 1187, 1193, 1195, 1198, 1202, 1205, 1208, 1214, 1220, 1226, 1232, 1234, 1237, 1241, 1244, 1247, 1253, 1259, 1265, 1271, 1273, 1276, 1280, 1283, 1286, 1297, 1300]
    sp_jac_trap_nia = 281
    sp_jac_trap_nja = 281
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
