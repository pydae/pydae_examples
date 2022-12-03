import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import euro_cffi as jacs

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


class euro_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 67
        self.N_y = 80 
        self.N_z = 42 
        self.N_store = 100000 
        self.params_list = ['S_base', 'g_ES_G_ES_T', 'b_ES_G_ES_T', 'bs_ES_G_ES_T', 'g_DE_G_DE_T', 'b_DE_G_DE_T', 'bs_DE_G_DE_T', 'g_FR_G_FR_T', 'b_FR_G_FR_T', 'bs_FR_G_FR_T', 'g_IT_G_IT_T', 'b_IT_G_IT_T', 'bs_IT_G_IT_T', 'g_PL_G_PL_T', 'b_PL_G_PL_T', 'bs_PL_G_PL_T', 'g_BA_G_BA_T', 'b_BA_G_BA_T', 'bs_BA_G_BA_T', 'g_ES_T_FR_T', 'b_ES_T_FR_T', 'bs_ES_T_FR_T', 'g_FR_T_DE_T', 'b_FR_T_DE_T', 'bs_FR_T_DE_T', 'g_FR_T_IT_T', 'b_FR_T_IT_T', 'bs_FR_T_IT_T', 'g_DE_T_PL_T', 'b_DE_T_PL_T', 'bs_DE_T_PL_T', 'g_DE_T_BA_T', 'b_DE_T_BA_T', 'bs_DE_T_BA_T', 'g_IT_T_BA_T', 'b_IT_T_BA_T', 'bs_IT_T_BA_T', 'U_ES_G_n', 'U_DE_G_n', 'U_FR_G_n', 'U_IT_G_n', 'U_PL_G_n', 'U_BA_G_n', 'U_ES_T_n', 'U_DE_T_n', 'U_FR_T_n', 'U_IT_T_n', 'U_PL_T_n', 'U_BA_T_n', 'S_n_ES_G', 'Omega_b_ES_G', 'H_ES_G', 'T1d0_ES_G', 'T1q0_ES_G', 'X_d_ES_G', 'X_q_ES_G', 'X1d_ES_G', 'X1q_ES_G', 'D_ES_G', 'R_a_ES_G', 'K_delta_ES_G', 'K_sec_ES_G', 'K_a_ES_G', 'K_ai_ES_G', 'T_r_ES_G', 'V_min_ES_G', 'V_max_ES_G', 'K_aw_ES_G', 'Droop_ES_G', 'T_gov_1_ES_G', 'T_gov_2_ES_G', 'T_gov_3_ES_G', 'K_imw_ES_G', 'omega_ref_ES_G', 'T_wo_ES_G', 'T_1_ES_G', 'T_2_ES_G', 'K_stab_ES_G', 'V_lim_ES_G', 'S_n_FR_G', 'Omega_b_FR_G', 'H_FR_G', 'T1d0_FR_G', 'T1q0_FR_G', 'X_d_FR_G', 'X_q_FR_G', 'X1d_FR_G', 'X1q_FR_G', 'D_FR_G', 'R_a_FR_G', 'K_delta_FR_G', 'K_sec_FR_G', 'K_a_FR_G', 'K_ai_FR_G', 'T_r_FR_G', 'V_min_FR_G', 'V_max_FR_G', 'K_aw_FR_G', 'Droop_FR_G', 'T_gov_1_FR_G', 'T_gov_2_FR_G', 'T_gov_3_FR_G', 'K_imw_FR_G', 'omega_ref_FR_G', 'T_wo_FR_G', 'T_1_FR_G', 'T_2_FR_G', 'K_stab_FR_G', 'V_lim_FR_G', 'S_n_DE_G', 'Omega_b_DE_G', 'H_DE_G', 'T1d0_DE_G', 'T1q0_DE_G', 'X_d_DE_G', 'X_q_DE_G', 'X1d_DE_G', 'X1q_DE_G', 'D_DE_G', 'R_a_DE_G', 'K_delta_DE_G', 'K_sec_DE_G', 'K_a_DE_G', 'K_ai_DE_G', 'T_r_DE_G', 'V_min_DE_G', 'V_max_DE_G', 'K_aw_DE_G', 'Droop_DE_G', 'T_gov_1_DE_G', 'T_gov_2_DE_G', 'T_gov_3_DE_G', 'K_imw_DE_G', 'omega_ref_DE_G', 'T_wo_DE_G', 'T_1_DE_G', 'T_2_DE_G', 'K_stab_DE_G', 'V_lim_DE_G', 'S_n_IT_G', 'Omega_b_IT_G', 'H_IT_G', 'T1d0_IT_G', 'T1q0_IT_G', 'X_d_IT_G', 'X_q_IT_G', 'X1d_IT_G', 'X1q_IT_G', 'D_IT_G', 'R_a_IT_G', 'K_delta_IT_G', 'K_sec_IT_G', 'K_a_IT_G', 'K_ai_IT_G', 'T_r_IT_G', 'V_min_IT_G', 'V_max_IT_G', 'K_aw_IT_G', 'Droop_IT_G', 'T_gov_1_IT_G', 'T_gov_2_IT_G', 'T_gov_3_IT_G', 'K_imw_IT_G', 'omega_ref_IT_G', 'T_wo_IT_G', 'T_1_IT_G', 'T_2_IT_G', 'K_stab_IT_G', 'V_lim_IT_G', 'S_n_PL_G', 'Omega_b_PL_G', 'H_PL_G', 'T1d0_PL_G', 'T1q0_PL_G', 'X_d_PL_G', 'X_q_PL_G', 'X1d_PL_G', 'X1q_PL_G', 'D_PL_G', 'R_a_PL_G', 'K_delta_PL_G', 'K_sec_PL_G', 'K_a_PL_G', 'K_ai_PL_G', 'T_r_PL_G', 'V_min_PL_G', 'V_max_PL_G', 'K_aw_PL_G', 'Droop_PL_G', 'T_gov_1_PL_G', 'T_gov_2_PL_G', 'T_gov_3_PL_G', 'K_imw_PL_G', 'omega_ref_PL_G', 'T_wo_PL_G', 'T_1_PL_G', 'T_2_PL_G', 'K_stab_PL_G', 'V_lim_PL_G', 'S_n_BA_G', 'Omega_b_BA_G', 'H_BA_G', 'T1d0_BA_G', 'T1q0_BA_G', 'X_d_BA_G', 'X_q_BA_G', 'X1d_BA_G', 'X1q_BA_G', 'D_BA_G', 'R_a_BA_G', 'K_delta_BA_G', 'K_sec_BA_G', 'K_a_BA_G', 'K_ai_BA_G', 'T_r_BA_G', 'V_min_BA_G', 'V_max_BA_G', 'K_aw_BA_G', 'Droop_BA_G', 'T_gov_1_BA_G', 'T_gov_2_BA_G', 'T_gov_3_BA_G', 'K_imw_BA_G', 'omega_ref_BA_G', 'T_wo_BA_G', 'T_1_BA_G', 'T_2_BA_G', 'K_stab_BA_G', 'V_lim_BA_G', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [1000000000.0, 0.0, -66666666.66666667, 0.0, 0.0, -66666666.66666667, 0.0, 0.0, -66666666.66666667, 0.0, 0.0, -66666666.66666667, 0.0, 0.0, -66666666.66666667, 0.0, 0.0, -66666666.66666667, 0.0, 0.0, -100.0, -0.05, 0.0, -100.0, -0.0, 0.0, -100.0, -0.0, 0.0, -100.0, -0.0, 0.0, -100.0, -0.0, 0.0, -100.0, -0.0, 20000.0, 20000.0, 20000.0, 20000.0, 20000.0, 20000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 1500000000.0, 314.1592653589793, 6.3, 6.47, 0.61, 2.135, 2.046, 0.34, 0.573, 0.0, 0.0, 0.0, 0.0, 100, 1e-06, 0.02, -5, 5.0, 2.0, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 1500000000.0, 314.1592653589793, 6.3, 6.47, 0.61, 2.135, 2.046, 0.34, 0.573, 0.0, 0.0, 0.0, 0.0, 100, 1e-06, 0.02, -5, 5.0, 2.0, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 5000000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.01, 0.01, 1.0, 100, 1e-06, 0.02, -5, 5.0, 2.0, 0.05, 1.0, 2.0, 10.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 1500000000.0, 314.1592653589793, 6.3, 6.47, 0.61, 2.135, 2.046, 0.34, 0.573, 0.0, 0.0, 0.0, 0.0, 100, 1e-06, 0.02, -5, 5.0, 2.0, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 1500000000.0, 314.1592653589793, 6.3, 6.47, 0.61, 2.135, 2.046, 0.34, 0.573, 0.0, 0.0, 0.0, 0.0, 100, 1e-06, 0.02, -5, 5.0, 2.0, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 1500000000.0, 314.1592653589793, 6.3, 6.47, 0.61, 2.135, 2.046, 0.34, 0.573, 0.0, 0.0, 0.0, 0.0, 100, 1e-06, 0.02, -5, 5.0, 2.0, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 1.0, 0.1] 
        self.inputs_ini_list = ['P_ES_G', 'Q_ES_G', 'P_DE_G', 'Q_DE_G', 'P_FR_G', 'Q_FR_G', 'P_IT_G', 'Q_IT_G', 'P_PL_G', 'Q_PL_G', 'P_BA_G', 'Q_BA_G', 'P_ES_T', 'Q_ES_T', 'P_DE_T', 'Q_DE_T', 'P_FR_T', 'Q_FR_T', 'P_IT_T', 'Q_IT_T', 'P_PL_T', 'Q_PL_T', 'P_BA_T', 'Q_BA_T', 'v_ref_ES_G', 'v_pss_ES_G', 'p_c_ES_G', 'p_r_ES_G', 'v_ref_FR_G', 'v_pss_FR_G', 'p_c_FR_G', 'p_r_FR_G', 'v_ref_DE_G', 'v_pss_DE_G', 'p_c_DE_G', 'p_r_DE_G', 'v_ref_IT_G', 'v_pss_IT_G', 'p_c_IT_G', 'p_r_IT_G', 'v_ref_PL_G', 'v_pss_PL_G', 'p_c_PL_G', 'p_r_PL_G', 'v_ref_BA_G', 'v_pss_BA_G', 'p_c_BA_G', 'p_r_BA_G'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0] 
        self.inputs_run_list = ['P_ES_G', 'Q_ES_G', 'P_DE_G', 'Q_DE_G', 'P_FR_G', 'Q_FR_G', 'P_IT_G', 'Q_IT_G', 'P_PL_G', 'Q_PL_G', 'P_BA_G', 'Q_BA_G', 'P_ES_T', 'Q_ES_T', 'P_DE_T', 'Q_DE_T', 'P_FR_T', 'Q_FR_T', 'P_IT_T', 'Q_IT_T', 'P_PL_T', 'Q_PL_T', 'P_BA_T', 'Q_BA_T', 'v_ref_ES_G', 'v_pss_ES_G', 'p_c_ES_G', 'p_r_ES_G', 'v_ref_FR_G', 'v_pss_FR_G', 'p_c_FR_G', 'p_r_FR_G', 'v_ref_DE_G', 'v_pss_DE_G', 'p_c_DE_G', 'p_r_DE_G', 'v_ref_IT_G', 'v_pss_IT_G', 'p_c_IT_G', 'p_r_IT_G', 'v_ref_PL_G', 'v_pss_PL_G', 'p_c_PL_G', 'p_r_PL_G', 'v_ref_BA_G', 'v_pss_BA_G', 'p_c_BA_G', 'p_r_BA_G'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0, 1.0, 0.0, 0.01, 0.0] 
        self.outputs_list = ['V_ES_G', 'V_DE_G', 'V_FR_G', 'V_IT_G', 'V_PL_G', 'V_BA_G', 'V_ES_T', 'V_DE_T', 'V_FR_T', 'V_IT_T', 'V_PL_T', 'V_BA_T', 'p_line_ES_G_ES_T', 'q_line_ES_G_ES_T', 'p_line_DE_G_DE_T', 'q_line_DE_G_DE_T', 'p_line_FR_G_FR_T', 'q_line_FR_G_FR_T', 'p_line_IT_G_IT_T', 'q_line_IT_G_IT_T', 'p_line_PL_G_PL_T', 'q_line_PL_G_PL_T', 'p_line_BA_G_BA_T', 'q_line_BA_G_BA_T', 'p_line_ES_T_FR_T', 'q_line_ES_T_FR_T', 'p_line_FR_T_DE_T', 'q_line_FR_T_DE_T', 'p_line_FR_T_IT_T', 'q_line_FR_T_IT_T', 'p_line_DE_T_PL_T', 'q_line_DE_T_PL_T', 'p_line_DE_T_BA_T', 'q_line_DE_T_BA_T', 'p_line_IT_T_BA_T', 'q_line_IT_T_BA_T', 'p_e_ES_G', 'p_e_FR_G', 'p_e_DE_G', 'p_e_IT_G', 'p_e_PL_G', 'p_e_BA_G'] 
        self.x_list = ['delta_ES_G', 'omega_ES_G', 'e1q_ES_G', 'e1d_ES_G', 'v_c_ES_G', 'xi_v_ES_G', 'x_gov_1_ES_G', 'x_gov_2_ES_G', 'xi_imw_ES_G', 'x_wo_ES_G', 'x_lead_ES_G', 'delta_FR_G', 'omega_FR_G', 'e1q_FR_G', 'e1d_FR_G', 'v_c_FR_G', 'xi_v_FR_G', 'x_gov_1_FR_G', 'x_gov_2_FR_G', 'xi_imw_FR_G', 'x_wo_FR_G', 'x_lead_FR_G', 'delta_DE_G', 'omega_DE_G', 'e1q_DE_G', 'e1d_DE_G', 'v_c_DE_G', 'xi_v_DE_G', 'x_gov_1_DE_G', 'x_gov_2_DE_G', 'xi_imw_DE_G', 'x_wo_DE_G', 'x_lead_DE_G', 'delta_IT_G', 'omega_IT_G', 'e1q_IT_G', 'e1d_IT_G', 'v_c_IT_G', 'xi_v_IT_G', 'x_gov_1_IT_G', 'x_gov_2_IT_G', 'xi_imw_IT_G', 'x_wo_IT_G', 'x_lead_IT_G', 'delta_PL_G', 'omega_PL_G', 'e1q_PL_G', 'e1d_PL_G', 'v_c_PL_G', 'xi_v_PL_G', 'x_gov_1_PL_G', 'x_gov_2_PL_G', 'xi_imw_PL_G', 'x_wo_PL_G', 'x_lead_PL_G', 'delta_BA_G', 'omega_BA_G', 'e1q_BA_G', 'e1d_BA_G', 'v_c_BA_G', 'xi_v_BA_G', 'x_gov_1_BA_G', 'x_gov_2_BA_G', 'xi_imw_BA_G', 'x_wo_BA_G', 'x_lead_BA_G', 'xi_freq'] 
        self.y_run_list = ['V_ES_G', 'theta_ES_G', 'V_DE_G', 'theta_DE_G', 'V_FR_G', 'theta_FR_G', 'V_IT_G', 'theta_IT_G', 'V_PL_G', 'theta_PL_G', 'V_BA_G', 'theta_BA_G', 'V_ES_T', 'theta_ES_T', 'V_DE_T', 'theta_DE_T', 'V_FR_T', 'theta_FR_T', 'V_IT_T', 'theta_IT_T', 'V_PL_T', 'theta_PL_T', 'V_BA_T', 'theta_BA_T', 'i_d_ES_G', 'i_q_ES_G', 'p_g_ES_G', 'q_g_ES_G', 'v_f_ES_G', 'p_m_ref_ES_G', 'p_m_ES_G', 'z_wo_ES_G', 'v_pss_ES_G', 'i_d_FR_G', 'i_q_FR_G', 'p_g_FR_G', 'q_g_FR_G', 'v_f_FR_G', 'p_m_ref_FR_G', 'p_m_FR_G', 'z_wo_FR_G', 'v_pss_FR_G', 'i_d_DE_G', 'i_q_DE_G', 'p_g_DE_G', 'q_g_DE_G', 'v_f_DE_G', 'p_m_ref_DE_G', 'p_m_DE_G', 'z_wo_DE_G', 'v_pss_DE_G', 'i_d_IT_G', 'i_q_IT_G', 'p_g_IT_G', 'q_g_IT_G', 'v_f_IT_G', 'p_m_ref_IT_G', 'p_m_IT_G', 'z_wo_IT_G', 'v_pss_IT_G', 'i_d_PL_G', 'i_q_PL_G', 'p_g_PL_G', 'q_g_PL_G', 'v_f_PL_G', 'p_m_ref_PL_G', 'p_m_PL_G', 'z_wo_PL_G', 'v_pss_PL_G', 'i_d_BA_G', 'i_q_BA_G', 'p_g_BA_G', 'q_g_BA_G', 'v_f_BA_G', 'p_m_ref_BA_G', 'p_m_BA_G', 'z_wo_BA_G', 'v_pss_BA_G', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_ES_G', 'theta_ES_G', 'V_DE_G', 'theta_DE_G', 'V_FR_G', 'theta_FR_G', 'V_IT_G', 'theta_IT_G', 'V_PL_G', 'theta_PL_G', 'V_BA_G', 'theta_BA_G', 'V_ES_T', 'theta_ES_T', 'V_DE_T', 'theta_DE_T', 'V_FR_T', 'theta_FR_T', 'V_IT_T', 'theta_IT_T', 'V_PL_T', 'theta_PL_T', 'V_BA_T', 'theta_BA_T', 'i_d_ES_G', 'i_q_ES_G', 'p_g_ES_G', 'q_g_ES_G', 'v_f_ES_G', 'p_m_ref_ES_G', 'p_m_ES_G', 'z_wo_ES_G', 'v_pss_ES_G', 'i_d_FR_G', 'i_q_FR_G', 'p_g_FR_G', 'q_g_FR_G', 'v_f_FR_G', 'p_m_ref_FR_G', 'p_m_FR_G', 'z_wo_FR_G', 'v_pss_FR_G', 'i_d_DE_G', 'i_q_DE_G', 'p_g_DE_G', 'q_g_DE_G', 'v_f_DE_G', 'p_m_ref_DE_G', 'p_m_DE_G', 'z_wo_DE_G', 'v_pss_DE_G', 'i_d_IT_G', 'i_q_IT_G', 'p_g_IT_G', 'q_g_IT_G', 'v_f_IT_G', 'p_m_ref_IT_G', 'p_m_IT_G', 'z_wo_IT_G', 'v_pss_IT_G', 'i_d_PL_G', 'i_q_PL_G', 'p_g_PL_G', 'q_g_PL_G', 'v_f_PL_G', 'p_m_ref_PL_G', 'p_m_PL_G', 'z_wo_PL_G', 'v_pss_PL_G', 'i_d_BA_G', 'i_q_BA_G', 'p_g_BA_G', 'q_g_BA_G', 'v_f_BA_G', 'p_m_ref_BA_G', 'p_m_BA_G', 'z_wo_BA_G', 'v_pss_BA_G', 'omega_coi', 'p_agc'] 
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
        self.sp_jac_ini = sspa.load_npz('euro_sp_jac_ini_num.npz')
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
        #self.sp_jac_run = sspa.csr_matrix((data, self.sp_jac_run_ia, self.sp_jac_run_ja), shape=(self.sp_jac_run_nia,self.sp_jac_run_nja))
        self.sp_jac_run = sspa.load_npz('euro_sp_jac_run_num.npz')
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
        self.sp_jac_trap = sspa.load_npz('euro_sp_jac_trap_num.npz')
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

        #self.sp_Fu_run = sspa.load_npz('euro_Fu_run_num.npz')
        #self.sp_Gu_run = sspa.load_npz('euro_Gu_run_num.npz')
        #self.sp_Hx_run = sspa.load_npz('euro_Hx_run_num.npz')
        #self.sp_Hy_run = sspa.load_npz('euro_Hy_run_num.npz')
        #self.sp_Hu_run = sspa.load_npz('euro_Hu_run_num.npz')        
 
        



        
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
        self.iparams_run = np.zeros(10,dtype=np.float64)
    
        t,it,it_store,xy = spdaesolver(t,t_end,it,it_store,xy,u,p,
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
                 ldamp=self.ldamp)

 
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
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
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
def spdaesolver(t,t_end,it,it_store,xy,u,p,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,
                iparams,
                max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0):

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
    
    #sp_jac_trap_num_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
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

    sp_jac_ini_ia = [0, 1, 145, 0, 1, 67, 68, 91, 92, 97, 145, 2, 91, 95, 3, 92, 4, 67, 4, 5, 95, 99, 6, 96, 6, 7, 8, 93, 1, 9, 10, 98, 11, 12, 145, 11, 12, 71, 72, 100, 101, 106, 145, 13, 100, 104, 14, 101, 15, 71, 15, 16, 104, 108, 17, 105, 17, 18, 19, 102, 12, 20, 21, 107, 22, 23, 145, 22, 23, 69, 70, 109, 110, 115, 145, 24, 109, 113, 25, 110, 26, 69, 26, 27, 113, 117, 28, 114, 28, 29, 30, 111, 23, 31, 32, 116, 33, 34, 145, 33, 34, 73, 74, 118, 119, 124, 145, 35, 118, 122, 36, 119, 37, 73, 37, 38, 122, 126, 39, 123, 39, 40, 41, 120, 34, 42, 43, 125, 44, 45, 145, 44, 45, 75, 76, 127, 128, 133, 145, 46, 127, 131, 47, 128, 48, 75, 48, 49, 131, 135, 50, 132, 50, 51, 52, 129, 45, 53, 54, 134, 55, 56, 145, 55, 56, 77, 78, 136, 137, 142, 145, 57, 136, 140, 58, 137, 59, 77, 59, 60, 140, 144, 61, 141, 61, 62, 63, 138, 56, 64, 65, 143, 145, 67, 68, 79, 80, 93, 67, 68, 79, 80, 94, 69, 70, 81, 82, 111, 69, 70, 81, 82, 112, 71, 72, 83, 84, 102, 71, 72, 83, 84, 103, 73, 74, 85, 86, 120, 73, 74, 85, 86, 121, 75, 76, 87, 88, 129, 75, 76, 87, 88, 130, 77, 78, 89, 90, 138, 77, 78, 89, 90, 139, 67, 68, 79, 80, 83, 84, 67, 68, 79, 80, 83, 84, 69, 70, 81, 82, 83, 84, 87, 88, 89, 90, 69, 70, 81, 82, 83, 84, 87, 88, 89, 90, 71, 72, 79, 80, 81, 82, 83, 84, 85, 86, 71, 72, 79, 80, 81, 82, 83, 84, 85, 86, 73, 74, 83, 84, 85, 86, 89, 90, 73, 74, 83, 84, 85, 86, 89, 90, 75, 76, 81, 82, 87, 88, 75, 76, 81, 82, 87, 88, 77, 78, 81, 82, 85, 86, 89, 90, 77, 78, 81, 82, 85, 86, 89, 90, 0, 2, 67, 68, 91, 92, 0, 3, 67, 68, 91, 92, 0, 67, 68, 91, 92, 93, 0, 67, 68, 91, 92, 94, 4, 5, 95, 99, 1, 8, 96, 146, 6, 7, 97, 1, 9, 98, 10, 98, 99, 11, 13, 71, 72, 100, 101, 11, 14, 71, 72, 100, 101, 11, 71, 72, 100, 101, 102, 11, 71, 72, 100, 101, 103, 15, 16, 104, 108, 12, 19, 105, 146, 17, 18, 106, 12, 20, 107, 21, 107, 108, 22, 24, 69, 70, 109, 110, 22, 25, 69, 70, 109, 110, 22, 69, 70, 109, 110, 111, 22, 69, 70, 109, 110, 112, 26, 27, 113, 117, 23, 30, 114, 146, 28, 29, 115, 23, 31, 116, 32, 116, 117, 33, 35, 73, 74, 118, 119, 33, 36, 73, 74, 118, 119, 33, 73, 74, 118, 119, 120, 33, 73, 74, 118, 119, 121, 37, 38, 122, 126, 34, 41, 123, 146, 39, 40, 124, 34, 42, 125, 43, 125, 126, 44, 46, 75, 76, 127, 128, 44, 47, 75, 76, 127, 128, 44, 75, 76, 127, 128, 129, 44, 75, 76, 127, 128, 130, 48, 49, 131, 135, 45, 52, 132, 146, 50, 51, 133, 45, 53, 134, 54, 134, 135, 55, 57, 77, 78, 136, 137, 55, 58, 77, 78, 136, 137, 55, 77, 78, 136, 137, 138, 55, 77, 78, 136, 137, 139, 59, 60, 140, 144, 56, 63, 141, 146, 61, 62, 142, 56, 64, 143, 65, 143, 144, 1, 12, 23, 34, 45, 56, 145, 66, 145, 146]
    sp_jac_ini_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 131, 139, 142, 144, 146, 150, 152, 154, 156, 158, 160, 163, 171, 174, 176, 178, 182, 184, 186, 188, 190, 192, 193, 198, 203, 208, 213, 218, 223, 228, 233, 238, 243, 248, 253, 259, 265, 275, 285, 295, 305, 313, 321, 327, 333, 341, 349, 355, 361, 367, 373, 377, 381, 384, 387, 390, 396, 402, 408, 414, 418, 422, 425, 428, 431, 437, 443, 449, 455, 459, 463, 466, 469, 472, 478, 484, 490, 496, 500, 504, 507, 510, 513, 519, 525, 531, 537, 541, 545, 548, 551, 554, 560, 566, 572, 578, 582, 586, 589, 592, 595, 602, 605]
    sp_jac_ini_nia = 147
    sp_jac_ini_nja = 147
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 1, 145, 0, 1, 67, 68, 91, 92, 97, 145, 2, 91, 95, 3, 92, 4, 67, 4, 5, 95, 99, 6, 96, 6, 7, 8, 93, 1, 9, 10, 98, 11, 12, 145, 11, 12, 71, 72, 100, 101, 106, 145, 13, 100, 104, 14, 101, 15, 71, 15, 16, 104, 108, 17, 105, 17, 18, 19, 102, 12, 20, 21, 107, 22, 23, 145, 22, 23, 69, 70, 109, 110, 115, 145, 24, 109, 113, 25, 110, 26, 69, 26, 27, 113, 117, 28, 114, 28, 29, 30, 111, 23, 31, 32, 116, 33, 34, 145, 33, 34, 73, 74, 118, 119, 124, 145, 35, 118, 122, 36, 119, 37, 73, 37, 38, 122, 126, 39, 123, 39, 40, 41, 120, 34, 42, 43, 125, 44, 45, 145, 44, 45, 75, 76, 127, 128, 133, 145, 46, 127, 131, 47, 128, 48, 75, 48, 49, 131, 135, 50, 132, 50, 51, 52, 129, 45, 53, 54, 134, 55, 56, 145, 55, 56, 77, 78, 136, 137, 142, 145, 57, 136, 140, 58, 137, 59, 77, 59, 60, 140, 144, 61, 141, 61, 62, 63, 138, 56, 64, 65, 143, 145, 67, 68, 79, 80, 93, 67, 68, 79, 80, 94, 69, 70, 81, 82, 111, 69, 70, 81, 82, 112, 71, 72, 83, 84, 102, 71, 72, 83, 84, 103, 73, 74, 85, 86, 120, 73, 74, 85, 86, 121, 75, 76, 87, 88, 129, 75, 76, 87, 88, 130, 77, 78, 89, 90, 138, 77, 78, 89, 90, 139, 67, 68, 79, 80, 83, 84, 67, 68, 79, 80, 83, 84, 69, 70, 81, 82, 83, 84, 87, 88, 89, 90, 69, 70, 81, 82, 83, 84, 87, 88, 89, 90, 71, 72, 79, 80, 81, 82, 83, 84, 85, 86, 71, 72, 79, 80, 81, 82, 83, 84, 85, 86, 73, 74, 83, 84, 85, 86, 89, 90, 73, 74, 83, 84, 85, 86, 89, 90, 75, 76, 81, 82, 87, 88, 75, 76, 81, 82, 87, 88, 77, 78, 81, 82, 85, 86, 89, 90, 77, 78, 81, 82, 85, 86, 89, 90, 0, 2, 67, 68, 91, 92, 0, 3, 67, 68, 91, 92, 0, 67, 68, 91, 92, 93, 0, 67, 68, 91, 92, 94, 4, 5, 95, 99, 1, 8, 96, 146, 6, 7, 97, 1, 9, 98, 10, 98, 99, 11, 13, 71, 72, 100, 101, 11, 14, 71, 72, 100, 101, 11, 71, 72, 100, 101, 102, 11, 71, 72, 100, 101, 103, 15, 16, 104, 108, 12, 19, 105, 146, 17, 18, 106, 12, 20, 107, 21, 107, 108, 22, 24, 69, 70, 109, 110, 22, 25, 69, 70, 109, 110, 22, 69, 70, 109, 110, 111, 22, 69, 70, 109, 110, 112, 26, 27, 113, 117, 23, 30, 114, 146, 28, 29, 115, 23, 31, 116, 32, 116, 117, 33, 35, 73, 74, 118, 119, 33, 36, 73, 74, 118, 119, 33, 73, 74, 118, 119, 120, 33, 73, 74, 118, 119, 121, 37, 38, 122, 126, 34, 41, 123, 146, 39, 40, 124, 34, 42, 125, 43, 125, 126, 44, 46, 75, 76, 127, 128, 44, 47, 75, 76, 127, 128, 44, 75, 76, 127, 128, 129, 44, 75, 76, 127, 128, 130, 48, 49, 131, 135, 45, 52, 132, 146, 50, 51, 133, 45, 53, 134, 54, 134, 135, 55, 57, 77, 78, 136, 137, 55, 58, 77, 78, 136, 137, 55, 77, 78, 136, 137, 138, 55, 77, 78, 136, 137, 139, 59, 60, 140, 144, 56, 63, 141, 146, 61, 62, 142, 56, 64, 143, 65, 143, 144, 1, 12, 23, 34, 45, 56, 145, 66, 145, 146]
    sp_jac_run_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 131, 139, 142, 144, 146, 150, 152, 154, 156, 158, 160, 163, 171, 174, 176, 178, 182, 184, 186, 188, 190, 192, 193, 198, 203, 208, 213, 218, 223, 228, 233, 238, 243, 248, 253, 259, 265, 275, 285, 295, 305, 313, 321, 327, 333, 341, 349, 355, 361, 367, 373, 377, 381, 384, 387, 390, 396, 402, 408, 414, 418, 422, 425, 428, 431, 437, 443, 449, 455, 459, 463, 466, 469, 472, 478, 484, 490, 496, 500, 504, 507, 510, 513, 519, 525, 531, 537, 541, 545, 548, 551, 554, 560, 566, 572, 578, 582, 586, 589, 592, 595, 602, 605]
    sp_jac_run_nia = 147
    sp_jac_run_nja = 147
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 145, 0, 1, 67, 68, 91, 92, 97, 145, 2, 91, 95, 3, 92, 4, 67, 4, 5, 95, 99, 6, 96, 6, 7, 8, 93, 1, 9, 10, 98, 11, 12, 145, 11, 12, 71, 72, 100, 101, 106, 145, 13, 100, 104, 14, 101, 15, 71, 15, 16, 104, 108, 17, 105, 17, 18, 19, 102, 12, 20, 21, 107, 22, 23, 145, 22, 23, 69, 70, 109, 110, 115, 145, 24, 109, 113, 25, 110, 26, 69, 26, 27, 113, 117, 28, 114, 28, 29, 30, 111, 23, 31, 32, 116, 33, 34, 145, 33, 34, 73, 74, 118, 119, 124, 145, 35, 118, 122, 36, 119, 37, 73, 37, 38, 122, 126, 39, 123, 39, 40, 41, 120, 34, 42, 43, 125, 44, 45, 145, 44, 45, 75, 76, 127, 128, 133, 145, 46, 127, 131, 47, 128, 48, 75, 48, 49, 131, 135, 50, 132, 50, 51, 52, 129, 45, 53, 54, 134, 55, 56, 145, 55, 56, 77, 78, 136, 137, 142, 145, 57, 136, 140, 58, 137, 59, 77, 59, 60, 140, 144, 61, 141, 61, 62, 63, 138, 56, 64, 65, 143, 66, 145, 67, 68, 79, 80, 93, 67, 68, 79, 80, 94, 69, 70, 81, 82, 111, 69, 70, 81, 82, 112, 71, 72, 83, 84, 102, 71, 72, 83, 84, 103, 73, 74, 85, 86, 120, 73, 74, 85, 86, 121, 75, 76, 87, 88, 129, 75, 76, 87, 88, 130, 77, 78, 89, 90, 138, 77, 78, 89, 90, 139, 67, 68, 79, 80, 83, 84, 67, 68, 79, 80, 83, 84, 69, 70, 81, 82, 83, 84, 87, 88, 89, 90, 69, 70, 81, 82, 83, 84, 87, 88, 89, 90, 71, 72, 79, 80, 81, 82, 83, 84, 85, 86, 71, 72, 79, 80, 81, 82, 83, 84, 85, 86, 73, 74, 83, 84, 85, 86, 89, 90, 73, 74, 83, 84, 85, 86, 89, 90, 75, 76, 81, 82, 87, 88, 75, 76, 81, 82, 87, 88, 77, 78, 81, 82, 85, 86, 89, 90, 77, 78, 81, 82, 85, 86, 89, 90, 0, 2, 67, 68, 91, 92, 0, 3, 67, 68, 91, 92, 0, 67, 68, 91, 92, 93, 0, 67, 68, 91, 92, 94, 4, 5, 95, 99, 1, 8, 96, 146, 6, 7, 97, 1, 9, 98, 10, 98, 99, 11, 13, 71, 72, 100, 101, 11, 14, 71, 72, 100, 101, 11, 71, 72, 100, 101, 102, 11, 71, 72, 100, 101, 103, 15, 16, 104, 108, 12, 19, 105, 146, 17, 18, 106, 12, 20, 107, 21, 107, 108, 22, 24, 69, 70, 109, 110, 22, 25, 69, 70, 109, 110, 22, 69, 70, 109, 110, 111, 22, 69, 70, 109, 110, 112, 26, 27, 113, 117, 23, 30, 114, 146, 28, 29, 115, 23, 31, 116, 32, 116, 117, 33, 35, 73, 74, 118, 119, 33, 36, 73, 74, 118, 119, 33, 73, 74, 118, 119, 120, 33, 73, 74, 118, 119, 121, 37, 38, 122, 126, 34, 41, 123, 146, 39, 40, 124, 34, 42, 125, 43, 125, 126, 44, 46, 75, 76, 127, 128, 44, 47, 75, 76, 127, 128, 44, 75, 76, 127, 128, 129, 44, 75, 76, 127, 128, 130, 48, 49, 131, 135, 45, 52, 132, 146, 50, 51, 133, 45, 53, 134, 54, 134, 135, 55, 57, 77, 78, 136, 137, 55, 58, 77, 78, 136, 137, 55, 77, 78, 136, 137, 138, 55, 77, 78, 136, 137, 139, 59, 60, 140, 144, 56, 63, 141, 146, 61, 62, 142, 56, 64, 143, 65, 143, 144, 1, 12, 23, 34, 45, 56, 145, 66, 145, 146]
    sp_jac_trap_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 131, 139, 142, 144, 146, 150, 152, 154, 156, 158, 160, 163, 171, 174, 176, 178, 182, 184, 186, 188, 190, 192, 194, 199, 204, 209, 214, 219, 224, 229, 234, 239, 244, 249, 254, 260, 266, 276, 286, 296, 306, 314, 322, 328, 334, 342, 350, 356, 362, 368, 374, 378, 382, 385, 388, 391, 397, 403, 409, 415, 419, 423, 426, 429, 432, 438, 444, 450, 456, 460, 464, 467, 470, 473, 479, 485, 491, 497, 501, 505, 508, 511, 514, 520, 526, 532, 538, 542, 546, 549, 552, 555, 561, 567, 573, 579, 583, 587, 590, 593, 596, 603, 606]
    sp_jac_trap_nia = 147
    sp_jac_trap_nja = 147
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
