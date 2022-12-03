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
    import wf19_cffi as jacs
if dae_file_mode == 'enviroment':
    import envus.no_enviroment.wf19_cffi as jacs
if dae_file_mode == 'colab':
    import wf19_cffi as jacs
    
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


class model: 

    def __init__(self): 
        
        self.dae_file_mode = 'local'
        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 38
        self.N_y = 86 
        self.N_z = 64 
        self.N_store = 100000 
        self.params_list = ['S_base', 'g_01_02', 'b_01_02', 'bs_01_02', 'g_03_04', 'b_03_04', 'bs_03_04', 'g_05_06', 'b_05_06', 'bs_05_06', 'g_07_08', 'b_07_08', 'bs_07_08', 'g_02_04', 'b_02_04', 'bs_02_04', 'g_04_06', 'b_04_06', 'bs_04_06', 'g_06_08', 'b_06_08', 'bs_06_08', 'g_08_17', 'b_08_17', 'bs_08_17', 'g_09_10', 'b_09_10', 'bs_09_10', 'g_11_12', 'b_11_12', 'bs_11_12', 'g_13_14', 'b_13_14', 'bs_13_14', 'g_15_16', 'b_15_16', 'bs_15_16', 'g_10_12', 'b_10_12', 'bs_10_12', 'g_12_14', 'b_12_14', 'bs_12_14', 'g_14_16', 'b_14_16', 'bs_14_16', 'g_16_17', 'b_16_17', 'bs_16_17', 'g_17_18', 'b_17_18', 'bs_17_18', 'g_18_19', 'b_18_19', 'bs_18_19', 'U_01_n', 'U_02_n', 'U_03_n', 'U_04_n', 'U_05_n', 'U_06_n', 'U_07_n', 'U_08_n', 'U_09_n', 'U_10_n', 'U_11_n', 'U_12_n', 'U_13_n', 'U_14_n', 'U_15_n', 'U_16_n', 'U_17_n', 'U_18_n', 'U_19_n', 'S_n_19', 'Omega_b_19', 'K_p_19', 'T_p_19', 'K_q_19', 'T_v_19', 'X_v_19', 'R_v_19', 'K_delta_19', 'K_sec_19', 'Droop_19', 'T_e_19', 'T_c_19', 'S_n_01', 'F_n_01', 'K_delta_01', 'K_p_01', 'K_i_01', 'K_g_01', 'R_v_01', 'X_v_01', 'K_q_01', 'T_q_01', 'K_i_q_01', 'S_n_03', 'F_n_03', 'K_delta_03', 'K_p_03', 'K_i_03', 'K_g_03', 'R_v_03', 'X_v_03', 'K_q_03', 'T_q_03', 'K_i_q_03', 'S_n_05', 'F_n_05', 'K_delta_05', 'K_p_05', 'K_i_05', 'K_g_05', 'R_v_05', 'X_v_05', 'K_q_05', 'T_q_05', 'K_i_q_05', 'S_n_07', 'F_n_07', 'K_delta_07', 'K_p_07', 'K_i_07', 'K_g_07', 'R_v_07', 'X_v_07', 'K_q_07', 'T_q_07', 'K_i_q_07', 'S_n_09', 'F_n_09', 'K_delta_09', 'K_p_09', 'K_i_09', 'K_g_09', 'R_v_09', 'X_v_09', 'K_q_09', 'T_q_09', 'K_i_q_09', 'S_n_11', 'F_n_11', 'K_delta_11', 'K_p_11', 'K_i_11', 'K_g_11', 'R_v_11', 'X_v_11', 'K_q_11', 'T_q_11', 'K_i_q_11', 'S_n_13', 'F_n_13', 'K_delta_13', 'K_p_13', 'K_i_13', 'K_g_13', 'R_v_13', 'X_v_13', 'K_q_13', 'T_q_13', 'K_i_q_13', 'S_n_15', 'F_n_15', 'K_delta_15', 'K_p_15', 'K_i_15', 'K_g_15', 'R_v_15', 'X_v_15', 'K_q_15', 'T_q_15', 'K_i_q_15', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [100000000.0, 0.0, -3.3967391304347823, 0.0, 0.0, -3.3967391304347823, 0.0, 0.0, -3.3967391304347823, 0.0, 0.0, -3.3967391304347823, 0.0, 3.694435687263556, -4.581100252206809, 0.0, 3.694435687263556, -4.581100252206809, 0.0, 3.694435687263556, -4.581100252206809, 0.0, 3.694435687263556, -4.581100252206809, 0.0, 0.0, -4.528985507246377, 0.0, 0.0, -4.528985507246377, 0.0, 0.0, -4.528985507246377, 0.0, 0.0, -4.528985507246377, 0.0, 4.925914249684741, -6.108133669609079, 0.0, 4.925914249684741, -6.108133669609079, 0.0, 4.925914249684741, -6.108133669609079, 0.0, 4.925914249684741, -6.108133669609079, 0.0, 0.0, -14.583333333333334, 0.0, 0.6930693069306932, -6.9306930693069315, 0.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 690.0, 500000000.0, 314.1592653589793, 0.01, 0.1, 0.01, 0.1, 0.1, 0.01, 0.001, 0.0, 0.05, 0.1, 0.1, 18750000.0, 50, 0.0, 0.0021850968611841584, 0.125, 0.0, 0.0, 0.3, 9.999999999999991, 0.03183098861837907, 1e-06, 18750000.0, 50, 0.0, 0.0021850968611841584, 0.125, 0.0, 0.0, 0.3, 9.999999999999991, 0.03183098861837907, 1e-06, 18750000.0, 50, 0.0, 0.0021850968611841584, 0.125, 0.0, 0.0, 0.3, 9.999999999999991, 0.03183098861837907, 1e-06, 18750000.0, 50, 0.0, 0.0021850968611841584, 0.125, 0.0, 0.0, 0.3, 9.999999999999991, 0.03183098861837907, 1e-06, 25000000.0, 50, 0.0, 0.0021850968611841584, 0.125, 0.0, 0.0, 0.3, 9.999999999999991, 0.03183098861837907, 1e-06, 25000000.0, 50, 0.0, 0.0021850968611841584, 0.125, 0.0, 0.0, 0.3, 9.999999999999991, 0.03183098861837907, 1e-06, 25000000.0, 50, 0.0, 0.0021850968611841584, 0.125, 0.0, 0.0, 0.3, 9.999999999999991, 0.03183098861837907, 1e-06, 25000000.0, 50, 0.0, 0.0021850968611841584, 0.125, 0.0, 0.0, 0.3, 9.999999999999991, 0.03183098861837907, 1e-06, 0.01, 0.01] 
        self.inputs_ini_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'P_13', 'Q_13', 'P_14', 'Q_14', 'P_15', 'Q_15', 'P_16', 'Q_16', 'P_17', 'Q_17', 'P_18', 'Q_18', 'P_19', 'Q_19', 'v_ref_19', 'p_c_19', 'omega_ref_19', 'q_ref_19', 'p_ref_01', 'q_ref_01', 'p_ref_03', 'q_ref_03', 'p_ref_05', 'q_ref_05', 'p_ref_07', 'q_ref_07', 'p_ref_09', 'q_ref_09', 'p_ref_11', 'q_ref_11', 'p_ref_13', 'q_ref_13', 'p_ref_15', 'q_ref_15'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.5, -0.05, 0.0, 0.0, 1.02, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.inputs_run_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'P_13', 'Q_13', 'P_14', 'Q_14', 'P_15', 'Q_15', 'P_16', 'Q_16', 'P_17', 'Q_17', 'P_18', 'Q_18', 'P_19', 'Q_19', 'v_ref_19', 'p_c_19', 'omega_ref_19', 'q_ref_19', 'p_ref_01', 'q_ref_01', 'p_ref_03', 'q_ref_03', 'p_ref_05', 'q_ref_05', 'p_ref_07', 'q_ref_07', 'p_ref_09', 'q_ref_09', 'p_ref_11', 'q_ref_11', 'p_ref_13', 'q_ref_13', 'p_ref_15', 'q_ref_15'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.5, -0.05, 0.0, 0.0, 1.02, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
        self.outputs_list = ['V_01', 'V_02', 'V_03', 'V_04', 'V_05', 'V_06', 'V_07', 'V_08', 'V_09', 'V_10', 'V_11', 'V_12', 'V_13', 'V_14', 'V_15', 'V_16', 'V_17', 'V_18', 'V_19', 'p_line_01_02', 'q_line_01_02', 'p_line_03_04', 'q_line_03_04', 'p_line_05_06', 'q_line_05_06', 'p_line_07_08', 'q_line_07_08', 'p_line_02_04', 'q_line_02_04', 'p_line_04_06', 'q_line_04_06', 'p_line_06_08', 'q_line_06_08', 'p_line_08_17', 'q_line_08_17', 'p_line_09_10', 'q_line_09_10', 'p_line_11_12', 'q_line_11_12', 'p_line_13_14', 'q_line_13_14', 'p_line_15_16', 'q_line_15_16', 'p_line_10_12', 'q_line_10_12', 'p_line_12_14', 'q_line_12_14', 'p_line_14_16', 'q_line_14_16', 'p_line_16_17', 'q_line_16_17', 'p_line_17_18', 'q_line_17_18', 'p_line_18_19', 'q_line_18_19', 'p_e_19', 'omega_v_01', 'omega_v_03', 'omega_v_05', 'omega_v_07', 'omega_v_09', 'omega_v_11', 'omega_v_13', 'omega_v_15'] 
        self.x_list = ['delta_19', 'xi_p_19', 'e_qv_19', 'p_ef_19', 'p_cf_19', 'delta_01', 'x_v_01', 'e_qm_01', 'xi_q_01', 'delta_03', 'x_v_03', 'e_qm_03', 'xi_q_03', 'delta_05', 'x_v_05', 'e_qm_05', 'xi_q_05', 'delta_07', 'x_v_07', 'e_qm_07', 'xi_q_07', 'delta_09', 'x_v_09', 'e_qm_09', 'xi_q_09', 'delta_11', 'x_v_11', 'e_qm_11', 'xi_q_11', 'delta_13', 'x_v_13', 'e_qm_13', 'xi_q_13', 'delta_15', 'x_v_15', 'e_qm_15', 'xi_q_15', 'xi_freq'] 
        self.y_run_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'V_13', 'theta_13', 'V_14', 'theta_14', 'V_15', 'theta_15', 'V_16', 'theta_16', 'V_17', 'theta_17', 'V_18', 'theta_18', 'V_19', 'theta_19', 'omega_19', 'i_d_19', 'i_q_19', 'p_g_19', 'q_g_19', 'p_m_19', 'i_d_ref_01', 'i_q_ref_01', 'p_01', 'q_01', 'e_qv_01', 'i_d_ref_03', 'i_q_ref_03', 'p_03', 'q_03', 'e_qv_03', 'i_d_ref_05', 'i_q_ref_05', 'p_05', 'q_05', 'e_qv_05', 'i_d_ref_07', 'i_q_ref_07', 'p_07', 'q_07', 'e_qv_07', 'i_d_ref_09', 'i_q_ref_09', 'p_09', 'q_09', 'e_qv_09', 'i_d_ref_11', 'i_q_ref_11', 'p_11', 'q_11', 'e_qv_11', 'i_d_ref_13', 'i_q_ref_13', 'p_13', 'q_13', 'e_qv_13', 'i_d_ref_15', 'i_q_ref_15', 'p_15', 'q_15', 'e_qv_15', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'V_13', 'theta_13', 'V_14', 'theta_14', 'V_15', 'theta_15', 'V_16', 'theta_16', 'V_17', 'theta_17', 'V_18', 'theta_18', 'V_19', 'theta_19', 'omega_19', 'i_d_19', 'i_q_19', 'p_g_19', 'q_g_19', 'p_m_19', 'i_d_ref_01', 'i_q_ref_01', 'p_01', 'q_01', 'e_qv_01', 'i_d_ref_03', 'i_q_ref_03', 'p_03', 'q_03', 'e_qv_03', 'i_d_ref_05', 'i_q_ref_05', 'p_05', 'q_05', 'e_qv_05', 'i_d_ref_07', 'i_q_ref_07', 'p_07', 'q_07', 'e_qv_07', 'i_d_ref_09', 'i_q_ref_09', 'p_09', 'q_09', 'e_qv_09', 'i_d_ref_11', 'i_q_ref_11', 'p_11', 'q_11', 'e_qv_11', 'i_d_ref_13', 'i_q_ref_13', 'p_13', 'q_13', 'e_qv_13', 'i_d_ref_15', 'i_q_ref_15', 'p_15', 'q_15', 'e_qv_15', 'omega_coi', 'p_agc'] 
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
           
        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './wf19_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz('./wf19_sp_jac_ini_num.npz')
            
            
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
            fobj = BytesIO(pkgutil.get_data(__name__, './wf19_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz('./wf19_sp_jac_run_num.npz')
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
            fobj = BytesIO(pkgutil.get_data(__name__, './wf19_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz('./wf19_sp_jac_trap_num.npz')
            

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

        #self.sp_Fu_run = sspa.load_npz('./wf19_Fu_run_num.npz')
        #self.sp_Gu_run = sspa.load_npz('./wf19_Gu_run_num.npz')
        #self.sp_Hx_run = sspa.load_npz('./wf19_Hx_run_num.npz')
        #self.sp_Hy_run = sspa.load_npz('./wf19_Hy_run_num.npz')
        #self.sp_Hu_run = sspa.load_npz('./wf19_Hu_run_num.npz')        
        
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

    sp_jac_ini_ia = [0, 76, 122, 3, 81, 2, 80, 0, 3, 74, 75, 77, 78, 4, 5, 6, 84, 122, 6, 84, 7, 85, 8, 9, 10, 89, 122, 10, 89, 11, 90, 12, 13, 14, 94, 122, 14, 94, 15, 95, 16, 17, 18, 99, 122, 18, 99, 19, 100, 20, 21, 22, 104, 122, 22, 104, 23, 105, 24, 25, 26, 109, 122, 26, 109, 27, 110, 28, 29, 30, 114, 122, 30, 114, 31, 115, 32, 33, 34, 119, 122, 34, 119, 35, 120, 36, 122, 38, 39, 40, 41, 84, 38, 39, 40, 41, 85, 38, 39, 40, 41, 44, 45, 38, 39, 40, 41, 44, 45, 42, 43, 44, 45, 89, 42, 43, 44, 45, 90, 40, 41, 42, 43, 44, 45, 48, 49, 40, 41, 42, 43, 44, 45, 48, 49, 46, 47, 48, 49, 94, 46, 47, 48, 49, 95, 44, 45, 46, 47, 48, 49, 52, 53, 44, 45, 46, 47, 48, 49, 52, 53, 50, 51, 52, 53, 99, 50, 51, 52, 53, 100, 48, 49, 50, 51, 52, 53, 70, 71, 48, 49, 50, 51, 52, 53, 70, 71, 54, 55, 56, 57, 104, 54, 55, 56, 57, 105, 54, 55, 56, 57, 60, 61, 54, 55, 56, 57, 60, 61, 58, 59, 60, 61, 109, 58, 59, 60, 61, 110, 56, 57, 58, 59, 60, 61, 64, 65, 56, 57, 58, 59, 60, 61, 64, 65, 62, 63, 64, 65, 114, 62, 63, 64, 65, 115, 60, 61, 62, 63, 64, 65, 68, 69, 60, 61, 62, 63, 64, 65, 68, 69, 66, 67, 68, 69, 119, 66, 67, 68, 69, 120, 64, 65, 66, 67, 68, 69, 70, 71, 64, 65, 66, 67, 68, 69, 70, 71, 52, 53, 68, 69, 70, 71, 72, 73, 52, 53, 68, 69, 70, 71, 72, 73, 70, 71, 72, 73, 74, 75, 70, 71, 72, 73, 74, 75, 72, 73, 74, 75, 79, 72, 73, 74, 75, 80, 1, 3, 76, 81, 0, 74, 75, 77, 78, 0, 2, 74, 75, 77, 78, 0, 74, 75, 77, 78, 79, 0, 74, 75, 77, 78, 80, 4, 76, 81, 123, 5, 38, 39, 82, 83, 5, 38, 39, 82, 83, 86, 5, 38, 39, 82, 83, 84, 5, 38, 39, 82, 83, 85, 7, 86, 9, 42, 43, 87, 88, 9, 42, 43, 87, 88, 91, 9, 42, 43, 87, 88, 89, 9, 42, 43, 87, 88, 90, 11, 91, 13, 46, 47, 92, 93, 13, 46, 47, 92, 93, 96, 13, 46, 47, 92, 93, 94, 13, 46, 47, 92, 93, 95, 15, 96, 17, 50, 51, 97, 98, 17, 50, 51, 97, 98, 101, 17, 50, 51, 97, 98, 99, 17, 50, 51, 97, 98, 100, 19, 101, 21, 54, 55, 102, 103, 21, 54, 55, 102, 103, 106, 21, 54, 55, 102, 103, 104, 21, 54, 55, 102, 103, 105, 23, 106, 25, 58, 59, 107, 108, 25, 58, 59, 107, 108, 111, 25, 58, 59, 107, 108, 109, 25, 58, 59, 107, 108, 110, 27, 111, 29, 62, 63, 112, 113, 29, 62, 63, 112, 113, 116, 29, 62, 63, 112, 113, 114, 29, 62, 63, 112, 113, 115, 31, 116, 33, 66, 67, 117, 118, 33, 66, 67, 117, 118, 121, 33, 66, 67, 117, 118, 119, 33, 66, 67, 117, 118, 120, 35, 121, 6, 10, 14, 18, 22, 26, 30, 34, 76, 84, 89, 94, 99, 104, 109, 114, 119, 122, 37, 122, 123]
    sp_jac_ini_ja = [0, 3, 5, 7, 13, 14, 18, 20, 22, 23, 27, 29, 31, 32, 36, 38, 40, 41, 45, 47, 49, 50, 54, 56, 58, 59, 63, 65, 67, 68, 72, 74, 76, 77, 81, 83, 85, 86, 87, 92, 97, 103, 109, 114, 119, 127, 135, 140, 145, 153, 161, 166, 171, 179, 187, 192, 197, 203, 209, 214, 219, 227, 235, 240, 245, 253, 261, 266, 271, 279, 287, 295, 303, 309, 315, 320, 325, 329, 334, 340, 346, 352, 356, 361, 367, 373, 379, 381, 386, 392, 398, 404, 406, 411, 417, 423, 429, 431, 436, 442, 448, 454, 456, 461, 467, 473, 479, 481, 486, 492, 498, 504, 506, 511, 517, 523, 529, 531, 536, 542, 548, 554, 556, 574, 577]
    sp_jac_ini_nia = 124
    sp_jac_ini_nja = 124
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 76, 122, 3, 81, 2, 80, 0, 3, 74, 75, 77, 78, 4, 5, 6, 84, 122, 6, 84, 7, 85, 8, 9, 10, 89, 122, 10, 89, 11, 90, 12, 13, 14, 94, 122, 14, 94, 15, 95, 16, 17, 18, 99, 122, 18, 99, 19, 100, 20, 21, 22, 104, 122, 22, 104, 23, 105, 24, 25, 26, 109, 122, 26, 109, 27, 110, 28, 29, 30, 114, 122, 30, 114, 31, 115, 32, 33, 34, 119, 122, 34, 119, 35, 120, 36, 122, 38, 39, 40, 41, 84, 38, 39, 40, 41, 85, 38, 39, 40, 41, 44, 45, 38, 39, 40, 41, 44, 45, 42, 43, 44, 45, 89, 42, 43, 44, 45, 90, 40, 41, 42, 43, 44, 45, 48, 49, 40, 41, 42, 43, 44, 45, 48, 49, 46, 47, 48, 49, 94, 46, 47, 48, 49, 95, 44, 45, 46, 47, 48, 49, 52, 53, 44, 45, 46, 47, 48, 49, 52, 53, 50, 51, 52, 53, 99, 50, 51, 52, 53, 100, 48, 49, 50, 51, 52, 53, 70, 71, 48, 49, 50, 51, 52, 53, 70, 71, 54, 55, 56, 57, 104, 54, 55, 56, 57, 105, 54, 55, 56, 57, 60, 61, 54, 55, 56, 57, 60, 61, 58, 59, 60, 61, 109, 58, 59, 60, 61, 110, 56, 57, 58, 59, 60, 61, 64, 65, 56, 57, 58, 59, 60, 61, 64, 65, 62, 63, 64, 65, 114, 62, 63, 64, 65, 115, 60, 61, 62, 63, 64, 65, 68, 69, 60, 61, 62, 63, 64, 65, 68, 69, 66, 67, 68, 69, 119, 66, 67, 68, 69, 120, 64, 65, 66, 67, 68, 69, 70, 71, 64, 65, 66, 67, 68, 69, 70, 71, 52, 53, 68, 69, 70, 71, 72, 73, 52, 53, 68, 69, 70, 71, 72, 73, 70, 71, 72, 73, 74, 75, 70, 71, 72, 73, 74, 75, 72, 73, 74, 75, 79, 72, 73, 74, 75, 80, 1, 3, 76, 81, 0, 74, 75, 77, 78, 0, 2, 74, 75, 77, 78, 0, 74, 75, 77, 78, 79, 0, 74, 75, 77, 78, 80, 4, 76, 81, 123, 5, 38, 39, 82, 83, 5, 38, 39, 82, 83, 86, 5, 38, 39, 82, 83, 84, 5, 38, 39, 82, 83, 85, 7, 86, 9, 42, 43, 87, 88, 9, 42, 43, 87, 88, 91, 9, 42, 43, 87, 88, 89, 9, 42, 43, 87, 88, 90, 11, 91, 13, 46, 47, 92, 93, 13, 46, 47, 92, 93, 96, 13, 46, 47, 92, 93, 94, 13, 46, 47, 92, 93, 95, 15, 96, 17, 50, 51, 97, 98, 17, 50, 51, 97, 98, 101, 17, 50, 51, 97, 98, 99, 17, 50, 51, 97, 98, 100, 19, 101, 21, 54, 55, 102, 103, 21, 54, 55, 102, 103, 106, 21, 54, 55, 102, 103, 104, 21, 54, 55, 102, 103, 105, 23, 106, 25, 58, 59, 107, 108, 25, 58, 59, 107, 108, 111, 25, 58, 59, 107, 108, 109, 25, 58, 59, 107, 108, 110, 27, 111, 29, 62, 63, 112, 113, 29, 62, 63, 112, 113, 116, 29, 62, 63, 112, 113, 114, 29, 62, 63, 112, 113, 115, 31, 116, 33, 66, 67, 117, 118, 33, 66, 67, 117, 118, 121, 33, 66, 67, 117, 118, 119, 33, 66, 67, 117, 118, 120, 35, 121, 6, 10, 14, 18, 22, 26, 30, 34, 76, 84, 89, 94, 99, 104, 109, 114, 119, 122, 37, 122, 123]
    sp_jac_run_ja = [0, 3, 5, 7, 13, 14, 18, 20, 22, 23, 27, 29, 31, 32, 36, 38, 40, 41, 45, 47, 49, 50, 54, 56, 58, 59, 63, 65, 67, 68, 72, 74, 76, 77, 81, 83, 85, 86, 87, 92, 97, 103, 109, 114, 119, 127, 135, 140, 145, 153, 161, 166, 171, 179, 187, 192, 197, 203, 209, 214, 219, 227, 235, 240, 245, 253, 261, 266, 271, 279, 287, 295, 303, 309, 315, 320, 325, 329, 334, 340, 346, 352, 356, 361, 367, 373, 379, 381, 386, 392, 398, 404, 406, 411, 417, 423, 429, 431, 436, 442, 448, 454, 456, 461, 467, 473, 479, 481, 486, 492, 498, 504, 506, 511, 517, 523, 529, 531, 536, 542, 548, 554, 556, 574, 577]
    sp_jac_run_nia = 124
    sp_jac_run_nja = 124
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 76, 122, 1, 3, 81, 2, 80, 0, 3, 74, 75, 77, 78, 4, 5, 6, 84, 122, 6, 84, 7, 85, 8, 9, 10, 89, 122, 10, 89, 11, 90, 12, 13, 14, 94, 122, 14, 94, 15, 95, 16, 17, 18, 99, 122, 18, 99, 19, 100, 20, 21, 22, 104, 122, 22, 104, 23, 105, 24, 25, 26, 109, 122, 26, 109, 27, 110, 28, 29, 30, 114, 122, 30, 114, 31, 115, 32, 33, 34, 119, 122, 34, 119, 35, 120, 36, 37, 122, 38, 39, 40, 41, 84, 38, 39, 40, 41, 85, 38, 39, 40, 41, 44, 45, 38, 39, 40, 41, 44, 45, 42, 43, 44, 45, 89, 42, 43, 44, 45, 90, 40, 41, 42, 43, 44, 45, 48, 49, 40, 41, 42, 43, 44, 45, 48, 49, 46, 47, 48, 49, 94, 46, 47, 48, 49, 95, 44, 45, 46, 47, 48, 49, 52, 53, 44, 45, 46, 47, 48, 49, 52, 53, 50, 51, 52, 53, 99, 50, 51, 52, 53, 100, 48, 49, 50, 51, 52, 53, 70, 71, 48, 49, 50, 51, 52, 53, 70, 71, 54, 55, 56, 57, 104, 54, 55, 56, 57, 105, 54, 55, 56, 57, 60, 61, 54, 55, 56, 57, 60, 61, 58, 59, 60, 61, 109, 58, 59, 60, 61, 110, 56, 57, 58, 59, 60, 61, 64, 65, 56, 57, 58, 59, 60, 61, 64, 65, 62, 63, 64, 65, 114, 62, 63, 64, 65, 115, 60, 61, 62, 63, 64, 65, 68, 69, 60, 61, 62, 63, 64, 65, 68, 69, 66, 67, 68, 69, 119, 66, 67, 68, 69, 120, 64, 65, 66, 67, 68, 69, 70, 71, 64, 65, 66, 67, 68, 69, 70, 71, 52, 53, 68, 69, 70, 71, 72, 73, 52, 53, 68, 69, 70, 71, 72, 73, 70, 71, 72, 73, 74, 75, 70, 71, 72, 73, 74, 75, 72, 73, 74, 75, 79, 72, 73, 74, 75, 80, 1, 3, 76, 81, 0, 74, 75, 77, 78, 0, 2, 74, 75, 77, 78, 0, 74, 75, 77, 78, 79, 0, 74, 75, 77, 78, 80, 4, 76, 81, 123, 5, 38, 39, 82, 83, 5, 38, 39, 82, 83, 86, 5, 38, 39, 82, 83, 84, 5, 38, 39, 82, 83, 85, 7, 86, 9, 42, 43, 87, 88, 9, 42, 43, 87, 88, 91, 9, 42, 43, 87, 88, 89, 9, 42, 43, 87, 88, 90, 11, 91, 13, 46, 47, 92, 93, 13, 46, 47, 92, 93, 96, 13, 46, 47, 92, 93, 94, 13, 46, 47, 92, 93, 95, 15, 96, 17, 50, 51, 97, 98, 17, 50, 51, 97, 98, 101, 17, 50, 51, 97, 98, 99, 17, 50, 51, 97, 98, 100, 19, 101, 21, 54, 55, 102, 103, 21, 54, 55, 102, 103, 106, 21, 54, 55, 102, 103, 104, 21, 54, 55, 102, 103, 105, 23, 106, 25, 58, 59, 107, 108, 25, 58, 59, 107, 108, 111, 25, 58, 59, 107, 108, 109, 25, 58, 59, 107, 108, 110, 27, 111, 29, 62, 63, 112, 113, 29, 62, 63, 112, 113, 116, 29, 62, 63, 112, 113, 114, 29, 62, 63, 112, 113, 115, 31, 116, 33, 66, 67, 117, 118, 33, 66, 67, 117, 118, 121, 33, 66, 67, 117, 118, 119, 33, 66, 67, 117, 118, 120, 35, 121, 6, 10, 14, 18, 22, 26, 30, 34, 76, 84, 89, 94, 99, 104, 109, 114, 119, 122, 37, 122, 123]
    sp_jac_trap_ja = [0, 3, 6, 8, 14, 15, 19, 21, 23, 24, 28, 30, 32, 33, 37, 39, 41, 42, 46, 48, 50, 51, 55, 57, 59, 60, 64, 66, 68, 69, 73, 75, 77, 78, 82, 84, 86, 87, 89, 94, 99, 105, 111, 116, 121, 129, 137, 142, 147, 155, 163, 168, 173, 181, 189, 194, 199, 205, 211, 216, 221, 229, 237, 242, 247, 255, 263, 268, 273, 281, 289, 297, 305, 311, 317, 322, 327, 331, 336, 342, 348, 354, 358, 363, 369, 375, 381, 383, 388, 394, 400, 406, 408, 413, 419, 425, 431, 433, 438, 444, 450, 456, 458, 463, 469, 475, 481, 483, 488, 494, 500, 506, 508, 513, 519, 525, 531, 533, 538, 544, 550, 556, 558, 576, 579]
    sp_jac_trap_nia = 124
    sp_jac_trap_nja = 124
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
