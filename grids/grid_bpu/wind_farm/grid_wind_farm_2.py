import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 


class grid_wind_farm_2_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 24
        self.N_y = 130 
        self.N_z = 0 
        self.N_store = 10000 
        self.params_list = ['u_ctrl_v_W1lv', 'K_p_v_W1lv', 'K_i_v_W1lv', 'V_base_W1lv', 'V_base_W1mv', 'S_base_W1lv', 'I_max_W1lv', 'u_ctrl_v_W2lv', 'K_p_v_W2lv', 'K_i_v_W2lv', 'V_base_W2lv', 'V_base_W2mv', 'S_base_W2lv', 'I_max_W2lv', 'u_ctrl_v_W3lv', 'K_p_v_W3lv', 'K_i_v_W3lv', 'V_base_W3lv', 'V_base_W3mv', 'S_base_W3lv', 'I_max_W3lv', 'u_ctrl_v_STlv', 'K_p_v_STlv', 'K_i_v_STlv', 'V_base_STlv', 'V_base_STmv', 'S_base_STlv', 'I_max_STlv'] 
        self.params_values_list  = [0.0, 0.1, 0.1, 400, 11547.005383792515, 2000000.0, 0.5, 0.0, 0.1, 0.1, 400, 11547.005383792515, 2000000.0, 0.5, 0.0, 0.1, 0.1, 400, 11547.005383792515, 2000000.0, 0.5, 0.0, 0.1, 0.1, 400, 11547.005383792515, 2000000.0, 0.5] 
        self.inputs_ini_list = ['v_GRID_a_r', 'v_GRID_a_i', 'v_GRID_b_r', 'v_GRID_b_i', 'v_GRID_c_r', 'v_GRID_c_i', 'i_POI_a_r', 'i_POI_a_i', 'i_POI_b_r', 'i_POI_b_i', 'i_POI_c_r', 'i_POI_c_i', 'i_POImv_a_r', 'i_POImv_a_i', 'i_POImv_b_r', 'i_POImv_b_i', 'i_POImv_c_r', 'i_POImv_c_i', 'i_W1mv_a_r', 'i_W1mv_a_i', 'i_W1mv_b_r', 'i_W1mv_b_i', 'i_W1mv_c_r', 'i_W1mv_c_i', 'i_W2mv_a_r', 'i_W2mv_a_i', 'i_W2mv_b_r', 'i_W2mv_b_i', 'i_W2mv_c_r', 'i_W2mv_c_i', 'i_W3mv_a_r', 'i_W3mv_a_i', 'i_W3mv_b_r', 'i_W3mv_b_i', 'i_W3mv_c_r', 'i_W3mv_c_i', 'i_STmv_a_r', 'i_STmv_a_i', 'i_STmv_b_r', 'i_STmv_b_i', 'i_STmv_c_r', 'i_STmv_c_i', 'p_ref_W1lv', 'T_pq_W1lv', 'v_loc_ref_W1lv', 'Dv_r_W1lv', 'Dq_r_W1lv', 'p_ref_W2lv', 'T_pq_W2lv', 'v_loc_ref_W2lv', 'Dv_r_W2lv', 'Dq_r_W2lv', 'p_ref_W3lv', 'T_pq_W3lv', 'v_loc_ref_W3lv', 'Dv_r_W3lv', 'Dq_r_W3lv', 'p_ref_STlv', 'T_pq_STlv', 'v_loc_ref_STlv', 'Dv_r_STlv', 'Dq_r_STlv'] 
        self.inputs_ini_values_list  = [32999.89801120604, 19052.499999999996, -32999.89801120604, 19052.499999999996, -6.999774942226484e-12, -38105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.630061867937911e-07, 1.523679202364292e-06, 0.0, 0.0, 0.0, 0.0, 2000000.0025185598, 0.2, 1, 0, 0, 2000000.0024943855, 0.2, 1, 0, 0, 2000000.0024460733, 0.2, 1, 0, 0, 0.0, 0.2, 1, 0, 0] 
        self.inputs_run_list = ['v_GRID_a_r', 'v_GRID_a_i', 'v_GRID_b_r', 'v_GRID_b_i', 'v_GRID_c_r', 'v_GRID_c_i', 'i_POI_a_r', 'i_POI_a_i', 'i_POI_b_r', 'i_POI_b_i', 'i_POI_c_r', 'i_POI_c_i', 'i_POImv_a_r', 'i_POImv_a_i', 'i_POImv_b_r', 'i_POImv_b_i', 'i_POImv_c_r', 'i_POImv_c_i', 'i_W1mv_a_r', 'i_W1mv_a_i', 'i_W1mv_b_r', 'i_W1mv_b_i', 'i_W1mv_c_r', 'i_W1mv_c_i', 'i_W2mv_a_r', 'i_W2mv_a_i', 'i_W2mv_b_r', 'i_W2mv_b_i', 'i_W2mv_c_r', 'i_W2mv_c_i', 'i_W3mv_a_r', 'i_W3mv_a_i', 'i_W3mv_b_r', 'i_W3mv_b_i', 'i_W3mv_c_r', 'i_W3mv_c_i', 'i_STmv_a_r', 'i_STmv_a_i', 'i_STmv_b_r', 'i_STmv_b_i', 'i_STmv_c_r', 'i_STmv_c_i', 'p_ref_W1lv', 'T_pq_W1lv', 'v_loc_ref_W1lv', 'Dv_r_W1lv', 'Dq_r_W1lv', 'p_ref_W2lv', 'T_pq_W2lv', 'v_loc_ref_W2lv', 'Dv_r_W2lv', 'Dq_r_W2lv', 'p_ref_W3lv', 'T_pq_W3lv', 'v_loc_ref_W3lv', 'Dv_r_W3lv', 'Dq_r_W3lv', 'p_ref_STlv', 'T_pq_STlv', 'v_loc_ref_STlv', 'Dv_r_STlv', 'Dq_r_STlv'] 
        self.inputs_run_values_list = [32999.89801120604, 19052.499999999996, -32999.89801120604, 19052.499999999996, -6.999774942226484e-12, -38105.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.630061867937911e-07, 1.523679202364292e-06, 0.0, 0.0, 0.0, 0.0, 2000000.0025185598, 0.2, 1, 0, 0, 2000000.0024943855, 0.2, 1, 0, 0, 2000000.0024460733, 0.2, 1, 0, 0, 0.0, 0.2, 1, 0, 0] 
        self.outputs_list = ['v_GRID_a_m', 'v_GRID_b_m', 'v_GRID_c_m', 'v_W1lv_a_m', 'v_W1lv_b_m', 'v_W1lv_c_m', 'v_W2lv_a_m', 'v_W2lv_b_m', 'v_W2lv_c_m', 'v_W3lv_a_m', 'v_W3lv_b_m', 'v_W3lv_c_m', 'v_STlv_a_m', 'v_STlv_b_m', 'v_STlv_c_m', 'v_POI_a_m', 'v_POI_b_m', 'v_POI_c_m', 'v_POImv_a_m', 'v_POImv_b_m', 'v_POImv_c_m', 'v_W1mv_a_m', 'v_W1mv_b_m', 'v_W1mv_c_m', 'v_W2mv_a_m', 'v_W2mv_b_m', 'v_W2mv_c_m', 'v_W3mv_a_m', 'v_W3mv_b_m', 'v_W3mv_c_m', 'v_STmv_a_m', 'v_STmv_b_m', 'v_STmv_c_m'] 
        self.x_list = ['p_W1lv_a', 'p_W1lv_b', 'p_W1lv_c', 'q_W1lv_a', 'q_W1lv_b', 'q_W1lv_c', 'p_W2lv_a', 'p_W2lv_b', 'p_W2lv_c', 'q_W2lv_a', 'q_W2lv_b', 'q_W2lv_c', 'p_W3lv_a', 'p_W3lv_b', 'p_W3lv_c', 'q_W3lv_a', 'q_W3lv_b', 'q_W3lv_c', 'p_STlv_a', 'p_STlv_b', 'p_STlv_c', 'q_STlv_a', 'q_STlv_b', 'q_STlv_c'] 
        self.y_run_list = ['v_W1lv_a_r', 'v_W1lv_a_i', 'v_W1lv_b_r', 'v_W1lv_b_i', 'v_W1lv_c_r', 'v_W1lv_c_i', 'v_W2lv_a_r', 'v_W2lv_a_i', 'v_W2lv_b_r', 'v_W2lv_b_i', 'v_W2lv_c_r', 'v_W2lv_c_i', 'v_W3lv_a_r', 'v_W3lv_a_i', 'v_W3lv_b_r', 'v_W3lv_b_i', 'v_W3lv_c_r', 'v_W3lv_c_i', 'v_STlv_a_r', 'v_STlv_a_i', 'v_STlv_b_r', 'v_STlv_b_i', 'v_STlv_c_r', 'v_STlv_c_i', 'v_POI_a_r', 'v_POI_a_i', 'v_POI_b_r', 'v_POI_b_i', 'v_POI_c_r', 'v_POI_c_i', 'v_POImv_a_r', 'v_POImv_a_i', 'v_POImv_b_r', 'v_POImv_b_i', 'v_POImv_c_r', 'v_POImv_c_i', 'v_W1mv_a_r', 'v_W1mv_a_i', 'v_W1mv_b_r', 'v_W1mv_b_i', 'v_W1mv_c_r', 'v_W1mv_c_i', 'v_W2mv_a_r', 'v_W2mv_a_i', 'v_W2mv_b_r', 'v_W2mv_b_i', 'v_W2mv_c_r', 'v_W2mv_c_i', 'v_W3mv_a_r', 'v_W3mv_a_i', 'v_W3mv_b_r', 'v_W3mv_b_i', 'v_W3mv_c_r', 'v_W3mv_c_i', 'v_STmv_a_r', 'v_STmv_a_i', 'v_STmv_b_r', 'v_STmv_b_i', 'v_STmv_c_r', 'v_STmv_c_i', 'i_l_W1mv_W2mv_a_r', 'i_l_W1mv_W2mv_a_i', 'i_l_W1mv_W2mv_b_r', 'i_l_W1mv_W2mv_b_i', 'i_l_W1mv_W2mv_c_r', 'i_l_W1mv_W2mv_c_i', 'i_l_W2mv_W3mv_a_r', 'i_l_W2mv_W3mv_a_i', 'i_l_W2mv_W3mv_b_r', 'i_l_W2mv_W3mv_b_i', 'i_l_W2mv_W3mv_c_r', 'i_l_W2mv_W3mv_c_i', 'i_l_W3mv_POImv_a_r', 'i_l_W3mv_POImv_a_i', 'i_l_W3mv_POImv_b_r', 'i_l_W3mv_POImv_b_i', 'i_l_W3mv_POImv_c_r', 'i_l_W3mv_POImv_c_i', 'i_l_STmv_POImv_a_r', 'i_l_STmv_POImv_a_i', 'i_l_STmv_POImv_b_r', 'i_l_STmv_POImv_b_i', 'i_l_STmv_POImv_c_r', 'i_l_STmv_POImv_c_i', 'i_l_POI_GRID_a_r', 'i_l_POI_GRID_a_i', 'i_l_POI_GRID_b_r', 'i_l_POI_GRID_b_i', 'i_l_POI_GRID_c_r', 'i_l_POI_GRID_c_i', 'i_W1lv_a_r', 'i_W1lv_a_i', 'i_W1lv_b_r', 'i_W1lv_b_i', 'i_W1lv_c_r', 'i_W1lv_c_i', 'v_m_W1lv', 'v_m_W1mv', 'i_reac_ref_W1lv', 'q_ref_W1lv', 'i_W2lv_a_r', 'i_W2lv_a_i', 'i_W2lv_b_r', 'i_W2lv_b_i', 'i_W2lv_c_r', 'i_W2lv_c_i', 'v_m_W2lv', 'v_m_W2mv', 'i_reac_ref_W2lv', 'q_ref_W2lv', 'i_W3lv_a_r', 'i_W3lv_a_i', 'i_W3lv_b_r', 'i_W3lv_b_i', 'i_W3lv_c_r', 'i_W3lv_c_i', 'v_m_W3lv', 'v_m_W3mv', 'i_reac_ref_W3lv', 'q_ref_W3lv', 'i_STlv_a_r', 'i_STlv_a_i', 'i_STlv_b_r', 'i_STlv_b_i', 'i_STlv_c_r', 'i_STlv_c_i', 'v_m_STlv', 'v_m_STmv', 'i_reac_ref_STlv', 'q_ref_STlv'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_W1lv_a_r', 'v_W1lv_a_i', 'v_W1lv_b_r', 'v_W1lv_b_i', 'v_W1lv_c_r', 'v_W1lv_c_i', 'v_W2lv_a_r', 'v_W2lv_a_i', 'v_W2lv_b_r', 'v_W2lv_b_i', 'v_W2lv_c_r', 'v_W2lv_c_i', 'v_W3lv_a_r', 'v_W3lv_a_i', 'v_W3lv_b_r', 'v_W3lv_b_i', 'v_W3lv_c_r', 'v_W3lv_c_i', 'v_STlv_a_r', 'v_STlv_a_i', 'v_STlv_b_r', 'v_STlv_b_i', 'v_STlv_c_r', 'v_STlv_c_i', 'v_POI_a_r', 'v_POI_a_i', 'v_POI_b_r', 'v_POI_b_i', 'v_POI_c_r', 'v_POI_c_i', 'v_POImv_a_r', 'v_POImv_a_i', 'v_POImv_b_r', 'v_POImv_b_i', 'v_POImv_c_r', 'v_POImv_c_i', 'v_W1mv_a_r', 'v_W1mv_a_i', 'v_W1mv_b_r', 'v_W1mv_b_i', 'v_W1mv_c_r', 'v_W1mv_c_i', 'v_W2mv_a_r', 'v_W2mv_a_i', 'v_W2mv_b_r', 'v_W2mv_b_i', 'v_W2mv_c_r', 'v_W2mv_c_i', 'v_W3mv_a_r', 'v_W3mv_a_i', 'v_W3mv_b_r', 'v_W3mv_b_i', 'v_W3mv_c_r', 'v_W3mv_c_i', 'v_STmv_a_r', 'v_STmv_a_i', 'v_STmv_b_r', 'v_STmv_b_i', 'v_STmv_c_r', 'v_STmv_c_i', 'i_l_W1mv_W2mv_a_r', 'i_l_W1mv_W2mv_a_i', 'i_l_W1mv_W2mv_b_r', 'i_l_W1mv_W2mv_b_i', 'i_l_W1mv_W2mv_c_r', 'i_l_W1mv_W2mv_c_i', 'i_l_W2mv_W3mv_a_r', 'i_l_W2mv_W3mv_a_i', 'i_l_W2mv_W3mv_b_r', 'i_l_W2mv_W3mv_b_i', 'i_l_W2mv_W3mv_c_r', 'i_l_W2mv_W3mv_c_i', 'i_l_W3mv_POImv_a_r', 'i_l_W3mv_POImv_a_i', 'i_l_W3mv_POImv_b_r', 'i_l_W3mv_POImv_b_i', 'i_l_W3mv_POImv_c_r', 'i_l_W3mv_POImv_c_i', 'i_l_STmv_POImv_a_r', 'i_l_STmv_POImv_a_i', 'i_l_STmv_POImv_b_r', 'i_l_STmv_POImv_b_i', 'i_l_STmv_POImv_c_r', 'i_l_STmv_POImv_c_i', 'i_l_POI_GRID_a_r', 'i_l_POI_GRID_a_i', 'i_l_POI_GRID_b_r', 'i_l_POI_GRID_b_i', 'i_l_POI_GRID_c_r', 'i_l_POI_GRID_c_i', 'i_W1lv_a_r', 'i_W1lv_a_i', 'i_W1lv_b_r', 'i_W1lv_b_i', 'i_W1lv_c_r', 'i_W1lv_c_i', 'v_m_W1lv', 'v_m_W1mv', 'i_reac_ref_W1lv', 'q_ref_W1lv', 'i_W2lv_a_r', 'i_W2lv_a_i', 'i_W2lv_b_r', 'i_W2lv_b_i', 'i_W2lv_c_r', 'i_W2lv_c_i', 'v_m_W2lv', 'v_m_W2mv', 'i_reac_ref_W2lv', 'q_ref_W2lv', 'i_W3lv_a_r', 'i_W3lv_a_i', 'i_W3lv_b_r', 'i_W3lv_b_i', 'i_W3lv_c_r', 'i_W3lv_c_i', 'v_m_W3lv', 'v_m_W3mv', 'i_reac_ref_W3lv', 'q_ref_W3lv', 'i_STlv_a_r', 'i_STlv_a_i', 'i_STlv_b_r', 'i_STlv_b_i', 'i_STlv_c_r', 'i_STlv_c_i', 'v_m_STlv', 'v_m_STmv', 'i_reac_ref_STlv', 'q_ref_STlv'] 
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
        
        self.update() 


    def update(self): 

        self.N_steps = int(np.ceil(self.t_end/self.Dt)) 
        dt = [  
              ('t_end', np.float64),
              ('Dt', np.float64),
              ('decimation', np.float64),
              ('itol', np.float64),
              ('Dt_max', np.float64),
              ('Dt_min', np.float64),
              ('solvern', np.int64),
              ('imax', np.int64),
              ('N_steps', np.int64),
              ('N_store', np.int64),
              ('N_x', np.int64),
              ('N_y', np.int64),
              ('N_z', np.int64),
              ('t', np.float64),
              ('it', np.int64),
              ('it_store', np.int64),
              ('idx', np.int64),
              ('idy', np.int64),
              ('f', np.float64, (self.N_x,1)),
              ('x', np.float64, (self.N_x,1)),
              ('x_0', np.float64, (self.N_x,1)),
              ('g', np.float64, (self.N_y,1)),
              ('y_run', np.float64, (self.N_y,1)),
              ('y_ini', np.float64, (self.N_y,1)),
              ('y_0', np.float64, (self.N_y,1)),
              ('h', np.float64, (self.N_z,1)),
              ('Fx', np.float64, (self.N_x,self.N_x)),
              ('Fy', np.float64, (self.N_x,self.N_y)),
              ('Gx', np.float64, (self.N_y,self.N_x)),
              ('Gy', np.float64, (self.N_y,self.N_y)),
              ('Fu', np.float64, (self.N_x,self.N_u)),
              ('Gu', np.float64, (self.N_y,self.N_u)),
              ('Hx', np.float64, (self.N_z,self.N_x)),
              ('Hy', np.float64, (self.N_z,self.N_y)),
              ('Hu', np.float64, (self.N_z,self.N_u)),
              ('Fx_ini', np.float64, (self.N_x,self.N_x)),
              ('Fy_ini', np.float64, (self.N_x,self.N_y)),
              ('Gx_ini', np.float64, (self.N_y,self.N_x)),
              ('Gy_ini', np.float64, (self.N_y,self.N_y)),
              ('T', np.float64, (self.N_store+1,1)),
              ('X', np.float64, (self.N_store+1,self.N_x)),
              ('Y', np.float64, (self.N_store+1,self.N_y)),
              ('Z', np.float64, (self.N_store+1,self.N_z)),
              ('iters', np.float64, (self.N_store+1,1)),
             ]

        values = [
                self.t_end,                          
                self.Dt,
                self.decimation,
                self.itol,
                self.Dt_max,
                self.Dt_min,
                self.solvern,
                self.imax,
                self.N_steps,
                self.N_store,
                self.N_x,
                self.N_y,
                self.N_z,
                self.t,
                self.it,
                self.it_store,
                0,                                     # idx
                0,                                     # idy
                np.zeros((self.N_x,1)),                # f
                np.zeros((self.N_x,1)),                # x
                np.zeros((self.N_x,1)),                # x_0
                np.zeros((self.N_y,1)),                # g
                np.zeros((self.N_y,1)),                # y_run
                np.zeros((self.N_y,1)),                # y_ini
                np.zeros((self.N_y,1)),                # y_0
                np.zeros((self.N_z,1)),                # h
                np.zeros((self.N_x,self.N_x)),         # Fx   
                np.zeros((self.N_x,self.N_y)),         # Fy 
                np.zeros((self.N_y,self.N_x)),         # Gx 
                np.zeros((self.N_y,self.N_y)),         # Fy
                np.zeros((self.N_x,self.N_u)),         # Fu 
                np.zeros((self.N_y,self.N_u)),         # Gu 
                np.zeros((self.N_z,self.N_x)),         # Hx 
                np.zeros((self.N_z,self.N_y)),         # Hy 
                np.zeros((self.N_z,self.N_u)),         # Hu 
                np.zeros((self.N_x,self.N_x)),         # Fx_ini  
                np.zeros((self.N_x,self.N_y)),         # Fy_ini 
                np.zeros((self.N_y,self.N_x)),         # Gx_ini 
                np.zeros((self.N_y,self.N_y)),         # Fy_ini 
                np.zeros((self.N_store+1,1)),          # T
                np.zeros((self.N_store+1,self.N_x)),   # X
                np.zeros((self.N_store+1,self.N_y)),   # Y
                np.zeros((self.N_store+1,self.N_z)),   # Z
                np.zeros((self.N_store+1,1)),          # iters
                ]  

        dt += [(item,np.float64) for item in self.params_list]
        values += [item for item in self.params_values_list]

        for item_id,item_val in zip(self.inputs_ini_list,self.inputs_ini_values_list):
            if item_id in self.inputs_run_list: continue
            dt += [(item_id,np.float64)]
            values += [item_val]

        dt += [(item,np.float64) for item in self.inputs_run_list]
        values += [item for item in self.inputs_run_values_list]

        self.struct = np.rec.array([tuple(values)], dtype=np.dtype(dt))

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
            self.params_values_list[self.params_list.index(item)] = self.data[item]



    def ini_problem(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini(self.struct,2)
        ini(self.struct,3)       
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg

    def run_problem(self,x):
        t = self.struct[0].t
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(t,self.struct,2)
        run(t,self.struct,3)
        run(t,self.struct,10)
        run(t,self.struct,11)
        run(t,self.struct,12)
        run(t,self.struct,13)
        
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg
    

    def run_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(0.0,self.struct,10)
        run(0.0,self.struct,11)     
        run(0.0,self.struct,12)
        run(0.0,self.struct,13)
        A_c = np.block([[self.struct[0].Fx,self.struct[0].Fy],
                        [self.struct[0].Gx,self.struct[0].Gy]])
        return A_c

    def run_dae_jacobian_nn(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run_nn(0.0,self.struct,10)
        run_nn(0.0,self.struct,11)     
        run_nn(0.0,self.struct,12)
        run_nn(0.0,self.struct,13)
 

    
    def eval_jacobians(self):

        run(0.0,self.struct,10)
        run(0.0,self.struct,11)  
        run(0.0,self.struct,12) 

        return 1


    def ini_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini(self.struct,10)
        ini(self.struct,11)       
        A_c = np.block([[self.struct[0].Fx_ini,self.struct[0].Fy_ini],
                        [self.struct[0].Gx_ini,self.struct[0].Gy_ini]])
        return A_c

    def ini_dae_jacobian_nn(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini_nn(self.struct,10)
        ini_nn(self.struct,11)       
 

    def f_ode(self,x):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def f_odeint(self,x,t):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def f_ivp(self,t,x):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def Fx_ode(self,x):
        self.struct[0].x[:,0] = x
        run(self.struct,10)
        return self.struct[0].Fx

    def eval_A(self):
        
        Fx = self.struct[0].Fx
        Fy = self.struct[0].Fy
        Gx = self.struct[0].Gx
        Gy = self.struct[0].Gy
        
        A = Fx - Fy @ np.linalg.solve(Gy,Gx)
        
        self.A = A
        
        return A

    def eval_A_ini(self):
        
        Fx = self.struct[0].Fx_ini
        Fy = self.struct[0].Fy_ini
        Gx = self.struct[0].Gx_ini
        Gy = self.struct[0].Gy_ini
        
        A = Fx - Fy @ np.linalg.solve(Gy,Gx)
        
        
        return A
    
    def reset(self):
        for param,param_value in zip(self.params_list,self.params_values_list):
            self.struct[0][param] = param_value
        for input_name,input_value in zip(self.inputs_ini_list,self.inputs_ini_values_list):
            self.struct[0][input_name] = input_value   
        for input_name,input_value in zip(self.inputs_run_list,self.inputs_run_values_list):
            self.struct[0][input_name] = input_value  

    def simulate(self,events,xy0=0):
        
        # initialize both the ini and the run system
        self.initialize(events,xy0=xy0)
        
        ## solve 
        #daesolver(self.struct)    # run until first event

        # simulation run
        for event in events[1:]:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            daesolver(self.struct)    # run until next event
            
        
        T,X,Y,Z = self.post()
        
        return T,X,Y,Z
    
    def run(self,events):
        

        # simulation run
        for event in events:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            daesolver(self.struct)    # run until next event
            
        return 1
    
    
    def post(self):
        
        # post process result    
        T = self.struct[0]['T'][:self.struct[0].it_store]
        X = self.struct[0]['X'][:self.struct[0].it_store,:]
        Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
        Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
        iters = self.struct[0]['iters'][:self.struct[0].it_store,:]
    
        self.T = T
        self.X = X
        self.Y = Y
        self.Z = Z
        self.iters = iters
        
        return T,X,Y,Z
        
        
    def initialize(self,events=[{}],xy0=0):
        '''
        

        Parameters
        ----------
        events : dictionary 
            Dictionary with at least 't_end' and all inputs and parameters 
            that need to be changed.
        xy0 : float or string, optional
            0 means all states should be zero as initial guess. 
            If not zero all the states initial guess are the given input.
            If 'prev' it uses the last known initialization result as initial guess.

        Returns
        -------
        T : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        Y : TYPE
            DESCRIPTION.
        Z : TYPE
            DESCRIPTION.

        '''
        # simulation parameters
        self.struct[0].it = 0       # set time step to zero
        self.struct[0].it_store = 0 # set storage to zero
        self.struct[0].t = 0.0      # set time to zero
                    
        # initialization
        it_event = 0
        event = events[it_event]
        for item in event:
            self.struct[0][item] = event[item]
            
        
        ## compute initial conditions using x and y_ini 
        if xy0 == 0:
            xy0 = np.zeros(self.N_x+self.N_y)
        elif xy0 == 1:
            xy0 = np.ones(self.N_x+self.N_y)
        elif xy0 == 'prev':
            xy0 = self.xy_prev
        else:
            xy0 = xy0*np.ones(self.N_x+self.N_y)

        #xy = sopt.fsolve(self.ini_problem,xy0, jac=self.ini_dae_jacobian )
        self.ini_dae_jacobian_nn(xy0)
        self.run_dae_jacobian_nn(xy0)
        
        if self.sopt_root_jac:
            sol = sopt.root(self.ini_problem, xy0, 
                            jac=self.ini_dae_jacobian, 
                            method=self.sopt_root_method, tol=self.initialization_tol)
        else:
            sol = sopt.root(self.ini_problem, xy0, method=self.sopt_root_method)

        self.initialization_ok = True
        if sol.success == False:
            print('initialization not found!')
            self.initialization_ok = False

            T = self.struct[0]['T'][:self.struct[0].it_store]
            X = self.struct[0]['X'][:self.struct[0].it_store,:]
            Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
            Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
            iters = self.struct[0]['iters'][:self.struct[0].it_store,:]

        if self.initialization_ok:
            xy = sol.x
            self.xy_prev = xy
            self.struct[0].x[:,0] = xy[0:self.N_x]
            self.struct[0].y_run[:,0] = xy[self.N_x:]

            ## y_ini to u_run
            for item in self.inputs_run_list:
                if item in self.y_ini_list:
                    self.struct[0][item] = self.struct[0].y_ini[self.y_ini_list.index(item)]

            ## u_ini to y_run
            for item in self.inputs_ini_list:
                if item in self.y_run_list:
                    self.struct[0].y_run[self.y_run_list.index(item)] = self.struct[0][item]


            #xy = sopt.fsolve(self.ini_problem,xy0, jac=self.ini_dae_jacobian )
            if self.sopt_root_jac:
                sol = sopt.root(self.run_problem, xy0, 
                                jac=self.run_dae_jacobian, 
                                method=self.sopt_root_method, tol=self.initialization_tol)
            else:
                sol = sopt.root(self.run_problem, xy0, method=self.sopt_root_method)

            # evaluate f and g
            run(0.0,self.struct,2)
            run(0.0,self.struct,3)                

            
            # evaluate run jacobians 
            run(0.0,self.struct,10)
            run(0.0,self.struct,11)                
            run(0.0,self.struct,12) 
            run(0.0,self.struct,14) 
             
            # post process result    
            T = self.struct[0]['T'][:self.struct[0].it_store]
            X = self.struct[0]['X'][:self.struct[0].it_store,:]
            Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
            Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
            iters = self.struct[0]['iters'][:self.struct[0].it_store,:]
        
            self.T = T
            self.X = X
            self.Y = Y
            self.Z = Z
            self.iters = iters
            
        return self.initialization_ok
    
    
    def get_value(self,name):
        if name in self.inputs_run_list:
            value = self.struct[0][name]
        if name in self.x_list:
            idx = self.x_list.index(name)
            value = self.struct[0].x[idx,0]
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.struct[0].y_run[idy,0]
        if name in self.params_list:
            value = self.struct[0][name]
        if name in self.outputs_list:
            value = self.struct[0].h[self.outputs_list.index(name),0] 

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
    
    def set_value(self,name,value):
        if name in self.inputs_run_list:
            self.struct[0][name] = value
        if name in self.params_list:
            self.struct[0][name] = value
            
    def report_x(self,value_format='5.2f'):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_y(self,value_format='5.2f'):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')
            
    def report_u(self,value_format='5.2f'):
        for item in self.inputs_run_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_z(self,value_format='5.2f'):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')
            
    def get_x(self):
        return self.struct[0].x


@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    u_ctrl_v_W1lv = struct[0].u_ctrl_v_W1lv
    K_p_v_W1lv = struct[0].K_p_v_W1lv
    K_i_v_W1lv = struct[0].K_i_v_W1lv
    V_base_W1lv = struct[0].V_base_W1lv
    V_base_W1mv = struct[0].V_base_W1mv
    S_base_W1lv = struct[0].S_base_W1lv
    I_max_W1lv = struct[0].I_max_W1lv
    u_ctrl_v_W2lv = struct[0].u_ctrl_v_W2lv
    K_p_v_W2lv = struct[0].K_p_v_W2lv
    K_i_v_W2lv = struct[0].K_i_v_W2lv
    V_base_W2lv = struct[0].V_base_W2lv
    V_base_W2mv = struct[0].V_base_W2mv
    S_base_W2lv = struct[0].S_base_W2lv
    I_max_W2lv = struct[0].I_max_W2lv
    u_ctrl_v_W3lv = struct[0].u_ctrl_v_W3lv
    K_p_v_W3lv = struct[0].K_p_v_W3lv
    K_i_v_W3lv = struct[0].K_i_v_W3lv
    V_base_W3lv = struct[0].V_base_W3lv
    V_base_W3mv = struct[0].V_base_W3mv
    S_base_W3lv = struct[0].S_base_W3lv
    I_max_W3lv = struct[0].I_max_W3lv
    u_ctrl_v_STlv = struct[0].u_ctrl_v_STlv
    K_p_v_STlv = struct[0].K_p_v_STlv
    K_i_v_STlv = struct[0].K_i_v_STlv
    V_base_STlv = struct[0].V_base_STlv
    V_base_STmv = struct[0].V_base_STmv
    S_base_STlv = struct[0].S_base_STlv
    I_max_STlv = struct[0].I_max_STlv
    
    # Inputs:
    v_GRID_a_r = struct[0].v_GRID_a_r
    v_GRID_a_i = struct[0].v_GRID_a_i
    v_GRID_b_r = struct[0].v_GRID_b_r
    v_GRID_b_i = struct[0].v_GRID_b_i
    v_GRID_c_r = struct[0].v_GRID_c_r
    v_GRID_c_i = struct[0].v_GRID_c_i
    i_POI_a_r = struct[0].i_POI_a_r
    i_POI_a_i = struct[0].i_POI_a_i
    i_POI_b_r = struct[0].i_POI_b_r
    i_POI_b_i = struct[0].i_POI_b_i
    i_POI_c_r = struct[0].i_POI_c_r
    i_POI_c_i = struct[0].i_POI_c_i
    i_POImv_a_r = struct[0].i_POImv_a_r
    i_POImv_a_i = struct[0].i_POImv_a_i
    i_POImv_b_r = struct[0].i_POImv_b_r
    i_POImv_b_i = struct[0].i_POImv_b_i
    i_POImv_c_r = struct[0].i_POImv_c_r
    i_POImv_c_i = struct[0].i_POImv_c_i
    i_W1mv_a_r = struct[0].i_W1mv_a_r
    i_W1mv_a_i = struct[0].i_W1mv_a_i
    i_W1mv_b_r = struct[0].i_W1mv_b_r
    i_W1mv_b_i = struct[0].i_W1mv_b_i
    i_W1mv_c_r = struct[0].i_W1mv_c_r
    i_W1mv_c_i = struct[0].i_W1mv_c_i
    i_W2mv_a_r = struct[0].i_W2mv_a_r
    i_W2mv_a_i = struct[0].i_W2mv_a_i
    i_W2mv_b_r = struct[0].i_W2mv_b_r
    i_W2mv_b_i = struct[0].i_W2mv_b_i
    i_W2mv_c_r = struct[0].i_W2mv_c_r
    i_W2mv_c_i = struct[0].i_W2mv_c_i
    i_W3mv_a_r = struct[0].i_W3mv_a_r
    i_W3mv_a_i = struct[0].i_W3mv_a_i
    i_W3mv_b_r = struct[0].i_W3mv_b_r
    i_W3mv_b_i = struct[0].i_W3mv_b_i
    i_W3mv_c_r = struct[0].i_W3mv_c_r
    i_W3mv_c_i = struct[0].i_W3mv_c_i
    i_STmv_a_r = struct[0].i_STmv_a_r
    i_STmv_a_i = struct[0].i_STmv_a_i
    i_STmv_b_r = struct[0].i_STmv_b_r
    i_STmv_b_i = struct[0].i_STmv_b_i
    i_STmv_c_r = struct[0].i_STmv_c_r
    i_STmv_c_i = struct[0].i_STmv_c_i
    p_ref_W1lv = struct[0].p_ref_W1lv
    T_pq_W1lv = struct[0].T_pq_W1lv
    v_loc_ref_W1lv = struct[0].v_loc_ref_W1lv
    Dv_r_W1lv = struct[0].Dv_r_W1lv
    Dq_r_W1lv = struct[0].Dq_r_W1lv
    p_ref_W2lv = struct[0].p_ref_W2lv
    T_pq_W2lv = struct[0].T_pq_W2lv
    v_loc_ref_W2lv = struct[0].v_loc_ref_W2lv
    Dv_r_W2lv = struct[0].Dv_r_W2lv
    Dq_r_W2lv = struct[0].Dq_r_W2lv
    p_ref_W3lv = struct[0].p_ref_W3lv
    T_pq_W3lv = struct[0].T_pq_W3lv
    v_loc_ref_W3lv = struct[0].v_loc_ref_W3lv
    Dv_r_W3lv = struct[0].Dv_r_W3lv
    Dq_r_W3lv = struct[0].Dq_r_W3lv
    p_ref_STlv = struct[0].p_ref_STlv
    T_pq_STlv = struct[0].T_pq_STlv
    v_loc_ref_STlv = struct[0].v_loc_ref_STlv
    Dv_r_STlv = struct[0].Dv_r_STlv
    Dq_r_STlv = struct[0].Dq_r_STlv
    
    # Dynamical states:
    p_W1lv_a = struct[0].x[0,0]
    p_W1lv_b = struct[0].x[1,0]
    p_W1lv_c = struct[0].x[2,0]
    q_W1lv_a = struct[0].x[3,0]
    q_W1lv_b = struct[0].x[4,0]
    q_W1lv_c = struct[0].x[5,0]
    p_W2lv_a = struct[0].x[6,0]
    p_W2lv_b = struct[0].x[7,0]
    p_W2lv_c = struct[0].x[8,0]
    q_W2lv_a = struct[0].x[9,0]
    q_W2lv_b = struct[0].x[10,0]
    q_W2lv_c = struct[0].x[11,0]
    p_W3lv_a = struct[0].x[12,0]
    p_W3lv_b = struct[0].x[13,0]
    p_W3lv_c = struct[0].x[14,0]
    q_W3lv_a = struct[0].x[15,0]
    q_W3lv_b = struct[0].x[16,0]
    q_W3lv_c = struct[0].x[17,0]
    p_STlv_a = struct[0].x[18,0]
    p_STlv_b = struct[0].x[19,0]
    p_STlv_c = struct[0].x[20,0]
    q_STlv_a = struct[0].x[21,0]
    q_STlv_b = struct[0].x[22,0]
    q_STlv_c = struct[0].x[23,0]
    
    # Algebraic states:
    v_W1lv_a_r = struct[0].y_ini[0,0]
    v_W1lv_a_i = struct[0].y_ini[1,0]
    v_W1lv_b_r = struct[0].y_ini[2,0]
    v_W1lv_b_i = struct[0].y_ini[3,0]
    v_W1lv_c_r = struct[0].y_ini[4,0]
    v_W1lv_c_i = struct[0].y_ini[5,0]
    v_W2lv_a_r = struct[0].y_ini[6,0]
    v_W2lv_a_i = struct[0].y_ini[7,0]
    v_W2lv_b_r = struct[0].y_ini[8,0]
    v_W2lv_b_i = struct[0].y_ini[9,0]
    v_W2lv_c_r = struct[0].y_ini[10,0]
    v_W2lv_c_i = struct[0].y_ini[11,0]
    v_W3lv_a_r = struct[0].y_ini[12,0]
    v_W3lv_a_i = struct[0].y_ini[13,0]
    v_W3lv_b_r = struct[0].y_ini[14,0]
    v_W3lv_b_i = struct[0].y_ini[15,0]
    v_W3lv_c_r = struct[0].y_ini[16,0]
    v_W3lv_c_i = struct[0].y_ini[17,0]
    v_STlv_a_r = struct[0].y_ini[18,0]
    v_STlv_a_i = struct[0].y_ini[19,0]
    v_STlv_b_r = struct[0].y_ini[20,0]
    v_STlv_b_i = struct[0].y_ini[21,0]
    v_STlv_c_r = struct[0].y_ini[22,0]
    v_STlv_c_i = struct[0].y_ini[23,0]
    v_POI_a_r = struct[0].y_ini[24,0]
    v_POI_a_i = struct[0].y_ini[25,0]
    v_POI_b_r = struct[0].y_ini[26,0]
    v_POI_b_i = struct[0].y_ini[27,0]
    v_POI_c_r = struct[0].y_ini[28,0]
    v_POI_c_i = struct[0].y_ini[29,0]
    v_POImv_a_r = struct[0].y_ini[30,0]
    v_POImv_a_i = struct[0].y_ini[31,0]
    v_POImv_b_r = struct[0].y_ini[32,0]
    v_POImv_b_i = struct[0].y_ini[33,0]
    v_POImv_c_r = struct[0].y_ini[34,0]
    v_POImv_c_i = struct[0].y_ini[35,0]
    v_W1mv_a_r = struct[0].y_ini[36,0]
    v_W1mv_a_i = struct[0].y_ini[37,0]
    v_W1mv_b_r = struct[0].y_ini[38,0]
    v_W1mv_b_i = struct[0].y_ini[39,0]
    v_W1mv_c_r = struct[0].y_ini[40,0]
    v_W1mv_c_i = struct[0].y_ini[41,0]
    v_W2mv_a_r = struct[0].y_ini[42,0]
    v_W2mv_a_i = struct[0].y_ini[43,0]
    v_W2mv_b_r = struct[0].y_ini[44,0]
    v_W2mv_b_i = struct[0].y_ini[45,0]
    v_W2mv_c_r = struct[0].y_ini[46,0]
    v_W2mv_c_i = struct[0].y_ini[47,0]
    v_W3mv_a_r = struct[0].y_ini[48,0]
    v_W3mv_a_i = struct[0].y_ini[49,0]
    v_W3mv_b_r = struct[0].y_ini[50,0]
    v_W3mv_b_i = struct[0].y_ini[51,0]
    v_W3mv_c_r = struct[0].y_ini[52,0]
    v_W3mv_c_i = struct[0].y_ini[53,0]
    v_STmv_a_r = struct[0].y_ini[54,0]
    v_STmv_a_i = struct[0].y_ini[55,0]
    v_STmv_b_r = struct[0].y_ini[56,0]
    v_STmv_b_i = struct[0].y_ini[57,0]
    v_STmv_c_r = struct[0].y_ini[58,0]
    v_STmv_c_i = struct[0].y_ini[59,0]
    i_l_W1mv_W2mv_a_r = struct[0].y_ini[60,0]
    i_l_W1mv_W2mv_a_i = struct[0].y_ini[61,0]
    i_l_W1mv_W2mv_b_r = struct[0].y_ini[62,0]
    i_l_W1mv_W2mv_b_i = struct[0].y_ini[63,0]
    i_l_W1mv_W2mv_c_r = struct[0].y_ini[64,0]
    i_l_W1mv_W2mv_c_i = struct[0].y_ini[65,0]
    i_l_W2mv_W3mv_a_r = struct[0].y_ini[66,0]
    i_l_W2mv_W3mv_a_i = struct[0].y_ini[67,0]
    i_l_W2mv_W3mv_b_r = struct[0].y_ini[68,0]
    i_l_W2mv_W3mv_b_i = struct[0].y_ini[69,0]
    i_l_W2mv_W3mv_c_r = struct[0].y_ini[70,0]
    i_l_W2mv_W3mv_c_i = struct[0].y_ini[71,0]
    i_l_W3mv_POImv_a_r = struct[0].y_ini[72,0]
    i_l_W3mv_POImv_a_i = struct[0].y_ini[73,0]
    i_l_W3mv_POImv_b_r = struct[0].y_ini[74,0]
    i_l_W3mv_POImv_b_i = struct[0].y_ini[75,0]
    i_l_W3mv_POImv_c_r = struct[0].y_ini[76,0]
    i_l_W3mv_POImv_c_i = struct[0].y_ini[77,0]
    i_l_STmv_POImv_a_r = struct[0].y_ini[78,0]
    i_l_STmv_POImv_a_i = struct[0].y_ini[79,0]
    i_l_STmv_POImv_b_r = struct[0].y_ini[80,0]
    i_l_STmv_POImv_b_i = struct[0].y_ini[81,0]
    i_l_STmv_POImv_c_r = struct[0].y_ini[82,0]
    i_l_STmv_POImv_c_i = struct[0].y_ini[83,0]
    i_l_POI_GRID_a_r = struct[0].y_ini[84,0]
    i_l_POI_GRID_a_i = struct[0].y_ini[85,0]
    i_l_POI_GRID_b_r = struct[0].y_ini[86,0]
    i_l_POI_GRID_b_i = struct[0].y_ini[87,0]
    i_l_POI_GRID_c_r = struct[0].y_ini[88,0]
    i_l_POI_GRID_c_i = struct[0].y_ini[89,0]
    i_W1lv_a_r = struct[0].y_ini[90,0]
    i_W1lv_a_i = struct[0].y_ini[91,0]
    i_W1lv_b_r = struct[0].y_ini[92,0]
    i_W1lv_b_i = struct[0].y_ini[93,0]
    i_W1lv_c_r = struct[0].y_ini[94,0]
    i_W1lv_c_i = struct[0].y_ini[95,0]
    v_m_W1lv = struct[0].y_ini[96,0]
    v_m_W1mv = struct[0].y_ini[97,0]
    i_reac_ref_W1lv = struct[0].y_ini[98,0]
    q_ref_W1lv = struct[0].y_ini[99,0]
    i_W2lv_a_r = struct[0].y_ini[100,0]
    i_W2lv_a_i = struct[0].y_ini[101,0]
    i_W2lv_b_r = struct[0].y_ini[102,0]
    i_W2lv_b_i = struct[0].y_ini[103,0]
    i_W2lv_c_r = struct[0].y_ini[104,0]
    i_W2lv_c_i = struct[0].y_ini[105,0]
    v_m_W2lv = struct[0].y_ini[106,0]
    v_m_W2mv = struct[0].y_ini[107,0]
    i_reac_ref_W2lv = struct[0].y_ini[108,0]
    q_ref_W2lv = struct[0].y_ini[109,0]
    i_W3lv_a_r = struct[0].y_ini[110,0]
    i_W3lv_a_i = struct[0].y_ini[111,0]
    i_W3lv_b_r = struct[0].y_ini[112,0]
    i_W3lv_b_i = struct[0].y_ini[113,0]
    i_W3lv_c_r = struct[0].y_ini[114,0]
    i_W3lv_c_i = struct[0].y_ini[115,0]
    v_m_W3lv = struct[0].y_ini[116,0]
    v_m_W3mv = struct[0].y_ini[117,0]
    i_reac_ref_W3lv = struct[0].y_ini[118,0]
    q_ref_W3lv = struct[0].y_ini[119,0]
    i_STlv_a_r = struct[0].y_ini[120,0]
    i_STlv_a_i = struct[0].y_ini[121,0]
    i_STlv_b_r = struct[0].y_ini[122,0]
    i_STlv_b_i = struct[0].y_ini[123,0]
    i_STlv_c_r = struct[0].y_ini[124,0]
    i_STlv_c_i = struct[0].y_ini[125,0]
    v_m_STlv = struct[0].y_ini[126,0]
    v_m_STmv = struct[0].y_ini[127,0]
    i_reac_ref_STlv = struct[0].y_ini[128,0]
    q_ref_STlv = struct[0].y_ini[129,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-p_W1lv_a + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[1,0] = (-p_W1lv_b + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[2,0] = (-p_W1lv_c + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[3,0] = (-q_W1lv_a + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[4,0] = (-q_W1lv_b + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[5,0] = (-q_W1lv_c + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[6,0] = (-p_W2lv_a + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[7,0] = (-p_W2lv_b + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[8,0] = (-p_W2lv_c + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[9,0] = (-q_W2lv_a + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[10,0] = (-q_W2lv_b + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[11,0] = (-q_W2lv_c + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[12,0] = (-p_W3lv_a + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[13,0] = (-p_W3lv_b + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[14,0] = (-p_W3lv_c + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[15,0] = (-q_W3lv_a + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[16,0] = (-q_W3lv_b + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[17,0] = (-q_W3lv_c + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[18,0] = (-p_STlv_a + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[19,0] = (-p_STlv_b + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[20,0] = (-p_STlv_c + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[21,0] = (-q_STlv_a + q_ref_STlv/3)/T_pq_STlv
        struct[0].f[22,0] = (-q_STlv_b + q_ref_STlv/3)/T_pq_STlv
        struct[0].f[23,0] = (-q_STlv_c + q_ref_STlv/3)/T_pq_STlv
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_W1lv_a_r - 85.1513138847732*v_W1lv_a_i - 14.1918856474622*v_W1lv_a_r + 1.69609362276623*v_W1mv_a_i + 0.282682270461039*v_W1mv_a_r - 1.69609362276623*v_W1mv_c_i - 0.282682270461039*v_W1mv_c_r
        struct[0].g[1,0] = i_W1lv_a_i - 14.1918856474622*v_W1lv_a_i + 85.1513138847732*v_W1lv_a_r + 0.282682270461039*v_W1mv_a_i - 1.69609362276623*v_W1mv_a_r - 0.282682270461039*v_W1mv_c_i + 1.69609362276623*v_W1mv_c_r
        struct[0].g[2,0] = i_W1lv_b_r - 85.1513138847732*v_W1lv_b_i - 14.1918856474622*v_W1lv_b_r - 1.69609362276623*v_W1mv_a_i - 0.282682270461039*v_W1mv_a_r + 1.69609362276623*v_W1mv_b_i + 0.282682270461039*v_W1mv_b_r
        struct[0].g[3,0] = i_W1lv_b_i - 14.1918856474622*v_W1lv_b_i + 85.1513138847732*v_W1lv_b_r - 0.282682270461039*v_W1mv_a_i + 1.69609362276623*v_W1mv_a_r + 0.282682270461039*v_W1mv_b_i - 1.69609362276623*v_W1mv_b_r
        struct[0].g[4,0] = i_W1lv_c_r - 85.1513138847732*v_W1lv_c_i - 14.1918856474622*v_W1lv_c_r - 1.69609362276623*v_W1mv_b_i - 0.282682270461039*v_W1mv_b_r + 1.69609362276623*v_W1mv_c_i + 0.282682270461039*v_W1mv_c_r
        struct[0].g[5,0] = i_W1lv_c_i - 14.1918856474622*v_W1lv_c_i + 85.1513138847732*v_W1lv_c_r - 0.282682270461039*v_W1mv_b_i + 1.69609362276623*v_W1mv_b_r + 0.282682270461039*v_W1mv_c_i - 1.69609362276623*v_W1mv_c_r
        struct[0].g[6,0] = i_W2lv_a_r - 85.1513138847732*v_W2lv_a_i - 14.1918856474622*v_W2lv_a_r + 1.69609362276623*v_W2mv_a_i + 0.282682270461039*v_W2mv_a_r - 1.69609362276623*v_W2mv_c_i - 0.282682270461039*v_W2mv_c_r
        struct[0].g[7,0] = i_W2lv_a_i - 14.1918856474622*v_W2lv_a_i + 85.1513138847732*v_W2lv_a_r + 0.282682270461039*v_W2mv_a_i - 1.69609362276623*v_W2mv_a_r - 0.282682270461039*v_W2mv_c_i + 1.69609362276623*v_W2mv_c_r
        struct[0].g[8,0] = i_W2lv_b_r - 85.1513138847732*v_W2lv_b_i - 14.1918856474622*v_W2lv_b_r - 1.69609362276623*v_W2mv_a_i - 0.282682270461039*v_W2mv_a_r + 1.69609362276623*v_W2mv_b_i + 0.282682270461039*v_W2mv_b_r
        struct[0].g[9,0] = i_W2lv_b_i - 14.1918856474622*v_W2lv_b_i + 85.1513138847732*v_W2lv_b_r - 0.282682270461039*v_W2mv_a_i + 1.69609362276623*v_W2mv_a_r + 0.282682270461039*v_W2mv_b_i - 1.69609362276623*v_W2mv_b_r
        struct[0].g[10,0] = i_W2lv_c_r - 85.1513138847732*v_W2lv_c_i - 14.1918856474622*v_W2lv_c_r - 1.69609362276623*v_W2mv_b_i - 0.282682270461039*v_W2mv_b_r + 1.69609362276623*v_W2mv_c_i + 0.282682270461039*v_W2mv_c_r
        struct[0].g[11,0] = i_W2lv_c_i - 14.1918856474622*v_W2lv_c_i + 85.1513138847732*v_W2lv_c_r - 0.282682270461039*v_W2mv_b_i + 1.69609362276623*v_W2mv_b_r + 0.282682270461039*v_W2mv_c_i - 1.69609362276623*v_W2mv_c_r
        struct[0].g[12,0] = i_W3lv_a_r - 85.1513138847732*v_W3lv_a_i - 14.1918856474622*v_W3lv_a_r + 1.69609362276623*v_W3mv_a_i + 0.282682270461039*v_W3mv_a_r - 1.69609362276623*v_W3mv_c_i - 0.282682270461039*v_W3mv_c_r
        struct[0].g[13,0] = i_W3lv_a_i - 14.1918856474622*v_W3lv_a_i + 85.1513138847732*v_W3lv_a_r + 0.282682270461039*v_W3mv_a_i - 1.69609362276623*v_W3mv_a_r - 0.282682270461039*v_W3mv_c_i + 1.69609362276623*v_W3mv_c_r
        struct[0].g[14,0] = i_W3lv_b_r - 85.1513138847732*v_W3lv_b_i - 14.1918856474622*v_W3lv_b_r - 1.69609362276623*v_W3mv_a_i - 0.282682270461039*v_W3mv_a_r + 1.69609362276623*v_W3mv_b_i + 0.282682270461039*v_W3mv_b_r
        struct[0].g[15,0] = i_W3lv_b_i - 14.1918856474622*v_W3lv_b_i + 85.1513138847732*v_W3lv_b_r - 0.282682270461039*v_W3mv_a_i + 1.69609362276623*v_W3mv_a_r + 0.282682270461039*v_W3mv_b_i - 1.69609362276623*v_W3mv_b_r
        struct[0].g[16,0] = i_W3lv_c_r - 85.1513138847732*v_W3lv_c_i - 14.1918856474622*v_W3lv_c_r - 1.69609362276623*v_W3mv_b_i - 0.282682270461039*v_W3mv_b_r + 1.69609362276623*v_W3mv_c_i + 0.282682270461039*v_W3mv_c_r
        struct[0].g[17,0] = i_W3lv_c_i - 14.1918856474622*v_W3lv_c_i + 85.1513138847732*v_W3lv_c_r - 0.282682270461039*v_W3mv_b_i + 1.69609362276623*v_W3mv_b_r + 0.282682270461039*v_W3mv_c_i - 1.69609362276623*v_W3mv_c_r
        struct[0].g[18,0] = i_STlv_a_r - 85.1513138847732*v_STlv_a_i - 14.1918856474622*v_STlv_a_r + 1.69609362276623*v_STmv_a_i + 0.282682270461039*v_STmv_a_r - 1.69609362276623*v_STmv_c_i - 0.282682270461039*v_STmv_c_r
        struct[0].g[19,0] = i_STlv_a_i - 14.1918856474622*v_STlv_a_i + 85.1513138847732*v_STlv_a_r + 0.282682270461039*v_STmv_a_i - 1.69609362276623*v_STmv_a_r - 0.282682270461039*v_STmv_c_i + 1.69609362276623*v_STmv_c_r
        struct[0].g[20,0] = i_STlv_b_r - 85.1513138847732*v_STlv_b_i - 14.1918856474622*v_STlv_b_r - 1.69609362276623*v_STmv_a_i - 0.282682270461039*v_STmv_a_r + 1.69609362276623*v_STmv_b_i + 0.282682270461039*v_STmv_b_r
        struct[0].g[21,0] = i_STlv_b_i - 14.1918856474622*v_STlv_b_i + 85.1513138847732*v_STlv_b_r - 0.282682270461039*v_STmv_a_i + 1.69609362276623*v_STmv_a_r + 0.282682270461039*v_STmv_b_i - 1.69609362276623*v_STmv_b_r
        struct[0].g[22,0] = i_STlv_c_r - 85.1513138847732*v_STlv_c_i - 14.1918856474622*v_STlv_c_r - 1.69609362276623*v_STmv_b_i - 0.282682270461039*v_STmv_b_r + 1.69609362276623*v_STmv_c_i + 0.282682270461039*v_STmv_c_r
        struct[0].g[23,0] = i_STlv_c_i - 14.1918856474622*v_STlv_c_i + 85.1513138847732*v_STlv_c_r - 0.282682270461039*v_STmv_b_i + 1.69609362276623*v_STmv_b_r + 0.282682270461039*v_STmv_c_i - 1.69609362276623*v_STmv_c_r
        struct[0].g[24,0] = i_POI_a_r + 0.040290088638195*v_GRID_a_i + 0.024174053182917*v_GRID_a_r + 4.66248501556824e-18*v_GRID_b_i - 4.31760362252812e-18*v_GRID_b_r + 4.19816664496737e-18*v_GRID_c_i - 3.49608108880335e-18*v_GRID_c_r - 0.0591264711109411*v_POI_a_i - 0.0265286009920103*v_POI_a_r + 0.00941819123637305*v_POI_b_i + 0.00117727390454664*v_POI_b_r + 0.00941819123637305*v_POI_c_i + 0.00117727390454664*v_POI_c_r + 0.0538321929314336*v_POImv_a_i + 0.0067290241164292*v_POImv_a_r - 0.0538321929314336*v_POImv_b_i - 0.0067290241164292*v_POImv_b_r
        struct[0].g[25,0] = i_POI_a_i + 0.024174053182917*v_GRID_a_i - 0.040290088638195*v_GRID_a_r - 4.31760362252812e-18*v_GRID_b_i - 4.66248501556824e-18*v_GRID_b_r - 3.49608108880335e-18*v_GRID_c_i - 4.19816664496737e-18*v_GRID_c_r - 0.0265286009920103*v_POI_a_i + 0.0591264711109411*v_POI_a_r + 0.00117727390454664*v_POI_b_i - 0.00941819123637305*v_POI_b_r + 0.00117727390454664*v_POI_c_i - 0.00941819123637305*v_POI_c_r + 0.0067290241164292*v_POImv_a_i - 0.0538321929314336*v_POImv_a_r - 0.0067290241164292*v_POImv_b_i + 0.0538321929314336*v_POImv_b_r
        struct[0].g[26,0] = i_POI_b_r + 6.30775359573304e-19*v_GRID_a_i - 2.07254761002657e-18*v_GRID_a_r + 0.040290088638195*v_GRID_b_i + 0.024174053182917*v_GRID_b_r + 9.01107656533306e-19*v_GRID_c_i - 1.78419315993592e-17*v_GRID_c_r + 0.00941819123637305*v_POI_a_i + 0.00117727390454663*v_POI_a_r - 0.0591264711109411*v_POI_b_i - 0.0265286009920103*v_POI_b_r + 0.00941819123637305*v_POI_c_i + 0.00117727390454665*v_POI_c_r + 0.0538321929314336*v_POImv_b_i + 0.0067290241164292*v_POImv_b_r - 0.0538321929314336*v_POImv_c_i - 0.0067290241164292*v_POImv_c_r
        struct[0].g[27,0] = i_POI_b_i - 2.07254761002657e-18*v_GRID_a_i - 6.30775359573304e-19*v_GRID_a_r + 0.024174053182917*v_GRID_b_i - 0.040290088638195*v_GRID_b_r - 1.78419315993592e-17*v_GRID_c_i - 9.01107656533306e-19*v_GRID_c_r + 0.00117727390454663*v_POI_a_i - 0.00941819123637305*v_POI_a_r - 0.0265286009920103*v_POI_b_i + 0.0591264711109411*v_POI_b_r + 0.00117727390454665*v_POI_c_i - 0.00941819123637305*v_POI_c_r + 0.0067290241164292*v_POImv_b_i - 0.0538321929314336*v_POImv_b_r - 0.0067290241164292*v_POImv_c_i + 0.0538321929314336*v_POImv_c_r
        struct[0].g[28,0] = i_POI_c_r - 7.20886125226632e-19*v_GRID_a_i - 1.35166148479994e-18*v_GRID_a_r - 4.50553828266631e-19*v_GRID_b_i - 1.71210454741325e-17*v_GRID_b_r + 0.040290088638195*v_GRID_c_i + 0.024174053182917*v_GRID_c_r + 0.00941819123637305*v_POI_a_i + 0.00117727390454663*v_POI_a_r + 0.00941819123637305*v_POI_b_i + 0.00117727390454665*v_POI_b_r - 0.0591264711109411*v_POI_c_i - 0.0265286009920103*v_POI_c_r - 0.0538321929314336*v_POImv_a_i - 0.0067290241164292*v_POImv_a_r + 0.0538321929314336*v_POImv_c_i + 0.0067290241164292*v_POImv_c_r
        struct[0].g[29,0] = i_POI_c_i - 1.35166148479994e-18*v_GRID_a_i + 7.20886125226632e-19*v_GRID_a_r - 1.71210454741325e-17*v_GRID_b_i + 4.50553828266631e-19*v_GRID_b_r + 0.024174053182917*v_GRID_c_i - 0.040290088638195*v_GRID_c_r + 0.00117727390454663*v_POI_a_i - 0.00941819123637305*v_POI_a_r + 0.00117727390454665*v_POI_b_i - 0.00941819123637305*v_POI_b_r - 0.0265286009920103*v_POI_c_i + 0.0591264711109411*v_POI_c_r - 0.0067290241164292*v_POImv_a_i + 0.0538321929314336*v_POImv_a_r + 0.0067290241164292*v_POImv_c_i - 0.0538321929314336*v_POImv_c_r
        struct[0].g[30,0] = i_POImv_a_r + 0.0538321929314336*v_POI_a_i + 0.0067290241164292*v_POI_a_r - 0.0538321929314336*v_POI_c_i - 0.0067290241164292*v_POI_c_r - 155.244588874881*v_POImv_a_i - 188.924390492986*v_POImv_a_r + 44.2677164725443*v_POImv_b_i + 53.9540151298641*v_POImv_b_r + 44.2677164725443*v_POImv_c_i + 53.9540151298641*v_POImv_c_r + 148.977829666654*v_STmv_a_i + 181.611068546017*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274334*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274334*v_STmv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[31,0] = i_POImv_a_i + 0.0067290241164292*v_POI_a_i - 0.0538321929314336*v_POI_a_r - 0.0067290241164292*v_POI_c_i + 0.0538321929314336*v_POI_c_r - 188.924390492986*v_POImv_a_i + 155.244588874881*v_POImv_a_r + 53.9540151298641*v_POImv_b_i - 44.2677164725443*v_POImv_b_r + 53.9540151298641*v_POImv_c_i - 44.2677164725443*v_POImv_c_r + 181.611068546017*v_STmv_a_i - 148.977829666654*v_STmv_a_r - 51.8888767274334*v_STmv_b_i + 42.5650941904727*v_STmv_b_r - 51.8888767274334*v_STmv_c_i + 42.5650941904727*v_STmv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[32,0] = i_POImv_b_r - 0.0538321929314336*v_POI_a_i - 0.0067290241164292*v_POI_a_r + 0.0538321929314336*v_POI_b_i + 0.0067290241164292*v_POI_b_r + 44.2677164725443*v_POImv_a_i + 53.9540151298641*v_POImv_a_r - 155.244588874881*v_POImv_b_i - 188.924390492986*v_POImv_b_r + 44.2677164725443*v_POImv_c_i + 53.9540151298642*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r + 148.977829666654*v_STmv_b_i + 181.611068546017*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274335*v_STmv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[33,0] = i_POImv_b_i - 0.0067290241164292*v_POI_a_i + 0.0538321929314336*v_POI_a_r + 0.0067290241164292*v_POI_b_i - 0.0538321929314336*v_POI_b_r + 53.9540151298641*v_POImv_a_i - 44.2677164725443*v_POImv_a_r - 188.924390492986*v_POImv_b_i + 155.244588874881*v_POImv_b_r + 53.9540151298642*v_POImv_c_i - 44.2677164725443*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r + 181.611068546017*v_STmv_b_i - 148.977829666654*v_STmv_b_r - 51.8888767274335*v_STmv_c_i + 42.5650941904727*v_STmv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[34,0] = i_POImv_c_r - 0.0538321929314336*v_POI_b_i - 0.0067290241164292*v_POI_b_r + 0.0538321929314336*v_POI_c_i + 0.0067290241164292*v_POI_c_r + 44.2677164725443*v_POImv_a_i + 53.9540151298641*v_POImv_a_r + 44.2677164725443*v_POImv_b_i + 53.9540151298642*v_POImv_b_r - 155.244588874881*v_POImv_c_i - 188.924390492986*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274335*v_STmv_b_r + 148.977829666654*v_STmv_c_i + 181.611068546017*v_STmv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[35,0] = i_POImv_c_i - 0.0067290241164292*v_POI_b_i + 0.0538321929314336*v_POI_b_r + 0.0067290241164292*v_POI_c_i - 0.0538321929314336*v_POI_c_r + 53.9540151298641*v_POImv_a_i - 44.2677164725443*v_POImv_a_r + 53.9540151298642*v_POImv_b_i - 44.2677164725443*v_POImv_b_r - 188.924390492986*v_POImv_c_i + 155.244588874881*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r - 51.8888767274335*v_STmv_b_i + 42.5650941904727*v_STmv_b_r + 181.611068546017*v_STmv_c_i - 148.977829666654*v_STmv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[36,0] = i_W1mv_a_r + 1.69609362276623*v_W1lv_a_i + 0.282682270461039*v_W1lv_a_r - 1.69609362276623*v_W1lv_b_i - 0.282682270461039*v_W1lv_b_r - 6.02663624833782*v_W1mv_a_i - 7.27570400310194*v_W1mv_a_r + 1.73640535376106*v_W1mv_b_i + 2.08118569972797*v_W1mv_b_r + 1.73640535376106*v_W1mv_c_i + 2.08118569972797*v_W1mv_c_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r
        struct[0].g[37,0] = i_W1mv_a_i + 0.282682270461039*v_W1lv_a_i - 1.69609362276623*v_W1lv_a_r - 0.282682270461039*v_W1lv_b_i + 1.69609362276623*v_W1lv_b_r - 7.27570400310194*v_W1mv_a_i + 6.02663624833782*v_W1mv_a_r + 2.08118569972797*v_W1mv_b_i - 1.73640535376106*v_W1mv_b_r + 2.08118569972797*v_W1mv_c_i - 1.73640535376106*v_W1mv_c_r + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r
        struct[0].g[38,0] = i_W1mv_b_r + 1.69609362276623*v_W1lv_b_i + 0.282682270461039*v_W1lv_b_r - 1.69609362276623*v_W1lv_c_i - 0.282682270461039*v_W1lv_c_r + 1.73640535376106*v_W1mv_a_i + 2.08118569972797*v_W1mv_a_r - 6.02663624833782*v_W1mv_b_i - 7.27570400310194*v_W1mv_b_r + 1.73640535376106*v_W1mv_c_i + 2.08118569972797*v_W1mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r
        struct[0].g[39,0] = i_W1mv_b_i + 0.282682270461039*v_W1lv_b_i - 1.69609362276623*v_W1lv_b_r - 0.282682270461039*v_W1lv_c_i + 1.69609362276623*v_W1lv_c_r + 2.08118569972797*v_W1mv_a_i - 1.73640535376106*v_W1mv_a_r - 7.27570400310194*v_W1mv_b_i + 6.02663624833782*v_W1mv_b_r + 2.08118569972797*v_W1mv_c_i - 1.73640535376106*v_W1mv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r
        struct[0].g[40,0] = i_W1mv_c_r - 1.69609362276623*v_W1lv_a_i - 0.282682270461039*v_W1lv_a_r + 1.69609362276623*v_W1lv_c_i + 0.282682270461039*v_W1lv_c_r + 1.73640535376106*v_W1mv_a_i + 2.08118569972797*v_W1mv_a_r + 1.73640535376106*v_W1mv_b_i + 2.08118569972797*v_W1mv_b_r - 6.02663624833782*v_W1mv_c_i - 7.27570400310194*v_W1mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r
        struct[0].g[41,0] = i_W1mv_c_i - 0.282682270461039*v_W1lv_a_i + 1.69609362276623*v_W1lv_a_r + 0.282682270461039*v_W1lv_c_i - 1.69609362276623*v_W1lv_c_r + 2.08118569972797*v_W1mv_a_i - 1.73640535376106*v_W1mv_a_r + 2.08118569972797*v_W1mv_b_i - 1.73640535376106*v_W1mv_b_r - 7.27570400310194*v_W1mv_c_i + 6.02663624833782*v_W1mv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r
        struct[0].g[42,0] = i_W2mv_a_r + 5.95911318666618*v_W1mv_a_i + 7.26444274184068*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.69609362276623*v_W2lv_a_i + 0.282682270461039*v_W2lv_a_r - 1.69609362276623*v_W2lv_b_i - 0.282682270461039*v_W2lv_b_r - 11.9857049291081*v_W2mv_a_i - 14.5401467449426*v_W2mv_a_r + 3.43902692373834*v_W2mv_b_i + 4.1567407688253*v_W2mv_b_r + 3.43902692373834*v_W2mv_c_i + 4.1567407688253*v_W2mv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[43,0] = i_W2mv_a_i + 7.26444274184068*v_W1mv_a_i - 5.95911318666618*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 0.282682270461039*v_W2lv_a_i - 1.69609362276623*v_W2lv_a_r - 0.282682270461039*v_W2lv_b_i + 1.69609362276623*v_W2lv_b_r - 14.5401467449426*v_W2mv_a_i + 11.9857049291081*v_W2mv_a_r + 4.1567407688253*v_W2mv_b_i - 3.43902692373834*v_W2mv_b_r + 4.1567407688253*v_W2mv_c_i - 3.43902692373834*v_W2mv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[44,0] = i_W2mv_b_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r + 5.95911318666618*v_W1mv_b_i + 7.26444274184068*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.69609362276623*v_W2lv_b_i + 0.282682270461039*v_W2lv_b_r - 1.69609362276623*v_W2lv_c_i - 0.282682270461039*v_W2lv_c_r + 3.43902692373834*v_W2mv_a_i + 4.1567407688253*v_W2mv_a_r - 11.9857049291081*v_W2mv_b_i - 14.5401467449426*v_W2mv_b_r + 3.43902692373834*v_W2mv_c_i + 4.15674076882531*v_W2mv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[45,0] = i_W2mv_b_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r + 7.26444274184068*v_W1mv_b_i - 5.95911318666618*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 0.282682270461039*v_W2lv_b_i - 1.69609362276623*v_W2lv_b_r - 0.282682270461039*v_W2lv_c_i + 1.69609362276623*v_W2lv_c_r + 4.1567407688253*v_W2mv_a_i - 3.43902692373834*v_W2mv_a_r - 14.5401467449426*v_W2mv_b_i + 11.9857049291081*v_W2mv_b_r + 4.15674076882531*v_W2mv_c_i - 3.43902692373834*v_W2mv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[46,0] = i_W2mv_c_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r + 5.95911318666618*v_W1mv_c_i + 7.26444274184068*v_W1mv_c_r - 1.69609362276623*v_W2lv_a_i - 0.282682270461039*v_W2lv_a_r + 1.69609362276623*v_W2lv_c_i + 0.282682270461039*v_W2lv_c_r + 3.43902692373834*v_W2mv_a_i + 4.1567407688253*v_W2mv_a_r + 3.43902692373834*v_W2mv_b_i + 4.15674076882531*v_W2mv_b_r - 11.9857049291081*v_W2mv_c_i - 14.5401467449426*v_W2mv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[47,0] = i_W2mv_c_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r + 7.26444274184068*v_W1mv_c_i - 5.95911318666618*v_W1mv_c_r - 0.282682270461039*v_W2lv_a_i + 1.69609362276623*v_W2lv_a_r + 0.282682270461039*v_W2lv_c_i - 1.69609362276623*v_W2lv_c_r + 4.1567407688253*v_W2mv_a_i - 3.43902692373834*v_W2mv_a_r + 4.15674076882531*v_W2mv_b_i - 3.43902692373834*v_W2mv_b_r - 14.5401467449426*v_W2mv_c_i + 11.9857049291081*v_W2mv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[48,0] = i_W3mv_a_r + 5.95911318666618*v_POImv_a_i + 7.26444274184068*v_POImv_a_r - 1.70260376761891*v_POImv_b_i - 2.07555506909734*v_POImv_b_r - 1.70260376761891*v_POImv_c_i - 2.07555506909734*v_POImv_c_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.69609362276623*v_W3lv_a_i + 0.282682270461039*v_W3lv_a_r - 1.69609362276623*v_W3lv_b_i - 0.282682270461039*v_W3lv_b_r - 11.9857049291081*v_W3mv_a_i - 14.5401467449426*v_W3mv_a_r + 3.43902692373834*v_W3mv_b_i + 4.1567407688253*v_W3mv_b_r + 3.43902692373834*v_W3mv_c_i + 4.1567407688253*v_W3mv_c_r
        struct[0].g[49,0] = i_W3mv_a_i + 7.26444274184068*v_POImv_a_i - 5.95911318666618*v_POImv_a_r - 2.07555506909734*v_POImv_b_i + 1.70260376761891*v_POImv_b_r - 2.07555506909734*v_POImv_c_i + 1.70260376761891*v_POImv_c_r + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 0.282682270461039*v_W3lv_a_i - 1.69609362276623*v_W3lv_a_r - 0.282682270461039*v_W3lv_b_i + 1.69609362276623*v_W3lv_b_r - 14.5401467449426*v_W3mv_a_i + 11.9857049291081*v_W3mv_a_r + 4.1567407688253*v_W3mv_b_i - 3.43902692373834*v_W3mv_b_r + 4.1567407688253*v_W3mv_c_i - 3.43902692373834*v_W3mv_c_r
        struct[0].g[50,0] = i_W3mv_b_r - 1.70260376761891*v_POImv_a_i - 2.07555506909734*v_POImv_a_r + 5.95911318666618*v_POImv_b_i + 7.26444274184068*v_POImv_b_r - 1.70260376761891*v_POImv_c_i - 2.07555506909734*v_POImv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.69609362276623*v_W3lv_b_i + 0.282682270461039*v_W3lv_b_r - 1.69609362276623*v_W3lv_c_i - 0.282682270461039*v_W3lv_c_r + 3.43902692373834*v_W3mv_a_i + 4.1567407688253*v_W3mv_a_r - 11.9857049291081*v_W3mv_b_i - 14.5401467449426*v_W3mv_b_r + 3.43902692373834*v_W3mv_c_i + 4.15674076882531*v_W3mv_c_r
        struct[0].g[51,0] = i_W3mv_b_i - 2.07555506909734*v_POImv_a_i + 1.70260376761891*v_POImv_a_r + 7.26444274184068*v_POImv_b_i - 5.95911318666618*v_POImv_b_r - 2.07555506909734*v_POImv_c_i + 1.70260376761891*v_POImv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 0.282682270461039*v_W3lv_b_i - 1.69609362276623*v_W3lv_b_r - 0.282682270461039*v_W3lv_c_i + 1.69609362276623*v_W3lv_c_r + 4.1567407688253*v_W3mv_a_i - 3.43902692373834*v_W3mv_a_r - 14.5401467449426*v_W3mv_b_i + 11.9857049291081*v_W3mv_b_r + 4.15674076882531*v_W3mv_c_i - 3.43902692373834*v_W3mv_c_r
        struct[0].g[52,0] = i_W3mv_c_r - 1.70260376761891*v_POImv_a_i - 2.07555506909734*v_POImv_a_r - 1.70260376761891*v_POImv_b_i - 2.07555506909734*v_POImv_b_r + 5.95911318666618*v_POImv_c_i + 7.26444274184068*v_POImv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r - 1.69609362276623*v_W3lv_a_i - 0.282682270461039*v_W3lv_a_r + 1.69609362276623*v_W3lv_c_i + 0.282682270461039*v_W3lv_c_r + 3.43902692373834*v_W3mv_a_i + 4.1567407688253*v_W3mv_a_r + 3.43902692373834*v_W3mv_b_i + 4.15674076882531*v_W3mv_b_r - 11.9857049291081*v_W3mv_c_i - 14.5401467449426*v_W3mv_c_r
        struct[0].g[53,0] = i_W3mv_c_i - 2.07555506909734*v_POImv_a_i + 1.70260376761891*v_POImv_a_r - 2.07555506909734*v_POImv_b_i + 1.70260376761891*v_POImv_b_r + 7.26444274184068*v_POImv_c_i - 5.95911318666618*v_POImv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r - 0.282682270461039*v_W3lv_a_i + 1.69609362276623*v_W3lv_a_r + 0.282682270461039*v_W3lv_c_i - 1.69609362276623*v_W3lv_c_r + 4.1567407688253*v_W3mv_a_i - 3.43902692373834*v_W3mv_a_r + 4.15674076882531*v_W3mv_b_i - 3.43902692373834*v_W3mv_b_r - 14.5401467449426*v_W3mv_c_i + 11.9857049291081*v_W3mv_c_r
        struct[0].g[54,0] = i_STmv_a_r + 148.977829666654*v_POImv_a_i + 181.611068546017*v_POImv_a_r - 42.5650941904727*v_POImv_b_i - 51.8888767274334*v_POImv_b_r - 42.5650941904727*v_POImv_c_i - 51.8888767274334*v_POImv_c_r + 1.69609362276623*v_STlv_a_i + 0.282682270461039*v_STlv_a_r - 1.69609362276623*v_STlv_b_i - 0.282682270461039*v_STlv_b_r - 149.045395453986*v_STmv_a_i - 181.622329807278*v_STmv_a_r + 42.5988786863508*v_STmv_b_i + 51.8945073580641*v_STmv_b_r + 42.5988786863508*v_STmv_c_i + 51.894507358064*v_STmv_c_r
        struct[0].g[55,0] = i_STmv_a_i + 181.611068546017*v_POImv_a_i - 148.977829666654*v_POImv_a_r - 51.8888767274334*v_POImv_b_i + 42.5650941904727*v_POImv_b_r - 51.8888767274334*v_POImv_c_i + 42.5650941904727*v_POImv_c_r + 0.282682270461039*v_STlv_a_i - 1.69609362276623*v_STlv_a_r - 0.282682270461039*v_STlv_b_i + 1.69609362276623*v_STlv_b_r - 181.622329807278*v_STmv_a_i + 149.045395453986*v_STmv_a_r + 51.8945073580641*v_STmv_b_i - 42.5988786863508*v_STmv_b_r + 51.894507358064*v_STmv_c_i - 42.5988786863508*v_STmv_c_r
        struct[0].g[56,0] = i_STmv_b_r - 42.5650941904727*v_POImv_a_i - 51.8888767274334*v_POImv_a_r + 148.977829666654*v_POImv_b_i + 181.611068546017*v_POImv_b_r - 42.5650941904727*v_POImv_c_i - 51.8888767274335*v_POImv_c_r + 1.69609362276623*v_STlv_b_i + 0.282682270461039*v_STlv_b_r - 1.69609362276623*v_STlv_c_i - 0.282682270461039*v_STlv_c_r + 42.5988786863508*v_STmv_a_i + 51.8945073580641*v_STmv_a_r - 149.045395453986*v_STmv_b_i - 181.622329807278*v_STmv_b_r + 42.5988786863508*v_STmv_c_i + 51.8945073580641*v_STmv_c_r
        struct[0].g[57,0] = i_STmv_b_i - 51.8888767274334*v_POImv_a_i + 42.5650941904727*v_POImv_a_r + 181.611068546017*v_POImv_b_i - 148.977829666654*v_POImv_b_r - 51.8888767274335*v_POImv_c_i + 42.5650941904727*v_POImv_c_r + 0.282682270461039*v_STlv_b_i - 1.69609362276623*v_STlv_b_r - 0.282682270461039*v_STlv_c_i + 1.69609362276623*v_STlv_c_r + 51.8945073580641*v_STmv_a_i - 42.5988786863508*v_STmv_a_r - 181.622329807278*v_STmv_b_i + 149.045395453986*v_STmv_b_r + 51.8945073580641*v_STmv_c_i - 42.5988786863508*v_STmv_c_r
        struct[0].g[58,0] = i_STmv_c_r - 42.5650941904727*v_POImv_a_i - 51.8888767274334*v_POImv_a_r - 42.5650941904727*v_POImv_b_i - 51.8888767274335*v_POImv_b_r + 148.977829666654*v_POImv_c_i + 181.611068546017*v_POImv_c_r - 1.69609362276623*v_STlv_a_i - 0.282682270461039*v_STlv_a_r + 1.69609362276623*v_STlv_c_i + 0.282682270461039*v_STlv_c_r + 42.5988786863508*v_STmv_a_i + 51.8945073580641*v_STmv_a_r + 42.5988786863508*v_STmv_b_i + 51.8945073580641*v_STmv_b_r - 149.045395453986*v_STmv_c_i - 181.622329807278*v_STmv_c_r
        struct[0].g[59,0] = i_STmv_c_i - 51.8888767274334*v_POImv_a_i + 42.5650941904727*v_POImv_a_r - 51.8888767274335*v_POImv_b_i + 42.5650941904727*v_POImv_b_r + 181.611068546017*v_POImv_c_i - 148.977829666654*v_POImv_c_r - 0.282682270461039*v_STlv_a_i + 1.69609362276623*v_STlv_a_r + 0.282682270461039*v_STlv_c_i - 1.69609362276623*v_STlv_c_r + 51.8945073580641*v_STmv_a_i - 42.5988786863508*v_STmv_a_r + 51.8945073580641*v_STmv_b_i - 42.5988786863508*v_STmv_b_r - 181.622329807278*v_STmv_c_i + 149.045395453986*v_STmv_c_r
        struct[0].g[60,0] = -i_l_W1mv_W2mv_a_r + 5.95911318666618*v_W1mv_a_i + 7.26444274184068*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r - 5.95911318666618*v_W2mv_a_i - 7.26444274184068*v_W2mv_a_r + 1.70260376761891*v_W2mv_b_i + 2.07555506909734*v_W2mv_b_r + 1.70260376761891*v_W2mv_c_i + 2.07555506909734*v_W2mv_c_r
        struct[0].g[61,0] = -i_l_W1mv_W2mv_a_i + 7.26444274184068*v_W1mv_a_i - 5.95911318666618*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r - 7.26444274184068*v_W2mv_a_i + 5.95911318666618*v_W2mv_a_r + 2.07555506909734*v_W2mv_b_i - 1.70260376761891*v_W2mv_b_r + 2.07555506909734*v_W2mv_c_i - 1.70260376761891*v_W2mv_c_r
        struct[0].g[62,0] = -i_l_W1mv_W2mv_b_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r + 5.95911318666618*v_W1mv_b_i + 7.26444274184068*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.70260376761891*v_W2mv_a_i + 2.07555506909734*v_W2mv_a_r - 5.95911318666618*v_W2mv_b_i - 7.26444274184068*v_W2mv_b_r + 1.70260376761891*v_W2mv_c_i + 2.07555506909734*v_W2mv_c_r
        struct[0].g[63,0] = -i_l_W1mv_W2mv_b_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r + 7.26444274184068*v_W1mv_b_i - 5.95911318666618*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 2.07555506909734*v_W2mv_a_i - 1.70260376761891*v_W2mv_a_r - 7.26444274184068*v_W2mv_b_i + 5.95911318666618*v_W2mv_b_r + 2.07555506909734*v_W2mv_c_i - 1.70260376761891*v_W2mv_c_r
        struct[0].g[64,0] = -i_l_W1mv_W2mv_c_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r + 5.95911318666618*v_W1mv_c_i + 7.26444274184068*v_W1mv_c_r + 1.70260376761891*v_W2mv_a_i + 2.07555506909734*v_W2mv_a_r + 1.70260376761891*v_W2mv_b_i + 2.07555506909734*v_W2mv_b_r - 5.95911318666618*v_W2mv_c_i - 7.26444274184068*v_W2mv_c_r
        struct[0].g[65,0] = -i_l_W1mv_W2mv_c_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r + 7.26444274184068*v_W1mv_c_i - 5.95911318666618*v_W1mv_c_r + 2.07555506909734*v_W2mv_a_i - 1.70260376761891*v_W2mv_a_r + 2.07555506909734*v_W2mv_b_i - 1.70260376761891*v_W2mv_b_r - 7.26444274184068*v_W2mv_c_i + 5.95911318666618*v_W2mv_c_r
        struct[0].g[66,0] = -i_l_W2mv_W3mv_a_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r - 5.95911318666618*v_W3mv_a_i - 7.26444274184068*v_W3mv_a_r + 1.70260376761891*v_W3mv_b_i + 2.07555506909734*v_W3mv_b_r + 1.70260376761891*v_W3mv_c_i + 2.07555506909734*v_W3mv_c_r
        struct[0].g[67,0] = -i_l_W2mv_W3mv_a_i + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r - 7.26444274184068*v_W3mv_a_i + 5.95911318666618*v_W3mv_a_r + 2.07555506909734*v_W3mv_b_i - 1.70260376761891*v_W3mv_b_r + 2.07555506909734*v_W3mv_c_i - 1.70260376761891*v_W3mv_c_r
        struct[0].g[68,0] = -i_l_W2mv_W3mv_b_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.70260376761891*v_W3mv_a_i + 2.07555506909734*v_W3mv_a_r - 5.95911318666618*v_W3mv_b_i - 7.26444274184068*v_W3mv_b_r + 1.70260376761891*v_W3mv_c_i + 2.07555506909734*v_W3mv_c_r
        struct[0].g[69,0] = -i_l_W2mv_W3mv_b_i - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 2.07555506909734*v_W3mv_a_i - 1.70260376761891*v_W3mv_a_r - 7.26444274184068*v_W3mv_b_i + 5.95911318666618*v_W3mv_b_r + 2.07555506909734*v_W3mv_c_i - 1.70260376761891*v_W3mv_c_r
        struct[0].g[70,0] = -i_l_W2mv_W3mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r + 1.70260376761891*v_W3mv_a_i + 2.07555506909734*v_W3mv_a_r + 1.70260376761891*v_W3mv_b_i + 2.07555506909734*v_W3mv_b_r - 5.95911318666618*v_W3mv_c_i - 7.26444274184068*v_W3mv_c_r
        struct[0].g[71,0] = -i_l_W2mv_W3mv_c_i - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r + 2.07555506909734*v_W3mv_a_i - 1.70260376761891*v_W3mv_a_r + 2.07555506909734*v_W3mv_b_i - 1.70260376761891*v_W3mv_b_r - 7.26444274184068*v_W3mv_c_i + 5.95911318666618*v_W3mv_c_r
        struct[0].g[72,0] = -i_l_W3mv_POImv_a_r - 5.95911318666618*v_POImv_a_i - 7.26444274184068*v_POImv_a_r + 1.70260376761891*v_POImv_b_i + 2.07555506909734*v_POImv_b_r + 1.70260376761891*v_POImv_c_i + 2.07555506909734*v_POImv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[73,0] = -i_l_W3mv_POImv_a_i - 7.26444274184068*v_POImv_a_i + 5.95911318666618*v_POImv_a_r + 2.07555506909734*v_POImv_b_i - 1.70260376761891*v_POImv_b_r + 2.07555506909734*v_POImv_c_i - 1.70260376761891*v_POImv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[74,0] = -i_l_W3mv_POImv_b_r + 1.70260376761891*v_POImv_a_i + 2.07555506909734*v_POImv_a_r - 5.95911318666618*v_POImv_b_i - 7.26444274184068*v_POImv_b_r + 1.70260376761891*v_POImv_c_i + 2.07555506909734*v_POImv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[75,0] = -i_l_W3mv_POImv_b_i + 2.07555506909734*v_POImv_a_i - 1.70260376761891*v_POImv_a_r - 7.26444274184068*v_POImv_b_i + 5.95911318666618*v_POImv_b_r + 2.07555506909734*v_POImv_c_i - 1.70260376761891*v_POImv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[76,0] = -i_l_W3mv_POImv_c_r + 1.70260376761891*v_POImv_a_i + 2.07555506909734*v_POImv_a_r + 1.70260376761891*v_POImv_b_i + 2.07555506909734*v_POImv_b_r - 5.95911318666618*v_POImv_c_i - 7.26444274184068*v_POImv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[77,0] = -i_l_W3mv_POImv_c_i + 2.07555506909734*v_POImv_a_i - 1.70260376761891*v_POImv_a_r + 2.07555506909734*v_POImv_b_i - 1.70260376761891*v_POImv_b_r - 7.26444274184068*v_POImv_c_i + 5.95911318666618*v_POImv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[78,0] = -i_l_STmv_POImv_a_r - 148.977829666654*v_POImv_a_i - 181.611068546017*v_POImv_a_r + 42.5650941904727*v_POImv_b_i + 51.8888767274334*v_POImv_b_r + 42.5650941904727*v_POImv_c_i + 51.8888767274334*v_POImv_c_r + 148.977829666654*v_STmv_a_i + 181.611068546017*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274334*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274334*v_STmv_c_r
        struct[0].g[79,0] = -i_l_STmv_POImv_a_i - 181.611068546017*v_POImv_a_i + 148.977829666654*v_POImv_a_r + 51.8888767274334*v_POImv_b_i - 42.5650941904727*v_POImv_b_r + 51.8888767274334*v_POImv_c_i - 42.5650941904727*v_POImv_c_r + 181.611068546017*v_STmv_a_i - 148.977829666654*v_STmv_a_r - 51.8888767274334*v_STmv_b_i + 42.5650941904727*v_STmv_b_r - 51.8888767274334*v_STmv_c_i + 42.5650941904727*v_STmv_c_r
        struct[0].g[80,0] = -i_l_STmv_POImv_b_r + 42.5650941904727*v_POImv_a_i + 51.8888767274334*v_POImv_a_r - 148.977829666654*v_POImv_b_i - 181.611068546017*v_POImv_b_r + 42.5650941904727*v_POImv_c_i + 51.8888767274335*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r + 148.977829666654*v_STmv_b_i + 181.611068546017*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274335*v_STmv_c_r
        struct[0].g[81,0] = -i_l_STmv_POImv_b_i + 51.8888767274334*v_POImv_a_i - 42.5650941904727*v_POImv_a_r - 181.611068546017*v_POImv_b_i + 148.977829666654*v_POImv_b_r + 51.8888767274335*v_POImv_c_i - 42.5650941904727*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r + 181.611068546017*v_STmv_b_i - 148.977829666654*v_STmv_b_r - 51.8888767274335*v_STmv_c_i + 42.5650941904727*v_STmv_c_r
        struct[0].g[82,0] = -i_l_STmv_POImv_c_r + 42.5650941904727*v_POImv_a_i + 51.8888767274334*v_POImv_a_r + 42.5650941904727*v_POImv_b_i + 51.8888767274335*v_POImv_b_r - 148.977829666654*v_POImv_c_i - 181.611068546017*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274335*v_STmv_b_r + 148.977829666654*v_STmv_c_i + 181.611068546017*v_STmv_c_r
        struct[0].g[83,0] = -i_l_STmv_POImv_c_i + 51.8888767274334*v_POImv_a_i - 42.5650941904727*v_POImv_a_r + 51.8888767274335*v_POImv_b_i - 42.5650941904727*v_POImv_b_r - 181.611068546017*v_POImv_c_i + 148.977829666654*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r - 51.8888767274335*v_STmv_b_i + 42.5650941904727*v_STmv_b_r + 181.611068546017*v_STmv_c_i - 148.977829666654*v_STmv_c_r
        struct[0].g[84,0] = -i_l_POI_GRID_a_r - 0.040290088638195*v_GRID_a_i - 0.024174053182917*v_GRID_a_r - 4.66248501556824e-18*v_GRID_b_i + 4.31760362252812e-18*v_GRID_b_r - 4.19816664496737e-18*v_GRID_c_i + 3.49608108880335e-18*v_GRID_c_r + 0.040290088638195*v_POI_a_i + 0.024174053182917*v_POI_a_r + 4.66248501556824e-18*v_POI_b_i - 4.31760362252812e-18*v_POI_b_r + 4.19816664496737e-18*v_POI_c_i - 3.49608108880335e-18*v_POI_c_r
        struct[0].g[85,0] = -i_l_POI_GRID_a_i - 0.024174053182917*v_GRID_a_i + 0.040290088638195*v_GRID_a_r + 4.31760362252812e-18*v_GRID_b_i + 4.66248501556824e-18*v_GRID_b_r + 3.49608108880335e-18*v_GRID_c_i + 4.19816664496737e-18*v_GRID_c_r + 0.024174053182917*v_POI_a_i - 0.040290088638195*v_POI_a_r - 4.31760362252812e-18*v_POI_b_i - 4.66248501556824e-18*v_POI_b_r - 3.49608108880335e-18*v_POI_c_i - 4.19816664496737e-18*v_POI_c_r
        struct[0].g[86,0] = -i_l_POI_GRID_b_r - 6.30775359573304e-19*v_GRID_a_i + 2.07254761002657e-18*v_GRID_a_r - 0.040290088638195*v_GRID_b_i - 0.024174053182917*v_GRID_b_r - 9.01107656533306e-19*v_GRID_c_i + 1.78419315993592e-17*v_GRID_c_r + 6.30775359573304e-19*v_POI_a_i - 2.07254761002657e-18*v_POI_a_r + 0.040290088638195*v_POI_b_i + 0.024174053182917*v_POI_b_r + 9.01107656533306e-19*v_POI_c_i - 1.78419315993592e-17*v_POI_c_r
        struct[0].g[87,0] = -i_l_POI_GRID_b_i + 2.07254761002657e-18*v_GRID_a_i + 6.30775359573304e-19*v_GRID_a_r - 0.024174053182917*v_GRID_b_i + 0.040290088638195*v_GRID_b_r + 1.78419315993592e-17*v_GRID_c_i + 9.01107656533306e-19*v_GRID_c_r - 2.07254761002657e-18*v_POI_a_i - 6.30775359573304e-19*v_POI_a_r + 0.024174053182917*v_POI_b_i - 0.040290088638195*v_POI_b_r - 1.78419315993592e-17*v_POI_c_i - 9.01107656533306e-19*v_POI_c_r
        struct[0].g[88,0] = -i_l_POI_GRID_c_r + 7.20886125226632e-19*v_GRID_a_i + 1.35166148479994e-18*v_GRID_a_r + 4.50553828266631e-19*v_GRID_b_i + 1.71210454741325e-17*v_GRID_b_r - 0.040290088638195*v_GRID_c_i - 0.024174053182917*v_GRID_c_r - 7.20886125226632e-19*v_POI_a_i - 1.35166148479994e-18*v_POI_a_r - 4.50553828266631e-19*v_POI_b_i - 1.71210454741325e-17*v_POI_b_r + 0.040290088638195*v_POI_c_i + 0.024174053182917*v_POI_c_r
        struct[0].g[89,0] = -i_l_POI_GRID_c_i + 1.35166148479994e-18*v_GRID_a_i - 7.20886125226632e-19*v_GRID_a_r + 1.71210454741325e-17*v_GRID_b_i - 4.50553828266631e-19*v_GRID_b_r - 0.024174053182917*v_GRID_c_i + 0.040290088638195*v_GRID_c_r - 1.35166148479994e-18*v_POI_a_i + 7.20886125226632e-19*v_POI_a_r - 1.71210454741325e-17*v_POI_b_i + 4.50553828266631e-19*v_POI_b_r + 0.024174053182917*v_POI_c_i - 0.040290088638195*v_POI_c_r
        struct[0].g[90,0] = i_W1lv_a_i*v_W1lv_a_i + i_W1lv_a_r*v_W1lv_a_r - p_W1lv_a
        struct[0].g[91,0] = i_W1lv_b_i*v_W1lv_b_i + i_W1lv_b_r*v_W1lv_b_r - p_W1lv_b
        struct[0].g[92,0] = i_W1lv_c_i*v_W1lv_c_i + i_W1lv_c_r*v_W1lv_c_r - p_W1lv_c
        struct[0].g[93,0] = -i_W1lv_a_i*v_W1lv_a_r + i_W1lv_a_r*v_W1lv_a_i - q_W1lv_a
        struct[0].g[94,0] = -i_W1lv_b_i*v_W1lv_b_r + i_W1lv_b_r*v_W1lv_b_i - q_W1lv_b
        struct[0].g[95,0] = -i_W1lv_c_i*v_W1lv_c_r + i_W1lv_c_r*v_W1lv_c_i - q_W1lv_c
        struct[0].g[96,0] = -v_m_W1lv + (v_W1lv_a_i**2 + v_W1lv_a_r**2)**0.5/V_base_W1lv
        struct[0].g[97,0] = -v_m_W1mv + (v_W1mv_a_i**2 + v_W1mv_a_r**2)**0.5/V_base_W1mv
        struct[0].g[98,0] = Dq_r_W1lv + K_p_v_W1lv*(Dv_r_W1lv - u_ctrl_v_W1lv*v_m_W1mv + v_loc_ref_W1lv - v_m_W1lv*(1.0 - u_ctrl_v_W1lv)) - i_reac_ref_W1lv
        struct[0].g[99,0] = S_base_W1lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)])) - q_ref_W1lv
        struct[0].g[100,0] = i_W2lv_a_i*v_W2lv_a_i + i_W2lv_a_r*v_W2lv_a_r - p_W2lv_a
        struct[0].g[101,0] = i_W2lv_b_i*v_W2lv_b_i + i_W2lv_b_r*v_W2lv_b_r - p_W2lv_b
        struct[0].g[102,0] = i_W2lv_c_i*v_W2lv_c_i + i_W2lv_c_r*v_W2lv_c_r - p_W2lv_c
        struct[0].g[103,0] = -i_W2lv_a_i*v_W2lv_a_r + i_W2lv_a_r*v_W2lv_a_i - q_W2lv_a
        struct[0].g[104,0] = -i_W2lv_b_i*v_W2lv_b_r + i_W2lv_b_r*v_W2lv_b_i - q_W2lv_b
        struct[0].g[105,0] = -i_W2lv_c_i*v_W2lv_c_r + i_W2lv_c_r*v_W2lv_c_i - q_W2lv_c
        struct[0].g[106,0] = -v_m_W2lv + (v_W2lv_a_i**2 + v_W2lv_a_r**2)**0.5/V_base_W2lv
        struct[0].g[107,0] = -v_m_W2mv + (v_W2mv_a_i**2 + v_W2mv_a_r**2)**0.5/V_base_W2mv
        struct[0].g[108,0] = Dq_r_W2lv + K_p_v_W2lv*(Dv_r_W2lv - u_ctrl_v_W2lv*v_m_W2mv + v_loc_ref_W2lv - v_m_W2lv*(1.0 - u_ctrl_v_W2lv)) - i_reac_ref_W2lv
        struct[0].g[109,0] = S_base_W2lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)])) - q_ref_W2lv
        struct[0].g[110,0] = i_W3lv_a_i*v_W3lv_a_i + i_W3lv_a_r*v_W3lv_a_r - p_W3lv_a
        struct[0].g[111,0] = i_W3lv_b_i*v_W3lv_b_i + i_W3lv_b_r*v_W3lv_b_r - p_W3lv_b
        struct[0].g[112,0] = i_W3lv_c_i*v_W3lv_c_i + i_W3lv_c_r*v_W3lv_c_r - p_W3lv_c
        struct[0].g[113,0] = -i_W3lv_a_i*v_W3lv_a_r + i_W3lv_a_r*v_W3lv_a_i - q_W3lv_a
        struct[0].g[114,0] = -i_W3lv_b_i*v_W3lv_b_r + i_W3lv_b_r*v_W3lv_b_i - q_W3lv_b
        struct[0].g[115,0] = -i_W3lv_c_i*v_W3lv_c_r + i_W3lv_c_r*v_W3lv_c_i - q_W3lv_c
        struct[0].g[116,0] = -v_m_W3lv + (v_W3lv_a_i**2 + v_W3lv_a_r**2)**0.5/V_base_W3lv
        struct[0].g[117,0] = -v_m_W3mv + (v_W3mv_a_i**2 + v_W3mv_a_r**2)**0.5/V_base_W3mv
        struct[0].g[118,0] = Dq_r_W3lv + K_p_v_W3lv*(Dv_r_W3lv - u_ctrl_v_W3lv*v_m_W3mv + v_loc_ref_W3lv - v_m_W3lv*(1.0 - u_ctrl_v_W3lv)) - i_reac_ref_W3lv
        struct[0].g[119,0] = S_base_W3lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)])) - q_ref_W3lv
        struct[0].g[120,0] = i_STlv_a_i*v_STlv_a_i + i_STlv_a_r*v_STlv_a_r - p_STlv_a
        struct[0].g[121,0] = i_STlv_b_i*v_STlv_b_i + i_STlv_b_r*v_STlv_b_r - p_STlv_b
        struct[0].g[122,0] = i_STlv_c_i*v_STlv_c_i + i_STlv_c_r*v_STlv_c_r - p_STlv_c
        struct[0].g[123,0] = -i_STlv_a_i*v_STlv_a_r + i_STlv_a_r*v_STlv_a_i - q_STlv_a
        struct[0].g[124,0] = -i_STlv_b_i*v_STlv_b_r + i_STlv_b_r*v_STlv_b_i - q_STlv_b
        struct[0].g[125,0] = -i_STlv_c_i*v_STlv_c_r + i_STlv_c_r*v_STlv_c_i - q_STlv_c
        struct[0].g[126,0] = -v_m_STlv + (v_STlv_a_i**2 + v_STlv_a_r**2)**0.5/V_base_STlv
        struct[0].g[127,0] = -v_m_STmv + (v_STmv_a_i**2 + v_STmv_a_r**2)**0.5/V_base_STmv
        struct[0].g[128,0] = Dq_r_STlv + K_p_v_STlv*(Dv_r_STlv - u_ctrl_v_STlv*v_m_STmv + v_loc_ref_STlv - v_m_STlv*(1.0 - u_ctrl_v_STlv)) - i_reac_ref_STlv
        struct[0].g[129,0] = S_base_STlv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)])) - q_ref_STlv
    
    # Outputs:
    if mode == 3:

    
        pass

    if mode == 10:

        struct[0].Fx_ini[0,0] = -1/T_pq_W1lv
        struct[0].Fx_ini[1,1] = -1/T_pq_W1lv
        struct[0].Fx_ini[2,2] = -1/T_pq_W1lv
        struct[0].Fx_ini[3,3] = -1/T_pq_W1lv
        struct[0].Fx_ini[4,4] = -1/T_pq_W1lv
        struct[0].Fx_ini[5,5] = -1/T_pq_W1lv
        struct[0].Fx_ini[6,6] = -1/T_pq_W2lv
        struct[0].Fx_ini[7,7] = -1/T_pq_W2lv
        struct[0].Fx_ini[8,8] = -1/T_pq_W2lv
        struct[0].Fx_ini[9,9] = -1/T_pq_W2lv
        struct[0].Fx_ini[10,10] = -1/T_pq_W2lv
        struct[0].Fx_ini[11,11] = -1/T_pq_W2lv
        struct[0].Fx_ini[12,12] = -1/T_pq_W3lv
        struct[0].Fx_ini[13,13] = -1/T_pq_W3lv
        struct[0].Fx_ini[14,14] = -1/T_pq_W3lv
        struct[0].Fx_ini[15,15] = -1/T_pq_W3lv
        struct[0].Fx_ini[16,16] = -1/T_pq_W3lv
        struct[0].Fx_ini[17,17] = -1/T_pq_W3lv
        struct[0].Fx_ini[18,18] = -1/T_pq_STlv
        struct[0].Fx_ini[19,19] = -1/T_pq_STlv
        struct[0].Fx_ini[20,20] = -1/T_pq_STlv
        struct[0].Fx_ini[21,21] = -1/T_pq_STlv
        struct[0].Fx_ini[22,22] = -1/T_pq_STlv
        struct[0].Fx_ini[23,23] = -1/T_pq_STlv

    if mode == 11:

        struct[0].Fy_ini[3,99] = 1/(3*T_pq_W1lv) 
        struct[0].Fy_ini[4,99] = 1/(3*T_pq_W1lv) 
        struct[0].Fy_ini[5,99] = 1/(3*T_pq_W1lv) 
        struct[0].Fy_ini[9,109] = 1/(3*T_pq_W2lv) 
        struct[0].Fy_ini[10,109] = 1/(3*T_pq_W2lv) 
        struct[0].Fy_ini[11,109] = 1/(3*T_pq_W2lv) 
        struct[0].Fy_ini[15,119] = 1/(3*T_pq_W3lv) 
        struct[0].Fy_ini[16,119] = 1/(3*T_pq_W3lv) 
        struct[0].Fy_ini[17,119] = 1/(3*T_pq_W3lv) 
        struct[0].Fy_ini[21,129] = 1/(3*T_pq_STlv) 
        struct[0].Fy_ini[22,129] = 1/(3*T_pq_STlv) 
        struct[0].Fy_ini[23,129] = 1/(3*T_pq_STlv) 

        struct[0].Gx_ini[90,0] = -1
        struct[0].Gx_ini[91,1] = -1
        struct[0].Gx_ini[92,2] = -1
        struct[0].Gx_ini[93,3] = -1
        struct[0].Gx_ini[94,4] = -1
        struct[0].Gx_ini[95,5] = -1
        struct[0].Gx_ini[100,6] = -1
        struct[0].Gx_ini[101,7] = -1
        struct[0].Gx_ini[102,8] = -1
        struct[0].Gx_ini[103,9] = -1
        struct[0].Gx_ini[104,10] = -1
        struct[0].Gx_ini[105,11] = -1
        struct[0].Gx_ini[110,12] = -1
        struct[0].Gx_ini[111,13] = -1
        struct[0].Gx_ini[112,14] = -1
        struct[0].Gx_ini[113,15] = -1
        struct[0].Gx_ini[114,16] = -1
        struct[0].Gx_ini[115,17] = -1
        struct[0].Gx_ini[120,18] = -1
        struct[0].Gx_ini[121,19] = -1
        struct[0].Gx_ini[122,20] = -1
        struct[0].Gx_ini[123,21] = -1
        struct[0].Gx_ini[124,22] = -1
        struct[0].Gx_ini[125,23] = -1

        struct[0].Gy_ini[90,0] = i_W1lv_a_r
        struct[0].Gy_ini[90,1] = i_W1lv_a_i
        struct[0].Gy_ini[90,90] = v_W1lv_a_r
        struct[0].Gy_ini[90,91] = v_W1lv_a_i
        struct[0].Gy_ini[91,2] = i_W1lv_b_r
        struct[0].Gy_ini[91,3] = i_W1lv_b_i
        struct[0].Gy_ini[91,92] = v_W1lv_b_r
        struct[0].Gy_ini[91,93] = v_W1lv_b_i
        struct[0].Gy_ini[92,4] = i_W1lv_c_r
        struct[0].Gy_ini[92,5] = i_W1lv_c_i
        struct[0].Gy_ini[92,94] = v_W1lv_c_r
        struct[0].Gy_ini[92,95] = v_W1lv_c_i
        struct[0].Gy_ini[93,0] = -i_W1lv_a_i
        struct[0].Gy_ini[93,1] = i_W1lv_a_r
        struct[0].Gy_ini[93,90] = v_W1lv_a_i
        struct[0].Gy_ini[93,91] = -v_W1lv_a_r
        struct[0].Gy_ini[94,2] = -i_W1lv_b_i
        struct[0].Gy_ini[94,3] = i_W1lv_b_r
        struct[0].Gy_ini[94,92] = v_W1lv_b_i
        struct[0].Gy_ini[94,93] = -v_W1lv_b_r
        struct[0].Gy_ini[95,4] = -i_W1lv_c_i
        struct[0].Gy_ini[95,5] = i_W1lv_c_r
        struct[0].Gy_ini[95,94] = v_W1lv_c_i
        struct[0].Gy_ini[95,95] = -v_W1lv_c_r
        struct[0].Gy_ini[96,0] = 1.0*v_W1lv_a_r*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)/V_base_W1lv
        struct[0].Gy_ini[96,1] = 1.0*v_W1lv_a_i*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)/V_base_W1lv
        struct[0].Gy_ini[97,36] = 1.0*v_W1mv_a_r*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)/V_base_W1mv
        struct[0].Gy_ini[97,37] = 1.0*v_W1mv_a_i*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)/V_base_W1mv
        struct[0].Gy_ini[98,96] = K_p_v_W1lv*(u_ctrl_v_W1lv - 1.0)
        struct[0].Gy_ini[98,97] = -K_p_v_W1lv*u_ctrl_v_W1lv
        struct[0].Gy_ini[99,58] = 1.0*S_base_W1lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)]))
        struct[0].Gy_ini[99,59] = 1.0*S_base_W1lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)]))
        struct[0].Gy_ini[99,98] = S_base_W1lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W1lv < i_reac_ref_W1lv) | (I_max_W1lv < -i_reac_ref_W1lv)), (1, True)]))
        struct[0].Gy_ini[100,6] = i_W2lv_a_r
        struct[0].Gy_ini[100,7] = i_W2lv_a_i
        struct[0].Gy_ini[100,100] = v_W2lv_a_r
        struct[0].Gy_ini[100,101] = v_W2lv_a_i
        struct[0].Gy_ini[101,8] = i_W2lv_b_r
        struct[0].Gy_ini[101,9] = i_W2lv_b_i
        struct[0].Gy_ini[101,102] = v_W2lv_b_r
        struct[0].Gy_ini[101,103] = v_W2lv_b_i
        struct[0].Gy_ini[102,10] = i_W2lv_c_r
        struct[0].Gy_ini[102,11] = i_W2lv_c_i
        struct[0].Gy_ini[102,104] = v_W2lv_c_r
        struct[0].Gy_ini[102,105] = v_W2lv_c_i
        struct[0].Gy_ini[103,6] = -i_W2lv_a_i
        struct[0].Gy_ini[103,7] = i_W2lv_a_r
        struct[0].Gy_ini[103,100] = v_W2lv_a_i
        struct[0].Gy_ini[103,101] = -v_W2lv_a_r
        struct[0].Gy_ini[104,8] = -i_W2lv_b_i
        struct[0].Gy_ini[104,9] = i_W2lv_b_r
        struct[0].Gy_ini[104,102] = v_W2lv_b_i
        struct[0].Gy_ini[104,103] = -v_W2lv_b_r
        struct[0].Gy_ini[105,10] = -i_W2lv_c_i
        struct[0].Gy_ini[105,11] = i_W2lv_c_r
        struct[0].Gy_ini[105,104] = v_W2lv_c_i
        struct[0].Gy_ini[105,105] = -v_W2lv_c_r
        struct[0].Gy_ini[106,6] = 1.0*v_W2lv_a_r*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)/V_base_W2lv
        struct[0].Gy_ini[106,7] = 1.0*v_W2lv_a_i*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)/V_base_W2lv
        struct[0].Gy_ini[107,42] = 1.0*v_W2mv_a_r*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)/V_base_W2mv
        struct[0].Gy_ini[107,43] = 1.0*v_W2mv_a_i*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)/V_base_W2mv
        struct[0].Gy_ini[108,106] = K_p_v_W2lv*(u_ctrl_v_W2lv - 1.0)
        struct[0].Gy_ini[108,107] = -K_p_v_W2lv*u_ctrl_v_W2lv
        struct[0].Gy_ini[109,58] = 1.0*S_base_W2lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)]))
        struct[0].Gy_ini[109,59] = 1.0*S_base_W2lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)]))
        struct[0].Gy_ini[109,108] = S_base_W2lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W2lv < i_reac_ref_W2lv) | (I_max_W2lv < -i_reac_ref_W2lv)), (1, True)]))
        struct[0].Gy_ini[110,12] = i_W3lv_a_r
        struct[0].Gy_ini[110,13] = i_W3lv_a_i
        struct[0].Gy_ini[110,110] = v_W3lv_a_r
        struct[0].Gy_ini[110,111] = v_W3lv_a_i
        struct[0].Gy_ini[111,14] = i_W3lv_b_r
        struct[0].Gy_ini[111,15] = i_W3lv_b_i
        struct[0].Gy_ini[111,112] = v_W3lv_b_r
        struct[0].Gy_ini[111,113] = v_W3lv_b_i
        struct[0].Gy_ini[112,16] = i_W3lv_c_r
        struct[0].Gy_ini[112,17] = i_W3lv_c_i
        struct[0].Gy_ini[112,114] = v_W3lv_c_r
        struct[0].Gy_ini[112,115] = v_W3lv_c_i
        struct[0].Gy_ini[113,12] = -i_W3lv_a_i
        struct[0].Gy_ini[113,13] = i_W3lv_a_r
        struct[0].Gy_ini[113,110] = v_W3lv_a_i
        struct[0].Gy_ini[113,111] = -v_W3lv_a_r
        struct[0].Gy_ini[114,14] = -i_W3lv_b_i
        struct[0].Gy_ini[114,15] = i_W3lv_b_r
        struct[0].Gy_ini[114,112] = v_W3lv_b_i
        struct[0].Gy_ini[114,113] = -v_W3lv_b_r
        struct[0].Gy_ini[115,16] = -i_W3lv_c_i
        struct[0].Gy_ini[115,17] = i_W3lv_c_r
        struct[0].Gy_ini[115,114] = v_W3lv_c_i
        struct[0].Gy_ini[115,115] = -v_W3lv_c_r
        struct[0].Gy_ini[116,12] = 1.0*v_W3lv_a_r*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)/V_base_W3lv
        struct[0].Gy_ini[116,13] = 1.0*v_W3lv_a_i*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)/V_base_W3lv
        struct[0].Gy_ini[117,48] = 1.0*v_W3mv_a_r*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)/V_base_W3mv
        struct[0].Gy_ini[117,49] = 1.0*v_W3mv_a_i*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)/V_base_W3mv
        struct[0].Gy_ini[118,116] = K_p_v_W3lv*(u_ctrl_v_W3lv - 1.0)
        struct[0].Gy_ini[118,117] = -K_p_v_W3lv*u_ctrl_v_W3lv
        struct[0].Gy_ini[119,58] = 1.0*S_base_W3lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)]))
        struct[0].Gy_ini[119,59] = 1.0*S_base_W3lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)]))
        struct[0].Gy_ini[119,118] = S_base_W3lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W3lv < i_reac_ref_W3lv) | (I_max_W3lv < -i_reac_ref_W3lv)), (1, True)]))
        struct[0].Gy_ini[120,18] = i_STlv_a_r
        struct[0].Gy_ini[120,19] = i_STlv_a_i
        struct[0].Gy_ini[120,120] = v_STlv_a_r
        struct[0].Gy_ini[120,121] = v_STlv_a_i
        struct[0].Gy_ini[121,20] = i_STlv_b_r
        struct[0].Gy_ini[121,21] = i_STlv_b_i
        struct[0].Gy_ini[121,122] = v_STlv_b_r
        struct[0].Gy_ini[121,123] = v_STlv_b_i
        struct[0].Gy_ini[122,22] = i_STlv_c_r
        struct[0].Gy_ini[122,23] = i_STlv_c_i
        struct[0].Gy_ini[122,124] = v_STlv_c_r
        struct[0].Gy_ini[122,125] = v_STlv_c_i
        struct[0].Gy_ini[123,18] = -i_STlv_a_i
        struct[0].Gy_ini[123,19] = i_STlv_a_r
        struct[0].Gy_ini[123,120] = v_STlv_a_i
        struct[0].Gy_ini[123,121] = -v_STlv_a_r
        struct[0].Gy_ini[124,20] = -i_STlv_b_i
        struct[0].Gy_ini[124,21] = i_STlv_b_r
        struct[0].Gy_ini[124,122] = v_STlv_b_i
        struct[0].Gy_ini[124,123] = -v_STlv_b_r
        struct[0].Gy_ini[125,22] = -i_STlv_c_i
        struct[0].Gy_ini[125,23] = i_STlv_c_r
        struct[0].Gy_ini[125,124] = v_STlv_c_i
        struct[0].Gy_ini[125,125] = -v_STlv_c_r
        struct[0].Gy_ini[126,18] = 1.0*v_STlv_a_r*(v_STlv_a_i**2 + v_STlv_a_r**2)**(-0.5)/V_base_STlv
        struct[0].Gy_ini[126,19] = 1.0*v_STlv_a_i*(v_STlv_a_i**2 + v_STlv_a_r**2)**(-0.5)/V_base_STlv
        struct[0].Gy_ini[127,54] = 1.0*v_STmv_a_r*(v_STmv_a_i**2 + v_STmv_a_r**2)**(-0.5)/V_base_STmv
        struct[0].Gy_ini[127,55] = 1.0*v_STmv_a_i*(v_STmv_a_i**2 + v_STmv_a_r**2)**(-0.5)/V_base_STmv
        struct[0].Gy_ini[128,126] = K_p_v_STlv*(u_ctrl_v_STlv - 1.0)
        struct[0].Gy_ini[128,127] = -K_p_v_STlv*u_ctrl_v_STlv
        struct[0].Gy_ini[129,58] = 1.0*S_base_STlv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)]))
        struct[0].Gy_ini[129,59] = 1.0*S_base_STlv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)]))
        struct[0].Gy_ini[129,128] = S_base_STlv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_STlv < i_reac_ref_STlv) | (I_max_STlv < -i_reac_ref_STlv)), (1, True)]))



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    u_ctrl_v_W1lv = struct[0].u_ctrl_v_W1lv
    K_p_v_W1lv = struct[0].K_p_v_W1lv
    K_i_v_W1lv = struct[0].K_i_v_W1lv
    V_base_W1lv = struct[0].V_base_W1lv
    V_base_W1mv = struct[0].V_base_W1mv
    S_base_W1lv = struct[0].S_base_W1lv
    I_max_W1lv = struct[0].I_max_W1lv
    u_ctrl_v_W2lv = struct[0].u_ctrl_v_W2lv
    K_p_v_W2lv = struct[0].K_p_v_W2lv
    K_i_v_W2lv = struct[0].K_i_v_W2lv
    V_base_W2lv = struct[0].V_base_W2lv
    V_base_W2mv = struct[0].V_base_W2mv
    S_base_W2lv = struct[0].S_base_W2lv
    I_max_W2lv = struct[0].I_max_W2lv
    u_ctrl_v_W3lv = struct[0].u_ctrl_v_W3lv
    K_p_v_W3lv = struct[0].K_p_v_W3lv
    K_i_v_W3lv = struct[0].K_i_v_W3lv
    V_base_W3lv = struct[0].V_base_W3lv
    V_base_W3mv = struct[0].V_base_W3mv
    S_base_W3lv = struct[0].S_base_W3lv
    I_max_W3lv = struct[0].I_max_W3lv
    u_ctrl_v_STlv = struct[0].u_ctrl_v_STlv
    K_p_v_STlv = struct[0].K_p_v_STlv
    K_i_v_STlv = struct[0].K_i_v_STlv
    V_base_STlv = struct[0].V_base_STlv
    V_base_STmv = struct[0].V_base_STmv
    S_base_STlv = struct[0].S_base_STlv
    I_max_STlv = struct[0].I_max_STlv
    
    # Inputs:
    v_GRID_a_r = struct[0].v_GRID_a_r
    v_GRID_a_i = struct[0].v_GRID_a_i
    v_GRID_b_r = struct[0].v_GRID_b_r
    v_GRID_b_i = struct[0].v_GRID_b_i
    v_GRID_c_r = struct[0].v_GRID_c_r
    v_GRID_c_i = struct[0].v_GRID_c_i
    i_POI_a_r = struct[0].i_POI_a_r
    i_POI_a_i = struct[0].i_POI_a_i
    i_POI_b_r = struct[0].i_POI_b_r
    i_POI_b_i = struct[0].i_POI_b_i
    i_POI_c_r = struct[0].i_POI_c_r
    i_POI_c_i = struct[0].i_POI_c_i
    i_POImv_a_r = struct[0].i_POImv_a_r
    i_POImv_a_i = struct[0].i_POImv_a_i
    i_POImv_b_r = struct[0].i_POImv_b_r
    i_POImv_b_i = struct[0].i_POImv_b_i
    i_POImv_c_r = struct[0].i_POImv_c_r
    i_POImv_c_i = struct[0].i_POImv_c_i
    i_W1mv_a_r = struct[0].i_W1mv_a_r
    i_W1mv_a_i = struct[0].i_W1mv_a_i
    i_W1mv_b_r = struct[0].i_W1mv_b_r
    i_W1mv_b_i = struct[0].i_W1mv_b_i
    i_W1mv_c_r = struct[0].i_W1mv_c_r
    i_W1mv_c_i = struct[0].i_W1mv_c_i
    i_W2mv_a_r = struct[0].i_W2mv_a_r
    i_W2mv_a_i = struct[0].i_W2mv_a_i
    i_W2mv_b_r = struct[0].i_W2mv_b_r
    i_W2mv_b_i = struct[0].i_W2mv_b_i
    i_W2mv_c_r = struct[0].i_W2mv_c_r
    i_W2mv_c_i = struct[0].i_W2mv_c_i
    i_W3mv_a_r = struct[0].i_W3mv_a_r
    i_W3mv_a_i = struct[0].i_W3mv_a_i
    i_W3mv_b_r = struct[0].i_W3mv_b_r
    i_W3mv_b_i = struct[0].i_W3mv_b_i
    i_W3mv_c_r = struct[0].i_W3mv_c_r
    i_W3mv_c_i = struct[0].i_W3mv_c_i
    i_STmv_a_r = struct[0].i_STmv_a_r
    i_STmv_a_i = struct[0].i_STmv_a_i
    i_STmv_b_r = struct[0].i_STmv_b_r
    i_STmv_b_i = struct[0].i_STmv_b_i
    i_STmv_c_r = struct[0].i_STmv_c_r
    i_STmv_c_i = struct[0].i_STmv_c_i
    p_ref_W1lv = struct[0].p_ref_W1lv
    T_pq_W1lv = struct[0].T_pq_W1lv
    v_loc_ref_W1lv = struct[0].v_loc_ref_W1lv
    Dv_r_W1lv = struct[0].Dv_r_W1lv
    Dq_r_W1lv = struct[0].Dq_r_W1lv
    p_ref_W2lv = struct[0].p_ref_W2lv
    T_pq_W2lv = struct[0].T_pq_W2lv
    v_loc_ref_W2lv = struct[0].v_loc_ref_W2lv
    Dv_r_W2lv = struct[0].Dv_r_W2lv
    Dq_r_W2lv = struct[0].Dq_r_W2lv
    p_ref_W3lv = struct[0].p_ref_W3lv
    T_pq_W3lv = struct[0].T_pq_W3lv
    v_loc_ref_W3lv = struct[0].v_loc_ref_W3lv
    Dv_r_W3lv = struct[0].Dv_r_W3lv
    Dq_r_W3lv = struct[0].Dq_r_W3lv
    p_ref_STlv = struct[0].p_ref_STlv
    T_pq_STlv = struct[0].T_pq_STlv
    v_loc_ref_STlv = struct[0].v_loc_ref_STlv
    Dv_r_STlv = struct[0].Dv_r_STlv
    Dq_r_STlv = struct[0].Dq_r_STlv
    
    # Dynamical states:
    p_W1lv_a = struct[0].x[0,0]
    p_W1lv_b = struct[0].x[1,0]
    p_W1lv_c = struct[0].x[2,0]
    q_W1lv_a = struct[0].x[3,0]
    q_W1lv_b = struct[0].x[4,0]
    q_W1lv_c = struct[0].x[5,0]
    p_W2lv_a = struct[0].x[6,0]
    p_W2lv_b = struct[0].x[7,0]
    p_W2lv_c = struct[0].x[8,0]
    q_W2lv_a = struct[0].x[9,0]
    q_W2lv_b = struct[0].x[10,0]
    q_W2lv_c = struct[0].x[11,0]
    p_W3lv_a = struct[0].x[12,0]
    p_W3lv_b = struct[0].x[13,0]
    p_W3lv_c = struct[0].x[14,0]
    q_W3lv_a = struct[0].x[15,0]
    q_W3lv_b = struct[0].x[16,0]
    q_W3lv_c = struct[0].x[17,0]
    p_STlv_a = struct[0].x[18,0]
    p_STlv_b = struct[0].x[19,0]
    p_STlv_c = struct[0].x[20,0]
    q_STlv_a = struct[0].x[21,0]
    q_STlv_b = struct[0].x[22,0]
    q_STlv_c = struct[0].x[23,0]
    
    # Algebraic states:
    v_W1lv_a_r = struct[0].y_run[0,0]
    v_W1lv_a_i = struct[0].y_run[1,0]
    v_W1lv_b_r = struct[0].y_run[2,0]
    v_W1lv_b_i = struct[0].y_run[3,0]
    v_W1lv_c_r = struct[0].y_run[4,0]
    v_W1lv_c_i = struct[0].y_run[5,0]
    v_W2lv_a_r = struct[0].y_run[6,0]
    v_W2lv_a_i = struct[0].y_run[7,0]
    v_W2lv_b_r = struct[0].y_run[8,0]
    v_W2lv_b_i = struct[0].y_run[9,0]
    v_W2lv_c_r = struct[0].y_run[10,0]
    v_W2lv_c_i = struct[0].y_run[11,0]
    v_W3lv_a_r = struct[0].y_run[12,0]
    v_W3lv_a_i = struct[0].y_run[13,0]
    v_W3lv_b_r = struct[0].y_run[14,0]
    v_W3lv_b_i = struct[0].y_run[15,0]
    v_W3lv_c_r = struct[0].y_run[16,0]
    v_W3lv_c_i = struct[0].y_run[17,0]
    v_STlv_a_r = struct[0].y_run[18,0]
    v_STlv_a_i = struct[0].y_run[19,0]
    v_STlv_b_r = struct[0].y_run[20,0]
    v_STlv_b_i = struct[0].y_run[21,0]
    v_STlv_c_r = struct[0].y_run[22,0]
    v_STlv_c_i = struct[0].y_run[23,0]
    v_POI_a_r = struct[0].y_run[24,0]
    v_POI_a_i = struct[0].y_run[25,0]
    v_POI_b_r = struct[0].y_run[26,0]
    v_POI_b_i = struct[0].y_run[27,0]
    v_POI_c_r = struct[0].y_run[28,0]
    v_POI_c_i = struct[0].y_run[29,0]
    v_POImv_a_r = struct[0].y_run[30,0]
    v_POImv_a_i = struct[0].y_run[31,0]
    v_POImv_b_r = struct[0].y_run[32,0]
    v_POImv_b_i = struct[0].y_run[33,0]
    v_POImv_c_r = struct[0].y_run[34,0]
    v_POImv_c_i = struct[0].y_run[35,0]
    v_W1mv_a_r = struct[0].y_run[36,0]
    v_W1mv_a_i = struct[0].y_run[37,0]
    v_W1mv_b_r = struct[0].y_run[38,0]
    v_W1mv_b_i = struct[0].y_run[39,0]
    v_W1mv_c_r = struct[0].y_run[40,0]
    v_W1mv_c_i = struct[0].y_run[41,0]
    v_W2mv_a_r = struct[0].y_run[42,0]
    v_W2mv_a_i = struct[0].y_run[43,0]
    v_W2mv_b_r = struct[0].y_run[44,0]
    v_W2mv_b_i = struct[0].y_run[45,0]
    v_W2mv_c_r = struct[0].y_run[46,0]
    v_W2mv_c_i = struct[0].y_run[47,0]
    v_W3mv_a_r = struct[0].y_run[48,0]
    v_W3mv_a_i = struct[0].y_run[49,0]
    v_W3mv_b_r = struct[0].y_run[50,0]
    v_W3mv_b_i = struct[0].y_run[51,0]
    v_W3mv_c_r = struct[0].y_run[52,0]
    v_W3mv_c_i = struct[0].y_run[53,0]
    v_STmv_a_r = struct[0].y_run[54,0]
    v_STmv_a_i = struct[0].y_run[55,0]
    v_STmv_b_r = struct[0].y_run[56,0]
    v_STmv_b_i = struct[0].y_run[57,0]
    v_STmv_c_r = struct[0].y_run[58,0]
    v_STmv_c_i = struct[0].y_run[59,0]
    i_l_W1mv_W2mv_a_r = struct[0].y_run[60,0]
    i_l_W1mv_W2mv_a_i = struct[0].y_run[61,0]
    i_l_W1mv_W2mv_b_r = struct[0].y_run[62,0]
    i_l_W1mv_W2mv_b_i = struct[0].y_run[63,0]
    i_l_W1mv_W2mv_c_r = struct[0].y_run[64,0]
    i_l_W1mv_W2mv_c_i = struct[0].y_run[65,0]
    i_l_W2mv_W3mv_a_r = struct[0].y_run[66,0]
    i_l_W2mv_W3mv_a_i = struct[0].y_run[67,0]
    i_l_W2mv_W3mv_b_r = struct[0].y_run[68,0]
    i_l_W2mv_W3mv_b_i = struct[0].y_run[69,0]
    i_l_W2mv_W3mv_c_r = struct[0].y_run[70,0]
    i_l_W2mv_W3mv_c_i = struct[0].y_run[71,0]
    i_l_W3mv_POImv_a_r = struct[0].y_run[72,0]
    i_l_W3mv_POImv_a_i = struct[0].y_run[73,0]
    i_l_W3mv_POImv_b_r = struct[0].y_run[74,0]
    i_l_W3mv_POImv_b_i = struct[0].y_run[75,0]
    i_l_W3mv_POImv_c_r = struct[0].y_run[76,0]
    i_l_W3mv_POImv_c_i = struct[0].y_run[77,0]
    i_l_STmv_POImv_a_r = struct[0].y_run[78,0]
    i_l_STmv_POImv_a_i = struct[0].y_run[79,0]
    i_l_STmv_POImv_b_r = struct[0].y_run[80,0]
    i_l_STmv_POImv_b_i = struct[0].y_run[81,0]
    i_l_STmv_POImv_c_r = struct[0].y_run[82,0]
    i_l_STmv_POImv_c_i = struct[0].y_run[83,0]
    i_l_POI_GRID_a_r = struct[0].y_run[84,0]
    i_l_POI_GRID_a_i = struct[0].y_run[85,0]
    i_l_POI_GRID_b_r = struct[0].y_run[86,0]
    i_l_POI_GRID_b_i = struct[0].y_run[87,0]
    i_l_POI_GRID_c_r = struct[0].y_run[88,0]
    i_l_POI_GRID_c_i = struct[0].y_run[89,0]
    i_W1lv_a_r = struct[0].y_run[90,0]
    i_W1lv_a_i = struct[0].y_run[91,0]
    i_W1lv_b_r = struct[0].y_run[92,0]
    i_W1lv_b_i = struct[0].y_run[93,0]
    i_W1lv_c_r = struct[0].y_run[94,0]
    i_W1lv_c_i = struct[0].y_run[95,0]
    v_m_W1lv = struct[0].y_run[96,0]
    v_m_W1mv = struct[0].y_run[97,0]
    i_reac_ref_W1lv = struct[0].y_run[98,0]
    q_ref_W1lv = struct[0].y_run[99,0]
    i_W2lv_a_r = struct[0].y_run[100,0]
    i_W2lv_a_i = struct[0].y_run[101,0]
    i_W2lv_b_r = struct[0].y_run[102,0]
    i_W2lv_b_i = struct[0].y_run[103,0]
    i_W2lv_c_r = struct[0].y_run[104,0]
    i_W2lv_c_i = struct[0].y_run[105,0]
    v_m_W2lv = struct[0].y_run[106,0]
    v_m_W2mv = struct[0].y_run[107,0]
    i_reac_ref_W2lv = struct[0].y_run[108,0]
    q_ref_W2lv = struct[0].y_run[109,0]
    i_W3lv_a_r = struct[0].y_run[110,0]
    i_W3lv_a_i = struct[0].y_run[111,0]
    i_W3lv_b_r = struct[0].y_run[112,0]
    i_W3lv_b_i = struct[0].y_run[113,0]
    i_W3lv_c_r = struct[0].y_run[114,0]
    i_W3lv_c_i = struct[0].y_run[115,0]
    v_m_W3lv = struct[0].y_run[116,0]
    v_m_W3mv = struct[0].y_run[117,0]
    i_reac_ref_W3lv = struct[0].y_run[118,0]
    q_ref_W3lv = struct[0].y_run[119,0]
    i_STlv_a_r = struct[0].y_run[120,0]
    i_STlv_a_i = struct[0].y_run[121,0]
    i_STlv_b_r = struct[0].y_run[122,0]
    i_STlv_b_i = struct[0].y_run[123,0]
    i_STlv_c_r = struct[0].y_run[124,0]
    i_STlv_c_i = struct[0].y_run[125,0]
    v_m_STlv = struct[0].y_run[126,0]
    v_m_STmv = struct[0].y_run[127,0]
    i_reac_ref_STlv = struct[0].y_run[128,0]
    q_ref_STlv = struct[0].y_run[129,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-p_W1lv_a + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[1,0] = (-p_W1lv_b + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[2,0] = (-p_W1lv_c + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[3,0] = (-q_W1lv_a + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[4,0] = (-q_W1lv_b + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[5,0] = (-q_W1lv_c + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[6,0] = (-p_W2lv_a + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[7,0] = (-p_W2lv_b + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[8,0] = (-p_W2lv_c + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[9,0] = (-q_W2lv_a + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[10,0] = (-q_W2lv_b + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[11,0] = (-q_W2lv_c + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[12,0] = (-p_W3lv_a + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[13,0] = (-p_W3lv_b + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[14,0] = (-p_W3lv_c + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[15,0] = (-q_W3lv_a + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[16,0] = (-q_W3lv_b + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[17,0] = (-q_W3lv_c + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[18,0] = (-p_STlv_a + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[19,0] = (-p_STlv_b + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[20,0] = (-p_STlv_c + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[21,0] = (-q_STlv_a + q_ref_STlv/3)/T_pq_STlv
        struct[0].f[22,0] = (-q_STlv_b + q_ref_STlv/3)/T_pq_STlv
        struct[0].f[23,0] = (-q_STlv_c + q_ref_STlv/3)/T_pq_STlv
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_W1lv_a_r - 85.1513138847732*v_W1lv_a_i - 14.1918856474622*v_W1lv_a_r + 1.69609362276623*v_W1mv_a_i + 0.282682270461039*v_W1mv_a_r - 1.69609362276623*v_W1mv_c_i - 0.282682270461039*v_W1mv_c_r
        struct[0].g[1,0] = i_W1lv_a_i - 14.1918856474622*v_W1lv_a_i + 85.1513138847732*v_W1lv_a_r + 0.282682270461039*v_W1mv_a_i - 1.69609362276623*v_W1mv_a_r - 0.282682270461039*v_W1mv_c_i + 1.69609362276623*v_W1mv_c_r
        struct[0].g[2,0] = i_W1lv_b_r - 85.1513138847732*v_W1lv_b_i - 14.1918856474622*v_W1lv_b_r - 1.69609362276623*v_W1mv_a_i - 0.282682270461039*v_W1mv_a_r + 1.69609362276623*v_W1mv_b_i + 0.282682270461039*v_W1mv_b_r
        struct[0].g[3,0] = i_W1lv_b_i - 14.1918856474622*v_W1lv_b_i + 85.1513138847732*v_W1lv_b_r - 0.282682270461039*v_W1mv_a_i + 1.69609362276623*v_W1mv_a_r + 0.282682270461039*v_W1mv_b_i - 1.69609362276623*v_W1mv_b_r
        struct[0].g[4,0] = i_W1lv_c_r - 85.1513138847732*v_W1lv_c_i - 14.1918856474622*v_W1lv_c_r - 1.69609362276623*v_W1mv_b_i - 0.282682270461039*v_W1mv_b_r + 1.69609362276623*v_W1mv_c_i + 0.282682270461039*v_W1mv_c_r
        struct[0].g[5,0] = i_W1lv_c_i - 14.1918856474622*v_W1lv_c_i + 85.1513138847732*v_W1lv_c_r - 0.282682270461039*v_W1mv_b_i + 1.69609362276623*v_W1mv_b_r + 0.282682270461039*v_W1mv_c_i - 1.69609362276623*v_W1mv_c_r
        struct[0].g[6,0] = i_W2lv_a_r - 85.1513138847732*v_W2lv_a_i - 14.1918856474622*v_W2lv_a_r + 1.69609362276623*v_W2mv_a_i + 0.282682270461039*v_W2mv_a_r - 1.69609362276623*v_W2mv_c_i - 0.282682270461039*v_W2mv_c_r
        struct[0].g[7,0] = i_W2lv_a_i - 14.1918856474622*v_W2lv_a_i + 85.1513138847732*v_W2lv_a_r + 0.282682270461039*v_W2mv_a_i - 1.69609362276623*v_W2mv_a_r - 0.282682270461039*v_W2mv_c_i + 1.69609362276623*v_W2mv_c_r
        struct[0].g[8,0] = i_W2lv_b_r - 85.1513138847732*v_W2lv_b_i - 14.1918856474622*v_W2lv_b_r - 1.69609362276623*v_W2mv_a_i - 0.282682270461039*v_W2mv_a_r + 1.69609362276623*v_W2mv_b_i + 0.282682270461039*v_W2mv_b_r
        struct[0].g[9,0] = i_W2lv_b_i - 14.1918856474622*v_W2lv_b_i + 85.1513138847732*v_W2lv_b_r - 0.282682270461039*v_W2mv_a_i + 1.69609362276623*v_W2mv_a_r + 0.282682270461039*v_W2mv_b_i - 1.69609362276623*v_W2mv_b_r
        struct[0].g[10,0] = i_W2lv_c_r - 85.1513138847732*v_W2lv_c_i - 14.1918856474622*v_W2lv_c_r - 1.69609362276623*v_W2mv_b_i - 0.282682270461039*v_W2mv_b_r + 1.69609362276623*v_W2mv_c_i + 0.282682270461039*v_W2mv_c_r
        struct[0].g[11,0] = i_W2lv_c_i - 14.1918856474622*v_W2lv_c_i + 85.1513138847732*v_W2lv_c_r - 0.282682270461039*v_W2mv_b_i + 1.69609362276623*v_W2mv_b_r + 0.282682270461039*v_W2mv_c_i - 1.69609362276623*v_W2mv_c_r
        struct[0].g[12,0] = i_W3lv_a_r - 85.1513138847732*v_W3lv_a_i - 14.1918856474622*v_W3lv_a_r + 1.69609362276623*v_W3mv_a_i + 0.282682270461039*v_W3mv_a_r - 1.69609362276623*v_W3mv_c_i - 0.282682270461039*v_W3mv_c_r
        struct[0].g[13,0] = i_W3lv_a_i - 14.1918856474622*v_W3lv_a_i + 85.1513138847732*v_W3lv_a_r + 0.282682270461039*v_W3mv_a_i - 1.69609362276623*v_W3mv_a_r - 0.282682270461039*v_W3mv_c_i + 1.69609362276623*v_W3mv_c_r
        struct[0].g[14,0] = i_W3lv_b_r - 85.1513138847732*v_W3lv_b_i - 14.1918856474622*v_W3lv_b_r - 1.69609362276623*v_W3mv_a_i - 0.282682270461039*v_W3mv_a_r + 1.69609362276623*v_W3mv_b_i + 0.282682270461039*v_W3mv_b_r
        struct[0].g[15,0] = i_W3lv_b_i - 14.1918856474622*v_W3lv_b_i + 85.1513138847732*v_W3lv_b_r - 0.282682270461039*v_W3mv_a_i + 1.69609362276623*v_W3mv_a_r + 0.282682270461039*v_W3mv_b_i - 1.69609362276623*v_W3mv_b_r
        struct[0].g[16,0] = i_W3lv_c_r - 85.1513138847732*v_W3lv_c_i - 14.1918856474622*v_W3lv_c_r - 1.69609362276623*v_W3mv_b_i - 0.282682270461039*v_W3mv_b_r + 1.69609362276623*v_W3mv_c_i + 0.282682270461039*v_W3mv_c_r
        struct[0].g[17,0] = i_W3lv_c_i - 14.1918856474622*v_W3lv_c_i + 85.1513138847732*v_W3lv_c_r - 0.282682270461039*v_W3mv_b_i + 1.69609362276623*v_W3mv_b_r + 0.282682270461039*v_W3mv_c_i - 1.69609362276623*v_W3mv_c_r
        struct[0].g[18,0] = i_STlv_a_r - 85.1513138847732*v_STlv_a_i - 14.1918856474622*v_STlv_a_r + 1.69609362276623*v_STmv_a_i + 0.282682270461039*v_STmv_a_r - 1.69609362276623*v_STmv_c_i - 0.282682270461039*v_STmv_c_r
        struct[0].g[19,0] = i_STlv_a_i - 14.1918856474622*v_STlv_a_i + 85.1513138847732*v_STlv_a_r + 0.282682270461039*v_STmv_a_i - 1.69609362276623*v_STmv_a_r - 0.282682270461039*v_STmv_c_i + 1.69609362276623*v_STmv_c_r
        struct[0].g[20,0] = i_STlv_b_r - 85.1513138847732*v_STlv_b_i - 14.1918856474622*v_STlv_b_r - 1.69609362276623*v_STmv_a_i - 0.282682270461039*v_STmv_a_r + 1.69609362276623*v_STmv_b_i + 0.282682270461039*v_STmv_b_r
        struct[0].g[21,0] = i_STlv_b_i - 14.1918856474622*v_STlv_b_i + 85.1513138847732*v_STlv_b_r - 0.282682270461039*v_STmv_a_i + 1.69609362276623*v_STmv_a_r + 0.282682270461039*v_STmv_b_i - 1.69609362276623*v_STmv_b_r
        struct[0].g[22,0] = i_STlv_c_r - 85.1513138847732*v_STlv_c_i - 14.1918856474622*v_STlv_c_r - 1.69609362276623*v_STmv_b_i - 0.282682270461039*v_STmv_b_r + 1.69609362276623*v_STmv_c_i + 0.282682270461039*v_STmv_c_r
        struct[0].g[23,0] = i_STlv_c_i - 14.1918856474622*v_STlv_c_i + 85.1513138847732*v_STlv_c_r - 0.282682270461039*v_STmv_b_i + 1.69609362276623*v_STmv_b_r + 0.282682270461039*v_STmv_c_i - 1.69609362276623*v_STmv_c_r
        struct[0].g[24,0] = i_POI_a_r + 0.040290088638195*v_GRID_a_i + 0.024174053182917*v_GRID_a_r + 4.66248501556824e-18*v_GRID_b_i - 4.31760362252812e-18*v_GRID_b_r + 4.19816664496737e-18*v_GRID_c_i - 3.49608108880335e-18*v_GRID_c_r - 0.0591264711109411*v_POI_a_i - 0.0265286009920103*v_POI_a_r + 0.00941819123637305*v_POI_b_i + 0.00117727390454664*v_POI_b_r + 0.00941819123637305*v_POI_c_i + 0.00117727390454664*v_POI_c_r + 0.0538321929314336*v_POImv_a_i + 0.0067290241164292*v_POImv_a_r - 0.0538321929314336*v_POImv_b_i - 0.0067290241164292*v_POImv_b_r
        struct[0].g[25,0] = i_POI_a_i + 0.024174053182917*v_GRID_a_i - 0.040290088638195*v_GRID_a_r - 4.31760362252812e-18*v_GRID_b_i - 4.66248501556824e-18*v_GRID_b_r - 3.49608108880335e-18*v_GRID_c_i - 4.19816664496737e-18*v_GRID_c_r - 0.0265286009920103*v_POI_a_i + 0.0591264711109411*v_POI_a_r + 0.00117727390454664*v_POI_b_i - 0.00941819123637305*v_POI_b_r + 0.00117727390454664*v_POI_c_i - 0.00941819123637305*v_POI_c_r + 0.0067290241164292*v_POImv_a_i - 0.0538321929314336*v_POImv_a_r - 0.0067290241164292*v_POImv_b_i + 0.0538321929314336*v_POImv_b_r
        struct[0].g[26,0] = i_POI_b_r + 6.30775359573304e-19*v_GRID_a_i - 2.07254761002657e-18*v_GRID_a_r + 0.040290088638195*v_GRID_b_i + 0.024174053182917*v_GRID_b_r + 9.01107656533306e-19*v_GRID_c_i - 1.78419315993592e-17*v_GRID_c_r + 0.00941819123637305*v_POI_a_i + 0.00117727390454663*v_POI_a_r - 0.0591264711109411*v_POI_b_i - 0.0265286009920103*v_POI_b_r + 0.00941819123637305*v_POI_c_i + 0.00117727390454665*v_POI_c_r + 0.0538321929314336*v_POImv_b_i + 0.0067290241164292*v_POImv_b_r - 0.0538321929314336*v_POImv_c_i - 0.0067290241164292*v_POImv_c_r
        struct[0].g[27,0] = i_POI_b_i - 2.07254761002657e-18*v_GRID_a_i - 6.30775359573304e-19*v_GRID_a_r + 0.024174053182917*v_GRID_b_i - 0.040290088638195*v_GRID_b_r - 1.78419315993592e-17*v_GRID_c_i - 9.01107656533306e-19*v_GRID_c_r + 0.00117727390454663*v_POI_a_i - 0.00941819123637305*v_POI_a_r - 0.0265286009920103*v_POI_b_i + 0.0591264711109411*v_POI_b_r + 0.00117727390454665*v_POI_c_i - 0.00941819123637305*v_POI_c_r + 0.0067290241164292*v_POImv_b_i - 0.0538321929314336*v_POImv_b_r - 0.0067290241164292*v_POImv_c_i + 0.0538321929314336*v_POImv_c_r
        struct[0].g[28,0] = i_POI_c_r - 7.20886125226632e-19*v_GRID_a_i - 1.35166148479994e-18*v_GRID_a_r - 4.50553828266631e-19*v_GRID_b_i - 1.71210454741325e-17*v_GRID_b_r + 0.040290088638195*v_GRID_c_i + 0.024174053182917*v_GRID_c_r + 0.00941819123637305*v_POI_a_i + 0.00117727390454663*v_POI_a_r + 0.00941819123637305*v_POI_b_i + 0.00117727390454665*v_POI_b_r - 0.0591264711109411*v_POI_c_i - 0.0265286009920103*v_POI_c_r - 0.0538321929314336*v_POImv_a_i - 0.0067290241164292*v_POImv_a_r + 0.0538321929314336*v_POImv_c_i + 0.0067290241164292*v_POImv_c_r
        struct[0].g[29,0] = i_POI_c_i - 1.35166148479994e-18*v_GRID_a_i + 7.20886125226632e-19*v_GRID_a_r - 1.71210454741325e-17*v_GRID_b_i + 4.50553828266631e-19*v_GRID_b_r + 0.024174053182917*v_GRID_c_i - 0.040290088638195*v_GRID_c_r + 0.00117727390454663*v_POI_a_i - 0.00941819123637305*v_POI_a_r + 0.00117727390454665*v_POI_b_i - 0.00941819123637305*v_POI_b_r - 0.0265286009920103*v_POI_c_i + 0.0591264711109411*v_POI_c_r - 0.0067290241164292*v_POImv_a_i + 0.0538321929314336*v_POImv_a_r + 0.0067290241164292*v_POImv_c_i - 0.0538321929314336*v_POImv_c_r
        struct[0].g[30,0] = i_POImv_a_r + 0.0538321929314336*v_POI_a_i + 0.0067290241164292*v_POI_a_r - 0.0538321929314336*v_POI_c_i - 0.0067290241164292*v_POI_c_r - 155.244588874881*v_POImv_a_i - 188.924390492986*v_POImv_a_r + 44.2677164725443*v_POImv_b_i + 53.9540151298641*v_POImv_b_r + 44.2677164725443*v_POImv_c_i + 53.9540151298641*v_POImv_c_r + 148.977829666654*v_STmv_a_i + 181.611068546017*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274334*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274334*v_STmv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[31,0] = i_POImv_a_i + 0.0067290241164292*v_POI_a_i - 0.0538321929314336*v_POI_a_r - 0.0067290241164292*v_POI_c_i + 0.0538321929314336*v_POI_c_r - 188.924390492986*v_POImv_a_i + 155.244588874881*v_POImv_a_r + 53.9540151298641*v_POImv_b_i - 44.2677164725443*v_POImv_b_r + 53.9540151298641*v_POImv_c_i - 44.2677164725443*v_POImv_c_r + 181.611068546017*v_STmv_a_i - 148.977829666654*v_STmv_a_r - 51.8888767274334*v_STmv_b_i + 42.5650941904727*v_STmv_b_r - 51.8888767274334*v_STmv_c_i + 42.5650941904727*v_STmv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[32,0] = i_POImv_b_r - 0.0538321929314336*v_POI_a_i - 0.0067290241164292*v_POI_a_r + 0.0538321929314336*v_POI_b_i + 0.0067290241164292*v_POI_b_r + 44.2677164725443*v_POImv_a_i + 53.9540151298641*v_POImv_a_r - 155.244588874881*v_POImv_b_i - 188.924390492986*v_POImv_b_r + 44.2677164725443*v_POImv_c_i + 53.9540151298642*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r + 148.977829666654*v_STmv_b_i + 181.611068546017*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274335*v_STmv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[33,0] = i_POImv_b_i - 0.0067290241164292*v_POI_a_i + 0.0538321929314336*v_POI_a_r + 0.0067290241164292*v_POI_b_i - 0.0538321929314336*v_POI_b_r + 53.9540151298641*v_POImv_a_i - 44.2677164725443*v_POImv_a_r - 188.924390492986*v_POImv_b_i + 155.244588874881*v_POImv_b_r + 53.9540151298642*v_POImv_c_i - 44.2677164725443*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r + 181.611068546017*v_STmv_b_i - 148.977829666654*v_STmv_b_r - 51.8888767274335*v_STmv_c_i + 42.5650941904727*v_STmv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[34,0] = i_POImv_c_r - 0.0538321929314336*v_POI_b_i - 0.0067290241164292*v_POI_b_r + 0.0538321929314336*v_POI_c_i + 0.0067290241164292*v_POI_c_r + 44.2677164725443*v_POImv_a_i + 53.9540151298641*v_POImv_a_r + 44.2677164725443*v_POImv_b_i + 53.9540151298642*v_POImv_b_r - 155.244588874881*v_POImv_c_i - 188.924390492986*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274335*v_STmv_b_r + 148.977829666654*v_STmv_c_i + 181.611068546017*v_STmv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[35,0] = i_POImv_c_i - 0.0067290241164292*v_POI_b_i + 0.0538321929314336*v_POI_b_r + 0.0067290241164292*v_POI_c_i - 0.0538321929314336*v_POI_c_r + 53.9540151298641*v_POImv_a_i - 44.2677164725443*v_POImv_a_r + 53.9540151298642*v_POImv_b_i - 44.2677164725443*v_POImv_b_r - 188.924390492986*v_POImv_c_i + 155.244588874881*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r - 51.8888767274335*v_STmv_b_i + 42.5650941904727*v_STmv_b_r + 181.611068546017*v_STmv_c_i - 148.977829666654*v_STmv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[36,0] = i_W1mv_a_r + 1.69609362276623*v_W1lv_a_i + 0.282682270461039*v_W1lv_a_r - 1.69609362276623*v_W1lv_b_i - 0.282682270461039*v_W1lv_b_r - 6.02663624833782*v_W1mv_a_i - 7.27570400310194*v_W1mv_a_r + 1.73640535376106*v_W1mv_b_i + 2.08118569972797*v_W1mv_b_r + 1.73640535376106*v_W1mv_c_i + 2.08118569972797*v_W1mv_c_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r
        struct[0].g[37,0] = i_W1mv_a_i + 0.282682270461039*v_W1lv_a_i - 1.69609362276623*v_W1lv_a_r - 0.282682270461039*v_W1lv_b_i + 1.69609362276623*v_W1lv_b_r - 7.27570400310194*v_W1mv_a_i + 6.02663624833782*v_W1mv_a_r + 2.08118569972797*v_W1mv_b_i - 1.73640535376106*v_W1mv_b_r + 2.08118569972797*v_W1mv_c_i - 1.73640535376106*v_W1mv_c_r + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r
        struct[0].g[38,0] = i_W1mv_b_r + 1.69609362276623*v_W1lv_b_i + 0.282682270461039*v_W1lv_b_r - 1.69609362276623*v_W1lv_c_i - 0.282682270461039*v_W1lv_c_r + 1.73640535376106*v_W1mv_a_i + 2.08118569972797*v_W1mv_a_r - 6.02663624833782*v_W1mv_b_i - 7.27570400310194*v_W1mv_b_r + 1.73640535376106*v_W1mv_c_i + 2.08118569972797*v_W1mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r
        struct[0].g[39,0] = i_W1mv_b_i + 0.282682270461039*v_W1lv_b_i - 1.69609362276623*v_W1lv_b_r - 0.282682270461039*v_W1lv_c_i + 1.69609362276623*v_W1lv_c_r + 2.08118569972797*v_W1mv_a_i - 1.73640535376106*v_W1mv_a_r - 7.27570400310194*v_W1mv_b_i + 6.02663624833782*v_W1mv_b_r + 2.08118569972797*v_W1mv_c_i - 1.73640535376106*v_W1mv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r
        struct[0].g[40,0] = i_W1mv_c_r - 1.69609362276623*v_W1lv_a_i - 0.282682270461039*v_W1lv_a_r + 1.69609362276623*v_W1lv_c_i + 0.282682270461039*v_W1lv_c_r + 1.73640535376106*v_W1mv_a_i + 2.08118569972797*v_W1mv_a_r + 1.73640535376106*v_W1mv_b_i + 2.08118569972797*v_W1mv_b_r - 6.02663624833782*v_W1mv_c_i - 7.27570400310194*v_W1mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r
        struct[0].g[41,0] = i_W1mv_c_i - 0.282682270461039*v_W1lv_a_i + 1.69609362276623*v_W1lv_a_r + 0.282682270461039*v_W1lv_c_i - 1.69609362276623*v_W1lv_c_r + 2.08118569972797*v_W1mv_a_i - 1.73640535376106*v_W1mv_a_r + 2.08118569972797*v_W1mv_b_i - 1.73640535376106*v_W1mv_b_r - 7.27570400310194*v_W1mv_c_i + 6.02663624833782*v_W1mv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r
        struct[0].g[42,0] = i_W2mv_a_r + 5.95911318666618*v_W1mv_a_i + 7.26444274184068*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.69609362276623*v_W2lv_a_i + 0.282682270461039*v_W2lv_a_r - 1.69609362276623*v_W2lv_b_i - 0.282682270461039*v_W2lv_b_r - 11.9857049291081*v_W2mv_a_i - 14.5401467449426*v_W2mv_a_r + 3.43902692373834*v_W2mv_b_i + 4.1567407688253*v_W2mv_b_r + 3.43902692373834*v_W2mv_c_i + 4.1567407688253*v_W2mv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[43,0] = i_W2mv_a_i + 7.26444274184068*v_W1mv_a_i - 5.95911318666618*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 0.282682270461039*v_W2lv_a_i - 1.69609362276623*v_W2lv_a_r - 0.282682270461039*v_W2lv_b_i + 1.69609362276623*v_W2lv_b_r - 14.5401467449426*v_W2mv_a_i + 11.9857049291081*v_W2mv_a_r + 4.1567407688253*v_W2mv_b_i - 3.43902692373834*v_W2mv_b_r + 4.1567407688253*v_W2mv_c_i - 3.43902692373834*v_W2mv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[44,0] = i_W2mv_b_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r + 5.95911318666618*v_W1mv_b_i + 7.26444274184068*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.69609362276623*v_W2lv_b_i + 0.282682270461039*v_W2lv_b_r - 1.69609362276623*v_W2lv_c_i - 0.282682270461039*v_W2lv_c_r + 3.43902692373834*v_W2mv_a_i + 4.1567407688253*v_W2mv_a_r - 11.9857049291081*v_W2mv_b_i - 14.5401467449426*v_W2mv_b_r + 3.43902692373834*v_W2mv_c_i + 4.15674076882531*v_W2mv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[45,0] = i_W2mv_b_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r + 7.26444274184068*v_W1mv_b_i - 5.95911318666618*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 0.282682270461039*v_W2lv_b_i - 1.69609362276623*v_W2lv_b_r - 0.282682270461039*v_W2lv_c_i + 1.69609362276623*v_W2lv_c_r + 4.1567407688253*v_W2mv_a_i - 3.43902692373834*v_W2mv_a_r - 14.5401467449426*v_W2mv_b_i + 11.9857049291081*v_W2mv_b_r + 4.15674076882531*v_W2mv_c_i - 3.43902692373834*v_W2mv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[46,0] = i_W2mv_c_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r + 5.95911318666618*v_W1mv_c_i + 7.26444274184068*v_W1mv_c_r - 1.69609362276623*v_W2lv_a_i - 0.282682270461039*v_W2lv_a_r + 1.69609362276623*v_W2lv_c_i + 0.282682270461039*v_W2lv_c_r + 3.43902692373834*v_W2mv_a_i + 4.1567407688253*v_W2mv_a_r + 3.43902692373834*v_W2mv_b_i + 4.15674076882531*v_W2mv_b_r - 11.9857049291081*v_W2mv_c_i - 14.5401467449426*v_W2mv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[47,0] = i_W2mv_c_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r + 7.26444274184068*v_W1mv_c_i - 5.95911318666618*v_W1mv_c_r - 0.282682270461039*v_W2lv_a_i + 1.69609362276623*v_W2lv_a_r + 0.282682270461039*v_W2lv_c_i - 1.69609362276623*v_W2lv_c_r + 4.1567407688253*v_W2mv_a_i - 3.43902692373834*v_W2mv_a_r + 4.15674076882531*v_W2mv_b_i - 3.43902692373834*v_W2mv_b_r - 14.5401467449426*v_W2mv_c_i + 11.9857049291081*v_W2mv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[48,0] = i_W3mv_a_r + 5.95911318666618*v_POImv_a_i + 7.26444274184068*v_POImv_a_r - 1.70260376761891*v_POImv_b_i - 2.07555506909734*v_POImv_b_r - 1.70260376761891*v_POImv_c_i - 2.07555506909734*v_POImv_c_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.69609362276623*v_W3lv_a_i + 0.282682270461039*v_W3lv_a_r - 1.69609362276623*v_W3lv_b_i - 0.282682270461039*v_W3lv_b_r - 11.9857049291081*v_W3mv_a_i - 14.5401467449426*v_W3mv_a_r + 3.43902692373834*v_W3mv_b_i + 4.1567407688253*v_W3mv_b_r + 3.43902692373834*v_W3mv_c_i + 4.1567407688253*v_W3mv_c_r
        struct[0].g[49,0] = i_W3mv_a_i + 7.26444274184068*v_POImv_a_i - 5.95911318666618*v_POImv_a_r - 2.07555506909734*v_POImv_b_i + 1.70260376761891*v_POImv_b_r - 2.07555506909734*v_POImv_c_i + 1.70260376761891*v_POImv_c_r + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 0.282682270461039*v_W3lv_a_i - 1.69609362276623*v_W3lv_a_r - 0.282682270461039*v_W3lv_b_i + 1.69609362276623*v_W3lv_b_r - 14.5401467449426*v_W3mv_a_i + 11.9857049291081*v_W3mv_a_r + 4.1567407688253*v_W3mv_b_i - 3.43902692373834*v_W3mv_b_r + 4.1567407688253*v_W3mv_c_i - 3.43902692373834*v_W3mv_c_r
        struct[0].g[50,0] = i_W3mv_b_r - 1.70260376761891*v_POImv_a_i - 2.07555506909734*v_POImv_a_r + 5.95911318666618*v_POImv_b_i + 7.26444274184068*v_POImv_b_r - 1.70260376761891*v_POImv_c_i - 2.07555506909734*v_POImv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.69609362276623*v_W3lv_b_i + 0.282682270461039*v_W3lv_b_r - 1.69609362276623*v_W3lv_c_i - 0.282682270461039*v_W3lv_c_r + 3.43902692373834*v_W3mv_a_i + 4.1567407688253*v_W3mv_a_r - 11.9857049291081*v_W3mv_b_i - 14.5401467449426*v_W3mv_b_r + 3.43902692373834*v_W3mv_c_i + 4.15674076882531*v_W3mv_c_r
        struct[0].g[51,0] = i_W3mv_b_i - 2.07555506909734*v_POImv_a_i + 1.70260376761891*v_POImv_a_r + 7.26444274184068*v_POImv_b_i - 5.95911318666618*v_POImv_b_r - 2.07555506909734*v_POImv_c_i + 1.70260376761891*v_POImv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 0.282682270461039*v_W3lv_b_i - 1.69609362276623*v_W3lv_b_r - 0.282682270461039*v_W3lv_c_i + 1.69609362276623*v_W3lv_c_r + 4.1567407688253*v_W3mv_a_i - 3.43902692373834*v_W3mv_a_r - 14.5401467449426*v_W3mv_b_i + 11.9857049291081*v_W3mv_b_r + 4.15674076882531*v_W3mv_c_i - 3.43902692373834*v_W3mv_c_r
        struct[0].g[52,0] = i_W3mv_c_r - 1.70260376761891*v_POImv_a_i - 2.07555506909734*v_POImv_a_r - 1.70260376761891*v_POImv_b_i - 2.07555506909734*v_POImv_b_r + 5.95911318666618*v_POImv_c_i + 7.26444274184068*v_POImv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r - 1.69609362276623*v_W3lv_a_i - 0.282682270461039*v_W3lv_a_r + 1.69609362276623*v_W3lv_c_i + 0.282682270461039*v_W3lv_c_r + 3.43902692373834*v_W3mv_a_i + 4.1567407688253*v_W3mv_a_r + 3.43902692373834*v_W3mv_b_i + 4.15674076882531*v_W3mv_b_r - 11.9857049291081*v_W3mv_c_i - 14.5401467449426*v_W3mv_c_r
        struct[0].g[53,0] = i_W3mv_c_i - 2.07555506909734*v_POImv_a_i + 1.70260376761891*v_POImv_a_r - 2.07555506909734*v_POImv_b_i + 1.70260376761891*v_POImv_b_r + 7.26444274184068*v_POImv_c_i - 5.95911318666618*v_POImv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r - 0.282682270461039*v_W3lv_a_i + 1.69609362276623*v_W3lv_a_r + 0.282682270461039*v_W3lv_c_i - 1.69609362276623*v_W3lv_c_r + 4.1567407688253*v_W3mv_a_i - 3.43902692373834*v_W3mv_a_r + 4.15674076882531*v_W3mv_b_i - 3.43902692373834*v_W3mv_b_r - 14.5401467449426*v_W3mv_c_i + 11.9857049291081*v_W3mv_c_r
        struct[0].g[54,0] = i_STmv_a_r + 148.977829666654*v_POImv_a_i + 181.611068546017*v_POImv_a_r - 42.5650941904727*v_POImv_b_i - 51.8888767274334*v_POImv_b_r - 42.5650941904727*v_POImv_c_i - 51.8888767274334*v_POImv_c_r + 1.69609362276623*v_STlv_a_i + 0.282682270461039*v_STlv_a_r - 1.69609362276623*v_STlv_b_i - 0.282682270461039*v_STlv_b_r - 149.045395453986*v_STmv_a_i - 181.622329807278*v_STmv_a_r + 42.5988786863508*v_STmv_b_i + 51.8945073580641*v_STmv_b_r + 42.5988786863508*v_STmv_c_i + 51.894507358064*v_STmv_c_r
        struct[0].g[55,0] = i_STmv_a_i + 181.611068546017*v_POImv_a_i - 148.977829666654*v_POImv_a_r - 51.8888767274334*v_POImv_b_i + 42.5650941904727*v_POImv_b_r - 51.8888767274334*v_POImv_c_i + 42.5650941904727*v_POImv_c_r + 0.282682270461039*v_STlv_a_i - 1.69609362276623*v_STlv_a_r - 0.282682270461039*v_STlv_b_i + 1.69609362276623*v_STlv_b_r - 181.622329807278*v_STmv_a_i + 149.045395453986*v_STmv_a_r + 51.8945073580641*v_STmv_b_i - 42.5988786863508*v_STmv_b_r + 51.894507358064*v_STmv_c_i - 42.5988786863508*v_STmv_c_r
        struct[0].g[56,0] = i_STmv_b_r - 42.5650941904727*v_POImv_a_i - 51.8888767274334*v_POImv_a_r + 148.977829666654*v_POImv_b_i + 181.611068546017*v_POImv_b_r - 42.5650941904727*v_POImv_c_i - 51.8888767274335*v_POImv_c_r + 1.69609362276623*v_STlv_b_i + 0.282682270461039*v_STlv_b_r - 1.69609362276623*v_STlv_c_i - 0.282682270461039*v_STlv_c_r + 42.5988786863508*v_STmv_a_i + 51.8945073580641*v_STmv_a_r - 149.045395453986*v_STmv_b_i - 181.622329807278*v_STmv_b_r + 42.5988786863508*v_STmv_c_i + 51.8945073580641*v_STmv_c_r
        struct[0].g[57,0] = i_STmv_b_i - 51.8888767274334*v_POImv_a_i + 42.5650941904727*v_POImv_a_r + 181.611068546017*v_POImv_b_i - 148.977829666654*v_POImv_b_r - 51.8888767274335*v_POImv_c_i + 42.5650941904727*v_POImv_c_r + 0.282682270461039*v_STlv_b_i - 1.69609362276623*v_STlv_b_r - 0.282682270461039*v_STlv_c_i + 1.69609362276623*v_STlv_c_r + 51.8945073580641*v_STmv_a_i - 42.5988786863508*v_STmv_a_r - 181.622329807278*v_STmv_b_i + 149.045395453986*v_STmv_b_r + 51.8945073580641*v_STmv_c_i - 42.5988786863508*v_STmv_c_r
        struct[0].g[58,0] = i_STmv_c_r - 42.5650941904727*v_POImv_a_i - 51.8888767274334*v_POImv_a_r - 42.5650941904727*v_POImv_b_i - 51.8888767274335*v_POImv_b_r + 148.977829666654*v_POImv_c_i + 181.611068546017*v_POImv_c_r - 1.69609362276623*v_STlv_a_i - 0.282682270461039*v_STlv_a_r + 1.69609362276623*v_STlv_c_i + 0.282682270461039*v_STlv_c_r + 42.5988786863508*v_STmv_a_i + 51.8945073580641*v_STmv_a_r + 42.5988786863508*v_STmv_b_i + 51.8945073580641*v_STmv_b_r - 149.045395453986*v_STmv_c_i - 181.622329807278*v_STmv_c_r
        struct[0].g[59,0] = i_STmv_c_i - 51.8888767274334*v_POImv_a_i + 42.5650941904727*v_POImv_a_r - 51.8888767274335*v_POImv_b_i + 42.5650941904727*v_POImv_b_r + 181.611068546017*v_POImv_c_i - 148.977829666654*v_POImv_c_r - 0.282682270461039*v_STlv_a_i + 1.69609362276623*v_STlv_a_r + 0.282682270461039*v_STlv_c_i - 1.69609362276623*v_STlv_c_r + 51.8945073580641*v_STmv_a_i - 42.5988786863508*v_STmv_a_r + 51.8945073580641*v_STmv_b_i - 42.5988786863508*v_STmv_b_r - 181.622329807278*v_STmv_c_i + 149.045395453986*v_STmv_c_r
        struct[0].g[60,0] = -i_l_W1mv_W2mv_a_r + 5.95911318666618*v_W1mv_a_i + 7.26444274184068*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r - 5.95911318666618*v_W2mv_a_i - 7.26444274184068*v_W2mv_a_r + 1.70260376761891*v_W2mv_b_i + 2.07555506909734*v_W2mv_b_r + 1.70260376761891*v_W2mv_c_i + 2.07555506909734*v_W2mv_c_r
        struct[0].g[61,0] = -i_l_W1mv_W2mv_a_i + 7.26444274184068*v_W1mv_a_i - 5.95911318666618*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r - 7.26444274184068*v_W2mv_a_i + 5.95911318666618*v_W2mv_a_r + 2.07555506909734*v_W2mv_b_i - 1.70260376761891*v_W2mv_b_r + 2.07555506909734*v_W2mv_c_i - 1.70260376761891*v_W2mv_c_r
        struct[0].g[62,0] = -i_l_W1mv_W2mv_b_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r + 5.95911318666618*v_W1mv_b_i + 7.26444274184068*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.70260376761891*v_W2mv_a_i + 2.07555506909734*v_W2mv_a_r - 5.95911318666618*v_W2mv_b_i - 7.26444274184068*v_W2mv_b_r + 1.70260376761891*v_W2mv_c_i + 2.07555506909734*v_W2mv_c_r
        struct[0].g[63,0] = -i_l_W1mv_W2mv_b_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r + 7.26444274184068*v_W1mv_b_i - 5.95911318666618*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 2.07555506909734*v_W2mv_a_i - 1.70260376761891*v_W2mv_a_r - 7.26444274184068*v_W2mv_b_i + 5.95911318666618*v_W2mv_b_r + 2.07555506909734*v_W2mv_c_i - 1.70260376761891*v_W2mv_c_r
        struct[0].g[64,0] = -i_l_W1mv_W2mv_c_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r + 5.95911318666618*v_W1mv_c_i + 7.26444274184068*v_W1mv_c_r + 1.70260376761891*v_W2mv_a_i + 2.07555506909734*v_W2mv_a_r + 1.70260376761891*v_W2mv_b_i + 2.07555506909734*v_W2mv_b_r - 5.95911318666618*v_W2mv_c_i - 7.26444274184068*v_W2mv_c_r
        struct[0].g[65,0] = -i_l_W1mv_W2mv_c_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r + 7.26444274184068*v_W1mv_c_i - 5.95911318666618*v_W1mv_c_r + 2.07555506909734*v_W2mv_a_i - 1.70260376761891*v_W2mv_a_r + 2.07555506909734*v_W2mv_b_i - 1.70260376761891*v_W2mv_b_r - 7.26444274184068*v_W2mv_c_i + 5.95911318666618*v_W2mv_c_r
        struct[0].g[66,0] = -i_l_W2mv_W3mv_a_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r - 5.95911318666618*v_W3mv_a_i - 7.26444274184068*v_W3mv_a_r + 1.70260376761891*v_W3mv_b_i + 2.07555506909734*v_W3mv_b_r + 1.70260376761891*v_W3mv_c_i + 2.07555506909734*v_W3mv_c_r
        struct[0].g[67,0] = -i_l_W2mv_W3mv_a_i + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r - 7.26444274184068*v_W3mv_a_i + 5.95911318666618*v_W3mv_a_r + 2.07555506909734*v_W3mv_b_i - 1.70260376761891*v_W3mv_b_r + 2.07555506909734*v_W3mv_c_i - 1.70260376761891*v_W3mv_c_r
        struct[0].g[68,0] = -i_l_W2mv_W3mv_b_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.70260376761891*v_W3mv_a_i + 2.07555506909734*v_W3mv_a_r - 5.95911318666618*v_W3mv_b_i - 7.26444274184068*v_W3mv_b_r + 1.70260376761891*v_W3mv_c_i + 2.07555506909734*v_W3mv_c_r
        struct[0].g[69,0] = -i_l_W2mv_W3mv_b_i - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 2.07555506909734*v_W3mv_a_i - 1.70260376761891*v_W3mv_a_r - 7.26444274184068*v_W3mv_b_i + 5.95911318666618*v_W3mv_b_r + 2.07555506909734*v_W3mv_c_i - 1.70260376761891*v_W3mv_c_r
        struct[0].g[70,0] = -i_l_W2mv_W3mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r + 1.70260376761891*v_W3mv_a_i + 2.07555506909734*v_W3mv_a_r + 1.70260376761891*v_W3mv_b_i + 2.07555506909734*v_W3mv_b_r - 5.95911318666618*v_W3mv_c_i - 7.26444274184068*v_W3mv_c_r
        struct[0].g[71,0] = -i_l_W2mv_W3mv_c_i - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r + 2.07555506909734*v_W3mv_a_i - 1.70260376761891*v_W3mv_a_r + 2.07555506909734*v_W3mv_b_i - 1.70260376761891*v_W3mv_b_r - 7.26444274184068*v_W3mv_c_i + 5.95911318666618*v_W3mv_c_r
        struct[0].g[72,0] = -i_l_W3mv_POImv_a_r - 5.95911318666618*v_POImv_a_i - 7.26444274184068*v_POImv_a_r + 1.70260376761891*v_POImv_b_i + 2.07555506909734*v_POImv_b_r + 1.70260376761891*v_POImv_c_i + 2.07555506909734*v_POImv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[73,0] = -i_l_W3mv_POImv_a_i - 7.26444274184068*v_POImv_a_i + 5.95911318666618*v_POImv_a_r + 2.07555506909734*v_POImv_b_i - 1.70260376761891*v_POImv_b_r + 2.07555506909734*v_POImv_c_i - 1.70260376761891*v_POImv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[74,0] = -i_l_W3mv_POImv_b_r + 1.70260376761891*v_POImv_a_i + 2.07555506909734*v_POImv_a_r - 5.95911318666618*v_POImv_b_i - 7.26444274184068*v_POImv_b_r + 1.70260376761891*v_POImv_c_i + 2.07555506909734*v_POImv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[75,0] = -i_l_W3mv_POImv_b_i + 2.07555506909734*v_POImv_a_i - 1.70260376761891*v_POImv_a_r - 7.26444274184068*v_POImv_b_i + 5.95911318666618*v_POImv_b_r + 2.07555506909734*v_POImv_c_i - 1.70260376761891*v_POImv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[76,0] = -i_l_W3mv_POImv_c_r + 1.70260376761891*v_POImv_a_i + 2.07555506909734*v_POImv_a_r + 1.70260376761891*v_POImv_b_i + 2.07555506909734*v_POImv_b_r - 5.95911318666618*v_POImv_c_i - 7.26444274184068*v_POImv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[77,0] = -i_l_W3mv_POImv_c_i + 2.07555506909734*v_POImv_a_i - 1.70260376761891*v_POImv_a_r + 2.07555506909734*v_POImv_b_i - 1.70260376761891*v_POImv_b_r - 7.26444274184068*v_POImv_c_i + 5.95911318666618*v_POImv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[78,0] = -i_l_STmv_POImv_a_r - 148.977829666654*v_POImv_a_i - 181.611068546017*v_POImv_a_r + 42.5650941904727*v_POImv_b_i + 51.8888767274334*v_POImv_b_r + 42.5650941904727*v_POImv_c_i + 51.8888767274334*v_POImv_c_r + 148.977829666654*v_STmv_a_i + 181.611068546017*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274334*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274334*v_STmv_c_r
        struct[0].g[79,0] = -i_l_STmv_POImv_a_i - 181.611068546017*v_POImv_a_i + 148.977829666654*v_POImv_a_r + 51.8888767274334*v_POImv_b_i - 42.5650941904727*v_POImv_b_r + 51.8888767274334*v_POImv_c_i - 42.5650941904727*v_POImv_c_r + 181.611068546017*v_STmv_a_i - 148.977829666654*v_STmv_a_r - 51.8888767274334*v_STmv_b_i + 42.5650941904727*v_STmv_b_r - 51.8888767274334*v_STmv_c_i + 42.5650941904727*v_STmv_c_r
        struct[0].g[80,0] = -i_l_STmv_POImv_b_r + 42.5650941904727*v_POImv_a_i + 51.8888767274334*v_POImv_a_r - 148.977829666654*v_POImv_b_i - 181.611068546017*v_POImv_b_r + 42.5650941904727*v_POImv_c_i + 51.8888767274335*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r + 148.977829666654*v_STmv_b_i + 181.611068546017*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274335*v_STmv_c_r
        struct[0].g[81,0] = -i_l_STmv_POImv_b_i + 51.8888767274334*v_POImv_a_i - 42.5650941904727*v_POImv_a_r - 181.611068546017*v_POImv_b_i + 148.977829666654*v_POImv_b_r + 51.8888767274335*v_POImv_c_i - 42.5650941904727*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r + 181.611068546017*v_STmv_b_i - 148.977829666654*v_STmv_b_r - 51.8888767274335*v_STmv_c_i + 42.5650941904727*v_STmv_c_r
        struct[0].g[82,0] = -i_l_STmv_POImv_c_r + 42.5650941904727*v_POImv_a_i + 51.8888767274334*v_POImv_a_r + 42.5650941904727*v_POImv_b_i + 51.8888767274335*v_POImv_b_r - 148.977829666654*v_POImv_c_i - 181.611068546017*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274335*v_STmv_b_r + 148.977829666654*v_STmv_c_i + 181.611068546017*v_STmv_c_r
        struct[0].g[83,0] = -i_l_STmv_POImv_c_i + 51.8888767274334*v_POImv_a_i - 42.5650941904727*v_POImv_a_r + 51.8888767274335*v_POImv_b_i - 42.5650941904727*v_POImv_b_r - 181.611068546017*v_POImv_c_i + 148.977829666654*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r - 51.8888767274335*v_STmv_b_i + 42.5650941904727*v_STmv_b_r + 181.611068546017*v_STmv_c_i - 148.977829666654*v_STmv_c_r
        struct[0].g[84,0] = -i_l_POI_GRID_a_r - 0.040290088638195*v_GRID_a_i - 0.024174053182917*v_GRID_a_r - 4.66248501556824e-18*v_GRID_b_i + 4.31760362252812e-18*v_GRID_b_r - 4.19816664496737e-18*v_GRID_c_i + 3.49608108880335e-18*v_GRID_c_r + 0.040290088638195*v_POI_a_i + 0.024174053182917*v_POI_a_r + 4.66248501556824e-18*v_POI_b_i - 4.31760362252812e-18*v_POI_b_r + 4.19816664496737e-18*v_POI_c_i - 3.49608108880335e-18*v_POI_c_r
        struct[0].g[85,0] = -i_l_POI_GRID_a_i - 0.024174053182917*v_GRID_a_i + 0.040290088638195*v_GRID_a_r + 4.31760362252812e-18*v_GRID_b_i + 4.66248501556824e-18*v_GRID_b_r + 3.49608108880335e-18*v_GRID_c_i + 4.19816664496737e-18*v_GRID_c_r + 0.024174053182917*v_POI_a_i - 0.040290088638195*v_POI_a_r - 4.31760362252812e-18*v_POI_b_i - 4.66248501556824e-18*v_POI_b_r - 3.49608108880335e-18*v_POI_c_i - 4.19816664496737e-18*v_POI_c_r
        struct[0].g[86,0] = -i_l_POI_GRID_b_r - 6.30775359573304e-19*v_GRID_a_i + 2.07254761002657e-18*v_GRID_a_r - 0.040290088638195*v_GRID_b_i - 0.024174053182917*v_GRID_b_r - 9.01107656533306e-19*v_GRID_c_i + 1.78419315993592e-17*v_GRID_c_r + 6.30775359573304e-19*v_POI_a_i - 2.07254761002657e-18*v_POI_a_r + 0.040290088638195*v_POI_b_i + 0.024174053182917*v_POI_b_r + 9.01107656533306e-19*v_POI_c_i - 1.78419315993592e-17*v_POI_c_r
        struct[0].g[87,0] = -i_l_POI_GRID_b_i + 2.07254761002657e-18*v_GRID_a_i + 6.30775359573304e-19*v_GRID_a_r - 0.024174053182917*v_GRID_b_i + 0.040290088638195*v_GRID_b_r + 1.78419315993592e-17*v_GRID_c_i + 9.01107656533306e-19*v_GRID_c_r - 2.07254761002657e-18*v_POI_a_i - 6.30775359573304e-19*v_POI_a_r + 0.024174053182917*v_POI_b_i - 0.040290088638195*v_POI_b_r - 1.78419315993592e-17*v_POI_c_i - 9.01107656533306e-19*v_POI_c_r
        struct[0].g[88,0] = -i_l_POI_GRID_c_r + 7.20886125226632e-19*v_GRID_a_i + 1.35166148479994e-18*v_GRID_a_r + 4.50553828266631e-19*v_GRID_b_i + 1.71210454741325e-17*v_GRID_b_r - 0.040290088638195*v_GRID_c_i - 0.024174053182917*v_GRID_c_r - 7.20886125226632e-19*v_POI_a_i - 1.35166148479994e-18*v_POI_a_r - 4.50553828266631e-19*v_POI_b_i - 1.71210454741325e-17*v_POI_b_r + 0.040290088638195*v_POI_c_i + 0.024174053182917*v_POI_c_r
        struct[0].g[89,0] = -i_l_POI_GRID_c_i + 1.35166148479994e-18*v_GRID_a_i - 7.20886125226632e-19*v_GRID_a_r + 1.71210454741325e-17*v_GRID_b_i - 4.50553828266631e-19*v_GRID_b_r - 0.024174053182917*v_GRID_c_i + 0.040290088638195*v_GRID_c_r - 1.35166148479994e-18*v_POI_a_i + 7.20886125226632e-19*v_POI_a_r - 1.71210454741325e-17*v_POI_b_i + 4.50553828266631e-19*v_POI_b_r + 0.024174053182917*v_POI_c_i - 0.040290088638195*v_POI_c_r
        struct[0].g[90,0] = i_W1lv_a_i*v_W1lv_a_i + i_W1lv_a_r*v_W1lv_a_r - p_W1lv_a
        struct[0].g[91,0] = i_W1lv_b_i*v_W1lv_b_i + i_W1lv_b_r*v_W1lv_b_r - p_W1lv_b
        struct[0].g[92,0] = i_W1lv_c_i*v_W1lv_c_i + i_W1lv_c_r*v_W1lv_c_r - p_W1lv_c
        struct[0].g[93,0] = -i_W1lv_a_i*v_W1lv_a_r + i_W1lv_a_r*v_W1lv_a_i - q_W1lv_a
        struct[0].g[94,0] = -i_W1lv_b_i*v_W1lv_b_r + i_W1lv_b_r*v_W1lv_b_i - q_W1lv_b
        struct[0].g[95,0] = -i_W1lv_c_i*v_W1lv_c_r + i_W1lv_c_r*v_W1lv_c_i - q_W1lv_c
        struct[0].g[96,0] = -v_m_W1lv + (v_W1lv_a_i**2 + v_W1lv_a_r**2)**0.5/V_base_W1lv
        struct[0].g[97,0] = -v_m_W1mv + (v_W1mv_a_i**2 + v_W1mv_a_r**2)**0.5/V_base_W1mv
        struct[0].g[98,0] = Dq_r_W1lv + K_p_v_W1lv*(Dv_r_W1lv - u_ctrl_v_W1lv*v_m_W1mv + v_loc_ref_W1lv - v_m_W1lv*(1.0 - u_ctrl_v_W1lv)) - i_reac_ref_W1lv
        struct[0].g[99,0] = S_base_W1lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)])) - q_ref_W1lv
        struct[0].g[100,0] = i_W2lv_a_i*v_W2lv_a_i + i_W2lv_a_r*v_W2lv_a_r - p_W2lv_a
        struct[0].g[101,0] = i_W2lv_b_i*v_W2lv_b_i + i_W2lv_b_r*v_W2lv_b_r - p_W2lv_b
        struct[0].g[102,0] = i_W2lv_c_i*v_W2lv_c_i + i_W2lv_c_r*v_W2lv_c_r - p_W2lv_c
        struct[0].g[103,0] = -i_W2lv_a_i*v_W2lv_a_r + i_W2lv_a_r*v_W2lv_a_i - q_W2lv_a
        struct[0].g[104,0] = -i_W2lv_b_i*v_W2lv_b_r + i_W2lv_b_r*v_W2lv_b_i - q_W2lv_b
        struct[0].g[105,0] = -i_W2lv_c_i*v_W2lv_c_r + i_W2lv_c_r*v_W2lv_c_i - q_W2lv_c
        struct[0].g[106,0] = -v_m_W2lv + (v_W2lv_a_i**2 + v_W2lv_a_r**2)**0.5/V_base_W2lv
        struct[0].g[107,0] = -v_m_W2mv + (v_W2mv_a_i**2 + v_W2mv_a_r**2)**0.5/V_base_W2mv
        struct[0].g[108,0] = Dq_r_W2lv + K_p_v_W2lv*(Dv_r_W2lv - u_ctrl_v_W2lv*v_m_W2mv + v_loc_ref_W2lv - v_m_W2lv*(1.0 - u_ctrl_v_W2lv)) - i_reac_ref_W2lv
        struct[0].g[109,0] = S_base_W2lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)])) - q_ref_W2lv
        struct[0].g[110,0] = i_W3lv_a_i*v_W3lv_a_i + i_W3lv_a_r*v_W3lv_a_r - p_W3lv_a
        struct[0].g[111,0] = i_W3lv_b_i*v_W3lv_b_i + i_W3lv_b_r*v_W3lv_b_r - p_W3lv_b
        struct[0].g[112,0] = i_W3lv_c_i*v_W3lv_c_i + i_W3lv_c_r*v_W3lv_c_r - p_W3lv_c
        struct[0].g[113,0] = -i_W3lv_a_i*v_W3lv_a_r + i_W3lv_a_r*v_W3lv_a_i - q_W3lv_a
        struct[0].g[114,0] = -i_W3lv_b_i*v_W3lv_b_r + i_W3lv_b_r*v_W3lv_b_i - q_W3lv_b
        struct[0].g[115,0] = -i_W3lv_c_i*v_W3lv_c_r + i_W3lv_c_r*v_W3lv_c_i - q_W3lv_c
        struct[0].g[116,0] = -v_m_W3lv + (v_W3lv_a_i**2 + v_W3lv_a_r**2)**0.5/V_base_W3lv
        struct[0].g[117,0] = -v_m_W3mv + (v_W3mv_a_i**2 + v_W3mv_a_r**2)**0.5/V_base_W3mv
        struct[0].g[118,0] = Dq_r_W3lv + K_p_v_W3lv*(Dv_r_W3lv - u_ctrl_v_W3lv*v_m_W3mv + v_loc_ref_W3lv - v_m_W3lv*(1.0 - u_ctrl_v_W3lv)) - i_reac_ref_W3lv
        struct[0].g[119,0] = S_base_W3lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)])) - q_ref_W3lv
        struct[0].g[120,0] = i_STlv_a_i*v_STlv_a_i + i_STlv_a_r*v_STlv_a_r - p_STlv_a
        struct[0].g[121,0] = i_STlv_b_i*v_STlv_b_i + i_STlv_b_r*v_STlv_b_r - p_STlv_b
        struct[0].g[122,0] = i_STlv_c_i*v_STlv_c_i + i_STlv_c_r*v_STlv_c_r - p_STlv_c
        struct[0].g[123,0] = -i_STlv_a_i*v_STlv_a_r + i_STlv_a_r*v_STlv_a_i - q_STlv_a
        struct[0].g[124,0] = -i_STlv_b_i*v_STlv_b_r + i_STlv_b_r*v_STlv_b_i - q_STlv_b
        struct[0].g[125,0] = -i_STlv_c_i*v_STlv_c_r + i_STlv_c_r*v_STlv_c_i - q_STlv_c
        struct[0].g[126,0] = -v_m_STlv + (v_STlv_a_i**2 + v_STlv_a_r**2)**0.5/V_base_STlv
        struct[0].g[127,0] = -v_m_STmv + (v_STmv_a_i**2 + v_STmv_a_r**2)**0.5/V_base_STmv
        struct[0].g[128,0] = Dq_r_STlv + K_p_v_STlv*(Dv_r_STlv - u_ctrl_v_STlv*v_m_STmv + v_loc_ref_STlv - v_m_STlv*(1.0 - u_ctrl_v_STlv)) - i_reac_ref_STlv
        struct[0].g[129,0] = S_base_STlv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)])) - q_ref_STlv
    
    # Outputs:
    if mode == 3:

    
        pass

    if mode == 10:

        struct[0].Fx[0,0] = -1/T_pq_W1lv
        struct[0].Fx[1,1] = -1/T_pq_W1lv
        struct[0].Fx[2,2] = -1/T_pq_W1lv
        struct[0].Fx[3,3] = -1/T_pq_W1lv
        struct[0].Fx[4,4] = -1/T_pq_W1lv
        struct[0].Fx[5,5] = -1/T_pq_W1lv
        struct[0].Fx[6,6] = -1/T_pq_W2lv
        struct[0].Fx[7,7] = -1/T_pq_W2lv
        struct[0].Fx[8,8] = -1/T_pq_W2lv
        struct[0].Fx[9,9] = -1/T_pq_W2lv
        struct[0].Fx[10,10] = -1/T_pq_W2lv
        struct[0].Fx[11,11] = -1/T_pq_W2lv
        struct[0].Fx[12,12] = -1/T_pq_W3lv
        struct[0].Fx[13,13] = -1/T_pq_W3lv
        struct[0].Fx[14,14] = -1/T_pq_W3lv
        struct[0].Fx[15,15] = -1/T_pq_W3lv
        struct[0].Fx[16,16] = -1/T_pq_W3lv
        struct[0].Fx[17,17] = -1/T_pq_W3lv
        struct[0].Fx[18,18] = -1/T_pq_STlv
        struct[0].Fx[19,19] = -1/T_pq_STlv
        struct[0].Fx[20,20] = -1/T_pq_STlv
        struct[0].Fx[21,21] = -1/T_pq_STlv
        struct[0].Fx[22,22] = -1/T_pq_STlv
        struct[0].Fx[23,23] = -1/T_pq_STlv

    if mode == 11:

        struct[0].Fy[3,99] = 1/(3*T_pq_W1lv)
        struct[0].Fy[4,99] = 1/(3*T_pq_W1lv)
        struct[0].Fy[5,99] = 1/(3*T_pq_W1lv)
        struct[0].Fy[9,109] = 1/(3*T_pq_W2lv)
        struct[0].Fy[10,109] = 1/(3*T_pq_W2lv)
        struct[0].Fy[11,109] = 1/(3*T_pq_W2lv)
        struct[0].Fy[15,119] = 1/(3*T_pq_W3lv)
        struct[0].Fy[16,119] = 1/(3*T_pq_W3lv)
        struct[0].Fy[17,119] = 1/(3*T_pq_W3lv)
        struct[0].Fy[21,129] = 1/(3*T_pq_STlv)
        struct[0].Fy[22,129] = 1/(3*T_pq_STlv)
        struct[0].Fy[23,129] = 1/(3*T_pq_STlv)

        struct[0].Gx[90,0] = -1
        struct[0].Gx[91,1] = -1
        struct[0].Gx[92,2] = -1
        struct[0].Gx[93,3] = -1
        struct[0].Gx[94,4] = -1
        struct[0].Gx[95,5] = -1
        struct[0].Gx[100,6] = -1
        struct[0].Gx[101,7] = -1
        struct[0].Gx[102,8] = -1
        struct[0].Gx[103,9] = -1
        struct[0].Gx[104,10] = -1
        struct[0].Gx[105,11] = -1
        struct[0].Gx[110,12] = -1
        struct[0].Gx[111,13] = -1
        struct[0].Gx[112,14] = -1
        struct[0].Gx[113,15] = -1
        struct[0].Gx[114,16] = -1
        struct[0].Gx[115,17] = -1
        struct[0].Gx[120,18] = -1
        struct[0].Gx[121,19] = -1
        struct[0].Gx[122,20] = -1
        struct[0].Gx[123,21] = -1
        struct[0].Gx[124,22] = -1
        struct[0].Gx[125,23] = -1

        struct[0].Gy[90,0] = i_W1lv_a_r
        struct[0].Gy[90,1] = i_W1lv_a_i
        struct[0].Gy[90,90] = v_W1lv_a_r
        struct[0].Gy[90,91] = v_W1lv_a_i
        struct[0].Gy[91,2] = i_W1lv_b_r
        struct[0].Gy[91,3] = i_W1lv_b_i
        struct[0].Gy[91,92] = v_W1lv_b_r
        struct[0].Gy[91,93] = v_W1lv_b_i
        struct[0].Gy[92,4] = i_W1lv_c_r
        struct[0].Gy[92,5] = i_W1lv_c_i
        struct[0].Gy[92,94] = v_W1lv_c_r
        struct[0].Gy[92,95] = v_W1lv_c_i
        struct[0].Gy[93,0] = -i_W1lv_a_i
        struct[0].Gy[93,1] = i_W1lv_a_r
        struct[0].Gy[93,90] = v_W1lv_a_i
        struct[0].Gy[93,91] = -v_W1lv_a_r
        struct[0].Gy[94,2] = -i_W1lv_b_i
        struct[0].Gy[94,3] = i_W1lv_b_r
        struct[0].Gy[94,92] = v_W1lv_b_i
        struct[0].Gy[94,93] = -v_W1lv_b_r
        struct[0].Gy[95,4] = -i_W1lv_c_i
        struct[0].Gy[95,5] = i_W1lv_c_r
        struct[0].Gy[95,94] = v_W1lv_c_i
        struct[0].Gy[95,95] = -v_W1lv_c_r
        struct[0].Gy[96,0] = 1.0*v_W1lv_a_r*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)/V_base_W1lv
        struct[0].Gy[96,1] = 1.0*v_W1lv_a_i*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)/V_base_W1lv
        struct[0].Gy[97,36] = 1.0*v_W1mv_a_r*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)/V_base_W1mv
        struct[0].Gy[97,37] = 1.0*v_W1mv_a_i*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)/V_base_W1mv
        struct[0].Gy[98,96] = K_p_v_W1lv*(u_ctrl_v_W1lv - 1.0)
        struct[0].Gy[98,97] = -K_p_v_W1lv*u_ctrl_v_W1lv
        struct[0].Gy[99,58] = 1.0*S_base_W1lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)]))
        struct[0].Gy[99,59] = 1.0*S_base_W1lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)]))
        struct[0].Gy[99,98] = S_base_W1lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W1lv < i_reac_ref_W1lv) | (I_max_W1lv < -i_reac_ref_W1lv)), (1, True)]))
        struct[0].Gy[100,6] = i_W2lv_a_r
        struct[0].Gy[100,7] = i_W2lv_a_i
        struct[0].Gy[100,100] = v_W2lv_a_r
        struct[0].Gy[100,101] = v_W2lv_a_i
        struct[0].Gy[101,8] = i_W2lv_b_r
        struct[0].Gy[101,9] = i_W2lv_b_i
        struct[0].Gy[101,102] = v_W2lv_b_r
        struct[0].Gy[101,103] = v_W2lv_b_i
        struct[0].Gy[102,10] = i_W2lv_c_r
        struct[0].Gy[102,11] = i_W2lv_c_i
        struct[0].Gy[102,104] = v_W2lv_c_r
        struct[0].Gy[102,105] = v_W2lv_c_i
        struct[0].Gy[103,6] = -i_W2lv_a_i
        struct[0].Gy[103,7] = i_W2lv_a_r
        struct[0].Gy[103,100] = v_W2lv_a_i
        struct[0].Gy[103,101] = -v_W2lv_a_r
        struct[0].Gy[104,8] = -i_W2lv_b_i
        struct[0].Gy[104,9] = i_W2lv_b_r
        struct[0].Gy[104,102] = v_W2lv_b_i
        struct[0].Gy[104,103] = -v_W2lv_b_r
        struct[0].Gy[105,10] = -i_W2lv_c_i
        struct[0].Gy[105,11] = i_W2lv_c_r
        struct[0].Gy[105,104] = v_W2lv_c_i
        struct[0].Gy[105,105] = -v_W2lv_c_r
        struct[0].Gy[106,6] = 1.0*v_W2lv_a_r*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)/V_base_W2lv
        struct[0].Gy[106,7] = 1.0*v_W2lv_a_i*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)/V_base_W2lv
        struct[0].Gy[107,42] = 1.0*v_W2mv_a_r*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)/V_base_W2mv
        struct[0].Gy[107,43] = 1.0*v_W2mv_a_i*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)/V_base_W2mv
        struct[0].Gy[108,106] = K_p_v_W2lv*(u_ctrl_v_W2lv - 1.0)
        struct[0].Gy[108,107] = -K_p_v_W2lv*u_ctrl_v_W2lv
        struct[0].Gy[109,58] = 1.0*S_base_W2lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)]))
        struct[0].Gy[109,59] = 1.0*S_base_W2lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)]))
        struct[0].Gy[109,108] = S_base_W2lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W2lv < i_reac_ref_W2lv) | (I_max_W2lv < -i_reac_ref_W2lv)), (1, True)]))
        struct[0].Gy[110,12] = i_W3lv_a_r
        struct[0].Gy[110,13] = i_W3lv_a_i
        struct[0].Gy[110,110] = v_W3lv_a_r
        struct[0].Gy[110,111] = v_W3lv_a_i
        struct[0].Gy[111,14] = i_W3lv_b_r
        struct[0].Gy[111,15] = i_W3lv_b_i
        struct[0].Gy[111,112] = v_W3lv_b_r
        struct[0].Gy[111,113] = v_W3lv_b_i
        struct[0].Gy[112,16] = i_W3lv_c_r
        struct[0].Gy[112,17] = i_W3lv_c_i
        struct[0].Gy[112,114] = v_W3lv_c_r
        struct[0].Gy[112,115] = v_W3lv_c_i
        struct[0].Gy[113,12] = -i_W3lv_a_i
        struct[0].Gy[113,13] = i_W3lv_a_r
        struct[0].Gy[113,110] = v_W3lv_a_i
        struct[0].Gy[113,111] = -v_W3lv_a_r
        struct[0].Gy[114,14] = -i_W3lv_b_i
        struct[0].Gy[114,15] = i_W3lv_b_r
        struct[0].Gy[114,112] = v_W3lv_b_i
        struct[0].Gy[114,113] = -v_W3lv_b_r
        struct[0].Gy[115,16] = -i_W3lv_c_i
        struct[0].Gy[115,17] = i_W3lv_c_r
        struct[0].Gy[115,114] = v_W3lv_c_i
        struct[0].Gy[115,115] = -v_W3lv_c_r
        struct[0].Gy[116,12] = 1.0*v_W3lv_a_r*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)/V_base_W3lv
        struct[0].Gy[116,13] = 1.0*v_W3lv_a_i*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)/V_base_W3lv
        struct[0].Gy[117,48] = 1.0*v_W3mv_a_r*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)/V_base_W3mv
        struct[0].Gy[117,49] = 1.0*v_W3mv_a_i*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)/V_base_W3mv
        struct[0].Gy[118,116] = K_p_v_W3lv*(u_ctrl_v_W3lv - 1.0)
        struct[0].Gy[118,117] = -K_p_v_W3lv*u_ctrl_v_W3lv
        struct[0].Gy[119,58] = 1.0*S_base_W3lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)]))
        struct[0].Gy[119,59] = 1.0*S_base_W3lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)]))
        struct[0].Gy[119,118] = S_base_W3lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W3lv < i_reac_ref_W3lv) | (I_max_W3lv < -i_reac_ref_W3lv)), (1, True)]))
        struct[0].Gy[120,18] = i_STlv_a_r
        struct[0].Gy[120,19] = i_STlv_a_i
        struct[0].Gy[120,120] = v_STlv_a_r
        struct[0].Gy[120,121] = v_STlv_a_i
        struct[0].Gy[121,20] = i_STlv_b_r
        struct[0].Gy[121,21] = i_STlv_b_i
        struct[0].Gy[121,122] = v_STlv_b_r
        struct[0].Gy[121,123] = v_STlv_b_i
        struct[0].Gy[122,22] = i_STlv_c_r
        struct[0].Gy[122,23] = i_STlv_c_i
        struct[0].Gy[122,124] = v_STlv_c_r
        struct[0].Gy[122,125] = v_STlv_c_i
        struct[0].Gy[123,18] = -i_STlv_a_i
        struct[0].Gy[123,19] = i_STlv_a_r
        struct[0].Gy[123,120] = v_STlv_a_i
        struct[0].Gy[123,121] = -v_STlv_a_r
        struct[0].Gy[124,20] = -i_STlv_b_i
        struct[0].Gy[124,21] = i_STlv_b_r
        struct[0].Gy[124,122] = v_STlv_b_i
        struct[0].Gy[124,123] = -v_STlv_b_r
        struct[0].Gy[125,22] = -i_STlv_c_i
        struct[0].Gy[125,23] = i_STlv_c_r
        struct[0].Gy[125,124] = v_STlv_c_i
        struct[0].Gy[125,125] = -v_STlv_c_r
        struct[0].Gy[126,18] = 1.0*v_STlv_a_r*(v_STlv_a_i**2 + v_STlv_a_r**2)**(-0.5)/V_base_STlv
        struct[0].Gy[126,19] = 1.0*v_STlv_a_i*(v_STlv_a_i**2 + v_STlv_a_r**2)**(-0.5)/V_base_STlv
        struct[0].Gy[127,54] = 1.0*v_STmv_a_r*(v_STmv_a_i**2 + v_STmv_a_r**2)**(-0.5)/V_base_STmv
        struct[0].Gy[127,55] = 1.0*v_STmv_a_i*(v_STmv_a_i**2 + v_STmv_a_r**2)**(-0.5)/V_base_STmv
        struct[0].Gy[128,126] = K_p_v_STlv*(u_ctrl_v_STlv - 1.0)
        struct[0].Gy[128,127] = -K_p_v_STlv*u_ctrl_v_STlv
        struct[0].Gy[129,58] = 1.0*S_base_STlv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)]))
        struct[0].Gy[129,59] = 1.0*S_base_STlv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)]))
        struct[0].Gy[129,128] = S_base_STlv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_STlv < i_reac_ref_STlv) | (I_max_STlv < -i_reac_ref_STlv)), (1, True)]))

    if mode > 12:

        struct[0].Fu[0,42] = 1/(3*T_pq_W1lv)
        struct[0].Fu[0,43] = -(-p_W1lv_a + p_ref_W1lv/3)/T_pq_W1lv**2
        struct[0].Fu[1,42] = 1/(3*T_pq_W1lv)
        struct[0].Fu[1,43] = -(-p_W1lv_b + p_ref_W1lv/3)/T_pq_W1lv**2
        struct[0].Fu[2,42] = 1/(3*T_pq_W1lv)
        struct[0].Fu[2,43] = -(-p_W1lv_c + p_ref_W1lv/3)/T_pq_W1lv**2
        struct[0].Fu[3,43] = -(-q_W1lv_a + q_ref_W1lv/3)/T_pq_W1lv**2
        struct[0].Fu[4,43] = -(-q_W1lv_b + q_ref_W1lv/3)/T_pq_W1lv**2
        struct[0].Fu[5,43] = -(-q_W1lv_c + q_ref_W1lv/3)/T_pq_W1lv**2
        struct[0].Fu[6,47] = 1/(3*T_pq_W2lv)
        struct[0].Fu[6,48] = -(-p_W2lv_a + p_ref_W2lv/3)/T_pq_W2lv**2
        struct[0].Fu[7,47] = 1/(3*T_pq_W2lv)
        struct[0].Fu[7,48] = -(-p_W2lv_b + p_ref_W2lv/3)/T_pq_W2lv**2
        struct[0].Fu[8,47] = 1/(3*T_pq_W2lv)
        struct[0].Fu[8,48] = -(-p_W2lv_c + p_ref_W2lv/3)/T_pq_W2lv**2
        struct[0].Fu[9,48] = -(-q_W2lv_a + q_ref_W2lv/3)/T_pq_W2lv**2
        struct[0].Fu[10,48] = -(-q_W2lv_b + q_ref_W2lv/3)/T_pq_W2lv**2
        struct[0].Fu[11,48] = -(-q_W2lv_c + q_ref_W2lv/3)/T_pq_W2lv**2
        struct[0].Fu[12,52] = 1/(3*T_pq_W3lv)
        struct[0].Fu[12,53] = -(-p_W3lv_a + p_ref_W3lv/3)/T_pq_W3lv**2
        struct[0].Fu[13,52] = 1/(3*T_pq_W3lv)
        struct[0].Fu[13,53] = -(-p_W3lv_b + p_ref_W3lv/3)/T_pq_W3lv**2
        struct[0].Fu[14,52] = 1/(3*T_pq_W3lv)
        struct[0].Fu[14,53] = -(-p_W3lv_c + p_ref_W3lv/3)/T_pq_W3lv**2
        struct[0].Fu[15,53] = -(-q_W3lv_a + q_ref_W3lv/3)/T_pq_W3lv**2
        struct[0].Fu[16,53] = -(-q_W3lv_b + q_ref_W3lv/3)/T_pq_W3lv**2
        struct[0].Fu[17,53] = -(-q_W3lv_c + q_ref_W3lv/3)/T_pq_W3lv**2
        struct[0].Fu[18,57] = 1/(3*T_pq_STlv)
        struct[0].Fu[18,58] = -(-p_STlv_a + p_ref_STlv/3)/T_pq_STlv**2
        struct[0].Fu[19,57] = 1/(3*T_pq_STlv)
        struct[0].Fu[19,58] = -(-p_STlv_b + p_ref_STlv/3)/T_pq_STlv**2
        struct[0].Fu[20,57] = 1/(3*T_pq_STlv)
        struct[0].Fu[20,58] = -(-p_STlv_c + p_ref_STlv/3)/T_pq_STlv**2
        struct[0].Fu[21,58] = -(-q_STlv_a + q_ref_STlv/3)/T_pq_STlv**2
        struct[0].Fu[22,58] = -(-q_STlv_b + q_ref_STlv/3)/T_pq_STlv**2
        struct[0].Fu[23,58] = -(-q_STlv_c + q_ref_STlv/3)/T_pq_STlv**2

        struct[0].Gu[98,44] = K_p_v_W1lv
        struct[0].Gu[98,45] = K_p_v_W1lv
        struct[0].Gu[108,49] = K_p_v_W2lv
        struct[0].Gu[108,50] = K_p_v_W2lv
        struct[0].Gu[118,54] = K_p_v_W3lv
        struct[0].Gu[118,55] = K_p_v_W3lv
        struct[0].Gu[128,59] = K_p_v_STlv
        struct[0].Gu[128,60] = K_p_v_STlv






def ini_nn(struct,mode):

    # Parameters:
    u_ctrl_v_W1lv = struct[0].u_ctrl_v_W1lv
    K_p_v_W1lv = struct[0].K_p_v_W1lv
    K_i_v_W1lv = struct[0].K_i_v_W1lv
    V_base_W1lv = struct[0].V_base_W1lv
    V_base_W1mv = struct[0].V_base_W1mv
    S_base_W1lv = struct[0].S_base_W1lv
    I_max_W1lv = struct[0].I_max_W1lv
    u_ctrl_v_W2lv = struct[0].u_ctrl_v_W2lv
    K_p_v_W2lv = struct[0].K_p_v_W2lv
    K_i_v_W2lv = struct[0].K_i_v_W2lv
    V_base_W2lv = struct[0].V_base_W2lv
    V_base_W2mv = struct[0].V_base_W2mv
    S_base_W2lv = struct[0].S_base_W2lv
    I_max_W2lv = struct[0].I_max_W2lv
    u_ctrl_v_W3lv = struct[0].u_ctrl_v_W3lv
    K_p_v_W3lv = struct[0].K_p_v_W3lv
    K_i_v_W3lv = struct[0].K_i_v_W3lv
    V_base_W3lv = struct[0].V_base_W3lv
    V_base_W3mv = struct[0].V_base_W3mv
    S_base_W3lv = struct[0].S_base_W3lv
    I_max_W3lv = struct[0].I_max_W3lv
    u_ctrl_v_STlv = struct[0].u_ctrl_v_STlv
    K_p_v_STlv = struct[0].K_p_v_STlv
    K_i_v_STlv = struct[0].K_i_v_STlv
    V_base_STlv = struct[0].V_base_STlv
    V_base_STmv = struct[0].V_base_STmv
    S_base_STlv = struct[0].S_base_STlv
    I_max_STlv = struct[0].I_max_STlv
    
    # Inputs:
    v_GRID_a_r = struct[0].v_GRID_a_r
    v_GRID_a_i = struct[0].v_GRID_a_i
    v_GRID_b_r = struct[0].v_GRID_b_r
    v_GRID_b_i = struct[0].v_GRID_b_i
    v_GRID_c_r = struct[0].v_GRID_c_r
    v_GRID_c_i = struct[0].v_GRID_c_i
    i_POI_a_r = struct[0].i_POI_a_r
    i_POI_a_i = struct[0].i_POI_a_i
    i_POI_b_r = struct[0].i_POI_b_r
    i_POI_b_i = struct[0].i_POI_b_i
    i_POI_c_r = struct[0].i_POI_c_r
    i_POI_c_i = struct[0].i_POI_c_i
    i_POImv_a_r = struct[0].i_POImv_a_r
    i_POImv_a_i = struct[0].i_POImv_a_i
    i_POImv_b_r = struct[0].i_POImv_b_r
    i_POImv_b_i = struct[0].i_POImv_b_i
    i_POImv_c_r = struct[0].i_POImv_c_r
    i_POImv_c_i = struct[0].i_POImv_c_i
    i_W1mv_a_r = struct[0].i_W1mv_a_r
    i_W1mv_a_i = struct[0].i_W1mv_a_i
    i_W1mv_b_r = struct[0].i_W1mv_b_r
    i_W1mv_b_i = struct[0].i_W1mv_b_i
    i_W1mv_c_r = struct[0].i_W1mv_c_r
    i_W1mv_c_i = struct[0].i_W1mv_c_i
    i_W2mv_a_r = struct[0].i_W2mv_a_r
    i_W2mv_a_i = struct[0].i_W2mv_a_i
    i_W2mv_b_r = struct[0].i_W2mv_b_r
    i_W2mv_b_i = struct[0].i_W2mv_b_i
    i_W2mv_c_r = struct[0].i_W2mv_c_r
    i_W2mv_c_i = struct[0].i_W2mv_c_i
    i_W3mv_a_r = struct[0].i_W3mv_a_r
    i_W3mv_a_i = struct[0].i_W3mv_a_i
    i_W3mv_b_r = struct[0].i_W3mv_b_r
    i_W3mv_b_i = struct[0].i_W3mv_b_i
    i_W3mv_c_r = struct[0].i_W3mv_c_r
    i_W3mv_c_i = struct[0].i_W3mv_c_i
    i_STmv_a_r = struct[0].i_STmv_a_r
    i_STmv_a_i = struct[0].i_STmv_a_i
    i_STmv_b_r = struct[0].i_STmv_b_r
    i_STmv_b_i = struct[0].i_STmv_b_i
    i_STmv_c_r = struct[0].i_STmv_c_r
    i_STmv_c_i = struct[0].i_STmv_c_i
    p_ref_W1lv = struct[0].p_ref_W1lv
    T_pq_W1lv = struct[0].T_pq_W1lv
    v_loc_ref_W1lv = struct[0].v_loc_ref_W1lv
    Dv_r_W1lv = struct[0].Dv_r_W1lv
    Dq_r_W1lv = struct[0].Dq_r_W1lv
    p_ref_W2lv = struct[0].p_ref_W2lv
    T_pq_W2lv = struct[0].T_pq_W2lv
    v_loc_ref_W2lv = struct[0].v_loc_ref_W2lv
    Dv_r_W2lv = struct[0].Dv_r_W2lv
    Dq_r_W2lv = struct[0].Dq_r_W2lv
    p_ref_W3lv = struct[0].p_ref_W3lv
    T_pq_W3lv = struct[0].T_pq_W3lv
    v_loc_ref_W3lv = struct[0].v_loc_ref_W3lv
    Dv_r_W3lv = struct[0].Dv_r_W3lv
    Dq_r_W3lv = struct[0].Dq_r_W3lv
    p_ref_STlv = struct[0].p_ref_STlv
    T_pq_STlv = struct[0].T_pq_STlv
    v_loc_ref_STlv = struct[0].v_loc_ref_STlv
    Dv_r_STlv = struct[0].Dv_r_STlv
    Dq_r_STlv = struct[0].Dq_r_STlv
    
    # Dynamical states:
    p_W1lv_a = struct[0].x[0,0]
    p_W1lv_b = struct[0].x[1,0]
    p_W1lv_c = struct[0].x[2,0]
    q_W1lv_a = struct[0].x[3,0]
    q_W1lv_b = struct[0].x[4,0]
    q_W1lv_c = struct[0].x[5,0]
    p_W2lv_a = struct[0].x[6,0]
    p_W2lv_b = struct[0].x[7,0]
    p_W2lv_c = struct[0].x[8,0]
    q_W2lv_a = struct[0].x[9,0]
    q_W2lv_b = struct[0].x[10,0]
    q_W2lv_c = struct[0].x[11,0]
    p_W3lv_a = struct[0].x[12,0]
    p_W3lv_b = struct[0].x[13,0]
    p_W3lv_c = struct[0].x[14,0]
    q_W3lv_a = struct[0].x[15,0]
    q_W3lv_b = struct[0].x[16,0]
    q_W3lv_c = struct[0].x[17,0]
    p_STlv_a = struct[0].x[18,0]
    p_STlv_b = struct[0].x[19,0]
    p_STlv_c = struct[0].x[20,0]
    q_STlv_a = struct[0].x[21,0]
    q_STlv_b = struct[0].x[22,0]
    q_STlv_c = struct[0].x[23,0]
    
    # Algebraic states:
    v_W1lv_a_r = struct[0].y_ini[0,0]
    v_W1lv_a_i = struct[0].y_ini[1,0]
    v_W1lv_b_r = struct[0].y_ini[2,0]
    v_W1lv_b_i = struct[0].y_ini[3,0]
    v_W1lv_c_r = struct[0].y_ini[4,0]
    v_W1lv_c_i = struct[0].y_ini[5,0]
    v_W2lv_a_r = struct[0].y_ini[6,0]
    v_W2lv_a_i = struct[0].y_ini[7,0]
    v_W2lv_b_r = struct[0].y_ini[8,0]
    v_W2lv_b_i = struct[0].y_ini[9,0]
    v_W2lv_c_r = struct[0].y_ini[10,0]
    v_W2lv_c_i = struct[0].y_ini[11,0]
    v_W3lv_a_r = struct[0].y_ini[12,0]
    v_W3lv_a_i = struct[0].y_ini[13,0]
    v_W3lv_b_r = struct[0].y_ini[14,0]
    v_W3lv_b_i = struct[0].y_ini[15,0]
    v_W3lv_c_r = struct[0].y_ini[16,0]
    v_W3lv_c_i = struct[0].y_ini[17,0]
    v_STlv_a_r = struct[0].y_ini[18,0]
    v_STlv_a_i = struct[0].y_ini[19,0]
    v_STlv_b_r = struct[0].y_ini[20,0]
    v_STlv_b_i = struct[0].y_ini[21,0]
    v_STlv_c_r = struct[0].y_ini[22,0]
    v_STlv_c_i = struct[0].y_ini[23,0]
    v_POI_a_r = struct[0].y_ini[24,0]
    v_POI_a_i = struct[0].y_ini[25,0]
    v_POI_b_r = struct[0].y_ini[26,0]
    v_POI_b_i = struct[0].y_ini[27,0]
    v_POI_c_r = struct[0].y_ini[28,0]
    v_POI_c_i = struct[0].y_ini[29,0]
    v_POImv_a_r = struct[0].y_ini[30,0]
    v_POImv_a_i = struct[0].y_ini[31,0]
    v_POImv_b_r = struct[0].y_ini[32,0]
    v_POImv_b_i = struct[0].y_ini[33,0]
    v_POImv_c_r = struct[0].y_ini[34,0]
    v_POImv_c_i = struct[0].y_ini[35,0]
    v_W1mv_a_r = struct[0].y_ini[36,0]
    v_W1mv_a_i = struct[0].y_ini[37,0]
    v_W1mv_b_r = struct[0].y_ini[38,0]
    v_W1mv_b_i = struct[0].y_ini[39,0]
    v_W1mv_c_r = struct[0].y_ini[40,0]
    v_W1mv_c_i = struct[0].y_ini[41,0]
    v_W2mv_a_r = struct[0].y_ini[42,0]
    v_W2mv_a_i = struct[0].y_ini[43,0]
    v_W2mv_b_r = struct[0].y_ini[44,0]
    v_W2mv_b_i = struct[0].y_ini[45,0]
    v_W2mv_c_r = struct[0].y_ini[46,0]
    v_W2mv_c_i = struct[0].y_ini[47,0]
    v_W3mv_a_r = struct[0].y_ini[48,0]
    v_W3mv_a_i = struct[0].y_ini[49,0]
    v_W3mv_b_r = struct[0].y_ini[50,0]
    v_W3mv_b_i = struct[0].y_ini[51,0]
    v_W3mv_c_r = struct[0].y_ini[52,0]
    v_W3mv_c_i = struct[0].y_ini[53,0]
    v_STmv_a_r = struct[0].y_ini[54,0]
    v_STmv_a_i = struct[0].y_ini[55,0]
    v_STmv_b_r = struct[0].y_ini[56,0]
    v_STmv_b_i = struct[0].y_ini[57,0]
    v_STmv_c_r = struct[0].y_ini[58,0]
    v_STmv_c_i = struct[0].y_ini[59,0]
    i_l_W1mv_W2mv_a_r = struct[0].y_ini[60,0]
    i_l_W1mv_W2mv_a_i = struct[0].y_ini[61,0]
    i_l_W1mv_W2mv_b_r = struct[0].y_ini[62,0]
    i_l_W1mv_W2mv_b_i = struct[0].y_ini[63,0]
    i_l_W1mv_W2mv_c_r = struct[0].y_ini[64,0]
    i_l_W1mv_W2mv_c_i = struct[0].y_ini[65,0]
    i_l_W2mv_W3mv_a_r = struct[0].y_ini[66,0]
    i_l_W2mv_W3mv_a_i = struct[0].y_ini[67,0]
    i_l_W2mv_W3mv_b_r = struct[0].y_ini[68,0]
    i_l_W2mv_W3mv_b_i = struct[0].y_ini[69,0]
    i_l_W2mv_W3mv_c_r = struct[0].y_ini[70,0]
    i_l_W2mv_W3mv_c_i = struct[0].y_ini[71,0]
    i_l_W3mv_POImv_a_r = struct[0].y_ini[72,0]
    i_l_W3mv_POImv_a_i = struct[0].y_ini[73,0]
    i_l_W3mv_POImv_b_r = struct[0].y_ini[74,0]
    i_l_W3mv_POImv_b_i = struct[0].y_ini[75,0]
    i_l_W3mv_POImv_c_r = struct[0].y_ini[76,0]
    i_l_W3mv_POImv_c_i = struct[0].y_ini[77,0]
    i_l_STmv_POImv_a_r = struct[0].y_ini[78,0]
    i_l_STmv_POImv_a_i = struct[0].y_ini[79,0]
    i_l_STmv_POImv_b_r = struct[0].y_ini[80,0]
    i_l_STmv_POImv_b_i = struct[0].y_ini[81,0]
    i_l_STmv_POImv_c_r = struct[0].y_ini[82,0]
    i_l_STmv_POImv_c_i = struct[0].y_ini[83,0]
    i_l_POI_GRID_a_r = struct[0].y_ini[84,0]
    i_l_POI_GRID_a_i = struct[0].y_ini[85,0]
    i_l_POI_GRID_b_r = struct[0].y_ini[86,0]
    i_l_POI_GRID_b_i = struct[0].y_ini[87,0]
    i_l_POI_GRID_c_r = struct[0].y_ini[88,0]
    i_l_POI_GRID_c_i = struct[0].y_ini[89,0]
    i_W1lv_a_r = struct[0].y_ini[90,0]
    i_W1lv_a_i = struct[0].y_ini[91,0]
    i_W1lv_b_r = struct[0].y_ini[92,0]
    i_W1lv_b_i = struct[0].y_ini[93,0]
    i_W1lv_c_r = struct[0].y_ini[94,0]
    i_W1lv_c_i = struct[0].y_ini[95,0]
    v_m_W1lv = struct[0].y_ini[96,0]
    v_m_W1mv = struct[0].y_ini[97,0]
    i_reac_ref_W1lv = struct[0].y_ini[98,0]
    q_ref_W1lv = struct[0].y_ini[99,0]
    i_W2lv_a_r = struct[0].y_ini[100,0]
    i_W2lv_a_i = struct[0].y_ini[101,0]
    i_W2lv_b_r = struct[0].y_ini[102,0]
    i_W2lv_b_i = struct[0].y_ini[103,0]
    i_W2lv_c_r = struct[0].y_ini[104,0]
    i_W2lv_c_i = struct[0].y_ini[105,0]
    v_m_W2lv = struct[0].y_ini[106,0]
    v_m_W2mv = struct[0].y_ini[107,0]
    i_reac_ref_W2lv = struct[0].y_ini[108,0]
    q_ref_W2lv = struct[0].y_ini[109,0]
    i_W3lv_a_r = struct[0].y_ini[110,0]
    i_W3lv_a_i = struct[0].y_ini[111,0]
    i_W3lv_b_r = struct[0].y_ini[112,0]
    i_W3lv_b_i = struct[0].y_ini[113,0]
    i_W3lv_c_r = struct[0].y_ini[114,0]
    i_W3lv_c_i = struct[0].y_ini[115,0]
    v_m_W3lv = struct[0].y_ini[116,0]
    v_m_W3mv = struct[0].y_ini[117,0]
    i_reac_ref_W3lv = struct[0].y_ini[118,0]
    q_ref_W3lv = struct[0].y_ini[119,0]
    i_STlv_a_r = struct[0].y_ini[120,0]
    i_STlv_a_i = struct[0].y_ini[121,0]
    i_STlv_b_r = struct[0].y_ini[122,0]
    i_STlv_b_i = struct[0].y_ini[123,0]
    i_STlv_c_r = struct[0].y_ini[124,0]
    i_STlv_c_i = struct[0].y_ini[125,0]
    v_m_STlv = struct[0].y_ini[126,0]
    v_m_STmv = struct[0].y_ini[127,0]
    i_reac_ref_STlv = struct[0].y_ini[128,0]
    q_ref_STlv = struct[0].y_ini[129,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-p_W1lv_a + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[1,0] = (-p_W1lv_b + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[2,0] = (-p_W1lv_c + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[3,0] = (-q_W1lv_a + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[4,0] = (-q_W1lv_b + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[5,0] = (-q_W1lv_c + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[6,0] = (-p_W2lv_a + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[7,0] = (-p_W2lv_b + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[8,0] = (-p_W2lv_c + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[9,0] = (-q_W2lv_a + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[10,0] = (-q_W2lv_b + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[11,0] = (-q_W2lv_c + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[12,0] = (-p_W3lv_a + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[13,0] = (-p_W3lv_b + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[14,0] = (-p_W3lv_c + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[15,0] = (-q_W3lv_a + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[16,0] = (-q_W3lv_b + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[17,0] = (-q_W3lv_c + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[18,0] = (-p_STlv_a + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[19,0] = (-p_STlv_b + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[20,0] = (-p_STlv_c + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[21,0] = (-q_STlv_a + q_ref_STlv/3)/T_pq_STlv
        struct[0].f[22,0] = (-q_STlv_b + q_ref_STlv/3)/T_pq_STlv
        struct[0].f[23,0] = (-q_STlv_c + q_ref_STlv/3)/T_pq_STlv
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_W1lv_a_r - 85.1513138847732*v_W1lv_a_i - 14.1918856474622*v_W1lv_a_r + 1.69609362276623*v_W1mv_a_i + 0.282682270461039*v_W1mv_a_r - 1.69609362276623*v_W1mv_c_i - 0.282682270461039*v_W1mv_c_r
        struct[0].g[1,0] = i_W1lv_a_i - 14.1918856474622*v_W1lv_a_i + 85.1513138847732*v_W1lv_a_r + 0.282682270461039*v_W1mv_a_i - 1.69609362276623*v_W1mv_a_r - 0.282682270461039*v_W1mv_c_i + 1.69609362276623*v_W1mv_c_r
        struct[0].g[2,0] = i_W1lv_b_r - 85.1513138847732*v_W1lv_b_i - 14.1918856474622*v_W1lv_b_r - 1.69609362276623*v_W1mv_a_i - 0.282682270461039*v_W1mv_a_r + 1.69609362276623*v_W1mv_b_i + 0.282682270461039*v_W1mv_b_r
        struct[0].g[3,0] = i_W1lv_b_i - 14.1918856474622*v_W1lv_b_i + 85.1513138847732*v_W1lv_b_r - 0.282682270461039*v_W1mv_a_i + 1.69609362276623*v_W1mv_a_r + 0.282682270461039*v_W1mv_b_i - 1.69609362276623*v_W1mv_b_r
        struct[0].g[4,0] = i_W1lv_c_r - 85.1513138847732*v_W1lv_c_i - 14.1918856474622*v_W1lv_c_r - 1.69609362276623*v_W1mv_b_i - 0.282682270461039*v_W1mv_b_r + 1.69609362276623*v_W1mv_c_i + 0.282682270461039*v_W1mv_c_r
        struct[0].g[5,0] = i_W1lv_c_i - 14.1918856474622*v_W1lv_c_i + 85.1513138847732*v_W1lv_c_r - 0.282682270461039*v_W1mv_b_i + 1.69609362276623*v_W1mv_b_r + 0.282682270461039*v_W1mv_c_i - 1.69609362276623*v_W1mv_c_r
        struct[0].g[6,0] = i_W2lv_a_r - 85.1513138847732*v_W2lv_a_i - 14.1918856474622*v_W2lv_a_r + 1.69609362276623*v_W2mv_a_i + 0.282682270461039*v_W2mv_a_r - 1.69609362276623*v_W2mv_c_i - 0.282682270461039*v_W2mv_c_r
        struct[0].g[7,0] = i_W2lv_a_i - 14.1918856474622*v_W2lv_a_i + 85.1513138847732*v_W2lv_a_r + 0.282682270461039*v_W2mv_a_i - 1.69609362276623*v_W2mv_a_r - 0.282682270461039*v_W2mv_c_i + 1.69609362276623*v_W2mv_c_r
        struct[0].g[8,0] = i_W2lv_b_r - 85.1513138847732*v_W2lv_b_i - 14.1918856474622*v_W2lv_b_r - 1.69609362276623*v_W2mv_a_i - 0.282682270461039*v_W2mv_a_r + 1.69609362276623*v_W2mv_b_i + 0.282682270461039*v_W2mv_b_r
        struct[0].g[9,0] = i_W2lv_b_i - 14.1918856474622*v_W2lv_b_i + 85.1513138847732*v_W2lv_b_r - 0.282682270461039*v_W2mv_a_i + 1.69609362276623*v_W2mv_a_r + 0.282682270461039*v_W2mv_b_i - 1.69609362276623*v_W2mv_b_r
        struct[0].g[10,0] = i_W2lv_c_r - 85.1513138847732*v_W2lv_c_i - 14.1918856474622*v_W2lv_c_r - 1.69609362276623*v_W2mv_b_i - 0.282682270461039*v_W2mv_b_r + 1.69609362276623*v_W2mv_c_i + 0.282682270461039*v_W2mv_c_r
        struct[0].g[11,0] = i_W2lv_c_i - 14.1918856474622*v_W2lv_c_i + 85.1513138847732*v_W2lv_c_r - 0.282682270461039*v_W2mv_b_i + 1.69609362276623*v_W2mv_b_r + 0.282682270461039*v_W2mv_c_i - 1.69609362276623*v_W2mv_c_r
        struct[0].g[12,0] = i_W3lv_a_r - 85.1513138847732*v_W3lv_a_i - 14.1918856474622*v_W3lv_a_r + 1.69609362276623*v_W3mv_a_i + 0.282682270461039*v_W3mv_a_r - 1.69609362276623*v_W3mv_c_i - 0.282682270461039*v_W3mv_c_r
        struct[0].g[13,0] = i_W3lv_a_i - 14.1918856474622*v_W3lv_a_i + 85.1513138847732*v_W3lv_a_r + 0.282682270461039*v_W3mv_a_i - 1.69609362276623*v_W3mv_a_r - 0.282682270461039*v_W3mv_c_i + 1.69609362276623*v_W3mv_c_r
        struct[0].g[14,0] = i_W3lv_b_r - 85.1513138847732*v_W3lv_b_i - 14.1918856474622*v_W3lv_b_r - 1.69609362276623*v_W3mv_a_i - 0.282682270461039*v_W3mv_a_r + 1.69609362276623*v_W3mv_b_i + 0.282682270461039*v_W3mv_b_r
        struct[0].g[15,0] = i_W3lv_b_i - 14.1918856474622*v_W3lv_b_i + 85.1513138847732*v_W3lv_b_r - 0.282682270461039*v_W3mv_a_i + 1.69609362276623*v_W3mv_a_r + 0.282682270461039*v_W3mv_b_i - 1.69609362276623*v_W3mv_b_r
        struct[0].g[16,0] = i_W3lv_c_r - 85.1513138847732*v_W3lv_c_i - 14.1918856474622*v_W3lv_c_r - 1.69609362276623*v_W3mv_b_i - 0.282682270461039*v_W3mv_b_r + 1.69609362276623*v_W3mv_c_i + 0.282682270461039*v_W3mv_c_r
        struct[0].g[17,0] = i_W3lv_c_i - 14.1918856474622*v_W3lv_c_i + 85.1513138847732*v_W3lv_c_r - 0.282682270461039*v_W3mv_b_i + 1.69609362276623*v_W3mv_b_r + 0.282682270461039*v_W3mv_c_i - 1.69609362276623*v_W3mv_c_r
        struct[0].g[18,0] = i_STlv_a_r - 85.1513138847732*v_STlv_a_i - 14.1918856474622*v_STlv_a_r + 1.69609362276623*v_STmv_a_i + 0.282682270461039*v_STmv_a_r - 1.69609362276623*v_STmv_c_i - 0.282682270461039*v_STmv_c_r
        struct[0].g[19,0] = i_STlv_a_i - 14.1918856474622*v_STlv_a_i + 85.1513138847732*v_STlv_a_r + 0.282682270461039*v_STmv_a_i - 1.69609362276623*v_STmv_a_r - 0.282682270461039*v_STmv_c_i + 1.69609362276623*v_STmv_c_r
        struct[0].g[20,0] = i_STlv_b_r - 85.1513138847732*v_STlv_b_i - 14.1918856474622*v_STlv_b_r - 1.69609362276623*v_STmv_a_i - 0.282682270461039*v_STmv_a_r + 1.69609362276623*v_STmv_b_i + 0.282682270461039*v_STmv_b_r
        struct[0].g[21,0] = i_STlv_b_i - 14.1918856474622*v_STlv_b_i + 85.1513138847732*v_STlv_b_r - 0.282682270461039*v_STmv_a_i + 1.69609362276623*v_STmv_a_r + 0.282682270461039*v_STmv_b_i - 1.69609362276623*v_STmv_b_r
        struct[0].g[22,0] = i_STlv_c_r - 85.1513138847732*v_STlv_c_i - 14.1918856474622*v_STlv_c_r - 1.69609362276623*v_STmv_b_i - 0.282682270461039*v_STmv_b_r + 1.69609362276623*v_STmv_c_i + 0.282682270461039*v_STmv_c_r
        struct[0].g[23,0] = i_STlv_c_i - 14.1918856474622*v_STlv_c_i + 85.1513138847732*v_STlv_c_r - 0.282682270461039*v_STmv_b_i + 1.69609362276623*v_STmv_b_r + 0.282682270461039*v_STmv_c_i - 1.69609362276623*v_STmv_c_r
        struct[0].g[24,0] = i_POI_a_r + 0.040290088638195*v_GRID_a_i + 0.024174053182917*v_GRID_a_r + 4.66248501556824e-18*v_GRID_b_i - 4.31760362252812e-18*v_GRID_b_r + 4.19816664496737e-18*v_GRID_c_i - 3.49608108880335e-18*v_GRID_c_r - 0.0591264711109411*v_POI_a_i - 0.0265286009920103*v_POI_a_r + 0.00941819123637305*v_POI_b_i + 0.00117727390454664*v_POI_b_r + 0.00941819123637305*v_POI_c_i + 0.00117727390454664*v_POI_c_r + 0.0538321929314336*v_POImv_a_i + 0.0067290241164292*v_POImv_a_r - 0.0538321929314336*v_POImv_b_i - 0.0067290241164292*v_POImv_b_r
        struct[0].g[25,0] = i_POI_a_i + 0.024174053182917*v_GRID_a_i - 0.040290088638195*v_GRID_a_r - 4.31760362252812e-18*v_GRID_b_i - 4.66248501556824e-18*v_GRID_b_r - 3.49608108880335e-18*v_GRID_c_i - 4.19816664496737e-18*v_GRID_c_r - 0.0265286009920103*v_POI_a_i + 0.0591264711109411*v_POI_a_r + 0.00117727390454664*v_POI_b_i - 0.00941819123637305*v_POI_b_r + 0.00117727390454664*v_POI_c_i - 0.00941819123637305*v_POI_c_r + 0.0067290241164292*v_POImv_a_i - 0.0538321929314336*v_POImv_a_r - 0.0067290241164292*v_POImv_b_i + 0.0538321929314336*v_POImv_b_r
        struct[0].g[26,0] = i_POI_b_r + 6.30775359573304e-19*v_GRID_a_i - 2.07254761002657e-18*v_GRID_a_r + 0.040290088638195*v_GRID_b_i + 0.024174053182917*v_GRID_b_r + 9.01107656533306e-19*v_GRID_c_i - 1.78419315993592e-17*v_GRID_c_r + 0.00941819123637305*v_POI_a_i + 0.00117727390454663*v_POI_a_r - 0.0591264711109411*v_POI_b_i - 0.0265286009920103*v_POI_b_r + 0.00941819123637305*v_POI_c_i + 0.00117727390454665*v_POI_c_r + 0.0538321929314336*v_POImv_b_i + 0.0067290241164292*v_POImv_b_r - 0.0538321929314336*v_POImv_c_i - 0.0067290241164292*v_POImv_c_r
        struct[0].g[27,0] = i_POI_b_i - 2.07254761002657e-18*v_GRID_a_i - 6.30775359573304e-19*v_GRID_a_r + 0.024174053182917*v_GRID_b_i - 0.040290088638195*v_GRID_b_r - 1.78419315993592e-17*v_GRID_c_i - 9.01107656533306e-19*v_GRID_c_r + 0.00117727390454663*v_POI_a_i - 0.00941819123637305*v_POI_a_r - 0.0265286009920103*v_POI_b_i + 0.0591264711109411*v_POI_b_r + 0.00117727390454665*v_POI_c_i - 0.00941819123637305*v_POI_c_r + 0.0067290241164292*v_POImv_b_i - 0.0538321929314336*v_POImv_b_r - 0.0067290241164292*v_POImv_c_i + 0.0538321929314336*v_POImv_c_r
        struct[0].g[28,0] = i_POI_c_r - 7.20886125226632e-19*v_GRID_a_i - 1.35166148479994e-18*v_GRID_a_r - 4.50553828266631e-19*v_GRID_b_i - 1.71210454741325e-17*v_GRID_b_r + 0.040290088638195*v_GRID_c_i + 0.024174053182917*v_GRID_c_r + 0.00941819123637305*v_POI_a_i + 0.00117727390454663*v_POI_a_r + 0.00941819123637305*v_POI_b_i + 0.00117727390454665*v_POI_b_r - 0.0591264711109411*v_POI_c_i - 0.0265286009920103*v_POI_c_r - 0.0538321929314336*v_POImv_a_i - 0.0067290241164292*v_POImv_a_r + 0.0538321929314336*v_POImv_c_i + 0.0067290241164292*v_POImv_c_r
        struct[0].g[29,0] = i_POI_c_i - 1.35166148479994e-18*v_GRID_a_i + 7.20886125226632e-19*v_GRID_a_r - 1.71210454741325e-17*v_GRID_b_i + 4.50553828266631e-19*v_GRID_b_r + 0.024174053182917*v_GRID_c_i - 0.040290088638195*v_GRID_c_r + 0.00117727390454663*v_POI_a_i - 0.00941819123637305*v_POI_a_r + 0.00117727390454665*v_POI_b_i - 0.00941819123637305*v_POI_b_r - 0.0265286009920103*v_POI_c_i + 0.0591264711109411*v_POI_c_r - 0.0067290241164292*v_POImv_a_i + 0.0538321929314336*v_POImv_a_r + 0.0067290241164292*v_POImv_c_i - 0.0538321929314336*v_POImv_c_r
        struct[0].g[30,0] = i_POImv_a_r + 0.0538321929314336*v_POI_a_i + 0.0067290241164292*v_POI_a_r - 0.0538321929314336*v_POI_c_i - 0.0067290241164292*v_POI_c_r - 155.244588874881*v_POImv_a_i - 188.924390492986*v_POImv_a_r + 44.2677164725443*v_POImv_b_i + 53.9540151298641*v_POImv_b_r + 44.2677164725443*v_POImv_c_i + 53.9540151298641*v_POImv_c_r + 148.977829666654*v_STmv_a_i + 181.611068546017*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274334*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274334*v_STmv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[31,0] = i_POImv_a_i + 0.0067290241164292*v_POI_a_i - 0.0538321929314336*v_POI_a_r - 0.0067290241164292*v_POI_c_i + 0.0538321929314336*v_POI_c_r - 188.924390492986*v_POImv_a_i + 155.244588874881*v_POImv_a_r + 53.9540151298641*v_POImv_b_i - 44.2677164725443*v_POImv_b_r + 53.9540151298641*v_POImv_c_i - 44.2677164725443*v_POImv_c_r + 181.611068546017*v_STmv_a_i - 148.977829666654*v_STmv_a_r - 51.8888767274334*v_STmv_b_i + 42.5650941904727*v_STmv_b_r - 51.8888767274334*v_STmv_c_i + 42.5650941904727*v_STmv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[32,0] = i_POImv_b_r - 0.0538321929314336*v_POI_a_i - 0.0067290241164292*v_POI_a_r + 0.0538321929314336*v_POI_b_i + 0.0067290241164292*v_POI_b_r + 44.2677164725443*v_POImv_a_i + 53.9540151298641*v_POImv_a_r - 155.244588874881*v_POImv_b_i - 188.924390492986*v_POImv_b_r + 44.2677164725443*v_POImv_c_i + 53.9540151298642*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r + 148.977829666654*v_STmv_b_i + 181.611068546017*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274335*v_STmv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[33,0] = i_POImv_b_i - 0.0067290241164292*v_POI_a_i + 0.0538321929314336*v_POI_a_r + 0.0067290241164292*v_POI_b_i - 0.0538321929314336*v_POI_b_r + 53.9540151298641*v_POImv_a_i - 44.2677164725443*v_POImv_a_r - 188.924390492986*v_POImv_b_i + 155.244588874881*v_POImv_b_r + 53.9540151298642*v_POImv_c_i - 44.2677164725443*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r + 181.611068546017*v_STmv_b_i - 148.977829666654*v_STmv_b_r - 51.8888767274335*v_STmv_c_i + 42.5650941904727*v_STmv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[34,0] = i_POImv_c_r - 0.0538321929314336*v_POI_b_i - 0.0067290241164292*v_POI_b_r + 0.0538321929314336*v_POI_c_i + 0.0067290241164292*v_POI_c_r + 44.2677164725443*v_POImv_a_i + 53.9540151298641*v_POImv_a_r + 44.2677164725443*v_POImv_b_i + 53.9540151298642*v_POImv_b_r - 155.244588874881*v_POImv_c_i - 188.924390492986*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274335*v_STmv_b_r + 148.977829666654*v_STmv_c_i + 181.611068546017*v_STmv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[35,0] = i_POImv_c_i - 0.0067290241164292*v_POI_b_i + 0.0538321929314336*v_POI_b_r + 0.0067290241164292*v_POI_c_i - 0.0538321929314336*v_POI_c_r + 53.9540151298641*v_POImv_a_i - 44.2677164725443*v_POImv_a_r + 53.9540151298642*v_POImv_b_i - 44.2677164725443*v_POImv_b_r - 188.924390492986*v_POImv_c_i + 155.244588874881*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r - 51.8888767274335*v_STmv_b_i + 42.5650941904727*v_STmv_b_r + 181.611068546017*v_STmv_c_i - 148.977829666654*v_STmv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[36,0] = i_W1mv_a_r + 1.69609362276623*v_W1lv_a_i + 0.282682270461039*v_W1lv_a_r - 1.69609362276623*v_W1lv_b_i - 0.282682270461039*v_W1lv_b_r - 6.02663624833782*v_W1mv_a_i - 7.27570400310194*v_W1mv_a_r + 1.73640535376106*v_W1mv_b_i + 2.08118569972797*v_W1mv_b_r + 1.73640535376106*v_W1mv_c_i + 2.08118569972797*v_W1mv_c_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r
        struct[0].g[37,0] = i_W1mv_a_i + 0.282682270461039*v_W1lv_a_i - 1.69609362276623*v_W1lv_a_r - 0.282682270461039*v_W1lv_b_i + 1.69609362276623*v_W1lv_b_r - 7.27570400310194*v_W1mv_a_i + 6.02663624833782*v_W1mv_a_r + 2.08118569972797*v_W1mv_b_i - 1.73640535376106*v_W1mv_b_r + 2.08118569972797*v_W1mv_c_i - 1.73640535376106*v_W1mv_c_r + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r
        struct[0].g[38,0] = i_W1mv_b_r + 1.69609362276623*v_W1lv_b_i + 0.282682270461039*v_W1lv_b_r - 1.69609362276623*v_W1lv_c_i - 0.282682270461039*v_W1lv_c_r + 1.73640535376106*v_W1mv_a_i + 2.08118569972797*v_W1mv_a_r - 6.02663624833782*v_W1mv_b_i - 7.27570400310194*v_W1mv_b_r + 1.73640535376106*v_W1mv_c_i + 2.08118569972797*v_W1mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r
        struct[0].g[39,0] = i_W1mv_b_i + 0.282682270461039*v_W1lv_b_i - 1.69609362276623*v_W1lv_b_r - 0.282682270461039*v_W1lv_c_i + 1.69609362276623*v_W1lv_c_r + 2.08118569972797*v_W1mv_a_i - 1.73640535376106*v_W1mv_a_r - 7.27570400310194*v_W1mv_b_i + 6.02663624833782*v_W1mv_b_r + 2.08118569972797*v_W1mv_c_i - 1.73640535376106*v_W1mv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r
        struct[0].g[40,0] = i_W1mv_c_r - 1.69609362276623*v_W1lv_a_i - 0.282682270461039*v_W1lv_a_r + 1.69609362276623*v_W1lv_c_i + 0.282682270461039*v_W1lv_c_r + 1.73640535376106*v_W1mv_a_i + 2.08118569972797*v_W1mv_a_r + 1.73640535376106*v_W1mv_b_i + 2.08118569972797*v_W1mv_b_r - 6.02663624833782*v_W1mv_c_i - 7.27570400310194*v_W1mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r
        struct[0].g[41,0] = i_W1mv_c_i - 0.282682270461039*v_W1lv_a_i + 1.69609362276623*v_W1lv_a_r + 0.282682270461039*v_W1lv_c_i - 1.69609362276623*v_W1lv_c_r + 2.08118569972797*v_W1mv_a_i - 1.73640535376106*v_W1mv_a_r + 2.08118569972797*v_W1mv_b_i - 1.73640535376106*v_W1mv_b_r - 7.27570400310194*v_W1mv_c_i + 6.02663624833782*v_W1mv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r
        struct[0].g[42,0] = i_W2mv_a_r + 5.95911318666618*v_W1mv_a_i + 7.26444274184068*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.69609362276623*v_W2lv_a_i + 0.282682270461039*v_W2lv_a_r - 1.69609362276623*v_W2lv_b_i - 0.282682270461039*v_W2lv_b_r - 11.9857049291081*v_W2mv_a_i - 14.5401467449426*v_W2mv_a_r + 3.43902692373834*v_W2mv_b_i + 4.1567407688253*v_W2mv_b_r + 3.43902692373834*v_W2mv_c_i + 4.1567407688253*v_W2mv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[43,0] = i_W2mv_a_i + 7.26444274184068*v_W1mv_a_i - 5.95911318666618*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 0.282682270461039*v_W2lv_a_i - 1.69609362276623*v_W2lv_a_r - 0.282682270461039*v_W2lv_b_i + 1.69609362276623*v_W2lv_b_r - 14.5401467449426*v_W2mv_a_i + 11.9857049291081*v_W2mv_a_r + 4.1567407688253*v_W2mv_b_i - 3.43902692373834*v_W2mv_b_r + 4.1567407688253*v_W2mv_c_i - 3.43902692373834*v_W2mv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[44,0] = i_W2mv_b_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r + 5.95911318666618*v_W1mv_b_i + 7.26444274184068*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.69609362276623*v_W2lv_b_i + 0.282682270461039*v_W2lv_b_r - 1.69609362276623*v_W2lv_c_i - 0.282682270461039*v_W2lv_c_r + 3.43902692373834*v_W2mv_a_i + 4.1567407688253*v_W2mv_a_r - 11.9857049291081*v_W2mv_b_i - 14.5401467449426*v_W2mv_b_r + 3.43902692373834*v_W2mv_c_i + 4.15674076882531*v_W2mv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[45,0] = i_W2mv_b_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r + 7.26444274184068*v_W1mv_b_i - 5.95911318666618*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 0.282682270461039*v_W2lv_b_i - 1.69609362276623*v_W2lv_b_r - 0.282682270461039*v_W2lv_c_i + 1.69609362276623*v_W2lv_c_r + 4.1567407688253*v_W2mv_a_i - 3.43902692373834*v_W2mv_a_r - 14.5401467449426*v_W2mv_b_i + 11.9857049291081*v_W2mv_b_r + 4.15674076882531*v_W2mv_c_i - 3.43902692373834*v_W2mv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[46,0] = i_W2mv_c_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r + 5.95911318666618*v_W1mv_c_i + 7.26444274184068*v_W1mv_c_r - 1.69609362276623*v_W2lv_a_i - 0.282682270461039*v_W2lv_a_r + 1.69609362276623*v_W2lv_c_i + 0.282682270461039*v_W2lv_c_r + 3.43902692373834*v_W2mv_a_i + 4.1567407688253*v_W2mv_a_r + 3.43902692373834*v_W2mv_b_i + 4.15674076882531*v_W2mv_b_r - 11.9857049291081*v_W2mv_c_i - 14.5401467449426*v_W2mv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[47,0] = i_W2mv_c_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r + 7.26444274184068*v_W1mv_c_i - 5.95911318666618*v_W1mv_c_r - 0.282682270461039*v_W2lv_a_i + 1.69609362276623*v_W2lv_a_r + 0.282682270461039*v_W2lv_c_i - 1.69609362276623*v_W2lv_c_r + 4.1567407688253*v_W2mv_a_i - 3.43902692373834*v_W2mv_a_r + 4.15674076882531*v_W2mv_b_i - 3.43902692373834*v_W2mv_b_r - 14.5401467449426*v_W2mv_c_i + 11.9857049291081*v_W2mv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[48,0] = i_W3mv_a_r + 5.95911318666618*v_POImv_a_i + 7.26444274184068*v_POImv_a_r - 1.70260376761891*v_POImv_b_i - 2.07555506909734*v_POImv_b_r - 1.70260376761891*v_POImv_c_i - 2.07555506909734*v_POImv_c_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.69609362276623*v_W3lv_a_i + 0.282682270461039*v_W3lv_a_r - 1.69609362276623*v_W3lv_b_i - 0.282682270461039*v_W3lv_b_r - 11.9857049291081*v_W3mv_a_i - 14.5401467449426*v_W3mv_a_r + 3.43902692373834*v_W3mv_b_i + 4.1567407688253*v_W3mv_b_r + 3.43902692373834*v_W3mv_c_i + 4.1567407688253*v_W3mv_c_r
        struct[0].g[49,0] = i_W3mv_a_i + 7.26444274184068*v_POImv_a_i - 5.95911318666618*v_POImv_a_r - 2.07555506909734*v_POImv_b_i + 1.70260376761891*v_POImv_b_r - 2.07555506909734*v_POImv_c_i + 1.70260376761891*v_POImv_c_r + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 0.282682270461039*v_W3lv_a_i - 1.69609362276623*v_W3lv_a_r - 0.282682270461039*v_W3lv_b_i + 1.69609362276623*v_W3lv_b_r - 14.5401467449426*v_W3mv_a_i + 11.9857049291081*v_W3mv_a_r + 4.1567407688253*v_W3mv_b_i - 3.43902692373834*v_W3mv_b_r + 4.1567407688253*v_W3mv_c_i - 3.43902692373834*v_W3mv_c_r
        struct[0].g[50,0] = i_W3mv_b_r - 1.70260376761891*v_POImv_a_i - 2.07555506909734*v_POImv_a_r + 5.95911318666618*v_POImv_b_i + 7.26444274184068*v_POImv_b_r - 1.70260376761891*v_POImv_c_i - 2.07555506909734*v_POImv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.69609362276623*v_W3lv_b_i + 0.282682270461039*v_W3lv_b_r - 1.69609362276623*v_W3lv_c_i - 0.282682270461039*v_W3lv_c_r + 3.43902692373834*v_W3mv_a_i + 4.1567407688253*v_W3mv_a_r - 11.9857049291081*v_W3mv_b_i - 14.5401467449426*v_W3mv_b_r + 3.43902692373834*v_W3mv_c_i + 4.15674076882531*v_W3mv_c_r
        struct[0].g[51,0] = i_W3mv_b_i - 2.07555506909734*v_POImv_a_i + 1.70260376761891*v_POImv_a_r + 7.26444274184068*v_POImv_b_i - 5.95911318666618*v_POImv_b_r - 2.07555506909734*v_POImv_c_i + 1.70260376761891*v_POImv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 0.282682270461039*v_W3lv_b_i - 1.69609362276623*v_W3lv_b_r - 0.282682270461039*v_W3lv_c_i + 1.69609362276623*v_W3lv_c_r + 4.1567407688253*v_W3mv_a_i - 3.43902692373834*v_W3mv_a_r - 14.5401467449426*v_W3mv_b_i + 11.9857049291081*v_W3mv_b_r + 4.15674076882531*v_W3mv_c_i - 3.43902692373834*v_W3mv_c_r
        struct[0].g[52,0] = i_W3mv_c_r - 1.70260376761891*v_POImv_a_i - 2.07555506909734*v_POImv_a_r - 1.70260376761891*v_POImv_b_i - 2.07555506909734*v_POImv_b_r + 5.95911318666618*v_POImv_c_i + 7.26444274184068*v_POImv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r - 1.69609362276623*v_W3lv_a_i - 0.282682270461039*v_W3lv_a_r + 1.69609362276623*v_W3lv_c_i + 0.282682270461039*v_W3lv_c_r + 3.43902692373834*v_W3mv_a_i + 4.1567407688253*v_W3mv_a_r + 3.43902692373834*v_W3mv_b_i + 4.15674076882531*v_W3mv_b_r - 11.9857049291081*v_W3mv_c_i - 14.5401467449426*v_W3mv_c_r
        struct[0].g[53,0] = i_W3mv_c_i - 2.07555506909734*v_POImv_a_i + 1.70260376761891*v_POImv_a_r - 2.07555506909734*v_POImv_b_i + 1.70260376761891*v_POImv_b_r + 7.26444274184068*v_POImv_c_i - 5.95911318666618*v_POImv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r - 0.282682270461039*v_W3lv_a_i + 1.69609362276623*v_W3lv_a_r + 0.282682270461039*v_W3lv_c_i - 1.69609362276623*v_W3lv_c_r + 4.1567407688253*v_W3mv_a_i - 3.43902692373834*v_W3mv_a_r + 4.15674076882531*v_W3mv_b_i - 3.43902692373834*v_W3mv_b_r - 14.5401467449426*v_W3mv_c_i + 11.9857049291081*v_W3mv_c_r
        struct[0].g[54,0] = i_STmv_a_r + 148.977829666654*v_POImv_a_i + 181.611068546017*v_POImv_a_r - 42.5650941904727*v_POImv_b_i - 51.8888767274334*v_POImv_b_r - 42.5650941904727*v_POImv_c_i - 51.8888767274334*v_POImv_c_r + 1.69609362276623*v_STlv_a_i + 0.282682270461039*v_STlv_a_r - 1.69609362276623*v_STlv_b_i - 0.282682270461039*v_STlv_b_r - 149.045395453986*v_STmv_a_i - 181.622329807278*v_STmv_a_r + 42.5988786863508*v_STmv_b_i + 51.8945073580641*v_STmv_b_r + 42.5988786863508*v_STmv_c_i + 51.894507358064*v_STmv_c_r
        struct[0].g[55,0] = i_STmv_a_i + 181.611068546017*v_POImv_a_i - 148.977829666654*v_POImv_a_r - 51.8888767274334*v_POImv_b_i + 42.5650941904727*v_POImv_b_r - 51.8888767274334*v_POImv_c_i + 42.5650941904727*v_POImv_c_r + 0.282682270461039*v_STlv_a_i - 1.69609362276623*v_STlv_a_r - 0.282682270461039*v_STlv_b_i + 1.69609362276623*v_STlv_b_r - 181.622329807278*v_STmv_a_i + 149.045395453986*v_STmv_a_r + 51.8945073580641*v_STmv_b_i - 42.5988786863508*v_STmv_b_r + 51.894507358064*v_STmv_c_i - 42.5988786863508*v_STmv_c_r
        struct[0].g[56,0] = i_STmv_b_r - 42.5650941904727*v_POImv_a_i - 51.8888767274334*v_POImv_a_r + 148.977829666654*v_POImv_b_i + 181.611068546017*v_POImv_b_r - 42.5650941904727*v_POImv_c_i - 51.8888767274335*v_POImv_c_r + 1.69609362276623*v_STlv_b_i + 0.282682270461039*v_STlv_b_r - 1.69609362276623*v_STlv_c_i - 0.282682270461039*v_STlv_c_r + 42.5988786863508*v_STmv_a_i + 51.8945073580641*v_STmv_a_r - 149.045395453986*v_STmv_b_i - 181.622329807278*v_STmv_b_r + 42.5988786863508*v_STmv_c_i + 51.8945073580641*v_STmv_c_r
        struct[0].g[57,0] = i_STmv_b_i - 51.8888767274334*v_POImv_a_i + 42.5650941904727*v_POImv_a_r + 181.611068546017*v_POImv_b_i - 148.977829666654*v_POImv_b_r - 51.8888767274335*v_POImv_c_i + 42.5650941904727*v_POImv_c_r + 0.282682270461039*v_STlv_b_i - 1.69609362276623*v_STlv_b_r - 0.282682270461039*v_STlv_c_i + 1.69609362276623*v_STlv_c_r + 51.8945073580641*v_STmv_a_i - 42.5988786863508*v_STmv_a_r - 181.622329807278*v_STmv_b_i + 149.045395453986*v_STmv_b_r + 51.8945073580641*v_STmv_c_i - 42.5988786863508*v_STmv_c_r
        struct[0].g[58,0] = i_STmv_c_r - 42.5650941904727*v_POImv_a_i - 51.8888767274334*v_POImv_a_r - 42.5650941904727*v_POImv_b_i - 51.8888767274335*v_POImv_b_r + 148.977829666654*v_POImv_c_i + 181.611068546017*v_POImv_c_r - 1.69609362276623*v_STlv_a_i - 0.282682270461039*v_STlv_a_r + 1.69609362276623*v_STlv_c_i + 0.282682270461039*v_STlv_c_r + 42.5988786863508*v_STmv_a_i + 51.8945073580641*v_STmv_a_r + 42.5988786863508*v_STmv_b_i + 51.8945073580641*v_STmv_b_r - 149.045395453986*v_STmv_c_i - 181.622329807278*v_STmv_c_r
        struct[0].g[59,0] = i_STmv_c_i - 51.8888767274334*v_POImv_a_i + 42.5650941904727*v_POImv_a_r - 51.8888767274335*v_POImv_b_i + 42.5650941904727*v_POImv_b_r + 181.611068546017*v_POImv_c_i - 148.977829666654*v_POImv_c_r - 0.282682270461039*v_STlv_a_i + 1.69609362276623*v_STlv_a_r + 0.282682270461039*v_STlv_c_i - 1.69609362276623*v_STlv_c_r + 51.8945073580641*v_STmv_a_i - 42.5988786863508*v_STmv_a_r + 51.8945073580641*v_STmv_b_i - 42.5988786863508*v_STmv_b_r - 181.622329807278*v_STmv_c_i + 149.045395453986*v_STmv_c_r
        struct[0].g[60,0] = -i_l_W1mv_W2mv_a_r + 5.95911318666618*v_W1mv_a_i + 7.26444274184068*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r - 5.95911318666618*v_W2mv_a_i - 7.26444274184068*v_W2mv_a_r + 1.70260376761891*v_W2mv_b_i + 2.07555506909734*v_W2mv_b_r + 1.70260376761891*v_W2mv_c_i + 2.07555506909734*v_W2mv_c_r
        struct[0].g[61,0] = -i_l_W1mv_W2mv_a_i + 7.26444274184068*v_W1mv_a_i - 5.95911318666618*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r - 7.26444274184068*v_W2mv_a_i + 5.95911318666618*v_W2mv_a_r + 2.07555506909734*v_W2mv_b_i - 1.70260376761891*v_W2mv_b_r + 2.07555506909734*v_W2mv_c_i - 1.70260376761891*v_W2mv_c_r
        struct[0].g[62,0] = -i_l_W1mv_W2mv_b_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r + 5.95911318666618*v_W1mv_b_i + 7.26444274184068*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.70260376761891*v_W2mv_a_i + 2.07555506909734*v_W2mv_a_r - 5.95911318666618*v_W2mv_b_i - 7.26444274184068*v_W2mv_b_r + 1.70260376761891*v_W2mv_c_i + 2.07555506909734*v_W2mv_c_r
        struct[0].g[63,0] = -i_l_W1mv_W2mv_b_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r + 7.26444274184068*v_W1mv_b_i - 5.95911318666618*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 2.07555506909734*v_W2mv_a_i - 1.70260376761891*v_W2mv_a_r - 7.26444274184068*v_W2mv_b_i + 5.95911318666618*v_W2mv_b_r + 2.07555506909734*v_W2mv_c_i - 1.70260376761891*v_W2mv_c_r
        struct[0].g[64,0] = -i_l_W1mv_W2mv_c_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r + 5.95911318666618*v_W1mv_c_i + 7.26444274184068*v_W1mv_c_r + 1.70260376761891*v_W2mv_a_i + 2.07555506909734*v_W2mv_a_r + 1.70260376761891*v_W2mv_b_i + 2.07555506909734*v_W2mv_b_r - 5.95911318666618*v_W2mv_c_i - 7.26444274184068*v_W2mv_c_r
        struct[0].g[65,0] = -i_l_W1mv_W2mv_c_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r + 7.26444274184068*v_W1mv_c_i - 5.95911318666618*v_W1mv_c_r + 2.07555506909734*v_W2mv_a_i - 1.70260376761891*v_W2mv_a_r + 2.07555506909734*v_W2mv_b_i - 1.70260376761891*v_W2mv_b_r - 7.26444274184068*v_W2mv_c_i + 5.95911318666618*v_W2mv_c_r
        struct[0].g[66,0] = -i_l_W2mv_W3mv_a_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r - 5.95911318666618*v_W3mv_a_i - 7.26444274184068*v_W3mv_a_r + 1.70260376761891*v_W3mv_b_i + 2.07555506909734*v_W3mv_b_r + 1.70260376761891*v_W3mv_c_i + 2.07555506909734*v_W3mv_c_r
        struct[0].g[67,0] = -i_l_W2mv_W3mv_a_i + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r - 7.26444274184068*v_W3mv_a_i + 5.95911318666618*v_W3mv_a_r + 2.07555506909734*v_W3mv_b_i - 1.70260376761891*v_W3mv_b_r + 2.07555506909734*v_W3mv_c_i - 1.70260376761891*v_W3mv_c_r
        struct[0].g[68,0] = -i_l_W2mv_W3mv_b_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.70260376761891*v_W3mv_a_i + 2.07555506909734*v_W3mv_a_r - 5.95911318666618*v_W3mv_b_i - 7.26444274184068*v_W3mv_b_r + 1.70260376761891*v_W3mv_c_i + 2.07555506909734*v_W3mv_c_r
        struct[0].g[69,0] = -i_l_W2mv_W3mv_b_i - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 2.07555506909734*v_W3mv_a_i - 1.70260376761891*v_W3mv_a_r - 7.26444274184068*v_W3mv_b_i + 5.95911318666618*v_W3mv_b_r + 2.07555506909734*v_W3mv_c_i - 1.70260376761891*v_W3mv_c_r
        struct[0].g[70,0] = -i_l_W2mv_W3mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r + 1.70260376761891*v_W3mv_a_i + 2.07555506909734*v_W3mv_a_r + 1.70260376761891*v_W3mv_b_i + 2.07555506909734*v_W3mv_b_r - 5.95911318666618*v_W3mv_c_i - 7.26444274184068*v_W3mv_c_r
        struct[0].g[71,0] = -i_l_W2mv_W3mv_c_i - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r + 2.07555506909734*v_W3mv_a_i - 1.70260376761891*v_W3mv_a_r + 2.07555506909734*v_W3mv_b_i - 1.70260376761891*v_W3mv_b_r - 7.26444274184068*v_W3mv_c_i + 5.95911318666618*v_W3mv_c_r
        struct[0].g[72,0] = -i_l_W3mv_POImv_a_r - 5.95911318666618*v_POImv_a_i - 7.26444274184068*v_POImv_a_r + 1.70260376761891*v_POImv_b_i + 2.07555506909734*v_POImv_b_r + 1.70260376761891*v_POImv_c_i + 2.07555506909734*v_POImv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[73,0] = -i_l_W3mv_POImv_a_i - 7.26444274184068*v_POImv_a_i + 5.95911318666618*v_POImv_a_r + 2.07555506909734*v_POImv_b_i - 1.70260376761891*v_POImv_b_r + 2.07555506909734*v_POImv_c_i - 1.70260376761891*v_POImv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[74,0] = -i_l_W3mv_POImv_b_r + 1.70260376761891*v_POImv_a_i + 2.07555506909734*v_POImv_a_r - 5.95911318666618*v_POImv_b_i - 7.26444274184068*v_POImv_b_r + 1.70260376761891*v_POImv_c_i + 2.07555506909734*v_POImv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[75,0] = -i_l_W3mv_POImv_b_i + 2.07555506909734*v_POImv_a_i - 1.70260376761891*v_POImv_a_r - 7.26444274184068*v_POImv_b_i + 5.95911318666618*v_POImv_b_r + 2.07555506909734*v_POImv_c_i - 1.70260376761891*v_POImv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[76,0] = -i_l_W3mv_POImv_c_r + 1.70260376761891*v_POImv_a_i + 2.07555506909734*v_POImv_a_r + 1.70260376761891*v_POImv_b_i + 2.07555506909734*v_POImv_b_r - 5.95911318666618*v_POImv_c_i - 7.26444274184068*v_POImv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[77,0] = -i_l_W3mv_POImv_c_i + 2.07555506909734*v_POImv_a_i - 1.70260376761891*v_POImv_a_r + 2.07555506909734*v_POImv_b_i - 1.70260376761891*v_POImv_b_r - 7.26444274184068*v_POImv_c_i + 5.95911318666618*v_POImv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[78,0] = -i_l_STmv_POImv_a_r - 148.977829666654*v_POImv_a_i - 181.611068546017*v_POImv_a_r + 42.5650941904727*v_POImv_b_i + 51.8888767274334*v_POImv_b_r + 42.5650941904727*v_POImv_c_i + 51.8888767274334*v_POImv_c_r + 148.977829666654*v_STmv_a_i + 181.611068546017*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274334*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274334*v_STmv_c_r
        struct[0].g[79,0] = -i_l_STmv_POImv_a_i - 181.611068546017*v_POImv_a_i + 148.977829666654*v_POImv_a_r + 51.8888767274334*v_POImv_b_i - 42.5650941904727*v_POImv_b_r + 51.8888767274334*v_POImv_c_i - 42.5650941904727*v_POImv_c_r + 181.611068546017*v_STmv_a_i - 148.977829666654*v_STmv_a_r - 51.8888767274334*v_STmv_b_i + 42.5650941904727*v_STmv_b_r - 51.8888767274334*v_STmv_c_i + 42.5650941904727*v_STmv_c_r
        struct[0].g[80,0] = -i_l_STmv_POImv_b_r + 42.5650941904727*v_POImv_a_i + 51.8888767274334*v_POImv_a_r - 148.977829666654*v_POImv_b_i - 181.611068546017*v_POImv_b_r + 42.5650941904727*v_POImv_c_i + 51.8888767274335*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r + 148.977829666654*v_STmv_b_i + 181.611068546017*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274335*v_STmv_c_r
        struct[0].g[81,0] = -i_l_STmv_POImv_b_i + 51.8888767274334*v_POImv_a_i - 42.5650941904727*v_POImv_a_r - 181.611068546017*v_POImv_b_i + 148.977829666654*v_POImv_b_r + 51.8888767274335*v_POImv_c_i - 42.5650941904727*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r + 181.611068546017*v_STmv_b_i - 148.977829666654*v_STmv_b_r - 51.8888767274335*v_STmv_c_i + 42.5650941904727*v_STmv_c_r
        struct[0].g[82,0] = -i_l_STmv_POImv_c_r + 42.5650941904727*v_POImv_a_i + 51.8888767274334*v_POImv_a_r + 42.5650941904727*v_POImv_b_i + 51.8888767274335*v_POImv_b_r - 148.977829666654*v_POImv_c_i - 181.611068546017*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274335*v_STmv_b_r + 148.977829666654*v_STmv_c_i + 181.611068546017*v_STmv_c_r
        struct[0].g[83,0] = -i_l_STmv_POImv_c_i + 51.8888767274334*v_POImv_a_i - 42.5650941904727*v_POImv_a_r + 51.8888767274335*v_POImv_b_i - 42.5650941904727*v_POImv_b_r - 181.611068546017*v_POImv_c_i + 148.977829666654*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r - 51.8888767274335*v_STmv_b_i + 42.5650941904727*v_STmv_b_r + 181.611068546017*v_STmv_c_i - 148.977829666654*v_STmv_c_r
        struct[0].g[84,0] = -i_l_POI_GRID_a_r - 0.040290088638195*v_GRID_a_i - 0.024174053182917*v_GRID_a_r - 4.66248501556824e-18*v_GRID_b_i + 4.31760362252812e-18*v_GRID_b_r - 4.19816664496737e-18*v_GRID_c_i + 3.49608108880335e-18*v_GRID_c_r + 0.040290088638195*v_POI_a_i + 0.024174053182917*v_POI_a_r + 4.66248501556824e-18*v_POI_b_i - 4.31760362252812e-18*v_POI_b_r + 4.19816664496737e-18*v_POI_c_i - 3.49608108880335e-18*v_POI_c_r
        struct[0].g[85,0] = -i_l_POI_GRID_a_i - 0.024174053182917*v_GRID_a_i + 0.040290088638195*v_GRID_a_r + 4.31760362252812e-18*v_GRID_b_i + 4.66248501556824e-18*v_GRID_b_r + 3.49608108880335e-18*v_GRID_c_i + 4.19816664496737e-18*v_GRID_c_r + 0.024174053182917*v_POI_a_i - 0.040290088638195*v_POI_a_r - 4.31760362252812e-18*v_POI_b_i - 4.66248501556824e-18*v_POI_b_r - 3.49608108880335e-18*v_POI_c_i - 4.19816664496737e-18*v_POI_c_r
        struct[0].g[86,0] = -i_l_POI_GRID_b_r - 6.30775359573304e-19*v_GRID_a_i + 2.07254761002657e-18*v_GRID_a_r - 0.040290088638195*v_GRID_b_i - 0.024174053182917*v_GRID_b_r - 9.01107656533306e-19*v_GRID_c_i + 1.78419315993592e-17*v_GRID_c_r + 6.30775359573304e-19*v_POI_a_i - 2.07254761002657e-18*v_POI_a_r + 0.040290088638195*v_POI_b_i + 0.024174053182917*v_POI_b_r + 9.01107656533306e-19*v_POI_c_i - 1.78419315993592e-17*v_POI_c_r
        struct[0].g[87,0] = -i_l_POI_GRID_b_i + 2.07254761002657e-18*v_GRID_a_i + 6.30775359573304e-19*v_GRID_a_r - 0.024174053182917*v_GRID_b_i + 0.040290088638195*v_GRID_b_r + 1.78419315993592e-17*v_GRID_c_i + 9.01107656533306e-19*v_GRID_c_r - 2.07254761002657e-18*v_POI_a_i - 6.30775359573304e-19*v_POI_a_r + 0.024174053182917*v_POI_b_i - 0.040290088638195*v_POI_b_r - 1.78419315993592e-17*v_POI_c_i - 9.01107656533306e-19*v_POI_c_r
        struct[0].g[88,0] = -i_l_POI_GRID_c_r + 7.20886125226632e-19*v_GRID_a_i + 1.35166148479994e-18*v_GRID_a_r + 4.50553828266631e-19*v_GRID_b_i + 1.71210454741325e-17*v_GRID_b_r - 0.040290088638195*v_GRID_c_i - 0.024174053182917*v_GRID_c_r - 7.20886125226632e-19*v_POI_a_i - 1.35166148479994e-18*v_POI_a_r - 4.50553828266631e-19*v_POI_b_i - 1.71210454741325e-17*v_POI_b_r + 0.040290088638195*v_POI_c_i + 0.024174053182917*v_POI_c_r
        struct[0].g[89,0] = -i_l_POI_GRID_c_i + 1.35166148479994e-18*v_GRID_a_i - 7.20886125226632e-19*v_GRID_a_r + 1.71210454741325e-17*v_GRID_b_i - 4.50553828266631e-19*v_GRID_b_r - 0.024174053182917*v_GRID_c_i + 0.040290088638195*v_GRID_c_r - 1.35166148479994e-18*v_POI_a_i + 7.20886125226632e-19*v_POI_a_r - 1.71210454741325e-17*v_POI_b_i + 4.50553828266631e-19*v_POI_b_r + 0.024174053182917*v_POI_c_i - 0.040290088638195*v_POI_c_r
        struct[0].g[90,0] = i_W1lv_a_i*v_W1lv_a_i + i_W1lv_a_r*v_W1lv_a_r - p_W1lv_a
        struct[0].g[91,0] = i_W1lv_b_i*v_W1lv_b_i + i_W1lv_b_r*v_W1lv_b_r - p_W1lv_b
        struct[0].g[92,0] = i_W1lv_c_i*v_W1lv_c_i + i_W1lv_c_r*v_W1lv_c_r - p_W1lv_c
        struct[0].g[93,0] = -i_W1lv_a_i*v_W1lv_a_r + i_W1lv_a_r*v_W1lv_a_i - q_W1lv_a
        struct[0].g[94,0] = -i_W1lv_b_i*v_W1lv_b_r + i_W1lv_b_r*v_W1lv_b_i - q_W1lv_b
        struct[0].g[95,0] = -i_W1lv_c_i*v_W1lv_c_r + i_W1lv_c_r*v_W1lv_c_i - q_W1lv_c
        struct[0].g[96,0] = -v_m_W1lv + (v_W1lv_a_i**2 + v_W1lv_a_r**2)**0.5/V_base_W1lv
        struct[0].g[97,0] = -v_m_W1mv + (v_W1mv_a_i**2 + v_W1mv_a_r**2)**0.5/V_base_W1mv
        struct[0].g[98,0] = Dq_r_W1lv + K_p_v_W1lv*(Dv_r_W1lv - u_ctrl_v_W1lv*v_m_W1mv + v_loc_ref_W1lv - v_m_W1lv*(1.0 - u_ctrl_v_W1lv)) - i_reac_ref_W1lv
        struct[0].g[99,0] = S_base_W1lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)])) - q_ref_W1lv
        struct[0].g[100,0] = i_W2lv_a_i*v_W2lv_a_i + i_W2lv_a_r*v_W2lv_a_r - p_W2lv_a
        struct[0].g[101,0] = i_W2lv_b_i*v_W2lv_b_i + i_W2lv_b_r*v_W2lv_b_r - p_W2lv_b
        struct[0].g[102,0] = i_W2lv_c_i*v_W2lv_c_i + i_W2lv_c_r*v_W2lv_c_r - p_W2lv_c
        struct[0].g[103,0] = -i_W2lv_a_i*v_W2lv_a_r + i_W2lv_a_r*v_W2lv_a_i - q_W2lv_a
        struct[0].g[104,0] = -i_W2lv_b_i*v_W2lv_b_r + i_W2lv_b_r*v_W2lv_b_i - q_W2lv_b
        struct[0].g[105,0] = -i_W2lv_c_i*v_W2lv_c_r + i_W2lv_c_r*v_W2lv_c_i - q_W2lv_c
        struct[0].g[106,0] = -v_m_W2lv + (v_W2lv_a_i**2 + v_W2lv_a_r**2)**0.5/V_base_W2lv
        struct[0].g[107,0] = -v_m_W2mv + (v_W2mv_a_i**2 + v_W2mv_a_r**2)**0.5/V_base_W2mv
        struct[0].g[108,0] = Dq_r_W2lv + K_p_v_W2lv*(Dv_r_W2lv - u_ctrl_v_W2lv*v_m_W2mv + v_loc_ref_W2lv - v_m_W2lv*(1.0 - u_ctrl_v_W2lv)) - i_reac_ref_W2lv
        struct[0].g[109,0] = S_base_W2lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)])) - q_ref_W2lv
        struct[0].g[110,0] = i_W3lv_a_i*v_W3lv_a_i + i_W3lv_a_r*v_W3lv_a_r - p_W3lv_a
        struct[0].g[111,0] = i_W3lv_b_i*v_W3lv_b_i + i_W3lv_b_r*v_W3lv_b_r - p_W3lv_b
        struct[0].g[112,0] = i_W3lv_c_i*v_W3lv_c_i + i_W3lv_c_r*v_W3lv_c_r - p_W3lv_c
        struct[0].g[113,0] = -i_W3lv_a_i*v_W3lv_a_r + i_W3lv_a_r*v_W3lv_a_i - q_W3lv_a
        struct[0].g[114,0] = -i_W3lv_b_i*v_W3lv_b_r + i_W3lv_b_r*v_W3lv_b_i - q_W3lv_b
        struct[0].g[115,0] = -i_W3lv_c_i*v_W3lv_c_r + i_W3lv_c_r*v_W3lv_c_i - q_W3lv_c
        struct[0].g[116,0] = -v_m_W3lv + (v_W3lv_a_i**2 + v_W3lv_a_r**2)**0.5/V_base_W3lv
        struct[0].g[117,0] = -v_m_W3mv + (v_W3mv_a_i**2 + v_W3mv_a_r**2)**0.5/V_base_W3mv
        struct[0].g[118,0] = Dq_r_W3lv + K_p_v_W3lv*(Dv_r_W3lv - u_ctrl_v_W3lv*v_m_W3mv + v_loc_ref_W3lv - v_m_W3lv*(1.0 - u_ctrl_v_W3lv)) - i_reac_ref_W3lv
        struct[0].g[119,0] = S_base_W3lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)])) - q_ref_W3lv
        struct[0].g[120,0] = i_STlv_a_i*v_STlv_a_i + i_STlv_a_r*v_STlv_a_r - p_STlv_a
        struct[0].g[121,0] = i_STlv_b_i*v_STlv_b_i + i_STlv_b_r*v_STlv_b_r - p_STlv_b
        struct[0].g[122,0] = i_STlv_c_i*v_STlv_c_i + i_STlv_c_r*v_STlv_c_r - p_STlv_c
        struct[0].g[123,0] = -i_STlv_a_i*v_STlv_a_r + i_STlv_a_r*v_STlv_a_i - q_STlv_a
        struct[0].g[124,0] = -i_STlv_b_i*v_STlv_b_r + i_STlv_b_r*v_STlv_b_i - q_STlv_b
        struct[0].g[125,0] = -i_STlv_c_i*v_STlv_c_r + i_STlv_c_r*v_STlv_c_i - q_STlv_c
        struct[0].g[126,0] = -v_m_STlv + (v_STlv_a_i**2 + v_STlv_a_r**2)**0.5/V_base_STlv
        struct[0].g[127,0] = -v_m_STmv + (v_STmv_a_i**2 + v_STmv_a_r**2)**0.5/V_base_STmv
        struct[0].g[128,0] = Dq_r_STlv + K_p_v_STlv*(Dv_r_STlv - u_ctrl_v_STlv*v_m_STmv + v_loc_ref_STlv - v_m_STlv*(1.0 - u_ctrl_v_STlv)) - i_reac_ref_STlv
        struct[0].g[129,0] = S_base_STlv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)])) - q_ref_STlv
    
    # Outputs:
    if mode == 3:

    
        pass

    if mode == 10:

        struct[0].Fx_ini[0,0] = -1/T_pq_W1lv
        struct[0].Fx_ini[1,1] = -1/T_pq_W1lv
        struct[0].Fx_ini[2,2] = -1/T_pq_W1lv
        struct[0].Fx_ini[3,3] = -1/T_pq_W1lv
        struct[0].Fx_ini[4,4] = -1/T_pq_W1lv
        struct[0].Fx_ini[5,5] = -1/T_pq_W1lv
        struct[0].Fx_ini[6,6] = -1/T_pq_W2lv
        struct[0].Fx_ini[7,7] = -1/T_pq_W2lv
        struct[0].Fx_ini[8,8] = -1/T_pq_W2lv
        struct[0].Fx_ini[9,9] = -1/T_pq_W2lv
        struct[0].Fx_ini[10,10] = -1/T_pq_W2lv
        struct[0].Fx_ini[11,11] = -1/T_pq_W2lv
        struct[0].Fx_ini[12,12] = -1/T_pq_W3lv
        struct[0].Fx_ini[13,13] = -1/T_pq_W3lv
        struct[0].Fx_ini[14,14] = -1/T_pq_W3lv
        struct[0].Fx_ini[15,15] = -1/T_pq_W3lv
        struct[0].Fx_ini[16,16] = -1/T_pq_W3lv
        struct[0].Fx_ini[17,17] = -1/T_pq_W3lv
        struct[0].Fx_ini[18,18] = -1/T_pq_STlv
        struct[0].Fx_ini[19,19] = -1/T_pq_STlv
        struct[0].Fx_ini[20,20] = -1/T_pq_STlv
        struct[0].Fx_ini[21,21] = -1/T_pq_STlv
        struct[0].Fx_ini[22,22] = -1/T_pq_STlv
        struct[0].Fx_ini[23,23] = -1/T_pq_STlv

    if mode == 11:

        struct[0].Fy_ini[3,99] = 1/(3*T_pq_W1lv) 
        struct[0].Fy_ini[4,99] = 1/(3*T_pq_W1lv) 
        struct[0].Fy_ini[5,99] = 1/(3*T_pq_W1lv) 
        struct[0].Fy_ini[9,109] = 1/(3*T_pq_W2lv) 
        struct[0].Fy_ini[10,109] = 1/(3*T_pq_W2lv) 
        struct[0].Fy_ini[11,109] = 1/(3*T_pq_W2lv) 
        struct[0].Fy_ini[15,119] = 1/(3*T_pq_W3lv) 
        struct[0].Fy_ini[16,119] = 1/(3*T_pq_W3lv) 
        struct[0].Fy_ini[17,119] = 1/(3*T_pq_W3lv) 
        struct[0].Fy_ini[21,129] = 1/(3*T_pq_STlv) 
        struct[0].Fy_ini[22,129] = 1/(3*T_pq_STlv) 
        struct[0].Fy_ini[23,129] = 1/(3*T_pq_STlv) 

        struct[0].Gy_ini[0,0] = -14.1918856474622
        struct[0].Gy_ini[0,1] = -85.1513138847732
        struct[0].Gy_ini[0,36] = 0.282682270461039
        struct[0].Gy_ini[0,37] = 1.69609362276623
        struct[0].Gy_ini[0,40] = -0.282682270461039
        struct[0].Gy_ini[0,41] = -1.69609362276623
        struct[0].Gy_ini[0,90] = 1
        struct[0].Gy_ini[1,0] = 85.1513138847732
        struct[0].Gy_ini[1,1] = -14.1918856474622
        struct[0].Gy_ini[1,36] = -1.69609362276623
        struct[0].Gy_ini[1,37] = 0.282682270461039
        struct[0].Gy_ini[1,40] = 1.69609362276623
        struct[0].Gy_ini[1,41] = -0.282682270461039
        struct[0].Gy_ini[1,91] = 1
        struct[0].Gy_ini[2,2] = -14.1918856474622
        struct[0].Gy_ini[2,3] = -85.1513138847732
        struct[0].Gy_ini[2,36] = -0.282682270461039
        struct[0].Gy_ini[2,37] = -1.69609362276623
        struct[0].Gy_ini[2,38] = 0.282682270461039
        struct[0].Gy_ini[2,39] = 1.69609362276623
        struct[0].Gy_ini[2,92] = 1
        struct[0].Gy_ini[3,2] = 85.1513138847732
        struct[0].Gy_ini[3,3] = -14.1918856474622
        struct[0].Gy_ini[3,36] = 1.69609362276623
        struct[0].Gy_ini[3,37] = -0.282682270461039
        struct[0].Gy_ini[3,38] = -1.69609362276623
        struct[0].Gy_ini[3,39] = 0.282682270461039
        struct[0].Gy_ini[3,93] = 1
        struct[0].Gy_ini[4,4] = -14.1918856474622
        struct[0].Gy_ini[4,5] = -85.1513138847732
        struct[0].Gy_ini[4,38] = -0.282682270461039
        struct[0].Gy_ini[4,39] = -1.69609362276623
        struct[0].Gy_ini[4,40] = 0.282682270461039
        struct[0].Gy_ini[4,41] = 1.69609362276623
        struct[0].Gy_ini[4,94] = 1
        struct[0].Gy_ini[5,4] = 85.1513138847732
        struct[0].Gy_ini[5,5] = -14.1918856474622
        struct[0].Gy_ini[5,38] = 1.69609362276623
        struct[0].Gy_ini[5,39] = -0.282682270461039
        struct[0].Gy_ini[5,40] = -1.69609362276623
        struct[0].Gy_ini[5,41] = 0.282682270461039
        struct[0].Gy_ini[5,95] = 1
        struct[0].Gy_ini[6,6] = -14.1918856474622
        struct[0].Gy_ini[6,7] = -85.1513138847732
        struct[0].Gy_ini[6,42] = 0.282682270461039
        struct[0].Gy_ini[6,43] = 1.69609362276623
        struct[0].Gy_ini[6,46] = -0.282682270461039
        struct[0].Gy_ini[6,47] = -1.69609362276623
        struct[0].Gy_ini[6,100] = 1
        struct[0].Gy_ini[7,6] = 85.1513138847732
        struct[0].Gy_ini[7,7] = -14.1918856474622
        struct[0].Gy_ini[7,42] = -1.69609362276623
        struct[0].Gy_ini[7,43] = 0.282682270461039
        struct[0].Gy_ini[7,46] = 1.69609362276623
        struct[0].Gy_ini[7,47] = -0.282682270461039
        struct[0].Gy_ini[7,101] = 1
        struct[0].Gy_ini[8,8] = -14.1918856474622
        struct[0].Gy_ini[8,9] = -85.1513138847732
        struct[0].Gy_ini[8,42] = -0.282682270461039
        struct[0].Gy_ini[8,43] = -1.69609362276623
        struct[0].Gy_ini[8,44] = 0.282682270461039
        struct[0].Gy_ini[8,45] = 1.69609362276623
        struct[0].Gy_ini[8,102] = 1
        struct[0].Gy_ini[9,8] = 85.1513138847732
        struct[0].Gy_ini[9,9] = -14.1918856474622
        struct[0].Gy_ini[9,42] = 1.69609362276623
        struct[0].Gy_ini[9,43] = -0.282682270461039
        struct[0].Gy_ini[9,44] = -1.69609362276623
        struct[0].Gy_ini[9,45] = 0.282682270461039
        struct[0].Gy_ini[9,103] = 1
        struct[0].Gy_ini[10,10] = -14.1918856474622
        struct[0].Gy_ini[10,11] = -85.1513138847732
        struct[0].Gy_ini[10,44] = -0.282682270461039
        struct[0].Gy_ini[10,45] = -1.69609362276623
        struct[0].Gy_ini[10,46] = 0.282682270461039
        struct[0].Gy_ini[10,47] = 1.69609362276623
        struct[0].Gy_ini[10,104] = 1
        struct[0].Gy_ini[11,10] = 85.1513138847732
        struct[0].Gy_ini[11,11] = -14.1918856474622
        struct[0].Gy_ini[11,44] = 1.69609362276623
        struct[0].Gy_ini[11,45] = -0.282682270461039
        struct[0].Gy_ini[11,46] = -1.69609362276623
        struct[0].Gy_ini[11,47] = 0.282682270461039
        struct[0].Gy_ini[11,105] = 1
        struct[0].Gy_ini[12,12] = -14.1918856474622
        struct[0].Gy_ini[12,13] = -85.1513138847732
        struct[0].Gy_ini[12,48] = 0.282682270461039
        struct[0].Gy_ini[12,49] = 1.69609362276623
        struct[0].Gy_ini[12,52] = -0.282682270461039
        struct[0].Gy_ini[12,53] = -1.69609362276623
        struct[0].Gy_ini[12,110] = 1
        struct[0].Gy_ini[13,12] = 85.1513138847732
        struct[0].Gy_ini[13,13] = -14.1918856474622
        struct[0].Gy_ini[13,48] = -1.69609362276623
        struct[0].Gy_ini[13,49] = 0.282682270461039
        struct[0].Gy_ini[13,52] = 1.69609362276623
        struct[0].Gy_ini[13,53] = -0.282682270461039
        struct[0].Gy_ini[13,111] = 1
        struct[0].Gy_ini[14,14] = -14.1918856474622
        struct[0].Gy_ini[14,15] = -85.1513138847732
        struct[0].Gy_ini[14,48] = -0.282682270461039
        struct[0].Gy_ini[14,49] = -1.69609362276623
        struct[0].Gy_ini[14,50] = 0.282682270461039
        struct[0].Gy_ini[14,51] = 1.69609362276623
        struct[0].Gy_ini[14,112] = 1
        struct[0].Gy_ini[15,14] = 85.1513138847732
        struct[0].Gy_ini[15,15] = -14.1918856474622
        struct[0].Gy_ini[15,48] = 1.69609362276623
        struct[0].Gy_ini[15,49] = -0.282682270461039
        struct[0].Gy_ini[15,50] = -1.69609362276623
        struct[0].Gy_ini[15,51] = 0.282682270461039
        struct[0].Gy_ini[15,113] = 1
        struct[0].Gy_ini[16,16] = -14.1918856474622
        struct[0].Gy_ini[16,17] = -85.1513138847732
        struct[0].Gy_ini[16,50] = -0.282682270461039
        struct[0].Gy_ini[16,51] = -1.69609362276623
        struct[0].Gy_ini[16,52] = 0.282682270461039
        struct[0].Gy_ini[16,53] = 1.69609362276623
        struct[0].Gy_ini[16,114] = 1
        struct[0].Gy_ini[17,16] = 85.1513138847732
        struct[0].Gy_ini[17,17] = -14.1918856474622
        struct[0].Gy_ini[17,50] = 1.69609362276623
        struct[0].Gy_ini[17,51] = -0.282682270461039
        struct[0].Gy_ini[17,52] = -1.69609362276623
        struct[0].Gy_ini[17,53] = 0.282682270461039
        struct[0].Gy_ini[17,115] = 1
        struct[0].Gy_ini[18,18] = -14.1918856474622
        struct[0].Gy_ini[18,19] = -85.1513138847732
        struct[0].Gy_ini[18,54] = 0.282682270461039
        struct[0].Gy_ini[18,55] = 1.69609362276623
        struct[0].Gy_ini[18,58] = -0.282682270461039
        struct[0].Gy_ini[18,59] = -1.69609362276623
        struct[0].Gy_ini[18,120] = 1
        struct[0].Gy_ini[19,18] = 85.1513138847732
        struct[0].Gy_ini[19,19] = -14.1918856474622
        struct[0].Gy_ini[19,54] = -1.69609362276623
        struct[0].Gy_ini[19,55] = 0.282682270461039
        struct[0].Gy_ini[19,58] = 1.69609362276623
        struct[0].Gy_ini[19,59] = -0.282682270461039
        struct[0].Gy_ini[19,121] = 1
        struct[0].Gy_ini[20,20] = -14.1918856474622
        struct[0].Gy_ini[20,21] = -85.1513138847732
        struct[0].Gy_ini[20,54] = -0.282682270461039
        struct[0].Gy_ini[20,55] = -1.69609362276623
        struct[0].Gy_ini[20,56] = 0.282682270461039
        struct[0].Gy_ini[20,57] = 1.69609362276623
        struct[0].Gy_ini[20,122] = 1
        struct[0].Gy_ini[21,20] = 85.1513138847732
        struct[0].Gy_ini[21,21] = -14.1918856474622
        struct[0].Gy_ini[21,54] = 1.69609362276623
        struct[0].Gy_ini[21,55] = -0.282682270461039
        struct[0].Gy_ini[21,56] = -1.69609362276623
        struct[0].Gy_ini[21,57] = 0.282682270461039
        struct[0].Gy_ini[21,123] = 1
        struct[0].Gy_ini[22,22] = -14.1918856474622
        struct[0].Gy_ini[22,23] = -85.1513138847732
        struct[0].Gy_ini[22,56] = -0.282682270461039
        struct[0].Gy_ini[22,57] = -1.69609362276623
        struct[0].Gy_ini[22,58] = 0.282682270461039
        struct[0].Gy_ini[22,59] = 1.69609362276623
        struct[0].Gy_ini[22,124] = 1
        struct[0].Gy_ini[23,22] = 85.1513138847732
        struct[0].Gy_ini[23,23] = -14.1918856474622
        struct[0].Gy_ini[23,56] = 1.69609362276623
        struct[0].Gy_ini[23,57] = -0.282682270461039
        struct[0].Gy_ini[23,58] = -1.69609362276623
        struct[0].Gy_ini[23,59] = 0.282682270461039
        struct[0].Gy_ini[23,125] = 1
        struct[0].Gy_ini[24,24] = -0.0265286009920103
        struct[0].Gy_ini[24,25] = -0.0591264711109411
        struct[0].Gy_ini[24,26] = 0.00117727390454664
        struct[0].Gy_ini[24,27] = 0.00941819123637305
        struct[0].Gy_ini[24,28] = 0.00117727390454664
        struct[0].Gy_ini[24,29] = 0.00941819123637305
        struct[0].Gy_ini[24,30] = 0.00672902411642920
        struct[0].Gy_ini[24,31] = 0.0538321929314336
        struct[0].Gy_ini[24,32] = -0.00672902411642920
        struct[0].Gy_ini[24,33] = -0.0538321929314336
        struct[0].Gy_ini[25,24] = 0.0591264711109411
        struct[0].Gy_ini[25,25] = -0.0265286009920103
        struct[0].Gy_ini[25,26] = -0.00941819123637305
        struct[0].Gy_ini[25,27] = 0.00117727390454664
        struct[0].Gy_ini[25,28] = -0.00941819123637305
        struct[0].Gy_ini[25,29] = 0.00117727390454664
        struct[0].Gy_ini[25,30] = -0.0538321929314336
        struct[0].Gy_ini[25,31] = 0.00672902411642920
        struct[0].Gy_ini[25,32] = 0.0538321929314336
        struct[0].Gy_ini[25,33] = -0.00672902411642920
        struct[0].Gy_ini[26,24] = 0.00117727390454663
        struct[0].Gy_ini[26,25] = 0.00941819123637305
        struct[0].Gy_ini[26,26] = -0.0265286009920103
        struct[0].Gy_ini[26,27] = -0.0591264711109411
        struct[0].Gy_ini[26,28] = 0.00117727390454665
        struct[0].Gy_ini[26,29] = 0.00941819123637305
        struct[0].Gy_ini[26,32] = 0.00672902411642920
        struct[0].Gy_ini[26,33] = 0.0538321929314336
        struct[0].Gy_ini[26,34] = -0.00672902411642920
        struct[0].Gy_ini[26,35] = -0.0538321929314336
        struct[0].Gy_ini[27,24] = -0.00941819123637305
        struct[0].Gy_ini[27,25] = 0.00117727390454663
        struct[0].Gy_ini[27,26] = 0.0591264711109411
        struct[0].Gy_ini[27,27] = -0.0265286009920103
        struct[0].Gy_ini[27,28] = -0.00941819123637305
        struct[0].Gy_ini[27,29] = 0.00117727390454665
        struct[0].Gy_ini[27,32] = -0.0538321929314336
        struct[0].Gy_ini[27,33] = 0.00672902411642920
        struct[0].Gy_ini[27,34] = 0.0538321929314336
        struct[0].Gy_ini[27,35] = -0.00672902411642920
        struct[0].Gy_ini[28,24] = 0.00117727390454663
        struct[0].Gy_ini[28,25] = 0.00941819123637305
        struct[0].Gy_ini[28,26] = 0.00117727390454665
        struct[0].Gy_ini[28,27] = 0.00941819123637305
        struct[0].Gy_ini[28,28] = -0.0265286009920103
        struct[0].Gy_ini[28,29] = -0.0591264711109411
        struct[0].Gy_ini[28,30] = -0.00672902411642920
        struct[0].Gy_ini[28,31] = -0.0538321929314336
        struct[0].Gy_ini[28,34] = 0.00672902411642920
        struct[0].Gy_ini[28,35] = 0.0538321929314336
        struct[0].Gy_ini[29,24] = -0.00941819123637305
        struct[0].Gy_ini[29,25] = 0.00117727390454663
        struct[0].Gy_ini[29,26] = -0.00941819123637305
        struct[0].Gy_ini[29,27] = 0.00117727390454665
        struct[0].Gy_ini[29,28] = 0.0591264711109411
        struct[0].Gy_ini[29,29] = -0.0265286009920103
        struct[0].Gy_ini[29,30] = 0.0538321929314336
        struct[0].Gy_ini[29,31] = -0.00672902411642920
        struct[0].Gy_ini[29,34] = -0.0538321929314336
        struct[0].Gy_ini[29,35] = 0.00672902411642920
        struct[0].Gy_ini[30,24] = 0.00672902411642920
        struct[0].Gy_ini[30,25] = 0.0538321929314336
        struct[0].Gy_ini[30,28] = -0.00672902411642920
        struct[0].Gy_ini[30,29] = -0.0538321929314336
        struct[0].Gy_ini[30,30] = -188.924390492986
        struct[0].Gy_ini[30,31] = -155.244588874881
        struct[0].Gy_ini[30,32] = 53.9540151298641
        struct[0].Gy_ini[30,33] = 44.2677164725443
        struct[0].Gy_ini[30,34] = 53.9540151298641
        struct[0].Gy_ini[30,35] = 44.2677164725443
        struct[0].Gy_ini[30,48] = 7.26444274184068
        struct[0].Gy_ini[30,49] = 5.95911318666618
        struct[0].Gy_ini[30,50] = -2.07555506909734
        struct[0].Gy_ini[30,51] = -1.70260376761891
        struct[0].Gy_ini[30,52] = -2.07555506909734
        struct[0].Gy_ini[30,53] = -1.70260376761891
        struct[0].Gy_ini[30,54] = 181.611068546017
        struct[0].Gy_ini[30,55] = 148.977829666654
        struct[0].Gy_ini[30,56] = -51.8888767274334
        struct[0].Gy_ini[30,57] = -42.5650941904727
        struct[0].Gy_ini[30,58] = -51.8888767274334
        struct[0].Gy_ini[30,59] = -42.5650941904727
        struct[0].Gy_ini[31,24] = -0.0538321929314336
        struct[0].Gy_ini[31,25] = 0.00672902411642920
        struct[0].Gy_ini[31,28] = 0.0538321929314336
        struct[0].Gy_ini[31,29] = -0.00672902411642920
        struct[0].Gy_ini[31,30] = 155.244588874881
        struct[0].Gy_ini[31,31] = -188.924390492986
        struct[0].Gy_ini[31,32] = -44.2677164725443
        struct[0].Gy_ini[31,33] = 53.9540151298641
        struct[0].Gy_ini[31,34] = -44.2677164725443
        struct[0].Gy_ini[31,35] = 53.9540151298641
        struct[0].Gy_ini[31,48] = -5.95911318666618
        struct[0].Gy_ini[31,49] = 7.26444274184068
        struct[0].Gy_ini[31,50] = 1.70260376761891
        struct[0].Gy_ini[31,51] = -2.07555506909734
        struct[0].Gy_ini[31,52] = 1.70260376761891
        struct[0].Gy_ini[31,53] = -2.07555506909734
        struct[0].Gy_ini[31,54] = -148.977829666654
        struct[0].Gy_ini[31,55] = 181.611068546017
        struct[0].Gy_ini[31,56] = 42.5650941904727
        struct[0].Gy_ini[31,57] = -51.8888767274334
        struct[0].Gy_ini[31,58] = 42.5650941904727
        struct[0].Gy_ini[31,59] = -51.8888767274334
        struct[0].Gy_ini[32,24] = -0.00672902411642920
        struct[0].Gy_ini[32,25] = -0.0538321929314336
        struct[0].Gy_ini[32,26] = 0.00672902411642920
        struct[0].Gy_ini[32,27] = 0.0538321929314336
        struct[0].Gy_ini[32,30] = 53.9540151298641
        struct[0].Gy_ini[32,31] = 44.2677164725443
        struct[0].Gy_ini[32,32] = -188.924390492986
        struct[0].Gy_ini[32,33] = -155.244588874881
        struct[0].Gy_ini[32,34] = 53.9540151298642
        struct[0].Gy_ini[32,35] = 44.2677164725443
        struct[0].Gy_ini[32,48] = -2.07555506909734
        struct[0].Gy_ini[32,49] = -1.70260376761891
        struct[0].Gy_ini[32,50] = 7.26444274184068
        struct[0].Gy_ini[32,51] = 5.95911318666618
        struct[0].Gy_ini[32,52] = -2.07555506909734
        struct[0].Gy_ini[32,53] = -1.70260376761891
        struct[0].Gy_ini[32,54] = -51.8888767274334
        struct[0].Gy_ini[32,55] = -42.5650941904727
        struct[0].Gy_ini[32,56] = 181.611068546017
        struct[0].Gy_ini[32,57] = 148.977829666654
        struct[0].Gy_ini[32,58] = -51.8888767274335
        struct[0].Gy_ini[32,59] = -42.5650941904727
        struct[0].Gy_ini[33,24] = 0.0538321929314336
        struct[0].Gy_ini[33,25] = -0.00672902411642920
        struct[0].Gy_ini[33,26] = -0.0538321929314336
        struct[0].Gy_ini[33,27] = 0.00672902411642920
        struct[0].Gy_ini[33,30] = -44.2677164725443
        struct[0].Gy_ini[33,31] = 53.9540151298641
        struct[0].Gy_ini[33,32] = 155.244588874881
        struct[0].Gy_ini[33,33] = -188.924390492986
        struct[0].Gy_ini[33,34] = -44.2677164725443
        struct[0].Gy_ini[33,35] = 53.9540151298642
        struct[0].Gy_ini[33,48] = 1.70260376761891
        struct[0].Gy_ini[33,49] = -2.07555506909734
        struct[0].Gy_ini[33,50] = -5.95911318666618
        struct[0].Gy_ini[33,51] = 7.26444274184068
        struct[0].Gy_ini[33,52] = 1.70260376761891
        struct[0].Gy_ini[33,53] = -2.07555506909734
        struct[0].Gy_ini[33,54] = 42.5650941904727
        struct[0].Gy_ini[33,55] = -51.8888767274334
        struct[0].Gy_ini[33,56] = -148.977829666654
        struct[0].Gy_ini[33,57] = 181.611068546017
        struct[0].Gy_ini[33,58] = 42.5650941904727
        struct[0].Gy_ini[33,59] = -51.8888767274335
        struct[0].Gy_ini[34,26] = -0.00672902411642920
        struct[0].Gy_ini[34,27] = -0.0538321929314336
        struct[0].Gy_ini[34,28] = 0.00672902411642920
        struct[0].Gy_ini[34,29] = 0.0538321929314336
        struct[0].Gy_ini[34,30] = 53.9540151298641
        struct[0].Gy_ini[34,31] = 44.2677164725443
        struct[0].Gy_ini[34,32] = 53.9540151298642
        struct[0].Gy_ini[34,33] = 44.2677164725443
        struct[0].Gy_ini[34,34] = -188.924390492986
        struct[0].Gy_ini[34,35] = -155.244588874881
        struct[0].Gy_ini[34,48] = -2.07555506909734
        struct[0].Gy_ini[34,49] = -1.70260376761891
        struct[0].Gy_ini[34,50] = -2.07555506909734
        struct[0].Gy_ini[34,51] = -1.70260376761891
        struct[0].Gy_ini[34,52] = 7.26444274184068
        struct[0].Gy_ini[34,53] = 5.95911318666618
        struct[0].Gy_ini[34,54] = -51.8888767274334
        struct[0].Gy_ini[34,55] = -42.5650941904727
        struct[0].Gy_ini[34,56] = -51.8888767274335
        struct[0].Gy_ini[34,57] = -42.5650941904727
        struct[0].Gy_ini[34,58] = 181.611068546017
        struct[0].Gy_ini[34,59] = 148.977829666654
        struct[0].Gy_ini[35,26] = 0.0538321929314336
        struct[0].Gy_ini[35,27] = -0.00672902411642920
        struct[0].Gy_ini[35,28] = -0.0538321929314336
        struct[0].Gy_ini[35,29] = 0.00672902411642920
        struct[0].Gy_ini[35,30] = -44.2677164725443
        struct[0].Gy_ini[35,31] = 53.9540151298641
        struct[0].Gy_ini[35,32] = -44.2677164725443
        struct[0].Gy_ini[35,33] = 53.9540151298642
        struct[0].Gy_ini[35,34] = 155.244588874881
        struct[0].Gy_ini[35,35] = -188.924390492986
        struct[0].Gy_ini[35,48] = 1.70260376761891
        struct[0].Gy_ini[35,49] = -2.07555506909734
        struct[0].Gy_ini[35,50] = 1.70260376761891
        struct[0].Gy_ini[35,51] = -2.07555506909734
        struct[0].Gy_ini[35,52] = -5.95911318666618
        struct[0].Gy_ini[35,53] = 7.26444274184068
        struct[0].Gy_ini[35,54] = 42.5650941904727
        struct[0].Gy_ini[35,55] = -51.8888767274334
        struct[0].Gy_ini[35,56] = 42.5650941904727
        struct[0].Gy_ini[35,57] = -51.8888767274335
        struct[0].Gy_ini[35,58] = -148.977829666654
        struct[0].Gy_ini[35,59] = 181.611068546017
        struct[0].Gy_ini[36,0] = 0.282682270461039
        struct[0].Gy_ini[36,1] = 1.69609362276623
        struct[0].Gy_ini[36,2] = -0.282682270461039
        struct[0].Gy_ini[36,3] = -1.69609362276623
        struct[0].Gy_ini[36,36] = -7.27570400310194
        struct[0].Gy_ini[36,37] = -6.02663624833782
        struct[0].Gy_ini[36,38] = 2.08118569972797
        struct[0].Gy_ini[36,39] = 1.73640535376106
        struct[0].Gy_ini[36,40] = 2.08118569972797
        struct[0].Gy_ini[36,41] = 1.73640535376106
        struct[0].Gy_ini[36,42] = 7.26444274184068
        struct[0].Gy_ini[36,43] = 5.95911318666618
        struct[0].Gy_ini[36,44] = -2.07555506909734
        struct[0].Gy_ini[36,45] = -1.70260376761891
        struct[0].Gy_ini[36,46] = -2.07555506909734
        struct[0].Gy_ini[36,47] = -1.70260376761891
        struct[0].Gy_ini[37,0] = -1.69609362276623
        struct[0].Gy_ini[37,1] = 0.282682270461039
        struct[0].Gy_ini[37,2] = 1.69609362276623
        struct[0].Gy_ini[37,3] = -0.282682270461039
        struct[0].Gy_ini[37,36] = 6.02663624833782
        struct[0].Gy_ini[37,37] = -7.27570400310194
        struct[0].Gy_ini[37,38] = -1.73640535376106
        struct[0].Gy_ini[37,39] = 2.08118569972797
        struct[0].Gy_ini[37,40] = -1.73640535376106
        struct[0].Gy_ini[37,41] = 2.08118569972797
        struct[0].Gy_ini[37,42] = -5.95911318666618
        struct[0].Gy_ini[37,43] = 7.26444274184068
        struct[0].Gy_ini[37,44] = 1.70260376761891
        struct[0].Gy_ini[37,45] = -2.07555506909734
        struct[0].Gy_ini[37,46] = 1.70260376761891
        struct[0].Gy_ini[37,47] = -2.07555506909734
        struct[0].Gy_ini[38,2] = 0.282682270461039
        struct[0].Gy_ini[38,3] = 1.69609362276623
        struct[0].Gy_ini[38,4] = -0.282682270461039
        struct[0].Gy_ini[38,5] = -1.69609362276623
        struct[0].Gy_ini[38,36] = 2.08118569972797
        struct[0].Gy_ini[38,37] = 1.73640535376106
        struct[0].Gy_ini[38,38] = -7.27570400310194
        struct[0].Gy_ini[38,39] = -6.02663624833782
        struct[0].Gy_ini[38,40] = 2.08118569972797
        struct[0].Gy_ini[38,41] = 1.73640535376106
        struct[0].Gy_ini[38,42] = -2.07555506909734
        struct[0].Gy_ini[38,43] = -1.70260376761891
        struct[0].Gy_ini[38,44] = 7.26444274184068
        struct[0].Gy_ini[38,45] = 5.95911318666618
        struct[0].Gy_ini[38,46] = -2.07555506909734
        struct[0].Gy_ini[38,47] = -1.70260376761891
        struct[0].Gy_ini[39,2] = -1.69609362276623
        struct[0].Gy_ini[39,3] = 0.282682270461039
        struct[0].Gy_ini[39,4] = 1.69609362276623
        struct[0].Gy_ini[39,5] = -0.282682270461039
        struct[0].Gy_ini[39,36] = -1.73640535376106
        struct[0].Gy_ini[39,37] = 2.08118569972797
        struct[0].Gy_ini[39,38] = 6.02663624833782
        struct[0].Gy_ini[39,39] = -7.27570400310194
        struct[0].Gy_ini[39,40] = -1.73640535376106
        struct[0].Gy_ini[39,41] = 2.08118569972797
        struct[0].Gy_ini[39,42] = 1.70260376761891
        struct[0].Gy_ini[39,43] = -2.07555506909734
        struct[0].Gy_ini[39,44] = -5.95911318666618
        struct[0].Gy_ini[39,45] = 7.26444274184068
        struct[0].Gy_ini[39,46] = 1.70260376761891
        struct[0].Gy_ini[39,47] = -2.07555506909734
        struct[0].Gy_ini[40,0] = -0.282682270461039
        struct[0].Gy_ini[40,1] = -1.69609362276623
        struct[0].Gy_ini[40,4] = 0.282682270461039
        struct[0].Gy_ini[40,5] = 1.69609362276623
        struct[0].Gy_ini[40,36] = 2.08118569972797
        struct[0].Gy_ini[40,37] = 1.73640535376106
        struct[0].Gy_ini[40,38] = 2.08118569972797
        struct[0].Gy_ini[40,39] = 1.73640535376106
        struct[0].Gy_ini[40,40] = -7.27570400310194
        struct[0].Gy_ini[40,41] = -6.02663624833782
        struct[0].Gy_ini[40,42] = -2.07555506909734
        struct[0].Gy_ini[40,43] = -1.70260376761891
        struct[0].Gy_ini[40,44] = -2.07555506909734
        struct[0].Gy_ini[40,45] = -1.70260376761891
        struct[0].Gy_ini[40,46] = 7.26444274184068
        struct[0].Gy_ini[40,47] = 5.95911318666618
        struct[0].Gy_ini[41,0] = 1.69609362276623
        struct[0].Gy_ini[41,1] = -0.282682270461039
        struct[0].Gy_ini[41,4] = -1.69609362276623
        struct[0].Gy_ini[41,5] = 0.282682270461039
        struct[0].Gy_ini[41,36] = -1.73640535376106
        struct[0].Gy_ini[41,37] = 2.08118569972797
        struct[0].Gy_ini[41,38] = -1.73640535376106
        struct[0].Gy_ini[41,39] = 2.08118569972797
        struct[0].Gy_ini[41,40] = 6.02663624833782
        struct[0].Gy_ini[41,41] = -7.27570400310194
        struct[0].Gy_ini[41,42] = 1.70260376761891
        struct[0].Gy_ini[41,43] = -2.07555506909734
        struct[0].Gy_ini[41,44] = 1.70260376761891
        struct[0].Gy_ini[41,45] = -2.07555506909734
        struct[0].Gy_ini[41,46] = -5.95911318666618
        struct[0].Gy_ini[41,47] = 7.26444274184068
        struct[0].Gy_ini[42,6] = 0.282682270461039
        struct[0].Gy_ini[42,7] = 1.69609362276623
        struct[0].Gy_ini[42,8] = -0.282682270461039
        struct[0].Gy_ini[42,9] = -1.69609362276623
        struct[0].Gy_ini[42,36] = 7.26444274184068
        struct[0].Gy_ini[42,37] = 5.95911318666618
        struct[0].Gy_ini[42,38] = -2.07555506909734
        struct[0].Gy_ini[42,39] = -1.70260376761891
        struct[0].Gy_ini[42,40] = -2.07555506909734
        struct[0].Gy_ini[42,41] = -1.70260376761891
        struct[0].Gy_ini[42,42] = -14.5401467449426
        struct[0].Gy_ini[42,43] = -11.9857049291081
        struct[0].Gy_ini[42,44] = 4.15674076882530
        struct[0].Gy_ini[42,45] = 3.43902692373834
        struct[0].Gy_ini[42,46] = 4.15674076882530
        struct[0].Gy_ini[42,47] = 3.43902692373834
        struct[0].Gy_ini[42,48] = 7.26444274184068
        struct[0].Gy_ini[42,49] = 5.95911318666618
        struct[0].Gy_ini[42,50] = -2.07555506909734
        struct[0].Gy_ini[42,51] = -1.70260376761891
        struct[0].Gy_ini[42,52] = -2.07555506909734
        struct[0].Gy_ini[42,53] = -1.70260376761891
        struct[0].Gy_ini[43,6] = -1.69609362276623
        struct[0].Gy_ini[43,7] = 0.282682270461039
        struct[0].Gy_ini[43,8] = 1.69609362276623
        struct[0].Gy_ini[43,9] = -0.282682270461039
        struct[0].Gy_ini[43,36] = -5.95911318666618
        struct[0].Gy_ini[43,37] = 7.26444274184068
        struct[0].Gy_ini[43,38] = 1.70260376761891
        struct[0].Gy_ini[43,39] = -2.07555506909734
        struct[0].Gy_ini[43,40] = 1.70260376761891
        struct[0].Gy_ini[43,41] = -2.07555506909734
        struct[0].Gy_ini[43,42] = 11.9857049291081
        struct[0].Gy_ini[43,43] = -14.5401467449426
        struct[0].Gy_ini[43,44] = -3.43902692373834
        struct[0].Gy_ini[43,45] = 4.15674076882530
        struct[0].Gy_ini[43,46] = -3.43902692373834
        struct[0].Gy_ini[43,47] = 4.15674076882530
        struct[0].Gy_ini[43,48] = -5.95911318666618
        struct[0].Gy_ini[43,49] = 7.26444274184068
        struct[0].Gy_ini[43,50] = 1.70260376761891
        struct[0].Gy_ini[43,51] = -2.07555506909734
        struct[0].Gy_ini[43,52] = 1.70260376761891
        struct[0].Gy_ini[43,53] = -2.07555506909734
        struct[0].Gy_ini[44,8] = 0.282682270461039
        struct[0].Gy_ini[44,9] = 1.69609362276623
        struct[0].Gy_ini[44,10] = -0.282682270461039
        struct[0].Gy_ini[44,11] = -1.69609362276623
        struct[0].Gy_ini[44,36] = -2.07555506909734
        struct[0].Gy_ini[44,37] = -1.70260376761891
        struct[0].Gy_ini[44,38] = 7.26444274184068
        struct[0].Gy_ini[44,39] = 5.95911318666618
        struct[0].Gy_ini[44,40] = -2.07555506909734
        struct[0].Gy_ini[44,41] = -1.70260376761891
        struct[0].Gy_ini[44,42] = 4.15674076882530
        struct[0].Gy_ini[44,43] = 3.43902692373834
        struct[0].Gy_ini[44,44] = -14.5401467449426
        struct[0].Gy_ini[44,45] = -11.9857049291081
        struct[0].Gy_ini[44,46] = 4.15674076882531
        struct[0].Gy_ini[44,47] = 3.43902692373834
        struct[0].Gy_ini[44,48] = -2.07555506909734
        struct[0].Gy_ini[44,49] = -1.70260376761891
        struct[0].Gy_ini[44,50] = 7.26444274184068
        struct[0].Gy_ini[44,51] = 5.95911318666618
        struct[0].Gy_ini[44,52] = -2.07555506909734
        struct[0].Gy_ini[44,53] = -1.70260376761891
        struct[0].Gy_ini[45,8] = -1.69609362276623
        struct[0].Gy_ini[45,9] = 0.282682270461039
        struct[0].Gy_ini[45,10] = 1.69609362276623
        struct[0].Gy_ini[45,11] = -0.282682270461039
        struct[0].Gy_ini[45,36] = 1.70260376761891
        struct[0].Gy_ini[45,37] = -2.07555506909734
        struct[0].Gy_ini[45,38] = -5.95911318666618
        struct[0].Gy_ini[45,39] = 7.26444274184068
        struct[0].Gy_ini[45,40] = 1.70260376761891
        struct[0].Gy_ini[45,41] = -2.07555506909734
        struct[0].Gy_ini[45,42] = -3.43902692373834
        struct[0].Gy_ini[45,43] = 4.15674076882530
        struct[0].Gy_ini[45,44] = 11.9857049291081
        struct[0].Gy_ini[45,45] = -14.5401467449426
        struct[0].Gy_ini[45,46] = -3.43902692373834
        struct[0].Gy_ini[45,47] = 4.15674076882531
        struct[0].Gy_ini[45,48] = 1.70260376761891
        struct[0].Gy_ini[45,49] = -2.07555506909734
        struct[0].Gy_ini[45,50] = -5.95911318666618
        struct[0].Gy_ini[45,51] = 7.26444274184068
        struct[0].Gy_ini[45,52] = 1.70260376761891
        struct[0].Gy_ini[45,53] = -2.07555506909734
        struct[0].Gy_ini[46,6] = -0.282682270461039
        struct[0].Gy_ini[46,7] = -1.69609362276623
        struct[0].Gy_ini[46,10] = 0.282682270461039
        struct[0].Gy_ini[46,11] = 1.69609362276623
        struct[0].Gy_ini[46,36] = -2.07555506909734
        struct[0].Gy_ini[46,37] = -1.70260376761891
        struct[0].Gy_ini[46,38] = -2.07555506909734
        struct[0].Gy_ini[46,39] = -1.70260376761891
        struct[0].Gy_ini[46,40] = 7.26444274184068
        struct[0].Gy_ini[46,41] = 5.95911318666618
        struct[0].Gy_ini[46,42] = 4.15674076882530
        struct[0].Gy_ini[46,43] = 3.43902692373834
        struct[0].Gy_ini[46,44] = 4.15674076882531
        struct[0].Gy_ini[46,45] = 3.43902692373834
        struct[0].Gy_ini[46,46] = -14.5401467449426
        struct[0].Gy_ini[46,47] = -11.9857049291081
        struct[0].Gy_ini[46,48] = -2.07555506909734
        struct[0].Gy_ini[46,49] = -1.70260376761891
        struct[0].Gy_ini[46,50] = -2.07555506909734
        struct[0].Gy_ini[46,51] = -1.70260376761891
        struct[0].Gy_ini[46,52] = 7.26444274184068
        struct[0].Gy_ini[46,53] = 5.95911318666618
        struct[0].Gy_ini[47,6] = 1.69609362276623
        struct[0].Gy_ini[47,7] = -0.282682270461039
        struct[0].Gy_ini[47,10] = -1.69609362276623
        struct[0].Gy_ini[47,11] = 0.282682270461039
        struct[0].Gy_ini[47,36] = 1.70260376761891
        struct[0].Gy_ini[47,37] = -2.07555506909734
        struct[0].Gy_ini[47,38] = 1.70260376761891
        struct[0].Gy_ini[47,39] = -2.07555506909734
        struct[0].Gy_ini[47,40] = -5.95911318666618
        struct[0].Gy_ini[47,41] = 7.26444274184068
        struct[0].Gy_ini[47,42] = -3.43902692373834
        struct[0].Gy_ini[47,43] = 4.15674076882530
        struct[0].Gy_ini[47,44] = -3.43902692373834
        struct[0].Gy_ini[47,45] = 4.15674076882531
        struct[0].Gy_ini[47,46] = 11.9857049291081
        struct[0].Gy_ini[47,47] = -14.5401467449426
        struct[0].Gy_ini[47,48] = 1.70260376761891
        struct[0].Gy_ini[47,49] = -2.07555506909734
        struct[0].Gy_ini[47,50] = 1.70260376761891
        struct[0].Gy_ini[47,51] = -2.07555506909734
        struct[0].Gy_ini[47,52] = -5.95911318666618
        struct[0].Gy_ini[47,53] = 7.26444274184068
        struct[0].Gy_ini[48,12] = 0.282682270461039
        struct[0].Gy_ini[48,13] = 1.69609362276623
        struct[0].Gy_ini[48,14] = -0.282682270461039
        struct[0].Gy_ini[48,15] = -1.69609362276623
        struct[0].Gy_ini[48,30] = 7.26444274184068
        struct[0].Gy_ini[48,31] = 5.95911318666618
        struct[0].Gy_ini[48,32] = -2.07555506909734
        struct[0].Gy_ini[48,33] = -1.70260376761891
        struct[0].Gy_ini[48,34] = -2.07555506909734
        struct[0].Gy_ini[48,35] = -1.70260376761891
        struct[0].Gy_ini[48,42] = 7.26444274184068
        struct[0].Gy_ini[48,43] = 5.95911318666618
        struct[0].Gy_ini[48,44] = -2.07555506909734
        struct[0].Gy_ini[48,45] = -1.70260376761891
        struct[0].Gy_ini[48,46] = -2.07555506909734
        struct[0].Gy_ini[48,47] = -1.70260376761891
        struct[0].Gy_ini[48,48] = -14.5401467449426
        struct[0].Gy_ini[48,49] = -11.9857049291081
        struct[0].Gy_ini[48,50] = 4.15674076882530
        struct[0].Gy_ini[48,51] = 3.43902692373834
        struct[0].Gy_ini[48,52] = 4.15674076882530
        struct[0].Gy_ini[48,53] = 3.43902692373834
        struct[0].Gy_ini[49,12] = -1.69609362276623
        struct[0].Gy_ini[49,13] = 0.282682270461039
        struct[0].Gy_ini[49,14] = 1.69609362276623
        struct[0].Gy_ini[49,15] = -0.282682270461039
        struct[0].Gy_ini[49,30] = -5.95911318666618
        struct[0].Gy_ini[49,31] = 7.26444274184068
        struct[0].Gy_ini[49,32] = 1.70260376761891
        struct[0].Gy_ini[49,33] = -2.07555506909734
        struct[0].Gy_ini[49,34] = 1.70260376761891
        struct[0].Gy_ini[49,35] = -2.07555506909734
        struct[0].Gy_ini[49,42] = -5.95911318666618
        struct[0].Gy_ini[49,43] = 7.26444274184068
        struct[0].Gy_ini[49,44] = 1.70260376761891
        struct[0].Gy_ini[49,45] = -2.07555506909734
        struct[0].Gy_ini[49,46] = 1.70260376761891
        struct[0].Gy_ini[49,47] = -2.07555506909734
        struct[0].Gy_ini[49,48] = 11.9857049291081
        struct[0].Gy_ini[49,49] = -14.5401467449426
        struct[0].Gy_ini[49,50] = -3.43902692373834
        struct[0].Gy_ini[49,51] = 4.15674076882530
        struct[0].Gy_ini[49,52] = -3.43902692373834
        struct[0].Gy_ini[49,53] = 4.15674076882530
        struct[0].Gy_ini[50,14] = 0.282682270461039
        struct[0].Gy_ini[50,15] = 1.69609362276623
        struct[0].Gy_ini[50,16] = -0.282682270461039
        struct[0].Gy_ini[50,17] = -1.69609362276623
        struct[0].Gy_ini[50,30] = -2.07555506909734
        struct[0].Gy_ini[50,31] = -1.70260376761891
        struct[0].Gy_ini[50,32] = 7.26444274184068
        struct[0].Gy_ini[50,33] = 5.95911318666618
        struct[0].Gy_ini[50,34] = -2.07555506909734
        struct[0].Gy_ini[50,35] = -1.70260376761891
        struct[0].Gy_ini[50,42] = -2.07555506909734
        struct[0].Gy_ini[50,43] = -1.70260376761891
        struct[0].Gy_ini[50,44] = 7.26444274184068
        struct[0].Gy_ini[50,45] = 5.95911318666618
        struct[0].Gy_ini[50,46] = -2.07555506909734
        struct[0].Gy_ini[50,47] = -1.70260376761891
        struct[0].Gy_ini[50,48] = 4.15674076882530
        struct[0].Gy_ini[50,49] = 3.43902692373834
        struct[0].Gy_ini[50,50] = -14.5401467449426
        struct[0].Gy_ini[50,51] = -11.9857049291081
        struct[0].Gy_ini[50,52] = 4.15674076882531
        struct[0].Gy_ini[50,53] = 3.43902692373834
        struct[0].Gy_ini[51,14] = -1.69609362276623
        struct[0].Gy_ini[51,15] = 0.282682270461039
        struct[0].Gy_ini[51,16] = 1.69609362276623
        struct[0].Gy_ini[51,17] = -0.282682270461039
        struct[0].Gy_ini[51,30] = 1.70260376761891
        struct[0].Gy_ini[51,31] = -2.07555506909734
        struct[0].Gy_ini[51,32] = -5.95911318666618
        struct[0].Gy_ini[51,33] = 7.26444274184068
        struct[0].Gy_ini[51,34] = 1.70260376761891
        struct[0].Gy_ini[51,35] = -2.07555506909734
        struct[0].Gy_ini[51,42] = 1.70260376761891
        struct[0].Gy_ini[51,43] = -2.07555506909734
        struct[0].Gy_ini[51,44] = -5.95911318666618
        struct[0].Gy_ini[51,45] = 7.26444274184068
        struct[0].Gy_ini[51,46] = 1.70260376761891
        struct[0].Gy_ini[51,47] = -2.07555506909734
        struct[0].Gy_ini[51,48] = -3.43902692373834
        struct[0].Gy_ini[51,49] = 4.15674076882530
        struct[0].Gy_ini[51,50] = 11.9857049291081
        struct[0].Gy_ini[51,51] = -14.5401467449426
        struct[0].Gy_ini[51,52] = -3.43902692373834
        struct[0].Gy_ini[51,53] = 4.15674076882531
        struct[0].Gy_ini[52,12] = -0.282682270461039
        struct[0].Gy_ini[52,13] = -1.69609362276623
        struct[0].Gy_ini[52,16] = 0.282682270461039
        struct[0].Gy_ini[52,17] = 1.69609362276623
        struct[0].Gy_ini[52,30] = -2.07555506909734
        struct[0].Gy_ini[52,31] = -1.70260376761891
        struct[0].Gy_ini[52,32] = -2.07555506909734
        struct[0].Gy_ini[52,33] = -1.70260376761891
        struct[0].Gy_ini[52,34] = 7.26444274184068
        struct[0].Gy_ini[52,35] = 5.95911318666618
        struct[0].Gy_ini[52,42] = -2.07555506909734
        struct[0].Gy_ini[52,43] = -1.70260376761891
        struct[0].Gy_ini[52,44] = -2.07555506909734
        struct[0].Gy_ini[52,45] = -1.70260376761891
        struct[0].Gy_ini[52,46] = 7.26444274184068
        struct[0].Gy_ini[52,47] = 5.95911318666618
        struct[0].Gy_ini[52,48] = 4.15674076882530
        struct[0].Gy_ini[52,49] = 3.43902692373834
        struct[0].Gy_ini[52,50] = 4.15674076882531
        struct[0].Gy_ini[52,51] = 3.43902692373834
        struct[0].Gy_ini[52,52] = -14.5401467449426
        struct[0].Gy_ini[52,53] = -11.9857049291081
        struct[0].Gy_ini[53,12] = 1.69609362276623
        struct[0].Gy_ini[53,13] = -0.282682270461039
        struct[0].Gy_ini[53,16] = -1.69609362276623
        struct[0].Gy_ini[53,17] = 0.282682270461039
        struct[0].Gy_ini[53,30] = 1.70260376761891
        struct[0].Gy_ini[53,31] = -2.07555506909734
        struct[0].Gy_ini[53,32] = 1.70260376761891
        struct[0].Gy_ini[53,33] = -2.07555506909734
        struct[0].Gy_ini[53,34] = -5.95911318666618
        struct[0].Gy_ini[53,35] = 7.26444274184068
        struct[0].Gy_ini[53,42] = 1.70260376761891
        struct[0].Gy_ini[53,43] = -2.07555506909734
        struct[0].Gy_ini[53,44] = 1.70260376761891
        struct[0].Gy_ini[53,45] = -2.07555506909734
        struct[0].Gy_ini[53,46] = -5.95911318666618
        struct[0].Gy_ini[53,47] = 7.26444274184068
        struct[0].Gy_ini[53,48] = -3.43902692373834
        struct[0].Gy_ini[53,49] = 4.15674076882530
        struct[0].Gy_ini[53,50] = -3.43902692373834
        struct[0].Gy_ini[53,51] = 4.15674076882531
        struct[0].Gy_ini[53,52] = 11.9857049291081
        struct[0].Gy_ini[53,53] = -14.5401467449426
        struct[0].Gy_ini[54,18] = 0.282682270461039
        struct[0].Gy_ini[54,19] = 1.69609362276623
        struct[0].Gy_ini[54,20] = -0.282682270461039
        struct[0].Gy_ini[54,21] = -1.69609362276623
        struct[0].Gy_ini[54,30] = 181.611068546017
        struct[0].Gy_ini[54,31] = 148.977829666654
        struct[0].Gy_ini[54,32] = -51.8888767274334
        struct[0].Gy_ini[54,33] = -42.5650941904727
        struct[0].Gy_ini[54,34] = -51.8888767274334
        struct[0].Gy_ini[54,35] = -42.5650941904727
        struct[0].Gy_ini[54,54] = -181.622329807278
        struct[0].Gy_ini[54,55] = -149.045395453986
        struct[0].Gy_ini[54,56] = 51.8945073580641
        struct[0].Gy_ini[54,57] = 42.5988786863508
        struct[0].Gy_ini[54,58] = 51.8945073580640
        struct[0].Gy_ini[54,59] = 42.5988786863508
        struct[0].Gy_ini[55,18] = -1.69609362276623
        struct[0].Gy_ini[55,19] = 0.282682270461039
        struct[0].Gy_ini[55,20] = 1.69609362276623
        struct[0].Gy_ini[55,21] = -0.282682270461039
        struct[0].Gy_ini[55,30] = -148.977829666654
        struct[0].Gy_ini[55,31] = 181.611068546017
        struct[0].Gy_ini[55,32] = 42.5650941904727
        struct[0].Gy_ini[55,33] = -51.8888767274334
        struct[0].Gy_ini[55,34] = 42.5650941904727
        struct[0].Gy_ini[55,35] = -51.8888767274334
        struct[0].Gy_ini[55,54] = 149.045395453986
        struct[0].Gy_ini[55,55] = -181.622329807278
        struct[0].Gy_ini[55,56] = -42.5988786863508
        struct[0].Gy_ini[55,57] = 51.8945073580641
        struct[0].Gy_ini[55,58] = -42.5988786863508
        struct[0].Gy_ini[55,59] = 51.8945073580640
        struct[0].Gy_ini[56,20] = 0.282682270461039
        struct[0].Gy_ini[56,21] = 1.69609362276623
        struct[0].Gy_ini[56,22] = -0.282682270461039
        struct[0].Gy_ini[56,23] = -1.69609362276623
        struct[0].Gy_ini[56,30] = -51.8888767274334
        struct[0].Gy_ini[56,31] = -42.5650941904727
        struct[0].Gy_ini[56,32] = 181.611068546017
        struct[0].Gy_ini[56,33] = 148.977829666654
        struct[0].Gy_ini[56,34] = -51.8888767274335
        struct[0].Gy_ini[56,35] = -42.5650941904727
        struct[0].Gy_ini[56,54] = 51.8945073580641
        struct[0].Gy_ini[56,55] = 42.5988786863508
        struct[0].Gy_ini[56,56] = -181.622329807278
        struct[0].Gy_ini[56,57] = -149.045395453986
        struct[0].Gy_ini[56,58] = 51.8945073580641
        struct[0].Gy_ini[56,59] = 42.5988786863508
        struct[0].Gy_ini[57,20] = -1.69609362276623
        struct[0].Gy_ini[57,21] = 0.282682270461039
        struct[0].Gy_ini[57,22] = 1.69609362276623
        struct[0].Gy_ini[57,23] = -0.282682270461039
        struct[0].Gy_ini[57,30] = 42.5650941904727
        struct[0].Gy_ini[57,31] = -51.8888767274334
        struct[0].Gy_ini[57,32] = -148.977829666654
        struct[0].Gy_ini[57,33] = 181.611068546017
        struct[0].Gy_ini[57,34] = 42.5650941904727
        struct[0].Gy_ini[57,35] = -51.8888767274335
        struct[0].Gy_ini[57,54] = -42.5988786863508
        struct[0].Gy_ini[57,55] = 51.8945073580641
        struct[0].Gy_ini[57,56] = 149.045395453986
        struct[0].Gy_ini[57,57] = -181.622329807278
        struct[0].Gy_ini[57,58] = -42.5988786863508
        struct[0].Gy_ini[57,59] = 51.8945073580641
        struct[0].Gy_ini[58,18] = -0.282682270461039
        struct[0].Gy_ini[58,19] = -1.69609362276623
        struct[0].Gy_ini[58,22] = 0.282682270461039
        struct[0].Gy_ini[58,23] = 1.69609362276623
        struct[0].Gy_ini[58,30] = -51.8888767274334
        struct[0].Gy_ini[58,31] = -42.5650941904727
        struct[0].Gy_ini[58,32] = -51.8888767274335
        struct[0].Gy_ini[58,33] = -42.5650941904727
        struct[0].Gy_ini[58,34] = 181.611068546017
        struct[0].Gy_ini[58,35] = 148.977829666654
        struct[0].Gy_ini[58,54] = 51.8945073580641
        struct[0].Gy_ini[58,55] = 42.5988786863508
        struct[0].Gy_ini[58,56] = 51.8945073580641
        struct[0].Gy_ini[58,57] = 42.5988786863508
        struct[0].Gy_ini[58,58] = -181.622329807278
        struct[0].Gy_ini[58,59] = -149.045395453986
        struct[0].Gy_ini[59,18] = 1.69609362276623
        struct[0].Gy_ini[59,19] = -0.282682270461039
        struct[0].Gy_ini[59,22] = -1.69609362276623
        struct[0].Gy_ini[59,23] = 0.282682270461039
        struct[0].Gy_ini[59,30] = 42.5650941904727
        struct[0].Gy_ini[59,31] = -51.8888767274334
        struct[0].Gy_ini[59,32] = 42.5650941904727
        struct[0].Gy_ini[59,33] = -51.8888767274335
        struct[0].Gy_ini[59,34] = -148.977829666654
        struct[0].Gy_ini[59,35] = 181.611068546017
        struct[0].Gy_ini[59,54] = -42.5988786863508
        struct[0].Gy_ini[59,55] = 51.8945073580641
        struct[0].Gy_ini[59,56] = -42.5988786863508
        struct[0].Gy_ini[59,57] = 51.8945073580641
        struct[0].Gy_ini[59,58] = 149.045395453986
        struct[0].Gy_ini[59,59] = -181.622329807278
        struct[0].Gy_ini[60,36] = 7.26444274184068
        struct[0].Gy_ini[60,37] = 5.95911318666618
        struct[0].Gy_ini[60,38] = -2.07555506909734
        struct[0].Gy_ini[60,39] = -1.70260376761891
        struct[0].Gy_ini[60,40] = -2.07555506909734
        struct[0].Gy_ini[60,41] = -1.70260376761891
        struct[0].Gy_ini[60,42] = -7.26444274184068
        struct[0].Gy_ini[60,43] = -5.95911318666618
        struct[0].Gy_ini[60,44] = 2.07555506909734
        struct[0].Gy_ini[60,45] = 1.70260376761891
        struct[0].Gy_ini[60,46] = 2.07555506909734
        struct[0].Gy_ini[60,47] = 1.70260376761891
        struct[0].Gy_ini[60,60] = -1
        struct[0].Gy_ini[61,36] = -5.95911318666618
        struct[0].Gy_ini[61,37] = 7.26444274184068
        struct[0].Gy_ini[61,38] = 1.70260376761891
        struct[0].Gy_ini[61,39] = -2.07555506909734
        struct[0].Gy_ini[61,40] = 1.70260376761891
        struct[0].Gy_ini[61,41] = -2.07555506909734
        struct[0].Gy_ini[61,42] = 5.95911318666618
        struct[0].Gy_ini[61,43] = -7.26444274184068
        struct[0].Gy_ini[61,44] = -1.70260376761891
        struct[0].Gy_ini[61,45] = 2.07555506909734
        struct[0].Gy_ini[61,46] = -1.70260376761891
        struct[0].Gy_ini[61,47] = 2.07555506909734
        struct[0].Gy_ini[61,61] = -1
        struct[0].Gy_ini[62,36] = -2.07555506909734
        struct[0].Gy_ini[62,37] = -1.70260376761891
        struct[0].Gy_ini[62,38] = 7.26444274184068
        struct[0].Gy_ini[62,39] = 5.95911318666618
        struct[0].Gy_ini[62,40] = -2.07555506909734
        struct[0].Gy_ini[62,41] = -1.70260376761891
        struct[0].Gy_ini[62,42] = 2.07555506909734
        struct[0].Gy_ini[62,43] = 1.70260376761891
        struct[0].Gy_ini[62,44] = -7.26444274184068
        struct[0].Gy_ini[62,45] = -5.95911318666618
        struct[0].Gy_ini[62,46] = 2.07555506909734
        struct[0].Gy_ini[62,47] = 1.70260376761891
        struct[0].Gy_ini[62,62] = -1
        struct[0].Gy_ini[63,36] = 1.70260376761891
        struct[0].Gy_ini[63,37] = -2.07555506909734
        struct[0].Gy_ini[63,38] = -5.95911318666618
        struct[0].Gy_ini[63,39] = 7.26444274184068
        struct[0].Gy_ini[63,40] = 1.70260376761891
        struct[0].Gy_ini[63,41] = -2.07555506909734
        struct[0].Gy_ini[63,42] = -1.70260376761891
        struct[0].Gy_ini[63,43] = 2.07555506909734
        struct[0].Gy_ini[63,44] = 5.95911318666618
        struct[0].Gy_ini[63,45] = -7.26444274184068
        struct[0].Gy_ini[63,46] = -1.70260376761891
        struct[0].Gy_ini[63,47] = 2.07555506909734
        struct[0].Gy_ini[63,63] = -1
        struct[0].Gy_ini[64,36] = -2.07555506909734
        struct[0].Gy_ini[64,37] = -1.70260376761891
        struct[0].Gy_ini[64,38] = -2.07555506909734
        struct[0].Gy_ini[64,39] = -1.70260376761891
        struct[0].Gy_ini[64,40] = 7.26444274184068
        struct[0].Gy_ini[64,41] = 5.95911318666618
        struct[0].Gy_ini[64,42] = 2.07555506909734
        struct[0].Gy_ini[64,43] = 1.70260376761891
        struct[0].Gy_ini[64,44] = 2.07555506909734
        struct[0].Gy_ini[64,45] = 1.70260376761891
        struct[0].Gy_ini[64,46] = -7.26444274184068
        struct[0].Gy_ini[64,47] = -5.95911318666618
        struct[0].Gy_ini[64,64] = -1
        struct[0].Gy_ini[65,36] = 1.70260376761891
        struct[0].Gy_ini[65,37] = -2.07555506909734
        struct[0].Gy_ini[65,38] = 1.70260376761891
        struct[0].Gy_ini[65,39] = -2.07555506909734
        struct[0].Gy_ini[65,40] = -5.95911318666618
        struct[0].Gy_ini[65,41] = 7.26444274184068
        struct[0].Gy_ini[65,42] = -1.70260376761891
        struct[0].Gy_ini[65,43] = 2.07555506909734
        struct[0].Gy_ini[65,44] = -1.70260376761891
        struct[0].Gy_ini[65,45] = 2.07555506909734
        struct[0].Gy_ini[65,46] = 5.95911318666618
        struct[0].Gy_ini[65,47] = -7.26444274184068
        struct[0].Gy_ini[65,65] = -1
        struct[0].Gy_ini[66,42] = 7.26444274184068
        struct[0].Gy_ini[66,43] = 5.95911318666618
        struct[0].Gy_ini[66,44] = -2.07555506909734
        struct[0].Gy_ini[66,45] = -1.70260376761891
        struct[0].Gy_ini[66,46] = -2.07555506909734
        struct[0].Gy_ini[66,47] = -1.70260376761891
        struct[0].Gy_ini[66,48] = -7.26444274184068
        struct[0].Gy_ini[66,49] = -5.95911318666618
        struct[0].Gy_ini[66,50] = 2.07555506909734
        struct[0].Gy_ini[66,51] = 1.70260376761891
        struct[0].Gy_ini[66,52] = 2.07555506909734
        struct[0].Gy_ini[66,53] = 1.70260376761891
        struct[0].Gy_ini[66,66] = -1
        struct[0].Gy_ini[67,42] = -5.95911318666618
        struct[0].Gy_ini[67,43] = 7.26444274184068
        struct[0].Gy_ini[67,44] = 1.70260376761891
        struct[0].Gy_ini[67,45] = -2.07555506909734
        struct[0].Gy_ini[67,46] = 1.70260376761891
        struct[0].Gy_ini[67,47] = -2.07555506909734
        struct[0].Gy_ini[67,48] = 5.95911318666618
        struct[0].Gy_ini[67,49] = -7.26444274184068
        struct[0].Gy_ini[67,50] = -1.70260376761891
        struct[0].Gy_ini[67,51] = 2.07555506909734
        struct[0].Gy_ini[67,52] = -1.70260376761891
        struct[0].Gy_ini[67,53] = 2.07555506909734
        struct[0].Gy_ini[67,67] = -1
        struct[0].Gy_ini[68,42] = -2.07555506909734
        struct[0].Gy_ini[68,43] = -1.70260376761891
        struct[0].Gy_ini[68,44] = 7.26444274184068
        struct[0].Gy_ini[68,45] = 5.95911318666618
        struct[0].Gy_ini[68,46] = -2.07555506909734
        struct[0].Gy_ini[68,47] = -1.70260376761891
        struct[0].Gy_ini[68,48] = 2.07555506909734
        struct[0].Gy_ini[68,49] = 1.70260376761891
        struct[0].Gy_ini[68,50] = -7.26444274184068
        struct[0].Gy_ini[68,51] = -5.95911318666618
        struct[0].Gy_ini[68,52] = 2.07555506909734
        struct[0].Gy_ini[68,53] = 1.70260376761891
        struct[0].Gy_ini[68,68] = -1
        struct[0].Gy_ini[69,42] = 1.70260376761891
        struct[0].Gy_ini[69,43] = -2.07555506909734
        struct[0].Gy_ini[69,44] = -5.95911318666618
        struct[0].Gy_ini[69,45] = 7.26444274184068
        struct[0].Gy_ini[69,46] = 1.70260376761891
        struct[0].Gy_ini[69,47] = -2.07555506909734
        struct[0].Gy_ini[69,48] = -1.70260376761891
        struct[0].Gy_ini[69,49] = 2.07555506909734
        struct[0].Gy_ini[69,50] = 5.95911318666618
        struct[0].Gy_ini[69,51] = -7.26444274184068
        struct[0].Gy_ini[69,52] = -1.70260376761891
        struct[0].Gy_ini[69,53] = 2.07555506909734
        struct[0].Gy_ini[69,69] = -1
        struct[0].Gy_ini[70,42] = -2.07555506909734
        struct[0].Gy_ini[70,43] = -1.70260376761891
        struct[0].Gy_ini[70,44] = -2.07555506909734
        struct[0].Gy_ini[70,45] = -1.70260376761891
        struct[0].Gy_ini[70,46] = 7.26444274184068
        struct[0].Gy_ini[70,47] = 5.95911318666618
        struct[0].Gy_ini[70,48] = 2.07555506909734
        struct[0].Gy_ini[70,49] = 1.70260376761891
        struct[0].Gy_ini[70,50] = 2.07555506909734
        struct[0].Gy_ini[70,51] = 1.70260376761891
        struct[0].Gy_ini[70,52] = -7.26444274184068
        struct[0].Gy_ini[70,53] = -5.95911318666618
        struct[0].Gy_ini[70,70] = -1
        struct[0].Gy_ini[71,42] = 1.70260376761891
        struct[0].Gy_ini[71,43] = -2.07555506909734
        struct[0].Gy_ini[71,44] = 1.70260376761891
        struct[0].Gy_ini[71,45] = -2.07555506909734
        struct[0].Gy_ini[71,46] = -5.95911318666618
        struct[0].Gy_ini[71,47] = 7.26444274184068
        struct[0].Gy_ini[71,48] = -1.70260376761891
        struct[0].Gy_ini[71,49] = 2.07555506909734
        struct[0].Gy_ini[71,50] = -1.70260376761891
        struct[0].Gy_ini[71,51] = 2.07555506909734
        struct[0].Gy_ini[71,52] = 5.95911318666618
        struct[0].Gy_ini[71,53] = -7.26444274184068
        struct[0].Gy_ini[71,71] = -1
        struct[0].Gy_ini[72,30] = -7.26444274184068
        struct[0].Gy_ini[72,31] = -5.95911318666618
        struct[0].Gy_ini[72,32] = 2.07555506909734
        struct[0].Gy_ini[72,33] = 1.70260376761891
        struct[0].Gy_ini[72,34] = 2.07555506909734
        struct[0].Gy_ini[72,35] = 1.70260376761891
        struct[0].Gy_ini[72,48] = 7.26444274184068
        struct[0].Gy_ini[72,49] = 5.95911318666618
        struct[0].Gy_ini[72,50] = -2.07555506909734
        struct[0].Gy_ini[72,51] = -1.70260376761891
        struct[0].Gy_ini[72,52] = -2.07555506909734
        struct[0].Gy_ini[72,53] = -1.70260376761891
        struct[0].Gy_ini[72,72] = -1
        struct[0].Gy_ini[73,30] = 5.95911318666618
        struct[0].Gy_ini[73,31] = -7.26444274184068
        struct[0].Gy_ini[73,32] = -1.70260376761891
        struct[0].Gy_ini[73,33] = 2.07555506909734
        struct[0].Gy_ini[73,34] = -1.70260376761891
        struct[0].Gy_ini[73,35] = 2.07555506909734
        struct[0].Gy_ini[73,48] = -5.95911318666618
        struct[0].Gy_ini[73,49] = 7.26444274184068
        struct[0].Gy_ini[73,50] = 1.70260376761891
        struct[0].Gy_ini[73,51] = -2.07555506909734
        struct[0].Gy_ini[73,52] = 1.70260376761891
        struct[0].Gy_ini[73,53] = -2.07555506909734
        struct[0].Gy_ini[73,73] = -1
        struct[0].Gy_ini[74,30] = 2.07555506909734
        struct[0].Gy_ini[74,31] = 1.70260376761891
        struct[0].Gy_ini[74,32] = -7.26444274184068
        struct[0].Gy_ini[74,33] = -5.95911318666618
        struct[0].Gy_ini[74,34] = 2.07555506909734
        struct[0].Gy_ini[74,35] = 1.70260376761891
        struct[0].Gy_ini[74,48] = -2.07555506909734
        struct[0].Gy_ini[74,49] = -1.70260376761891
        struct[0].Gy_ini[74,50] = 7.26444274184068
        struct[0].Gy_ini[74,51] = 5.95911318666618
        struct[0].Gy_ini[74,52] = -2.07555506909734
        struct[0].Gy_ini[74,53] = -1.70260376761891
        struct[0].Gy_ini[74,74] = -1
        struct[0].Gy_ini[75,30] = -1.70260376761891
        struct[0].Gy_ini[75,31] = 2.07555506909734
        struct[0].Gy_ini[75,32] = 5.95911318666618
        struct[0].Gy_ini[75,33] = -7.26444274184068
        struct[0].Gy_ini[75,34] = -1.70260376761891
        struct[0].Gy_ini[75,35] = 2.07555506909734
        struct[0].Gy_ini[75,48] = 1.70260376761891
        struct[0].Gy_ini[75,49] = -2.07555506909734
        struct[0].Gy_ini[75,50] = -5.95911318666618
        struct[0].Gy_ini[75,51] = 7.26444274184068
        struct[0].Gy_ini[75,52] = 1.70260376761891
        struct[0].Gy_ini[75,53] = -2.07555506909734
        struct[0].Gy_ini[75,75] = -1
        struct[0].Gy_ini[76,30] = 2.07555506909734
        struct[0].Gy_ini[76,31] = 1.70260376761891
        struct[0].Gy_ini[76,32] = 2.07555506909734
        struct[0].Gy_ini[76,33] = 1.70260376761891
        struct[0].Gy_ini[76,34] = -7.26444274184068
        struct[0].Gy_ini[76,35] = -5.95911318666618
        struct[0].Gy_ini[76,48] = -2.07555506909734
        struct[0].Gy_ini[76,49] = -1.70260376761891
        struct[0].Gy_ini[76,50] = -2.07555506909734
        struct[0].Gy_ini[76,51] = -1.70260376761891
        struct[0].Gy_ini[76,52] = 7.26444274184068
        struct[0].Gy_ini[76,53] = 5.95911318666618
        struct[0].Gy_ini[76,76] = -1
        struct[0].Gy_ini[77,30] = -1.70260376761891
        struct[0].Gy_ini[77,31] = 2.07555506909734
        struct[0].Gy_ini[77,32] = -1.70260376761891
        struct[0].Gy_ini[77,33] = 2.07555506909734
        struct[0].Gy_ini[77,34] = 5.95911318666618
        struct[0].Gy_ini[77,35] = -7.26444274184068
        struct[0].Gy_ini[77,48] = 1.70260376761891
        struct[0].Gy_ini[77,49] = -2.07555506909734
        struct[0].Gy_ini[77,50] = 1.70260376761891
        struct[0].Gy_ini[77,51] = -2.07555506909734
        struct[0].Gy_ini[77,52] = -5.95911318666618
        struct[0].Gy_ini[77,53] = 7.26444274184068
        struct[0].Gy_ini[77,77] = -1
        struct[0].Gy_ini[78,30] = -181.611068546017
        struct[0].Gy_ini[78,31] = -148.977829666654
        struct[0].Gy_ini[78,32] = 51.8888767274334
        struct[0].Gy_ini[78,33] = 42.5650941904727
        struct[0].Gy_ini[78,34] = 51.8888767274334
        struct[0].Gy_ini[78,35] = 42.5650941904727
        struct[0].Gy_ini[78,54] = 181.611068546017
        struct[0].Gy_ini[78,55] = 148.977829666654
        struct[0].Gy_ini[78,56] = -51.8888767274334
        struct[0].Gy_ini[78,57] = -42.5650941904727
        struct[0].Gy_ini[78,58] = -51.8888767274334
        struct[0].Gy_ini[78,59] = -42.5650941904727
        struct[0].Gy_ini[78,78] = -1
        struct[0].Gy_ini[79,30] = 148.977829666654
        struct[0].Gy_ini[79,31] = -181.611068546017
        struct[0].Gy_ini[79,32] = -42.5650941904727
        struct[0].Gy_ini[79,33] = 51.8888767274334
        struct[0].Gy_ini[79,34] = -42.5650941904727
        struct[0].Gy_ini[79,35] = 51.8888767274334
        struct[0].Gy_ini[79,54] = -148.977829666654
        struct[0].Gy_ini[79,55] = 181.611068546017
        struct[0].Gy_ini[79,56] = 42.5650941904727
        struct[0].Gy_ini[79,57] = -51.8888767274334
        struct[0].Gy_ini[79,58] = 42.5650941904727
        struct[0].Gy_ini[79,59] = -51.8888767274334
        struct[0].Gy_ini[79,79] = -1
        struct[0].Gy_ini[80,30] = 51.8888767274334
        struct[0].Gy_ini[80,31] = 42.5650941904727
        struct[0].Gy_ini[80,32] = -181.611068546017
        struct[0].Gy_ini[80,33] = -148.977829666654
        struct[0].Gy_ini[80,34] = 51.8888767274335
        struct[0].Gy_ini[80,35] = 42.5650941904727
        struct[0].Gy_ini[80,54] = -51.8888767274334
        struct[0].Gy_ini[80,55] = -42.5650941904727
        struct[0].Gy_ini[80,56] = 181.611068546017
        struct[0].Gy_ini[80,57] = 148.977829666654
        struct[0].Gy_ini[80,58] = -51.8888767274335
        struct[0].Gy_ini[80,59] = -42.5650941904727
        struct[0].Gy_ini[80,80] = -1
        struct[0].Gy_ini[81,30] = -42.5650941904727
        struct[0].Gy_ini[81,31] = 51.8888767274334
        struct[0].Gy_ini[81,32] = 148.977829666654
        struct[0].Gy_ini[81,33] = -181.611068546017
        struct[0].Gy_ini[81,34] = -42.5650941904727
        struct[0].Gy_ini[81,35] = 51.8888767274335
        struct[0].Gy_ini[81,54] = 42.5650941904727
        struct[0].Gy_ini[81,55] = -51.8888767274334
        struct[0].Gy_ini[81,56] = -148.977829666654
        struct[0].Gy_ini[81,57] = 181.611068546017
        struct[0].Gy_ini[81,58] = 42.5650941904727
        struct[0].Gy_ini[81,59] = -51.8888767274335
        struct[0].Gy_ini[81,81] = -1
        struct[0].Gy_ini[82,30] = 51.8888767274334
        struct[0].Gy_ini[82,31] = 42.5650941904727
        struct[0].Gy_ini[82,32] = 51.8888767274335
        struct[0].Gy_ini[82,33] = 42.5650941904727
        struct[0].Gy_ini[82,34] = -181.611068546017
        struct[0].Gy_ini[82,35] = -148.977829666654
        struct[0].Gy_ini[82,54] = -51.8888767274334
        struct[0].Gy_ini[82,55] = -42.5650941904727
        struct[0].Gy_ini[82,56] = -51.8888767274335
        struct[0].Gy_ini[82,57] = -42.5650941904727
        struct[0].Gy_ini[82,58] = 181.611068546017
        struct[0].Gy_ini[82,59] = 148.977829666654
        struct[0].Gy_ini[82,82] = -1
        struct[0].Gy_ini[83,30] = -42.5650941904727
        struct[0].Gy_ini[83,31] = 51.8888767274334
        struct[0].Gy_ini[83,32] = -42.5650941904727
        struct[0].Gy_ini[83,33] = 51.8888767274335
        struct[0].Gy_ini[83,34] = 148.977829666654
        struct[0].Gy_ini[83,35] = -181.611068546017
        struct[0].Gy_ini[83,54] = 42.5650941904727
        struct[0].Gy_ini[83,55] = -51.8888767274334
        struct[0].Gy_ini[83,56] = 42.5650941904727
        struct[0].Gy_ini[83,57] = -51.8888767274335
        struct[0].Gy_ini[83,58] = -148.977829666654
        struct[0].Gy_ini[83,59] = 181.611068546017
        struct[0].Gy_ini[83,83] = -1
        struct[0].Gy_ini[84,24] = 0.0241740531829170
        struct[0].Gy_ini[84,25] = 0.0402900886381950
        struct[0].Gy_ini[84,26] = -4.31760362252812E-18
        struct[0].Gy_ini[84,27] = 4.66248501556824E-18
        struct[0].Gy_ini[84,28] = -3.49608108880335E-18
        struct[0].Gy_ini[84,29] = 4.19816664496737E-18
        struct[0].Gy_ini[84,84] = -1
        struct[0].Gy_ini[85,24] = -0.0402900886381950
        struct[0].Gy_ini[85,25] = 0.0241740531829170
        struct[0].Gy_ini[85,26] = -4.66248501556824E-18
        struct[0].Gy_ini[85,27] = -4.31760362252812E-18
        struct[0].Gy_ini[85,28] = -4.19816664496737E-18
        struct[0].Gy_ini[85,29] = -3.49608108880335E-18
        struct[0].Gy_ini[85,85] = -1
        struct[0].Gy_ini[86,24] = -2.07254761002657E-18
        struct[0].Gy_ini[86,25] = 6.30775359573304E-19
        struct[0].Gy_ini[86,26] = 0.0241740531829170
        struct[0].Gy_ini[86,27] = 0.0402900886381950
        struct[0].Gy_ini[86,28] = -1.78419315993592E-17
        struct[0].Gy_ini[86,29] = 9.01107656533306E-19
        struct[0].Gy_ini[86,86] = -1
        struct[0].Gy_ini[87,24] = -6.30775359573304E-19
        struct[0].Gy_ini[87,25] = -2.07254761002657E-18
        struct[0].Gy_ini[87,26] = -0.0402900886381950
        struct[0].Gy_ini[87,27] = 0.0241740531829170
        struct[0].Gy_ini[87,28] = -9.01107656533306E-19
        struct[0].Gy_ini[87,29] = -1.78419315993592E-17
        struct[0].Gy_ini[87,87] = -1
        struct[0].Gy_ini[88,24] = -1.35166148479994E-18
        struct[0].Gy_ini[88,25] = -7.20886125226632E-19
        struct[0].Gy_ini[88,26] = -1.71210454741325E-17
        struct[0].Gy_ini[88,27] = -4.50553828266631E-19
        struct[0].Gy_ini[88,28] = 0.0241740531829170
        struct[0].Gy_ini[88,29] = 0.0402900886381950
        struct[0].Gy_ini[88,88] = -1
        struct[0].Gy_ini[89,24] = 7.20886125226632E-19
        struct[0].Gy_ini[89,25] = -1.35166148479994E-18
        struct[0].Gy_ini[89,26] = 4.50553828266631E-19
        struct[0].Gy_ini[89,27] = -1.71210454741325E-17
        struct[0].Gy_ini[89,28] = -0.0402900886381950
        struct[0].Gy_ini[89,29] = 0.0241740531829170
        struct[0].Gy_ini[89,89] = -1
        struct[0].Gy_ini[90,0] = i_W1lv_a_r
        struct[0].Gy_ini[90,1] = i_W1lv_a_i
        struct[0].Gy_ini[90,90] = v_W1lv_a_r
        struct[0].Gy_ini[90,91] = v_W1lv_a_i
        struct[0].Gy_ini[91,2] = i_W1lv_b_r
        struct[0].Gy_ini[91,3] = i_W1lv_b_i
        struct[0].Gy_ini[91,92] = v_W1lv_b_r
        struct[0].Gy_ini[91,93] = v_W1lv_b_i
        struct[0].Gy_ini[92,4] = i_W1lv_c_r
        struct[0].Gy_ini[92,5] = i_W1lv_c_i
        struct[0].Gy_ini[92,94] = v_W1lv_c_r
        struct[0].Gy_ini[92,95] = v_W1lv_c_i
        struct[0].Gy_ini[93,0] = -i_W1lv_a_i
        struct[0].Gy_ini[93,1] = i_W1lv_a_r
        struct[0].Gy_ini[93,90] = v_W1lv_a_i
        struct[0].Gy_ini[93,91] = -v_W1lv_a_r
        struct[0].Gy_ini[94,2] = -i_W1lv_b_i
        struct[0].Gy_ini[94,3] = i_W1lv_b_r
        struct[0].Gy_ini[94,92] = v_W1lv_b_i
        struct[0].Gy_ini[94,93] = -v_W1lv_b_r
        struct[0].Gy_ini[95,4] = -i_W1lv_c_i
        struct[0].Gy_ini[95,5] = i_W1lv_c_r
        struct[0].Gy_ini[95,94] = v_W1lv_c_i
        struct[0].Gy_ini[95,95] = -v_W1lv_c_r
        struct[0].Gy_ini[96,0] = 1.0*v_W1lv_a_r*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)/V_base_W1lv
        struct[0].Gy_ini[96,1] = 1.0*v_W1lv_a_i*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)/V_base_W1lv
        struct[0].Gy_ini[96,96] = -1
        struct[0].Gy_ini[97,36] = 1.0*v_W1mv_a_r*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)/V_base_W1mv
        struct[0].Gy_ini[97,37] = 1.0*v_W1mv_a_i*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)/V_base_W1mv
        struct[0].Gy_ini[97,97] = -1
        struct[0].Gy_ini[98,96] = K_p_v_W1lv*(u_ctrl_v_W1lv - 1.0)
        struct[0].Gy_ini[98,97] = -K_p_v_W1lv*u_ctrl_v_W1lv
        struct[0].Gy_ini[98,98] = -1
        struct[0].Gy_ini[99,58] = 1.0*S_base_W1lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)]))
        struct[0].Gy_ini[99,59] = 1.0*S_base_W1lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)]))
        struct[0].Gy_ini[99,98] = S_base_W1lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W1lv < i_reac_ref_W1lv) | (I_max_W1lv < -i_reac_ref_W1lv)), (1, True)]))
        struct[0].Gy_ini[99,99] = -1
        struct[0].Gy_ini[100,6] = i_W2lv_a_r
        struct[0].Gy_ini[100,7] = i_W2lv_a_i
        struct[0].Gy_ini[100,100] = v_W2lv_a_r
        struct[0].Gy_ini[100,101] = v_W2lv_a_i
        struct[0].Gy_ini[101,8] = i_W2lv_b_r
        struct[0].Gy_ini[101,9] = i_W2lv_b_i
        struct[0].Gy_ini[101,102] = v_W2lv_b_r
        struct[0].Gy_ini[101,103] = v_W2lv_b_i
        struct[0].Gy_ini[102,10] = i_W2lv_c_r
        struct[0].Gy_ini[102,11] = i_W2lv_c_i
        struct[0].Gy_ini[102,104] = v_W2lv_c_r
        struct[0].Gy_ini[102,105] = v_W2lv_c_i
        struct[0].Gy_ini[103,6] = -i_W2lv_a_i
        struct[0].Gy_ini[103,7] = i_W2lv_a_r
        struct[0].Gy_ini[103,100] = v_W2lv_a_i
        struct[0].Gy_ini[103,101] = -v_W2lv_a_r
        struct[0].Gy_ini[104,8] = -i_W2lv_b_i
        struct[0].Gy_ini[104,9] = i_W2lv_b_r
        struct[0].Gy_ini[104,102] = v_W2lv_b_i
        struct[0].Gy_ini[104,103] = -v_W2lv_b_r
        struct[0].Gy_ini[105,10] = -i_W2lv_c_i
        struct[0].Gy_ini[105,11] = i_W2lv_c_r
        struct[0].Gy_ini[105,104] = v_W2lv_c_i
        struct[0].Gy_ini[105,105] = -v_W2lv_c_r
        struct[0].Gy_ini[106,6] = 1.0*v_W2lv_a_r*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)/V_base_W2lv
        struct[0].Gy_ini[106,7] = 1.0*v_W2lv_a_i*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)/V_base_W2lv
        struct[0].Gy_ini[106,106] = -1
        struct[0].Gy_ini[107,42] = 1.0*v_W2mv_a_r*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)/V_base_W2mv
        struct[0].Gy_ini[107,43] = 1.0*v_W2mv_a_i*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)/V_base_W2mv
        struct[0].Gy_ini[107,107] = -1
        struct[0].Gy_ini[108,106] = K_p_v_W2lv*(u_ctrl_v_W2lv - 1.0)
        struct[0].Gy_ini[108,107] = -K_p_v_W2lv*u_ctrl_v_W2lv
        struct[0].Gy_ini[108,108] = -1
        struct[0].Gy_ini[109,58] = 1.0*S_base_W2lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)]))
        struct[0].Gy_ini[109,59] = 1.0*S_base_W2lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)]))
        struct[0].Gy_ini[109,108] = S_base_W2lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W2lv < i_reac_ref_W2lv) | (I_max_W2lv < -i_reac_ref_W2lv)), (1, True)]))
        struct[0].Gy_ini[109,109] = -1
        struct[0].Gy_ini[110,12] = i_W3lv_a_r
        struct[0].Gy_ini[110,13] = i_W3lv_a_i
        struct[0].Gy_ini[110,110] = v_W3lv_a_r
        struct[0].Gy_ini[110,111] = v_W3lv_a_i
        struct[0].Gy_ini[111,14] = i_W3lv_b_r
        struct[0].Gy_ini[111,15] = i_W3lv_b_i
        struct[0].Gy_ini[111,112] = v_W3lv_b_r
        struct[0].Gy_ini[111,113] = v_W3lv_b_i
        struct[0].Gy_ini[112,16] = i_W3lv_c_r
        struct[0].Gy_ini[112,17] = i_W3lv_c_i
        struct[0].Gy_ini[112,114] = v_W3lv_c_r
        struct[0].Gy_ini[112,115] = v_W3lv_c_i
        struct[0].Gy_ini[113,12] = -i_W3lv_a_i
        struct[0].Gy_ini[113,13] = i_W3lv_a_r
        struct[0].Gy_ini[113,110] = v_W3lv_a_i
        struct[0].Gy_ini[113,111] = -v_W3lv_a_r
        struct[0].Gy_ini[114,14] = -i_W3lv_b_i
        struct[0].Gy_ini[114,15] = i_W3lv_b_r
        struct[0].Gy_ini[114,112] = v_W3lv_b_i
        struct[0].Gy_ini[114,113] = -v_W3lv_b_r
        struct[0].Gy_ini[115,16] = -i_W3lv_c_i
        struct[0].Gy_ini[115,17] = i_W3lv_c_r
        struct[0].Gy_ini[115,114] = v_W3lv_c_i
        struct[0].Gy_ini[115,115] = -v_W3lv_c_r
        struct[0].Gy_ini[116,12] = 1.0*v_W3lv_a_r*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)/V_base_W3lv
        struct[0].Gy_ini[116,13] = 1.0*v_W3lv_a_i*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)/V_base_W3lv
        struct[0].Gy_ini[116,116] = -1
        struct[0].Gy_ini[117,48] = 1.0*v_W3mv_a_r*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)/V_base_W3mv
        struct[0].Gy_ini[117,49] = 1.0*v_W3mv_a_i*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)/V_base_W3mv
        struct[0].Gy_ini[117,117] = -1
        struct[0].Gy_ini[118,116] = K_p_v_W3lv*(u_ctrl_v_W3lv - 1.0)
        struct[0].Gy_ini[118,117] = -K_p_v_W3lv*u_ctrl_v_W3lv
        struct[0].Gy_ini[118,118] = -1
        struct[0].Gy_ini[119,58] = 1.0*S_base_W3lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)]))
        struct[0].Gy_ini[119,59] = 1.0*S_base_W3lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)]))
        struct[0].Gy_ini[119,118] = S_base_W3lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W3lv < i_reac_ref_W3lv) | (I_max_W3lv < -i_reac_ref_W3lv)), (1, True)]))
        struct[0].Gy_ini[119,119] = -1
        struct[0].Gy_ini[120,18] = i_STlv_a_r
        struct[0].Gy_ini[120,19] = i_STlv_a_i
        struct[0].Gy_ini[120,120] = v_STlv_a_r
        struct[0].Gy_ini[120,121] = v_STlv_a_i
        struct[0].Gy_ini[121,20] = i_STlv_b_r
        struct[0].Gy_ini[121,21] = i_STlv_b_i
        struct[0].Gy_ini[121,122] = v_STlv_b_r
        struct[0].Gy_ini[121,123] = v_STlv_b_i
        struct[0].Gy_ini[122,22] = i_STlv_c_r
        struct[0].Gy_ini[122,23] = i_STlv_c_i
        struct[0].Gy_ini[122,124] = v_STlv_c_r
        struct[0].Gy_ini[122,125] = v_STlv_c_i
        struct[0].Gy_ini[123,18] = -i_STlv_a_i
        struct[0].Gy_ini[123,19] = i_STlv_a_r
        struct[0].Gy_ini[123,120] = v_STlv_a_i
        struct[0].Gy_ini[123,121] = -v_STlv_a_r
        struct[0].Gy_ini[124,20] = -i_STlv_b_i
        struct[0].Gy_ini[124,21] = i_STlv_b_r
        struct[0].Gy_ini[124,122] = v_STlv_b_i
        struct[0].Gy_ini[124,123] = -v_STlv_b_r
        struct[0].Gy_ini[125,22] = -i_STlv_c_i
        struct[0].Gy_ini[125,23] = i_STlv_c_r
        struct[0].Gy_ini[125,124] = v_STlv_c_i
        struct[0].Gy_ini[125,125] = -v_STlv_c_r
        struct[0].Gy_ini[126,18] = 1.0*v_STlv_a_r*(v_STlv_a_i**2 + v_STlv_a_r**2)**(-0.5)/V_base_STlv
        struct[0].Gy_ini[126,19] = 1.0*v_STlv_a_i*(v_STlv_a_i**2 + v_STlv_a_r**2)**(-0.5)/V_base_STlv
        struct[0].Gy_ini[126,126] = -1
        struct[0].Gy_ini[127,54] = 1.0*v_STmv_a_r*(v_STmv_a_i**2 + v_STmv_a_r**2)**(-0.5)/V_base_STmv
        struct[0].Gy_ini[127,55] = 1.0*v_STmv_a_i*(v_STmv_a_i**2 + v_STmv_a_r**2)**(-0.5)/V_base_STmv
        struct[0].Gy_ini[127,127] = -1
        struct[0].Gy_ini[128,126] = K_p_v_STlv*(u_ctrl_v_STlv - 1.0)
        struct[0].Gy_ini[128,127] = -K_p_v_STlv*u_ctrl_v_STlv
        struct[0].Gy_ini[128,128] = -1
        struct[0].Gy_ini[129,58] = 1.0*S_base_STlv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)]))
        struct[0].Gy_ini[129,59] = 1.0*S_base_STlv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)]))
        struct[0].Gy_ini[129,128] = S_base_STlv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_STlv < i_reac_ref_STlv) | (I_max_STlv < -i_reac_ref_STlv)), (1, True)]))
        struct[0].Gy_ini[129,129] = -1



def run_nn(t,struct,mode):

    # Parameters:
    u_ctrl_v_W1lv = struct[0].u_ctrl_v_W1lv
    K_p_v_W1lv = struct[0].K_p_v_W1lv
    K_i_v_W1lv = struct[0].K_i_v_W1lv
    V_base_W1lv = struct[0].V_base_W1lv
    V_base_W1mv = struct[0].V_base_W1mv
    S_base_W1lv = struct[0].S_base_W1lv
    I_max_W1lv = struct[0].I_max_W1lv
    u_ctrl_v_W2lv = struct[0].u_ctrl_v_W2lv
    K_p_v_W2lv = struct[0].K_p_v_W2lv
    K_i_v_W2lv = struct[0].K_i_v_W2lv
    V_base_W2lv = struct[0].V_base_W2lv
    V_base_W2mv = struct[0].V_base_W2mv
    S_base_W2lv = struct[0].S_base_W2lv
    I_max_W2lv = struct[0].I_max_W2lv
    u_ctrl_v_W3lv = struct[0].u_ctrl_v_W3lv
    K_p_v_W3lv = struct[0].K_p_v_W3lv
    K_i_v_W3lv = struct[0].K_i_v_W3lv
    V_base_W3lv = struct[0].V_base_W3lv
    V_base_W3mv = struct[0].V_base_W3mv
    S_base_W3lv = struct[0].S_base_W3lv
    I_max_W3lv = struct[0].I_max_W3lv
    u_ctrl_v_STlv = struct[0].u_ctrl_v_STlv
    K_p_v_STlv = struct[0].K_p_v_STlv
    K_i_v_STlv = struct[0].K_i_v_STlv
    V_base_STlv = struct[0].V_base_STlv
    V_base_STmv = struct[0].V_base_STmv
    S_base_STlv = struct[0].S_base_STlv
    I_max_STlv = struct[0].I_max_STlv
    
    # Inputs:
    v_GRID_a_r = struct[0].v_GRID_a_r
    v_GRID_a_i = struct[0].v_GRID_a_i
    v_GRID_b_r = struct[0].v_GRID_b_r
    v_GRID_b_i = struct[0].v_GRID_b_i
    v_GRID_c_r = struct[0].v_GRID_c_r
    v_GRID_c_i = struct[0].v_GRID_c_i
    i_POI_a_r = struct[0].i_POI_a_r
    i_POI_a_i = struct[0].i_POI_a_i
    i_POI_b_r = struct[0].i_POI_b_r
    i_POI_b_i = struct[0].i_POI_b_i
    i_POI_c_r = struct[0].i_POI_c_r
    i_POI_c_i = struct[0].i_POI_c_i
    i_POImv_a_r = struct[0].i_POImv_a_r
    i_POImv_a_i = struct[0].i_POImv_a_i
    i_POImv_b_r = struct[0].i_POImv_b_r
    i_POImv_b_i = struct[0].i_POImv_b_i
    i_POImv_c_r = struct[0].i_POImv_c_r
    i_POImv_c_i = struct[0].i_POImv_c_i
    i_W1mv_a_r = struct[0].i_W1mv_a_r
    i_W1mv_a_i = struct[0].i_W1mv_a_i
    i_W1mv_b_r = struct[0].i_W1mv_b_r
    i_W1mv_b_i = struct[0].i_W1mv_b_i
    i_W1mv_c_r = struct[0].i_W1mv_c_r
    i_W1mv_c_i = struct[0].i_W1mv_c_i
    i_W2mv_a_r = struct[0].i_W2mv_a_r
    i_W2mv_a_i = struct[0].i_W2mv_a_i
    i_W2mv_b_r = struct[0].i_W2mv_b_r
    i_W2mv_b_i = struct[0].i_W2mv_b_i
    i_W2mv_c_r = struct[0].i_W2mv_c_r
    i_W2mv_c_i = struct[0].i_W2mv_c_i
    i_W3mv_a_r = struct[0].i_W3mv_a_r
    i_W3mv_a_i = struct[0].i_W3mv_a_i
    i_W3mv_b_r = struct[0].i_W3mv_b_r
    i_W3mv_b_i = struct[0].i_W3mv_b_i
    i_W3mv_c_r = struct[0].i_W3mv_c_r
    i_W3mv_c_i = struct[0].i_W3mv_c_i
    i_STmv_a_r = struct[0].i_STmv_a_r
    i_STmv_a_i = struct[0].i_STmv_a_i
    i_STmv_b_r = struct[0].i_STmv_b_r
    i_STmv_b_i = struct[0].i_STmv_b_i
    i_STmv_c_r = struct[0].i_STmv_c_r
    i_STmv_c_i = struct[0].i_STmv_c_i
    p_ref_W1lv = struct[0].p_ref_W1lv
    T_pq_W1lv = struct[0].T_pq_W1lv
    v_loc_ref_W1lv = struct[0].v_loc_ref_W1lv
    Dv_r_W1lv = struct[0].Dv_r_W1lv
    Dq_r_W1lv = struct[0].Dq_r_W1lv
    p_ref_W2lv = struct[0].p_ref_W2lv
    T_pq_W2lv = struct[0].T_pq_W2lv
    v_loc_ref_W2lv = struct[0].v_loc_ref_W2lv
    Dv_r_W2lv = struct[0].Dv_r_W2lv
    Dq_r_W2lv = struct[0].Dq_r_W2lv
    p_ref_W3lv = struct[0].p_ref_W3lv
    T_pq_W3lv = struct[0].T_pq_W3lv
    v_loc_ref_W3lv = struct[0].v_loc_ref_W3lv
    Dv_r_W3lv = struct[0].Dv_r_W3lv
    Dq_r_W3lv = struct[0].Dq_r_W3lv
    p_ref_STlv = struct[0].p_ref_STlv
    T_pq_STlv = struct[0].T_pq_STlv
    v_loc_ref_STlv = struct[0].v_loc_ref_STlv
    Dv_r_STlv = struct[0].Dv_r_STlv
    Dq_r_STlv = struct[0].Dq_r_STlv
    
    # Dynamical states:
    p_W1lv_a = struct[0].x[0,0]
    p_W1lv_b = struct[0].x[1,0]
    p_W1lv_c = struct[0].x[2,0]
    q_W1lv_a = struct[0].x[3,0]
    q_W1lv_b = struct[0].x[4,0]
    q_W1lv_c = struct[0].x[5,0]
    p_W2lv_a = struct[0].x[6,0]
    p_W2lv_b = struct[0].x[7,0]
    p_W2lv_c = struct[0].x[8,0]
    q_W2lv_a = struct[0].x[9,0]
    q_W2lv_b = struct[0].x[10,0]
    q_W2lv_c = struct[0].x[11,0]
    p_W3lv_a = struct[0].x[12,0]
    p_W3lv_b = struct[0].x[13,0]
    p_W3lv_c = struct[0].x[14,0]
    q_W3lv_a = struct[0].x[15,0]
    q_W3lv_b = struct[0].x[16,0]
    q_W3lv_c = struct[0].x[17,0]
    p_STlv_a = struct[0].x[18,0]
    p_STlv_b = struct[0].x[19,0]
    p_STlv_c = struct[0].x[20,0]
    q_STlv_a = struct[0].x[21,0]
    q_STlv_b = struct[0].x[22,0]
    q_STlv_c = struct[0].x[23,0]
    
    # Algebraic states:
    v_W1lv_a_r = struct[0].y_run[0,0]
    v_W1lv_a_i = struct[0].y_run[1,0]
    v_W1lv_b_r = struct[0].y_run[2,0]
    v_W1lv_b_i = struct[0].y_run[3,0]
    v_W1lv_c_r = struct[0].y_run[4,0]
    v_W1lv_c_i = struct[0].y_run[5,0]
    v_W2lv_a_r = struct[0].y_run[6,0]
    v_W2lv_a_i = struct[0].y_run[7,0]
    v_W2lv_b_r = struct[0].y_run[8,0]
    v_W2lv_b_i = struct[0].y_run[9,0]
    v_W2lv_c_r = struct[0].y_run[10,0]
    v_W2lv_c_i = struct[0].y_run[11,0]
    v_W3lv_a_r = struct[0].y_run[12,0]
    v_W3lv_a_i = struct[0].y_run[13,0]
    v_W3lv_b_r = struct[0].y_run[14,0]
    v_W3lv_b_i = struct[0].y_run[15,0]
    v_W3lv_c_r = struct[0].y_run[16,0]
    v_W3lv_c_i = struct[0].y_run[17,0]
    v_STlv_a_r = struct[0].y_run[18,0]
    v_STlv_a_i = struct[0].y_run[19,0]
    v_STlv_b_r = struct[0].y_run[20,0]
    v_STlv_b_i = struct[0].y_run[21,0]
    v_STlv_c_r = struct[0].y_run[22,0]
    v_STlv_c_i = struct[0].y_run[23,0]
    v_POI_a_r = struct[0].y_run[24,0]
    v_POI_a_i = struct[0].y_run[25,0]
    v_POI_b_r = struct[0].y_run[26,0]
    v_POI_b_i = struct[0].y_run[27,0]
    v_POI_c_r = struct[0].y_run[28,0]
    v_POI_c_i = struct[0].y_run[29,0]
    v_POImv_a_r = struct[0].y_run[30,0]
    v_POImv_a_i = struct[0].y_run[31,0]
    v_POImv_b_r = struct[0].y_run[32,0]
    v_POImv_b_i = struct[0].y_run[33,0]
    v_POImv_c_r = struct[0].y_run[34,0]
    v_POImv_c_i = struct[0].y_run[35,0]
    v_W1mv_a_r = struct[0].y_run[36,0]
    v_W1mv_a_i = struct[0].y_run[37,0]
    v_W1mv_b_r = struct[0].y_run[38,0]
    v_W1mv_b_i = struct[0].y_run[39,0]
    v_W1mv_c_r = struct[0].y_run[40,0]
    v_W1mv_c_i = struct[0].y_run[41,0]
    v_W2mv_a_r = struct[0].y_run[42,0]
    v_W2mv_a_i = struct[0].y_run[43,0]
    v_W2mv_b_r = struct[0].y_run[44,0]
    v_W2mv_b_i = struct[0].y_run[45,0]
    v_W2mv_c_r = struct[0].y_run[46,0]
    v_W2mv_c_i = struct[0].y_run[47,0]
    v_W3mv_a_r = struct[0].y_run[48,0]
    v_W3mv_a_i = struct[0].y_run[49,0]
    v_W3mv_b_r = struct[0].y_run[50,0]
    v_W3mv_b_i = struct[0].y_run[51,0]
    v_W3mv_c_r = struct[0].y_run[52,0]
    v_W3mv_c_i = struct[0].y_run[53,0]
    v_STmv_a_r = struct[0].y_run[54,0]
    v_STmv_a_i = struct[0].y_run[55,0]
    v_STmv_b_r = struct[0].y_run[56,0]
    v_STmv_b_i = struct[0].y_run[57,0]
    v_STmv_c_r = struct[0].y_run[58,0]
    v_STmv_c_i = struct[0].y_run[59,0]
    i_l_W1mv_W2mv_a_r = struct[0].y_run[60,0]
    i_l_W1mv_W2mv_a_i = struct[0].y_run[61,0]
    i_l_W1mv_W2mv_b_r = struct[0].y_run[62,0]
    i_l_W1mv_W2mv_b_i = struct[0].y_run[63,0]
    i_l_W1mv_W2mv_c_r = struct[0].y_run[64,0]
    i_l_W1mv_W2mv_c_i = struct[0].y_run[65,0]
    i_l_W2mv_W3mv_a_r = struct[0].y_run[66,0]
    i_l_W2mv_W3mv_a_i = struct[0].y_run[67,0]
    i_l_W2mv_W3mv_b_r = struct[0].y_run[68,0]
    i_l_W2mv_W3mv_b_i = struct[0].y_run[69,0]
    i_l_W2mv_W3mv_c_r = struct[0].y_run[70,0]
    i_l_W2mv_W3mv_c_i = struct[0].y_run[71,0]
    i_l_W3mv_POImv_a_r = struct[0].y_run[72,0]
    i_l_W3mv_POImv_a_i = struct[0].y_run[73,0]
    i_l_W3mv_POImv_b_r = struct[0].y_run[74,0]
    i_l_W3mv_POImv_b_i = struct[0].y_run[75,0]
    i_l_W3mv_POImv_c_r = struct[0].y_run[76,0]
    i_l_W3mv_POImv_c_i = struct[0].y_run[77,0]
    i_l_STmv_POImv_a_r = struct[0].y_run[78,0]
    i_l_STmv_POImv_a_i = struct[0].y_run[79,0]
    i_l_STmv_POImv_b_r = struct[0].y_run[80,0]
    i_l_STmv_POImv_b_i = struct[0].y_run[81,0]
    i_l_STmv_POImv_c_r = struct[0].y_run[82,0]
    i_l_STmv_POImv_c_i = struct[0].y_run[83,0]
    i_l_POI_GRID_a_r = struct[0].y_run[84,0]
    i_l_POI_GRID_a_i = struct[0].y_run[85,0]
    i_l_POI_GRID_b_r = struct[0].y_run[86,0]
    i_l_POI_GRID_b_i = struct[0].y_run[87,0]
    i_l_POI_GRID_c_r = struct[0].y_run[88,0]
    i_l_POI_GRID_c_i = struct[0].y_run[89,0]
    i_W1lv_a_r = struct[0].y_run[90,0]
    i_W1lv_a_i = struct[0].y_run[91,0]
    i_W1lv_b_r = struct[0].y_run[92,0]
    i_W1lv_b_i = struct[0].y_run[93,0]
    i_W1lv_c_r = struct[0].y_run[94,0]
    i_W1lv_c_i = struct[0].y_run[95,0]
    v_m_W1lv = struct[0].y_run[96,0]
    v_m_W1mv = struct[0].y_run[97,0]
    i_reac_ref_W1lv = struct[0].y_run[98,0]
    q_ref_W1lv = struct[0].y_run[99,0]
    i_W2lv_a_r = struct[0].y_run[100,0]
    i_W2lv_a_i = struct[0].y_run[101,0]
    i_W2lv_b_r = struct[0].y_run[102,0]
    i_W2lv_b_i = struct[0].y_run[103,0]
    i_W2lv_c_r = struct[0].y_run[104,0]
    i_W2lv_c_i = struct[0].y_run[105,0]
    v_m_W2lv = struct[0].y_run[106,0]
    v_m_W2mv = struct[0].y_run[107,0]
    i_reac_ref_W2lv = struct[0].y_run[108,0]
    q_ref_W2lv = struct[0].y_run[109,0]
    i_W3lv_a_r = struct[0].y_run[110,0]
    i_W3lv_a_i = struct[0].y_run[111,0]
    i_W3lv_b_r = struct[0].y_run[112,0]
    i_W3lv_b_i = struct[0].y_run[113,0]
    i_W3lv_c_r = struct[0].y_run[114,0]
    i_W3lv_c_i = struct[0].y_run[115,0]
    v_m_W3lv = struct[0].y_run[116,0]
    v_m_W3mv = struct[0].y_run[117,0]
    i_reac_ref_W3lv = struct[0].y_run[118,0]
    q_ref_W3lv = struct[0].y_run[119,0]
    i_STlv_a_r = struct[0].y_run[120,0]
    i_STlv_a_i = struct[0].y_run[121,0]
    i_STlv_b_r = struct[0].y_run[122,0]
    i_STlv_b_i = struct[0].y_run[123,0]
    i_STlv_c_r = struct[0].y_run[124,0]
    i_STlv_c_i = struct[0].y_run[125,0]
    v_m_STlv = struct[0].y_run[126,0]
    v_m_STmv = struct[0].y_run[127,0]
    i_reac_ref_STlv = struct[0].y_run[128,0]
    q_ref_STlv = struct[0].y_run[129,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = (-p_W1lv_a + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[1,0] = (-p_W1lv_b + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[2,0] = (-p_W1lv_c + p_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[3,0] = (-q_W1lv_a + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[4,0] = (-q_W1lv_b + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[5,0] = (-q_W1lv_c + q_ref_W1lv/3)/T_pq_W1lv
        struct[0].f[6,0] = (-p_W2lv_a + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[7,0] = (-p_W2lv_b + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[8,0] = (-p_W2lv_c + p_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[9,0] = (-q_W2lv_a + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[10,0] = (-q_W2lv_b + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[11,0] = (-q_W2lv_c + q_ref_W2lv/3)/T_pq_W2lv
        struct[0].f[12,0] = (-p_W3lv_a + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[13,0] = (-p_W3lv_b + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[14,0] = (-p_W3lv_c + p_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[15,0] = (-q_W3lv_a + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[16,0] = (-q_W3lv_b + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[17,0] = (-q_W3lv_c + q_ref_W3lv/3)/T_pq_W3lv
        struct[0].f[18,0] = (-p_STlv_a + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[19,0] = (-p_STlv_b + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[20,0] = (-p_STlv_c + p_ref_STlv/3)/T_pq_STlv
        struct[0].f[21,0] = (-q_STlv_a + q_ref_STlv/3)/T_pq_STlv
        struct[0].f[22,0] = (-q_STlv_b + q_ref_STlv/3)/T_pq_STlv
        struct[0].f[23,0] = (-q_STlv_c + q_ref_STlv/3)/T_pq_STlv
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_W1lv_a_r - 85.1513138847732*v_W1lv_a_i - 14.1918856474622*v_W1lv_a_r + 1.69609362276623*v_W1mv_a_i + 0.282682270461039*v_W1mv_a_r - 1.69609362276623*v_W1mv_c_i - 0.282682270461039*v_W1mv_c_r
        struct[0].g[1,0] = i_W1lv_a_i - 14.1918856474622*v_W1lv_a_i + 85.1513138847732*v_W1lv_a_r + 0.282682270461039*v_W1mv_a_i - 1.69609362276623*v_W1mv_a_r - 0.282682270461039*v_W1mv_c_i + 1.69609362276623*v_W1mv_c_r
        struct[0].g[2,0] = i_W1lv_b_r - 85.1513138847732*v_W1lv_b_i - 14.1918856474622*v_W1lv_b_r - 1.69609362276623*v_W1mv_a_i - 0.282682270461039*v_W1mv_a_r + 1.69609362276623*v_W1mv_b_i + 0.282682270461039*v_W1mv_b_r
        struct[0].g[3,0] = i_W1lv_b_i - 14.1918856474622*v_W1lv_b_i + 85.1513138847732*v_W1lv_b_r - 0.282682270461039*v_W1mv_a_i + 1.69609362276623*v_W1mv_a_r + 0.282682270461039*v_W1mv_b_i - 1.69609362276623*v_W1mv_b_r
        struct[0].g[4,0] = i_W1lv_c_r - 85.1513138847732*v_W1lv_c_i - 14.1918856474622*v_W1lv_c_r - 1.69609362276623*v_W1mv_b_i - 0.282682270461039*v_W1mv_b_r + 1.69609362276623*v_W1mv_c_i + 0.282682270461039*v_W1mv_c_r
        struct[0].g[5,0] = i_W1lv_c_i - 14.1918856474622*v_W1lv_c_i + 85.1513138847732*v_W1lv_c_r - 0.282682270461039*v_W1mv_b_i + 1.69609362276623*v_W1mv_b_r + 0.282682270461039*v_W1mv_c_i - 1.69609362276623*v_W1mv_c_r
        struct[0].g[6,0] = i_W2lv_a_r - 85.1513138847732*v_W2lv_a_i - 14.1918856474622*v_W2lv_a_r + 1.69609362276623*v_W2mv_a_i + 0.282682270461039*v_W2mv_a_r - 1.69609362276623*v_W2mv_c_i - 0.282682270461039*v_W2mv_c_r
        struct[0].g[7,0] = i_W2lv_a_i - 14.1918856474622*v_W2lv_a_i + 85.1513138847732*v_W2lv_a_r + 0.282682270461039*v_W2mv_a_i - 1.69609362276623*v_W2mv_a_r - 0.282682270461039*v_W2mv_c_i + 1.69609362276623*v_W2mv_c_r
        struct[0].g[8,0] = i_W2lv_b_r - 85.1513138847732*v_W2lv_b_i - 14.1918856474622*v_W2lv_b_r - 1.69609362276623*v_W2mv_a_i - 0.282682270461039*v_W2mv_a_r + 1.69609362276623*v_W2mv_b_i + 0.282682270461039*v_W2mv_b_r
        struct[0].g[9,0] = i_W2lv_b_i - 14.1918856474622*v_W2lv_b_i + 85.1513138847732*v_W2lv_b_r - 0.282682270461039*v_W2mv_a_i + 1.69609362276623*v_W2mv_a_r + 0.282682270461039*v_W2mv_b_i - 1.69609362276623*v_W2mv_b_r
        struct[0].g[10,0] = i_W2lv_c_r - 85.1513138847732*v_W2lv_c_i - 14.1918856474622*v_W2lv_c_r - 1.69609362276623*v_W2mv_b_i - 0.282682270461039*v_W2mv_b_r + 1.69609362276623*v_W2mv_c_i + 0.282682270461039*v_W2mv_c_r
        struct[0].g[11,0] = i_W2lv_c_i - 14.1918856474622*v_W2lv_c_i + 85.1513138847732*v_W2lv_c_r - 0.282682270461039*v_W2mv_b_i + 1.69609362276623*v_W2mv_b_r + 0.282682270461039*v_W2mv_c_i - 1.69609362276623*v_W2mv_c_r
        struct[0].g[12,0] = i_W3lv_a_r - 85.1513138847732*v_W3lv_a_i - 14.1918856474622*v_W3lv_a_r + 1.69609362276623*v_W3mv_a_i + 0.282682270461039*v_W3mv_a_r - 1.69609362276623*v_W3mv_c_i - 0.282682270461039*v_W3mv_c_r
        struct[0].g[13,0] = i_W3lv_a_i - 14.1918856474622*v_W3lv_a_i + 85.1513138847732*v_W3lv_a_r + 0.282682270461039*v_W3mv_a_i - 1.69609362276623*v_W3mv_a_r - 0.282682270461039*v_W3mv_c_i + 1.69609362276623*v_W3mv_c_r
        struct[0].g[14,0] = i_W3lv_b_r - 85.1513138847732*v_W3lv_b_i - 14.1918856474622*v_W3lv_b_r - 1.69609362276623*v_W3mv_a_i - 0.282682270461039*v_W3mv_a_r + 1.69609362276623*v_W3mv_b_i + 0.282682270461039*v_W3mv_b_r
        struct[0].g[15,0] = i_W3lv_b_i - 14.1918856474622*v_W3lv_b_i + 85.1513138847732*v_W3lv_b_r - 0.282682270461039*v_W3mv_a_i + 1.69609362276623*v_W3mv_a_r + 0.282682270461039*v_W3mv_b_i - 1.69609362276623*v_W3mv_b_r
        struct[0].g[16,0] = i_W3lv_c_r - 85.1513138847732*v_W3lv_c_i - 14.1918856474622*v_W3lv_c_r - 1.69609362276623*v_W3mv_b_i - 0.282682270461039*v_W3mv_b_r + 1.69609362276623*v_W3mv_c_i + 0.282682270461039*v_W3mv_c_r
        struct[0].g[17,0] = i_W3lv_c_i - 14.1918856474622*v_W3lv_c_i + 85.1513138847732*v_W3lv_c_r - 0.282682270461039*v_W3mv_b_i + 1.69609362276623*v_W3mv_b_r + 0.282682270461039*v_W3mv_c_i - 1.69609362276623*v_W3mv_c_r
        struct[0].g[18,0] = i_STlv_a_r - 85.1513138847732*v_STlv_a_i - 14.1918856474622*v_STlv_a_r + 1.69609362276623*v_STmv_a_i + 0.282682270461039*v_STmv_a_r - 1.69609362276623*v_STmv_c_i - 0.282682270461039*v_STmv_c_r
        struct[0].g[19,0] = i_STlv_a_i - 14.1918856474622*v_STlv_a_i + 85.1513138847732*v_STlv_a_r + 0.282682270461039*v_STmv_a_i - 1.69609362276623*v_STmv_a_r - 0.282682270461039*v_STmv_c_i + 1.69609362276623*v_STmv_c_r
        struct[0].g[20,0] = i_STlv_b_r - 85.1513138847732*v_STlv_b_i - 14.1918856474622*v_STlv_b_r - 1.69609362276623*v_STmv_a_i - 0.282682270461039*v_STmv_a_r + 1.69609362276623*v_STmv_b_i + 0.282682270461039*v_STmv_b_r
        struct[0].g[21,0] = i_STlv_b_i - 14.1918856474622*v_STlv_b_i + 85.1513138847732*v_STlv_b_r - 0.282682270461039*v_STmv_a_i + 1.69609362276623*v_STmv_a_r + 0.282682270461039*v_STmv_b_i - 1.69609362276623*v_STmv_b_r
        struct[0].g[22,0] = i_STlv_c_r - 85.1513138847732*v_STlv_c_i - 14.1918856474622*v_STlv_c_r - 1.69609362276623*v_STmv_b_i - 0.282682270461039*v_STmv_b_r + 1.69609362276623*v_STmv_c_i + 0.282682270461039*v_STmv_c_r
        struct[0].g[23,0] = i_STlv_c_i - 14.1918856474622*v_STlv_c_i + 85.1513138847732*v_STlv_c_r - 0.282682270461039*v_STmv_b_i + 1.69609362276623*v_STmv_b_r + 0.282682270461039*v_STmv_c_i - 1.69609362276623*v_STmv_c_r
        struct[0].g[24,0] = i_POI_a_r + 0.040290088638195*v_GRID_a_i + 0.024174053182917*v_GRID_a_r + 4.66248501556824e-18*v_GRID_b_i - 4.31760362252812e-18*v_GRID_b_r + 4.19816664496737e-18*v_GRID_c_i - 3.49608108880335e-18*v_GRID_c_r - 0.0591264711109411*v_POI_a_i - 0.0265286009920103*v_POI_a_r + 0.00941819123637305*v_POI_b_i + 0.00117727390454664*v_POI_b_r + 0.00941819123637305*v_POI_c_i + 0.00117727390454664*v_POI_c_r + 0.0538321929314336*v_POImv_a_i + 0.0067290241164292*v_POImv_a_r - 0.0538321929314336*v_POImv_b_i - 0.0067290241164292*v_POImv_b_r
        struct[0].g[25,0] = i_POI_a_i + 0.024174053182917*v_GRID_a_i - 0.040290088638195*v_GRID_a_r - 4.31760362252812e-18*v_GRID_b_i - 4.66248501556824e-18*v_GRID_b_r - 3.49608108880335e-18*v_GRID_c_i - 4.19816664496737e-18*v_GRID_c_r - 0.0265286009920103*v_POI_a_i + 0.0591264711109411*v_POI_a_r + 0.00117727390454664*v_POI_b_i - 0.00941819123637305*v_POI_b_r + 0.00117727390454664*v_POI_c_i - 0.00941819123637305*v_POI_c_r + 0.0067290241164292*v_POImv_a_i - 0.0538321929314336*v_POImv_a_r - 0.0067290241164292*v_POImv_b_i + 0.0538321929314336*v_POImv_b_r
        struct[0].g[26,0] = i_POI_b_r + 6.30775359573304e-19*v_GRID_a_i - 2.07254761002657e-18*v_GRID_a_r + 0.040290088638195*v_GRID_b_i + 0.024174053182917*v_GRID_b_r + 9.01107656533306e-19*v_GRID_c_i - 1.78419315993592e-17*v_GRID_c_r + 0.00941819123637305*v_POI_a_i + 0.00117727390454663*v_POI_a_r - 0.0591264711109411*v_POI_b_i - 0.0265286009920103*v_POI_b_r + 0.00941819123637305*v_POI_c_i + 0.00117727390454665*v_POI_c_r + 0.0538321929314336*v_POImv_b_i + 0.0067290241164292*v_POImv_b_r - 0.0538321929314336*v_POImv_c_i - 0.0067290241164292*v_POImv_c_r
        struct[0].g[27,0] = i_POI_b_i - 2.07254761002657e-18*v_GRID_a_i - 6.30775359573304e-19*v_GRID_a_r + 0.024174053182917*v_GRID_b_i - 0.040290088638195*v_GRID_b_r - 1.78419315993592e-17*v_GRID_c_i - 9.01107656533306e-19*v_GRID_c_r + 0.00117727390454663*v_POI_a_i - 0.00941819123637305*v_POI_a_r - 0.0265286009920103*v_POI_b_i + 0.0591264711109411*v_POI_b_r + 0.00117727390454665*v_POI_c_i - 0.00941819123637305*v_POI_c_r + 0.0067290241164292*v_POImv_b_i - 0.0538321929314336*v_POImv_b_r - 0.0067290241164292*v_POImv_c_i + 0.0538321929314336*v_POImv_c_r
        struct[0].g[28,0] = i_POI_c_r - 7.20886125226632e-19*v_GRID_a_i - 1.35166148479994e-18*v_GRID_a_r - 4.50553828266631e-19*v_GRID_b_i - 1.71210454741325e-17*v_GRID_b_r + 0.040290088638195*v_GRID_c_i + 0.024174053182917*v_GRID_c_r + 0.00941819123637305*v_POI_a_i + 0.00117727390454663*v_POI_a_r + 0.00941819123637305*v_POI_b_i + 0.00117727390454665*v_POI_b_r - 0.0591264711109411*v_POI_c_i - 0.0265286009920103*v_POI_c_r - 0.0538321929314336*v_POImv_a_i - 0.0067290241164292*v_POImv_a_r + 0.0538321929314336*v_POImv_c_i + 0.0067290241164292*v_POImv_c_r
        struct[0].g[29,0] = i_POI_c_i - 1.35166148479994e-18*v_GRID_a_i + 7.20886125226632e-19*v_GRID_a_r - 1.71210454741325e-17*v_GRID_b_i + 4.50553828266631e-19*v_GRID_b_r + 0.024174053182917*v_GRID_c_i - 0.040290088638195*v_GRID_c_r + 0.00117727390454663*v_POI_a_i - 0.00941819123637305*v_POI_a_r + 0.00117727390454665*v_POI_b_i - 0.00941819123637305*v_POI_b_r - 0.0265286009920103*v_POI_c_i + 0.0591264711109411*v_POI_c_r - 0.0067290241164292*v_POImv_a_i + 0.0538321929314336*v_POImv_a_r + 0.0067290241164292*v_POImv_c_i - 0.0538321929314336*v_POImv_c_r
        struct[0].g[30,0] = i_POImv_a_r + 0.0538321929314336*v_POI_a_i + 0.0067290241164292*v_POI_a_r - 0.0538321929314336*v_POI_c_i - 0.0067290241164292*v_POI_c_r - 155.244588874881*v_POImv_a_i - 188.924390492986*v_POImv_a_r + 44.2677164725443*v_POImv_b_i + 53.9540151298641*v_POImv_b_r + 44.2677164725443*v_POImv_c_i + 53.9540151298641*v_POImv_c_r + 148.977829666654*v_STmv_a_i + 181.611068546017*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274334*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274334*v_STmv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[31,0] = i_POImv_a_i + 0.0067290241164292*v_POI_a_i - 0.0538321929314336*v_POI_a_r - 0.0067290241164292*v_POI_c_i + 0.0538321929314336*v_POI_c_r - 188.924390492986*v_POImv_a_i + 155.244588874881*v_POImv_a_r + 53.9540151298641*v_POImv_b_i - 44.2677164725443*v_POImv_b_r + 53.9540151298641*v_POImv_c_i - 44.2677164725443*v_POImv_c_r + 181.611068546017*v_STmv_a_i - 148.977829666654*v_STmv_a_r - 51.8888767274334*v_STmv_b_i + 42.5650941904727*v_STmv_b_r - 51.8888767274334*v_STmv_c_i + 42.5650941904727*v_STmv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[32,0] = i_POImv_b_r - 0.0538321929314336*v_POI_a_i - 0.0067290241164292*v_POI_a_r + 0.0538321929314336*v_POI_b_i + 0.0067290241164292*v_POI_b_r + 44.2677164725443*v_POImv_a_i + 53.9540151298641*v_POImv_a_r - 155.244588874881*v_POImv_b_i - 188.924390492986*v_POImv_b_r + 44.2677164725443*v_POImv_c_i + 53.9540151298642*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r + 148.977829666654*v_STmv_b_i + 181.611068546017*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274335*v_STmv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[33,0] = i_POImv_b_i - 0.0067290241164292*v_POI_a_i + 0.0538321929314336*v_POI_a_r + 0.0067290241164292*v_POI_b_i - 0.0538321929314336*v_POI_b_r + 53.9540151298641*v_POImv_a_i - 44.2677164725443*v_POImv_a_r - 188.924390492986*v_POImv_b_i + 155.244588874881*v_POImv_b_r + 53.9540151298642*v_POImv_c_i - 44.2677164725443*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r + 181.611068546017*v_STmv_b_i - 148.977829666654*v_STmv_b_r - 51.8888767274335*v_STmv_c_i + 42.5650941904727*v_STmv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[34,0] = i_POImv_c_r - 0.0538321929314336*v_POI_b_i - 0.0067290241164292*v_POI_b_r + 0.0538321929314336*v_POI_c_i + 0.0067290241164292*v_POI_c_r + 44.2677164725443*v_POImv_a_i + 53.9540151298641*v_POImv_a_r + 44.2677164725443*v_POImv_b_i + 53.9540151298642*v_POImv_b_r - 155.244588874881*v_POImv_c_i - 188.924390492986*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274335*v_STmv_b_r + 148.977829666654*v_STmv_c_i + 181.611068546017*v_STmv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[35,0] = i_POImv_c_i - 0.0067290241164292*v_POI_b_i + 0.0538321929314336*v_POI_b_r + 0.0067290241164292*v_POI_c_i - 0.0538321929314336*v_POI_c_r + 53.9540151298641*v_POImv_a_i - 44.2677164725443*v_POImv_a_r + 53.9540151298642*v_POImv_b_i - 44.2677164725443*v_POImv_b_r - 188.924390492986*v_POImv_c_i + 155.244588874881*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r - 51.8888767274335*v_STmv_b_i + 42.5650941904727*v_STmv_b_r + 181.611068546017*v_STmv_c_i - 148.977829666654*v_STmv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[36,0] = i_W1mv_a_r + 1.69609362276623*v_W1lv_a_i + 0.282682270461039*v_W1lv_a_r - 1.69609362276623*v_W1lv_b_i - 0.282682270461039*v_W1lv_b_r - 6.02663624833782*v_W1mv_a_i - 7.27570400310194*v_W1mv_a_r + 1.73640535376106*v_W1mv_b_i + 2.08118569972797*v_W1mv_b_r + 1.73640535376106*v_W1mv_c_i + 2.08118569972797*v_W1mv_c_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r
        struct[0].g[37,0] = i_W1mv_a_i + 0.282682270461039*v_W1lv_a_i - 1.69609362276623*v_W1lv_a_r - 0.282682270461039*v_W1lv_b_i + 1.69609362276623*v_W1lv_b_r - 7.27570400310194*v_W1mv_a_i + 6.02663624833782*v_W1mv_a_r + 2.08118569972797*v_W1mv_b_i - 1.73640535376106*v_W1mv_b_r + 2.08118569972797*v_W1mv_c_i - 1.73640535376106*v_W1mv_c_r + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r
        struct[0].g[38,0] = i_W1mv_b_r + 1.69609362276623*v_W1lv_b_i + 0.282682270461039*v_W1lv_b_r - 1.69609362276623*v_W1lv_c_i - 0.282682270461039*v_W1lv_c_r + 1.73640535376106*v_W1mv_a_i + 2.08118569972797*v_W1mv_a_r - 6.02663624833782*v_W1mv_b_i - 7.27570400310194*v_W1mv_b_r + 1.73640535376106*v_W1mv_c_i + 2.08118569972797*v_W1mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r
        struct[0].g[39,0] = i_W1mv_b_i + 0.282682270461039*v_W1lv_b_i - 1.69609362276623*v_W1lv_b_r - 0.282682270461039*v_W1lv_c_i + 1.69609362276623*v_W1lv_c_r + 2.08118569972797*v_W1mv_a_i - 1.73640535376106*v_W1mv_a_r - 7.27570400310194*v_W1mv_b_i + 6.02663624833782*v_W1mv_b_r + 2.08118569972797*v_W1mv_c_i - 1.73640535376106*v_W1mv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r
        struct[0].g[40,0] = i_W1mv_c_r - 1.69609362276623*v_W1lv_a_i - 0.282682270461039*v_W1lv_a_r + 1.69609362276623*v_W1lv_c_i + 0.282682270461039*v_W1lv_c_r + 1.73640535376106*v_W1mv_a_i + 2.08118569972797*v_W1mv_a_r + 1.73640535376106*v_W1mv_b_i + 2.08118569972797*v_W1mv_b_r - 6.02663624833782*v_W1mv_c_i - 7.27570400310194*v_W1mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r
        struct[0].g[41,0] = i_W1mv_c_i - 0.282682270461039*v_W1lv_a_i + 1.69609362276623*v_W1lv_a_r + 0.282682270461039*v_W1lv_c_i - 1.69609362276623*v_W1lv_c_r + 2.08118569972797*v_W1mv_a_i - 1.73640535376106*v_W1mv_a_r + 2.08118569972797*v_W1mv_b_i - 1.73640535376106*v_W1mv_b_r - 7.27570400310194*v_W1mv_c_i + 6.02663624833782*v_W1mv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r
        struct[0].g[42,0] = i_W2mv_a_r + 5.95911318666618*v_W1mv_a_i + 7.26444274184068*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.69609362276623*v_W2lv_a_i + 0.282682270461039*v_W2lv_a_r - 1.69609362276623*v_W2lv_b_i - 0.282682270461039*v_W2lv_b_r - 11.9857049291081*v_W2mv_a_i - 14.5401467449426*v_W2mv_a_r + 3.43902692373834*v_W2mv_b_i + 4.1567407688253*v_W2mv_b_r + 3.43902692373834*v_W2mv_c_i + 4.1567407688253*v_W2mv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[43,0] = i_W2mv_a_i + 7.26444274184068*v_W1mv_a_i - 5.95911318666618*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 0.282682270461039*v_W2lv_a_i - 1.69609362276623*v_W2lv_a_r - 0.282682270461039*v_W2lv_b_i + 1.69609362276623*v_W2lv_b_r - 14.5401467449426*v_W2mv_a_i + 11.9857049291081*v_W2mv_a_r + 4.1567407688253*v_W2mv_b_i - 3.43902692373834*v_W2mv_b_r + 4.1567407688253*v_W2mv_c_i - 3.43902692373834*v_W2mv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[44,0] = i_W2mv_b_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r + 5.95911318666618*v_W1mv_b_i + 7.26444274184068*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.69609362276623*v_W2lv_b_i + 0.282682270461039*v_W2lv_b_r - 1.69609362276623*v_W2lv_c_i - 0.282682270461039*v_W2lv_c_r + 3.43902692373834*v_W2mv_a_i + 4.1567407688253*v_W2mv_a_r - 11.9857049291081*v_W2mv_b_i - 14.5401467449426*v_W2mv_b_r + 3.43902692373834*v_W2mv_c_i + 4.15674076882531*v_W2mv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[45,0] = i_W2mv_b_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r + 7.26444274184068*v_W1mv_b_i - 5.95911318666618*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 0.282682270461039*v_W2lv_b_i - 1.69609362276623*v_W2lv_b_r - 0.282682270461039*v_W2lv_c_i + 1.69609362276623*v_W2lv_c_r + 4.1567407688253*v_W2mv_a_i - 3.43902692373834*v_W2mv_a_r - 14.5401467449426*v_W2mv_b_i + 11.9857049291081*v_W2mv_b_r + 4.15674076882531*v_W2mv_c_i - 3.43902692373834*v_W2mv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[46,0] = i_W2mv_c_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r + 5.95911318666618*v_W1mv_c_i + 7.26444274184068*v_W1mv_c_r - 1.69609362276623*v_W2lv_a_i - 0.282682270461039*v_W2lv_a_r + 1.69609362276623*v_W2lv_c_i + 0.282682270461039*v_W2lv_c_r + 3.43902692373834*v_W2mv_a_i + 4.1567407688253*v_W2mv_a_r + 3.43902692373834*v_W2mv_b_i + 4.15674076882531*v_W2mv_b_r - 11.9857049291081*v_W2mv_c_i - 14.5401467449426*v_W2mv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[47,0] = i_W2mv_c_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r + 7.26444274184068*v_W1mv_c_i - 5.95911318666618*v_W1mv_c_r - 0.282682270461039*v_W2lv_a_i + 1.69609362276623*v_W2lv_a_r + 0.282682270461039*v_W2lv_c_i - 1.69609362276623*v_W2lv_c_r + 4.1567407688253*v_W2mv_a_i - 3.43902692373834*v_W2mv_a_r + 4.15674076882531*v_W2mv_b_i - 3.43902692373834*v_W2mv_b_r - 14.5401467449426*v_W2mv_c_i + 11.9857049291081*v_W2mv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[48,0] = i_W3mv_a_r + 5.95911318666618*v_POImv_a_i + 7.26444274184068*v_POImv_a_r - 1.70260376761891*v_POImv_b_i - 2.07555506909734*v_POImv_b_r - 1.70260376761891*v_POImv_c_i - 2.07555506909734*v_POImv_c_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.69609362276623*v_W3lv_a_i + 0.282682270461039*v_W3lv_a_r - 1.69609362276623*v_W3lv_b_i - 0.282682270461039*v_W3lv_b_r - 11.9857049291081*v_W3mv_a_i - 14.5401467449426*v_W3mv_a_r + 3.43902692373834*v_W3mv_b_i + 4.1567407688253*v_W3mv_b_r + 3.43902692373834*v_W3mv_c_i + 4.1567407688253*v_W3mv_c_r
        struct[0].g[49,0] = i_W3mv_a_i + 7.26444274184068*v_POImv_a_i - 5.95911318666618*v_POImv_a_r - 2.07555506909734*v_POImv_b_i + 1.70260376761891*v_POImv_b_r - 2.07555506909734*v_POImv_c_i + 1.70260376761891*v_POImv_c_r + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 0.282682270461039*v_W3lv_a_i - 1.69609362276623*v_W3lv_a_r - 0.282682270461039*v_W3lv_b_i + 1.69609362276623*v_W3lv_b_r - 14.5401467449426*v_W3mv_a_i + 11.9857049291081*v_W3mv_a_r + 4.1567407688253*v_W3mv_b_i - 3.43902692373834*v_W3mv_b_r + 4.1567407688253*v_W3mv_c_i - 3.43902692373834*v_W3mv_c_r
        struct[0].g[50,0] = i_W3mv_b_r - 1.70260376761891*v_POImv_a_i - 2.07555506909734*v_POImv_a_r + 5.95911318666618*v_POImv_b_i + 7.26444274184068*v_POImv_b_r - 1.70260376761891*v_POImv_c_i - 2.07555506909734*v_POImv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.69609362276623*v_W3lv_b_i + 0.282682270461039*v_W3lv_b_r - 1.69609362276623*v_W3lv_c_i - 0.282682270461039*v_W3lv_c_r + 3.43902692373834*v_W3mv_a_i + 4.1567407688253*v_W3mv_a_r - 11.9857049291081*v_W3mv_b_i - 14.5401467449426*v_W3mv_b_r + 3.43902692373834*v_W3mv_c_i + 4.15674076882531*v_W3mv_c_r
        struct[0].g[51,0] = i_W3mv_b_i - 2.07555506909734*v_POImv_a_i + 1.70260376761891*v_POImv_a_r + 7.26444274184068*v_POImv_b_i - 5.95911318666618*v_POImv_b_r - 2.07555506909734*v_POImv_c_i + 1.70260376761891*v_POImv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 0.282682270461039*v_W3lv_b_i - 1.69609362276623*v_W3lv_b_r - 0.282682270461039*v_W3lv_c_i + 1.69609362276623*v_W3lv_c_r + 4.1567407688253*v_W3mv_a_i - 3.43902692373834*v_W3mv_a_r - 14.5401467449426*v_W3mv_b_i + 11.9857049291081*v_W3mv_b_r + 4.15674076882531*v_W3mv_c_i - 3.43902692373834*v_W3mv_c_r
        struct[0].g[52,0] = i_W3mv_c_r - 1.70260376761891*v_POImv_a_i - 2.07555506909734*v_POImv_a_r - 1.70260376761891*v_POImv_b_i - 2.07555506909734*v_POImv_b_r + 5.95911318666618*v_POImv_c_i + 7.26444274184068*v_POImv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r - 1.69609362276623*v_W3lv_a_i - 0.282682270461039*v_W3lv_a_r + 1.69609362276623*v_W3lv_c_i + 0.282682270461039*v_W3lv_c_r + 3.43902692373834*v_W3mv_a_i + 4.1567407688253*v_W3mv_a_r + 3.43902692373834*v_W3mv_b_i + 4.15674076882531*v_W3mv_b_r - 11.9857049291081*v_W3mv_c_i - 14.5401467449426*v_W3mv_c_r
        struct[0].g[53,0] = i_W3mv_c_i - 2.07555506909734*v_POImv_a_i + 1.70260376761891*v_POImv_a_r - 2.07555506909734*v_POImv_b_i + 1.70260376761891*v_POImv_b_r + 7.26444274184068*v_POImv_c_i - 5.95911318666618*v_POImv_c_r - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r - 0.282682270461039*v_W3lv_a_i + 1.69609362276623*v_W3lv_a_r + 0.282682270461039*v_W3lv_c_i - 1.69609362276623*v_W3lv_c_r + 4.1567407688253*v_W3mv_a_i - 3.43902692373834*v_W3mv_a_r + 4.15674076882531*v_W3mv_b_i - 3.43902692373834*v_W3mv_b_r - 14.5401467449426*v_W3mv_c_i + 11.9857049291081*v_W3mv_c_r
        struct[0].g[54,0] = i_STmv_a_r + 148.977829666654*v_POImv_a_i + 181.611068546017*v_POImv_a_r - 42.5650941904727*v_POImv_b_i - 51.8888767274334*v_POImv_b_r - 42.5650941904727*v_POImv_c_i - 51.8888767274334*v_POImv_c_r + 1.69609362276623*v_STlv_a_i + 0.282682270461039*v_STlv_a_r - 1.69609362276623*v_STlv_b_i - 0.282682270461039*v_STlv_b_r - 149.045395453986*v_STmv_a_i - 181.622329807278*v_STmv_a_r + 42.5988786863508*v_STmv_b_i + 51.8945073580641*v_STmv_b_r + 42.5988786863508*v_STmv_c_i + 51.894507358064*v_STmv_c_r
        struct[0].g[55,0] = i_STmv_a_i + 181.611068546017*v_POImv_a_i - 148.977829666654*v_POImv_a_r - 51.8888767274334*v_POImv_b_i + 42.5650941904727*v_POImv_b_r - 51.8888767274334*v_POImv_c_i + 42.5650941904727*v_POImv_c_r + 0.282682270461039*v_STlv_a_i - 1.69609362276623*v_STlv_a_r - 0.282682270461039*v_STlv_b_i + 1.69609362276623*v_STlv_b_r - 181.622329807278*v_STmv_a_i + 149.045395453986*v_STmv_a_r + 51.8945073580641*v_STmv_b_i - 42.5988786863508*v_STmv_b_r + 51.894507358064*v_STmv_c_i - 42.5988786863508*v_STmv_c_r
        struct[0].g[56,0] = i_STmv_b_r - 42.5650941904727*v_POImv_a_i - 51.8888767274334*v_POImv_a_r + 148.977829666654*v_POImv_b_i + 181.611068546017*v_POImv_b_r - 42.5650941904727*v_POImv_c_i - 51.8888767274335*v_POImv_c_r + 1.69609362276623*v_STlv_b_i + 0.282682270461039*v_STlv_b_r - 1.69609362276623*v_STlv_c_i - 0.282682270461039*v_STlv_c_r + 42.5988786863508*v_STmv_a_i + 51.8945073580641*v_STmv_a_r - 149.045395453986*v_STmv_b_i - 181.622329807278*v_STmv_b_r + 42.5988786863508*v_STmv_c_i + 51.8945073580641*v_STmv_c_r
        struct[0].g[57,0] = i_STmv_b_i - 51.8888767274334*v_POImv_a_i + 42.5650941904727*v_POImv_a_r + 181.611068546017*v_POImv_b_i - 148.977829666654*v_POImv_b_r - 51.8888767274335*v_POImv_c_i + 42.5650941904727*v_POImv_c_r + 0.282682270461039*v_STlv_b_i - 1.69609362276623*v_STlv_b_r - 0.282682270461039*v_STlv_c_i + 1.69609362276623*v_STlv_c_r + 51.8945073580641*v_STmv_a_i - 42.5988786863508*v_STmv_a_r - 181.622329807278*v_STmv_b_i + 149.045395453986*v_STmv_b_r + 51.8945073580641*v_STmv_c_i - 42.5988786863508*v_STmv_c_r
        struct[0].g[58,0] = i_STmv_c_r - 42.5650941904727*v_POImv_a_i - 51.8888767274334*v_POImv_a_r - 42.5650941904727*v_POImv_b_i - 51.8888767274335*v_POImv_b_r + 148.977829666654*v_POImv_c_i + 181.611068546017*v_POImv_c_r - 1.69609362276623*v_STlv_a_i - 0.282682270461039*v_STlv_a_r + 1.69609362276623*v_STlv_c_i + 0.282682270461039*v_STlv_c_r + 42.5988786863508*v_STmv_a_i + 51.8945073580641*v_STmv_a_r + 42.5988786863508*v_STmv_b_i + 51.8945073580641*v_STmv_b_r - 149.045395453986*v_STmv_c_i - 181.622329807278*v_STmv_c_r
        struct[0].g[59,0] = i_STmv_c_i - 51.8888767274334*v_POImv_a_i + 42.5650941904727*v_POImv_a_r - 51.8888767274335*v_POImv_b_i + 42.5650941904727*v_POImv_b_r + 181.611068546017*v_POImv_c_i - 148.977829666654*v_POImv_c_r - 0.282682270461039*v_STlv_a_i + 1.69609362276623*v_STlv_a_r + 0.282682270461039*v_STlv_c_i - 1.69609362276623*v_STlv_c_r + 51.8945073580641*v_STmv_a_i - 42.5988786863508*v_STmv_a_r + 51.8945073580641*v_STmv_b_i - 42.5988786863508*v_STmv_b_r - 181.622329807278*v_STmv_c_i + 149.045395453986*v_STmv_c_r
        struct[0].g[60,0] = -i_l_W1mv_W2mv_a_r + 5.95911318666618*v_W1mv_a_i + 7.26444274184068*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r - 5.95911318666618*v_W2mv_a_i - 7.26444274184068*v_W2mv_a_r + 1.70260376761891*v_W2mv_b_i + 2.07555506909734*v_W2mv_b_r + 1.70260376761891*v_W2mv_c_i + 2.07555506909734*v_W2mv_c_r
        struct[0].g[61,0] = -i_l_W1mv_W2mv_a_i + 7.26444274184068*v_W1mv_a_i - 5.95911318666618*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r - 7.26444274184068*v_W2mv_a_i + 5.95911318666618*v_W2mv_a_r + 2.07555506909734*v_W2mv_b_i - 1.70260376761891*v_W2mv_b_r + 2.07555506909734*v_W2mv_c_i - 1.70260376761891*v_W2mv_c_r
        struct[0].g[62,0] = -i_l_W1mv_W2mv_b_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r + 5.95911318666618*v_W1mv_b_i + 7.26444274184068*v_W1mv_b_r - 1.70260376761891*v_W1mv_c_i - 2.07555506909734*v_W1mv_c_r + 1.70260376761891*v_W2mv_a_i + 2.07555506909734*v_W2mv_a_r - 5.95911318666618*v_W2mv_b_i - 7.26444274184068*v_W2mv_b_r + 1.70260376761891*v_W2mv_c_i + 2.07555506909734*v_W2mv_c_r
        struct[0].g[63,0] = -i_l_W1mv_W2mv_b_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r + 7.26444274184068*v_W1mv_b_i - 5.95911318666618*v_W1mv_b_r - 2.07555506909734*v_W1mv_c_i + 1.70260376761891*v_W1mv_c_r + 2.07555506909734*v_W2mv_a_i - 1.70260376761891*v_W2mv_a_r - 7.26444274184068*v_W2mv_b_i + 5.95911318666618*v_W2mv_b_r + 2.07555506909734*v_W2mv_c_i - 1.70260376761891*v_W2mv_c_r
        struct[0].g[64,0] = -i_l_W1mv_W2mv_c_r - 1.70260376761891*v_W1mv_a_i - 2.07555506909734*v_W1mv_a_r - 1.70260376761891*v_W1mv_b_i - 2.07555506909734*v_W1mv_b_r + 5.95911318666618*v_W1mv_c_i + 7.26444274184068*v_W1mv_c_r + 1.70260376761891*v_W2mv_a_i + 2.07555506909734*v_W2mv_a_r + 1.70260376761891*v_W2mv_b_i + 2.07555506909734*v_W2mv_b_r - 5.95911318666618*v_W2mv_c_i - 7.26444274184068*v_W2mv_c_r
        struct[0].g[65,0] = -i_l_W1mv_W2mv_c_i - 2.07555506909734*v_W1mv_a_i + 1.70260376761891*v_W1mv_a_r - 2.07555506909734*v_W1mv_b_i + 1.70260376761891*v_W1mv_b_r + 7.26444274184068*v_W1mv_c_i - 5.95911318666618*v_W1mv_c_r + 2.07555506909734*v_W2mv_a_i - 1.70260376761891*v_W2mv_a_r + 2.07555506909734*v_W2mv_b_i - 1.70260376761891*v_W2mv_b_r - 7.26444274184068*v_W2mv_c_i + 5.95911318666618*v_W2mv_c_r
        struct[0].g[66,0] = -i_l_W2mv_W3mv_a_r + 5.95911318666618*v_W2mv_a_i + 7.26444274184068*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r - 5.95911318666618*v_W3mv_a_i - 7.26444274184068*v_W3mv_a_r + 1.70260376761891*v_W3mv_b_i + 2.07555506909734*v_W3mv_b_r + 1.70260376761891*v_W3mv_c_i + 2.07555506909734*v_W3mv_c_r
        struct[0].g[67,0] = -i_l_W2mv_W3mv_a_i + 7.26444274184068*v_W2mv_a_i - 5.95911318666618*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r - 7.26444274184068*v_W3mv_a_i + 5.95911318666618*v_W3mv_a_r + 2.07555506909734*v_W3mv_b_i - 1.70260376761891*v_W3mv_b_r + 2.07555506909734*v_W3mv_c_i - 1.70260376761891*v_W3mv_c_r
        struct[0].g[68,0] = -i_l_W2mv_W3mv_b_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r + 5.95911318666618*v_W2mv_b_i + 7.26444274184068*v_W2mv_b_r - 1.70260376761891*v_W2mv_c_i - 2.07555506909734*v_W2mv_c_r + 1.70260376761891*v_W3mv_a_i + 2.07555506909734*v_W3mv_a_r - 5.95911318666618*v_W3mv_b_i - 7.26444274184068*v_W3mv_b_r + 1.70260376761891*v_W3mv_c_i + 2.07555506909734*v_W3mv_c_r
        struct[0].g[69,0] = -i_l_W2mv_W3mv_b_i - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r + 7.26444274184068*v_W2mv_b_i - 5.95911318666618*v_W2mv_b_r - 2.07555506909734*v_W2mv_c_i + 1.70260376761891*v_W2mv_c_r + 2.07555506909734*v_W3mv_a_i - 1.70260376761891*v_W3mv_a_r - 7.26444274184068*v_W3mv_b_i + 5.95911318666618*v_W3mv_b_r + 2.07555506909734*v_W3mv_c_i - 1.70260376761891*v_W3mv_c_r
        struct[0].g[70,0] = -i_l_W2mv_W3mv_c_r - 1.70260376761891*v_W2mv_a_i - 2.07555506909734*v_W2mv_a_r - 1.70260376761891*v_W2mv_b_i - 2.07555506909734*v_W2mv_b_r + 5.95911318666618*v_W2mv_c_i + 7.26444274184068*v_W2mv_c_r + 1.70260376761891*v_W3mv_a_i + 2.07555506909734*v_W3mv_a_r + 1.70260376761891*v_W3mv_b_i + 2.07555506909734*v_W3mv_b_r - 5.95911318666618*v_W3mv_c_i - 7.26444274184068*v_W3mv_c_r
        struct[0].g[71,0] = -i_l_W2mv_W3mv_c_i - 2.07555506909734*v_W2mv_a_i + 1.70260376761891*v_W2mv_a_r - 2.07555506909734*v_W2mv_b_i + 1.70260376761891*v_W2mv_b_r + 7.26444274184068*v_W2mv_c_i - 5.95911318666618*v_W2mv_c_r + 2.07555506909734*v_W3mv_a_i - 1.70260376761891*v_W3mv_a_r + 2.07555506909734*v_W3mv_b_i - 1.70260376761891*v_W3mv_b_r - 7.26444274184068*v_W3mv_c_i + 5.95911318666618*v_W3mv_c_r
        struct[0].g[72,0] = -i_l_W3mv_POImv_a_r - 5.95911318666618*v_POImv_a_i - 7.26444274184068*v_POImv_a_r + 1.70260376761891*v_POImv_b_i + 2.07555506909734*v_POImv_b_r + 1.70260376761891*v_POImv_c_i + 2.07555506909734*v_POImv_c_r + 5.95911318666618*v_W3mv_a_i + 7.26444274184068*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[73,0] = -i_l_W3mv_POImv_a_i - 7.26444274184068*v_POImv_a_i + 5.95911318666618*v_POImv_a_r + 2.07555506909734*v_POImv_b_i - 1.70260376761891*v_POImv_b_r + 2.07555506909734*v_POImv_c_i - 1.70260376761891*v_POImv_c_r + 7.26444274184068*v_W3mv_a_i - 5.95911318666618*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[74,0] = -i_l_W3mv_POImv_b_r + 1.70260376761891*v_POImv_a_i + 2.07555506909734*v_POImv_a_r - 5.95911318666618*v_POImv_b_i - 7.26444274184068*v_POImv_b_r + 1.70260376761891*v_POImv_c_i + 2.07555506909734*v_POImv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r + 5.95911318666618*v_W3mv_b_i + 7.26444274184068*v_W3mv_b_r - 1.70260376761891*v_W3mv_c_i - 2.07555506909734*v_W3mv_c_r
        struct[0].g[75,0] = -i_l_W3mv_POImv_b_i + 2.07555506909734*v_POImv_a_i - 1.70260376761891*v_POImv_a_r - 7.26444274184068*v_POImv_b_i + 5.95911318666618*v_POImv_b_r + 2.07555506909734*v_POImv_c_i - 1.70260376761891*v_POImv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r + 7.26444274184068*v_W3mv_b_i - 5.95911318666618*v_W3mv_b_r - 2.07555506909734*v_W3mv_c_i + 1.70260376761891*v_W3mv_c_r
        struct[0].g[76,0] = -i_l_W3mv_POImv_c_r + 1.70260376761891*v_POImv_a_i + 2.07555506909734*v_POImv_a_r + 1.70260376761891*v_POImv_b_i + 2.07555506909734*v_POImv_b_r - 5.95911318666618*v_POImv_c_i - 7.26444274184068*v_POImv_c_r - 1.70260376761891*v_W3mv_a_i - 2.07555506909734*v_W3mv_a_r - 1.70260376761891*v_W3mv_b_i - 2.07555506909734*v_W3mv_b_r + 5.95911318666618*v_W3mv_c_i + 7.26444274184068*v_W3mv_c_r
        struct[0].g[77,0] = -i_l_W3mv_POImv_c_i + 2.07555506909734*v_POImv_a_i - 1.70260376761891*v_POImv_a_r + 2.07555506909734*v_POImv_b_i - 1.70260376761891*v_POImv_b_r - 7.26444274184068*v_POImv_c_i + 5.95911318666618*v_POImv_c_r - 2.07555506909734*v_W3mv_a_i + 1.70260376761891*v_W3mv_a_r - 2.07555506909734*v_W3mv_b_i + 1.70260376761891*v_W3mv_b_r + 7.26444274184068*v_W3mv_c_i - 5.95911318666618*v_W3mv_c_r
        struct[0].g[78,0] = -i_l_STmv_POImv_a_r - 148.977829666654*v_POImv_a_i - 181.611068546017*v_POImv_a_r + 42.5650941904727*v_POImv_b_i + 51.8888767274334*v_POImv_b_r + 42.5650941904727*v_POImv_c_i + 51.8888767274334*v_POImv_c_r + 148.977829666654*v_STmv_a_i + 181.611068546017*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274334*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274334*v_STmv_c_r
        struct[0].g[79,0] = -i_l_STmv_POImv_a_i - 181.611068546017*v_POImv_a_i + 148.977829666654*v_POImv_a_r + 51.8888767274334*v_POImv_b_i - 42.5650941904727*v_POImv_b_r + 51.8888767274334*v_POImv_c_i - 42.5650941904727*v_POImv_c_r + 181.611068546017*v_STmv_a_i - 148.977829666654*v_STmv_a_r - 51.8888767274334*v_STmv_b_i + 42.5650941904727*v_STmv_b_r - 51.8888767274334*v_STmv_c_i + 42.5650941904727*v_STmv_c_r
        struct[0].g[80,0] = -i_l_STmv_POImv_b_r + 42.5650941904727*v_POImv_a_i + 51.8888767274334*v_POImv_a_r - 148.977829666654*v_POImv_b_i - 181.611068546017*v_POImv_b_r + 42.5650941904727*v_POImv_c_i + 51.8888767274335*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r + 148.977829666654*v_STmv_b_i + 181.611068546017*v_STmv_b_r - 42.5650941904727*v_STmv_c_i - 51.8888767274335*v_STmv_c_r
        struct[0].g[81,0] = -i_l_STmv_POImv_b_i + 51.8888767274334*v_POImv_a_i - 42.5650941904727*v_POImv_a_r - 181.611068546017*v_POImv_b_i + 148.977829666654*v_POImv_b_r + 51.8888767274335*v_POImv_c_i - 42.5650941904727*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r + 181.611068546017*v_STmv_b_i - 148.977829666654*v_STmv_b_r - 51.8888767274335*v_STmv_c_i + 42.5650941904727*v_STmv_c_r
        struct[0].g[82,0] = -i_l_STmv_POImv_c_r + 42.5650941904727*v_POImv_a_i + 51.8888767274334*v_POImv_a_r + 42.5650941904727*v_POImv_b_i + 51.8888767274335*v_POImv_b_r - 148.977829666654*v_POImv_c_i - 181.611068546017*v_POImv_c_r - 42.5650941904727*v_STmv_a_i - 51.8888767274334*v_STmv_a_r - 42.5650941904727*v_STmv_b_i - 51.8888767274335*v_STmv_b_r + 148.977829666654*v_STmv_c_i + 181.611068546017*v_STmv_c_r
        struct[0].g[83,0] = -i_l_STmv_POImv_c_i + 51.8888767274334*v_POImv_a_i - 42.5650941904727*v_POImv_a_r + 51.8888767274335*v_POImv_b_i - 42.5650941904727*v_POImv_b_r - 181.611068546017*v_POImv_c_i + 148.977829666654*v_POImv_c_r - 51.8888767274334*v_STmv_a_i + 42.5650941904727*v_STmv_a_r - 51.8888767274335*v_STmv_b_i + 42.5650941904727*v_STmv_b_r + 181.611068546017*v_STmv_c_i - 148.977829666654*v_STmv_c_r
        struct[0].g[84,0] = -i_l_POI_GRID_a_r - 0.040290088638195*v_GRID_a_i - 0.024174053182917*v_GRID_a_r - 4.66248501556824e-18*v_GRID_b_i + 4.31760362252812e-18*v_GRID_b_r - 4.19816664496737e-18*v_GRID_c_i + 3.49608108880335e-18*v_GRID_c_r + 0.040290088638195*v_POI_a_i + 0.024174053182917*v_POI_a_r + 4.66248501556824e-18*v_POI_b_i - 4.31760362252812e-18*v_POI_b_r + 4.19816664496737e-18*v_POI_c_i - 3.49608108880335e-18*v_POI_c_r
        struct[0].g[85,0] = -i_l_POI_GRID_a_i - 0.024174053182917*v_GRID_a_i + 0.040290088638195*v_GRID_a_r + 4.31760362252812e-18*v_GRID_b_i + 4.66248501556824e-18*v_GRID_b_r + 3.49608108880335e-18*v_GRID_c_i + 4.19816664496737e-18*v_GRID_c_r + 0.024174053182917*v_POI_a_i - 0.040290088638195*v_POI_a_r - 4.31760362252812e-18*v_POI_b_i - 4.66248501556824e-18*v_POI_b_r - 3.49608108880335e-18*v_POI_c_i - 4.19816664496737e-18*v_POI_c_r
        struct[0].g[86,0] = -i_l_POI_GRID_b_r - 6.30775359573304e-19*v_GRID_a_i + 2.07254761002657e-18*v_GRID_a_r - 0.040290088638195*v_GRID_b_i - 0.024174053182917*v_GRID_b_r - 9.01107656533306e-19*v_GRID_c_i + 1.78419315993592e-17*v_GRID_c_r + 6.30775359573304e-19*v_POI_a_i - 2.07254761002657e-18*v_POI_a_r + 0.040290088638195*v_POI_b_i + 0.024174053182917*v_POI_b_r + 9.01107656533306e-19*v_POI_c_i - 1.78419315993592e-17*v_POI_c_r
        struct[0].g[87,0] = -i_l_POI_GRID_b_i + 2.07254761002657e-18*v_GRID_a_i + 6.30775359573304e-19*v_GRID_a_r - 0.024174053182917*v_GRID_b_i + 0.040290088638195*v_GRID_b_r + 1.78419315993592e-17*v_GRID_c_i + 9.01107656533306e-19*v_GRID_c_r - 2.07254761002657e-18*v_POI_a_i - 6.30775359573304e-19*v_POI_a_r + 0.024174053182917*v_POI_b_i - 0.040290088638195*v_POI_b_r - 1.78419315993592e-17*v_POI_c_i - 9.01107656533306e-19*v_POI_c_r
        struct[0].g[88,0] = -i_l_POI_GRID_c_r + 7.20886125226632e-19*v_GRID_a_i + 1.35166148479994e-18*v_GRID_a_r + 4.50553828266631e-19*v_GRID_b_i + 1.71210454741325e-17*v_GRID_b_r - 0.040290088638195*v_GRID_c_i - 0.024174053182917*v_GRID_c_r - 7.20886125226632e-19*v_POI_a_i - 1.35166148479994e-18*v_POI_a_r - 4.50553828266631e-19*v_POI_b_i - 1.71210454741325e-17*v_POI_b_r + 0.040290088638195*v_POI_c_i + 0.024174053182917*v_POI_c_r
        struct[0].g[89,0] = -i_l_POI_GRID_c_i + 1.35166148479994e-18*v_GRID_a_i - 7.20886125226632e-19*v_GRID_a_r + 1.71210454741325e-17*v_GRID_b_i - 4.50553828266631e-19*v_GRID_b_r - 0.024174053182917*v_GRID_c_i + 0.040290088638195*v_GRID_c_r - 1.35166148479994e-18*v_POI_a_i + 7.20886125226632e-19*v_POI_a_r - 1.71210454741325e-17*v_POI_b_i + 4.50553828266631e-19*v_POI_b_r + 0.024174053182917*v_POI_c_i - 0.040290088638195*v_POI_c_r
        struct[0].g[90,0] = i_W1lv_a_i*v_W1lv_a_i + i_W1lv_a_r*v_W1lv_a_r - p_W1lv_a
        struct[0].g[91,0] = i_W1lv_b_i*v_W1lv_b_i + i_W1lv_b_r*v_W1lv_b_r - p_W1lv_b
        struct[0].g[92,0] = i_W1lv_c_i*v_W1lv_c_i + i_W1lv_c_r*v_W1lv_c_r - p_W1lv_c
        struct[0].g[93,0] = -i_W1lv_a_i*v_W1lv_a_r + i_W1lv_a_r*v_W1lv_a_i - q_W1lv_a
        struct[0].g[94,0] = -i_W1lv_b_i*v_W1lv_b_r + i_W1lv_b_r*v_W1lv_b_i - q_W1lv_b
        struct[0].g[95,0] = -i_W1lv_c_i*v_W1lv_c_r + i_W1lv_c_r*v_W1lv_c_i - q_W1lv_c
        struct[0].g[96,0] = -v_m_W1lv + (v_W1lv_a_i**2 + v_W1lv_a_r**2)**0.5/V_base_W1lv
        struct[0].g[97,0] = -v_m_W1mv + (v_W1mv_a_i**2 + v_W1mv_a_r**2)**0.5/V_base_W1mv
        struct[0].g[98,0] = Dq_r_W1lv + K_p_v_W1lv*(Dv_r_W1lv - u_ctrl_v_W1lv*v_m_W1mv + v_loc_ref_W1lv - v_m_W1lv*(1.0 - u_ctrl_v_W1lv)) - i_reac_ref_W1lv
        struct[0].g[99,0] = S_base_W1lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)])) - q_ref_W1lv
        struct[0].g[100,0] = i_W2lv_a_i*v_W2lv_a_i + i_W2lv_a_r*v_W2lv_a_r - p_W2lv_a
        struct[0].g[101,0] = i_W2lv_b_i*v_W2lv_b_i + i_W2lv_b_r*v_W2lv_b_r - p_W2lv_b
        struct[0].g[102,0] = i_W2lv_c_i*v_W2lv_c_i + i_W2lv_c_r*v_W2lv_c_r - p_W2lv_c
        struct[0].g[103,0] = -i_W2lv_a_i*v_W2lv_a_r + i_W2lv_a_r*v_W2lv_a_i - q_W2lv_a
        struct[0].g[104,0] = -i_W2lv_b_i*v_W2lv_b_r + i_W2lv_b_r*v_W2lv_b_i - q_W2lv_b
        struct[0].g[105,0] = -i_W2lv_c_i*v_W2lv_c_r + i_W2lv_c_r*v_W2lv_c_i - q_W2lv_c
        struct[0].g[106,0] = -v_m_W2lv + (v_W2lv_a_i**2 + v_W2lv_a_r**2)**0.5/V_base_W2lv
        struct[0].g[107,0] = -v_m_W2mv + (v_W2mv_a_i**2 + v_W2mv_a_r**2)**0.5/V_base_W2mv
        struct[0].g[108,0] = Dq_r_W2lv + K_p_v_W2lv*(Dv_r_W2lv - u_ctrl_v_W2lv*v_m_W2mv + v_loc_ref_W2lv - v_m_W2lv*(1.0 - u_ctrl_v_W2lv)) - i_reac_ref_W2lv
        struct[0].g[109,0] = S_base_W2lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)])) - q_ref_W2lv
        struct[0].g[110,0] = i_W3lv_a_i*v_W3lv_a_i + i_W3lv_a_r*v_W3lv_a_r - p_W3lv_a
        struct[0].g[111,0] = i_W3lv_b_i*v_W3lv_b_i + i_W3lv_b_r*v_W3lv_b_r - p_W3lv_b
        struct[0].g[112,0] = i_W3lv_c_i*v_W3lv_c_i + i_W3lv_c_r*v_W3lv_c_r - p_W3lv_c
        struct[0].g[113,0] = -i_W3lv_a_i*v_W3lv_a_r + i_W3lv_a_r*v_W3lv_a_i - q_W3lv_a
        struct[0].g[114,0] = -i_W3lv_b_i*v_W3lv_b_r + i_W3lv_b_r*v_W3lv_b_i - q_W3lv_b
        struct[0].g[115,0] = -i_W3lv_c_i*v_W3lv_c_r + i_W3lv_c_r*v_W3lv_c_i - q_W3lv_c
        struct[0].g[116,0] = -v_m_W3lv + (v_W3lv_a_i**2 + v_W3lv_a_r**2)**0.5/V_base_W3lv
        struct[0].g[117,0] = -v_m_W3mv + (v_W3mv_a_i**2 + v_W3mv_a_r**2)**0.5/V_base_W3mv
        struct[0].g[118,0] = Dq_r_W3lv + K_p_v_W3lv*(Dv_r_W3lv - u_ctrl_v_W3lv*v_m_W3mv + v_loc_ref_W3lv - v_m_W3lv*(1.0 - u_ctrl_v_W3lv)) - i_reac_ref_W3lv
        struct[0].g[119,0] = S_base_W3lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)])) - q_ref_W3lv
        struct[0].g[120,0] = i_STlv_a_i*v_STlv_a_i + i_STlv_a_r*v_STlv_a_r - p_STlv_a
        struct[0].g[121,0] = i_STlv_b_i*v_STlv_b_i + i_STlv_b_r*v_STlv_b_r - p_STlv_b
        struct[0].g[122,0] = i_STlv_c_i*v_STlv_c_i + i_STlv_c_r*v_STlv_c_r - p_STlv_c
        struct[0].g[123,0] = -i_STlv_a_i*v_STlv_a_r + i_STlv_a_r*v_STlv_a_i - q_STlv_a
        struct[0].g[124,0] = -i_STlv_b_i*v_STlv_b_r + i_STlv_b_r*v_STlv_b_i - q_STlv_b
        struct[0].g[125,0] = -i_STlv_c_i*v_STlv_c_r + i_STlv_c_r*v_STlv_c_i - q_STlv_c
        struct[0].g[126,0] = -v_m_STlv + (v_STlv_a_i**2 + v_STlv_a_r**2)**0.5/V_base_STlv
        struct[0].g[127,0] = -v_m_STmv + (v_STmv_a_i**2 + v_STmv_a_r**2)**0.5/V_base_STmv
        struct[0].g[128,0] = Dq_r_STlv + K_p_v_STlv*(Dv_r_STlv - u_ctrl_v_STlv*v_m_STmv + v_loc_ref_STlv - v_m_STlv*(1.0 - u_ctrl_v_STlv)) - i_reac_ref_STlv
        struct[0].g[129,0] = S_base_STlv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)])) - q_ref_STlv
    
    # Outputs:
    if mode == 3:

    
        pass

    if mode == 10:

        struct[0].Fx[0,0] = -1/T_pq_W1lv
        struct[0].Fx[1,1] = -1/T_pq_W1lv
        struct[0].Fx[2,2] = -1/T_pq_W1lv
        struct[0].Fx[3,3] = -1/T_pq_W1lv
        struct[0].Fx[4,4] = -1/T_pq_W1lv
        struct[0].Fx[5,5] = -1/T_pq_W1lv
        struct[0].Fx[6,6] = -1/T_pq_W2lv
        struct[0].Fx[7,7] = -1/T_pq_W2lv
        struct[0].Fx[8,8] = -1/T_pq_W2lv
        struct[0].Fx[9,9] = -1/T_pq_W2lv
        struct[0].Fx[10,10] = -1/T_pq_W2lv
        struct[0].Fx[11,11] = -1/T_pq_W2lv
        struct[0].Fx[12,12] = -1/T_pq_W3lv
        struct[0].Fx[13,13] = -1/T_pq_W3lv
        struct[0].Fx[14,14] = -1/T_pq_W3lv
        struct[0].Fx[15,15] = -1/T_pq_W3lv
        struct[0].Fx[16,16] = -1/T_pq_W3lv
        struct[0].Fx[17,17] = -1/T_pq_W3lv
        struct[0].Fx[18,18] = -1/T_pq_STlv
        struct[0].Fx[19,19] = -1/T_pq_STlv
        struct[0].Fx[20,20] = -1/T_pq_STlv
        struct[0].Fx[21,21] = -1/T_pq_STlv
        struct[0].Fx[22,22] = -1/T_pq_STlv
        struct[0].Fx[23,23] = -1/T_pq_STlv

    if mode == 11:

        struct[0].Fy[3,99] = 1/(3*T_pq_W1lv)
        struct[0].Fy[4,99] = 1/(3*T_pq_W1lv)
        struct[0].Fy[5,99] = 1/(3*T_pq_W1lv)
        struct[0].Fy[9,109] = 1/(3*T_pq_W2lv)
        struct[0].Fy[10,109] = 1/(3*T_pq_W2lv)
        struct[0].Fy[11,109] = 1/(3*T_pq_W2lv)
        struct[0].Fy[15,119] = 1/(3*T_pq_W3lv)
        struct[0].Fy[16,119] = 1/(3*T_pq_W3lv)
        struct[0].Fy[17,119] = 1/(3*T_pq_W3lv)
        struct[0].Fy[21,129] = 1/(3*T_pq_STlv)
        struct[0].Fy[22,129] = 1/(3*T_pq_STlv)
        struct[0].Fy[23,129] = 1/(3*T_pq_STlv)

        struct[0].Gy[0,0] = -14.1918856474622
        struct[0].Gy[0,1] = -85.1513138847732
        struct[0].Gy[0,36] = 0.282682270461039
        struct[0].Gy[0,37] = 1.69609362276623
        struct[0].Gy[0,40] = -0.282682270461039
        struct[0].Gy[0,41] = -1.69609362276623
        struct[0].Gy[0,90] = 1
        struct[0].Gy[1,0] = 85.1513138847732
        struct[0].Gy[1,1] = -14.1918856474622
        struct[0].Gy[1,36] = -1.69609362276623
        struct[0].Gy[1,37] = 0.282682270461039
        struct[0].Gy[1,40] = 1.69609362276623
        struct[0].Gy[1,41] = -0.282682270461039
        struct[0].Gy[1,91] = 1
        struct[0].Gy[2,2] = -14.1918856474622
        struct[0].Gy[2,3] = -85.1513138847732
        struct[0].Gy[2,36] = -0.282682270461039
        struct[0].Gy[2,37] = -1.69609362276623
        struct[0].Gy[2,38] = 0.282682270461039
        struct[0].Gy[2,39] = 1.69609362276623
        struct[0].Gy[2,92] = 1
        struct[0].Gy[3,2] = 85.1513138847732
        struct[0].Gy[3,3] = -14.1918856474622
        struct[0].Gy[3,36] = 1.69609362276623
        struct[0].Gy[3,37] = -0.282682270461039
        struct[0].Gy[3,38] = -1.69609362276623
        struct[0].Gy[3,39] = 0.282682270461039
        struct[0].Gy[3,93] = 1
        struct[0].Gy[4,4] = -14.1918856474622
        struct[0].Gy[4,5] = -85.1513138847732
        struct[0].Gy[4,38] = -0.282682270461039
        struct[0].Gy[4,39] = -1.69609362276623
        struct[0].Gy[4,40] = 0.282682270461039
        struct[0].Gy[4,41] = 1.69609362276623
        struct[0].Gy[4,94] = 1
        struct[0].Gy[5,4] = 85.1513138847732
        struct[0].Gy[5,5] = -14.1918856474622
        struct[0].Gy[5,38] = 1.69609362276623
        struct[0].Gy[5,39] = -0.282682270461039
        struct[0].Gy[5,40] = -1.69609362276623
        struct[0].Gy[5,41] = 0.282682270461039
        struct[0].Gy[5,95] = 1
        struct[0].Gy[6,6] = -14.1918856474622
        struct[0].Gy[6,7] = -85.1513138847732
        struct[0].Gy[6,42] = 0.282682270461039
        struct[0].Gy[6,43] = 1.69609362276623
        struct[0].Gy[6,46] = -0.282682270461039
        struct[0].Gy[6,47] = -1.69609362276623
        struct[0].Gy[6,100] = 1
        struct[0].Gy[7,6] = 85.1513138847732
        struct[0].Gy[7,7] = -14.1918856474622
        struct[0].Gy[7,42] = -1.69609362276623
        struct[0].Gy[7,43] = 0.282682270461039
        struct[0].Gy[7,46] = 1.69609362276623
        struct[0].Gy[7,47] = -0.282682270461039
        struct[0].Gy[7,101] = 1
        struct[0].Gy[8,8] = -14.1918856474622
        struct[0].Gy[8,9] = -85.1513138847732
        struct[0].Gy[8,42] = -0.282682270461039
        struct[0].Gy[8,43] = -1.69609362276623
        struct[0].Gy[8,44] = 0.282682270461039
        struct[0].Gy[8,45] = 1.69609362276623
        struct[0].Gy[8,102] = 1
        struct[0].Gy[9,8] = 85.1513138847732
        struct[0].Gy[9,9] = -14.1918856474622
        struct[0].Gy[9,42] = 1.69609362276623
        struct[0].Gy[9,43] = -0.282682270461039
        struct[0].Gy[9,44] = -1.69609362276623
        struct[0].Gy[9,45] = 0.282682270461039
        struct[0].Gy[9,103] = 1
        struct[0].Gy[10,10] = -14.1918856474622
        struct[0].Gy[10,11] = -85.1513138847732
        struct[0].Gy[10,44] = -0.282682270461039
        struct[0].Gy[10,45] = -1.69609362276623
        struct[0].Gy[10,46] = 0.282682270461039
        struct[0].Gy[10,47] = 1.69609362276623
        struct[0].Gy[10,104] = 1
        struct[0].Gy[11,10] = 85.1513138847732
        struct[0].Gy[11,11] = -14.1918856474622
        struct[0].Gy[11,44] = 1.69609362276623
        struct[0].Gy[11,45] = -0.282682270461039
        struct[0].Gy[11,46] = -1.69609362276623
        struct[0].Gy[11,47] = 0.282682270461039
        struct[0].Gy[11,105] = 1
        struct[0].Gy[12,12] = -14.1918856474622
        struct[0].Gy[12,13] = -85.1513138847732
        struct[0].Gy[12,48] = 0.282682270461039
        struct[0].Gy[12,49] = 1.69609362276623
        struct[0].Gy[12,52] = -0.282682270461039
        struct[0].Gy[12,53] = -1.69609362276623
        struct[0].Gy[12,110] = 1
        struct[0].Gy[13,12] = 85.1513138847732
        struct[0].Gy[13,13] = -14.1918856474622
        struct[0].Gy[13,48] = -1.69609362276623
        struct[0].Gy[13,49] = 0.282682270461039
        struct[0].Gy[13,52] = 1.69609362276623
        struct[0].Gy[13,53] = -0.282682270461039
        struct[0].Gy[13,111] = 1
        struct[0].Gy[14,14] = -14.1918856474622
        struct[0].Gy[14,15] = -85.1513138847732
        struct[0].Gy[14,48] = -0.282682270461039
        struct[0].Gy[14,49] = -1.69609362276623
        struct[0].Gy[14,50] = 0.282682270461039
        struct[0].Gy[14,51] = 1.69609362276623
        struct[0].Gy[14,112] = 1
        struct[0].Gy[15,14] = 85.1513138847732
        struct[0].Gy[15,15] = -14.1918856474622
        struct[0].Gy[15,48] = 1.69609362276623
        struct[0].Gy[15,49] = -0.282682270461039
        struct[0].Gy[15,50] = -1.69609362276623
        struct[0].Gy[15,51] = 0.282682270461039
        struct[0].Gy[15,113] = 1
        struct[0].Gy[16,16] = -14.1918856474622
        struct[0].Gy[16,17] = -85.1513138847732
        struct[0].Gy[16,50] = -0.282682270461039
        struct[0].Gy[16,51] = -1.69609362276623
        struct[0].Gy[16,52] = 0.282682270461039
        struct[0].Gy[16,53] = 1.69609362276623
        struct[0].Gy[16,114] = 1
        struct[0].Gy[17,16] = 85.1513138847732
        struct[0].Gy[17,17] = -14.1918856474622
        struct[0].Gy[17,50] = 1.69609362276623
        struct[0].Gy[17,51] = -0.282682270461039
        struct[0].Gy[17,52] = -1.69609362276623
        struct[0].Gy[17,53] = 0.282682270461039
        struct[0].Gy[17,115] = 1
        struct[0].Gy[18,18] = -14.1918856474622
        struct[0].Gy[18,19] = -85.1513138847732
        struct[0].Gy[18,54] = 0.282682270461039
        struct[0].Gy[18,55] = 1.69609362276623
        struct[0].Gy[18,58] = -0.282682270461039
        struct[0].Gy[18,59] = -1.69609362276623
        struct[0].Gy[18,120] = 1
        struct[0].Gy[19,18] = 85.1513138847732
        struct[0].Gy[19,19] = -14.1918856474622
        struct[0].Gy[19,54] = -1.69609362276623
        struct[0].Gy[19,55] = 0.282682270461039
        struct[0].Gy[19,58] = 1.69609362276623
        struct[0].Gy[19,59] = -0.282682270461039
        struct[0].Gy[19,121] = 1
        struct[0].Gy[20,20] = -14.1918856474622
        struct[0].Gy[20,21] = -85.1513138847732
        struct[0].Gy[20,54] = -0.282682270461039
        struct[0].Gy[20,55] = -1.69609362276623
        struct[0].Gy[20,56] = 0.282682270461039
        struct[0].Gy[20,57] = 1.69609362276623
        struct[0].Gy[20,122] = 1
        struct[0].Gy[21,20] = 85.1513138847732
        struct[0].Gy[21,21] = -14.1918856474622
        struct[0].Gy[21,54] = 1.69609362276623
        struct[0].Gy[21,55] = -0.282682270461039
        struct[0].Gy[21,56] = -1.69609362276623
        struct[0].Gy[21,57] = 0.282682270461039
        struct[0].Gy[21,123] = 1
        struct[0].Gy[22,22] = -14.1918856474622
        struct[0].Gy[22,23] = -85.1513138847732
        struct[0].Gy[22,56] = -0.282682270461039
        struct[0].Gy[22,57] = -1.69609362276623
        struct[0].Gy[22,58] = 0.282682270461039
        struct[0].Gy[22,59] = 1.69609362276623
        struct[0].Gy[22,124] = 1
        struct[0].Gy[23,22] = 85.1513138847732
        struct[0].Gy[23,23] = -14.1918856474622
        struct[0].Gy[23,56] = 1.69609362276623
        struct[0].Gy[23,57] = -0.282682270461039
        struct[0].Gy[23,58] = -1.69609362276623
        struct[0].Gy[23,59] = 0.282682270461039
        struct[0].Gy[23,125] = 1
        struct[0].Gy[24,24] = -0.0265286009920103
        struct[0].Gy[24,25] = -0.0591264711109411
        struct[0].Gy[24,26] = 0.00117727390454664
        struct[0].Gy[24,27] = 0.00941819123637305
        struct[0].Gy[24,28] = 0.00117727390454664
        struct[0].Gy[24,29] = 0.00941819123637305
        struct[0].Gy[24,30] = 0.00672902411642920
        struct[0].Gy[24,31] = 0.0538321929314336
        struct[0].Gy[24,32] = -0.00672902411642920
        struct[0].Gy[24,33] = -0.0538321929314336
        struct[0].Gy[25,24] = 0.0591264711109411
        struct[0].Gy[25,25] = -0.0265286009920103
        struct[0].Gy[25,26] = -0.00941819123637305
        struct[0].Gy[25,27] = 0.00117727390454664
        struct[0].Gy[25,28] = -0.00941819123637305
        struct[0].Gy[25,29] = 0.00117727390454664
        struct[0].Gy[25,30] = -0.0538321929314336
        struct[0].Gy[25,31] = 0.00672902411642920
        struct[0].Gy[25,32] = 0.0538321929314336
        struct[0].Gy[25,33] = -0.00672902411642920
        struct[0].Gy[26,24] = 0.00117727390454663
        struct[0].Gy[26,25] = 0.00941819123637305
        struct[0].Gy[26,26] = -0.0265286009920103
        struct[0].Gy[26,27] = -0.0591264711109411
        struct[0].Gy[26,28] = 0.00117727390454665
        struct[0].Gy[26,29] = 0.00941819123637305
        struct[0].Gy[26,32] = 0.00672902411642920
        struct[0].Gy[26,33] = 0.0538321929314336
        struct[0].Gy[26,34] = -0.00672902411642920
        struct[0].Gy[26,35] = -0.0538321929314336
        struct[0].Gy[27,24] = -0.00941819123637305
        struct[0].Gy[27,25] = 0.00117727390454663
        struct[0].Gy[27,26] = 0.0591264711109411
        struct[0].Gy[27,27] = -0.0265286009920103
        struct[0].Gy[27,28] = -0.00941819123637305
        struct[0].Gy[27,29] = 0.00117727390454665
        struct[0].Gy[27,32] = -0.0538321929314336
        struct[0].Gy[27,33] = 0.00672902411642920
        struct[0].Gy[27,34] = 0.0538321929314336
        struct[0].Gy[27,35] = -0.00672902411642920
        struct[0].Gy[28,24] = 0.00117727390454663
        struct[0].Gy[28,25] = 0.00941819123637305
        struct[0].Gy[28,26] = 0.00117727390454665
        struct[0].Gy[28,27] = 0.00941819123637305
        struct[0].Gy[28,28] = -0.0265286009920103
        struct[0].Gy[28,29] = -0.0591264711109411
        struct[0].Gy[28,30] = -0.00672902411642920
        struct[0].Gy[28,31] = -0.0538321929314336
        struct[0].Gy[28,34] = 0.00672902411642920
        struct[0].Gy[28,35] = 0.0538321929314336
        struct[0].Gy[29,24] = -0.00941819123637305
        struct[0].Gy[29,25] = 0.00117727390454663
        struct[0].Gy[29,26] = -0.00941819123637305
        struct[0].Gy[29,27] = 0.00117727390454665
        struct[0].Gy[29,28] = 0.0591264711109411
        struct[0].Gy[29,29] = -0.0265286009920103
        struct[0].Gy[29,30] = 0.0538321929314336
        struct[0].Gy[29,31] = -0.00672902411642920
        struct[0].Gy[29,34] = -0.0538321929314336
        struct[0].Gy[29,35] = 0.00672902411642920
        struct[0].Gy[30,24] = 0.00672902411642920
        struct[0].Gy[30,25] = 0.0538321929314336
        struct[0].Gy[30,28] = -0.00672902411642920
        struct[0].Gy[30,29] = -0.0538321929314336
        struct[0].Gy[30,30] = -188.924390492986
        struct[0].Gy[30,31] = -155.244588874881
        struct[0].Gy[30,32] = 53.9540151298641
        struct[0].Gy[30,33] = 44.2677164725443
        struct[0].Gy[30,34] = 53.9540151298641
        struct[0].Gy[30,35] = 44.2677164725443
        struct[0].Gy[30,48] = 7.26444274184068
        struct[0].Gy[30,49] = 5.95911318666618
        struct[0].Gy[30,50] = -2.07555506909734
        struct[0].Gy[30,51] = -1.70260376761891
        struct[0].Gy[30,52] = -2.07555506909734
        struct[0].Gy[30,53] = -1.70260376761891
        struct[0].Gy[30,54] = 181.611068546017
        struct[0].Gy[30,55] = 148.977829666654
        struct[0].Gy[30,56] = -51.8888767274334
        struct[0].Gy[30,57] = -42.5650941904727
        struct[0].Gy[30,58] = -51.8888767274334
        struct[0].Gy[30,59] = -42.5650941904727
        struct[0].Gy[31,24] = -0.0538321929314336
        struct[0].Gy[31,25] = 0.00672902411642920
        struct[0].Gy[31,28] = 0.0538321929314336
        struct[0].Gy[31,29] = -0.00672902411642920
        struct[0].Gy[31,30] = 155.244588874881
        struct[0].Gy[31,31] = -188.924390492986
        struct[0].Gy[31,32] = -44.2677164725443
        struct[0].Gy[31,33] = 53.9540151298641
        struct[0].Gy[31,34] = -44.2677164725443
        struct[0].Gy[31,35] = 53.9540151298641
        struct[0].Gy[31,48] = -5.95911318666618
        struct[0].Gy[31,49] = 7.26444274184068
        struct[0].Gy[31,50] = 1.70260376761891
        struct[0].Gy[31,51] = -2.07555506909734
        struct[0].Gy[31,52] = 1.70260376761891
        struct[0].Gy[31,53] = -2.07555506909734
        struct[0].Gy[31,54] = -148.977829666654
        struct[0].Gy[31,55] = 181.611068546017
        struct[0].Gy[31,56] = 42.5650941904727
        struct[0].Gy[31,57] = -51.8888767274334
        struct[0].Gy[31,58] = 42.5650941904727
        struct[0].Gy[31,59] = -51.8888767274334
        struct[0].Gy[32,24] = -0.00672902411642920
        struct[0].Gy[32,25] = -0.0538321929314336
        struct[0].Gy[32,26] = 0.00672902411642920
        struct[0].Gy[32,27] = 0.0538321929314336
        struct[0].Gy[32,30] = 53.9540151298641
        struct[0].Gy[32,31] = 44.2677164725443
        struct[0].Gy[32,32] = -188.924390492986
        struct[0].Gy[32,33] = -155.244588874881
        struct[0].Gy[32,34] = 53.9540151298642
        struct[0].Gy[32,35] = 44.2677164725443
        struct[0].Gy[32,48] = -2.07555506909734
        struct[0].Gy[32,49] = -1.70260376761891
        struct[0].Gy[32,50] = 7.26444274184068
        struct[0].Gy[32,51] = 5.95911318666618
        struct[0].Gy[32,52] = -2.07555506909734
        struct[0].Gy[32,53] = -1.70260376761891
        struct[0].Gy[32,54] = -51.8888767274334
        struct[0].Gy[32,55] = -42.5650941904727
        struct[0].Gy[32,56] = 181.611068546017
        struct[0].Gy[32,57] = 148.977829666654
        struct[0].Gy[32,58] = -51.8888767274335
        struct[0].Gy[32,59] = -42.5650941904727
        struct[0].Gy[33,24] = 0.0538321929314336
        struct[0].Gy[33,25] = -0.00672902411642920
        struct[0].Gy[33,26] = -0.0538321929314336
        struct[0].Gy[33,27] = 0.00672902411642920
        struct[0].Gy[33,30] = -44.2677164725443
        struct[0].Gy[33,31] = 53.9540151298641
        struct[0].Gy[33,32] = 155.244588874881
        struct[0].Gy[33,33] = -188.924390492986
        struct[0].Gy[33,34] = -44.2677164725443
        struct[0].Gy[33,35] = 53.9540151298642
        struct[0].Gy[33,48] = 1.70260376761891
        struct[0].Gy[33,49] = -2.07555506909734
        struct[0].Gy[33,50] = -5.95911318666618
        struct[0].Gy[33,51] = 7.26444274184068
        struct[0].Gy[33,52] = 1.70260376761891
        struct[0].Gy[33,53] = -2.07555506909734
        struct[0].Gy[33,54] = 42.5650941904727
        struct[0].Gy[33,55] = -51.8888767274334
        struct[0].Gy[33,56] = -148.977829666654
        struct[0].Gy[33,57] = 181.611068546017
        struct[0].Gy[33,58] = 42.5650941904727
        struct[0].Gy[33,59] = -51.8888767274335
        struct[0].Gy[34,26] = -0.00672902411642920
        struct[0].Gy[34,27] = -0.0538321929314336
        struct[0].Gy[34,28] = 0.00672902411642920
        struct[0].Gy[34,29] = 0.0538321929314336
        struct[0].Gy[34,30] = 53.9540151298641
        struct[0].Gy[34,31] = 44.2677164725443
        struct[0].Gy[34,32] = 53.9540151298642
        struct[0].Gy[34,33] = 44.2677164725443
        struct[0].Gy[34,34] = -188.924390492986
        struct[0].Gy[34,35] = -155.244588874881
        struct[0].Gy[34,48] = -2.07555506909734
        struct[0].Gy[34,49] = -1.70260376761891
        struct[0].Gy[34,50] = -2.07555506909734
        struct[0].Gy[34,51] = -1.70260376761891
        struct[0].Gy[34,52] = 7.26444274184068
        struct[0].Gy[34,53] = 5.95911318666618
        struct[0].Gy[34,54] = -51.8888767274334
        struct[0].Gy[34,55] = -42.5650941904727
        struct[0].Gy[34,56] = -51.8888767274335
        struct[0].Gy[34,57] = -42.5650941904727
        struct[0].Gy[34,58] = 181.611068546017
        struct[0].Gy[34,59] = 148.977829666654
        struct[0].Gy[35,26] = 0.0538321929314336
        struct[0].Gy[35,27] = -0.00672902411642920
        struct[0].Gy[35,28] = -0.0538321929314336
        struct[0].Gy[35,29] = 0.00672902411642920
        struct[0].Gy[35,30] = -44.2677164725443
        struct[0].Gy[35,31] = 53.9540151298641
        struct[0].Gy[35,32] = -44.2677164725443
        struct[0].Gy[35,33] = 53.9540151298642
        struct[0].Gy[35,34] = 155.244588874881
        struct[0].Gy[35,35] = -188.924390492986
        struct[0].Gy[35,48] = 1.70260376761891
        struct[0].Gy[35,49] = -2.07555506909734
        struct[0].Gy[35,50] = 1.70260376761891
        struct[0].Gy[35,51] = -2.07555506909734
        struct[0].Gy[35,52] = -5.95911318666618
        struct[0].Gy[35,53] = 7.26444274184068
        struct[0].Gy[35,54] = 42.5650941904727
        struct[0].Gy[35,55] = -51.8888767274334
        struct[0].Gy[35,56] = 42.5650941904727
        struct[0].Gy[35,57] = -51.8888767274335
        struct[0].Gy[35,58] = -148.977829666654
        struct[0].Gy[35,59] = 181.611068546017
        struct[0].Gy[36,0] = 0.282682270461039
        struct[0].Gy[36,1] = 1.69609362276623
        struct[0].Gy[36,2] = -0.282682270461039
        struct[0].Gy[36,3] = -1.69609362276623
        struct[0].Gy[36,36] = -7.27570400310194
        struct[0].Gy[36,37] = -6.02663624833782
        struct[0].Gy[36,38] = 2.08118569972797
        struct[0].Gy[36,39] = 1.73640535376106
        struct[0].Gy[36,40] = 2.08118569972797
        struct[0].Gy[36,41] = 1.73640535376106
        struct[0].Gy[36,42] = 7.26444274184068
        struct[0].Gy[36,43] = 5.95911318666618
        struct[0].Gy[36,44] = -2.07555506909734
        struct[0].Gy[36,45] = -1.70260376761891
        struct[0].Gy[36,46] = -2.07555506909734
        struct[0].Gy[36,47] = -1.70260376761891
        struct[0].Gy[37,0] = -1.69609362276623
        struct[0].Gy[37,1] = 0.282682270461039
        struct[0].Gy[37,2] = 1.69609362276623
        struct[0].Gy[37,3] = -0.282682270461039
        struct[0].Gy[37,36] = 6.02663624833782
        struct[0].Gy[37,37] = -7.27570400310194
        struct[0].Gy[37,38] = -1.73640535376106
        struct[0].Gy[37,39] = 2.08118569972797
        struct[0].Gy[37,40] = -1.73640535376106
        struct[0].Gy[37,41] = 2.08118569972797
        struct[0].Gy[37,42] = -5.95911318666618
        struct[0].Gy[37,43] = 7.26444274184068
        struct[0].Gy[37,44] = 1.70260376761891
        struct[0].Gy[37,45] = -2.07555506909734
        struct[0].Gy[37,46] = 1.70260376761891
        struct[0].Gy[37,47] = -2.07555506909734
        struct[0].Gy[38,2] = 0.282682270461039
        struct[0].Gy[38,3] = 1.69609362276623
        struct[0].Gy[38,4] = -0.282682270461039
        struct[0].Gy[38,5] = -1.69609362276623
        struct[0].Gy[38,36] = 2.08118569972797
        struct[0].Gy[38,37] = 1.73640535376106
        struct[0].Gy[38,38] = -7.27570400310194
        struct[0].Gy[38,39] = -6.02663624833782
        struct[0].Gy[38,40] = 2.08118569972797
        struct[0].Gy[38,41] = 1.73640535376106
        struct[0].Gy[38,42] = -2.07555506909734
        struct[0].Gy[38,43] = -1.70260376761891
        struct[0].Gy[38,44] = 7.26444274184068
        struct[0].Gy[38,45] = 5.95911318666618
        struct[0].Gy[38,46] = -2.07555506909734
        struct[0].Gy[38,47] = -1.70260376761891
        struct[0].Gy[39,2] = -1.69609362276623
        struct[0].Gy[39,3] = 0.282682270461039
        struct[0].Gy[39,4] = 1.69609362276623
        struct[0].Gy[39,5] = -0.282682270461039
        struct[0].Gy[39,36] = -1.73640535376106
        struct[0].Gy[39,37] = 2.08118569972797
        struct[0].Gy[39,38] = 6.02663624833782
        struct[0].Gy[39,39] = -7.27570400310194
        struct[0].Gy[39,40] = -1.73640535376106
        struct[0].Gy[39,41] = 2.08118569972797
        struct[0].Gy[39,42] = 1.70260376761891
        struct[0].Gy[39,43] = -2.07555506909734
        struct[0].Gy[39,44] = -5.95911318666618
        struct[0].Gy[39,45] = 7.26444274184068
        struct[0].Gy[39,46] = 1.70260376761891
        struct[0].Gy[39,47] = -2.07555506909734
        struct[0].Gy[40,0] = -0.282682270461039
        struct[0].Gy[40,1] = -1.69609362276623
        struct[0].Gy[40,4] = 0.282682270461039
        struct[0].Gy[40,5] = 1.69609362276623
        struct[0].Gy[40,36] = 2.08118569972797
        struct[0].Gy[40,37] = 1.73640535376106
        struct[0].Gy[40,38] = 2.08118569972797
        struct[0].Gy[40,39] = 1.73640535376106
        struct[0].Gy[40,40] = -7.27570400310194
        struct[0].Gy[40,41] = -6.02663624833782
        struct[0].Gy[40,42] = -2.07555506909734
        struct[0].Gy[40,43] = -1.70260376761891
        struct[0].Gy[40,44] = -2.07555506909734
        struct[0].Gy[40,45] = -1.70260376761891
        struct[0].Gy[40,46] = 7.26444274184068
        struct[0].Gy[40,47] = 5.95911318666618
        struct[0].Gy[41,0] = 1.69609362276623
        struct[0].Gy[41,1] = -0.282682270461039
        struct[0].Gy[41,4] = -1.69609362276623
        struct[0].Gy[41,5] = 0.282682270461039
        struct[0].Gy[41,36] = -1.73640535376106
        struct[0].Gy[41,37] = 2.08118569972797
        struct[0].Gy[41,38] = -1.73640535376106
        struct[0].Gy[41,39] = 2.08118569972797
        struct[0].Gy[41,40] = 6.02663624833782
        struct[0].Gy[41,41] = -7.27570400310194
        struct[0].Gy[41,42] = 1.70260376761891
        struct[0].Gy[41,43] = -2.07555506909734
        struct[0].Gy[41,44] = 1.70260376761891
        struct[0].Gy[41,45] = -2.07555506909734
        struct[0].Gy[41,46] = -5.95911318666618
        struct[0].Gy[41,47] = 7.26444274184068
        struct[0].Gy[42,6] = 0.282682270461039
        struct[0].Gy[42,7] = 1.69609362276623
        struct[0].Gy[42,8] = -0.282682270461039
        struct[0].Gy[42,9] = -1.69609362276623
        struct[0].Gy[42,36] = 7.26444274184068
        struct[0].Gy[42,37] = 5.95911318666618
        struct[0].Gy[42,38] = -2.07555506909734
        struct[0].Gy[42,39] = -1.70260376761891
        struct[0].Gy[42,40] = -2.07555506909734
        struct[0].Gy[42,41] = -1.70260376761891
        struct[0].Gy[42,42] = -14.5401467449426
        struct[0].Gy[42,43] = -11.9857049291081
        struct[0].Gy[42,44] = 4.15674076882530
        struct[0].Gy[42,45] = 3.43902692373834
        struct[0].Gy[42,46] = 4.15674076882530
        struct[0].Gy[42,47] = 3.43902692373834
        struct[0].Gy[42,48] = 7.26444274184068
        struct[0].Gy[42,49] = 5.95911318666618
        struct[0].Gy[42,50] = -2.07555506909734
        struct[0].Gy[42,51] = -1.70260376761891
        struct[0].Gy[42,52] = -2.07555506909734
        struct[0].Gy[42,53] = -1.70260376761891
        struct[0].Gy[43,6] = -1.69609362276623
        struct[0].Gy[43,7] = 0.282682270461039
        struct[0].Gy[43,8] = 1.69609362276623
        struct[0].Gy[43,9] = -0.282682270461039
        struct[0].Gy[43,36] = -5.95911318666618
        struct[0].Gy[43,37] = 7.26444274184068
        struct[0].Gy[43,38] = 1.70260376761891
        struct[0].Gy[43,39] = -2.07555506909734
        struct[0].Gy[43,40] = 1.70260376761891
        struct[0].Gy[43,41] = -2.07555506909734
        struct[0].Gy[43,42] = 11.9857049291081
        struct[0].Gy[43,43] = -14.5401467449426
        struct[0].Gy[43,44] = -3.43902692373834
        struct[0].Gy[43,45] = 4.15674076882530
        struct[0].Gy[43,46] = -3.43902692373834
        struct[0].Gy[43,47] = 4.15674076882530
        struct[0].Gy[43,48] = -5.95911318666618
        struct[0].Gy[43,49] = 7.26444274184068
        struct[0].Gy[43,50] = 1.70260376761891
        struct[0].Gy[43,51] = -2.07555506909734
        struct[0].Gy[43,52] = 1.70260376761891
        struct[0].Gy[43,53] = -2.07555506909734
        struct[0].Gy[44,8] = 0.282682270461039
        struct[0].Gy[44,9] = 1.69609362276623
        struct[0].Gy[44,10] = -0.282682270461039
        struct[0].Gy[44,11] = -1.69609362276623
        struct[0].Gy[44,36] = -2.07555506909734
        struct[0].Gy[44,37] = -1.70260376761891
        struct[0].Gy[44,38] = 7.26444274184068
        struct[0].Gy[44,39] = 5.95911318666618
        struct[0].Gy[44,40] = -2.07555506909734
        struct[0].Gy[44,41] = -1.70260376761891
        struct[0].Gy[44,42] = 4.15674076882530
        struct[0].Gy[44,43] = 3.43902692373834
        struct[0].Gy[44,44] = -14.5401467449426
        struct[0].Gy[44,45] = -11.9857049291081
        struct[0].Gy[44,46] = 4.15674076882531
        struct[0].Gy[44,47] = 3.43902692373834
        struct[0].Gy[44,48] = -2.07555506909734
        struct[0].Gy[44,49] = -1.70260376761891
        struct[0].Gy[44,50] = 7.26444274184068
        struct[0].Gy[44,51] = 5.95911318666618
        struct[0].Gy[44,52] = -2.07555506909734
        struct[0].Gy[44,53] = -1.70260376761891
        struct[0].Gy[45,8] = -1.69609362276623
        struct[0].Gy[45,9] = 0.282682270461039
        struct[0].Gy[45,10] = 1.69609362276623
        struct[0].Gy[45,11] = -0.282682270461039
        struct[0].Gy[45,36] = 1.70260376761891
        struct[0].Gy[45,37] = -2.07555506909734
        struct[0].Gy[45,38] = -5.95911318666618
        struct[0].Gy[45,39] = 7.26444274184068
        struct[0].Gy[45,40] = 1.70260376761891
        struct[0].Gy[45,41] = -2.07555506909734
        struct[0].Gy[45,42] = -3.43902692373834
        struct[0].Gy[45,43] = 4.15674076882530
        struct[0].Gy[45,44] = 11.9857049291081
        struct[0].Gy[45,45] = -14.5401467449426
        struct[0].Gy[45,46] = -3.43902692373834
        struct[0].Gy[45,47] = 4.15674076882531
        struct[0].Gy[45,48] = 1.70260376761891
        struct[0].Gy[45,49] = -2.07555506909734
        struct[0].Gy[45,50] = -5.95911318666618
        struct[0].Gy[45,51] = 7.26444274184068
        struct[0].Gy[45,52] = 1.70260376761891
        struct[0].Gy[45,53] = -2.07555506909734
        struct[0].Gy[46,6] = -0.282682270461039
        struct[0].Gy[46,7] = -1.69609362276623
        struct[0].Gy[46,10] = 0.282682270461039
        struct[0].Gy[46,11] = 1.69609362276623
        struct[0].Gy[46,36] = -2.07555506909734
        struct[0].Gy[46,37] = -1.70260376761891
        struct[0].Gy[46,38] = -2.07555506909734
        struct[0].Gy[46,39] = -1.70260376761891
        struct[0].Gy[46,40] = 7.26444274184068
        struct[0].Gy[46,41] = 5.95911318666618
        struct[0].Gy[46,42] = 4.15674076882530
        struct[0].Gy[46,43] = 3.43902692373834
        struct[0].Gy[46,44] = 4.15674076882531
        struct[0].Gy[46,45] = 3.43902692373834
        struct[0].Gy[46,46] = -14.5401467449426
        struct[0].Gy[46,47] = -11.9857049291081
        struct[0].Gy[46,48] = -2.07555506909734
        struct[0].Gy[46,49] = -1.70260376761891
        struct[0].Gy[46,50] = -2.07555506909734
        struct[0].Gy[46,51] = -1.70260376761891
        struct[0].Gy[46,52] = 7.26444274184068
        struct[0].Gy[46,53] = 5.95911318666618
        struct[0].Gy[47,6] = 1.69609362276623
        struct[0].Gy[47,7] = -0.282682270461039
        struct[0].Gy[47,10] = -1.69609362276623
        struct[0].Gy[47,11] = 0.282682270461039
        struct[0].Gy[47,36] = 1.70260376761891
        struct[0].Gy[47,37] = -2.07555506909734
        struct[0].Gy[47,38] = 1.70260376761891
        struct[0].Gy[47,39] = -2.07555506909734
        struct[0].Gy[47,40] = -5.95911318666618
        struct[0].Gy[47,41] = 7.26444274184068
        struct[0].Gy[47,42] = -3.43902692373834
        struct[0].Gy[47,43] = 4.15674076882530
        struct[0].Gy[47,44] = -3.43902692373834
        struct[0].Gy[47,45] = 4.15674076882531
        struct[0].Gy[47,46] = 11.9857049291081
        struct[0].Gy[47,47] = -14.5401467449426
        struct[0].Gy[47,48] = 1.70260376761891
        struct[0].Gy[47,49] = -2.07555506909734
        struct[0].Gy[47,50] = 1.70260376761891
        struct[0].Gy[47,51] = -2.07555506909734
        struct[0].Gy[47,52] = -5.95911318666618
        struct[0].Gy[47,53] = 7.26444274184068
        struct[0].Gy[48,12] = 0.282682270461039
        struct[0].Gy[48,13] = 1.69609362276623
        struct[0].Gy[48,14] = -0.282682270461039
        struct[0].Gy[48,15] = -1.69609362276623
        struct[0].Gy[48,30] = 7.26444274184068
        struct[0].Gy[48,31] = 5.95911318666618
        struct[0].Gy[48,32] = -2.07555506909734
        struct[0].Gy[48,33] = -1.70260376761891
        struct[0].Gy[48,34] = -2.07555506909734
        struct[0].Gy[48,35] = -1.70260376761891
        struct[0].Gy[48,42] = 7.26444274184068
        struct[0].Gy[48,43] = 5.95911318666618
        struct[0].Gy[48,44] = -2.07555506909734
        struct[0].Gy[48,45] = -1.70260376761891
        struct[0].Gy[48,46] = -2.07555506909734
        struct[0].Gy[48,47] = -1.70260376761891
        struct[0].Gy[48,48] = -14.5401467449426
        struct[0].Gy[48,49] = -11.9857049291081
        struct[0].Gy[48,50] = 4.15674076882530
        struct[0].Gy[48,51] = 3.43902692373834
        struct[0].Gy[48,52] = 4.15674076882530
        struct[0].Gy[48,53] = 3.43902692373834
        struct[0].Gy[49,12] = -1.69609362276623
        struct[0].Gy[49,13] = 0.282682270461039
        struct[0].Gy[49,14] = 1.69609362276623
        struct[0].Gy[49,15] = -0.282682270461039
        struct[0].Gy[49,30] = -5.95911318666618
        struct[0].Gy[49,31] = 7.26444274184068
        struct[0].Gy[49,32] = 1.70260376761891
        struct[0].Gy[49,33] = -2.07555506909734
        struct[0].Gy[49,34] = 1.70260376761891
        struct[0].Gy[49,35] = -2.07555506909734
        struct[0].Gy[49,42] = -5.95911318666618
        struct[0].Gy[49,43] = 7.26444274184068
        struct[0].Gy[49,44] = 1.70260376761891
        struct[0].Gy[49,45] = -2.07555506909734
        struct[0].Gy[49,46] = 1.70260376761891
        struct[0].Gy[49,47] = -2.07555506909734
        struct[0].Gy[49,48] = 11.9857049291081
        struct[0].Gy[49,49] = -14.5401467449426
        struct[0].Gy[49,50] = -3.43902692373834
        struct[0].Gy[49,51] = 4.15674076882530
        struct[0].Gy[49,52] = -3.43902692373834
        struct[0].Gy[49,53] = 4.15674076882530
        struct[0].Gy[50,14] = 0.282682270461039
        struct[0].Gy[50,15] = 1.69609362276623
        struct[0].Gy[50,16] = -0.282682270461039
        struct[0].Gy[50,17] = -1.69609362276623
        struct[0].Gy[50,30] = -2.07555506909734
        struct[0].Gy[50,31] = -1.70260376761891
        struct[0].Gy[50,32] = 7.26444274184068
        struct[0].Gy[50,33] = 5.95911318666618
        struct[0].Gy[50,34] = -2.07555506909734
        struct[0].Gy[50,35] = -1.70260376761891
        struct[0].Gy[50,42] = -2.07555506909734
        struct[0].Gy[50,43] = -1.70260376761891
        struct[0].Gy[50,44] = 7.26444274184068
        struct[0].Gy[50,45] = 5.95911318666618
        struct[0].Gy[50,46] = -2.07555506909734
        struct[0].Gy[50,47] = -1.70260376761891
        struct[0].Gy[50,48] = 4.15674076882530
        struct[0].Gy[50,49] = 3.43902692373834
        struct[0].Gy[50,50] = -14.5401467449426
        struct[0].Gy[50,51] = -11.9857049291081
        struct[0].Gy[50,52] = 4.15674076882531
        struct[0].Gy[50,53] = 3.43902692373834
        struct[0].Gy[51,14] = -1.69609362276623
        struct[0].Gy[51,15] = 0.282682270461039
        struct[0].Gy[51,16] = 1.69609362276623
        struct[0].Gy[51,17] = -0.282682270461039
        struct[0].Gy[51,30] = 1.70260376761891
        struct[0].Gy[51,31] = -2.07555506909734
        struct[0].Gy[51,32] = -5.95911318666618
        struct[0].Gy[51,33] = 7.26444274184068
        struct[0].Gy[51,34] = 1.70260376761891
        struct[0].Gy[51,35] = -2.07555506909734
        struct[0].Gy[51,42] = 1.70260376761891
        struct[0].Gy[51,43] = -2.07555506909734
        struct[0].Gy[51,44] = -5.95911318666618
        struct[0].Gy[51,45] = 7.26444274184068
        struct[0].Gy[51,46] = 1.70260376761891
        struct[0].Gy[51,47] = -2.07555506909734
        struct[0].Gy[51,48] = -3.43902692373834
        struct[0].Gy[51,49] = 4.15674076882530
        struct[0].Gy[51,50] = 11.9857049291081
        struct[0].Gy[51,51] = -14.5401467449426
        struct[0].Gy[51,52] = -3.43902692373834
        struct[0].Gy[51,53] = 4.15674076882531
        struct[0].Gy[52,12] = -0.282682270461039
        struct[0].Gy[52,13] = -1.69609362276623
        struct[0].Gy[52,16] = 0.282682270461039
        struct[0].Gy[52,17] = 1.69609362276623
        struct[0].Gy[52,30] = -2.07555506909734
        struct[0].Gy[52,31] = -1.70260376761891
        struct[0].Gy[52,32] = -2.07555506909734
        struct[0].Gy[52,33] = -1.70260376761891
        struct[0].Gy[52,34] = 7.26444274184068
        struct[0].Gy[52,35] = 5.95911318666618
        struct[0].Gy[52,42] = -2.07555506909734
        struct[0].Gy[52,43] = -1.70260376761891
        struct[0].Gy[52,44] = -2.07555506909734
        struct[0].Gy[52,45] = -1.70260376761891
        struct[0].Gy[52,46] = 7.26444274184068
        struct[0].Gy[52,47] = 5.95911318666618
        struct[0].Gy[52,48] = 4.15674076882530
        struct[0].Gy[52,49] = 3.43902692373834
        struct[0].Gy[52,50] = 4.15674076882531
        struct[0].Gy[52,51] = 3.43902692373834
        struct[0].Gy[52,52] = -14.5401467449426
        struct[0].Gy[52,53] = -11.9857049291081
        struct[0].Gy[53,12] = 1.69609362276623
        struct[0].Gy[53,13] = -0.282682270461039
        struct[0].Gy[53,16] = -1.69609362276623
        struct[0].Gy[53,17] = 0.282682270461039
        struct[0].Gy[53,30] = 1.70260376761891
        struct[0].Gy[53,31] = -2.07555506909734
        struct[0].Gy[53,32] = 1.70260376761891
        struct[0].Gy[53,33] = -2.07555506909734
        struct[0].Gy[53,34] = -5.95911318666618
        struct[0].Gy[53,35] = 7.26444274184068
        struct[0].Gy[53,42] = 1.70260376761891
        struct[0].Gy[53,43] = -2.07555506909734
        struct[0].Gy[53,44] = 1.70260376761891
        struct[0].Gy[53,45] = -2.07555506909734
        struct[0].Gy[53,46] = -5.95911318666618
        struct[0].Gy[53,47] = 7.26444274184068
        struct[0].Gy[53,48] = -3.43902692373834
        struct[0].Gy[53,49] = 4.15674076882530
        struct[0].Gy[53,50] = -3.43902692373834
        struct[0].Gy[53,51] = 4.15674076882531
        struct[0].Gy[53,52] = 11.9857049291081
        struct[0].Gy[53,53] = -14.5401467449426
        struct[0].Gy[54,18] = 0.282682270461039
        struct[0].Gy[54,19] = 1.69609362276623
        struct[0].Gy[54,20] = -0.282682270461039
        struct[0].Gy[54,21] = -1.69609362276623
        struct[0].Gy[54,30] = 181.611068546017
        struct[0].Gy[54,31] = 148.977829666654
        struct[0].Gy[54,32] = -51.8888767274334
        struct[0].Gy[54,33] = -42.5650941904727
        struct[0].Gy[54,34] = -51.8888767274334
        struct[0].Gy[54,35] = -42.5650941904727
        struct[0].Gy[54,54] = -181.622329807278
        struct[0].Gy[54,55] = -149.045395453986
        struct[0].Gy[54,56] = 51.8945073580641
        struct[0].Gy[54,57] = 42.5988786863508
        struct[0].Gy[54,58] = 51.8945073580640
        struct[0].Gy[54,59] = 42.5988786863508
        struct[0].Gy[55,18] = -1.69609362276623
        struct[0].Gy[55,19] = 0.282682270461039
        struct[0].Gy[55,20] = 1.69609362276623
        struct[0].Gy[55,21] = -0.282682270461039
        struct[0].Gy[55,30] = -148.977829666654
        struct[0].Gy[55,31] = 181.611068546017
        struct[0].Gy[55,32] = 42.5650941904727
        struct[0].Gy[55,33] = -51.8888767274334
        struct[0].Gy[55,34] = 42.5650941904727
        struct[0].Gy[55,35] = -51.8888767274334
        struct[0].Gy[55,54] = 149.045395453986
        struct[0].Gy[55,55] = -181.622329807278
        struct[0].Gy[55,56] = -42.5988786863508
        struct[0].Gy[55,57] = 51.8945073580641
        struct[0].Gy[55,58] = -42.5988786863508
        struct[0].Gy[55,59] = 51.8945073580640
        struct[0].Gy[56,20] = 0.282682270461039
        struct[0].Gy[56,21] = 1.69609362276623
        struct[0].Gy[56,22] = -0.282682270461039
        struct[0].Gy[56,23] = -1.69609362276623
        struct[0].Gy[56,30] = -51.8888767274334
        struct[0].Gy[56,31] = -42.5650941904727
        struct[0].Gy[56,32] = 181.611068546017
        struct[0].Gy[56,33] = 148.977829666654
        struct[0].Gy[56,34] = -51.8888767274335
        struct[0].Gy[56,35] = -42.5650941904727
        struct[0].Gy[56,54] = 51.8945073580641
        struct[0].Gy[56,55] = 42.5988786863508
        struct[0].Gy[56,56] = -181.622329807278
        struct[0].Gy[56,57] = -149.045395453986
        struct[0].Gy[56,58] = 51.8945073580641
        struct[0].Gy[56,59] = 42.5988786863508
        struct[0].Gy[57,20] = -1.69609362276623
        struct[0].Gy[57,21] = 0.282682270461039
        struct[0].Gy[57,22] = 1.69609362276623
        struct[0].Gy[57,23] = -0.282682270461039
        struct[0].Gy[57,30] = 42.5650941904727
        struct[0].Gy[57,31] = -51.8888767274334
        struct[0].Gy[57,32] = -148.977829666654
        struct[0].Gy[57,33] = 181.611068546017
        struct[0].Gy[57,34] = 42.5650941904727
        struct[0].Gy[57,35] = -51.8888767274335
        struct[0].Gy[57,54] = -42.5988786863508
        struct[0].Gy[57,55] = 51.8945073580641
        struct[0].Gy[57,56] = 149.045395453986
        struct[0].Gy[57,57] = -181.622329807278
        struct[0].Gy[57,58] = -42.5988786863508
        struct[0].Gy[57,59] = 51.8945073580641
        struct[0].Gy[58,18] = -0.282682270461039
        struct[0].Gy[58,19] = -1.69609362276623
        struct[0].Gy[58,22] = 0.282682270461039
        struct[0].Gy[58,23] = 1.69609362276623
        struct[0].Gy[58,30] = -51.8888767274334
        struct[0].Gy[58,31] = -42.5650941904727
        struct[0].Gy[58,32] = -51.8888767274335
        struct[0].Gy[58,33] = -42.5650941904727
        struct[0].Gy[58,34] = 181.611068546017
        struct[0].Gy[58,35] = 148.977829666654
        struct[0].Gy[58,54] = 51.8945073580641
        struct[0].Gy[58,55] = 42.5988786863508
        struct[0].Gy[58,56] = 51.8945073580641
        struct[0].Gy[58,57] = 42.5988786863508
        struct[0].Gy[58,58] = -181.622329807278
        struct[0].Gy[58,59] = -149.045395453986
        struct[0].Gy[59,18] = 1.69609362276623
        struct[0].Gy[59,19] = -0.282682270461039
        struct[0].Gy[59,22] = -1.69609362276623
        struct[0].Gy[59,23] = 0.282682270461039
        struct[0].Gy[59,30] = 42.5650941904727
        struct[0].Gy[59,31] = -51.8888767274334
        struct[0].Gy[59,32] = 42.5650941904727
        struct[0].Gy[59,33] = -51.8888767274335
        struct[0].Gy[59,34] = -148.977829666654
        struct[0].Gy[59,35] = 181.611068546017
        struct[0].Gy[59,54] = -42.5988786863508
        struct[0].Gy[59,55] = 51.8945073580641
        struct[0].Gy[59,56] = -42.5988786863508
        struct[0].Gy[59,57] = 51.8945073580641
        struct[0].Gy[59,58] = 149.045395453986
        struct[0].Gy[59,59] = -181.622329807278
        struct[0].Gy[60,36] = 7.26444274184068
        struct[0].Gy[60,37] = 5.95911318666618
        struct[0].Gy[60,38] = -2.07555506909734
        struct[0].Gy[60,39] = -1.70260376761891
        struct[0].Gy[60,40] = -2.07555506909734
        struct[0].Gy[60,41] = -1.70260376761891
        struct[0].Gy[60,42] = -7.26444274184068
        struct[0].Gy[60,43] = -5.95911318666618
        struct[0].Gy[60,44] = 2.07555506909734
        struct[0].Gy[60,45] = 1.70260376761891
        struct[0].Gy[60,46] = 2.07555506909734
        struct[0].Gy[60,47] = 1.70260376761891
        struct[0].Gy[60,60] = -1
        struct[0].Gy[61,36] = -5.95911318666618
        struct[0].Gy[61,37] = 7.26444274184068
        struct[0].Gy[61,38] = 1.70260376761891
        struct[0].Gy[61,39] = -2.07555506909734
        struct[0].Gy[61,40] = 1.70260376761891
        struct[0].Gy[61,41] = -2.07555506909734
        struct[0].Gy[61,42] = 5.95911318666618
        struct[0].Gy[61,43] = -7.26444274184068
        struct[0].Gy[61,44] = -1.70260376761891
        struct[0].Gy[61,45] = 2.07555506909734
        struct[0].Gy[61,46] = -1.70260376761891
        struct[0].Gy[61,47] = 2.07555506909734
        struct[0].Gy[61,61] = -1
        struct[0].Gy[62,36] = -2.07555506909734
        struct[0].Gy[62,37] = -1.70260376761891
        struct[0].Gy[62,38] = 7.26444274184068
        struct[0].Gy[62,39] = 5.95911318666618
        struct[0].Gy[62,40] = -2.07555506909734
        struct[0].Gy[62,41] = -1.70260376761891
        struct[0].Gy[62,42] = 2.07555506909734
        struct[0].Gy[62,43] = 1.70260376761891
        struct[0].Gy[62,44] = -7.26444274184068
        struct[0].Gy[62,45] = -5.95911318666618
        struct[0].Gy[62,46] = 2.07555506909734
        struct[0].Gy[62,47] = 1.70260376761891
        struct[0].Gy[62,62] = -1
        struct[0].Gy[63,36] = 1.70260376761891
        struct[0].Gy[63,37] = -2.07555506909734
        struct[0].Gy[63,38] = -5.95911318666618
        struct[0].Gy[63,39] = 7.26444274184068
        struct[0].Gy[63,40] = 1.70260376761891
        struct[0].Gy[63,41] = -2.07555506909734
        struct[0].Gy[63,42] = -1.70260376761891
        struct[0].Gy[63,43] = 2.07555506909734
        struct[0].Gy[63,44] = 5.95911318666618
        struct[0].Gy[63,45] = -7.26444274184068
        struct[0].Gy[63,46] = -1.70260376761891
        struct[0].Gy[63,47] = 2.07555506909734
        struct[0].Gy[63,63] = -1
        struct[0].Gy[64,36] = -2.07555506909734
        struct[0].Gy[64,37] = -1.70260376761891
        struct[0].Gy[64,38] = -2.07555506909734
        struct[0].Gy[64,39] = -1.70260376761891
        struct[0].Gy[64,40] = 7.26444274184068
        struct[0].Gy[64,41] = 5.95911318666618
        struct[0].Gy[64,42] = 2.07555506909734
        struct[0].Gy[64,43] = 1.70260376761891
        struct[0].Gy[64,44] = 2.07555506909734
        struct[0].Gy[64,45] = 1.70260376761891
        struct[0].Gy[64,46] = -7.26444274184068
        struct[0].Gy[64,47] = -5.95911318666618
        struct[0].Gy[64,64] = -1
        struct[0].Gy[65,36] = 1.70260376761891
        struct[0].Gy[65,37] = -2.07555506909734
        struct[0].Gy[65,38] = 1.70260376761891
        struct[0].Gy[65,39] = -2.07555506909734
        struct[0].Gy[65,40] = -5.95911318666618
        struct[0].Gy[65,41] = 7.26444274184068
        struct[0].Gy[65,42] = -1.70260376761891
        struct[0].Gy[65,43] = 2.07555506909734
        struct[0].Gy[65,44] = -1.70260376761891
        struct[0].Gy[65,45] = 2.07555506909734
        struct[0].Gy[65,46] = 5.95911318666618
        struct[0].Gy[65,47] = -7.26444274184068
        struct[0].Gy[65,65] = -1
        struct[0].Gy[66,42] = 7.26444274184068
        struct[0].Gy[66,43] = 5.95911318666618
        struct[0].Gy[66,44] = -2.07555506909734
        struct[0].Gy[66,45] = -1.70260376761891
        struct[0].Gy[66,46] = -2.07555506909734
        struct[0].Gy[66,47] = -1.70260376761891
        struct[0].Gy[66,48] = -7.26444274184068
        struct[0].Gy[66,49] = -5.95911318666618
        struct[0].Gy[66,50] = 2.07555506909734
        struct[0].Gy[66,51] = 1.70260376761891
        struct[0].Gy[66,52] = 2.07555506909734
        struct[0].Gy[66,53] = 1.70260376761891
        struct[0].Gy[66,66] = -1
        struct[0].Gy[67,42] = -5.95911318666618
        struct[0].Gy[67,43] = 7.26444274184068
        struct[0].Gy[67,44] = 1.70260376761891
        struct[0].Gy[67,45] = -2.07555506909734
        struct[0].Gy[67,46] = 1.70260376761891
        struct[0].Gy[67,47] = -2.07555506909734
        struct[0].Gy[67,48] = 5.95911318666618
        struct[0].Gy[67,49] = -7.26444274184068
        struct[0].Gy[67,50] = -1.70260376761891
        struct[0].Gy[67,51] = 2.07555506909734
        struct[0].Gy[67,52] = -1.70260376761891
        struct[0].Gy[67,53] = 2.07555506909734
        struct[0].Gy[67,67] = -1
        struct[0].Gy[68,42] = -2.07555506909734
        struct[0].Gy[68,43] = -1.70260376761891
        struct[0].Gy[68,44] = 7.26444274184068
        struct[0].Gy[68,45] = 5.95911318666618
        struct[0].Gy[68,46] = -2.07555506909734
        struct[0].Gy[68,47] = -1.70260376761891
        struct[0].Gy[68,48] = 2.07555506909734
        struct[0].Gy[68,49] = 1.70260376761891
        struct[0].Gy[68,50] = -7.26444274184068
        struct[0].Gy[68,51] = -5.95911318666618
        struct[0].Gy[68,52] = 2.07555506909734
        struct[0].Gy[68,53] = 1.70260376761891
        struct[0].Gy[68,68] = -1
        struct[0].Gy[69,42] = 1.70260376761891
        struct[0].Gy[69,43] = -2.07555506909734
        struct[0].Gy[69,44] = -5.95911318666618
        struct[0].Gy[69,45] = 7.26444274184068
        struct[0].Gy[69,46] = 1.70260376761891
        struct[0].Gy[69,47] = -2.07555506909734
        struct[0].Gy[69,48] = -1.70260376761891
        struct[0].Gy[69,49] = 2.07555506909734
        struct[0].Gy[69,50] = 5.95911318666618
        struct[0].Gy[69,51] = -7.26444274184068
        struct[0].Gy[69,52] = -1.70260376761891
        struct[0].Gy[69,53] = 2.07555506909734
        struct[0].Gy[69,69] = -1
        struct[0].Gy[70,42] = -2.07555506909734
        struct[0].Gy[70,43] = -1.70260376761891
        struct[0].Gy[70,44] = -2.07555506909734
        struct[0].Gy[70,45] = -1.70260376761891
        struct[0].Gy[70,46] = 7.26444274184068
        struct[0].Gy[70,47] = 5.95911318666618
        struct[0].Gy[70,48] = 2.07555506909734
        struct[0].Gy[70,49] = 1.70260376761891
        struct[0].Gy[70,50] = 2.07555506909734
        struct[0].Gy[70,51] = 1.70260376761891
        struct[0].Gy[70,52] = -7.26444274184068
        struct[0].Gy[70,53] = -5.95911318666618
        struct[0].Gy[70,70] = -1
        struct[0].Gy[71,42] = 1.70260376761891
        struct[0].Gy[71,43] = -2.07555506909734
        struct[0].Gy[71,44] = 1.70260376761891
        struct[0].Gy[71,45] = -2.07555506909734
        struct[0].Gy[71,46] = -5.95911318666618
        struct[0].Gy[71,47] = 7.26444274184068
        struct[0].Gy[71,48] = -1.70260376761891
        struct[0].Gy[71,49] = 2.07555506909734
        struct[0].Gy[71,50] = -1.70260376761891
        struct[0].Gy[71,51] = 2.07555506909734
        struct[0].Gy[71,52] = 5.95911318666618
        struct[0].Gy[71,53] = -7.26444274184068
        struct[0].Gy[71,71] = -1
        struct[0].Gy[72,30] = -7.26444274184068
        struct[0].Gy[72,31] = -5.95911318666618
        struct[0].Gy[72,32] = 2.07555506909734
        struct[0].Gy[72,33] = 1.70260376761891
        struct[0].Gy[72,34] = 2.07555506909734
        struct[0].Gy[72,35] = 1.70260376761891
        struct[0].Gy[72,48] = 7.26444274184068
        struct[0].Gy[72,49] = 5.95911318666618
        struct[0].Gy[72,50] = -2.07555506909734
        struct[0].Gy[72,51] = -1.70260376761891
        struct[0].Gy[72,52] = -2.07555506909734
        struct[0].Gy[72,53] = -1.70260376761891
        struct[0].Gy[72,72] = -1
        struct[0].Gy[73,30] = 5.95911318666618
        struct[0].Gy[73,31] = -7.26444274184068
        struct[0].Gy[73,32] = -1.70260376761891
        struct[0].Gy[73,33] = 2.07555506909734
        struct[0].Gy[73,34] = -1.70260376761891
        struct[0].Gy[73,35] = 2.07555506909734
        struct[0].Gy[73,48] = -5.95911318666618
        struct[0].Gy[73,49] = 7.26444274184068
        struct[0].Gy[73,50] = 1.70260376761891
        struct[0].Gy[73,51] = -2.07555506909734
        struct[0].Gy[73,52] = 1.70260376761891
        struct[0].Gy[73,53] = -2.07555506909734
        struct[0].Gy[73,73] = -1
        struct[0].Gy[74,30] = 2.07555506909734
        struct[0].Gy[74,31] = 1.70260376761891
        struct[0].Gy[74,32] = -7.26444274184068
        struct[0].Gy[74,33] = -5.95911318666618
        struct[0].Gy[74,34] = 2.07555506909734
        struct[0].Gy[74,35] = 1.70260376761891
        struct[0].Gy[74,48] = -2.07555506909734
        struct[0].Gy[74,49] = -1.70260376761891
        struct[0].Gy[74,50] = 7.26444274184068
        struct[0].Gy[74,51] = 5.95911318666618
        struct[0].Gy[74,52] = -2.07555506909734
        struct[0].Gy[74,53] = -1.70260376761891
        struct[0].Gy[74,74] = -1
        struct[0].Gy[75,30] = -1.70260376761891
        struct[0].Gy[75,31] = 2.07555506909734
        struct[0].Gy[75,32] = 5.95911318666618
        struct[0].Gy[75,33] = -7.26444274184068
        struct[0].Gy[75,34] = -1.70260376761891
        struct[0].Gy[75,35] = 2.07555506909734
        struct[0].Gy[75,48] = 1.70260376761891
        struct[0].Gy[75,49] = -2.07555506909734
        struct[0].Gy[75,50] = -5.95911318666618
        struct[0].Gy[75,51] = 7.26444274184068
        struct[0].Gy[75,52] = 1.70260376761891
        struct[0].Gy[75,53] = -2.07555506909734
        struct[0].Gy[75,75] = -1
        struct[0].Gy[76,30] = 2.07555506909734
        struct[0].Gy[76,31] = 1.70260376761891
        struct[0].Gy[76,32] = 2.07555506909734
        struct[0].Gy[76,33] = 1.70260376761891
        struct[0].Gy[76,34] = -7.26444274184068
        struct[0].Gy[76,35] = -5.95911318666618
        struct[0].Gy[76,48] = -2.07555506909734
        struct[0].Gy[76,49] = -1.70260376761891
        struct[0].Gy[76,50] = -2.07555506909734
        struct[0].Gy[76,51] = -1.70260376761891
        struct[0].Gy[76,52] = 7.26444274184068
        struct[0].Gy[76,53] = 5.95911318666618
        struct[0].Gy[76,76] = -1
        struct[0].Gy[77,30] = -1.70260376761891
        struct[0].Gy[77,31] = 2.07555506909734
        struct[0].Gy[77,32] = -1.70260376761891
        struct[0].Gy[77,33] = 2.07555506909734
        struct[0].Gy[77,34] = 5.95911318666618
        struct[0].Gy[77,35] = -7.26444274184068
        struct[0].Gy[77,48] = 1.70260376761891
        struct[0].Gy[77,49] = -2.07555506909734
        struct[0].Gy[77,50] = 1.70260376761891
        struct[0].Gy[77,51] = -2.07555506909734
        struct[0].Gy[77,52] = -5.95911318666618
        struct[0].Gy[77,53] = 7.26444274184068
        struct[0].Gy[77,77] = -1
        struct[0].Gy[78,30] = -181.611068546017
        struct[0].Gy[78,31] = -148.977829666654
        struct[0].Gy[78,32] = 51.8888767274334
        struct[0].Gy[78,33] = 42.5650941904727
        struct[0].Gy[78,34] = 51.8888767274334
        struct[0].Gy[78,35] = 42.5650941904727
        struct[0].Gy[78,54] = 181.611068546017
        struct[0].Gy[78,55] = 148.977829666654
        struct[0].Gy[78,56] = -51.8888767274334
        struct[0].Gy[78,57] = -42.5650941904727
        struct[0].Gy[78,58] = -51.8888767274334
        struct[0].Gy[78,59] = -42.5650941904727
        struct[0].Gy[78,78] = -1
        struct[0].Gy[79,30] = 148.977829666654
        struct[0].Gy[79,31] = -181.611068546017
        struct[0].Gy[79,32] = -42.5650941904727
        struct[0].Gy[79,33] = 51.8888767274334
        struct[0].Gy[79,34] = -42.5650941904727
        struct[0].Gy[79,35] = 51.8888767274334
        struct[0].Gy[79,54] = -148.977829666654
        struct[0].Gy[79,55] = 181.611068546017
        struct[0].Gy[79,56] = 42.5650941904727
        struct[0].Gy[79,57] = -51.8888767274334
        struct[0].Gy[79,58] = 42.5650941904727
        struct[0].Gy[79,59] = -51.8888767274334
        struct[0].Gy[79,79] = -1
        struct[0].Gy[80,30] = 51.8888767274334
        struct[0].Gy[80,31] = 42.5650941904727
        struct[0].Gy[80,32] = -181.611068546017
        struct[0].Gy[80,33] = -148.977829666654
        struct[0].Gy[80,34] = 51.8888767274335
        struct[0].Gy[80,35] = 42.5650941904727
        struct[0].Gy[80,54] = -51.8888767274334
        struct[0].Gy[80,55] = -42.5650941904727
        struct[0].Gy[80,56] = 181.611068546017
        struct[0].Gy[80,57] = 148.977829666654
        struct[0].Gy[80,58] = -51.8888767274335
        struct[0].Gy[80,59] = -42.5650941904727
        struct[0].Gy[80,80] = -1
        struct[0].Gy[81,30] = -42.5650941904727
        struct[0].Gy[81,31] = 51.8888767274334
        struct[0].Gy[81,32] = 148.977829666654
        struct[0].Gy[81,33] = -181.611068546017
        struct[0].Gy[81,34] = -42.5650941904727
        struct[0].Gy[81,35] = 51.8888767274335
        struct[0].Gy[81,54] = 42.5650941904727
        struct[0].Gy[81,55] = -51.8888767274334
        struct[0].Gy[81,56] = -148.977829666654
        struct[0].Gy[81,57] = 181.611068546017
        struct[0].Gy[81,58] = 42.5650941904727
        struct[0].Gy[81,59] = -51.8888767274335
        struct[0].Gy[81,81] = -1
        struct[0].Gy[82,30] = 51.8888767274334
        struct[0].Gy[82,31] = 42.5650941904727
        struct[0].Gy[82,32] = 51.8888767274335
        struct[0].Gy[82,33] = 42.5650941904727
        struct[0].Gy[82,34] = -181.611068546017
        struct[0].Gy[82,35] = -148.977829666654
        struct[0].Gy[82,54] = -51.8888767274334
        struct[0].Gy[82,55] = -42.5650941904727
        struct[0].Gy[82,56] = -51.8888767274335
        struct[0].Gy[82,57] = -42.5650941904727
        struct[0].Gy[82,58] = 181.611068546017
        struct[0].Gy[82,59] = 148.977829666654
        struct[0].Gy[82,82] = -1
        struct[0].Gy[83,30] = -42.5650941904727
        struct[0].Gy[83,31] = 51.8888767274334
        struct[0].Gy[83,32] = -42.5650941904727
        struct[0].Gy[83,33] = 51.8888767274335
        struct[0].Gy[83,34] = 148.977829666654
        struct[0].Gy[83,35] = -181.611068546017
        struct[0].Gy[83,54] = 42.5650941904727
        struct[0].Gy[83,55] = -51.8888767274334
        struct[0].Gy[83,56] = 42.5650941904727
        struct[0].Gy[83,57] = -51.8888767274335
        struct[0].Gy[83,58] = -148.977829666654
        struct[0].Gy[83,59] = 181.611068546017
        struct[0].Gy[83,83] = -1
        struct[0].Gy[84,24] = 0.0241740531829170
        struct[0].Gy[84,25] = 0.0402900886381950
        struct[0].Gy[84,26] = -4.31760362252812E-18
        struct[0].Gy[84,27] = 4.66248501556824E-18
        struct[0].Gy[84,28] = -3.49608108880335E-18
        struct[0].Gy[84,29] = 4.19816664496737E-18
        struct[0].Gy[84,84] = -1
        struct[0].Gy[85,24] = -0.0402900886381950
        struct[0].Gy[85,25] = 0.0241740531829170
        struct[0].Gy[85,26] = -4.66248501556824E-18
        struct[0].Gy[85,27] = -4.31760362252812E-18
        struct[0].Gy[85,28] = -4.19816664496737E-18
        struct[0].Gy[85,29] = -3.49608108880335E-18
        struct[0].Gy[85,85] = -1
        struct[0].Gy[86,24] = -2.07254761002657E-18
        struct[0].Gy[86,25] = 6.30775359573304E-19
        struct[0].Gy[86,26] = 0.0241740531829170
        struct[0].Gy[86,27] = 0.0402900886381950
        struct[0].Gy[86,28] = -1.78419315993592E-17
        struct[0].Gy[86,29] = 9.01107656533306E-19
        struct[0].Gy[86,86] = -1
        struct[0].Gy[87,24] = -6.30775359573304E-19
        struct[0].Gy[87,25] = -2.07254761002657E-18
        struct[0].Gy[87,26] = -0.0402900886381950
        struct[0].Gy[87,27] = 0.0241740531829170
        struct[0].Gy[87,28] = -9.01107656533306E-19
        struct[0].Gy[87,29] = -1.78419315993592E-17
        struct[0].Gy[87,87] = -1
        struct[0].Gy[88,24] = -1.35166148479994E-18
        struct[0].Gy[88,25] = -7.20886125226632E-19
        struct[0].Gy[88,26] = -1.71210454741325E-17
        struct[0].Gy[88,27] = -4.50553828266631E-19
        struct[0].Gy[88,28] = 0.0241740531829170
        struct[0].Gy[88,29] = 0.0402900886381950
        struct[0].Gy[88,88] = -1
        struct[0].Gy[89,24] = 7.20886125226632E-19
        struct[0].Gy[89,25] = -1.35166148479994E-18
        struct[0].Gy[89,26] = 4.50553828266631E-19
        struct[0].Gy[89,27] = -1.71210454741325E-17
        struct[0].Gy[89,28] = -0.0402900886381950
        struct[0].Gy[89,29] = 0.0241740531829170
        struct[0].Gy[89,89] = -1
        struct[0].Gy[90,0] = i_W1lv_a_r
        struct[0].Gy[90,1] = i_W1lv_a_i
        struct[0].Gy[90,90] = v_W1lv_a_r
        struct[0].Gy[90,91] = v_W1lv_a_i
        struct[0].Gy[91,2] = i_W1lv_b_r
        struct[0].Gy[91,3] = i_W1lv_b_i
        struct[0].Gy[91,92] = v_W1lv_b_r
        struct[0].Gy[91,93] = v_W1lv_b_i
        struct[0].Gy[92,4] = i_W1lv_c_r
        struct[0].Gy[92,5] = i_W1lv_c_i
        struct[0].Gy[92,94] = v_W1lv_c_r
        struct[0].Gy[92,95] = v_W1lv_c_i
        struct[0].Gy[93,0] = -i_W1lv_a_i
        struct[0].Gy[93,1] = i_W1lv_a_r
        struct[0].Gy[93,90] = v_W1lv_a_i
        struct[0].Gy[93,91] = -v_W1lv_a_r
        struct[0].Gy[94,2] = -i_W1lv_b_i
        struct[0].Gy[94,3] = i_W1lv_b_r
        struct[0].Gy[94,92] = v_W1lv_b_i
        struct[0].Gy[94,93] = -v_W1lv_b_r
        struct[0].Gy[95,4] = -i_W1lv_c_i
        struct[0].Gy[95,5] = i_W1lv_c_r
        struct[0].Gy[95,94] = v_W1lv_c_i
        struct[0].Gy[95,95] = -v_W1lv_c_r
        struct[0].Gy[96,0] = 1.0*v_W1lv_a_r*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)/V_base_W1lv
        struct[0].Gy[96,1] = 1.0*v_W1lv_a_i*(v_W1lv_a_i**2 + v_W1lv_a_r**2)**(-0.5)/V_base_W1lv
        struct[0].Gy[96,96] = -1
        struct[0].Gy[97,36] = 1.0*v_W1mv_a_r*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)/V_base_W1mv
        struct[0].Gy[97,37] = 1.0*v_W1mv_a_i*(v_W1mv_a_i**2 + v_W1mv_a_r**2)**(-0.5)/V_base_W1mv
        struct[0].Gy[97,97] = -1
        struct[0].Gy[98,96] = K_p_v_W1lv*(u_ctrl_v_W1lv - 1.0)
        struct[0].Gy[98,97] = -K_p_v_W1lv*u_ctrl_v_W1lv
        struct[0].Gy[98,98] = -1
        struct[0].Gy[99,58] = 1.0*S_base_W1lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)]))
        struct[0].Gy[99,59] = 1.0*S_base_W1lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W1lv, I_max_W1lv < -i_reac_ref_W1lv), (I_max_W1lv, I_max_W1lv < i_reac_ref_W1lv), (i_reac_ref_W1lv, True)]))
        struct[0].Gy[99,98] = S_base_W1lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W1lv < i_reac_ref_W1lv) | (I_max_W1lv < -i_reac_ref_W1lv)), (1, True)]))
        struct[0].Gy[99,99] = -1
        struct[0].Gy[100,6] = i_W2lv_a_r
        struct[0].Gy[100,7] = i_W2lv_a_i
        struct[0].Gy[100,100] = v_W2lv_a_r
        struct[0].Gy[100,101] = v_W2lv_a_i
        struct[0].Gy[101,8] = i_W2lv_b_r
        struct[0].Gy[101,9] = i_W2lv_b_i
        struct[0].Gy[101,102] = v_W2lv_b_r
        struct[0].Gy[101,103] = v_W2lv_b_i
        struct[0].Gy[102,10] = i_W2lv_c_r
        struct[0].Gy[102,11] = i_W2lv_c_i
        struct[0].Gy[102,104] = v_W2lv_c_r
        struct[0].Gy[102,105] = v_W2lv_c_i
        struct[0].Gy[103,6] = -i_W2lv_a_i
        struct[0].Gy[103,7] = i_W2lv_a_r
        struct[0].Gy[103,100] = v_W2lv_a_i
        struct[0].Gy[103,101] = -v_W2lv_a_r
        struct[0].Gy[104,8] = -i_W2lv_b_i
        struct[0].Gy[104,9] = i_W2lv_b_r
        struct[0].Gy[104,102] = v_W2lv_b_i
        struct[0].Gy[104,103] = -v_W2lv_b_r
        struct[0].Gy[105,10] = -i_W2lv_c_i
        struct[0].Gy[105,11] = i_W2lv_c_r
        struct[0].Gy[105,104] = v_W2lv_c_i
        struct[0].Gy[105,105] = -v_W2lv_c_r
        struct[0].Gy[106,6] = 1.0*v_W2lv_a_r*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)/V_base_W2lv
        struct[0].Gy[106,7] = 1.0*v_W2lv_a_i*(v_W2lv_a_i**2 + v_W2lv_a_r**2)**(-0.5)/V_base_W2lv
        struct[0].Gy[106,106] = -1
        struct[0].Gy[107,42] = 1.0*v_W2mv_a_r*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)/V_base_W2mv
        struct[0].Gy[107,43] = 1.0*v_W2mv_a_i*(v_W2mv_a_i**2 + v_W2mv_a_r**2)**(-0.5)/V_base_W2mv
        struct[0].Gy[107,107] = -1
        struct[0].Gy[108,106] = K_p_v_W2lv*(u_ctrl_v_W2lv - 1.0)
        struct[0].Gy[108,107] = -K_p_v_W2lv*u_ctrl_v_W2lv
        struct[0].Gy[108,108] = -1
        struct[0].Gy[109,58] = 1.0*S_base_W2lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)]))
        struct[0].Gy[109,59] = 1.0*S_base_W2lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W2lv, I_max_W2lv < -i_reac_ref_W2lv), (I_max_W2lv, I_max_W2lv < i_reac_ref_W2lv), (i_reac_ref_W2lv, True)]))
        struct[0].Gy[109,108] = S_base_W2lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W2lv < i_reac_ref_W2lv) | (I_max_W2lv < -i_reac_ref_W2lv)), (1, True)]))
        struct[0].Gy[109,109] = -1
        struct[0].Gy[110,12] = i_W3lv_a_r
        struct[0].Gy[110,13] = i_W3lv_a_i
        struct[0].Gy[110,110] = v_W3lv_a_r
        struct[0].Gy[110,111] = v_W3lv_a_i
        struct[0].Gy[111,14] = i_W3lv_b_r
        struct[0].Gy[111,15] = i_W3lv_b_i
        struct[0].Gy[111,112] = v_W3lv_b_r
        struct[0].Gy[111,113] = v_W3lv_b_i
        struct[0].Gy[112,16] = i_W3lv_c_r
        struct[0].Gy[112,17] = i_W3lv_c_i
        struct[0].Gy[112,114] = v_W3lv_c_r
        struct[0].Gy[112,115] = v_W3lv_c_i
        struct[0].Gy[113,12] = -i_W3lv_a_i
        struct[0].Gy[113,13] = i_W3lv_a_r
        struct[0].Gy[113,110] = v_W3lv_a_i
        struct[0].Gy[113,111] = -v_W3lv_a_r
        struct[0].Gy[114,14] = -i_W3lv_b_i
        struct[0].Gy[114,15] = i_W3lv_b_r
        struct[0].Gy[114,112] = v_W3lv_b_i
        struct[0].Gy[114,113] = -v_W3lv_b_r
        struct[0].Gy[115,16] = -i_W3lv_c_i
        struct[0].Gy[115,17] = i_W3lv_c_r
        struct[0].Gy[115,114] = v_W3lv_c_i
        struct[0].Gy[115,115] = -v_W3lv_c_r
        struct[0].Gy[116,12] = 1.0*v_W3lv_a_r*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)/V_base_W3lv
        struct[0].Gy[116,13] = 1.0*v_W3lv_a_i*(v_W3lv_a_i**2 + v_W3lv_a_r**2)**(-0.5)/V_base_W3lv
        struct[0].Gy[116,116] = -1
        struct[0].Gy[117,48] = 1.0*v_W3mv_a_r*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)/V_base_W3mv
        struct[0].Gy[117,49] = 1.0*v_W3mv_a_i*(v_W3mv_a_i**2 + v_W3mv_a_r**2)**(-0.5)/V_base_W3mv
        struct[0].Gy[117,117] = -1
        struct[0].Gy[118,116] = K_p_v_W3lv*(u_ctrl_v_W3lv - 1.0)
        struct[0].Gy[118,117] = -K_p_v_W3lv*u_ctrl_v_W3lv
        struct[0].Gy[118,118] = -1
        struct[0].Gy[119,58] = 1.0*S_base_W3lv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)]))
        struct[0].Gy[119,59] = 1.0*S_base_W3lv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_W3lv, I_max_W3lv < -i_reac_ref_W3lv), (I_max_W3lv, I_max_W3lv < i_reac_ref_W3lv), (i_reac_ref_W3lv, True)]))
        struct[0].Gy[119,118] = S_base_W3lv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_W3lv < i_reac_ref_W3lv) | (I_max_W3lv < -i_reac_ref_W3lv)), (1, True)]))
        struct[0].Gy[119,119] = -1
        struct[0].Gy[120,18] = i_STlv_a_r
        struct[0].Gy[120,19] = i_STlv_a_i
        struct[0].Gy[120,120] = v_STlv_a_r
        struct[0].Gy[120,121] = v_STlv_a_i
        struct[0].Gy[121,20] = i_STlv_b_r
        struct[0].Gy[121,21] = i_STlv_b_i
        struct[0].Gy[121,122] = v_STlv_b_r
        struct[0].Gy[121,123] = v_STlv_b_i
        struct[0].Gy[122,22] = i_STlv_c_r
        struct[0].Gy[122,23] = i_STlv_c_i
        struct[0].Gy[122,124] = v_STlv_c_r
        struct[0].Gy[122,125] = v_STlv_c_i
        struct[0].Gy[123,18] = -i_STlv_a_i
        struct[0].Gy[123,19] = i_STlv_a_r
        struct[0].Gy[123,120] = v_STlv_a_i
        struct[0].Gy[123,121] = -v_STlv_a_r
        struct[0].Gy[124,20] = -i_STlv_b_i
        struct[0].Gy[124,21] = i_STlv_b_r
        struct[0].Gy[124,122] = v_STlv_b_i
        struct[0].Gy[124,123] = -v_STlv_b_r
        struct[0].Gy[125,22] = -i_STlv_c_i
        struct[0].Gy[125,23] = i_STlv_c_r
        struct[0].Gy[125,124] = v_STlv_c_i
        struct[0].Gy[125,125] = -v_STlv_c_r
        struct[0].Gy[126,18] = 1.0*v_STlv_a_r*(v_STlv_a_i**2 + v_STlv_a_r**2)**(-0.5)/V_base_STlv
        struct[0].Gy[126,19] = 1.0*v_STlv_a_i*(v_STlv_a_i**2 + v_STlv_a_r**2)**(-0.5)/V_base_STlv
        struct[0].Gy[126,126] = -1
        struct[0].Gy[127,54] = 1.0*v_STmv_a_r*(v_STmv_a_i**2 + v_STmv_a_r**2)**(-0.5)/V_base_STmv
        struct[0].Gy[127,55] = 1.0*v_STmv_a_i*(v_STmv_a_i**2 + v_STmv_a_r**2)**(-0.5)/V_base_STmv
        struct[0].Gy[127,127] = -1
        struct[0].Gy[128,126] = K_p_v_STlv*(u_ctrl_v_STlv - 1.0)
        struct[0].Gy[128,127] = -K_p_v_STlv*u_ctrl_v_STlv
        struct[0].Gy[128,128] = -1
        struct[0].Gy[129,58] = 1.0*S_base_STlv*v_STmv_c_r*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)]))
        struct[0].Gy[129,59] = 1.0*S_base_STlv*v_STmv_c_i*(v_STmv_c_i**2 + v_STmv_c_r**2)**(-0.5)*Piecewise(np.array([(-I_max_STlv, I_max_STlv < -i_reac_ref_STlv), (I_max_STlv, I_max_STlv < i_reac_ref_STlv), (i_reac_ref_STlv, True)]))
        struct[0].Gy[129,128] = S_base_STlv*(v_STmv_c_i**2 + v_STmv_c_r**2)**0.5*Piecewise(np.array([(0, (I_max_STlv < i_reac_ref_STlv) | (I_max_STlv < -i_reac_ref_STlv)), (1, True)]))
        struct[0].Gy[129,129] = -1

        struct[0].Gu[24,0] = 0.0241740531829170
        struct[0].Gu[24,1] = 0.0402900886381950
        struct[0].Gu[24,2] = -4.31760362252812E-18
        struct[0].Gu[24,3] = 4.66248501556824E-18
        struct[0].Gu[24,4] = -3.49608108880335E-18
        struct[0].Gu[24,5] = 4.19816664496737E-18
        struct[0].Gu[24,6] = 1
        struct[0].Gu[25,0] = -0.0402900886381950
        struct[0].Gu[25,1] = 0.0241740531829170
        struct[0].Gu[25,2] = -4.66248501556824E-18
        struct[0].Gu[25,3] = -4.31760362252812E-18
        struct[0].Gu[25,4] = -4.19816664496737E-18
        struct[0].Gu[25,5] = -3.49608108880335E-18
        struct[0].Gu[25,7] = 1
        struct[0].Gu[26,0] = -2.07254761002657E-18
        struct[0].Gu[26,1] = 6.30775359573304E-19
        struct[0].Gu[26,2] = 0.0241740531829170
        struct[0].Gu[26,3] = 0.0402900886381950
        struct[0].Gu[26,4] = -1.78419315993592E-17
        struct[0].Gu[26,5] = 9.01107656533306E-19
        struct[0].Gu[26,8] = 1
        struct[0].Gu[27,0] = -6.30775359573304E-19
        struct[0].Gu[27,1] = -2.07254761002657E-18
        struct[0].Gu[27,2] = -0.0402900886381950
        struct[0].Gu[27,3] = 0.0241740531829170
        struct[0].Gu[27,4] = -9.01107656533306E-19
        struct[0].Gu[27,5] = -1.78419315993592E-17
        struct[0].Gu[27,9] = 1
        struct[0].Gu[28,0] = -1.35166148479994E-18
        struct[0].Gu[28,1] = -7.20886125226632E-19
        struct[0].Gu[28,2] = -1.71210454741325E-17
        struct[0].Gu[28,3] = -4.50553828266631E-19
        struct[0].Gu[28,4] = 0.0241740531829170
        struct[0].Gu[28,5] = 0.0402900886381950
        struct[0].Gu[28,10] = 1
        struct[0].Gu[29,0] = 7.20886125226632E-19
        struct[0].Gu[29,1] = -1.35166148479994E-18
        struct[0].Gu[29,2] = 4.50553828266631E-19
        struct[0].Gu[29,3] = -1.71210454741325E-17
        struct[0].Gu[29,4] = -0.0402900886381950
        struct[0].Gu[29,5] = 0.0241740531829170
        struct[0].Gu[29,11] = 1
        struct[0].Gu[30,12] = 1
        struct[0].Gu[31,13] = 1
        struct[0].Gu[32,14] = 1
        struct[0].Gu[33,15] = 1
        struct[0].Gu[34,16] = 1
        struct[0].Gu[35,17] = 1
        struct[0].Gu[36,18] = 1
        struct[0].Gu[37,19] = 1
        struct[0].Gu[38,20] = 1
        struct[0].Gu[39,21] = 1
        struct[0].Gu[40,22] = 1
        struct[0].Gu[41,23] = 1
        struct[0].Gu[42,24] = 1
        struct[0].Gu[43,25] = 1
        struct[0].Gu[44,26] = 1
        struct[0].Gu[45,27] = 1
        struct[0].Gu[46,28] = 1
        struct[0].Gu[47,29] = 1
        struct[0].Gu[48,30] = 1
        struct[0].Gu[49,31] = 1
        struct[0].Gu[50,32] = 1
        struct[0].Gu[51,33] = 1
        struct[0].Gu[52,34] = 1
        struct[0].Gu[53,35] = 1
        struct[0].Gu[54,36] = 1
        struct[0].Gu[55,37] = 1
        struct[0].Gu[56,38] = 1
        struct[0].Gu[57,39] = 1
        struct[0].Gu[58,40] = 1
        struct[0].Gu[59,41] = 1
        struct[0].Gu[84,0] = -0.0241740531829170
        struct[0].Gu[84,1] = -0.0402900886381950
        struct[0].Gu[84,2] = 4.31760362252812E-18
        struct[0].Gu[84,3] = -4.66248501556824E-18
        struct[0].Gu[84,4] = 3.49608108880335E-18
        struct[0].Gu[84,5] = -4.19816664496737E-18
        struct[0].Gu[85,0] = 0.0402900886381950
        struct[0].Gu[85,1] = -0.0241740531829170
        struct[0].Gu[85,2] = 4.66248501556824E-18
        struct[0].Gu[85,3] = 4.31760362252812E-18
        struct[0].Gu[85,4] = 4.19816664496737E-18
        struct[0].Gu[85,5] = 3.49608108880335E-18
        struct[0].Gu[86,0] = 2.07254761002657E-18
        struct[0].Gu[86,1] = -6.30775359573304E-19
        struct[0].Gu[86,2] = -0.0241740531829170
        struct[0].Gu[86,3] = -0.0402900886381950
        struct[0].Gu[86,4] = 1.78419315993592E-17
        struct[0].Gu[86,5] = -9.01107656533306E-19
        struct[0].Gu[87,0] = 6.30775359573304E-19
        struct[0].Gu[87,1] = 2.07254761002657E-18
        struct[0].Gu[87,2] = 0.0402900886381950
        struct[0].Gu[87,3] = -0.0241740531829170
        struct[0].Gu[87,4] = 9.01107656533306E-19
        struct[0].Gu[87,5] = 1.78419315993592E-17
        struct[0].Gu[88,0] = 1.35166148479994E-18
        struct[0].Gu[88,1] = 7.20886125226632E-19
        struct[0].Gu[88,2] = 1.71210454741325E-17
        struct[0].Gu[88,3] = 4.50553828266631E-19
        struct[0].Gu[88,4] = -0.0241740531829170
        struct[0].Gu[88,5] = -0.0402900886381950
        struct[0].Gu[89,0] = -7.20886125226632E-19
        struct[0].Gu[89,1] = 1.35166148479994E-18
        struct[0].Gu[89,2] = -4.50553828266631E-19
        struct[0].Gu[89,3] = 1.71210454741325E-17
        struct[0].Gu[89,4] = 0.0402900886381950
        struct[0].Gu[89,5] = -0.0241740531829170
        struct[0].Gu[98,44] = K_p_v_W1lv
        struct[0].Gu[98,45] = K_p_v_W1lv
        struct[0].Gu[98,46] = 1
        struct[0].Gu[108,49] = K_p_v_W2lv
        struct[0].Gu[108,50] = K_p_v_W2lv
        struct[0].Gu[108,51] = 1
        struct[0].Gu[118,54] = K_p_v_W3lv
        struct[0].Gu[118,55] = K_p_v_W3lv
        struct[0].Gu[118,56] = 1
        struct[0].Gu[128,59] = K_p_v_STlv
        struct[0].Gu[128,60] = K_p_v_STlv
        struct[0].Gu[128,61] = 1





@numba.njit(cache=True)
def Piecewise(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out

@numba.njit(cache=True)
def ITE(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out


@numba.njit(cache=True)
def Abs(x):
    return np.abs(x)



@numba.njit(cache=True) 
def daesolver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 
    
    Dt = struct[i].Dt 

    N_x = struct[i].N_x
    N_y = struct[i].N_y
    N_z = struct[i].N_z

    decimation = struct[i].decimation 
    eye = np.eye(N_x)
    t = struct[i].t 
    t_end = struct[i].t_end 
    if struct[i].it == 0:
        run(t,struct, 1) 
        struct[i].it_store = 0  
        struct[i]['T'][0] = t 
        struct[i].X[0,:] = struct[i].x[:,0]  
        struct[i].Y[0,:] = struct[i].y_run[:,0]  
        struct[i].Z[0,:] = struct[i].h[:,0]  

    solver = struct[i].solvern 
    while t<t_end: 
        struct[i].it += 1
        struct[i].t += Dt
        
        t = struct[i].t


            
        if solver == 5: # Teapezoidal DAE as in Milano's book

            run(t,struct, 2) 
            run(t,struct, 3) 

            x = np.copy(struct[i].x[:]) 
            y = np.copy(struct[i].y_run[:]) 
            f = np.copy(struct[i].f[:]) 
            g = np.copy(struct[i].g[:]) 
            
            for iter in range(struct[i].imax):
                run(t,struct, 2) 
                run(t,struct, 3) 
                run(t,struct,10) 
                run(t,struct,11) 
                
                x_i = struct[i].x[:] 
                y_i = struct[i].y_run[:]  
                f_i = struct[i].f[:] 
                g_i = struct[i].g[:]                 
                F_x_i = struct[i].Fx[:,:]
                F_y_i = struct[i].Fy[:,:] 
                G_x_i = struct[i].Gx[:,:] 
                G_y_i = struct[i].Gy[:,:]                

                A_c_i = np.vstack((np.hstack((eye-0.5*Dt*F_x_i, -0.5*Dt*F_y_i)),
                                   np.hstack((G_x_i,         G_y_i))))
                     
                f_n_i = x_i - x - 0.5*Dt*(f_i+f) 
                # print(t,iter,g_i)
                Dxy_i = np.linalg.solve(-A_c_i,np.vstack((f_n_i,g_i))) 
                
                x_i = x_i + Dxy_i[0:N_x]
                y_i = y_i + Dxy_i[N_x:(N_x+N_y)]

                struct[i].x[:] = x_i
                struct[i].y_run[:] = y_i

        # [f_i,g_i,F_x_i,F_y_i,G_x_i,G_y_i] =  smib_transient(x_i,y_i,u);
        
        # A_c_i = [[eye(N_x)-0.5*Dt*F_x_i, -0.5*Dt*F_y_i],
        #          [                G_x_i,         G_y_i]];
             
        # f_n_i = x_i - x - 0.5*Dt*(f_i+f);
        
        # Dxy_i = -A_c_i\[f_n_i.',g_i.'].';
        
        # x_i = x_i + Dxy_i(1:N_x);
        # y_i = y_i + Dxy_i(N_x+1:N_x+N_y);
                
                xy = np.vstack((x_i,y_i))
                max_relative = 0.0
                for it_var in range(N_x+N_y):
                    abs_value = np.abs(xy[it_var,0])
                    if abs_value < 0.001:
                        abs_value = 0.001
                                             
                    relative_error = np.abs(Dxy_i[it_var,0])/abs_value
                    
                    if relative_error > max_relative: max_relative = relative_error
                    
                if max_relative<struct[i].itol:
                    
                    break
                
                # if iter>struct[i].imax-2:
                    
                #     print('Convergence problem')

            struct[i].x[:] = x_i
            struct[i].y_run[:] = y_i
                
        # channels 
        it_store = struct[i].it_store
        if struct[i].it >= it_store*decimation: 
            struct[i]['T'][it_store+1] = t 
            struct[i].X[it_store+1,:] = struct[i].x[:,0] 
            struct[i].Y[it_store+1,:] = struct[i].y_run[:,0]
            struct[i].Z[it_store+1,:] = struct[i].h[:,0]
            struct[i].iters[it_store+1,0] = iter
            struct[i].it_store += 1 
            
    struct[i].t = t

    return t


