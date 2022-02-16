# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 16:28:03 2022

@author: jmmau
"""

import numpy as np
import pandas as pd



def set_loads(excel_file,params,case,xy_0=False):
    
    if xy_0: 
        load_factor= 0.0 
    else: 
        load_factor = 1.0
        
    df_loads_ac = pd.read_excel(excel_file, 
                  sheet_name=case, 
                  header=1, names=None, index_col=0, 
                  usecols=[1,4,5], 
                  skiprows=[8,9,10,12,13,14,23,24,25,26])

    df_loads_dc = pd.read_excel(excel_file, 
                  sheet_name=case, 
                  header=1, names=None, index_col=0, 
                  usecols=[7,8,10], 
                  skiprows=[2,8,9,10,12,13,14,15,17,18,20,21,23,24,25,26])



    if case != 'Case 3':
        for item in df_loads_ac.index:
            p_kw   = df_loads_ac.loc[item]['P (kW)']
            q_kvar = df_loads_ac.loc[item]['Q (kvar)']
    
            for ph in ['a','b','c']:
                params.update({f'p_load_{item}_{ph}':load_factor*1e3*p_kw/3})
                params.update({f'q_load_{item}_{ph}':load_factor*1e3*q_kvar/3})
        
    if case == 'Case 3':
        for item in df_loads_ac.index:
            p_kw   = df_loads_ac.loc[item]['P (kW)']
            q_kvar = df_loads_ac.loc[item]['Q (kvar)']
    
            for ph in ['a']:
                params.update({f'p_load_{item}_{ph}':load_factor*1e3*p_kw})
                params.update({f'q_load_{item}_{ph}':load_factor*1e3*q_kvar})        
    
            for ph in ['b','c']:
                params.update({f'p_load_{item}_{ph}':0.0})
                params.update({f'q_load_{item}_{ph}':0.0})    
            
    for item in df_loads_dc.index:
        p_kw   = df_loads_dc.loc[item]['Pdc (kW)']

        for ph in ['a']:
            params.update({f'p_load_{item}_{ph}':load_factor*1e3*p_kw})
            params.update({f'q_load_{item}_{ph}':0.0})
        for ph in ['b','c']:
            params.update({f'p_load_{item}_{ph}':0.0})
            params.update({f'q_load_{item}_{ph}':0.0})

def get_head_power(grid):
    v_a_r,v_a_i,v_b_r,v_b_i,v_c_r,v_c_i = grid.get_mvalue(['v_C00_a_r','v_C00_a_i',
                                                           'v_C00_b_r','v_C00_b_i',
                                                           'v_C00_c_r','v_C00_c_i',])
    v_abc = np.array([v_a_r+1j*v_a_i, v_b_r+1j*v_b_i, v_c_r +1j*v_c_i])

    i_a_r,i_a_i,i_b_r,i_b_i,i_c_r,i_c_i = grid.get_mvalue(['i_t_C00_C01_1_a_r','i_t_C00_C01_1_a_i',
                                                           'i_t_C00_C01_1_b_r','i_t_C00_C01_1_b_i',
                                                           'i_t_C00_C01_1_c_r','i_t_C00_C01_1_c_i'])
    i_abc = np.array([i_a_r+1j*i_a_i, i_b_r+1j*i_b_i, i_c_r +1j*i_c_i])

    s_com = np.sum(v_abc*np.conj(i_abc))
   
    s_total = s_com
    
    return s_total

def set_v(grid,bus,V_rms,phases=['a','b','c'],phi=0.0):
        
    for ph,mult in zip(phases,[0.0,1.0,2.0]):
        v = V_rms/np.sqrt(3)*np.exp(1j*2/3*np.pi*mult)
        grid.set_value(f'v_{bus}_{ph}_r',v.real)
        grid.set_value(f'v_{bus}_{ph}_i',v.imag)
        
def get_bus_powers(gt_grid,bus_name):
    idx = gt_grid.bus_data['bus_id'].index(bus_name)
    p_a,q_a = gt_grid.buses[idx]['p_a'],gt_grid.buses[idx]['p_a']
    p_b,q_b = gt_grid.buses[idx]['p_b'],gt_grid.buses[idx]['q_b']
    p_c,q_c = gt_grid.buses[idx]['p_c'],gt_grid.buses[idx]['q_c']
    p = p_a + p_b + p_c
    q = q_a + q_b + q_c
    return p,q
    
