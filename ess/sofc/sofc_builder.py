

# out[0] = (2*K_r*u[1]/p[5] - q_h2)/Tau_f;
# out[1] = (-p_h2 + (-2*K_r*u[1] + q_h2)/p[1])/Tau_h2;
# out[2] = (-p_h2 + 2*K_r*u[1]/K_h2)/Tau_h2o;
# out[3] = (-p_o2 + (-K_r*u[1] + q_h2R_h01)/K_o2)/Tau_o2;


# 'q_h2', 'p_h2', 'p_h2o', 'p_o2'

# out[0] = 4.3083524205333364e-5*N0*u[0]*log(p_h2*sqrt(p_o2)) +N0 - y[0];


# 'K_r', 'K_h2', 'K_h2o', 'K_o2', 'R_h01', 'U_opt', 'Tau_f', 'Tau_h2', 'Tau_h2o', 'Tau_o2', 'N0', 'R', 'E0'

# params_list = ['K_r', 'K_h2', 'K_h2o', 'K_o2', 'R_h01', 'U_opt', 'Tau_f', 'Tau_h2', 'Tau_h2o', 'Tau_o2', 'N0', 'R', 'E0'] 
# params_values_list  = [1.166e-06, 0.000843, 0.000281, 0.00252, 1.145, 0.75, 5, 26.1, 78.3, 2.91, 450, 0.00032813, 1.18] 
# inputs_ini_list = ['temp', 'i_dc'] 
# inputs_ini_values_list  = [1273, 0.0] 
# inputs_run_list = ['temp', 'i_dc'] 
# inputs_run_values_list = [1273, 0.0] 
# outputs_list = ['V_nernst'] 
# x_list = ['q_h2', 'p_h2', 'p_h2o', 'p_o2'] 
# y_run_list = ['v_dc'] 
# y_ini_list = ['v_dc'] 

'''
out[0] = (2*p[0]*u[1]/p[5] - x[0])/p[6];
out[1] = (-x[1] + (-2*p[0]*u[1] + x[0])/p[1])/p[7];
out[2] = (-x[1] + 2*p[0]*u[1]/p[2])/p[8];
out[3] = (-x[3] + (-p[0]*u[1] + x[0]/p[4])/p[3])/p[9];
out[0] = 4.3083524205333364e-5*p[10]*u[0]*log(x[1]*sqrt(x[3])) + p[10] - y[0];
out[0] = 4.3083524205333364e-5*p[10]*u[0]*log(x[1]*sqrt(x[3]));
'''

import numpy as np
import sympy as sym
import pydae.build_cffi as db

# inputs
temp = sym.Symbol('temp', real=True)
i_dc_ref = sym.Symbol('i_dc_ref', real=True)
di_dc  = sym.Symbol('di_dc', real=True)

# dynamic states
q_h2 = sym.Symbol('q_h2', real=True)
p_h2 = sym.Symbol('p_h2', real=True)
p_h2o = sym.Symbol('p_h2o', real=True)
p_o2 = sym.Symbol('p_o2', real=True)
Di_dc  = sym.Symbol('Di_dc', real=True)

# algebraic states
v_dc = sym.Symbol('v_dc', real=True)

# parameters
K_r = sym.Symbol('K_r', real=True)
K_h2 = sym.Symbol('K_h2', real=True)
K_h2o = sym.Symbol('K_h2o', real=True)
K_o2 = sym.Symbol('K_o2', real=True)
R_h01 = sym.Symbol('R_h01', real=True)
U_opt = sym.Symbol('U_opt', real=True)
Tau_f = sym.Symbol('Tau_f', real=True)
Tau_h2 = sym.Symbol('Tau_h2', real=True)
Tau_h2o = sym.Symbol('Tau_h2o', real=True)
Tau_o2 = sym.Symbol('Tau_o2', real=True)
N0 = sym.Symbol('N0', real=True)
E0 = sym.Symbol('E0', real=True)
R_ohm = sym.Symbol('R_ohm', real=True)

F = 96.487e6 # Faraday's constant
R = 8314 # Universal gas constant
i_dc = i_dc_ref + Di_dc

dq_h2  = (2*K_r/U_opt*i_dc - q_h2)/Tau_f 
dp_h2  = (-p_h2 + (-2*K_r*i_dc + q_h2)/K_h2)/Tau_h2 
dp_h2o = (2*K_r*i_dc/K_h2o - p_h2o)/Tau_h2o 
dp_o2  = ((q_h2/R_h01 - K_r*i_dc)/K_o2 - p_o2)/Tau_o2 

dDi_dc  = di_dc - 1e-6*Di_dc  

V_ernst = R*temp/(2*F)*sym.ln(p_h2*sym.sqrt(p_o2)/p_h2o)*N0
g_v_dc = V_ernst + E0*N0 - R_ohm*N0*i_dc - v_dc 


 
u_dict = {}
u_dict.update({'temp':1273})
u_dict.update({'i_dc_ref':0.0})
u_dict.update({'di_dc':0.0})

params_dict = {}

params_dict.update({'K_r':1.166e-06})
params_dict.update({'K_h2':0.000843})
params_dict.update({'K_h2o':0.000281})
params_dict.update({'K_o2':0.00252})
params_dict.update({'R_h01':1.145})
params_dict.update({'U_opt':0.85})
params_dict.update({'Tau_f':5})
params_dict.update({'Tau_h2':26.1})
params_dict.update({'Tau_h2o':78.3})
params_dict.update({'Tau_o2':2.91})
params_dict.update({'N0':450})
params_dict.update({'E0':1.18})
params_dict.update({'R_ohm':3.2813e-004})
 
doc_dict = {}
doc_dict.update({'K_r':{'default':1.166e-06}, 'description':'', 'tex':''})

sys_dict = {'name':'sofc',
            'params_dict':params_dict,
            'f_list':[dq_h2,dp_h2,dp_h2o,dp_o2,dDi_dc],
            'x_list':[ q_h2, p_h2, p_h2o, p_o2,Di_dc],
            'g_list':[g_v_dc],
            'y_ini_list':[v_dc],
            'y_run_list':[v_dc],
            'u_ini_dict':u_dict,
            'u_run_dict':u_dict,
            'h_dict':{'V_ernst':V_ernst,'i_dc':i_dc}}

bldr = db.builder(sys_dict)
bldr.build()


# for it in range(len(inputs_ini_list)):
#     string = string.replace(f'u[{it}]', inputs_ini_list[it])
#     print(f'{inputs_ini_list[it]} = sym.Symbol({inputs_ini_list[it]}, real=True)')

# for it in range(len(x_list)):
#     string = string.replace(f'x[{it}]', x_list[it])
#     print(f'{x_list[it]} = sym.Symbol({x_list[it]}, real=True)')

# for it in range(len(y_run_list)):
#     string = string.replace(f'y[{it}]', params_list[it])
#     print(f'{y_run_list[it]} = sym.Symbol({y_run_list[it]}, real=True)')

# for it in range(len(params_list)):
#     string = string.replace(f'p[{it}]', params_list[it])
#     print(f'{params_list[it]} = sym.Symbol({params_list[it]}, real=True)')
    
# for it in range(len(params_list)):
#     name = f"'{params_list[it]}'"
#     print(f"{params_list[it]} = params_dict.update({{name}:{params_values_list[it]}})")
    

# print(string)
