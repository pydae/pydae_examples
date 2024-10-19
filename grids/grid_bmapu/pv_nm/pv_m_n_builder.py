import pydae.build_cffi as db
from pydae.bmapu import bmapu_builder
from pydae.build_v2 import builder
import time 

t_0 = time.time()
M = 1
N = 1

S_pv_mva = 1.0

data = {
    "system":{"name":f"pv_{M}_{N}","S_base":100e6,"K_p_agc":0.0,"K_i_agc":0.0,"K_xif":0.01},
    "buses":[
        {"name":"POI_MV","P_W":0.0,"Q_var":0.0,"U_kV":20.0},
        {"name":   "POI","P_W":0.0,"Q_var":0.0,"U_kV":132.0},
        {"name":  "GRID","P_W":0.0,"Q_var":0.0,"U_kV":132.0}
    ],
    "lines":[
        {"bus_j":"POI_MV","bus_k": "POI","X_pu":0.05,"R_pu":0.0,"Bs_pu":0.0,"S_mva":120},
        {"bus_j":   "POI","bus_k":"GRID","X_pu":0.02,"R_pu":0.0,"Bs_pu":0.0,"S_mva":120, 'sym':True, 'monitor':True}
        ],
    "pvs":[],
    "genapes":[{
          "bus":"GRID","S_n":1000e6,"F_n":50.0,"X_v":0.001,"R_v":0.0,
          "K_delta":0.001,"K_alpha":1e-6}]
    }

for i_m in range(1,M+1):
    name_j = "POI_MV"
    for i_n in range(1,N+1):
        name = f"{i_m}".zfill(2) + f"{i_n}".zfill(2)
        name_k = 'MV' + name

        data['buses'].append({"name":f"LV{name}","P_W":0.0,"Q_var":0.0,"U_kV":0.4})
        data['buses'].append({"name":f"MV{name}","P_W":0.0,"Q_var":0.0,"U_kV":20.0})

        data['lines'].append({"bus_j":f"LV{name}","bus_k":f"MV{name}","X_pu":0.05,"R_pu":0.0,"Bs_pu":0.0,"S_mva":1.2*S_pv_mva,"monitor":False})
        data['lines'].append({"bus_j":f"{name_k}","bus_k":f"{name_j}","X_pu":0.01,"R_pu":0.01,"Bs_pu":0.0,"S_mva":1.2*S_pv_mva*(N-i_n+1),"monitor":False})
        name_j = name_k
        data['pvs'].append({"bus":f"LV{name}","type":"pv_dq_d","S_n":S_pv_mva*1e6,"U_n":400.0,"F_n":50.0,"X_s":0.1,"R_s":0.01,"monitor":False,
                            "I_sc":8,"V_oc":42.1,"I_mp":3.56,"V_mp":33.7,"K_vt":-0.160,"K_it":0.065,"N_pv_s":25,"N_pv_p":250})
    
#grid = bmapu_builder.bmapu('spvib.json')
grid = bmapu_builder.bmapu(data)

grid.uz_jacs = False
grid.verbose = True
grid.construct(f'pv_{M}_{N}')

t_1 = time.time()


b = builder(grid.sys_dict,verbose=True)
b.sparse = True
b.mkl = True

b.dict2system()
b.functions()
b.jacobians()
b.cwrite()

self = b
N_x = len(self.f_ini_list)
N_y = len(self.g_ini_list)
N = N_x + N_y
print(f"Number of dynamic equations: {N_x}")
print(f"Number of algebraic equations: {N_y}")
print(f"Dense jac_ini elements: {N**2}")
# print(f"Non zeros jac_ini num elements: {b.N_jac_ini_num}")
# print(f"Non zeros jac_ini up elements: {b.N_jac_ini_up}")
# print(f"Non zeros jac_ini xy elements: {b.N_jac_ini_xy}")
# print(f"Non zeros jac_ini total elements: {b.N_jac_ini_num+b.N_jac_ini_up+b.N_jac_ini_xy}")
 
b.template()
b.compile_mkl()
t_2 = time.time()

print(t_1-t_0)
print(t_2-t_1)