import numpy as np
import matplotlib.pyplot as plt
import pv_1_2 as pv

### Instantiation
model = pv.model()

### Initial state computation
params = {}
M  = 1
N  = 2
for i_m in range(1,M+1):
    for i_n in range(1,N+1):
        name = f"{i_m}".zfill(2) + f"{i_n}".zfill(2)
        params.update({f'T_lp1_LV{name}':0.01})
        params.update({f'irrad_LV{name}':1000+(np.random.rand()-0.5)*0})
        params.update({f'p_s_ppc_LV{name}':0.1,f'q_s_ppc_LV{name}':0.2})
        params.update({f'N_pv_s_LV{name}':20, f'N_pv_p_LV{name}':200})

#model.report_u()
#model.report_x()
#model.report_y()
model.ini(params,'xy_0.json')


model.report_y()
model.run(10.0,{'q_s_ppc_LV0101':1})
model.post();

print(model.get_value('q_s_ppc_LV0101'))

print(model.get_value('q_s_LV0101'))
