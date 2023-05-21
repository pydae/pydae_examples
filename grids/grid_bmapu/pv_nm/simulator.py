import threading
import json
import time
from flask import Flask, request
import inv100
import numpy as np
import signal
import functools
import pandas as pd


app = Flask(__name__)
setpoints = {}
measurements = {} 
t_ini = 0

# Ruta de lectura GET
@app.route('/api/read', methods=['GET'])
def read_json():
    global measurements
    return json.dumps(measurements)
     

# Ruta de escritura POST
@app.route('/api/write', methods=['POST'])
def write_json():
    global setpoints
    data = request.get_json()
    setpoints = dict(data)
    return 'Exito en el POST '

def start_server():
    app.run(debug=True, port=8080,use_reloader=False)

# Listen for incoming datagrams
def simulation(intervalo):
    #model.Dt = intervalo/2000000000
    model.Dt = 0.025
    #Primera ejecucción siempre es la que más tarda
    print("Tiempo que tarda la primera ejecución: tstart = {:.2f}".format(time.time()))
    model.step(0.5,setpoints)
    print("Tiempo que tarda la primera ejecución: tstop = {:.2f}".format(time.time()))


    tiempo_ejecucion = np.zeros(5000)
    tiempo_simulacion = np.zeros(5000)
    i = 0
    t_ini = time.perf_counter_ns()
        
    # while tiempo_ejecucion[9999] == 0.0:
    #     t_start = time.perf_counter_ns()
    #     t_sim= time.perf_counter_ns() - t_ini
    #     model.step(t_sim/1000000000 + 0.5 ,setpoints)
    #     tiempo_ejecucion[i] = time.perf_counter_ns() - t_start
    #     tiempo_simulacion[i] = t_sim
    #     max = np.max([0.0,intervalo-(time.perf_counter_ns()-t_start)])
    #     print("me duermo {:.6f}\n\n".format(max))
    #     t_start_while = time.perf_counter_ns()
    #     while True:
    #         if time.perf_counter_ns() - t_start_while > max:
    #             break
    #     i+=1
    #     print(i)
    while tiempo_ejecucion[4999] == 0.0:
        t_start = time.perf_counter_ns()
        model.step(0.05*i+0.5,{})
        tiempo_ejecucion[i] = time.perf_counter_ns() - t_start
        i+=1
        print(i)
    j = 4999
    for i in range(5000):
        if j !=0:
            tiempo_simulacion[j]= tiempo_simulacion[j] - tiempo_simulacion[j-1]
            j -= 1
    df = pd.DataFrame({"tiempo_ejecucion":tiempo_ejecucion})
    print(df['tiempo_ejecucion'].describe())
    df.to_csv('inv100(3)withoutsetpoint_50ms.csv')
    

if __name__ == '__main__':
 
    model = inv100.model()
    inidiccionario={}
    v_dc_name = "v_dc_ref_"
    p_s_ppc_name = "p_s_ppc_"
    q_s_ppc_name = "q_s_ppc_" 
    q_s_name = "q_s_ref_"
    k_pdc_name = "K_pdc_"
    irrad_name = "irrad_"
    C_dc_name = "C_dc_"
    
    for i in range(11):
        if i != 0:
            inidiccionario[v_dc_name + str(i)] = 1.5
            inidiccionario[k_pdc_name + str(i)] = 100
            inidiccionario[irrad_name + str(i)] = 500
            inidiccionario[C_dc_name + str(i)] = 0.5
            
    print(inidiccionario)
    model.ini({},'xy_0.json')
   
    for i in range(100):
        if i != 0:
            setpoints[p_s_ppc_name + str(i)] = model.get_value(p_s_ppc_name+str(i))
            setpoints[q_s_ppc_name + str(i)] = model.get_value(q_s_ppc_name+str(i))
    print(setpoints)
    server_thread = threading.Thread(target=start_server)
    server_thread.start()
    print('Servidor iniciado en segundo plano.')
    time.sleep(1)
    simulation(50000000)
    #
    #start_server()
    
     
    