import socket
import time
import json
from threading import Thread
import pv_100
import numpy as np
import zmq

class emulator():

    def __init__(self) -> None:

        self.model = pv_100.model()
        self.N_gen = 10
        self.gen_names_list = ['LV' + f'{ig}'.zfill(3) for ig in range(1,1+self.N_gen)]

        # convenient indices:
        self.p_s_ppc_u_idxs = [  self.model.inputs_run_list.index(f'p_s_ppc_{name}') for name in self.gen_names_list]
        self.q_s_ppc_u_idxs = [  self.model.inputs_run_list.index(f'q_s_ppc_{name}') for name in self.gen_names_list]
        self.p_s_y_idxs = [  self.model.y_run_list.index(f'p_s_{name}') for name in self.gen_names_list]
        self.V_y_idxs = [  self.model.y_run_list.index(f'V_{name}') for name in self.gen_names_list]
        self.lvrt_ext_y_idxs = [  self.model.inputs_run_list.index(f'lvrt_ext_{name}') for name in self.gen_names_list]
        self.v_lvrt_idxs = [  self.model.params_list.index(f'v_lvrt_{name}') for name in self.gen_names_list]

        self.context = zmq.Context()

        # Socket to send messages
        self.sender_socket = self.context.socket(zmq.PAIR)
        self.sender_socket.bind("tcp://*:5555")

        # Socket to receive messages
        self.receiver_socket = self.context.socket(zmq.PAIR)
        self.receiver_socket.bind("tcp://*:5556")

        self.halt = False
        self.Dt_mid = 0.2

    def ini(self):
        self.model.Dt = 0.05
        params_ini = {}
        for it in range(1,1+self.N_gen):
            name = f'{it}'.zfill(3)
            params_ini.update({f'irrad_LV{name}':500,f'p_s_ppc_LV{name}':2})
            params_ini.update({f'i_sr_ref_LV{name}':0.8})
            

        self.model.ini(params_ini,'xy_0.json')
        self.model.p[self.v_lvrt_idxs] =0.0

    def start(self):

        self.halt = False
        self.step_loop_thread = Thread(target = self.step_loop)
        self.step_loop_thread.start()
        self.receive_loop = Thread(target = self.receive_loop)
        self.receive_loop.start()
        self.send_loop = Thread(target = self.send_loop)
        self.send_loop.start()

    def send_loop(self):
 
        while self.halt == False:
            # Send random float data to process A
            p_s = self.model.xy[np.array(self.p_s_y_idxs)+self.model.N_x]
            self.sender_socket.send_json({'p_s':list(p_s)})
            time.sleep(0.1) 


    def receive_loop(self):

        while self.halt == False:
            # Send random float data to process A

            # Wait for a response from process A
            received = self.receiver_socket.recv_json()

            if 'halt' in received:
                if received['halt'] == True:
                    self.halt = True

            if 'p_s_ppc' in received:
                p_s_ppc = np.array(received['p_s_ppc'])  
                self.model.u_run[self.p_s_ppc_u_idxs] = p_s_ppc

            if 'q_s_ppc' in received:
                q_s_ppc = np.array(received['q_s_ppc'])  
                self.model.u_run[self.q_s_ppc_u_idxs] = q_s_ppc

            if 'v_ref_GRID' in received:
                self.model.u_run[self.model.inputs_run_list.index('v_ref_GRID')] = received['v_ref_GRID']


            time.sleep(0.001)  # wait for 1 second before sending the next message




    def step_loop(self):

        t = 0.0
        self.model.step(t+self.Dt_mid,{})
         

        t_0 = time.perf_counter()

        while self.halt == False:

            t = time.perf_counter()-t_0

            self.model.step(t+self.Dt_mid,{})


            V_poi = self.model.get_value('V_POI')

            V_LV = self.model.xy[np.array(self.V_y_idxs)+self.model.N_x]
            lvrt_ext = np.zeros(self.N_gen)
            lvrt_ext[V_LV < 0.8] = 1.0
            self.model.u_run[self.lvrt_ext_y_idxs] = lvrt_ext

            p_s_LV001 = self.model.get_value('p_s_LV001')
            q_s_LV001 = self.model.get_value('q_s_LV001')
            p_line_POI_GRID = self.model.get_value('p_line_POI_GRID')*self.model.get_value('S_base')/1e6
            q_line_POI_GRID = self.model.get_value('q_line_POI_GRID')*self.model.get_value('S_base')/1e6

            t_end = time.perf_counter()-t_0
            print(f't = {t:0.3f}, t_s = {t_end-t:0.3f}, V_poi = {V_poi:0.3f}, P_poi = {p_line_POI_GRID:4.2f} MW, Q_poi = {q_line_POI_GRID:4.2f} Mvar, p_s_LV001 = {p_s_LV001:4.2f},  q_s_LV001 = {q_s_LV001:4.2f}')

            while (time.perf_counter() - t_0) < (t + self.Dt_mid):
                pass


if __name__ == "__main__":
    emu = emulator()
    emu.ini()
    emu.start()