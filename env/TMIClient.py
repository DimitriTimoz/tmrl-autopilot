import time
from threading import Lock, Thread

from tminterface.client import Client
from tminterface.interface import TMInterface


class SimStateClient(Client):
    """
    Client for a TMInterface instance.
    Its only job is to get the simulation state that is used by the gym env for reward computation.
    """

    def __init__(self):
        super().__init__()
        self.sim_state = None
        self.action_buffer = (0, 0)
        self.steer = 0
        self.gas = 0
        self.restart = False

    def on_run_step(self, iface, _time: int):
        self.sim_state = iface.get_simulation_state()
        
        accelerate = False
        brake = False
        if self.gas == 1:
            accelerate = True
        elif self.gas == -1:
            brake = True
            
        iface.set_input_state(accelerate=accelerate, brake=brake, steer=self.steer)
        if self.restart:
            iface.give_up()
            self.restart = False

class ThreadedClient:
    """
    Allows to run the Client in a separate thread, so that the gym env can run in the main thread.
    """

    def __init__(self) -> None:
        self.iface = TMInterface()
        self.tmi_client = SimStateClient()
        self._client_thread = Thread(target=self.client_thread, daemon=True)
        self._lock = Lock()
        self.data = None
        self._client_thread.start()
        self.gas = 0
        self.steer = 0
        self.restart_buff = False

    def client_thread(self):
        client = SimStateClient()
        print("ok")

        self.iface.register(client)
        self.iface.set_simulation_time_limit(100_000_000)
        while self.iface.running:
            time.sleep(0)
            self._lock.acquire()
            self.data = client.sim_state
            client.gas = self.gas
            client.steer = self.steer
            if self.restart_buff:
                client.restart = True
                self.restart_buff = False
            self._lock.release()
            
    def apply_action(self, gas: int, steer: int):
        assert type(gas) == int and type(steer) == int
        assert -1 <= gas <= 1 
        assert -65536 <= steer <= 65536
        self.gas = gas
        self.steer = steer

    def restart(self):
        self.restart_buff = True

if __name__ == "__main__":
    simthread = ThreadedClient()
