import numpy as np
from utils import normalize



# kinematic model
# for now discard dynamics
class Unicycle:

    v_primitives = {6.0}
    om_primitives = {-50.0, 0, 50}

    # (x,y,th, dx,dy,dth)
    def __init__(self, initial_state = np.array([0,0,0]), input_limits=None, dt=0.001):
        
        self.dt = dt

        self.initial_state = initial_state
        self.input_limits = input_limits

        self.motion_primitives = self._build_primitives()
        
    def f(self, q, u):
        dq = np.zeros(3)

        dq[0] = u[0]*np.cos(q[2])
        dq[1] = u[0]*np.sin(q[2])
        dq[2] = u[1]

        return dq

    # return next state (theta, theta_dot) after applying control u for time dt
    def step(self, q, u):
        # euler: q_k+1 = q_k + f(q_k, u_k)*dt

        q_next = q + self.f(q,u)*self.dt
        q_next[2] = normalize(q_next[2])

        return q_next
    
    # return q_new
    def extend_to(self, q_near, q_rand):
        min_d = np.inf
        q_next = None
        u = None
        for control in self.motion_primitives:
            q_cand = self.step(q_near, control)
            delta = q_rand-q_cand
            delta[2] = normalize(delta[2    ])

            # TODO find better metrics
            d = np.linalg.norm(delta)
            if d <= min_d:
                q_next = q_cand
                u = control
                min_d = d

        return q_next, u

    def _build_primitives(self):
        primitives = []

        for v in (self.v_primitives):
            for om in (self.om_primitives):
                primitive = np.array([v,om])
                primitives.append(primitive)

        return primitives

    def get_reachable_points(self, state):

        states = []
        controls = []

        for control in self.motion_primitives:
            cand = self.step(state, control)
            states.append(cand)
            controls.append(control)

        return states, controls