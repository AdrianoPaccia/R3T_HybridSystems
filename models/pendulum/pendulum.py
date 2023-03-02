import numpy as np
from utils import normalize, convex_hull_of_point_and_polytope
import pypolycontain as pp
from models.model import Model
import pypolycontain as pp


class Pendulum(Model):
    
    def __init__(self, m=1, m_l=0, l=0.5, g=9.81, b=0.1, 
                       initial_state = np.array([0,0]), 
                       goal_states   = [np.array([np.pi, 0.0]), np.array([-np.pi, 0])],
                       input_limits = np.array([-1, 1]).reshape(1,-1),
                       dt=0.05,
                       eps_goal=0.05):
        super().__init__(initial_state, input_limits, dt)

        self.x_dim = 2
        self.u_dim = 1

        self.m = m
        self.m_l = m_l
        self.l = l
        self.g = g
        self.b = b
        self.I = m*l**2 + (m_l*l**2)/12

        self.goal_states = goal_states


        self.motion_primitives = [self.input_limits[:,0],
                                  self.u_bar,
                                  self.input_limits[:,1]]


        self.eps_goal = eps_goal

    def f(self, x:np.ndarray, u:np.ndarray):
        dx = np.zeros_like(x)

        t = -(self.m*self.g*self.l*np.sin(x[0]) + self.m_l*self.g*self.l*np.sin(x[0])/2)

        dx[0] = x[1]
        dx[1] = (1/self.I) * (t + u[0] - self.b*x[1])

        return dx

    # return next state (theta, theta_dot) after applying control u for time dt
    def step(self, x:np.ndarray, u:np.ndarray, dt:float):
        # euler: q_k+1 = q_k + f(q_k, u_k)*dt
        return x + self.f(x,u)*dt

    
    def goal_check(self, x: np.ndarray) :#-> tuple[bool, float]:
        
        min_dist = np.inf
        goal = False

        x_ = x.copy()
        x_[0] = normalize(x_[0])
        for goal_state in self.goal_states:
            dist = np.linalg.norm(x_-goal_state)
            if dist<min_dist:
                min_dist = dist

        if min_dist < self.eps_goal:
            goal = True
        return goal, min_dist


    def linearize_at(self, x: np.ndarray, u: np.ndarray, dt: float) :#-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.zeros((self.x_dim, self.x_dim))
        B = np.zeros((self.x_dim, self.u_dim))
        c = np.zeros(self.x_dim)

        A[0,0] = 0
        A[0,1] = 1
        A[1,0] = -(1/self.I)*(self.m*self.g*self.l*np.cos(x[0]) + self.m_l*self.g*self.l*np.cos(x[0])/2)
        A[1,1] = -(1/self.I)*self.b

        A = (np.eye(len(x)) + dt*A)

        B[0,0] = 0
        B[1,0] = (1/self.I)
        B *= dt
        
        c = np.ndarray.flatten(self.step(x,u,dt)) - np.ndarray.flatten(A@x) - np.ndarray.flatten(B*u)

        return A, B, c

    def sample(self, **kwargs) :#-> np.ndarray:
        goal_bias = np.random.rand(1)
        if goal_bias < 0.3:
            if goal_bias < 0.15: return self.goal_states[0]
            else:                return self.goal_states[1]
        else:
            rnd = (np.random.rand(2) -0.5)*2 # range between -1 and 1

            rnd[0]*= 4*np.pi/3
            rnd[1]*= 10

            return rnd
    
    def expand_toward_pinv(self, x_near:np.ndarray, x_rand:np.ndarray, dt:float):#->tuple[np.ndarray, np.ndarray]:
        # expand using pseudoinverse on linearized system
        A, B, c = self.linearize_at(x_near, self.u_bar, dt)

        u = np.linalg.pinv(B)@(x_rand - A@x_near - c)

        if u < self.input_limits[0][0]: u[0] = self.input_limits[0][0]
        if u > self.input_limits[0][1]: u[0] = self.input_limits[0][1]

        # the state has to be actually reachable so I step on the real environment with the systems's dt
        iters = int(dt//self.dt)

        states = np.zeros((iters,self.x_dim))
        controls = np.zeros((iters, self.u_dim))
        x = x_near
        for i in range(iters):
            x = self.step(x, u, self.dt)
            states[i] = x
            controls[i] = u
        
        return states, controls

    def expand_toward_samples(self, x_near: np.ndarray, x_rand: np.ndarray, dt: float) :#-> tuple[np.ndarray, np.ndarray]:
        states, controls = self.get_reachable_sampled(x_near, dt)
        
        min_distance = np.inf
        closest_state = None
        closest_u = None

        for i in range(len(states)):
            delta = np.linalg.norm(x_rand-states[i][-1])
            if delta < min_distance:
                min_distance = delta
                closest_state = states[i]
                closest_u = controls[i]

        return closest_state, closest_u


    def get_reachable_sampled(self, x: np.ndarray, dt: float) :#-> tuple[np.ndarray, np.ndarray]:

        iters = int(dt//self.dt)
        states = []
        controls = []

        for mp in self.motion_primitives:
            s = np.zeros((iters,self.x_dim))
            u = np.zeros((iters,self.u_dim))
            x_r = x
            for i in range(iters):
                x_r = self.step(x_r, u, self.dt)
                s[i] = x_r
                u[i] = mp

            states.append(s)
            controls.append(u)
        return np.array(states), np.array(controls)
    
    def get_reachable_AH(self, x: np.ndarray, dt: float, convex_hull: bool = False):
        A, B, c = self.linearize_at(x, self.u_bar, dt)
        x_next = (A@x + B@self.u_bar + c)

        G = (B@self.u_diff).reshape(self.x_dim, self.u_dim)
        AH = pp.to_AH_polytope(pp.zonotope(G,x_next.reshape(-1,1)))
        if convex_hull:
            AH = convex_hull_of_point_and_polytope(x.reshape(-1,1), AH)
        return [(x_next, AH)]
