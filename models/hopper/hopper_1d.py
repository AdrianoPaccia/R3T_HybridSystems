import numpy as np
import pypolycontain as pp
from utils import convex_hull_of_point_and_polytope
from models.model import Model

class Hopper1D(Model):

    FLIGHT = 0
    CONTACT = 1
    BOUNCE = -1

    modes = {FLIGHT, CONTACT, BOUNCE}

    def __init__(self, m=1, l=1, p=0.1, b=0.9, g=9.8, 
                 initial_state=np.array([2.0,0.0]), 
                 input_limits =np.array([0,80]).reshape(1,2),
                 goal_states  =[np.array([3, 0.0])],
                 eps_goal = 0.1,
                 epsilon = 1e-7, 
                 dt=0.01,
                 fast_forward=True):
        super().__init__(initial_state, input_limits, dt)

        self.x_dim = 2
        self.u_dim = 1

        self.m = m
        self.l = l
        self.p = p
        self.b = b
        self.g = g

        self.epsilon = epsilon
        
        self.dt = dt

        self.u_dim = 1

        self.motion_primitives = [self.input_limits[:,0],
                                  self.u_bar,
                                  self.input_limits[:,1]]

        self.goal_states = goal_states
        self.eps_goal    = eps_goal

        self.fast_forward = fast_forward

    def f_flight(self, x):
        # not actuated
        dx = np.array([x[1], -self.g])
        return dx

    def f_contact(self, x, u):
        dx = np.asarray([x[1], u[0]/self.m-self.g])
        return dx
    
    def f_bounce(self, x):
        x_plus = np.array([self.l+self.epsilon, -x[1]*self.b])
        return x_plus


    def check_flight(self, x): return x[0]>self.l+self.p
    def check_contact(self, x): return self.l<x[0] and x[0]<=self.l+self.p
    def check_bounce(self, x): return x[0]<=self.l

    def get_mode(self, x):
        # modes are mutually exclusive
        if self.check_flight(x):
            return self.FLIGHT
        elif self.check_contact(x):
            return self.CONTACT
        elif self.check_bounce(x):
            return self.BOUNCE
        else:
            raise Exception()

       

    # return next state (x, x_dot) after applying control u for time dt
    def step(self, x, u, dt):
        mode = self.get_mode(x)
        if   mode == Hopper1D.FLIGHT:
            # print("flight")
            x_next = x + self.f_flight(x)*dt
        elif mode == Hopper1D.CONTACT:
            # print("contact")
            x_next = x + self.f_contact(x,u)*dt
        elif mode == Hopper1D.BOUNCE:
            # print("bounce")
            x_next = self.f_bounce(x)
        else:
            raise Exception()
        return x_next

    def goal_check(self, x:np.ndarray):#->tuple[bool, float]:      
        min_dist = np.inf
        goal = False

        for goal_state in self.goal_states:
            dist = np.linalg.norm(x-goal_state)
            if dist<min_dist:
                min_dist = dist

        if min_dist < self.eps_goal:
            goal = True
        return goal, min_dist
    
    def sample(self, **kwargs):#->np.ndarray:
        gaussian_ratio = 0.4
        rnd = np.zeros(2)
        rnd[0] = np.random.normal(self.l+0.5*self.p,3*self.p)
        rnd[1] = (np.random.rand(1)-0.5)*2*10
        if np.random.rand(1) > gaussian_ratio:
            rnd = np.random.rand(2)
            rnd[0] = rnd[0]*5+0.5
            rnd[1] = (rnd[1]-0.5)*2*10
            goal_bias = np.random.rand(1)
            if goal_bias<0.1:
                return self.goal_states[0]
            return rnd
        return rnd

    def expand_toward(self, x_near:np.ndarray, x_rand:np.ndarray, dt:float):#->tuple[np.ndarray, np.ndarray]:
        pass
    
    def linearize_at(self, x:np.ndarray, u:np.ndarray, dt:float, mode=None):
        if mode == None:
            mode = self.get_mode(x)
        # no dependence on x,u as expected (linear dynamics)
        if mode == self.FLIGHT:
            A = (np.array([[0,1],[0,0]])*dt + np.eye(2))
            B = np.zeros((2,1))*dt
            c = np.array([0, -self.g])*dt
        elif mode == self.CONTACT:
            A = (np.array([[0,1],[0,0]])*dt + np.eye(2))
            B = (np.array([[0, 1/self.m]])*dt).reshape(2,1)
            c = np.array([0, -self.g])*dt
        elif mode == self.BOUNCE:
            # xdot_k+1 = -b xdot_k
            B = np.array([[0,0]]).reshape(2,1)
            A = np.array([[0, 0],[0,-self.b]])
            c = np.array([self.l+self.epsilon, 0.0])

        else:
            raise Exception()
        
        return A,B,c
    
    def get_reachable_sampled(self, x:np.ndarray, dt:float):#->tuple[np.ndarray, np.ndarray]:

        iters = int(dt//self.dt)
        states = []
        controls = []
        for mp in self.motion_primitives:
            s = np.zeros((iters,self.x_dim))
            u = np.zeros((iters,self.u_dim))
            x_r = x.copy()
            for i in range(iters):
                m = self.get_mode(x_r)
                if m == self.FLIGHT:
                    inp = np.array([0.0]).reshape(1,-1)
                else:
                    inp = mp
                x_r = self.step(x_r, inp, self.dt)
                
                s[i] = x_r
                u[i] = inp

            states.append(s)
            controls.append(u)
        return states, controls

    def ffw(self, x: np.ndarray) :#-> list[np.ndarray]:
        if self.fast_forward:
                    
            no_inputs = []

            x_ = x.copy()
            while self.get_mode(x_) != self.CONTACT:
                x_ = self.step(x_,None,self.dt)
                no_inputs.append(x_)

            return np.array(no_inputs).reshape(-1,self.x_dim), np.zeros((len(no_inputs),self.u_dim))

        else:
            return super().ffw(x)
        

    def get_reachable_AH(self, x: np.ndarray, dt: float, convex_hull: bool = False) :#-> list[tuple[np.ndarray, pp.AH_polytope]]:
        A, B, c = self.linearize_at(x, self.u_bar, dt)
        x_next = (A@x + B@self.u_bar + c)
        G = (B@self.u_diff).reshape(self.x_dim, self.u_dim)
        AH = pp.to_AH_polytope(pp.zonotope(G,x_next.reshape(-1,1)))
        if convex_hull:
            AH = convex_hull_of_point_and_polytope(x.reshape(-1,1), AH)
        return [(x_next, AH)]
    
    def expand_toward_pinv(self, x_near: np.ndarray, x_rand: np.ndarray, dt: float) :#-> tuple[np.ndarray, np.ndarray]:
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

