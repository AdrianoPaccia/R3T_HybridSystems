import numpy as np
import pypolycontain as pp

class Model:

    def __init__(self, initial_state: np.ndarray, # shape (x_dim, )
                       input_limits:  np.ndarray,  # shape (u_dim, 2)
                       dt: float,
                       ):
        self.initial_state = initial_state.reshape(-1)
        self.input_limits = input_limits
        self.dt = dt

        self.u_bar =  ( self.input_limits[:,0] + self.input_limits[:,1] )/2
        self.u_diff = ( self.input_limits[:,1] - self.input_limits[:,0] )/2

        # TODO either this or motion primitives
        # both rely on distance metric which is not the best with hybrid systems
        # samples looks like the best because it does not do linearization thus it can handle
        # mode switches. It does discretize input space however...
        self.expand_toward = self.expand_toward_pinv
        # self.expand_toward = self.expand_toward_pinv

    def step(self, x: np.ndarray, u: np.ndarray, dt: float)->np.ndarray:
        raise NotImplementedError()

    def goal_check(self, x:np.ndarray):#->tuple[bool, float]:
        raise NotImplementedError()
    
    def sample(self, **kwargs)->np.ndarray:
        raise NotImplementedError()

    def expand_toward(self, x_near:np.ndarray, x_rand:np.ndarray, dt:float):#->tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    def linearize_at(self, x:np.ndarray, u:np.ndarray, dt:float):#->tuple[np.ndarray,  # A, shape (x_dim, x_dim)
                                                                       # np.ndarray,  # B, shape (x_dim, u_dim)
                                                                        #np.ndarray]: # c, shape (x_dim, )
        raise NotImplementedError()
    
    def get_reachable_sampled(self, x:np.ndarray, dt:float):#->tuple[np.ndarray, np.ndarray]:
        # returns a set of sampled points associated with the respective inputs
        # these points are generatet by stepping in the dynamics starting from x and applying
        # a predefined discrete set of inputs
        raise NotImplementedError()
    
    def get_reachable_AH(self, x:np.ndarray, dt:float, convex_hull:bool=False):#->list[tuple[np.ndarray, pp.AH_polytope]]:
        # get reachable set approximation using AH-polytopes
        # returns the a keypoint and a polytope for each reachable polytope
        raise NotImplementedError()


    def expand_toward_pinv(self, x_near:np.ndarray, x_rand:np.ndarray, dt:float):#->tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    def expand_toward_samples(self, x_near: np.ndarray, x_rand: np.ndarray, dt: float):# -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    def ffw(self, x: np.ndarray):#->list[np.ndarray]:
        # for hybrid systems especially it can be useful to skip phases in which you
        # do not have inputs available to avoid cluttering the sample space with
        # nodes that cannot be further expanded
        return np.array([]).reshape(-1,self.x_dim), np.array([]).reshape(-1,self.u_dim)
    
    def calc_input(self, frm: np.ndarray, to: np.ndarray, dt:float):#->tuple[np.ndarray, np.ndarray]:
        # hybrid systems override this
        return self.expand_toward_pinv(x_near=frm, x_rand=to, dt=dt)