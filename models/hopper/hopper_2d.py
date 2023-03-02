import numpy as np
import pypolycontain as pp
from utils import convex_hull_of_point_and_polytope
from models.model import Model
import sympy

class Hopper2D(Model):
    FLIGHT_ASCEND = 0
    FLIGHT_DESCEND = 1
    CONTACT_ASCEND = 3
    CONTACT_DESCEND = 2

    modes = {FLIGHT_ASCEND, FLIGHT_DESCEND, CONTACT_ASCEND, CONTACT_DESCEND}

    def __init__(self, 
                 m=5, J=500, m_l=1, J_l=0.5, l1=0.0, l2=0.0, k_g=2e3, b_g=20, g=9.8,
                 initial_state=np.array([0., 1., 0, 0, 1.5, 0, 0., 0., 0., 0.]), 
                 input_limits=np.array([[-500,500], [1.4e3,2e3]]), 
                 goal_states = [np.array([10,1.,0.,0.,1.5,0,0.,0.,0.,0., 0.])],
                 dt=0.005,
                 fast_forward=True):
        super().__init__(initial_state, input_limits, dt)

        self.x_dim = 10
        self.u_dim = 2

        self.goal_states = goal_states
        self.fast_forward = fast_forward

        self.m = m
        self.J = J
        self.m_l = m_l
        self.J_l = J_l
        self.l1 = l1
        self.l2 = l2
        self.k_g_y = k_g
        self.k_g_x = 2e3
        self.b_g_x = 200
        self.b_g = b_g
        self.g = g
        self.r0 = 1.5

        self.body_attitude_limit = np.pi/3-1e-2
        self.leg_attitude_limit = np.pi/3

        self.k0 = 800
        self.b_leg = 2
        self.k0_stabilize = 40
        self.b0_stabilize = 10
        self.k0_restore = 60
        self.b0_restore = 15

        self.b_r_ascend = 0.
        self.tau_p = 400.
        self.tau_d = 10.

        self.x_ = sympy.MatrixSymbol('x',self.x_dim,1)
        self.u_ = sympy.MatrixSymbol('u',self.u_dim, 1)

        self.flight_ascend_dynamics =   sympy.lambdify(self.x_, self._flight_ascend_dynamics(self.x_))
        self.flight_descend_dynamics =  sympy.lambdify(self.x_, self._flight_descend_dynamics(self.x_))
        self.contact_ascend_dynamics =  sympy.lambdify([self.x_, self.u_], self._contact_ascend_dynamics(self.x_,  self.u_))
        self.contact_descend_dynamics = sympy.lambdify([self.x_, self.u_],self._contact_descend_dynamics(self.x_, self.u_))

        self.flight_ascend_J = {
            "A": sympy.lambdify([self.x_, self.u_],self._flight_ascend_dynamics(self.x_).jacobian(self.x_)),
            "B": sympy.lambdify([self.x_, self.u_],self._flight_ascend_dynamics(self.x_).jacobian(self.u_))
        }
        self.flight_descend_J = {
            "A": sympy.lambdify([self.x_, self.u_],self._flight_descend_dynamics(self.x_).jacobian(self.x_)),
            "B": sympy.lambdify([self.x_, self.u_],self._flight_descend_dynamics(self.x_).jacobian(self.u_))
        }
        self.contact_ascend_J = {
            "A": sympy.lambdify([self.x_, self.u_],self._contact_ascend_dynamics(self.x_,  self.u_).jacobian(self.x_)),
            "B": sympy.lambdify([self.x_, self.u_],self._contact_ascend_dynamics(self.x_,  self.u_).jacobian(self.u_))
        }
        self.contact_descend_J = {
            "A": sympy.lambdify([self.x_, self.u_],self._contact_descend_dynamics(self.x_, self.u_).jacobian(self.x_)),
            "B": sympy.lambdify([self.x_, self.u_],self._contact_descend_dynamics(self.x_, self.u_).jacobian(self.u_))
        }
        

    def get_ddots(self, x, Fx, Fy, F_leg, u0):
        R = x[4]-self.l1

        alpha = (self.l1*Fy*sympy.sin(x[2])-self.l1*Fx*sympy.cos(x[2])-u0)
        A = sympy.cos(x[2])*alpha-R*(Fx-F_leg*sympy.sin(x[2])-self.m_l*self.l1*x[7]**2*sympy.sin(x[2]))
        B = sympy.sin(x[2])*alpha+R*(self.m_l*self.l1*x[7]**2*sympy.cos(x[2])+Fy-F_leg*sympy.cos(x[2])-self.m_l*self.g)
        C = sympy.cos(x[2])*alpha+R*F_leg*sympy.sin(x[2])+self.m*R*(x[4]*x[7]**2*sympy.sin(x[2])+self.l2*x[8]**2*sympy.sin(x[3])-2*x[9]*x[7]*sympy.cos(x[2]))
        D = sympy.sin(x[2])*alpha-R*(F_leg*sympy.cos(x[2])-self.m*self.g)-self.m*R*(2*x[9]*x[7]*sympy.sin(x[2])+x[4]*x[7]**2*sympy.cos(x[2])+self.l2*x[8]**2*sympy.cos(x[3]))
        E = self.l2*sympy.cos(x[2]-x[3])*alpha-R*(self.l2*F_leg*sympy.sin(x[3]-x[2])+u0)

        
        a1 = -self.m_l*R
        a2 = (self.J_l-self.m_l*R*self.l1)*sympy.cos(x[2])
        b1 = self.m_l*R
        b2 = (self.J_l -self.m_l*R*self.l1)*sympy.sin(x[2])
        c1 = self.m*R
        c2 = (self.J_l+self.m*R*x[4])*sympy.cos(x[2])
        c3 = self.m*R*self.l2*sympy.cos(x[3])
        c4 = self.m*R*sympy.sin(x[2])
        d1 = -self.m*R
        d2 = (self.J_l+self.m*R*x[4])*sympy.sin(x[2])
        d3 = self.m*R*self.l2*sympy.sin(x[3])
        d4 = -self.m*R*sympy.cos(x[2])
        e1 = self.J_l*self.l2*sympy.cos(x[2]-x[3])
        e2 = -self.J*R

        return np.array([(A*b1*c2*d4*e2 - A*b1*c3*d4*e1 - A*b1*c4*d2*e2 + A*b1*c4*d3*e1 + A*b2*c4*d1*e2 - B*a2*c4*d1*e2 - C*a2*b1*d4*e2 + D*a2*b1*c4*e2 + E*a2*b1*c3*d4 - E*a2*b1*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                            (A*b2*c1*d4*e2 + B*a1*c2*d4*e2 - B*a1*c3*d4*e1 - B*a1*c4*d2*e2 + B*a1*c4*d3*e1 - B*a2*c1*d4*e2 - C*a1*b2*d4*e2 + D*a1*b2*c4*e2 + E*a1*b2*c3*d4 - E*a1*b2*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                            -(A*b1*c1*d4*e2 - B*a1*c4*d1*e2 - C*a1*b1*d4*e2 + D*a1*b1*c4*e2 + E*a1*b1*c3*d4 - E*a1*b1*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                            (A*b1*c1*d4*e1 - B*a1*c4*d1*e1 - C*a1*b1*d4*e1 + D*a1*b1*c4*e1 + E*a1*b1*c2*d4 - E*a1*b1*c4*d2 + E*a1*b2*c4*d1 - E*a2*b1*c1*d4)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                            (A*b1*c1*d2*e2 - A*b1*c1*d3*e1 - A*b2*c1*d1*e2 - B*a1*c2*d1*e2 + B*a1*c3*d1*e1 + B*a2*c1*d1*e2 - C*a1*b1*d2*e2 + C*a1*b1*d3*e1 + C*a1*b2*d1*e2 + D*a1*b1*c2*e2 - D*a1*b1*c3*e1 - D*a2*b1*c1*e2 - E*a1*b1*c2*d3 + E*a1*b1*c3*d2 - E*a1*b2*c3*d1 + E*a2*b1*c1*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2)])

    def check_flight_ascend(self, x):
        hip_y_dot = x[6]+x[9]*np.cos(x[2])-x[4]*np.sin(x[2])*x[7]
        return x[1] > 0 and hip_y_dot>0
    def check_flight_descend(self, x):
        hip_y_dot = x[6]+x[9]*np.cos(x[2])-x[4]*np.sin(x[2])*x[7]
        return x[1] > 0 and hip_y_dot<=0
    def check_contact_descend(self, x):
        return x[1] <= 0 and x[9] < 0
    def check_contact_ascend(self, x):
        return x[1] <= 0 and x[9] >= 0

    def get_mode(self, x):

        if self.check_flight_ascend(x): return (self.FLIGHT_ASCEND)
        if self.check_flight_descend(x): return (self.FLIGHT_DESCEND)
        if self.check_contact_ascend(x): return (self.CONTACT_ASCEND)
        if self.check_contact_descend(x): return (self.CONTACT_DESCEND)

        raise Exception()
    

    def _flight_ascend_dynamics(self,x):
        r_diff = x[4]-self.r0
        F_leg_flight = -self.k0_restore*r_diff-self.b0_restore*x[9]
        hip_x_dot = x[5]+x[9]*sympy.sin(x[2])+x[4]*sympy.cos(x[2])*x[7]
        hip_y_dot = x[6]+x[9]*sympy.cos(x[2])-x[4]*sympy.sin(x[2])*x[7]
        alpha_des_ascend = 0.6*sympy.atan(hip_x_dot/(-hip_y_dot-1e-6))

        tau_leg_flight_ascend = (self.tau_p*(alpha_des_ascend-x[2])-self.tau_d*x[7])*-1

        dx = sympy.Matrix([*x[5:], *self.get_ddots(x,0,0, F_leg_flight, tau_leg_flight_ascend)])

        return dx
    def _flight_descend_dynamics(self, x):
        r_diff = x[4]-self.r0
        F_leg_flight = -self.k0_restore*r_diff-self.b0_restore*x[9]
        hip_x_dot = x[5]+x[9]*sympy.sin(x[2])+x[4]*sympy.cos(x[2])*x[7]
        hip_y_dot = x[6]+x[9]*sympy.cos(x[2])-x[4]*sympy.sin(x[2])*x[7]
        alpha_des_descend = 0.6*sympy.atan(hip_x_dot/(hip_y_dot+1e-6)) # point toward landing point
        tau_leg_flight_descend = (self.tau_p*(alpha_des_descend-x[2])-self.tau_d*x[7])*-1
        
        dx = sympy.Matrix([*x[5:],*self.get_ddots(x, 0, 0, F_leg_flight, tau_leg_flight_descend) ])

        return dx
    def _contact_ascend_dynamics(self, x, u):
        r_diff = x[4]-self.r0
        Fx_contact = -self.b_g_x*x[5]
        Fy_contact = -self.k_g_y*(x[1])-self.b_g*x[6]*(1-sympy.exp(x[1]*16))
        F_leg_ascend = -u[1]*r_diff - self.b_r_ascend * x[9]
        tau_leg_contact = u[0]

        dx = sympy.Matrix([*x[5:],*self.get_ddots(x,Fx_contact, Fy_contact, F_leg_ascend, tau_leg_contact)])

        return dx
    def _contact_descend_dynamics(self, x, u):
        r_diff = x[4]-self.r0
        Fx_contact = -self.b_g_x*x[5]
        Fy_contact = -self.k_g_y*(x[1])-self.b_g*x[6]*(1-sympy.exp(x[1]*16))
        F_leg_descend = -self.k0*r_diff-self.b_leg*x[9]
        tau_leg_contact = u[0]

        dx = sympy.Matrix([*x[5:],*self.get_ddots(x,Fx_contact, Fy_contact, F_leg_descend, tau_leg_contact)])
        return dx


    def step(self, x: np.ndarray, u: np.ndarray, dt: float) :#-> np.ndarray:
        start_mode = self.get_mode(x)

        x_ = x.reshape(self.x_dim, 1)
        if u is not None:
            u_ = u.reshape(self.u_dim, 1)

        if start_mode == self.FLIGHT_ASCEND:
            dx = self.flight_ascend_dynamics(x_).reshape(self.x_dim)
        elif start_mode == self.FLIGHT_DESCEND:
            dx = self.flight_descend_dynamics(x_).reshape(self.x_dim)
        elif start_mode == self.CONTACT_ASCEND:
            dx = self.contact_ascend_dynamics(x_,u_).reshape(self.x_dim)
        elif start_mode == self.CONTACT_DESCEND:
            dx = self.contact_descend_dynamics(x_,u_).reshape(self.x_dim)
        else:
            raise Exception()
        #print("dx ours", dx*dt)
        x_next = x + dx*dt

        return x_next
    
    def linearize_at(self, x: np.ndarray, u: np.ndarray, dt: float, mode=None) :#-> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mode == None:
            mode = self.get_mode(x)

        x_ = x.reshape(self.x_dim, 1)
        u_ = u.reshape(self.u_dim, 1)
        if mode == self.FLIGHT_ASCEND:
            A = (np.eye(x_.shape[0]) + dt*self.flight_ascend_J["A"](x_,u_))
            B = dt*self.flight_ascend_J["B"](x_,u_)
            f_bar = self.flight_ascend_dynamics(x_).reshape(self.x_dim)
            c = dt*(f_bar - A@x - B@u)
        elif mode == self.FLIGHT_DESCEND:
            A = (np.eye(x_.shape[0]) + dt*self.flight_descend_J["A"](x_,u_))
            B = dt*self.flight_descend_J["B"](x_,u_)
            f_bar = self.flight_descend_dynamics(x_).reshape(self.x_dim)
            c = dt*(f_bar - A@x - B@u)
        elif mode == self.CONTACT_ASCEND:
            A = (np.eye(x_.shape[0]) + dt*self.contact_ascend_J["A"](x_,u_))
            B = dt*self.contact_ascend_J["B"](x_,u_)
            f_bar = self.contact_ascend_dynamics(x_,u_).reshape(self.x_dim)
            c = dt*(f_bar - A@x - B@u)
        elif mode == self.CONTACT_DESCEND:
            A = (np.eye(x_.shape[0]) + dt*self.contact_descend_J["A"](x_,u_))
            B = dt*self.contact_descend_J["B"](x_,u_)
            f_bar = self.contact_descend_dynamics(x_,u_).reshape(self.x_dim)
            c = dt*(f_bar - A@x - B@u)
        else:
            return None    
        
        assert A.shape == (self.x_dim, self.x_dim)
        assert B.shape == (self.x_dim, self.u_dim)
        assert c.shape == (self.x_dim, )

        return A,B,c

    def goal_check(self, x: np.ndarray) :#-> tuple[bool, float]:
        for goal_state in self.goal_states:
            delta = x[0] - goal_state[0]
            if delta > 0:
                return True, np.abs(delta)
            
        return False, np.abs(delta)
    
    def ffw(self, x: np.ndarray) :#-> list[np.ndarray]:
        if self.fast_forward:
                    
            no_inputs = []
            x_ = x.copy()
            mode = self.get_mode(x_)
            while mode == self.FLIGHT_ASCEND or mode == self.FLIGHT_DESCEND:
                x_ = self.step(x_,None,self.dt)
                no_inputs.append(x_)
                mode = self.get_mode(x_)

            return np.array(no_inputs).reshape(-1,self.x_dim), np.zeros((len(no_inputs), self.u_dim))

        else:
            return super().ffw(x)
        
    def get_reachable_AH(self, x: np.ndarray, dt: float, convex_hull: bool = False) :#-> list[tuple[np.ndarray, pp.AH_polytope]]:
        A, B, c = self.linearize_at(x, self.u_bar, dt)
        x_next = (A@x + B@self.u_bar + c)
        G = (B@np.diag(self.u_diff)).reshape(self.x_dim, self.u_dim)
        AH = pp.to_AH_polytope(pp.zonotope(G,x_next.reshape(-1,1)))
        if convex_hull:
            AH = convex_hull_of_point_and_polytope(x.reshape(-1,1), AH)
        return [(x_next, AH)]
    
    def expand_toward_pinv(self, x_near: np.ndarray, x_rand: np.ndarray, dt: float) :#-> tuple[np.ndarray, np.ndarray]:
        # expand using pseudoinverse on linearized system
        A, B, c = self.linearize_at(x_near, self.u_bar, dt)

        u = np.linalg.pinv(B)@(x_rand - A@x_near - c)

        if u[0] < self.input_limits[0][0]: u[0] = self.input_limits[0][0]
        if u[0] > self.input_limits[0][1]: u[0] = self.input_limits[0][1]
        if u[1] < self.input_limits[1][0]: u[1] = self.input_limits[1][0]
        if u[1] > self.input_limits[1][1]: u[1] = self.input_limits[1][1]

        # the state has to be actually reachable so I step on the real environment with the systems's dt
        iters = int(dt//self.dt)

        states = np.zeros((iters,self.x_dim))
        controls = np.zeros((iters, self.u_dim))
        x = x_near.copy()
        for i in range(iters):
            x = self.step(x, u, self.dt)
            states[i] = x
            controls[i] = u
        
        return states, controls
    

    def sample(self, **kwargs) :#-> np.ndarray:
        if np.random.rand(1)<0.5:
            return self.hip_coordinates_sampler()
        else:
            return self.contact_sampler()
        
    def contact_sampler(self):
        rnd = np.random.rand(self.x_dim)
        
        rnd[0] = rnd[0]*10-0.5
        rnd[1] = np.random.normal(-0.5,0.2)
        rnd[2] = np.random.normal(0, np.pi/4) #np.random.normal(0, np.pi/6)
        rnd[3] = np.random.normal(0, np.pi/6) #np.random.normal(0, np.pi/16)
        rnd[4] = (rnd[4]-0.5)*2*4+5
        rnd[5] = (rnd[5]-0.5)*2*3
        rnd[6] = (rnd[5]-0.5)*2*6#np.random.normal(0, 6)
        rnd[7] = np.random.normal(0, 30)
        rnd[8] = np.random.normal(0, 2)
        rnd[9] = (rnd[9] - 0.1) * 2 * 20
        # goal_bias = np.random.rand(1)
        return rnd
    def hip_coordinates_sampler(self):
        # [x_hip, y_hip, theta(leg), phi(body), r]
        # if np.random.rand(1)<0.5:
        #     return uniform_sampler()
        rnd = np.random.rand(self.x_dim)
        rnd[0] = rnd[0] * 15 
        rnd[1] = (rnd[1] - 0.5) * 2 * 0.75 + 1.5
        rnd[2] = np.random.normal(0, np.pi / 4) # (np.random.rand(1)-0.5)*2*np.pi/12
        rnd[3] = np.random.normal(0, np.pi / 8)#np.random.normal(0, np.pi / 16)
        rnd[4] = (rnd[4] - 0.5) * 2 * 0.5 + 2
        rnd[5] = np.random.normal(1.5, 3) #(rnd[5] - 0.5) * 2 * 6
        rnd[6] = (rnd[6] - 0.5) * 2 * 12 # np.random.normal(0, 6)
        rnd[7] = np.random.normal(0, 20) # (np.random.rand(1)-0.5)*2*20
        rnd[8] = np.random.normal(0, 3) # (np.random.rand(1)-0.5)*2*5
        rnd[9] = (rnd[9] - 0.5) * 2 * 10 + 3 #np.random.normal(2, 12)
        
        # convert to hopper foot coordinates
        rnd_ft = np.zeros(self.x_dim)
        rnd_ft[0] = rnd[0]-np.sin(rnd[2])*rnd[4]
        rnd_ft[1] = rnd[0]-np.cos(rnd[2])*rnd[4]
        # if np.random.rand(1)<0.7:
        #     rnd_ft[1]=(rnd[1]/2-0.2)
        rnd_ft[5] = rnd[5]-rnd[9]*np.sin(rnd[2])-rnd[4]*np.cos(rnd[2])*rnd[7]
        rnd_ft[6] = rnd[6] - rnd[9] * np.cos(rnd[2]) + rnd[4] * np.sin(rnd[2]) * rnd[7]
        if rnd_ft[1]<=0:
            rnd_ft[2]=rnd[2]*2
            rnd_ft[7]=rnd[7]*5
        else:
            rnd_ft[2] = rnd[2]
            rnd_ft[7] = rnd[7]
        rnd_ft[3:5] = rnd[3:5]
        rnd_ft[8:] = rnd[8:]
        
        return rnd_ft