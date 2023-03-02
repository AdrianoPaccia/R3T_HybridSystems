import numpy as np
from matplotlib.patches import Polygon,Rectangle
from scipy.spatial import ConvexHull
from lib.operations import AH_polytope_vertices 
import os
import pypolycontain as pp
import qpsolvers
import cv2
import matplotlib.pyplot as plt

def normalize(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def plot_hopper_2d(root_node, plt):
    states = np.array(root_node.states)[:,:2]
    state = root_node.state[:2]
    plt.scatter(states[:-1,0], states[:-1,1],c="green",s=2)
    plt.plot(states[:,0], states[:,1], linewidth=0.5,c="green")
    for child in root_node.children:
    
        child_state = child.states[0][:2]

        plt.plot([state[0], child_state[0]], [state[1], child_state[1]], linewidth=0.5, c="blue")
        plt.scatter(child_state[0], child_state[1],c="blue",s=2)

        plot_hopper_2d(child, plt)


def plot(nodes, ax,int_color='pink', last_color="red", size=5, lw=1, th=100, plot_all=True, zorder=0, convex_hull=False, polytopes=True):
    # speed up plots by plotting all at once
    from matplotlib import collections as mc


    lines = []
    lines_int = []

    scatters = []
    scatters_int = []

    for node in nodes:
        if polytopes and hasattr(node, "polytopes"):
            visualize_polytope_convexhull(node.polytopes[0], node.state, ax=ax, convex_hull=convex_hull)
        # always plot the last node
        scatters.append(node.state)

        # if it has a parent, add a line from this node's state 
        # to the parent's last
        if node.parent is not None:
            x_from = node.parent.state
            x_to   = node.states[0] if plot_all else node.state
            lines.append([x_from[:2],x_to[:2]])

        # if plot_all, also plot intermediate states and lines in between them
        n_int = len(node.states)
        if plot_all:
            for i in range(n_int-1): # I already plotted the last state in any case
                scatters_int.append(node.states[i,:2])
                lines_int.append([node.states[i,:2], node.states[i+1,:2]])
        
    scatters = np.array(scatters)
    lines = np.array(lines)
    lc = mc.LineCollection(lines, color=int_color,zorder=zorder+2,linewidth=lw)

    if plot_all and len(scatters_int) > 0 :
        scatters_int = np.array(scatters_int)
        ax.scatter(scatters_int[:,0], scatters_int[:,1], color=int_color, s=size*3/5, zorder=zorder+3)
        lines_int = np.array(lines_int)
        lc_int = mc.LineCollection(lines_int, color=int_color,zorder=zorder+1, linewidth=lw)
        ax.add_collection(lc_int)



    ax.add_collection(lc)
    ax.scatter(scatters[:,0], scatters[:,1], color=last_color, s=size, zorder=zorder+4)
    
def distance_point_polytope(query:np.ndarray, AH:pp.AH_polytope):
    n_dim = query.reshape(-1).shape[0]

    AH = pp.to_AH_polytope(AH)

    q_dim, m_dim = AH.P.H.shape

    P = np.zeros((n_dim+m_dim,n_dim+m_dim))
    P[:n_dim,:n_dim] = 0.5*np.eye(n_dim)

    q = np.zeros(n_dim+m_dim).reshape(-1)

    G = np.zeros((q_dim,n_dim+m_dim))
    G[:,n_dim:] = AH.P.H

    h = AH.P.h.reshape(-1)

    A = np.zeros((n_dim, n_dim+m_dim)) 
    A[:,:n_dim] = - np.eye(n_dim)
    A[:,n_dim:] = AH.T

    b = (query.reshape(-1) - AH.t.reshape(-1)).reshape(-1)

    solution = qpsolvers.solve_qp(P,q,G=G,h=h,A=A,b=b, solver="gurobi")
    try:
        delta = solution[:n_dim]
        return delta
    except TypeError:
        print(query)
        print(AH.t)
        print(AH.T)
        print("----")
        print(q)
        print(h)
        print(b)
        return None

class AABB: # axis aligned bounding box

    def __init__(self, l, u):
        assert l.shape == u.shape
    
        self.l = l
        self.u = u
    
    @staticmethod
    def from_AH(AH:pp.AH_polytope):
        
        n_dim = AH.n

        H = AH.P.H
        h = AH.P.h

        m_dim = AH.P.H.shape[1]

        G = AH.T
        g = AH.t

        U = np.zeros(n_dim)
        L = np.zeros(n_dim)

        for d in range(n_dim):
            # Gd: d-esima riga di G
            # dot(Gd,x) == qTx

            Gd = G[d,:]

            x = qpsolvers.solve_qp(P=np.zeros((m_dim,m_dim)), q =  Gd, G=H, h=h, solver="gurobi")
            L[d] = np.dot(Gd,x) + g[d] 

            x = qpsolvers.solve_qp(P=np.zeros((m_dim,m_dim)), q = -Gd, G=H, h=h, solver="gurobi")
            U[d] = np.dot(Gd,x) + g[d] 
        
        return AABB(L,U)
    
    def plot_AABB(self,plt,col, plot = None):
        if plot != None:
            plot.remove()
        anchor_point = self.l
        width = abs(self.l[0]-self.u[0])
        heigth = abs(self.l[1]-self.u[1])
        plot = plt.gca().add_patch(Rectangle(anchor_point,
                    width,heigth,
                    edgecolor = col,
                    fill=False,lw=1))
        return plot
        
def visualize_polytope_convexhull(polytope,state,color='blue',alpha=0.4,N=20,epsilon=0.001,ax=None, convex_hull=False):
    v,w=AH_polytope_vertices(polytope,N=N,epsilon=epsilon, solver="osqp")
    try:
        v=v[ConvexHull(v).vertices,:]
    except:
        v=v[ConvexHull(w).vertices,:]
    # x = v[0:2,:]
    if convex_hull:
        v = np.append(v,[state],axis=0)
    p=Polygon(v,edgecolor = color,facecolor = color,alpha = alpha,lw=1)
    ax.add_patch(p)
    return p

def convex_hull_of_point_and_polytope(x, Q):
    r"""
    Inputs:
        x: numpy n*1 array
        Q: AH-polytope in R^n
    Returns:
        AH-polytope representing convexhull(x,Q)
    
    .. math::
        \text{conv}(x,Q):=\{y | y= \lambda q + (1-\lambda) x, q \in Q\}.
    """
    Q=pp.to_AH_polytope(Q)
    q=Q.P.H.shape[1]

    new_T=np.hstack((Q.T,Q.t-x))
    new_t=x
    new_H_1=np.hstack((Q.P.H,-Q.P.h))
    new_H_2=np.zeros((2,q+1))
    new_H_2[0,q],new_H_2[1,q]=1,-1
    new_H=np.vstack((new_H_1,new_H_2))
    new_h=np.zeros((Q.P.h.shape[0]+2,1))
    new_h[Q.P.h.shape[0],0],new_h[Q.P.h.shape[0]+1,0]=1,0
    new_P=pp.H_polytope(new_H,new_h)
    return pp.AH_polytope(t=new_t,T=new_T,P=new_P)

def edit_video(path,N,dt, speed=1.0):

    img_array = []
    img_path = path + "/imgs/"
    for n in range(N):
        filename = img_path + str(n)+ '.png'
        img = cv2.imread(filename)
        
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    fps = int(1/dt)*speed
    out = cv2.VideoWriter(path+'/video.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    return out

def plot_plan(states, dt, video=False, goal_x=10, dir="./"):

    if video:
        os.mkdir(dir+"/imgs")
    

    alpha = 1.0 if video else 0.5
    step  = 1 if video else 16

    fig, ax = plt.subplots()
    ax.axvline(x = goal_x, color = 'g', label = 'goal')

    i = 0
    for state in states:
        X = state[:5]
        img_name = dir + "/imgs/" + str(i) +'.png'
        if i % step ==0:
            # plot
            hopper = hopper_plot(X,ax, xlim=[-2,17], ylim=[0,5], alpha=alpha)
            if video:
                fig.savefig(img_name)
                [x.remove() for x in hopper]

        i+= 1

    if video:
        edit_video(path=dir,N=i,dt=dt)
        plt.close()
    else:
        fig.savefig(dir+"/strobo.png")

    plt.close()
    return



import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

def hopper_plot(X,ax,scaling_factor=0.7, alpha=0.5, xlim=[0,5], ylim=[0,5]):
    x,y,theta,phi,r=X[0:5]
    # theta and phi are clockwise positive
    theta *= -1
    phi *= -1
    w_1=0.1*scaling_factor
    w_2=0.1*scaling_factor
    h=0.2*scaling_factor
    L=3*scaling_factor
    a=1*scaling_factor
    alpha = alpha
    R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    # Good now plot
    ax.set_xlabel("x [m]",fontsize=14)
    ax.set_ylabel("y [m]",fontsize=14)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_aspect('equal')
    # leg
    corner=np.array([x,y])
    down_left=np.array([x,y])+np.dot(R,[-w_1,h])
    down_right=np.array([x,y])+np.dot(R,[w_1,h])
    up_left=np.array([x,y])+np.dot(R,[-w_1,h])+np.dot(R,[-w_1/2,L])
    up_right=np.array([x,y])+np.dot(R,[w_1,h])+np.dot(R,[w_1/2,L])
    leg=[patches.Polygon(np.array([[corner[0],down_right[0],up_right[0],up_left[0],down_left[0]],\
                                   [corner[1],down_right[1],up_right[1],up_left[1],down_left[1]]]).reshape(2,5).T, True)]
    leg_ = ax.add_collection(PatchCollection(leg,color=(0.8,0.3,0.4),alpha=alpha,edgecolor=None))
    # Body
    center=np.array([x,y])+np.dot(R,[0,r*scaling_factor])
    R=np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    up_right,up_left,down_left,down_right=center+np.dot(R,[a,w_2]),\
                                            center+np.dot(R,[-a,w_2]),\
                                            center+np.dot(R,[-a,-w_2]),\
                                            center+np.dot(R,[a,-w_2])
    body=[patches.Polygon(np.array([[up_right[0],up_left[0],down_left[0],down_right[0]],\
                                    [up_right[1],up_left[1],down_left[1],down_right[1]]]).reshape(2,4).T, True)]
    body_ = ax.add_collection(PatchCollection(body,color=(0.2,0.2,0.8),alpha=alpha,edgecolor=None))
    ax.grid(color=(0,0,0), linestyle='--', linewidth=0.5)
    return leg_, body_