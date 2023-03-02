import numpy as np
import matplotlib.pyplot as plt

import utils, plot

from algorithms import RRT, RGRRT, R3T
from models import Pendulum, Hopper1D, Hopper2D

def test_calc_input(ax):
    p = Pendulum(dt=.001)
    tau=0.01
    x0 = np.array([1.05,-1])

    l = p.get_reachable_AH(x0, tau, convex_hull=False)
    kp = l[0][0]
    AH = l[0][1]

    ax.scatter(x0[0], x0[1], color="red")
    ax.scatter(kp[0],kp[1], color="yellow")
    utils.visualize_polytope_convexhull(AH, x0, ax=ax, convex_hull=False)

    x_rand = np.array([2,2])

    delta = utils.distance_point_polytope(AH=AH, query=x_rand)

    point = x_rand+delta

    ax.scatter(x_rand[0],x_rand[1],marker="x",color="red")
    ax.scatter(point[0],point[1],color="green")

    states, controls = p.calc_input(frm=x0, to=point, dt=tau)

    states = np.vstack((x0.reshape(1,-1),states))
    ax.plot(states[:,0],states[:,1],color="purple")

    error = (states[-1]-point)
    print(error)

    states, c = p.get_reachable_sampled(x0, tau)
    for state in states:
        ax.scatter(state[-1,0], state[-1,1],color="teal")

    plt.show()

def test_rrt_pendulum(seed=None, ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.001)

    planner = RRT(p, 0.2)

    goal, plan = planner.plan(max_nodes=1000, plt=None)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive", lw=3)
    print(seed)
    plt.show()

def test_rgrrt_pendulum(seed=None,ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.01)

    ax.scatter(p.initial_state[0],p.initial_state[1], color="purple")
    for goal_state in p.goal_states:
        ax.scatter(goal_state[0],goal_state[1], color="orange",marker="x")

    planner = RGRRT(p, 0.1)

    goal, plan = planner.plan(max_nodes=1000)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive")
    print(seed)
    plt.show()

def test_rgrrt_hopper_1d(seed=None,ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    h = Hopper1D(dt=0.01, eps_goal=0.05, fast_forward=True)

    planner = RGRRT(h, 0.04)

    goal, plan = planner.plan(max_nodes=700)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive")
    print(seed)
    plt.show()

def test_r3t_pendulum(seed=None,ax=None):
    # 554901
    # 202450
    # 603719
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.05, eps_goal=0.05)

    ax.scatter(p.initial_state[0],p.initial_state[1], color="purple")
    for goal_state in p.goal_states:
        ax.scatter(goal_state[0],goal_state[1], color="orange",marker="x",zorder=200)

    planner = R3T(p, 0.2, convex_hull=False, ax=None)

    goal, plan = planner.plan(max_nodes=800)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True, polytopes=False)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive", polytopes=False, zorder=199)
    print(seed)
    plt.show()

def test_r3t_hopper_1d(seed=None,ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Hopper1D(dt=0.01, eps_goal=0.05, fast_forward=True)

    planner = R3T(p, 0.04)

    goal, plan = planner.plan(max_nodes=1000)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True, polytopes=False)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive", polytopes=False)
    print(seed)
    plt.show()

def test_r3t_hopper_2d(seed=None,ax=None, animate=False):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Hopper2D(dt=0.005, fast_forward=True)

    planner = R3T(p, 0.1, thr=1e-9)
    goal, plan = planner.plan(max_nodes=1000)

    utils.plot(planner.nodes(), ax, plot_all=True, polytopes=False)
    print(goal)

    import pickle, os, random
    if plan is not None:
        out = {"plan":plan,
               "dt": p.dt,
               "tau": planner.tau,
               "goal": p.goal_states}
        
        dir = os.getcwd() 
        dir += '/trajectories/'+str(seed)

        if os.path.exists(dir):
            name_rnd = '_v'+str(random.randint(0, 100))
            dir += name_rnd

        dir +='/'
        os.mkdir(dir)
        with open(dir+"out.pickle", "wb") as out_file:
            pickle.dump(out, out_file)

    """
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True, polytopes=False)
        if plan is not None: 
            utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive", polytopes=False)
            if animate:
                states = []
                controls = []
                for node in plan:
                    for i in range(node.states.shape[0]):
                        states.append(node.states[i,:])
                        controls.append(node.u)
                        assert(node.states[i,:].shape == (10,))

            print(len(states), states[0].shape)
            utils.plot_plan(states, seed, save_video=True, goal_x=p.goal_states[0][0])
    """

    print(seed)
    plt.show()

def test(model_name, planner_name, tau, seed=None, max_nodes=1000):
    import os, random, pickle
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)

    if model_name == "pendulum":
        model = Pendulum()
    elif model_name == "hopper1d":
        model = Hopper1D()
    elif model_name == "hopper2d":
        model = Hopper2D()
    else:
        print("No model named ", model_name)
        return
    if planner_name == "RRT":
        planner = RRT(model, tau)
    elif planner_name == "RGRRT":
        planner = RGRRT(model, tau)
    elif planner_name == "R3T":
        planner = R3T(model, tau)
    else:
        print("no planner named ", planner_name)
        return

    goal, plan = planner.plan( max_nodes )

    out = {}
    out["model_name"]   = model_name
    out["planner_name"] = planner_name
    out["seed"]         = seed
    out["nodes"]        = list(planner.nodes())
    out["dt"]           = model.dt
    out["plan"]         = plan

    dir = os.getcwd() 
    filename = f"{model_name}_{planner_name}_{seed}.pickle"

    if os.path.exists(dir+"/"+filename):
        name_rnd = '_v'+str(random.randint(0, 100))
        filename += name_rnd

    with open(dir+"/"+filename, "wb") as out_file:
        pickle.dump(out, out_file)

    return dir+"/"+filename

# test("hopper1d", "R3T", 0.04)
pickle_name = test("pendulum", "R3T", 0.1, max_nodes=2000)

plot.plot(pickle_name)