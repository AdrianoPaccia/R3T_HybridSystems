from models import Model
from rtree import index
import numpy as np
import pypolycontain as pp
import utils
import time
import matplotlib.pyplot as plt

class Node:
    def __init__(self, states:np.ndarray, # an array of dimension   (n, dim_x)
                       controls:np.ndarray, # an array of dimension (n, dim_u)
                       parent=None, 
                       cost=0.,
                       dt = None):

        self.states = states
        self.controls = controls

        self.parent:Node = parent

        self.cost = cost
        self.dt = dt

        self.children = set()

    @property
    def state(self):
        return self.states[-1,:]

    def cumulative_cost(self):
        cost = self.cost
        if self.parent is None:
            return 0
        return cost + self.parent.cumulative_cost()
        
    def add_child(self, child):
        if child in self.children:
            return False
        else:
            self.children.add(child)
            return True

    def __hash__(self) :#-> int:
        return hash(str(np.hstack((self.states.flatten(), self.controls.flatten()))))
    
    def __eq__(self, __o: object) :#-> bool:
        return self.__hash__() == __o.__hash__()

    def __repr__(self) :#-> str:
        return f"[{self.state}, {self.controls[-1]}]"
    
class Planner:
    def __init__(self, model: Model, tau, thr=1e-9, ax=None):

        self.model = model
        self.x_dim = model.x_dim
        self.u_dim = model.u_dim
        self.state_tree = StateTree(self.x_dim)
        self.tau = tau
        self.thr = thr # threshold to alias nodes

        self.min_distance = np.inf

        self.n_nodes = 0

        self.id_to_node: dict[int, Node] = {}
        self.ax = ax
                    
    def nodes(self):
        return self.id_to_node.values()
    
    def get_plan(self, node):
        nodes = [node]

        while node.parent is not None:
            nodes = [node.parent] + nodes
            node = node.parent
            assert node.states.shape[0] == node.controls.shape[0]
        return nodes
    
    def add_node(self, states, controls = None, cost=None, parent:Node=None):#->Node:
        raise NotImplementedError()
    def expand(self, x_rand: np.ndarray):#-> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def plan(self, max_nodes):
        # add the first node with the initial state
        initial_state = self.model.initial_state

        self.initial_node = self.add_node(initial_state)

        goal, distance = self.model.goal_check(self.model.initial_state)
        if distance < self.min_distance:
            self.min_distance = distance
        if goal:
            plan = self.get_plan(self.initial_node)
            print()
            return True, plan
        
        it = 0
        dropped = 0
        start = time.time()
        while self.n_nodes < max_nodes:
            t = time.time()
            if it%1 == 0:
                print(f"n_nodes: {self.n_nodes}, dist: {self.min_distance}, dropped: {dropped}, t: {t-start} sec", end='\r')
            it+=1
            x_rand = self.model.sample()

            node_next, node_near = self.expand(x_rand)

            if node_next is None:
                dropped+=1
                continue

            x_next = node_next.state

            if self.ax != None: # debug
                x_near = node_near.state
                try: x_rand_plot.remove()
                except: pass
                try: x_next_plot.remove()
                except: pass
                try: x_near_plot.remove()
                except: pass
                x_rand_plot = self.ax.scatter(x_rand[0], x_rand[1], marker="x", color="green")
                x_near_plot = self.ax.scatter(x_near[0], x_near[1], color="purple")
                x_next_plot = self.ax.scatter(x_next[0], x_next[1], color="cyan")
                self.ax.plot([x_near[0], x_next[0]], [x_near[1],x_next[1]], color="blue")
                
                plt.draw()
                plt.pause(0.01)
                input()

                self.ax.scatter(x_near[0], x_near[1], color="blue")
                self.ax.scatter(x_next[0], x_next[1], color="blue")
               
            goal, plan = self.goal_check(node_next)
            if goal:
                return goal,plan

        print()
        return False, None

    def goal_check(self, node):
        for i in range(node.states.shape[0]):
            state = node.states[i,:]
            goal, distance = self.model.goal_check(state)
            if distance < self.min_distance:
                self.min_distance = distance
            if goal:
                node.states = node.states[:i+1,:]
                node.controls = node.controls[:i+1,:]
                plan = self.get_plan(node)
                return True, plan
        return False, None
        


class StateTree:
    def __init__(self, dim):
        self.dim = dim

        # set properties for nearest neighbor
        self.state_tree_p = index.Property()
        self.state_tree_p.dimension = dim
        
        self.state_idx = index.Index(properties=self.state_tree_p,
                                     interleaved=True)

    def insert(self, state):
        state_id = self._id(state)
        self.state_idx.insert(state_id, state)

        return state_id

    def nearest(self, state):
        nearest_id = list(self.state_idx.nearest(state, num_results=1))[0]

        return nearest_id
    
    def ids_in_box(self, box:utils.AABB):
        lu = np.concatenate((box.l, box.u))

        intersections = set(self.state_idx.intersection(lu))

        return intersections

    def _id(self,x:np.ndarray):
        return hash(str(x))
    

class PolytopeTree:
    # a polytope tree is a data structure which is
    # optimized for closest polytope to point queries
    # the implementation follows the AABB-query algorithm
    # in: "The Nearest Polytope Problem: Algorithms and Application to Controlling Hybrid Systems"

    def __init__(self, dim: int):

        self.dim = dim

        self.aabb_tree = AABBTree(dim)
        self.kpoint_tree = StateTree(dim)

        self.kp_id_to_polytope = {}
        self.bbox_id_to_polytope = {}

        self.kp_id_to_bbox_id = {}

    def insert(self, AH: pp.AH_polytope, kpoint: np.ndarray):
                                   # one keypoint per polytope
        kp_id = self.kpoint_tree.insert(kpoint)
        self.kp_id_to_polytope[kp_id] = AH


        bbox = utils.AABB.from_AH(AH)
        bbox_id = self.aabb_tree.insert(bbox)
        self.bbox_id_to_polytope[bbox_id] = AH

        self.kp_id_to_bbox_id[kp_id] = bbox_id

    def nearest_polytope(self, query: np.ndarray):#->pp.AH_polytope:
        
        nearest_kp_id = self.kpoint_tree.nearest(query)

        polytope_star = self.kp_id_to_polytope[nearest_kp_id]

        delta = utils.distance_point_polytope(query, polytope_star)
        d_star = np.linalg.norm(delta)
        p_star = query + delta

        if d_star <= 1e-6:
            return polytope_star, p_star, d_star

        abs_delta = np.abs(delta)

        # a little tighter than using the same distance for every dimension
        heuristic_box = utils.AABB(query-abs_delta, query+abs_delta)

        intersecting_boxes = self.aabb_tree.intersection(heuristic_box)

        bbox_id = self.kp_id_to_bbox_id[nearest_kp_id]
        already_computed = {bbox_id, } # do not re-call the solvers on the same polytope
        while len(intersecting_boxes) > 0:
            # candidate_id
            bbox_id = intersecting_boxes.pop()

            if bbox_id in already_computed:
                continue
            else:
                polytope = self.bbox_id_to_polytope[bbox_id]
                delta = utils.distance_point_polytope(query, polytope)

                already_computed.add(bbox_id)
                
                d = np.linalg.norm(delta)
                if d < d_star:
                    d_star = d
                    polytope_star = polytope
                    p_star = query + delta
                    abs_delta = np.abs(delta)
                    heuristic_box = utils.AABB(query-abs_delta, query+abs_delta)
                    intersecting_boxes = self.aabb_tree.intersection(heuristic_box)


        return polytope_star, p_star, d_star



class AABBTree:

    def __init__(self, dim):

        self.AABB_tree_p = index.Property()
        self.AABB_tree_p.dimension = dim

        self.AABB_idx = index.Index(properties=self.AABB_tree_p,
                                    interleaved=True)

    def insert(self, bbox: utils.AABB):

        lu = np.concatenate((bbox.l, bbox.u))

        bbox_id = hash(bbox)

        self.AABB_idx.insert(bbox_id, lu)

        return bbox_id

    def intersection(self, bbox: utils.AABB):
        
        lu = np.concatenate((bbox.l, bbox.u))

        intersections = set(self.AABB_idx.intersection(lu))

        return intersections