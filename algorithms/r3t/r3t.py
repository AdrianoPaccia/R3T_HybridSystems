from algorithms.planner import Planner,StateTree,PolytopeTree,Node
import numpy as np
from models.model import Model
import utils
import matplotlib.pyplot as plt

class R3T(Planner):
    
    def __init__(self, model: Model, tau, thr=1e-7, convex_hull=True, ax=None, rewire=False):
        super().__init__(model, tau, thr, ax)

        self.convex_hull = convex_hull

        # we need a polytope tree to find the nearest polytope given a query point
        self.polytope_tree = PolytopeTree(model.x_dim)

        self.polytope_id_to_node = {}

        self.do_rewire = rewire

        # debug
        self.polytope_plot = None
        self.closest_plot = None
        
        
    def add_node(self, states, controls = None, cost=None, parent:Node=None):
        if len(states.shape) == 1:
            # if it's just a single state then reshape it to still be a collection of states
            states = states.reshape(1,-1)

        if controls is None:
            cost = 0
            controls = np.zeros((self.u_dim,))
        if len(controls.shape) == 1:
            controls = controls.reshape(1,-1)

        if cost is None:
            cost = np.sum( np.linalg.norm(controls, axis=1) ) # sum the cost of every control
        node = Node(states, controls, parent, cost, self.model.dt)

        # manage the parent's children
        if parent is not None:
            is_new = parent.add_child(node)
            # assert is_new
            if not is_new:
                return None
        self.n_nodes += 1

        state_id = self.state_tree.insert(states[-1])
        self.id_to_node[state_id] = node

        # we need to get polytopes and respective keypoints and put those in the tree
        reachables = self.model.get_reachable_AH(states[-1], self.tau, convex_hull=True)
        node.polytopes = []
        for reachable in reachables:
            kpoint = reachable[0]
            polytope = reachable[1]
            if self.ax != None:
                if self.polytope_plot != None:
                    pass#self.polytope_plot.remove()
                #self.polytope_plot = utils.visualize_polytope_convexhull(polytope, states[-1,:2], ax=self.ax)

            
            """
            p = utils.visualize_polytope_convexhull(polytope, states[-1], ax=ax)
            plt.draw()
            plt.pause(0.01)
            """

            self.polytope_tree.insert(polytope, kpoint)
            self.polytope_id_to_node[hash(polytope)] = node

            node.polytopes.append(polytope)

        return node
    
    def expand(self, x_rand):

        # get the nearest reachable point
        polytope, point, dist = self.polytope_tree.nearest_polytope(x_rand)

        if self.ax != None:
            if self.closest_plot != None:
                self.closest_plot.remove()
            self.closest_plot = self.ax.scatter(point[0],point[1], color="teal")

        polytope_id = hash(polytope)
        node_near = self.polytope_id_to_node[polytope_id]

        x_near = node_near.state

        states, controls = self.model.calc_input(frm=x_near, to=point, dt=self.tau)
        

        ffw = self.model.ffw(states[-1])

        states = np.vstack(( states , ffw[0] ))
        controls = np.vstack(( controls , ffw[1] ))

        x_next = states[-1]

        closest_idx = self.state_tree.nearest(x_next)
        closest_state = self.id_to_node[closest_idx].state


        if self.thr != None and np.linalg.norm(x_next - closest_state) < self.thr:
            # there is already a node at this location
            # TODO consider rewiring if the cost is less
            return None, None
        cost = np.sum( np.linalg.norm(controls, axis=1) )

        # add node to tree
        node_next = self.add_node(states, controls, cost, node_near)

        if self.do_rewire:
            self.rewire(node_next)

        return node_next, node_near
    
    def rewire(self, node: Node):
        # rewire 2
        for polytope in node.polytopes:
            # use the bounding box as an approximation for the polytope
            bbox = utils.AABB.from_AH(polytope)
            
            cand_ids = self.state_tree.ids_in_box(bbox)

            for cand_id in cand_ids:
                cand_node = self.id_to_node[cand_id]
                cand_state = cand_node.state

                deltas = utils.distance_point_polytope(cand_state, polytope)
                if np.linalg.norm(deltas) > 1e-9:
                    continue # the state is not actually contained

                states, controls = self.model.expand_toward_pinv(node.state, cand_state, self.tau)

                difference = states[-1]-cand_state
                if np.linalg.norm(difference) > 1e-2 and False:
                    print("--")
                    continue # did not get to cand_state


                # cost to go from node.state to cand_state
                cost = np.sum( np.linalg.norm( controls, axis=1) )

                if cand_node.cumulative_cost() > cost + node.cumulative_cost():
                    # then rewire
                    
                    # to have a consistent trajectory
                    #cand_node.states = states
                    #cand_node.controls = controls
                    cand_node.parent.children.remove(cand_node)
                    cand_node.parent = node
                    node.add_child(cand_node)

    """
    def goal_check(self, node: Node):
        state = node.states[-1]

        goal, states, controls, min_distance = self.model.expand_toward_samples(state, self.model.dt, n_samples=18)
        if min_distance < self.min_distance:
            self.min_distance = min_distance
        if goal:
            goal_node = Node(states, controls, node)
            plan = self.get_plan(goal_node)
            return True, plan
        return False, None
    """