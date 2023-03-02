from algorithms.planner import Planner, Node
import numpy as np
from models.model import Model
import time

class RRT(Planner):
    
    def __init__(self, model: Model, tau, thr=1e-9, ax=None):
        super().__init__(model, tau, thr, ax)

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
            if not is_new:
                return None
        
        self.n_nodes += 1

        state_id = self.state_tree.insert(states[-1])
        self.id_to_node[state_id] = node
        return node
    
    def expand(self, x_rand: np.ndarray):

        id_near = self.state_tree.nearest(x_rand)
        node_near = self.id_to_node[id_near]

        x_near = node_near.state

        states, controls = self.model.expand_toward(x_near, x_rand, self.tau)

        # optimization for hybrid systems
        ffw = self.model.ffw(states[-1])

        states = np.vstack(( states , ffw[0] ))
        controls = np.vstack(( controls , ffw[1] ))

        if states is None:
            return None # cannot reach that state
        
        x_next: np.ndarray = states[-1]

        closest_idx = self.state_tree.nearest(x_next)
        closest_state = self.id_to_node[closest_idx].state


        if np.linalg.norm(x_next - closest_state) < self.thr:
            # there is already a node at this location
            # TODO consider rewiring if the cost is less       
            return None, None


        cost = np.sum( np.linalg.norm(controls, axis=1) )

        # add node to tree
        node_next = self.add_node(states, controls, cost, node_near)

        return node_next, node_near