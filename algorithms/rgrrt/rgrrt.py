from algorithms.planner import Planner,StateTree, Node
import numpy as np
from models.model import Model


class RGRRT(Planner):
    def __init__(self, model: Model, tau, thr=1e-9, ax=None):
        super().__init__(model, tau, thr, ax)

        # we need an additional tree in which we store sampled reachable points
        self.reachable_tree = StateTree(self.x_dim)

        # and an additional map to link the reachable points to their nodes
        # (and the dynamic trajectory that generated them)
        self.r_id_to_node: dict[int, tuple[Node, np.ndarray, np.ndarray]] = {}


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


        # when you add a new node also remember
        # to compute their reachable points and put them into the tree
        x_r, u_r = self.model.get_reachable_sampled(states[-1],self.tau)

        for i in range(len(x_r)):
            # x_r represents the whole trajectory to get to the reachable state
            x_r_i = x_r[i][-1]

            x_r_i_id = self.reachable_tree.insert(x_r_i)
            self.r_id_to_node[x_r_i_id] = (node, x_r[i], u_r[i])

        return node


    
    def expand(self, x_rand):

        # get the nearest reachable point
        id_near = self.reachable_tree.nearest(x_rand)

        node_near, states, controls = self.r_id_to_node[id_near]
        x_near = node_near.state
        r_near = states[-1]

        # check if the expansion will be in the direction of x_rand
        # if not, discard
        if np.linalg.norm(x_rand-x_near) < np.linalg.norm(x_rand-r_near):
            return None, None


        # check for fast forward possibility
        ffw = self.model.ffw(states[-1])

        states = np.vstack(( states , ffw[0] ))
        controls = np.vstack(( controls , ffw[1] ))

        x_next: np.ndarray = states[-1]

        closest_idx = self.state_tree.nearest(x_next)
        closest_state = self.id_to_node[closest_idx].state

        if np.linalg.norm(x_next - closest_state) < self.thr:
            # there is already a node at this location
            # TODO consider rewiring if the cost is less
            return None, None
        
        cost = np.sum( np.linalg.norm(controls) )

        # add node to tree
        node_next = self.add_node(states, controls, cost, node_near)

        return node_next, node_near


