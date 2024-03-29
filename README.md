# R3T_HybridSystems

Sampling-based motion planning algorithms such as PRM and RRT are commonly used to solve planning problems due to their ability to efficiently find solutions. Nevertheless, when applied to kinodynamic and hybrid systems, they typically perform poorly and they do not guarantee probabilistic completeness. Consequently, with my fellow collegues Fabio Scaparro and Patrizio Perugini, we implemented the R3T algorithm, a probabilistic complete and asymptotically optimal variant of RRT for kinodynamic planning of nonlinear hybrid systems, and compared it with RG-RRT and RRT considering different systems, such as the pendulum, the 1D hopper and the 2D hopper (in an obstacle-free environment).

Requirements are indicated in requirements.txt .

All theorethical and implementation details can be foud in the report at: https://drive.google.com/file/d/1oim0-c6Y9pFDmPO6cw81zYFFmn7ygfbd/view?usp=share_link.
