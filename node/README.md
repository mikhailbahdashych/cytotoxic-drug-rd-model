Getting to know the issues and applications of NODE neural differential equations.

Preparation for exercise.

1. Generate reference trajectories ("real" data) from your own PDE model (project) or an accurate base model,
2. Select a time period and time grid (e.g. evenly spaced points).

- Generate N trajectories: different initial conditions or different parameter values, 
- So you get the data: x_i(t_j), i = 1, ..., N, j = 1, ..., M
- (Optional) Add measurement noise to simulate measurement inaccuracy.
- Divide the data into: train (e.g. 80-90% of the trajectory), test (e.g. 10-20%),

3. Learn about NODE neural ordinary differential equations.
4. Build the Neural ODE model: dx/dt = f(x, t, theta), where f(...) is a neural network with theta weights.

5. Design different network architectures, e.g.:
- A: MLP "small" (fewer layers / narrow layers),
- B: MLP "large" (more layers / wide layers),
- or e.g. MLP vs residual architecture (ResNet). You can change:
    - number of layers,
    - number of neurons,
    - kind of non-linearity.

6. Get to know the libraries for training NODE, e.g. GitHub - caidao22/pnode: A Python library for training neural ODEs. - https://github.com/caidao22/pnode, link to readme - https://raw.githubusercontent.com/caidao22/pnode/refs/heads/main/README.md