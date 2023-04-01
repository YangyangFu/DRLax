# DRLax
Implementation of DRL algorithms in JAX, and currently focus more on model-based implementation

# Thouthgts
- seems JAX cannot jit stateful program, which has to be converted to stateless program. -> https://jax.readthedocs.io/en/latest/jax-101/07-state.html
  - then seems no need for object-oriented implementation
  - 
# Algorithms

## Off-policy
- DDQN
- 

## On-policy
- REINFORCE
- 

## model-based 
- DYNA-Q

# Useful Links
- [CleanRL](https://docs.cleanrl.dev): seems have standalone implementation of different algorithms
- [EnvPool](https://github.com/sail-sg/envpool): a env wrapper that can jit the environment model and speed up the simulation. 2000 times faster than typical DRL training as shown [here](https://github.com/google/flax/tree/main/examples/ppo). 

# Contact


