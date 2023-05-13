from pathos.pools import ProcessPool
import torch
import random
import numpy as np


def run_learning(env, agent, agent_seed):
    """
    run learning agent on some environment
    returns:
        results: {q_values: ..., policy: ..., event_listener_results: ...}
    """
    assert agent_seed == agent.seed
    random.seed(agent_seed)
    np.random.seed(agent_seed)
    torch.manual_seed(agent_seed)
    res = agent.train_on(env)
    return res


def run_multiprocessing_learning(env, agents, agent_seeds):
    """
    speed up run_learning with multiprocessing
    each process runs a different agent that carries a different seed, but the environment is exactly the same
    """
    pool = ProcessPool(nodes=len(agents))
    results = pool.amap(
        run_learning,
        [env]*len(agents),
        [agents[i] for i in range(len(agents))],
        [agent_seeds[i] for i in range(len(agents))]
    )
    while not results.ready():
        pass
    results = results.get()

    return results
