from pathos.pools import ProcessPool


def run_learning(env, agent):
    """
    run learning agent on some environment
    returns:
        results: {q_values: ..., policy: ..., event_listener_results: ...}
    """
    res = agent.train_on(env)
    return res


def run_multiprocessing_learning(env, agents):
    """
    speed up run_learning with multiprocessing
    each process runs a different agent that carries a different seed, but the environment is exactly the same
    """
    pool = ProcessPool(nodes=len(agents))
    results = pool.amap(
        run_learning,
        [env]*len(agents),
        [agents[i] for i in range(len(agents))],
    )
    while not results.ready():
        pass
    results = results.get()

    return results
