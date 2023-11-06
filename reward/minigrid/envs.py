from reward.minigrid.minigrid_wrappers import environment_builder as minigrid_env_builder
from reward.minigrid.d4rl_env import environment_builder as d4rl_env_builder


def environment_builder(
    level_name,
    **kwargs,
):
    if 'MiniGrid' in level_name:
        return minigrid_env_builder(
            level_name,
            **kwargs,
        )
    
    elif 'antmaze' in level_name:
        return d4rl_env_builder(
            level_name,
            **kwargs,
        )
        
    else:
        raise NotImplementedError('environment not supported')
            