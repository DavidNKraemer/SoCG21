class SimStats:
    """
    Object that logs and stores statistics describing one simulation
    trajectory.

    Intended for use with the genetic algorithm approach via a callback
    function.

    Fields
    ------
    dist_trav: int
        Total distance travelled by team of agents.
    time: int
        Total time elapsed (board clock ticks).
    obs_hit: int
        Total number of objects hit by the team of agents.
    agent_collisions: int
        Total number of collisions between agents. When agent i collides with
        agent j, this is counted as one collision, not two.
    """

    def __init__(self):
        """
        Construct an initialized SimStats instance before simulating.
        """
        self.dist_trav = 0
        self.time = 0
        self.obs_hit = 0
        self.agent_collisions = 0
