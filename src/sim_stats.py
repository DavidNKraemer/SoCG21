from src.board import DistributedBoard


class SimStats:
    """
    Object that logs and stores statistics describing one simulation
    trajectory.

    Intended for use with the genetic algorithm approach via a callback
    function.

    Fields
    ------
    dist_trav: int
        Total distance travelled by team of bots.
    time: int
        Total time elapsed (board clock ticks).
    obs_hit: int
        Total number of objects hit by the team of bots.
    bot_collisions: int
        Total number of collisions between bots. When bot i collides with
        bot j, this is counted as one collision, not two.
    """

    def __init__(self):
        """
        Construct an initialized SimStats instance before simulating.
        """
        self.dist_trav = 0
        self.time = 0
        self.obs_hit = 0
        self.bot_collisions = 0
        self.error = 0.
        self.finished = False

    def compute_l1error(self, board):
        """
        Return the sum of L1 distances between bots' final positions and
        their targets.

        Parameter
        ---------
        board: DistributedBoard
        """
        # equivalent: self.error = sum(bot.dist_to_go() for bot in board.bots)
        aggregate_error = 0
        for bot in board.bots:
            aggregate_error += bot.dist_to_go
        self.error = aggregate_error
