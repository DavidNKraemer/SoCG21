import numpy as np

from itertools import product

def obstacles_hit(bot):
    """
    Return the number of obstacles with which the specified bot collided
    over the previous time-step.

    The output will either be 0 or 1, for an bot moves only one unit per
    time-step, and no two obstacles overlap (obstacles are merely pixels, not
    collections thereof). We chose to make this function integer-valued -- not
    boolean-valued -- to maintain continuity with bots_hit().
    """
    # obstacles don't move. Collision occurred in previous time-step iff
    # bot's position in present time-step is an obstacle pixel.
    return int(bot.position in bot.board.obstacles)


def bots_hit(bot):
    """
    Return the number of other bots with which the specified bot collided
    over the previous time-step.
    """
    # a collision occurred iff in bot's neighborhood, another bot a' moved
    # cardinal directions (i.e., from N to E, from S to N, from E to N, etc.)
    # TODO: import the dictionary from board.py
    N = np.array([0, 1])
    S = np.array([0, -1])
    W = np.array([-1, 0])
    E = np.array([1, 0])

    # number of other bots sharing the same pixel as bot
    n_collisions = len(bot.board.occupied_pixels[tuple(bot.position)]) - 1
    # DEBUG
    # print(f"Position: {bot.board.occupied_pixels[tuple(bot.position)]}")

    for direction, other_direction in product([N, S, W, E], [N, S, W, E]):
        if direction is other_direction:
            continue

        # DEBUG
        # print("direction: ", bot.board.prev_active_pixels[
        #    tuple(bot.prev_position + direction)])

        # store set of bot ids occupying this pixel at previous time-step
        prev_ids = bot.board.prev_occupied_pixels[
            tuple(bot.prev_position + direction)
        ]

        # DEBUG
        # print("other_direction: ", bot.board.prev_active_pixels[
        #    tuple(bot.position + other_direction)])

        # store set of bot ids occupying this pixel at current time-step
        curr_ids = bot.board.occupied_pixels[
            tuple(bot.position + other_direction)
        ]
        # take intersection of two sets, and increment by cardinality
        # print(f"Intersection: {curr_ids & prev_ids}")
        n_collisions += len(curr_ids & prev_ids)

    return n_collisions


def bot_reward(bot, dist_pen, obs_hit_pen, bots_hit_pen, finish_bonus):
    """
    Return the "instantaneous" reward signal for the specified bot.

    IMPORTANT
    ---------
    Higher *_pen values are associated with more penalty (i.e., more negative
    reward). Penalties are negated to convert them into rewards.

    Params
    ------
    bot: Bot
        Bot under consideration.
    dist_pen: float
        Penalty multiplier for distance-to-go.
    obs_hit_pen: float
        Penalty multiplier for hitting an obstacle.
    bots_hit_pen: float
        Penalty multiplier for hitting another bot.
    finish_bonus: float
        Reward for reaching one's target.

    Returns
    -------
    reward: float
        Instantaneous reward signal.

    Preconditions
    -------------
    board.reset() has already been called.
    """
    # dist_to_go() returns the l_1 distance between an bot's position
    # and its target, ignoring intermediate obstacles and other bots that
    # might be in the way. See board.py's Bot class.
    return finish_bonus * bot.attarget() - (
        dist_pen * bot.dist_to_go
        + obs_hit_pen * obstacles_hit(bot)
        + bots_hit_pen * bots_hit(bot)
    )


# TODO: decide if we need this function
def board_reward(board, alpha, beta, gamma):
    """
    Given a DistributedBoard, return the "instantaneous" reward signal.

    This is to be highly engineered. Currently it's stupid.

    Params
    ------
    board: DistributedBoard
        Game board whose state determines the reward signal.
    alpha: float
        Penalty multiplier for distance-to-go.
    beta: float
        Pentalty multiplier for hitting an obstacle.
    gamma: float
        Penalty multiplier for hitting another bot.

    Returns
    -------
    reward: float
        Instantaneous reward signal

    Preconditions
    -------------
    board.reset() has already been called
    """
    return 1  # TODO: unneeded?

