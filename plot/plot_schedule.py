import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product

def pixel(xy, **kwargs):
    """
    Construct and return a unit Rectangle centered at (x, y).

    Parameters
    ----------
    xy: (float, float) or (int, int)
        Tuple of floats describing x, y coordinates; if you pass an int, then
        this function will handle the type-casting.
    **kwargs
        Splat keyword arguments; e.g., color="red", alpha=0.5, etc.
    """
    if isinstance(xy[0], int) and isinstance(xy[1], int):
        xy = (float(xy[0]), float(xy[1]))
    # Rectangle(xy: tuple[float], width, height)
    return Rectangle(xy, 1, 1, **kwargs)

def plot(board_env, xlim=(-10,10), ylim=(-10,10)):
    """
    Plot the current time-step of board_env.

    Call this function at each board-clock time-step to produce a "flipbook."

    Parameters
    ----------
    board_env: src.envs.BoardEnv
        Board environment.
    xlim: tuple[int]
        Lower and upper x-axis boundaries; default=(-10,10).
    ylim: tuple[int]
        Lower and upper y-axis boundaries; default=(-10,10).
    """
    fig, ax = plt.subplots()
    # extract and alias the DistributedBoard
    board = board_env.board

    # plot current positions
    for agent in board.agents:
        ax.add_patch(pixel(agent.position, color='green', alpha=0.5))

    # plot targets
    for target in board._targets:
        ax.add_patch(pixel(target, color='red', alpha=0.5))

    # plot obstacles
    for obstacle in board.obstacles:
        ax.add_patch(pixel(obstacle, color='grey', alpha=0.5))

    # "plot" invisible pixels to keep plot from zooming in; this does *not*
    # affect the collision detector
    for coords in product(xlim, ylim):
        ax.add_patch(pixel(coords, alpha=0.0))

    ax.set_xticks([i for i in range(xlim[0], xlim[1]+1)])
    ax.set_yticks([i for i in range(ylim[0], ylim[1]+1)])
    ax.grid(True)
    ax.axis('equal')
    plt.show()


if __name__ == "__main__":
    # call plot
    plot()
