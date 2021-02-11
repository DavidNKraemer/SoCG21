import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
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

def padded_bbox(board, pad=5):
    """
    Return a tuple: four corners that define the board's axis-aligned bounding
    box (with some padding), in addition to xlimits and ylimits.

    This is useful for plotting. In particular, it helps us set a "zoom-level"
    that is appropriate.

    Parameters
    ----------
    board: src.board.DistributedBoard
    pad: int
    """
    xmax, ymax, xmin, ymin = -np.inf, -np.inf, np.inf, np.inf
    # concatenate with *
    for pixel in [*board._starts, *board._targets]:
        if pixel[0] > xmax:
            xmax = pixel[0]
        if pixel[0] < xmin:
            xmin = pixel[0]

        if pixel[1] > ymax:
            ymax = pixel[1]
        if pixel[1] < ymin:
            ymin = pixel[1]

    corners = [pixel for pixel in product((xmin, xmax), (ymin, ymax))]
    # pad corners
    corners[0] += np.array([-pad, -pad])  # (xmin, ymin)
    corners[1] += np.array([-pad, pad])  # (xmin, ymax)
    corners[2] += np.array([pad, -pad])  # (xmax, ymin)
    corners[3] += np.array([pad, pad])  # (xmax, ymax)
    # 30 is some arbitrary padding; ensures there are enough ticks if agents
    # run away on some tangent
    xlims = (int(xmin-30), int(xmax+30))
    ylims = (int(ymin-30), int(ymax+30))
    return corners, xlims, ylims

def plot(board_env, pad=5):
    """
    Plot the current time-step of board_env.

    Call this function at each board-clock time-step to produce a "flipbook."

    Parameters
    ----------
    board_env: src.envs.BoardEnv
        Board environment.
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
    # affect the collision detector or any of our scheduling algorithms
    corners, xlims, ylims = padded_bbox(board)
    for corner in corners:
        ax.add_patch(pixel(corner, alpha=0.0))

    ax.set_xticks([i for i in range(xlims[0], xlims[1]+1)])
    ax.set_yticks([i for i in range(ylims[0], ylims[1]+1)])
    ax.grid(True)
    ax.axis('equal')
    #plt.show()
    #pp.savefig()
    return fig
