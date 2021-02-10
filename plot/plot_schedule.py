import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def pixel(xy, **kwargs):
    """
    Construct and return a unit Rectangle centered at (x, y).

    Parameters
    ----------
    xy: (float, float) or (int, int)
        Tuple of floats describing x, y coordinates; if you pass an int, then
        this function will handle the type-casting.
    **kwargs
        Splat keyword arguments here; e.g., color="red", alpha=0.6, etc.
    """
    if isinstance(xy[0], int) and isinstance(xy[1], int):
        xy[0], xy[1] = float(xy[0]), float(xy[1])
    # Rectangle(xy: tuple[float], width, height)
    return Rectangle(xy, 1, 1, **kwargs)

def plot(board_env):
    """
    Plot the current time-step of board_env.

    Call this function at each board-clock time-step to produce a "flipbook."

    Parameter
    ---------
    board_env: src.envs.BoardEnv
    """
    fig, ax = plt.subplots()
    # extract the DistributedBoard
    board = board_env.board

    # plot starts
    for start in board._starts:
        ax.add_patch(pixel(start, color='green', alpha=0.7))

    # plot targets
    for target in board._targets:
        ax.add_patch(pixel(target, color='red', alpha=0.7))

    ax.grid()
    ax.axis('equal')
    plt.show()


if __name__ == "__main__":
    # call plot
    plot()
