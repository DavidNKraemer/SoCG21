import numpy as np
from collections import defaultdict
from itertools import product
import heapq
from copy import copy, deepcopy

from pettingzoo.utils import agent_selector as bot_selector


MOVES = {
    "E": np.array([1, 0]),
    "W": np.array([-1, 0]),
    "N": np.array([0, 1]),
    "S": np.array([0, -1]),
    "": np.array([0, 0]),
}


def get_pixels(direction, radius):
    """
    Given a specified direction and radius, return every pixel in the L
    infinity neighborhood of (0,0) of the given radius along the given
    direction.

    So for example, if the direction is [1,0] (east), the pixels returned are
    all of the pixels to the right of (0,0).

    Params
    ------
    direction: numpy ndarray
        One of the four cardinal directions, or the zero vector (for no
        movement)
    radius: int
        Radius of the neighborhood to compute

    Returns
    -------
    pixels: numpy ndarray
        Every pixel along the given direction in the L infinity neighborhood of
        (0,0) within the given radius.

    Preconditions
    -------------
    direction in {[1,0], [-1,0], [0,1], [0,-1], [0,0]}
    radius > 0

    Postconditions
    --------------
    If direction == [0, 0], an empty array is returned.
    Otherwise, pixels.shape[1] == 2
    """
    if np.all(direction == 0):
        return np.array([[]]).reshape(-1, 2)

    pixels = []
    parallel = (direction != 0).astype(np.int)  # parallel axis
    sign = int(direction[parallel == 1])
    orthogonal = (direction == 0).astype(np.int)  # orthogonal axis

    for i in range(1, radius + 1):
        for j in range(-radius, radius + 1):
            pixels.append(list(sign * i * parallel + j * orthogonal))

    return np.array(pixels)


class DumbPolicy:
    """
    Do not use, just for demo

    Might suggest a good abstract base class for policies in general
    """

    def __init__(self, board):
        """
        There's one policy for the entire board, and calling the policy takes a
        single bot.

        Params
        ------
        board: Board
            The game board under consideration
        """
        self.board = board

    def __call__(self, bot):
        """
        Given an bot, determine a move

        Move horizontally until no longer needed. Then move vertically until
        arrived at the target

        Params
        ------
        bot: Bot
            The current bot to direct

        Returns
        -------
        move: str

        Postconditions
        --------------
        move in {'E', 'W', 'N', 'S', ''}
        """
        # compute displacements
        horiz, vert = bot.target - bot.position

        # horizontal displacement
        if horiz != 0:
            return "E" if horiz > 0 else "W"

        # vertical displacement
        elif vert != 0:
            return "N" if vert > 0 else "S"

        # if no displacement, just stay put
        else:
            return ""


class DistributedBoard:  # TODO: why isn't this a subclass of gym.Environment?
    """
    A distributed state-representation comprised of Bots and some auxilliary,
    global bookeeping.

    Fields
    ------
    bots
        a list of Bots;
    obstacles
        a list of pixels that are blocked-off and can't be used;
    active_pixels
        defaultdict: pixels -> Set[int]. These are pixels either occupied by
        bots *or* those within bots' local neighborhoods;
    prev_active_pixels
        active_pixels as seen in the previous (board-clock) time-step;
    occupied_pixels
        defaultdict: pixels -> Set[int]. These are pixels occupied by bots.
    prev_occupied_pixels
        occupied_pixels as seen in the previous (board-clock) time-step;
    queue
        heap used to handle orders in which bots are to move;
    clock
        time-step counter.
    """

    def __init__(
        self, starts, targets, obstacles, neighborhood_radius=2, **kwargs
    ):
        """
        Construct a DistributedBoard object.
        """
        self._starts = starts
        self._targets = targets
        self.obstacles = obstacles  # Set of length-two numpy arrays
        self.max_clock = kwargs.get("max_clock", None)
        self.neighborhood_radius = neighborhood_radius

        self._obs_shape = Bot.StateRepresentation.shape(neighborhood_radius)

    def _snapshot(self):
        """
        Stash current timestep info.

        Allows recovery of local neighborhood of a given bot from the stashed
        timestep.
        """
        self.prev_active_pixels = deepcopy(self.active_pixels)
        self.prev_occupied_pixels = deepcopy(self.occupied_pixels)

    def reset(self):
        """
        Resets the DistributedBoard, useful for reusing the same object for gym
        environment.
        """
        # initialize all the bots to original start/target info
        self.bots = []
        for i, (start, target) in enumerate(zip(self._starts, self._targets)):
            self.bots.append(Bot(start, target, self, i))

        self._bot_selector = bot_selector(self.bots)
        self.selected_bot = self._bot_selector.next()

        self.bot_actions = ["" for _ in self.bots]

        # reset the clock
        self.clock = 0

        # init a dict: length-two numpy arrays -> sets of Bot ids
        self.active_pixels = defaultdict(set)

        # init a dict: tuple -> sets of Bot ids
        self.occupied_pixels = defaultdict(set)

        # values are bot ids
        for bot in self.bots:
            # activate pixels
            for pixel in bot.neighborhood(self.neighborhood_radius):
                # this is the intended use of active_pixels
                self.active_pixels[tuple(pixel)].add(bot.bot_id)

            # occupied pixels
            self.occupied_pixels[tuple(bot.position)].add(bot.bot_id)
            # TODO: why don't obstacles get added to the occupied_pixels?

        self._snapshot()

    def update_bots(self):
        """ """
        self.clock += 1

        for bot_id, bot in enumerate(self.bots):
            bot.move(self.bot_actions[bot_id])
            self.bot_actions[bot_id] = ""

    def isdone(self):
        """
        Returns whether all bots have found their target

        Returns
        -------
        True iff every bot is at its target position
        """
        targets_reached = all(bot.attarget() for bot in self.bots)

        clock_expired = (
            False if self.max_clock is None else self.clock >= self.max_clock
        )

        return clock_expired or targets_reached


def cart_to_imag(origin, position, radius):
    """
    Cartesian coordinates to image coordinates

    Params
    ------
    origin: numpy ndarray
    position: numpy ndarray
    radius: int
    """
    diff = position - origin
    imag_diff = np.array([-diff[1], diff[0]])

    array_center = np.full(imag_diff.shape, radius)

    return array_center + imag_diff


class LocalState:
    """
    Local state information for one Bot.
    """

    @staticmethod
    def shape(radius):
        length = 2 * radius + 1
        return (2 * length * length + 4,)

    def __init__(self, bot):
        """
        Each object is attached ("privately", no need for external use) to a
        single bot.

        Return
        ------
        bot: Bot
        """
        self.bot = bot
        self.board = self.bot.board
        self.neighborhood_radius = self.board.neighborhood_radius

    @property
    def state(self):
        """
        Encode current position, target position, and neighborhood.

        Current features:
            * for every tile in the neighborhood of the bot, return the
              number of bots occupying the tile as well as whether an
              obstacle is occupying the tile
            * current location of the bot
            * target location of the bot

        TODO: make more interesting

        Returns
        -------
        encoded_state: nump ndarray
        """
        n = 2 * self.neighborhood_radius + 1

        neighborhood = np.zeros((n * n, 2))
        square_nbd = neighborhood.view().reshape((n, n, 2))

        BOTS, OBSTACLES = 0, 1

        # for every pixel in the neighborhood
        for index, pixel in enumerate(
            self.bot.neighborhood(self.neighborhood_radius)
        ):
            # check if the pixel is occupied by an obstacle
            neighborhood[index, OBSTACLES] = int(pixel in self.board.obstacles)

            # for every (other) bot in the active_pixels set corresponding to
            # the corresponding pixel
            for bot_id in self.board.occupied_pixels[tuple(pixel)]:
                # if the other bot is in the neighborhood, increment that
                # position pixel
                if self.bot.bot_id != bot_id:
                    other = self.board.bots[bot_id]
                    i, j = cart_to_imag(
                        self.bot.position,
                        other.position,
                        self.neighborhood_radius,
                    )
                    square_nbd[i, j, BOTS] += 1

        return np.r_[
            self.bot.position, self.bot.target, neighborhood.reshape(-1)
        ]


class LocalStateDQN:
    """
    Local state representation for a single Bot for use in DQN.

    Each state is a stack, or tensor, of 3 square images composed of pixels,
    centered on the Bot. Each image in the stack represents a different
    piece of information: the number of neighboring Bots occupying nearby
    pixels, a map of nearby obstacles, direction to the Bot's target, etc.

    In the case where side length is 3 pixels, an example state is:

        [[[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]],

         [[1, 0, 0],
          [0, 0, 0],
          [0, 0, 0]],

         [[0, 0, 1],
          [0, 0, 0],
          [0, 0, 0]]]

    where the first image gives the bot neighborhood, the second gives
    the obstacle neighborhood, and the third has a 1 in the direction of the
    Bot's target.
    """

    @staticmethod
    def shape(radius):
        length = 2 * radius + 1

        return (3, length, length)

    def __init__(self, bot):
        self.bot = bot
        self.board = self.bot.board
        self.neighborhood_radius = self.board.neighborhood_radius
        self.side_length = 2 * self.neighborhood_radius + 1

        # cartesian coordinates of the bottom left corner of the neighborhood
        self.offset = self.bot.position - np.full(
            self.bot.position.shape, self.neighborhood_radius
        )

        self.shape = (3, self.side_length, self.side_length)

    def _is_row_in(self, arr, rows):
        """
        Check if a 1D numpy array is a row in a 2D numpy array.
        """
        return (arr == rows).all(axis=1).any()

    @property
    def state(self):
        state = np.zeros((3, self.side_length, self.side_length))
        # cartesian coordinates of the bottom left corner of the neighborhood
        self.offset = self.bot.position - np.full(
            self.bot.position.shape, self.neighborhood_radius
        )

        BOTS, OBSTACLES, DIRECTION = 0, 1, 2

        neighborhood = self.bot.neighborhood(self.neighborhood_radius)

        for pixel in neighborhood:
            # add number of bots to each pixel in first image

            # Wes: should the below line be occupied_pixels[tuple(pixel)],
            # or is it okay as-is? Just double-checking... I do not want to
            # pre-emptively influence your opinion.
            #
            # ANSWER (10/26/21): should be occupied_pixels
            for bot_id in self.board.occupied_pixels[tuple(pixel)]:
                bot = self.board.bots[bot_id]
                if self.bot.bot_id != bot_id:
                    indices = BOTS, *(bot.position - self.offset)
                    state[indices] += 1

            # add obstacles to second image
            indices = OBSTACLES, *(pixel - self.offset)
            state[indices] = (self.board.obstacles == pixel).all(axis=1).sum()

        # add target direction to third image, projecting onto neighborhood
        # if necessary
        if self._is_row_in(self.bot.target, neighborhood):
            indices = DIRECTION, *(self.bot.target - self.offset)
            state[indices] = 1
        else:
            # TODO: only need to project on the *boundary* (which is Omega(n)) of
            # the neighborhood, not the entire neighborhood (which is Omega(n^2))

            # should be L2?? I would think L1 projection <-- write a GH issue
            diff = self.bot.target - self.bot.position
            linf_dist = np.abs(diff).max()

            intersection = (
                self.bot.position
                + (self.neighborhood_radius / linf_dist) * diff
            )
            boundary = self.bot.boundary(self.neighborhood_radius)
            distances = np.linalg.norm(boundary - intersection, axis=1, ord=2)

            indices = DIRECTION, *(boundary[np.argmin(distances)] - self.offset)
            state[indices] = 1

        # flip each image, as images were modified upside-down
        return np.flip(np.transpose(state, axes=(0, 2, 1)), axis=1)


class Bot:  # TODO: rename!  Current candidate: "Bot"
    """
    A pixel-robot aware of only its own local state, not those of its peers.

    Fields:
        -position: current position, a coordinate pair;
        -target: target coordinate pair;
        -neighborhood: representation of the surrounding 9 pixels.
    """

    StateRepresentation = LocalStateDQN

    def __init__(self, start, target, board, bot_id):
        """
        Construct an Bot object.

        Params
        ------
        start: numpy ndarray
            bot's starting position
        target: numpy ndarray
            bot's targett position
        board: Board
            game board object on which bot resides
        bot_id: int
            unique identifier with respect to the game board

        Preconditions
        -------------
        start.shape == target.shape == (2,)
        """
        self.position = deepcopy(start)  # length two numpy array
        self.target = target  # length two numpy array
        self.board = board
        self._local_state = Bot.StateRepresentation(
            self
        )  # state representer class
        self.bot_id = bot_id
        self.prev_position = self.position  # initialize; update with move()

    @property
    def state(self):
        """
        The bot communicates its "state" through its state representation
        class

        Returns
        -------
        state_data: numpy ndarray
            numpy-friendly state data aggregator
        """
        return self._local_state.state

    @property
    def dist_to_go(self):
        """
        Return the l_1 distance between self's current position and that of
        the target position.

        Note: this method ignores obstacles and other bots that might be in
        the way! We return only the l_1 distance between two pixels. Nothing
        fancy is done w.r.t. obstacle avoidance.
        """
        dist = np.linalg.norm(self.target - self.position, ord=1)
        return dist

    def attarget(self):
        """
        Returns whether the bot is in its target position

        Returns
        -------
        True iff the current position equals the target position
        """
        return np.all(self.position == self.target)

    def move(self, direction):
        """
        "Move" the bot according to the specified direction. After calling
        this method, the following will have been updated:
            * the bot's position will be in the adjacent tile corresponding
              to the direction
            * the bot will have "exited" from the tiles opposite of the
              direction of movement
            * the bot will have "entered" the tiles along the direction of
              movement

        Params
        ------
        direction: str
            movement direction

        Preconditions
        -------------
        direction in {'W', 'E', 'N', 'S', ''}
        """
        # the following three steps are noncommutative. it has to be that we
        # remove the old axis before updating the position, before adding the
        # new axis

        # find all of the pixels *leaving* the neighborhood
        old_axis = self.position + get_pixels(-MOVES[direction], 1)

        # remove bot from position
        self.board.occupied_pixels[tuple(self.position)].remove(self.bot_id)
        # if an empty set
        if not self.board.occupied_pixels[tuple(self.position)]:
            del self.board.occupied_pixels[tuple(self.position)]

        # update previous position (needed for bots_hit() in env.py)
        self.prev_position = copy(self.position)
        # move the bot's position
        self.position += MOVES[direction]

        # update position in occupied_pixels
        self.board.occupied_pixels[tuple(self.position)].add(self.bot_id)

        new_axis = self.position + get_pixels(MOVES[direction], 1)

        for pixel in old_axis:
            # remove pixels no longer in the neighborhood
            self.board.active_pixels[tuple(pixel)].remove(self.bot_id)
            if not self.board.active_pixels[tuple(pixel)]:
                del self.board.active_pixels[tuple(pixel)]

        for pixel in new_axis:
            # add pixels now in the neighborhood
            self.board.active_pixels[tuple(pixel)].add(self.bot_id)

    def neighborhood(self, dist):
        """
        Returns a numpy array of the L_infinity ball of radius `dist`

        Params
        ------
        dist: int
            radius (discrete)

        Returns
        -------
        pixels: numpy ndarray shape=((2*dist+1)**2, 2)
            all pixels with L_infinity distance at most `dist`
        """
        x = np.arange(-dist, dist + 1)
        X, Y = np.meshgrid(x, x)

        return self.position + np.c_[X.reshape(-1), Y.reshape(-1)]

    def boundary(self, dist):
        """
        Returns a generator of the L_infinity sphere of radius `dist`

        The sphere is the *boundary* of the ball.

        Params
        ------
        dist: int
            radius (discrete)

        Returns
        -------
        pixels: generator
            all pixels with L_infinity distance equalling `dist`
        """
        sphere = []

        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        next_directions = np.roll(directions, 1, axis=0)

        for direction, next_direction in zip(directions, next_directions):

            # parallel axis
            parallel = (direction != 0).astype(int)

            # orthogonal axis
            orthogonal = (direction == 0).astype(int)

            # direction along the parallel axis
            sign = int(direction[parallel == 1])

            sequence = np.arange(-dist + 1, dist).reshape((-1, 1))

            tiles = sign * dist * parallel + sequence * orthogonal
            corner = dist * (direction + next_direction)

            sphere += list(tiles) + [corner]

        return self.position + np.array(sphere)

    def __str__(self):
        """
        Print-friendly description string
        """
        return f"Bot(position={self.position}, target={self.target})"

    def __repr__(self):
        """
        Debug-friendly info
        """
        return str(self)
        # return f"Bot<{id(self)}>(Board<{id(self.board)}>)"


class BotForDQN(Bot):
    """
    Same as Bot, but uses LocalStateDQN instead of LocalState.
    """

    def __init__(self, start, target, board, bot_id):

        super().__init__(start, target, board, bot_id)

        self._local_state = LocalStateDQN(self)
