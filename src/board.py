import numpy as np
from collections import defaultdict
from itertools import product
import heapq
from copy import deepcopy


MOVES = {
    "E": np.array([ 1,  0]),
    "W": np.array([-1,  0]),
    "N": np.array([ 0,  1]),
    "S": np.array([ 0, -1]),
    "":  np.array([ 0,  0]),
}


def get_pixels(direction, radius):
    """
    Given a specified direction and radius, return every pixel in the L infinity
    neighborhood of (0,0) of the given radius along the given direction.

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
        Every pixel along the given direction in the L infinity neighborhood  of
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
        return np.array([[]])

    pixels = []
    parallel = (direction != 0).astype(np.int)
    sign = int(direction[parallel == 1])
    orthogonal = (direction == 0).astype(np.int)
    for i in range(1, radius+1):
        for j in range(-radius, radius+1):
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
        single agent.

        Params
        ------
        board: Board
            The game board under consideration
        """
        self.board = board

    def __call__(self, agent):
        """
        Given an agent, determine a move

        Move horizontally until no longer needed. Then move vertically until
        arrived at the target

        Params
        ------
        agent: Agent
            The current agent to direct

        Returns
        -------
        move: str

        Postconditions
        --------------
        move in {'E', 'W', 'N', 'S', ''}
        """
        # compute displacements
        horiz, vert = agent.target - agent.position

        # horizontal displacement
        if horiz != 0:
            return 'E' if horiz > 0 else 'W'

        # vertical displacement
        elif vert != 0:
            return 'N' if vert > 0 else 'S'

        # if no displacement, just stay put
        else:
            return ''


class Agent:
    """
    A pixel-robot aware of only its own local data, not those of its peers.

    Fields:
        -position: current position, a coordinate pair;
        -target: target coordinate pair;
        -neighborhood: representation of the surrounding 9 pixels.
    """
    def __init__(self, start, target, board, agent_id):
        """
        Construct an Agent object.

        Params
        ------
        start: numpy ndarray
            agent's starting position
        target: numpy ndarray
            agent's targett position
        board: Board
            game board object on which agent resides

        Preconditions
        -------------
        start.shape == target.shape == (2,)
        """
        self.position = start  # length two numpy array
        self.target = target  # length two numpy array
        self.board = board
        self._local_state = LocalState(self)  # state representer class
        self.local_clock = 0  # local clock
        self.agent_id = agent_id
        self.prev_position = self.position  # initialize; update with move()

    @property
    def state(self):
        """
        The agent communicates its "state" through its state representation
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

        Note: this method ignores obstacles and other agents that might be in
        the way! We return only the l_1 distance between two pixels. Nothing
        fancy is done w.r.t. obstacle avoidance.
        """
        return np.linalg.norm(self.target - self.position, ord=1)

    def attarget(self):
        """
        Returns whether the agent is in its target position

        Returns
        -------
        True iff the current position equals the target position
        """
        return np.all(self.position == self.target)

    def move(self, direction):
        """
        "Move" the agent according to the specified direction. After calling
        this method, the following will have been updated:
            * the agent's position will be in the adjacent tile corresponding to
              the direction
            * the agent will have "exited" from the tiles opposite of the
              direcction of movement
            * the agent will have "entered" the tiles along the direction of
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

        # update previous position (needed for agents_hit() in env.py)
        self.prev_position = self.position
        # move the agent's position
        self.position += MOVES[direction]

        new_axis = self.position + get_pixels(MOVES[direction], 1)

        for pixel in old_axis:
            # remove pixels no longer in the neighborhood
            self.board.active_pixels[tuple(pixel)].remove(self)
            if not self.board.active_pixels[tuple(pixel)]:
                del self.board.active_pixels[tuple(pixel)]
        for pixel in new_axis:
            # add pixels now in the neighborhood
            self.board.active_pixels[tuple(pixel)].add(self)

        # update the local clock
        self.local_clock += 1

    def neighborhood(self, dist):
        """
        Returns a generator of the L_infinity ball of radius `dist`

        Params
        ------
        dist: int
            radius (discrete)

        Returns
        -------
        pixels: generator
            all pixels with L_infinity distance at most `dist`
        """
        for x, y in product(np.arange(-dist, dist+1), repeat=2):
            yield self.position + np.array([x,y])

    def priority(self):
        """
        Priority value of the agent to be evaluated in the movement queue in the
        board

        TODO: make a better priority function, possibly give it over to a
        learner?

        Returns
        -------
        priority_val: float
            higher iz gooder
        """
        return np.linalg.norm(self.target - self.position, ord=1)

    def __lt__(self, other):
        """
        Compares agents for ordering in the priority queue.

        Params
        ------
        other: Agent
            to be compared against

        Returns
        -------
        True iff self has higher priority than other
        """
        return self.priority() > other.priority()

    def __str__(self):
        """
        Print-friendly description string
        """
        return f"Agent({self.position}, {self.target}, {self.priority()})"

    def __repr__(self):
        """
        Debug-friendly info
        """
        return f"Agent<{id(self)}>(Board<{id(self.board)}>)"


class DistributedBoard:
    """
    A distributed state-representation comprised of Agents and some auxilliary,
    global bookeeping.

    Fields:
        -agents: a set of Agents;
        -obstacles: a set of pixels that are blocked-off and can't be used;
        -active_pixels: defaultdict: pixels -> Set[Agent]. These are pixels
         either occupied by agents or those within agents' local neighborhoods;
        -prev_active_pixels: active_pixels as seen in the previous
         (board-clock) time-step;
        -queue: heap used to handle orders in which agents are to move;
        -clock: time-step counter.
    """
    def __init__(self, starts, targets, obstacles):
        """
        Construct a DistributedBoard object.
        """
        self._starts = starts
        self._targets = targets
        self.obstacles = obstacles  # Set of length-two numpy arrays

    def _snapshot(self):
        """
        Stash current timestep info.

        Allows recovery of local neighborhood of a given agent from
        the stashed timestep.
        """
        self.prev_active_pixels = deepcopy(self.active_pixels)

    def reset(self):
        """
        Resets the DistributedBoard, useful for reusing the same object for gym
        Environment
        """
        self.agents = [Agent(s, t, self, i) \
                       for i, (s, t) in enumerate(zip(self._starts, self._targets))]
        self.queue = []
        self.clock = 0

        # init a dict: length-two numpy arrays -> sets of Agents
        self.active_pixels = defaultdict(set)

        # values are pointers to Agent objects
        for agent in self.agents:
            # activate pixels
            for pixel in agent.neighborhood(1):  # change for bigger nbds
                self.active_pixels[tuple(pixel)].add(agent)

            # populate the queue
            self.insert(agent)

        self._snapshot()

    def pop(self):
        """
        Pops the next agent from the queue and returns it. During processing, we
        also manage the board's clock and the agent's local clock.

        See https://github.com/DavidNKraemer/SoCG21/issues/3#issue-784378247 for
        details of this implementation.

        Returns
        -------
        next_agent: Agent
            the agent with the highest priority
        """
        agent = heapq.heappop(self.queue)
        self._snapshot()
        self.clock = max(agent.local_clock, self.clock)
        agent.local_clock = self.clock

        return agent

    def peek(self):
        """
        Returns the next agent from the queue without removing it. During
        processing, the board's clock and the agent's local clock is updated.
        """
        agent = self.queue[0]
        self._snapshot()
        self.clock = max(agent.local_clock, self.clock)
        agent.local_clock = self.clock

        return agent

    def insert(self, agent):
        """
        Inserts an agent from the queue and returns it

        Params
        -------
        agent: Agent
            the agent to be inserted into the queue
        """
        heapq.heappush(self.queue, agent)

    def isdone(self):
        """
        Returns whether all agents have found their target

        Returns
        -------
        True iff every agent is at its target position
        """
        return all(agent.attarget() for agent in self.agents)


class LocalState:
    """
    Local state information for one Agent.
    """
    shape = (22,)
    def __init__(self, agent):
        """
        Each object is attached ("privately", no need for external use) to a
        single agent.

        Return
        ------
        agent: Agent
        """
        self.agent = agent
        self.board = self.agent.board

    @property
    def state(self):
        """
        Encode current position, target position, and neighborhood.

        Current features:
            * for every tile in the neighborhood of the agent, return the number
            of agents occupying the tile as well as whether an obstacle is
            occupying the tile
            * current location of the agent
            * target location of the agent

        TODO: make more interesting

        Returns
        -------
        encoded_state: nump ndarray
        """
        neighborhood = np.zeros((9, 2))

        # update neighborhood
        # might need to update the below -- perhaps attach to Agent.move()?
        # it probably belongs to either Agent or Board
        agent_positions = defaultdict(int)
        for agent in self.board.agents:
            agent_positions[tuple(agent.position)] += 1

        # search the neighborhood for agents and obstacles
        for index, pixel in enumerate(self.agent.neighborhood(1)):
            neighborhood[index, 0] = agent_positions[tuple(pixel)]
            neighborhood[index, 1] = int(pixel in self.board.obstacles)

        return np.r_[
            self.agent.position, self.agent.target, neighborhood.reshape(-1)
        ]

