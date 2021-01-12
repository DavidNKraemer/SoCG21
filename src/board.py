import numpy as np
from collections import defaultdict
from itertools import product
import heapq

MOVES = {
    "E": {
        'dir': np.array([1, 0]),
        'out': np.array([[-1,-1], [-1,0], [-1,1]]),
        'in': np.array([[1,-1], [1,0], [1,1]]),
    },
    "W" : {
        'dir': np.array([-1, 0]),
        'out': np.array([[1,1], [1,0], [1,-1]]),
        'in': np.array([[-1,1], [-1,0], [-1,-1]]),
    },
    "N" : {
        'dir': np.array([0, 1]),
        'out': np.array([[-1,-1],[0,-1],[1,-1]]),
        'in': np.array([[1,1],[0,1],[-1,1]]),
    },
    "S" : {
        'dir': np.array([0, -1]),
        'out': np.array([[-1,1],[0,1],[1,1]]),
        'in': np.array([[-1,-1],[0,-1],[1,-1]]),
    },
    "": {
        'dir': np.array([0, 0]),
        'out': [],
        'in': [],
    }
}

class DumbPolicy:
    """
    Do not use, just for demo
    """

    def __init__(self, board):
        self.board = board

    def __call__(self, agent):
        horiz, vert = agent.target - agent.position
        if horiz != 0:
            return 'E' if horiz > 0 else 'W'
        elif vert != 0:
            return 'N' if vert > 0 else 'S'
        else:
            return ''


class Agent:
    """
    A pixel-robot aware of only its own local data, not those of its peers.

    Fields:
        -position: current position, a coordinate pair;
        -target: target coordinate pair;
        -neighborhood: representation of the surrounding 8 pixels.
    """
    def __init__(self, start, target, board):
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
        self._local_state = LocalState(self)

    @property
    def state(self):
        return self._local_state.state

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
        direction in {'up', 'down', 'left', 'right'}
        """
        # find all of the pixels *leaving* the neighborhood
        old_axis = self.position + MOVES[direction]['out']

        # move the agent's position
        self.position += MOVES[direction]['dir']

        new_axis = self.position + MOVES[direction]['in']

        for pixel in old_axis:
            # remove pixels no longer in the neighborhood
            self.board.relevant_pixels[tuple(pixel)].remove(self)
            if not self.board.relevant_pixels[tuple(pixel)]:
                del self.board.relevant_pixels[tuple(pixel)]
        for pixel in new_axis:
            # add pixels now in the neighborhood
            self.board.relevant_pixels[tuple(pixel)].add(self)

    def neighborhood(self, dist):
        """
        Returns a generator of the neighborhood of all pixels of the agent of
        distance <= dist
        """
        for x, y in product(np.arange(-dist, dist+1), repeat=2):
            yield self.position + np.array([x,y])

    def priority(self):
        """
        Priority value of the agent
        """
        return np.linalg.norm(self.target - self.position, ord=1)

    def __lt__(self, other):
        return self.priority() > other.priority()

    def __repr__(self):
        """
        Debug friendly info
        """
        return f"Agent({self.position}, {self.target}, {self.priority()})"


class DistributedBoard:
    """
    A distributed state-representation comprised of Agents and some auxilliary,
    global bookeeping.

    Fields:
        -agents: a set of Agents;
        -obstacles: a set of pixels that are blocked-off and can't be used;
        -relevant_pixels: a dict: pixels -> Agents. These are pixels either
         occupied by agents or those within agents' local neighborhoods.
    """
    def __init__(self, starts, targets, obstacles):
        """
        Construct a DistributedState object.
        """
        self.agents = [Agent(start, target, self) for start, target in \
                       zip(starts, targets)]
        self.obstacles = obstacles  # Set of length-two numpy arrays
        self.queue = []

        # init a dict: length-two numpy arrays -> sets of Agents
        self.relevant_pixels = defaultdict(set)
        # values are pointers to Agent objects
        for agent in self.agents:
            heapq.heappush(self.queue, agent)
            for pixel in agent.neighborhood(1):  # change for bigger nbds
                self.relevant_pixels[tuple(pixel)].add(agent)

    def pop(self):
        agent = heapq.heappop(self.queue)
        return agent

    def insert(self, agent):
        heapq.heappush(self.queue, agent)

    def isdone(self):
        """
        Returns whether all agents have found their target
        """
        return all(agent.attarget() for agent in self.agents)

class LocalState:
    """
    Local state information for one Agent.
    """
    def __init__(self, agent):
        self.agent = agent

    @property
    def state(self):
        """
        Encode current position, target position, and neighborhood.
        """
        neighborhood = np.zeros((9, 2))
        # update neighborhood
        # might need to update the below -- perhaps attach to Agent.move()?
        agent_positions = defaultdict(int)
        for agent in self.agent.board.agents:
            agent_positions[tuple(agent.position)] += 1

        for index, pixel in enumerate(self.agent.neighborhood(1)):
            neighborhood[index, 0] = agent_positions[tuple(pixel)]
            neighborhood[index, 1] = int(pixel in self.agent.board.obstacles)

        return np.r_[self.agent.position, self.agent.target,
                     neighborhood.reshape(-1)]











