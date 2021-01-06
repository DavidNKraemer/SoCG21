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
        'out': np.array([[1,-1], [1,0], [1,1]]),
        'in': np.array([[-1,-1], [-1,0], [-1,1]]),
    },
    "N" : {
        'dir': np.array([0, 1]),
        'out': np.array([[-1,-1],[0,-1],[1,-1]]),
        'in': np.array([[-1,1],[0,1],[1,1]]),
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
        """
        self.position = start  # length two numpy array
        self.target = target  # length two numpy array
        self.board = board 

    def attarget(self):
        return np.all(self.position == self.target)

    def move(self, direction):
        """
        Move self to new_position, a length-two list denoting x,y coordinates.
        """
        move = MOVES[direction]['dir']

        # find all of the pixels *leaving* the neighborhood
        old_axis = [self.position + x for x in MOVES[direction]['out']]

        # find all of the pixels *entering* the neighborhood
        new_axis = [self.position + x for x in MOVES[direction]['in']]
        
        # move the agent's position
        self.position += move

        for pixel in old_axis:
            # remove pixels no longer in the neighborhood
            print(self.board.relevant_pixels[tuple(pixel)])
            self.board.relevant_pixels[tuple(pixel)].remove(self)
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

        # init a dict: length-two numpy arrays -> lists of Agents
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
