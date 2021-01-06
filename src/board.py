MOVES = dict()
MOVES["N"] = np.array([0, 1])
MOVES["S"] = np.array([0, -1])
MOVES["E"] = np.array([1, 0])
MOVES["W"] = np.array([-1, 0])
MOVES[""] = np.array([0, 0])


class Agent:
    """
    A pixel-robot aware of only its own local data, not those of its peers.

    Fields:
        -position: current position, a coordinate pair;
        -target: target coordinate pair;
        -neighborhood: representation of the surrounding 8 pixels.
    """
    def __init__(self, start, target):
        """
        Construct an Agent object.
        """
        self.position = start  # length two numpy array
        self.target = target  # length two numpy array
        self.board = None

    def move(self, direction):
        """
        Move self to new_position, a length-two list denoting x,y coordinates.
        """
        self.position += MOVES[direction]
        self.broadcast(board, self.position)
        
        new_neighbors = board.relevant_pixels(self.position)  # set of Agents
        for neighbor in new_neighbors:
            for value in MOVES.values():
                coord_pair = self.position + value
                
                neighbor.position

    def link(self, board):
        self.board = board
        return self


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
        self.agents = [Agent(start, target).link(self) for start, target in \
                       zip(starts, targets)]
        self.obstacles = obstacles  # Set of length-two numpy arrays

        # init a dict: length-two numpy arrays -> lists of Agents
        self.relevant_pixels = dict()
        # values are pointers to Agent objects
        for agent in agents:
            self.relevant_pixels[agent.position].append(agent)
            for pixel in agent.neighborhood:
                self.relevant_pixels[pixel].append(agent)

    def isempty(self, pixel):
        """
        Decide if the passed pixel is empty, returning True if and only if it
        is neither in self.obstacles or in self.relevant_pixels.
        """
        if (pixel not in self.obstacles) and (pixel not in
                                              self.relevant_pixels):
            return True
        else:
            return False

    def step(self, agent, new_position):
        """
        Move exactly one Agent.
        """
        assert new_position not in self.obstacles, "Obstacle in the way!"
        # need only update agent's local data
        if self.isempty(new_position):
            agent.move(new_position)
        # must update other agents' local data too
        else:
            affected_agents = self.relevant_pixels[new_position]
