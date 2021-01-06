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
        self.neighborhood = list()

    def move(self, new_position):
        """
        Move self to new_position, a length-two list denoting x,y coordinates.
        """
        # Can someone please tell me how one writes a Pythonic, multi-line
        # error message? Is triple-quoting the best way to do it? See the
        # error messages following the commas in the assert statements.

        assert new_position != self.position, "Oops; the Agent did not move!"
        assert abs(self.position[0]-self.new_position[0]) <= 1, """Tried to
        move too far horizontally."""
        assert abs(self.position[1]-self.new_position[1]) <= 1, """Tried to
        move too far vertically."""
        assert not(abs(self.position[0]-self.new_position[0]) == 1 and (
        abs(self.position[1]-self.new_position[1]) == 1)), """Can't move
        diagonally."""

        # update position; observe that neighborhood hasn't been updated...
        # neighborhood updates are handled by DistributedState
        self.position = new_position


class DistributedState:
    """
    A distributed state-representation comprised of Agents and some auxilliary,
    global bookeeping.

    Fields:
        -agents: a set of Agents;
        -obstacles: a set of pixels that are blocked-off and can't be used;
        -relevant_pixels: a dict: pixels -> Agents. These are pixels either
         occupied by agents or those within agents' local neighborhoods.
    """
    def __init__(self, agents, obstacles):
        """
        Construct a DistributedState object.
        """
        self.agents = agents  # Set of Agents
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
