import numpy as np


def training_plan(sequence, myopia_rate, len_epoch):
    """
    Training plan where the sequence of problems are repeated in "epochs".
    During the kth epoch, the same subsequence of problems are available for
    training. On a given step inside of the epoch, the kth item in the sequence
    is returned with probability `myopia_rate`. The probability of drawing from
    {0,...,k-1} is 1-`myopia_rate`.

    Some examples:
    epoch = 0
    weights = [1.]

    epoch = 1
    weights = [(1-myopia_rate), myopia_rate]

    epoch = 2
    weights = [(1-myopia_rate)**2, (1-myopia_rate) * myopia_rate, myopia_rate]

    This returns a generator of length len(sequence) * len_epoch
    """
    indices = np.arange(len(sequence))
    weights = np.ones(len(sequence))

    for k in range(len(sequence)):
        if k > 0:
            weights[k] = myopia_rate
            weights[:k] *= (1. - myopia_rate)
        for _ in range(len_epoch):
            index = np.random.choice(indices[:k+1], p=weights[:k+1])
            yield tuple(sequence[index].values())


training_sequence = [
    {
        'description': 'one agent, move east',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,0]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'one agent, move west',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,0]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'one agent, move north',
        'sources': np.array([[0,0]]),
        'targets': np.array([[0,5]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'one agent, move south',
        'sources': np.array([[0,0]]),
        'targets': np.array([[0,-5]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'one agent, random target',
        'sources': np.array([[0,0]]),
        'targets': np.random.randint(-10, 10, size=(1,2)),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'one agent, move northwest',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,5]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'one agent, move southwest',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,-5]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'one agent, move southeast',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,-5]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'one agent, move northeast',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,5]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
#########################
    # introduce obstacles
#########################
    {
        'description': 'one agent, one obstacle, move east',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,0]]),
        'obstacles': np.array([[3,0]]),
    },
    {
        'description': 'one agent, one obstacle, move west',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,0]]),
        'obstacles': np.array([[-3,0]]),
    },
    {
        'description': 'one agent, one obstacle, move north',
        'sources': np.array([[0,0]]),
        'targets': np.array([[0,5]]),
        'obstacles': np.array([[0,3]]),
    },
    {
        'description': 'one agent, one obstacle, move south',
        'sources': np.array([[0,0]]),
        'targets': np.array([[0,-5]]),
        'obstacles': np.array([[0,-3]]),
    },
    {
        'description': 'one agent, one random obstacle, random target',
        'sources': np.array([[0,0]]),
        'targets': np.random.randint(-10, 10, size=(1,2)),
        'obstacles': np.random.randint(-10, 10, size=(1,2)),
    },
    {
        'description': 'one agent, one obstacle, move northwest',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,5]]),
        'obstacles': np.array([[-3,3]]),
    },
    {
        'description': 'one agent, one obstacle, move southwest',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,-5]]),
        'obstacles': np.array([[-3,-3]]),
    },
    {
        'description': 'one agent, one obstacle, move southeast',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,-5]]),
        'obstacles': np.array([[3,-3]]),
    },
    {
        'description': 'one agent, one obstacle, move northeast',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,5]]),
        'obstacles': np.array([3,3]),
    },
################
    # two agents
################
    {
        'description': 'two agents, move horizontal towards one another',
        'sources': np.array([[0,0], [5,0]]),
        'targets': np.array([[5,0], [0,0]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'two agents, move horizontal toward one another, one obstacle',
        'sources': np.array([[0,0], [5,0]]),
        'targets': np.array([[5,0], [0,0]]),
        'obstacles': np.array([[3,0]]),
    },
    {
        'description': 'two agents, move vertical towards one another',
        'sources': np.array([[0,0], [0,5]]),
        'targets': np.array([[0,5], [0,0]]),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'two agents, move vertical toward one another, one obstacle',
        'sources': np.array([[0,0], [0,5]]),
        'targets': np.array([[0,5], [0,0]]),
        'obstacles': np.array([[0,3]]),
    },
    {
        'description': 'two agents, random targets',
        'sources': np.array([[0,0], [5,0]]),
        'targets': np.random.randint(-10, 10, size=(2,2)),
        'obstacles': np.array([[]]).reshape(-1,2)
    },
    {
        'description': 'two agents, two random obstacles, random targets',
        'sources': np.array([[0,0], [5,0]]),
        'targets': np.random.randint(-10, 10, size=(2,2)),
        'obstacles': np.random.randint(-10, 10, size=(2,2)),
    },
]

for tup in training_plan(training_sequence, 0.9, 1):
    #print(tup)
    continue
