import numpy as np

training_sequence = [
    {
        'description': 'one agent, move east',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,0]]),
    },
    {
        'description': 'one agent, move west',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,0]]),
    },
    {
        'description': 'one agent, move north',
        'sources': np.array([[0,0]]),
        'targets': np.array([[0,5]]),
    },
    {
        'description': 'one agent, move south',
        'sources': np.array([[0,0]]),
        'targets': np.array([[0,-5]]),
    },
    {
        'description': 'one agent, random target',
        'sources': np.array([[0,0]]),
        'targets': np.random.randint(-10, 10, size=(1,2)),
    },
    {
        'description': 'one agent, move northwest',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,5]]),
    },
    {
        'description': 'one agent, move southwest',
        'sources': np.array([[0,0]]),
        'targets': np.array([[-5,-5]]),
    },
    {
        'description': 'one agent, move southeast',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,-5]]),
    },
    {
        'description': 'one agent, move northeast',
        'sources': np.array([[0,0]]),
        'targets': np.array([[5,5]]),
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
        'targets': np.array([5,0], [0,0]),
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
        'targets': np.array([0,5], [0,0]),
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
    },
    {
        'description': 'two agents, two random obstacles, random targets',
        'sources': np.array([[0,0], [5,0]]),
        'targets': np.random.randint(-10, 10, size=(2,2)),
        'obstacles': np.random.randint(-10, 10, size=(2,2)),
    },
]
