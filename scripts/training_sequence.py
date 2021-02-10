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
        'targets': np.random.randint(-10,10, size=(1,2))
    },
]
