import cgshop2021_pyutils as socg
from src.board import DistributedBoard

def unzip_instances():
    """
    Return the official instances, organized as a list.
    """
    instances_iter = socg.InstanceDatabase("official_instances.zip")
    return [inst for inst in instances_iter]

def count_agents_obs():
    """
    Print the number of agents and obstacles in each instance.

    This function is merely for the purpose of inspection.
    """
    instances = unzip_instances()
    for i, instance in enumerate(instances):
        print(f"Instance: {i} has {instance.number_of_robots} agents and\
 {len(instance.obstacles)} obstacles.")

def parse_instance(index, unzipped_instances):
    """
    Return a DistributedBoard representation of the instance at the specified
    index.

    Parameter
    ---------
    index: int
    unzipped_instances: list
        A list of instances, obtained by calling unzip_instances().
    """
    # extract the instance of interest
    inst = unzipped_instances[index]
    return inst.start, inst.target, inst.obstacles
