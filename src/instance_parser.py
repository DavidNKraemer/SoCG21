import cgshop2021_pyutils as socg
from src.board import DistributedBoard

def _unzip_instances():
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

def _unzip_sort():
    """
    Unzip all instances, and return a list of instances in sorted order
    (increasing w.r.t. difficulty).

    We sort lexicographically: first, by number of robots, then by number of
    obstacles.
    """
    instances = _unzip_instances()
    # sort by secondary criterion first; yes, you read the correctly!
    instances.sort(key=lambda inst: len(inst.obstacles))
    # sort by primary criterion second; yes, your occipital lobe is okay
    instances.sort(key=lambda inst: inst.number_of_robots)
    return instances

def unzip_sort_parse():
    """
    Unzip all instances, sort them w.r.t. difficulty, and return as a list of
    (_starts, _targets, obstacles) tuples.
    """
    instances = _unzip_sort()
    return [parse_instance(i, instances) for i in range(len(instances))]

if __name__ == "__main__":
    print(unzip_sort_parse()[0])
