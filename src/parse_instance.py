import cgshop2021_pyutils as socg

instances = socg.InstanceDatabase("official_instances.zip")
for instance in instances:
    print(f"Instance: {instance}")
