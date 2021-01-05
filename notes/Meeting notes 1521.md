# Meeting notes 1/5/21

## State representation

### "Distributed" representation

* Agent classes, each having attributes:
  * current position 
  * target position
  * local neighborhood, three options:
    * list of categorical data based on characteristics of each neighboring tile
    * one-hot encoding of the above
    * "image" representation (passed on to a CNN)
  * can we also incorporate other agents' planned moves
  * additional characteristics (?)
    * nearest neighbor data
    * direction to target
    * relative crowding of the quadrants formed by the agent's origin.
* We can efficiently update each agent's state information based on moving another agent on the board. 
  * **Clever** application of dictionaries!

### "Global" representation



## Game Board representation

### How do we decide which agent moves next?

* In the case of the *makespan* problem criteron
  * It sounds like a priority queue is a good idea .
  * How do we compute priorities? Can they be *learned*?
* In the case of the *total distance* problem criterion
  * Maybe a priority queue as well, since how do we determine who moves in any given step?

* Going from internal representation to Tensor-friendly representation
* Wrapping around the standard OpenAI Gym environment for Wes, etc.
* There will be some data structures engineering to deal with some of these computations.
* Constraint violation will be "allowed" but heavily penalized in all learning systems
* 