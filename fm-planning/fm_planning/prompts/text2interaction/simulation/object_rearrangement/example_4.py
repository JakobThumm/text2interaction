from .system_prompt import TASK_PREFIX


TASK_PREFIX = \
"""Detected Objects:
- A table with dimensions L x W x H = [2.0, 2.0, 0.05] and origin in its center.
- A screwdriver with a rod and a handle.
- A red box with a bounding box [0.05, 0.05, 0.118] and origin in its center.
- A cyan box with a bounding box [0.05, 0.05, 0.070] and origin in its center.
- A blue box with a bounding box [0.05, 0.05, 0.106] and origin in its center.

Orientation:
- Front/Behind: [+/-1, 0, 0]
- Right/Left: [0, +/-1, 0]
- Above/Below: [0, 0, +/-1]

Object Relationships:
- free(screwdriver)
- on(screwdriver, table)
- free(cyan_box)
- on(cyan_box, table)
- free(red_box)
- on(red_box, table)
- free(blue_box)
- on(blue_box, table)

Task Plan:
1. Pick(red_box, table)
2. Place(red_box, table)
3. Pick(cyan_box, table)
4. Place(cyan_box, table)
5. Pick(blue_box, table)
6. Place(blue_box, table)
"""

EXAMPLE_USER = \
f"""{TASK_PREFIX}

Instruction:
Arrange the blue box, red box, and cyan box in a circle of radius 15 cm around the screwdriver.
"""

EXAMPLE_ASSISTANT = \
"""
custom_fns:
  - null
  - PlaceNextToScrewdriver15cmFn
  - null
  - PlaceNextToScrewdriver15cmFn
  - null
  - PlaceNextToScrewdriver15cmFn

```
def PlaceNextToScrewdriver15cmFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluate if the object is placed 15cm next to the screwdriver.

    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed place primitive [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    # Get the object ID from the primitive.
    object_id = get_object_id_from_primitive(0, primitive)
    # Get the non-manipulated object IDs from the environment.
    screwdriver_id = get_object_id_from_name("screwdriver", env, primitive)
    # For the manipulated object, the state after placing the object is relevant. 
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    screwdriver_pose = get_pose(state, screwdriver_id, -1)
    # Evaluate if the object is placed 15cm next to the screwdriver.
    distance_metric = position_norm_metric(next_object_pose, screwdriver_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    ideal_point = 0.15
    upper_threshold = 0.20
    smaller_than_ideal_probability = linear_probability(
        distance_metric, lower_threshold, ideal_point, is_smaller_then=False
    )
    bigger_than_ideal_probability = linear_probability(
        distance_metric, ideal_point, upper_threshold, is_smaller_then=True
    )
    return probability_intersection(smaller_than_ideal_probability, bigger_than_ideal_probability)
```
"""