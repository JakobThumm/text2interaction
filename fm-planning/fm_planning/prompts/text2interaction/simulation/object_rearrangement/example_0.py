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
1. Pick(cyan_box, table)
2. Place(cyan_box, table)
"""

EXAMPLE_USER = \
f"""{TASK_PREFIX}

Instruction:
Make sure that the red box, the blue box, and the cyan box are in a line on the table.
"""

EXAMPLE_ASSISTANT = \
"""
custom_fns:
  - null
  - PlaceInLineWithRedAndBlueBoxFn

```
def PlaceInLineWithRedAndBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates if the object is placed in line with the red and blue box.

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
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    # For the manipulated object, the state after placing the object is relevant. 
    next_object_pose = get_pose(next_state, object_id, -1)
    # For the non-manipulated objects, the current state is more reliable.
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)
    normal_distance_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, red_to_blue)
    # The x difference should be as small as possible but no larger than 5cm (close).
    lower_threshold = 0.0
    upper_threshold = 0.05
    probability = linear_probability(normal_distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)
    return probability
```
"""