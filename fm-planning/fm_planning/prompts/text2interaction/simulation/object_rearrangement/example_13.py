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
1. Pick(blue_box, table)
2. Place(blue_box, table)
"""

EXAMPLE_USER = \
f"""{TASK_PREFIX}

Instruction:
The blue box should be oriented the same as the cyan box or the red box.
"""

EXAMPLE_ASSISTANT = \
"""
custom_fns:
  - null
  - OrientedSameAsCyanOrRedBoxFn

```
def OrientedSameAsCyanOrRedBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates the if the object is oriented the same as the cyan box or the red box.

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
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    # Evaluate if the object has the same orientation as the cyan box
    cyan_orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    cyan_orientation_probability = linear_probability(
        cyan_orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object has the same orientation as the red box
    red_orientation_metric = great_circle_distance_metric(next_object_pose, red_box_pose)
    red_orientation_probability = linear_probability(
        red_orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # combine the two probabilities with OR logic
    total_probability = probability_union(cyan_orientation_probability, red_orientation_probability)
    return total_probability
```
"""