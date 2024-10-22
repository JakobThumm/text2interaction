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
The cyan box should be close to the red box or close to the blue box."
"""

EXAMPLE_ASSISTANT = \
"""
custom_fns:
  - null
  - PlaceNextToRedOrBlueBoxFn

```
def PlaceNextToRedOrBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates if the object is placed next to the red box or the blue box.

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
    object_id = get_object_id_from_primitive(0, primitive)
    red_box_id = get_object_id_from_name("red_box", env, primitive)
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    red_box_pose = get_pose(state, red_box_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    # Evaluate if the object is placed at least 10cm next to the red box.
    distance_red_metric = position_norm_metric(next_object_pose, red_box_pose, norm="L2", axes=["x", "y"])
    lower_threshold = 0.10
    upper_threshold = 0.15
    close_to_red_probability = linear_probability(
        distance_red_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Evaluate if the object is placed at least 10cm next to the blue box.
    distance_blue_metric = position_norm_metric(next_object_pose, blue_box_pose, norm="L2", axes=["x", "y"])
    close_to_blue_probability = linear_probability(
        distance_blue_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the two probabilities with OR logic
    total_probability = probability_union(close_to_red_probability, close_to_blue_probability)
    return total_probability
```
"""
