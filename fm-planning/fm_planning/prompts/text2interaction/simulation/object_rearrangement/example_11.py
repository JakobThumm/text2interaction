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
"""

EXAMPLE_USER = \
f"""{TASK_PREFIX}

Instruction:
Place the red box in a straight line right of or behind the blue box.
"""

EXAMPLE_ASSISTANT = \
"""
custom_fns:
  - null
  - StraightRightOfOrBehindBlueBoxFn

```
def StraightRightOfOrBehindBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates if the object is placed in a straight line right of or behind the blue box.

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
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    blue_box_pose = get_pose(state, blue_box_id, -1)
    ## Evaluate if the object is placed in a straight line behind the blue box
    # Evaluate if the object is placed behind the blue box
    behind = [-1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, behind)
    lower_threshold = 0.0
    is_behind_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, behind)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the left or right.
    total_straight_front_probability = probability_intersection(is_behind_probability, normal_diff_probability)
    ## Evaluate if the object is placed in a straight line right of the blue box
    # Evaluate if the object is placed right of the blue box
    right = [0.0, 1.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, right)
    lower_threshold = 0.0
    is_right_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the front or back
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, right)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the left or right.
    total_straight_right_probability = probability_intersection(is_right_of_probability, normal_diff_probability)
    ## Combine the two probabilities with OR logic
    total_probability = probability_union(total_straight_front_probability, total_straight_right_probability)
    return total_probability
```
"""