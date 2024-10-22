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
3. Pick(red_box, table)
4. Place(red_box, table)
"""

EXAMPLE_USER = \
f"""{TASK_PREFIX}

Instruction:
Arrange the three boxes in a straight line. The blue box should be in front of the cyan box and the red box should be in front of the blue box.
"""

EXAMPLE_ASSISTANT = \
"""
custom_fns:
  - null
  - StraightInFrontOfCyanBoxFn
  - null
  - StraightInFrontOfBlueBoxFn

```
def StraightInFrontOfCyanBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates if the object is placed in a straight line in front of the cyan box.

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
    cyan_box_id = get_object_id_from_name("cyan_box", env, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Evaluate if the object is placed in front of the cyan box
    in_front_of = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, cyan_box_pose, in_front_of)
    lower_threshold = 0.0
    is_in_front_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, cyan_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the cyan box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_of_probability, normal_diff_probability)
    return total_probability
```

```
def StraightInFrontOfBlueBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates if the object is placed in a straight line in front of the blue box.

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
    # Evaluate if the object is placed in front of the blue box
    in_front_of = [1.0, 0.0, 0.0]
    direction_difference = position_diff_along_direction(next_object_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    is_in_front_of_probability = threshold_probability(direction_difference, lower_threshold, is_smaller_then=False)
    # Evaluate the deviation to the left or right
    normal_to_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, in_front_of)
    lower_threshold = 0.0
    upper_threshold = 0.05
    normal_diff_probability = linear_probability(
        normal_to_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # The object should be in front of the blue box *and* not deviate too much to the left or right.
    total_probability = probability_intersection(is_in_front_of_probability, normal_diff_probability)
    return total_probability
```
"""
