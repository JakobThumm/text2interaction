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
The blue box and the red box have to be oriented in the same direction as the cyan box or orthogonal to the cyan box.
"""

EXAMPLE_ASSISTANT = \
"""
custom_fns:
  - null
  - OrientedSameOrOrthogonalToCyanBoxFn
  - null
  - OrientedSameOrOrthogonalToCyanBoxFn

```
def OrientedSameOrOrthogonalToCyanBoxFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluate if the object has the same orientation as the cyan box or is oriented orthogonal to the cyan box.

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
    object_id = get_object_id_from_primitive(0, primitive)
    next_object_pose = get_pose(next_state, object_id, -1)
    cyan_box_pose = get_pose(state, cyan_box_id, -1)
    # Calculate the orientational difference between the object and the cyan box
    orientation_metric = great_circle_distance_metric(next_object_pose, cyan_box_pose)
    lower_threshold = torch.pi / 8.0
    upper_threshold = torch.pi / 6.0
    # Calculate the probability that the object has the same orientation as the cyan box
    same_orientation_probability = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Orthogonal orientation is defined as the object orientation being pi/2 away from the cyan box orientation
    target_value = torch.pi / 2
    # Orthogonal orientation is symmetric around the cyan box orientation
    orthogonal_orientation_metric_1 = orientation_metric - target_value
    orthogonal_orientation_metric_2 = orientation_metric + target_value
    # Calculate the probability that the object has an orthogonal orientation to the cyan box
    orientation_orthogonal_probability_1 = linear_probability(
        orthogonal_orientation_metric_1, lower_threshold, upper_threshold, is_smaller_then=True
    )
    orientation_orthogonal_probability_2 = linear_probability(
        orthogonal_orientation_metric_2, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # Combine the two orthogonal orientation probabilities with OR logic
    orientation_orthogonal_probability = probability_union(
        orientation_orthogonal_probability_1, orientation_orthogonal_probability_2
    )
    # Combine the two probabilities with OR logic
    total_probability = probability_union(same_orientation_probability, orientation_orthogonal_probability)
    return total_probability
```
"""