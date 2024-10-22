from .system_prompt import TASK_PREFIX


EXAMPLE_USER = \
f"""{TASK_PREFIX}

Instruction:
When performing the handover, please ensure that the handle both points toward me and is sufficiently close to my hand. Ignore the Pick preference function for now."""


EXAMPLE_ASSISTANT = \
"""def ScrewdriverHandoverFn(
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None
) -> torch.Tensor:
    r\"\"\"Evaluates the orientation and position of the screwdriver handover.
    Args:
        state [batch_size, state_dim]: Current state.
        action [batch_size, action_dim]: Action.
        next_state [batch_size, state_dim]: Next state.
        primitive: optional primitive to receive the object orientation from
    Returns:
        Evaluation of the performed handover [batch_size] \in [0, 1].
    \"\"\"
    assert primitive is not None and isinstance(primitive, Primitive)
    env = primitive.env
    object_id = get_object_id_from_primitive(0, primitive)
    hand_id = get_object_id_from_primitive(1, primitive)
    next_object_pose = get_pose(next_state, object_id)
    current_hand_pose = get_pose(state, hand_id)
    handle_main_axis = [-1.0, 0.0, 0.0]
    # We want to know if the handle is pointing towards the hand position after the handover action.
    orientation_metric = pointing_in_direction_metric(next_object_pose, current_hand_pose, handle_main_axis)
    lower_threshold = torch.pi / 6.0
    upper_threshold = torch.pi / 4.0
    # Calculate the probability
    probability_handover_orientation = linear_probability(
        orientation_metric, lower_threshold, upper_threshold, is_smaller_then=True
    )
    # We want to be close to the human hand.
    position_metric = position_norm_metric(next_object_pose, current_hand_pose, norm="L2", axes=["x", "y", "z"])
    # Closeness is hard to achieve and we are okay with a small deviations, therefore we pick the normal distribution.
    mean = 0.0  # Closer to the hand is preferred
    std = 0.1  # standard deviation of 0.1 is considered a good handover
    # Calculate the probability for position
    probability_handover_position = normal_probability(position_metric, mean, std, is_smaller_then=True)
    total_probability = probability_intersection(probability_handover_position, probability_handover_orientation)
    return total_probability"""