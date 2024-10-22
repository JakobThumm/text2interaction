'''This script is used to create the files necessary for the cross evaluation of the ablation study of Text2Interaction.

Author: Jakob Thumm
Date: 2024-04-22
'''

import argparse
import numpy as np
import os
from typing import Sequence, Tuple
from ablation_preference_fn import generate_preference_function, EXAMPLES

TEST_PREFERENCE_OUTPUT = 'custom_fns:\n\
  - null\n\
  - PlaceInLineWithRedAndBlueBoxFn\n\
\n\
```python\n\
def PlaceInLineWithRedAndBlueBoxFn(\n\
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None\n\
) -> torch.Tensor:\n\
    r"""Evaluates if the object is placed in line with the red and blue box.\n\
\n\
    Args:\n\
        state [batch_size, state_dim]: Current state.\n\
        action [batch_size, action_dim]: Action.\n\
        next_state [batch_size, state_dim]: Next state.\n\
        primitive: optional primitive to receive the object orientation from\n\
\n\
    Returns:\n\
        Evaluation of the performed handover [batch_size] \in [0, 1].\n\
    """\n\
    assert primitive is not None and isinstance(primitive, Primitive)\n\
    env = primitive.env\n\
    object_id = get_object_id_from_primitive(0, primitive)\n\
    red_box_id = get_object_id_from_name("red_box", env, primitive)\n\
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)\n\
    next_object_pose = get_pose(next_state, object_id, -1)\n\
    red_box_pose = get_pose(state, red_box_id, -1)\n\
    blue_box_pose = get_pose(state, blue_box_id, -1)\n\
    red_to_blue = build_direction_vector(red_box_pose, blue_box_pose)\n\
    normal_distance_metric = position_metric_normal_to_direction(next_object_pose, blue_box_pose, red_to_blue)\n\
    lower_threshold = 0.0\n\
    upper_threshold = 0.05\n\
    # The x difference should be as small as possible but no larger than 5cm.\n\
    probability = linear_probability(normal_distance_metric, lower_threshold, upper_threshold, is_smaller_then=True)\n\
    return probability\n\
```\n\
'

TEST_PREFERENCE_OUTPUT_2 = 'custom_fns:\n\
  - null\n\
  - NextToBlueBoxFn\n\
  - null\n\
  - FarFromBlueAndRedBoxFn\n\
\n\
```python\n\
def NextToBlueBoxFn(\n\
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None\n\
) -> torch.Tensor:\n\
    r"""Evaluate if the red box is placed next to the blue box.\n\
\n\
    Args:\n\
        state [batch_size, state_dim]: Current state.\n\
        action [batch_size, action_dim]: Action.\n\
        next_state [batch_size, state_dim]: Next state.\n\
        primitive: optional primitive to receive the object orientation from\n\
    Returns:\n\
        Evaluation of the performed handover [batch_size] \in [0, 1].\n\
    """\n\
    assert primitive is not None and isinstance(primitive, Primitive)\n\
    env = primitive.env\n\
    red_box_id = get_object_id_from_primitive(0, primitive)\n\
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)\n\
    next_red_box_pose = get_pose(next_state, red_box_id, -1)\n\
    blue_box_pose = get_pose(state, blue_box_id, -1)\n\
    # Evaluate if the red box is placed next to the blue box.\n\
    distance_metric = position_norm_metric(next_red_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])\n\
    ideal_distance = 0.05  # Assuming next to means within 5cm\n\
    return threshold_probability(distance_metric, ideal_distance, is_smaller_then=True)\n\
```\n\
\n\
```python\n\
def FarFromBlueAndRedBoxFn(\n\
    state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, primitive: Optional[Primitive] = None\n\
) -> torch.Tensor:\n\
    r"""Evaluate if the cyan box is placed as far away as possible from both the blue and red boxes.\n\
\n\
    Args:\n\
        state [batch_size, state_dim]: Current state.\n\
        action [batch_size, action_dim]: Action.\n\
        next_state [batch_size, state_dim]: Next state.\n\
        primitive: optional primitive to receive the object orientation from\n\
    Returns:\n\
        Evaluation of the performed handover [batch_size] \in [0, 1].\n\
    """\n\
    assert primitive is not None and isinstance(primitive, Primitive)\n\
    env = primitive.env\n\
    cyan_box_id = get_object_id_from_primitive(0, primitive)\n\
    blue_box_id = get_object_id_from_name("blue_box", env, primitive)\n\
    red_box_id = get_object_id_from_name("red_box", env, primitive)\n\
    next_cyan_box_pose = get_pose(next_state, cyan_box_id, -1)\n\
    blue_box_pose = get_pose(state, blue_box_id, -1)\n\
    red_box_pose = get_pose(state, red_box_id, -1)\n\
    # Evaluate distance from both boxes and encourage maximizing this distance.\n\
    distance_to_blue = position_norm_metric(next_cyan_box_pose, blue_box_pose, norm="L2", axes=["x", "y"])\n\
    distance_to_red = position_norm_metric(next_cyan_box_pose, red_box_pose, norm="L2", axes=["x", "y"])\n\
    # Assuming far away means more than 20cm from either box\n\
    min_distance_threshold = 0.20\n\
    probability_from_blue = threshold_probability(distance_to_blue, min_distance_threshold, is_smaller_then=False)\n\
    probability_from_red = threshold_probability(distance_to_red, min_distance_threshold, is_smaller_then=False)\n\
    # Combine probabilities to ensure cyan box is far from both boxes\n\
    return probability_intersection(probability_from_blue, probability_from_red)\n\
```\n\
'


def generate_cross_eval(n_trials: int, n_examples: int, seed: int = 0, loop_until_each_example_used: bool = False) -> Sequence[Tuple[int, Sequence[int]]]:
    """Generate the cross evaluation trials."""
    assert n_trials > 0
    assert n_examples > 0
    eval_list = []
    trials_used = [0 for _ in range(len(EXAMPLES))]
    np.random.seed(seed)
    for i in range(n_trials):
        eval_trial = []
        eval_id = i % len(EXAMPLES)
        example_array = np.arange(len(EXAMPLES))
        example_array = np.delete(example_array, eval_id)
        random_examples = np.random.choice(example_array, n_examples, False)
        for example in random_examples:
            eval_trial.append(example)
            trials_used[example] += 1
        tuple_trial = (eval_id, eval_trial)
        eval_list.append(tuple_trial)
    if not loop_until_each_example_used or np.all(np.array(trials_used) > 0):
        return eval_list
    print("Not all examples were used at least once.")
    return generate_cross_eval(n_trials, n_examples)


def split_preference_output_string(preference_output: str) -> Sequence[str]:
    """Split the preference output string into the components config, [functions]."""
    strings = preference_output.split("```")
    output_strings = [strings[0]]
    for i in range(1, len(strings)-1):
        pos_def = strings[i].find("def ")
        if pos_def == -1:
            continue
        output_strings.append(strings[i][pos_def:])
    return output_strings


def get_custom_fn_name(custom_fn: str) -> str:
    """Return the name of the custom function."""
    return custom_fn.split('(')[0].split('def ')[1]


def add_trial_number_to_custom_fn(custom_fn_list: str, custom_fns: Sequence[str], trial_id: int) -> Tuple[str, Sequence[str]]:
    """Add the trial number to the custom functions.

    Example: Trial 0:
    custom_fn_list = "custom_fns:
      - null
      - AlignBoxesInLineFn"
    custom_fns[0] = "def AlignBoxesInLineFn(
      ...
      "
    -->
    Output
    custom_fn_list = "custom_fns:
      - null
      - AlignBoxesInLineFn_trial_0"
    custom_fns[0] = "def AlignBoxesInLineFn_trial_0(
      ...
      "
    """
    new_custom_fn_list = ""
    custom_fn_list_lines = custom_fn_list.split("\n")
    for line in custom_fn_list_lines:
        if line.strip().startswith("-") and line.strip().find("null") == -1:
            new_custom_fn_list += f"{line}_trial_{trial_id}\n"
        else:
            new_custom_fn_list += f"{line}\n"
    new_custom_fns = []
    for custom_fn in custom_fns:
        custom_fn_name = get_custom_fn_name(custom_fn)
        new_custom_fns.append(custom_fn.replace(custom_fn_name, f"{custom_fn_name}_trial_{trial_id}"))
    return new_custom_fn_list, new_custom_fns


def get_config_file_in(stap_path: str, eval_id: int) -> str:
    """Return the path to the original config file.

    The file is located in {stap_path}/configs/pybullet/envs/official/sim_domains/object_arrangement/ablation_task_{eval_id}.yaml.
    """
    file_name = f"{stap_path}/configs/pybullet/envs/official/sim_domains/object_arrangement/ablation_task_{eval_id}.yaml"
    assert os.path.exists(file_name)
    return file_name


def get_config_file_out(stap_path: str, trial_id: int) -> str:
    """Return the path to the original config file.

    The file is located in {stap_path}/configs/pybullet/envs/official/sim_domains/object_arrangement/ablation_task_{eval_id}.yaml.
    """
    return f"{stap_path}/configs/pybullet/envs/official/sim_domains/object_arrangement/generated_ablation_trial_{trial_id}.yaml"


def get_custom_preference_function_file(stap_path: str) -> str:
    """Return the path to the original config file.

    The file is located in {stap_path}/stap/planners/custom_fns.py.
    """
    return f"{stap_path}/stap/planners/custom_fns.py"


def write_config_file(config_file_in: str, config_file_out: str, custom_fns_list_str: str) -> None:
    """Write the custom functions to the config file."""
    with open(config_file_in, "r") as f_in:
        with open(config_file_out, "w") as f_out:
            for line in f_in:
                if line.strip().find("instruction:") != -1:
                    f_out.write(line)
                    custom_fns_list_lines = custom_fns_list_str.split("\n")
                    for custom_fns_list_line in custom_fns_list_lines:
                        f_out.write(f"      {custom_fns_list_line}\n")
                else:
                    f_out.write(line)


def add_custom_preference_functions_to_function_file(custom_preference_function_file: str, custom_fn: str) -> None:
    """Add the custom functions to the custom preference function file."""
    with open(custom_preference_function_file, "r") as f:
        lines = f.readlines()
    with open(custom_preference_function_file, "w") as f:
        for line in lines:
            if line.strip() == "CUSTOM_FNS = {":
                f.write(custom_fn)
                f.write("\n\n")
                f.write(line)
                # Find name between 'def ' and the first '('
                custom_fn_name = get_custom_fn_name(custom_fn)
                f.write(f"    \"{custom_fn_name}\": {custom_fn_name},\n")
            else:
                f.write(line)


def main(args: argparse.Namespace) -> None:
    eval_list = generate_cross_eval(
        args.n_trials,
        args.n_examples,
        args.seed,
        args.n_trials * args.n_examples > 2 * len(EXAMPLES)
    )
    for i in range(len(eval_list)):
        print(f"Trial {i} with eval function {eval_list[i][0]} and examples {eval_list[i][1]}.")
        preference_output = generate_preference_function(
            api_key=args.api_key,
            model_config=args.model_config,
            examples=eval_list[i][1],
            eval_example=eval_list[i][0],
            device=args.device,
            seed=args.seed,
            verbose=False
        )
        # preference_output = TEST_PREFERENCE_OUTPUT
        print(preference_output)
        # Get the file names
        config_file_in = get_config_file_in(args.stap_path, eval_list[i][0])
        config_file_out = get_config_file_out(args.stap_path, i)
        custom_preference_function_file = get_custom_preference_function_file(args.stap_path)
        # Split the preference output
        preference_output_split = split_preference_output_string(preference_output)
        # Add trial number to custom functions
        custom_fns_list_str, custom_fns = add_trial_number_to_custom_fn(preference_output_split[0], preference_output_split[1:], i)
        # Write the config file
        write_config_file(config_file_in, config_file_out, custom_fns_list_str)
        # Add custom preference functions to function file
        for custom_fn in custom_fns:
            add_custom_preference_functions_to_function_file(custom_preference_function_file, custom_fn)


if __name__ == "__main__":
    # Defaults.
    model_config = "configs/models/pretrained/generative/gpt_4_cot.yaml"

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--stap-path", type=str, required=True)
    parser.add_argument("--model-config", type=str, default=model_config)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=1)
    main(parser.parse_args())
