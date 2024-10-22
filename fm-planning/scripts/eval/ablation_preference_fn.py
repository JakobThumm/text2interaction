from typing import Sequence, Union

import argparse
import pathlib


from fm_planning import models, utils
from fm_planning.prompts.text2interaction.simulation.object_rearrangement import (
    system_prompt,
    example_0,
    example_1,
    example_2,
    example_3,
    example_4,
    example_5,
    example_6,
    example_7,
    example_8,
    example_9,
    example_10,
    example_11,
    example_12,
    example_13,
    example_14,
)

EXAMPLES = [
    example_0,
    example_1,
    example_2,
    example_3,
    example_4,
    example_5,
    example_6,
    example_7,
    example_8,
    example_9,
    example_10,
    example_11,
    example_12,
    example_13,
    example_14,
]


def generate_preference_function(
    api_key: str,
    model_config: Union[str, pathlib.Path],
    examples: Sequence[int],
    eval_example: int,
    device: str = "auto",
    seed: int = 0,
    verbose: bool = True,
) -> str:
    """Run a demo."""
    # Set seed.
    utils.random.seed(seed)

    # Load model.
    model_factory = models.PretrainedModelFactory(
        model_config, api_key=api_key, device=device
    )
    model = model_factory()

    # Construct prompt.
    prompt = [{
                  "role": "system",
                  "content": system_prompt.SYSTEM_PROMPT
              }]
    for example in examples:
        prompt.append(
            {
                "role": "system",
                "name": "example_user",
                "content": EXAMPLES[example].EXAMPLE_USER
            }
        )
        prompt.append(
            {
                "role": "system",
                "name": "example_assistant",
                "content": EXAMPLES[example].EXAMPLE_ASSISTANT
            }
        )

    prompt.append(
        {
            "role": "user",
            "name": "user_instruction",
            "content": EXAMPLES[eval_example].EXAMPLE_USER +\
                "Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable)."
        }
    )
    prompt.append(
        {
            "role": "user",
            "name": "rules",
            "content": "1. Make sure that the output follows the example structure! custom_fns: [...]\n ```def [Function 0] [...]```\n ```def [Function 1] [...]```(if applicable).\n\
                        2. If an action does not need a custom function, add a `- null` entry to the custom_fns list to make sure the list is complete.\n\
                        3. Use the `linear_probability()` function over the `threshold_probability()` or `normal_probability()` functions when possible to improve performance of the planner."
        }
    )

    # Predict preference functions.
    if verbose:
        print(f"{prompt}")
    response = model.forward(prompt)
    preference_fns = response["choices"][0]["message"]["content"]
    if verbose:
        print(preference_fns)
    return preference_fns


def main(args: argparse.Namespace) -> None:
    generate_preference_function(**vars(args))


if __name__ == "__main__":
    # Defaults.
    model_config = "configs/models/pretrained/generative/gpt_4_cot.yaml"

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model-config", type=str, default=model_config)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", nargs="+", type=int, default=[1, 6, 10])
    parser.add_argument("--eval-example", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", default=False)
    main(parser.parse_args())
