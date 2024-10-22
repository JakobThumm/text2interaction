from typing import Union

import argparse
import pathlib
from copy import deepcopy

from fm_planning import models, utils
from fm_planning.prompts.text2interaction.real_world import (
    system_prompt,
    example_1,
    example_2,
    example_3,
)


OPENAI_PROMPT = [
    {
        "role": "system",
        "content": system_prompt.SYSTEM_PROMPT
    },
    {
        "role": "system",
        "name": "example_user",
        "content": example_1.EXAMPLE_USER
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": example_1.EXAMPLE_ASSISTANT
    },
    {
        "role": "system",
        "name": "example_user",
        "content": example_2.EXAMPLE_USER
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": example_2.EXAMPLE_ASSISTANT
    },
    {
        "role": "system",
        "name": "example_user",
        "content": example_3.EXAMPLE_USER
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": example_3.EXAMPLE_ASSISTANT
    },
]



def demo(
    api_key: str,
    model_config: Union[str, pathlib.Path],
    device: str = "auto",
    seed: int = 0,
) -> None:
    """Run a demo."""
    # Set seed.
    utils.random.seed(seed)

    # Load model.
    model_factory = models.PretrainedModelFactory(
        model_config, api_key=api_key, device=device
    )
    model = model_factory()

    # Construct prompt.
    instruction = input("User instruction: ")
    prompt = deepcopy(OPENAI_PROMPT)
    prompt.append(
        {
            "role": "user",
            "content": \
                f"""{system_prompt.TASK_PREFIX}
                
                Instruction: 
                {instruction}"""
        }
    )

    # Predict preference functions.
    response = model.forward(prompt)
    preference_fns = response["choices"][0]["message"]["content"]
    print(f"Preference Functions: {preference_fns}")


def main(args: argparse.Namespace) -> None:
    demo(**vars(args))


if __name__ == "__main__":
    # Defaults.
    model_config = "configs/models/pretrained/generative/gpt_4_cot.yaml"

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model-config", type=str, default=model_config)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
