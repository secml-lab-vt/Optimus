import sys, os
_LM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LM_ROOT not in sys.path:
    sys.path.insert(0, _LM_ROOT)


def prompt_generator(model_name, text):
    """Generates a prompt for the language model."""


    zero_shot_prompt = f"""[INST] <<SYS>>
        {system_prompt}
        <</SYS>>

        {sys_command}
        {inference_example}

        {questions[index]} [/INST]"""

    return f"Prompt: {text}\n"