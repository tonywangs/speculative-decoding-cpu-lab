"""Synthetic tasks; no external data or tokenizer required."""
import random

BOS, EOS, SEPARATOR = 0, 4, 21
TASKS = ("repetitive", "structured", "unpredictable")
PROMPT_LENGTH = 26
SEQUENCE_LENGTH = 51


def sequence(rng, task):
    if task == "repetitive":
        motif = [rng.randrange(5, 21) for _ in range(4)]
        payload = motif * 12
    elif task == "structured":
        payload = []
        for _ in range(16):
            key = rng.randrange(5, 13)
            payload.extend((key, key + 8, SEPARATOR))
    elif task == "unpredictable":
        payload = [rng.randrange(5, 21) for _ in range(48)]
    else:
        raise ValueError(task)
    return [BOS, TASKS.index(task) + 1, *payload, EOS]


def generate_data(seed=17, train_per_task=256, eval_per_task=12):
    if min(train_per_task, eval_per_task) < 1 or train_per_task + eval_per_task > 65536:
        raise ValueError("positive split sizes with total <= 65536 required")
    train, evaluation = [], []
    seen = set()
    # Separate PRNG streams and explicit prefix exclusion: no evaluation prompt
    # (nor full sequence) occurs in training, even on the periodic task.
    for split, count, target, split_seed in (
        ("train", train_per_task, train, seed),
        ("eval", eval_per_task, evaluation, seed + 1),
    ):
        rng = random.Random(split_seed)
        for task in TASKS:
            for i in range(count):
                while True:
                    tokens = sequence(rng, task)
                    prefix = tuple(tokens[:PROMPT_LENGTH])
                    if prefix not in seen:
                        seen.add(prefix)
                        break
                target.append({"id": f"{split}-{task}-{i:04d}", "task": task,
                               "tokens": tokens, "prompt_length": PROMPT_LENGTH})
    return {"seed": seed, "split_seeds": {"train": seed, "eval": seed + 1},
            "train": train, "eval": evaluation}
