"""Build action-sequence training data for the Dancing With My Code chatbot.

The model never sees raw text. The website classifies user text into an input
Action + language, then this model predicts output Action IDs.

Simplified vocabulary (v4): 15 tokens, 9 output actions.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary constants (must match action_engine.rs)
# ---------------------------------------------------------------------------

PAD = 0
SOS = 1
EOS = 2
SEP = 3

UNKNOWN = 4
GREETING = 5
FAREWELL = 6
INSULT = 7
PROGRAMMING = 8
IDENTITY = 9
RESUME = 10
LINKS = 11
COURSE = 12

LANG_TH = 13
LANG_EN = 14
VOCAB_SIZE = 15

ALL_OUTPUT_ACTIONS = [
    GREETING,
    FAREWELL,
    INSULT,
    PROGRAMMING,
    IDENTITY,
    RESUME,
    LINKS,
    COURSE,
    UNKNOWN,
]

NON_FAREWELL_ACTIONS = [a for a in ALL_OUTPUT_ACTIONS if a != FAREWELL]

# ---------------------------------------------------------------------------
# Output-action patterns keyed by input action
# ---------------------------------------------------------------------------

BASE_PATTERNS: dict[int, list[list[int]]] = {
    GREETING: [
        [GREETING, EOS],
        [GREETING, INSULT, EOS],
        [GREETING, PROGRAMMING, EOS],
        [GREETING, IDENTITY, EOS],
        [GREETING, LINKS, EOS],
        [GREETING, RESUME, EOS],
        [GREETING, COURSE, EOS],
        [GREETING, UNKNOWN, EOS],
        [GREETING, IDENTITY, PROGRAMMING, EOS],
    ],
    FAREWELL: [
        [FAREWELL, EOS],
        [GREETING, FAREWELL, EOS],
        [FAREWELL, GREETING, EOS],
    ],
    INSULT: [
        [INSULT, INSULT, EOS],
        [INSULT, PROGRAMMING, EOS],
        [INSULT, IDENTITY, EOS],
        [INSULT, UNKNOWN, EOS],
        [INSULT, INSULT, PROGRAMMING, EOS],
        [INSULT, INSULT, INSULT, EOS],
        [INSULT, LINKS, EOS],
    ],
    PROGRAMMING: [
        [PROGRAMMING, INSULT, EOS],
        [PROGRAMMING, PROGRAMMING, EOS],
        [PROGRAMMING, IDENTITY, EOS],
        [PROGRAMMING, INSULT, PROGRAMMING, EOS],
        [PROGRAMMING, UNKNOWN, EOS],
        [PROGRAMMING, PROGRAMMING, INSULT, EOS],
        [PROGRAMMING, LINKS, EOS],
        [PROGRAMMING, COURSE, EOS],
    ],
    IDENTITY: [
        [IDENTITY, PROGRAMMING, EOS],
        [IDENTITY, GREETING, EOS],
        [IDENTITY, INSULT, EOS],
        [IDENTITY, RESUME, EOS],
        [IDENTITY, LINKS, EOS],
        [IDENTITY, COURSE, EOS],
        [IDENTITY, UNKNOWN, EOS],
    ],
    RESUME: [
        [RESUME, GREETING, EOS],
        [RESUME, IDENTITY, EOS],
        [RESUME, PROGRAMMING, EOS],
        [RESUME, UNKNOWN, EOS],
    ],
    LINKS: [
        [LINKS, GREETING, EOS],
        [LINKS, IDENTITY, EOS],
        [LINKS, PROGRAMMING, EOS],
        [LINKS, UNKNOWN, EOS],
    ],
    COURSE: [
        [COURSE, PROGRAMMING, EOS],
        [COURSE, IDENTITY, EOS],
        [COURSE, GREETING, EOS],
    ],
    UNKNOWN: [
        [UNKNOWN, GREETING, EOS],
        [UNKNOWN, INSULT, EOS],
        [UNKNOWN, UNKNOWN, EOS],
        [UNKNOWN, PROGRAMMING, EOS],
        [UNKNOWN, IDENTITY, EOS],
    ],
}

MULTI_TURN_TEMPLATES: list[tuple[list[int], int, list[int]]] = [
    ([GREETING, EOS], INSULT, [INSULT, INSULT, EOS]),
    ([GREETING, IDENTITY, EOS], PROGRAMMING, [PROGRAMMING, INSULT, EOS]),
    ([GREETING, PROGRAMMING, EOS], IDENTITY, [IDENTITY, RESUME, EOS]),
    ([INSULT, INSULT, EOS], GREETING, [GREETING, UNKNOWN, EOS]),
    ([INSULT, PROGRAMMING, EOS], INSULT, [INSULT, INSULT, INSULT, EOS]),
    ([PROGRAMMING, INSULT, EOS], PROGRAMMING, [PROGRAMMING, PROGRAMMING, EOS]),
    ([PROGRAMMING, PROGRAMMING, EOS], IDENTITY, [IDENTITY, PROGRAMMING, EOS]),
    ([IDENTITY, PROGRAMMING, EOS], RESUME, [RESUME, IDENTITY, EOS]),
    ([IDENTITY, RESUME, EOS], COURSE, [COURSE, PROGRAMMING, EOS]),
    ([COURSE, PROGRAMMING, EOS], LINKS, [LINKS, IDENTITY, EOS]),
    ([RESUME, GREETING, EOS], FAREWELL, [GREETING, FAREWELL, EOS]),
    ([LINKS, GREETING, EOS], PROGRAMMING, [PROGRAMMING, INSULT, EOS]),
    ([COURSE, GREETING, EOS], PROGRAMMING, [PROGRAMMING, PROGRAMMING, EOS]),
    ([UNKNOWN, GREETING, EOS], IDENTITY, [IDENTITY, PROGRAMMING, EOS]),
    ([GREETING, INSULT, EOS], INSULT, [INSULT, PROGRAMMING, INSULT, EOS]),
    ([PROGRAMMING, IDENTITY, EOS], INSULT, [INSULT, INSULT, EOS]),
    ([IDENTITY, LINKS, EOS], UNKNOWN, [UNKNOWN, GREETING, EOS]),
    ([INSULT, UNKNOWN, EOS], GREETING, [GREETING, IDENTITY, EOS]),
    ([LINKS, IDENTITY, EOS], GREETING, [GREETING, PROGRAMMING, EOS]),
    ([PROGRAMMING, COURSE, EOS], LINKS, [LINKS, GREETING, EOS]),
]

# Extra weight for personality-heavy topics.
TOPIC_WEIGHTS = {
    GREETING: 1.5,
    INSULT: 3.0,
    PROGRAMMING: 2.8,
    IDENTITY: 2.0,
    RESUME: 1.5,
    LINKS: 1.2,
    COURSE: 1.5,
    FAREWELL: 0.5,
    UNKNOWN: 1.2,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_single_turn_row(action: int, lang: int, output_seq: list[int]) -> str:
    return " ".join(str(t) for t in [action, lang, SEP] + output_seq)


def make_multi_turn_row(
    context: list[int], action: int, lang: int, output_seq: list[int]
) -> str:
    return " ".join(str(t) for t in list(context) + [action, lang, SEP] + output_seq)


def random_output_action(rng: random.Random, allow_farewell: bool = False) -> int:
    pool = ALL_OUTPUT_ACTIONS if allow_farewell else NON_FAREWELL_ACTIONS
    return rng.choice(pool)


def perturb_output(seq: list[int], rng: random.Random, max_swaps: int = 1) -> list[int]:
    if len(seq) <= 2:
        return list(seq)

    body = list(seq[:-1])
    for _ in range(rng.randint(0, max_swaps)):
        if len(body) < 2:
            break
        i = rng.randint(0, len(body) - 2)
        if FAREWELL not in (body[i], body[i + 1]):
            body[i], body[i + 1] = body[i + 1], body[i]

    return body + [EOS]


def extend_output(seq: list[int], rng: random.Random, action: int) -> list[int]:
    body = list(seq[:-1])
    allow_farewell = action == FAREWELL
    insert = random_output_action(rng, allow_farewell=allow_farewell)
    if not allow_farewell and insert == FAREWELL:
        insert = UNKNOWN
    body.insert(rng.randint(0, len(body)), insert)
    return body + [EOS]


def trim_output(seq: list[int], rng: random.Random) -> list[int]:
    if len(seq) <= 3:
        return list(seq)
    body = list(seq[:-1])
    body.pop(rng.randint(0, len(body) - 1))
    if not body:
        body.append(UNKNOWN)
    return body + [EOS]


def random_context(rng: random.Random, max_len: int = 4) -> list[int]:
    length = rng.randint(1, max_len)
    return [rng.choice(NON_FAREWELL_ACTIONS) for _ in range(length)] + [EOS]


def weighted_actions(rng: random.Random) -> int:
    actions = list(BASE_PATTERNS.keys())
    weights = [TOPIC_WEIGHTS.get(action, 1.0) for action in actions]
    return rng.choices(actions, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_base_rows(rng: random.Random) -> list[str]:
    rows: list[str] = []
    for lang in [LANG_TH, LANG_EN]:
        for action, output_seqs in BASE_PATTERNS.items():
            copies = max(8, int(12 * TOPIC_WEIGHTS.get(action, 1.0)))
            for seq in output_seqs:
                for _ in range(copies * 4):
                    rows.append(make_single_turn_row(action, lang, list(seq)))
    return rows


def generate_multi_turn_rows(
    rng: random.Random, lang: int, copies: int = 30
) -> list[str]:
    rows: list[str] = []
    for context, action, output_seq in MULTI_TURN_TEMPLATES:
        for _ in range(copies):
            seq = list(output_seq)
            if rng.random() < 0.4:
                seq = perturb_output(seq, rng)
            rows.append(make_multi_turn_row(context, action, lang, seq))
    return rows


def generate_random_multi_turn_rows(
    rng: random.Random, lang: int, count: int
) -> list[str]:
    rows: list[str] = []
    for _ in range(count):
        action = weighted_actions(rng)
        seq = list(rng.choice(BASE_PATTERNS[action]))
        if rng.random() < 0.3:
            seq = perturb_output(seq, rng)
        if rng.random() < 0.15:
            seq = extend_output(seq, rng, action)
        rows.append(make_multi_turn_row(random_context(rng), action, lang, seq))
    return rows


def generate_topic_drill_rows(rng: random.Random, lang: int, count: int) -> list[str]:
    topic_actions = list(BASE_PATTERNS.keys())
    rows: list[str] = []
    for _ in range(count):
        action = rng.choice(topic_actions)
        seq = list(rng.choice(BASE_PATTERNS[action]))
        if action != FAREWELL and FAREWELL in seq:
            seq = [UNKNOWN if token == FAREWELL else token for token in seq]
        rows.append(make_single_turn_row(action, lang, seq))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "examples" / "action-data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build action-sequence training data for the chatbot model."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="Directory to write train.txt / val.txt",
    )
    parser.add_argument("--seed", type=int, default=20260525, help="Random seed")
    parser.add_argument("--offline", action="store_true", help="Compatibility no-op")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[str] = []

    base = generate_base_rows(rng)
    all_rows.extend(base)
    print(f"  base patterns:       {len(base):>5} rows")

    for lang, label in [(LANG_TH, "TH"), (LANG_EN, "EN")]:
        mt = generate_multi_turn_rows(rng, lang, copies=30)
        all_rows.extend(mt)
        print(f"  multi-turn {label}:      {len(mt):>5} rows")

    for lang, label in [(LANG_TH, "TH"), (LANG_EN, "EN")]:
        random_rows = generate_random_multi_turn_rows(rng, lang, count=500)
        all_rows.extend(random_rows)
        print(f"  random-multi {label}:   {len(random_rows):>5} rows")

    for lang, label in [(LANG_TH, "TH"), (LANG_EN, "EN")]:
        topic_rows = generate_topic_drill_rows(rng, lang, count=500)
        all_rows.extend(topic_rows)
        print(f"  topic-drill {label}:    {len(topic_rows):>5} rows")

    rng.shuffle(all_rows)
    total = len(all_rows)
    val_count = max(500, int(total * 0.15))
    val_count = min(val_count, total - 100)

    val_rows = all_rows[:val_count]
    train_rows = all_rows[val_count:]

    train_path = data_dir / "train.txt"
    val_path = data_dir / "val.txt"
    train_path.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_rows) + "\n", encoding="utf-8")

    print()
    print(f"  train.txt: {len(train_rows):>5} rows  ({train_path})")
    print(f"  val.txt:   {len(val_rows):>5} rows  ({val_path})")
    print(f"  total:     {total:>5} rows")
    print()

    for name, rows in [("train", train_rows), ("val", val_rows)]:
        lengths = [len(r.split()) for r in rows]
        vocab = set()
        farewell_bad = 0
        for row in rows:
            ids = [int(t) for t in row.split()]
            vocab.update(ids)
            sep_index = ids.index(SEP) if SEP in ids else 2
            input_action = ids[sep_index - 2] if sep_index >= 2 else ids[0]
            output = ids[sep_index + 1 :] if SEP in ids else []
            if input_action != FAREWELL and FAREWELL in output:
                farewell_bad += 1
        print(
            f"  {name}: seq_len min={min(lengths)} max={max(lengths)} "
            f"avg={sum(lengths) / len(lengths):.1f} vocab_ids={sorted(vocab)} "
            f"bad_farewell={farewell_bad}"
        )
        if max(vocab) >= VOCAB_SIZE:
            raise SystemExit(f"{name}: vocab id exceeds VOCAB_SIZE={VOCAB_SIZE}")
        if farewell_bad:
            raise SystemExit(f"{name}: found non-farewell rows that output Farewell")

    print("\nDone.")


if __name__ == "__main__":
    main()
