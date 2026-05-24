"""Build action-sequence training data for the Dancing With My Code chatbot.

The model never sees raw text. The website classifies user text into an input
Action + language, then this model predicts output Action IDs.

HF datasets used as style/topic references, not copied into the repo:
  - li2017dailydialog/daily_dialog: English daily-life conversation shape.
  - pythainlp/oasst2_thai_top1_chat_format: Thai chat-format examples.
  - ZombitX64/ThaiChatbotConversation: Thai chatbot conversation coverage.
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
FRUSTRATED = 7
SAD = 8
HAPPY = 9
QUESTION = 10
INSULT = 11
COMPLIMENT = 12
AGREE = 13
DISAGREE = 14
GENERAL = 15
EATING = 16
DAILY_LIFE = 17
RUST_GO = 18
IDENTITY = 19
SHIT_TALK = 20

LANG_TH = 21
LANG_EN = 22
VOCAB_SIZE = 23

ALL_OUTPUT_ACTIONS = [
    GREETING,
    FAREWELL,
    FRUSTRATED,
    SAD,
    HAPPY,
    QUESTION,
    INSULT,
    COMPLIMENT,
    AGREE,
    DISAGREE,
    GENERAL,
    EATING,
    DAILY_LIFE,
    RUST_GO,
    IDENTITY,
    SHIT_TALK,
]

NON_FAREWELL_ACTIONS = [a for a in ALL_OUTPUT_ACTIONS if a != FAREWELL]

# ---------------------------------------------------------------------------
# Output-action patterns keyed by input action
# ---------------------------------------------------------------------------

BASE_PATTERNS: dict[int, list[list[int]]] = {
    GREETING: [
        [GREETING, DAILY_LIFE, EOS],
        [GREETING, EATING, EOS],
        [GREETING, QUESTION, EOS],
        [GREETING, SHIT_TALK, EOS],
        [GREETING, DAILY_LIFE, QUESTION, EOS],
    ],
    FAREWELL: [
        [FAREWELL, EOS],
        [FAREWELL, HAPPY, EOS],
        [GENERAL, FAREWELL, EOS],
    ],
    FRUSTRATED: [
        [FRUSTRATED, DAILY_LIFE, EOS],
        [FRUSTRATED, QUESTION, EOS],
        [FRUSTRATED, SHIT_TALK, EOS],
        [SHIT_TALK, FRUSTRATED, EOS],
    ],
    SAD: [
        [SAD, DAILY_LIFE, EOS],
        [SAD, EATING, EOS],
        [SAD, QUESTION, EOS],
        [SAD, HAPPY, EOS],
    ],
    HAPPY: [
        [HAPPY, COMPLIMENT, EOS],
        [HAPPY, DAILY_LIFE, EOS],
        [HAPPY, RUST_GO, EOS],
        [HAPPY, SHIT_TALK, EOS],
    ],
    QUESTION: [
        [QUESTION, GENERAL, EOS],
        [QUESTION, DAILY_LIFE, EOS],
        [QUESTION, QUESTION, EOS],
        [GENERAL, QUESTION, EOS],
    ],
    INSULT: [
        [INSULT, SHIT_TALK, EOS],
        [FRUSTRATED, INSULT, EOS],
        [SHIT_TALK, INSULT, EOS],
        [INSULT, EATING, EOS],
        [INSULT, FRUSTRATED, SHIT_TALK, EOS],
    ],
    COMPLIMENT: [
        [COMPLIMENT, HAPPY, EOS],
        [HAPPY, COMPLIMENT, EOS],
        [COMPLIMENT, SHIT_TALK, EOS],
    ],
    AGREE: [
        [AGREE, GENERAL, EOS],
        [AGREE, DAILY_LIFE, EOS],
        [AGREE, HAPPY, EOS],
    ],
    DISAGREE: [
        [DISAGREE, QUESTION, EOS],
        [DISAGREE, SHIT_TALK, EOS],
        [DISAGREE, GENERAL, EOS],
    ],
    GENERAL: [
        [GENERAL, DAILY_LIFE, EOS],
        [GENERAL, QUESTION, EOS],
        [GENERAL, EATING, EOS],
        [GENERAL, SHIT_TALK, EOS],
    ],
    EATING: [
        [EATING, DAILY_LIFE, EOS],
        [EATING, HAPPY, EOS],
        [EATING, SHIT_TALK, EOS],
        [EATING, QUESTION, EOS],
    ],
    DAILY_LIFE: [
        [DAILY_LIFE, EATING, EOS],
        [DAILY_LIFE, QUESTION, EOS],
        [DAILY_LIFE, SHIT_TALK, EOS],
        [DAILY_LIFE, HAPPY, EOS],
    ],
    RUST_GO: [
        [RUST_GO, SHIT_TALK, EOS],
        [RUST_GO, HAPPY, EOS],
        [RUST_GO, INSULT, EOS],
        [RUST_GO, QUESTION, EOS],
        [RUST_GO, RUST_GO, SHIT_TALK, EOS],
    ],
    IDENTITY: [
        [IDENTITY, GENERAL, EOS],
        [IDENTITY, HAPPY, EOS],
        [IDENTITY, DAILY_LIFE, EOS],
        [IDENTITY, SHIT_TALK, EOS],
    ],
    SHIT_TALK: [
        [SHIT_TALK, INSULT, EOS],
        [SHIT_TALK, FRUSTRATED, EOS],
        [SHIT_TALK, GENERAL, EOS],
        [SHIT_TALK, RUST_GO, EOS],
    ],
}

MULTI_TURN_TEMPLATES: list[tuple[list[int], int, list[int]]] = [
    ([GREETING, DAILY_LIFE, EOS], EATING, [EATING, SHIT_TALK, EOS]),
    ([GREETING, EATING, EOS], DAILY_LIFE, [DAILY_LIFE, QUESTION, EOS]),
    ([GREETING, SHIT_TALK, EOS], INSULT, [INSULT, SHIT_TALK, EOS]),
    ([DAILY_LIFE, QUESTION, EOS], RUST_GO, [RUST_GO, SHIT_TALK, EOS]),
    ([RUST_GO, SHIT_TALK, EOS], DISAGREE, [DISAGREE, RUST_GO, EOS]),
    ([RUST_GO, INSULT, EOS], AGREE, [AGREE, HAPPY, EOS]),
    ([INSULT, SHIT_TALK, EOS], GREETING, [GREETING, DAILY_LIFE, EOS]),
    ([INSULT, EATING, EOS], EATING, [EATING, HAPPY, EOS]),
    ([SAD, EATING, EOS], DAILY_LIFE, [DAILY_LIFE, HAPPY, EOS]),
    ([HAPPY, RUST_GO, EOS], SHIT_TALK, [SHIT_TALK, RUST_GO, EOS]),
    ([IDENTITY, GENERAL, EOS], QUESTION, [QUESTION, DAILY_LIFE, EOS]),
    ([IDENTITY, SHIT_TALK, EOS], INSULT, [INSULT, SHIT_TALK, EOS]),
    ([GENERAL, EATING, EOS], FAREWELL, [FAREWELL, EOS]),
    ([DAILY_LIFE, HAPPY, EOS], FAREWELL, [FAREWELL, HAPPY, EOS]),
    ([QUESTION, GENERAL, EOS], EATING, [EATING, DAILY_LIFE, EOS]),
    ([SHIT_TALK, RUST_GO, EOS], RUST_GO, [RUST_GO, SHIT_TALK, EOS]),
]

# Extra weight for the requested personality topics.
TOPIC_WEIGHTS = {
    GREETING: 1.3,
    EATING: 1.8,
    DAILY_LIFE: 1.8,
    RUST_GO: 2.4,
    SHIT_TALK: 2.2,
    IDENTITY: 1.8,
    INSULT: 2.2,
    FAREWELL: 0.6,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_single_turn_row(action: int, lang: int, output_seq: list[int]) -> str:
    return " ".join(str(t) for t in [action, lang, SEP] + output_seq)


def make_multi_turn_row(context: list[int], action: int, lang: int, output_seq: list[int]) -> str:
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
        insert = GENERAL
    body.insert(rng.randint(0, len(body)), insert)
    return body + [EOS]


def trim_output(seq: list[int], rng: random.Random) -> list[int]:
    if len(seq) <= 3:
        return list(seq)
    body = list(seq[:-1])
    body.pop(rng.randint(0, len(body) - 1))
    if not body:
        body.append(GENERAL)
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
            copies = max(2, int(4 * TOPIC_WEIGHTS.get(action, 1.0)))
            for seq in output_seqs:
                for _ in range(copies):
                    rows.append(make_single_turn_row(action, lang, list(seq)))
                for _ in range(copies):
                    rows.append(make_single_turn_row(action, lang, perturb_output(seq, rng)))
                for _ in range(max(1, copies // 2)):
                    rows.append(make_single_turn_row(action, lang, extend_output(seq, rng, action)))
                for _ in range(max(1, copies // 2)):
                    rows.append(make_single_turn_row(action, lang, trim_output(seq, rng)))
    return rows


def generate_multi_turn_rows(rng: random.Random, lang: int, copies: int = 10) -> list[str]:
    rows: list[str] = []
    for context, action, output_seq in MULTI_TURN_TEMPLATES:
        for _ in range(copies):
            seq = list(output_seq)
            if rng.random() < 0.55:
                seq = perturb_output(seq, rng)
            rows.append(make_multi_turn_row(context, action, lang, seq))
    return rows


def generate_random_multi_turn_rows(rng: random.Random, lang: int, count: int) -> list[str]:
    rows: list[str] = []
    for _ in range(count):
        action = weighted_actions(rng)
        seq = list(rng.choice(BASE_PATTERNS[action]))
        if rng.random() < 0.45:
            seq = perturb_output(seq, rng)
        if rng.random() < 0.25:
            seq = extend_output(seq, rng, action)
        rows.append(make_multi_turn_row(random_context(rng), action, lang, seq))
    return rows


def generate_topic_drill_rows(rng: random.Random, lang: int, count: int) -> list[str]:
    topic_actions = [GREETING, EATING, DAILY_LIFE, RUST_GO, SHIT_TALK, IDENTITY, INSULT]
    rows: list[str] = []
    for _ in range(count):
        action = rng.choice(topic_actions)
        seq = list(rng.choice(BASE_PATTERNS[action]))
        if action != FAREWELL and FAREWELL in seq:
            seq = [GENERAL if token == FAREWELL else token for token in seq]
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
        mt = generate_multi_turn_rows(rng, lang, copies=12)
        all_rows.extend(mt)
        print(f"  multi-turn {label}:      {len(mt):>5} rows")

    for lang, label in [(LANG_TH, "TH"), (LANG_EN, "EN")]:
        random_rows = generate_random_multi_turn_rows(rng, lang, count=900)
        all_rows.extend(random_rows)
        print(f"  random-multi {label}:   {len(random_rows):>5} rows")

    for lang, label in [(LANG_TH, "TH"), (LANG_EN, "EN")]:
        topic_rows = generate_topic_drill_rows(rng, lang, count=900)
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
