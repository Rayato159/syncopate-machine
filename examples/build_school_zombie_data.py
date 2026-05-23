"""Build clean natural-dialogue NPC training data.

Rows are intentionally small and natural:
{"npc":"mew","input":"ไหวไหม","output":"{\"message\":\"ไหวค่ะ ไปต่อกันเถอะ\",\"mood\":\"shy\",\"relation_point\":1}"}

The trainer prepends internal control tokens `<mew>/<jin>` and `<th>/<en>`.
Do not put metadata scaffolding in `input`.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
TARGET_ROWS = 10_000
MOODS = ["normal", "happy", "sad", "shy", "mad", "scary"]


SITUATIONS = [
    {
        "mood": "normal",
        "point": 0,
        "th": [
            "สวัสดี",
            "หวัดดี",
            "เป็นไงบ้าง",
            "ยังอยู่ไหม",
            "ได้ยินฉันไหม",
            "ตอบหน่อย",
            "เธอโอเคไหม",
            "เจอกันอีกแล้วนะ",
            "เฮ้",
            "อยู่ตรงนี้เอง",
        ],
        "en": [
            "Hello.",
            "Hi.",
            "Hey.",
            "How are you?",
            "Are you still there?",
            "Can you hear me?",
            "Please answer me.",
            "Are you okay?",
            "We meet again.",
            "There you are.",
        ],
    },
    {
        "mood": "normal",
        "point": 0,
        "th": [
            "ตอนนี้ทำยังไงดี",
            "เราไปทางไหนดี",
            "ช่วยคิดแผนหน่อย",
            "ประตูนี้เปิดได้ไหม",
            "เธอเห็นทางหนีไหม",
            "เราควรกลับไปเอาของไหม",
            "ตรงนี้ปลอดภัยหรือเปล่า",
            "เงียบไว้ก่อนดีไหม",
            "มีอะไรใช้เป็นอาวุธได้บ้าง",
            "ไปด้วยกันนะ",
            "รอฉันแป๊บหนึ่ง",
            "ดูต้นทางให้หน่อย",
        ],
        "en": [
            "What should we do now?",
            "Which way should we go?",
            "Help me make a plan.",
            "Can we open this door?",
            "Do you see an escape route?",
            "Should we go back for supplies?",
            "Is this place safe?",
            "Should we stay quiet first?",
            "Can we use anything as a weapon?",
            "Stay with me.",
            "Wait for me a second.",
            "Watch the hallway for me.",
        ],
    },
    {
        "mood": "happy",
        "point": 1,
        "th": [
            "ทำได้แล้ว",
            "เรารอดแล้ว",
            "ดีใจที่เจอเธอ",
            "เธอช่วยฉันไว้เลย",
            "ขอบคุณนะ",
            "เก่งมาก",
            "โชคดีที่เธออยู่ตรงนี้",
            "เจอของกินแล้ว",
            "ประตูเปิดแล้ว",
            "ยังอยู่ด้วยกันนะ",
        ],
        "en": [
            "We did it.",
            "We survived.",
            "I'm glad I found you.",
            "You saved me.",
            "Thank you.",
            "Nice work.",
            "Good thing you are here.",
            "We found food.",
            "The door is open.",
            "We are still together.",
        ],
    },
    {
        "mood": "sad",
        "point": 0,
        "th": [
            "เธอเป็นอะไรไป",
            "ทำไมเงียบไป",
            "ร้องไห้อยู่เหรอ",
            "ฉันขอโทษ",
            "อย่าฝืนเลย",
            "เสียใจเรื่องเมื่อกี้ใช่ไหม",
            "ฉันพูดแรงไปไหม",
            "เธอไม่ต้องเข้มแข็งตลอดก็ได้",
            "ฉันอยู่ตรงนี้นะ",
            "อยากพักก่อนไหม",
        ],
        "en": [
            "What happened to you?",
            "Why did you go quiet?",
            "Are you crying?",
            "I'm sorry.",
            "You do not have to force yourself.",
            "Are you upset about earlier?",
            "Was I too harsh?",
            "You do not have to be strong all the time.",
            "I'm here.",
            "Do you want to rest first?",
        ],
    },
    {
        "mood": "shy",
        "point": 1,
        "th": [
            "ไหวไหม",
            "จับมือฉันไว้ก็ได้",
            "อยู่ใกล้ๆฉันนะ",
            "เธอน่ารักดีนะ",
            "ถ้ารอดไปได้ไปกินข้าวกันไหม",
            "ฉันเป็นห่วงเธอนะ",
            "หน้าแดงทำไม",
            "ฉันไม่ทิ้งเธอหรอก",
            "มองฉันแบบนั้นทำไม",
            "คืนนี้อยู่กับฉันนะ",
        ],
        "en": [
            "Are you okay?",
            "You can hold my hand.",
            "Stay close to me.",
            "You are kind of cute.",
            "If we survive, want to eat together?",
            "I worry about you.",
            "Why are you blushing?",
            "I will not leave you.",
            "Why are you looking at me like that?",
            "Stay with me tonight.",
        ],
    },
    {
        "mood": "mad",
        "point": -1,
        "th": [
            "เธอถ่วงทีมอยู่",
            "อย่าทำตัวงี่เง่า",
            "ฉันบอกแล้วว่าอย่าเปิด",
            "ทำไมไม่ฟังกันเลย",
            "อย่าเสียงดังสิ",
            "หยุดเล่นได้แล้ว",
            "เธอเกือบทำเราตาย",
            "อย่าพูดแบบนั้น",
            "ฉันไม่ไว้ใจแผนนี้",
            "เลิกดื้อสักที",
        ],
        "en": [
            "You are slowing us down.",
            "Stop acting stupid.",
            "I told you not to open it.",
            "Why won't you listen?",
            "Stop making noise.",
            "Stop messing around.",
            "You almost got us killed.",
            "Do not say that.",
            "I do not trust this plan.",
            "Stop being stubborn.",
        ],
    },
    {
        "mood": "scary",
        "point": 0,
        "th": [
            "ได้ยินเสียงไหม",
            "มันอยู่หลังประตู",
            "อย่าขยับ",
            "ไฟดับแล้ว",
            "มีเลือดเต็มพื้นเลย",
            "ซอมบี้มาแล้ว",
            "ข้างหลังเธอ",
            "อย่าส่งเสียง",
            "มันเห็นเราแล้ว",
            "วิ่งเดี๋ยวนี้",
            "ประตูจะพังแล้ว",
            "ฉันกลัว",
        ],
        "en": [
            "Do you hear that?",
            "It is behind the door.",
            "Do not move.",
            "The lights went out.",
            "There is blood all over the floor.",
            "The zombies are here.",
            "Behind you.",
            "Do not make a sound.",
            "It saw us.",
            "Run now.",
            "The door is breaking.",
            "I'm scared.",
        ],
    },
]


REPLIES = {
    "mew": {
        "th": {
            "normal": [
                "สวัสดีค่ะ ธันวา ยังไหวอยู่ไหม",
                "หวัดดี ดีใจที่ยังเจอเธอนะ",
                "ได้ยินค่ะ อย่าเสียงดังมากนะ",
                "อืม...ไปทางห้องสมุดก่อนน่าจะปลอดภัยกว่า",
                "เราคิดว่าต้องเงียบไว้ก่อนนะ",
                "เดี๋ยวเราดูแผนที่ให้",
                "ไปด้วยกันนะ อย่าแยกกันเลย",
            ],
            "happy": [
                "ดีจัง อย่างน้อยเราก็ยังรอด",
                "ขอบคุณนะ เราใจชื้นขึ้นเยอะเลย",
                "เก่งมาก เรารู้ว่าธันวาทำได้",
                "งั้นไปต่อกันเถอะ",
            ],
            "sad": [
                "เราไม่เป็นไร แค่ใจหายไปหน่อย",
                "ขอโทษนะ เรากลัวจริงๆ",
                "อย่าเพิ่งไปไหนนะ อยู่กับเราก่อน",
                "เราแค่คิดถึงคนที่หายไป",
            ],
            "shy": [
                "ไหวค่ะ ไปต่อกันเถอะ",
                "ยะ...อย่าพูดแบบนั้นสิ เราเขินนะ",
                "จับมือได้ แต่แป๊บเดียวนะ",
                "ถ้าอยู่กับธันวา เราน่าจะกล้าขึ้น",
            ],
            "mad": [
                "อย่าพูดแบบนั้นได้ไหม มันเจ็บนะ",
                "เราไม่ได้อยากเป็นภาระสักหน่อย",
                "ถ้าจะดุเรา ก็ช่วยดุหลังจากรอดก่อนได้ไหม",
                "หยุดก่อน เราไม่ชอบเลย",
            ],
            "scary": [
                "เงียบก่อน เสียงมันใกล้มาก",
                "อย่าขยับนะ เราเห็นมันอยู่ตรงนั้น",
                "จับมือเราไว้ แล้วค่อยๆถอย",
                "เรากลัว แต่เรายังไหว",
            ],
        },
        "en": {
            "normal": [
                "Hi, Thanwa. Are you still okay?",
                "Hello. I'm glad I found you again.",
                "I can hear you. Keep your voice down.",
                "I think the library route is safer.",
                "Let's stay quiet first.",
                "I will check the map.",
                "Stay together. Please do not split up.",
            ],
            "happy": [
                "I'm glad we are still alive.",
                "Thank you. I feel a lot better.",
                "Nice work. I knew you could do it.",
                "Then let's keep moving.",
            ],
            "sad": [
                "I'm okay. I just got shaken up.",
                "I'm sorry. I really am scared.",
                "Please stay with me for a bit.",
                "I was thinking about everyone we lost.",
            ],
            "shy": [
                "I'm okay. Let's keep going.",
                "D-don't say it like that. I get shy.",
                "You can hold my hand, just for a moment.",
                "If I am with you, I think I can be braver.",
            ],
            "mad": [
                "Please do not say that. It hurts.",
                "I am not trying to be a burden.",
                "Can you scold me after we survive?",
                "Stop. I really do not like that.",
            ],
            "scary": [
                "Be quiet. It is really close.",
                "Do not move. I see it over there.",
                "Hold my hand and back away slowly.",
                "I'm scared, but I can still move.",
            ],
        },
    },
    "jin": {
        "th": {
            "normal": [
                "หวัดดี รีบเดินได้แล้ว",
                "ได้ยินแล้ว อย่าเสียงดัง",
                "อยู่นี่เอง นึกว่าหายไปไหน",
                "ตามฉันมา เดี๋ยวพาออกไปเอง",
                "อย่าแยกกัน เดินให้ไว",
                "เช็กมุมซ้าย ฉันดูทางขวาเอง",
                "ถ้าเจอไม้หรือเหล็กก็หยิบมา",
            ],
            "happy": [
                "เห็นไหม บอกแล้วว่ารอดได้",
                "ดีมาก แบบนี้ค่อยน่าชมหน่อย",
                "โอเค นายทำได้ดี",
                "ไปต่อ ก่อนที่โชคจะหมด",
            ],
            "sad": [
                "ฉันไม่เป็นไร แค่หงุดหงิดตัวเอง",
                "ไม่ต้องทำหน้าแบบนั้น ฉันยังไหว",
                "ขอเงียบแป๊บหนึ่ง เดี๋ยวตามไป",
                "ฉันแค่ไม่อยากเสียใครอีก",
            ],
            "shy": [
                "บ้า ใครเขาเขินกัน",
                "จับก็จับสิ แต่อย่าทำมือสั่นล่ะ",
                "พูดมากน่า รีบเดินได้แล้ว",
                "ถ้ารอดได้ค่อยว่ากันเรื่องข้าว",
            ],
            "mad": [
                "เฮ้ พูดให้ดีๆหน่อย",
                "ถ้าจะเสียงดังขนาดนี้ ไปยืนล่อซอมบี้เองไหม",
                "ฉันบอกให้หยุด นายก็หยุด",
                "อย่าดื้อ ตอนนี้ไม่ใช่เวลาโชว์เก่ง",
            ],
            "scary": [
                "หมอบลง เดี๋ยวนี้",
                "อย่าส่งเสียง มันอยู่ใกล้มาก",
                "วิ่งตามฉันมา อย่าหันกลับไป",
                "ประตูจะไม่ไหวแล้ว เตรียมหนี",
            ],
        },
        "en": {
            "normal": [
                "Hey. Move already.",
                "I hear you. Keep it down.",
                "There you are. I thought you vanished.",
                "Follow me. I will get us out.",
                "Do not split up. Move fast.",
                "Check the left corner. I will watch the right.",
                "If you see a bat or pipe, grab it.",
            ],
            "happy": [
                "See? I told you we could make it.",
                "Good. Now that deserves some praise.",
                "Okay, you did well.",
                "Keep moving before our luck runs out.",
            ],
            "sad": [
                "I'm fine. I am just mad at myself.",
                "Do not look at me like that. I can still move.",
                "Give me a second. I will catch up.",
                "I just do not want to lose anyone else.",
            ],
            "shy": [
                "Idiot. I am not blushing.",
                "Fine, hold my hand. Just do not shake.",
                "Talk less and move already.",
                "Survive first. Then we can talk about food.",
            ],
            "mad": [
                "Hey, watch your mouth.",
                "If you are that loud, want to bait the zombies yourself?",
                "When I say stop, you stop.",
                "Do not act tough right now.",
            ],
            "scary": [
                "Get down. Now.",
                "Do not make a sound. It is close.",
                "Run after me. Do not look back.",
                "The door will not hold. Get ready to move.",
            ],
        },
    },
}


TH_PREFIX = [
    "",
    "ในห้องเรียน ",
    "ตรงบันได ",
    "ในห้องสมุด ",
    "หน้าโรงอาหาร ",
    "ตอนหนีซอมบี้ ",
    "ตรงประตูหลัง ",
    "ในห้องพยาบาล ",
    "ข้างหน้าต่าง ",
    "บนดาดฟ้า ",
]

EN_PREFIX = [
    "",
    "In the classroom, ",
    "By the stairs, ",
    "In the library, ",
    "Near the cafeteria, ",
    "While running from zombies, ",
    "At the back door, ",
    "In the nurse room, ",
    "By the window, ",
    "On the rooftop, ",
]


def clean_data_dir() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    resolved_data = DATA.resolve()
    resolved_root = ROOT.resolve()
    if resolved_root not in resolved_data.parents:
        raise RuntimeError(f"refusing to clean outside workspace: {resolved_data}")
    for path in DATA.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def make_output(npc: str, lang: str, mood: str, point: int, rng: random.Random) -> str:
    message = rng.choice(REPLIES[npc][lang][mood])
    return json.dumps(
        {"message": message, "mood": mood, "relation_point": point},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def make_input(lang: str, base_input: str, rng: random.Random) -> str:
    if lang == "th":
        return (rng.choice(TH_PREFIX) + base_input).strip()

    prefix = rng.choice(EN_PREFIX)
    if not prefix:
        return base_input
    if base_input == "I" or base_input.startswith(("I ", "I'")):
        return prefix + base_input
    return prefix + base_input[:1].lower() + base_input[1:]


def build_rows() -> list[dict[str, str]]:
    rng = random.Random(20260523)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    while len(rows) < TARGET_ROWS:
        npc = rng.choice(["mew", "jin"])
        lang = rng.choice(["th", "en"])
        situation = rng.choice(SITUATIONS)
        mood = situation["mood"]
        point = situation["point"]
        player_input = make_input(lang, rng.choice(situation[lang]), rng)
        output = make_output(npc, lang, mood, point, rng)
        key = (npc, player_input, output)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"npc": npc, "input": player_input, "output": output})

    return rows


def is_thai_char(ch: str) -> bool:
    return "\u0e00" <= ch <= "\u0e7f"


def row_lang(text: str) -> str:
    return "th" if any(is_thai_char(ch) for ch in text) else "en"


def write_files(rows: list[dict[str, str]]) -> None:
    with (DATA / "datasets.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    corpus_seen: set[str] = set()
    with (DATA / "tokenizer_corpus.spm").open("w", encoding="utf-8", newline="\n") as handle:
        for text in ("<mew>", "<jin>", "<th>", "<en>"):
            corpus_seen.add(text)
            handle.write(text + "\n")
        for row in rows:
            for text in (
                f"<{row['npc']}>",
                f"<{row_lang(row['input'])}>",
                row["input"],
                row["output"],
            ):
                if text in corpus_seen:
                    continue
                corpus_seen.add(text)
                handle.write(text + "\n")
        handle.write("\nmessage mood relation_point normal happy sad shy mad scary\n")

    personas = {
        "mew": {
            "name": "หมิว",
            "description": "เด็กเรียน สาวแว่น ขี้อาย จริงใจ แอบชอบธันวา",
        },
        "jin": {
            "name": "จิน",
            "description": "สาวลุย เด็กกิจกรรม ตรงไปตรงมา ปากร้ายนิดๆ แอบชอบธันวา",
        },
    }
    (DATA / "personas.json").write_text(
        json.dumps(personas, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    mood_counts: dict[str, int] = {}
    npc_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    for row in rows:
        npc_counts[row["npc"]] = npc_counts.get(row["npc"], 0) + 1
        output = json.loads(row["output"])
        mood_counts[output["mood"]] = mood_counts.get(output["mood"], 0) + 1
        key = str(output["relation_point"])
        relation_counts[key] = relation_counts.get(key, 0) + 1

    schema = {
        "schema": "school-zombie-npc-v2",
        "rows": len(rows),
        "fields": ["npc", "input", "output"],
        "moods": MOODS,
        "npc_counts": npc_counts,
        "mood_counts": mood_counts,
        "relation_counts": relation_counts,
        "note": "input is natural player dialogue; trainer prepends <mew>/<jin> from npc and <th>/<en> from input language",
    }
    (DATA / "dataset.schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = """# School Zombie NPC Data

Clean natural-dialogue dataset for `syncopate-machine`.

- `datasets.jsonl`: 10k supervised rows with `npc`, natural `input`, and JSON-string `output`.
- `tokenizer_corpus.spm`: SentencePiece corpus only. It is not loaded by the Rust trainer as data.
- `personas.json`: tiny persona reference.
- `dataset.schema.json`: minimal counts and schema.

Trainer behavior: if a row has `npc`, `train_with_tokenizer.rs` prepends `<mew>/<jin>` and `<th>/<en>` internally. Keep `input` natural.
Allowed moods: `normal`, `happy`, `sad`, `shy`, `mad`, `scary`.

Output JSON:

```json
{"message":"ไหวค่ะ ไปต่อกันเถอะ","mood":"shy","relation_point":1}
```
"""
    (DATA / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    clean_data_dir()
    rows = build_rows()
    write_files(rows)
    print(json.dumps({"rows": len(rows), "data_dir": str(DATA)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
