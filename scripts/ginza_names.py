"""GiNZA（日本語の解析モデル）で、人名・社名の候補を拾う道具.

名前の確認の画面（``shokoban.name_review.run_ginza``）が ``--stdin`` で呼ぶ。
**spaCy / GiNZA が入った別の venv（``venv-ginza``）の Python で動かす**
（書庫番本体の venv には spaCy を入れない）。このファイルは標準ライブラリと spaCy だけに頼り、
``shokoban`` を import しない（別の venv には ``shokoban`` が入っていないため。文の分け方
``split_sentences`` は本体と同じ中身で持ち、同じ結果になることをテストで確かめる）。spaCy の
import は ``main`` の中だけ（純粋な関数は spaCy なしでテストできる）。

フォルダに置いた複数のファイルをまとめて処理する使い方（``--cases``・``--out``）もある。
そのときは ``<out>/<id>.json`` を書く。

出力 ``<out>/<id>.json``（0600）::

    {"id": "...", "items": [{"word": "...", "kind": "人名"|"社名・店名・団体",
                              "label": "<GiNZA のラベル>", "sentence": <文番号。1始まり>}]}

拾うもの:

- Sudachi の品詞が ``名詞-固有名詞-人名-*``（姓・名・一般）の語 → ``人名``。
- NER の ``Person`` → ``人名``（末尾の さん・様・ちゃん・くん・君・殿 を外す）。
- NER の組織系のラベル（``ORG_LABELS``）→ ``社名・店名・団体``（末尾の さん・さんとこ・のとこ などを外す）。
- 地名・日付・商品名・その他は拾わない。語は文にそのまま有るものだけ。

標準出力には件数だけを出し、文・語は出さない。使い方::

    venv-ginza/bin/python scripts/ginza_names.py --cases <ケースのフォルダ> \\
        --ids n01,n02 --out <出力先>

``--stdin``（名前の確認画面 ``shokoban.name_review.run_ginza`` が使う）は、標準入力の JSON
``{"sentences": ["文", ...]}`` を読み、標準出力に JSON ``{"items": [...]}`` を出す
（ファイルは読み書きしない。語・文は呼んだ側だけが受け取る。失敗は標準エラーに型名だけ）::

    echo '{"sentences": ["..."]}' | venv-ginza/bin/python -I scripts/ginza_names.py --stdin
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

KIND_PERSON = "人名"
KIND_ORG = "社名・店名・団体"

PERSON_LABEL = "Person"
# Sudachi の品詞（GiNZA の ``token.tag_``）。``名詞-固有名詞-人名-姓`` / ``-名`` / ``-一般``。
PERSON_TAG_PREFIX = "名詞-固有名詞-人名"

# NER（拡張固有表現 ENE）のラベルのうち、伏せたい「お客様の商売・取引・所属の相手」になりうる組織系。
# ja_ginza 5.2 のラベル一覧から選んだ:
#   入れる: Company（会社・店）, Company_Group（グループ会社）, Corporation_Other（信金・農協・公庫・法人）,
#           Organization_Other（組合・団体・サークル）, Political_Organization_Other（○○会・連盟）,
#           School（学校・園）, Research_Institute（研究所）, Public_Institution（役所・病院・公的機関）,
#           Sports_Organization_Other（スポーツ団体・クラブ）, Show_Organization（劇団・楽団）
#   入れない: 地名・施設の建物名（Facility_Other・Museum・Station・Park…。お店の建物の言い方が混ざり、
#           伏せすぎになる）、Government・Political_Party・International_Organization・Military・Cabinet
#           （広く知られた公的な名前）、Pro_Sports_Organization・Sports_League（広く知られた名前）、
#           N_Organization（「3組」のような数え方）、Position_Vocation（役職）、Family（家族の呼び方）
ORG_LABELS: frozenset[str] = frozenset(
    {
        "Company",
        "Company_Group",
        "Corporation_Other",
        "Organization_Other",
        "Political_Organization_Other",
        "School",
        "Research_Institute",
        "Public_Institution",
        "Sports_Organization_Other",
        "Show_Organization",
    }
)

# 人名の末尾の敬称・呼び名（長い方から外す）。
PERSON_TAILS: tuple[str, ...] = ("さん", "様", "ちゃん", "くん", "君", "殿")
# 社名・店名・団体の末尾（「丸栄材木さんとこ」「丸栄材木のとこ」「丸栄さん」）。長い順。
ORG_TAILS: tuple[str, ...] = (
    "さんところ",
    "さんとこ",
    "のところ",
    "のとこ",
    "ところ",
    "とこ",
    "さん",
    "様",
    "殿",
)

MAX_WORD_CHARS = 20
_TERMINALS = "。！？!?"
_HAS_LETTER_RE = re.compile(r"[ぁ-んァ-ヶー一-龥々A-Za-zＡ-Ｚａ-ｚ]")


# ---------------------------------------------------------------- 純粋な関数（spaCy なしでテストできる）

def split_sentences(text: str) -> list[str]:
    """「。」「！」「？」「!」「?」と改行で文に分ける（``shokoban.pick_bench.split_sentences`` と同じ中身）."""
    if type(text) is not str:
        raise ValueError("text は文字列で指定してください")
    sentences: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(text)

    def flush() -> None:
        sentence = "".join(buffer).strip()
        if sentence:
            sentences.append(sentence)
        buffer.clear()

    while index < length:
        char = text[index]
        if char in "\r\n":
            flush()
            index += 1
            continue
        buffer.append(char)
        index += 1
        if char in _TERMINALS:
            while index < length and text[index] in _TERMINALS:
                buffer.append(text[index])
                index += 1
            flush()
    flush()
    return sentences


def strip_tail(word: str, tails: Sequence[str]) -> str:
    """末尾の敬称・呼び名を1回だけ外す（外すと空になるときは外さない）."""
    for tail in sorted(tails, key=len, reverse=True):
        if word.endswith(tail) and len(word) > len(tail):
            return word[: -len(tail)]
    return word


def kind_for_ner_label(label: str) -> str | None:
    """NER のラベルを kind にする（人名・組織系以外は ``None``＝拾わない）."""
    if label == PERSON_LABEL:
        return KIND_PERSON
    if label in ORG_LABELS:
        return KIND_ORG
    return None


def is_person_tag(tag: str) -> bool:
    return tag == PERSON_TAG_PREFIX or tag.startswith(PERSON_TAG_PREFIX + "-")


def items_from_analysis(
    sentence: str,
    number: int,
    tokens: Iterable[tuple[str, str]],
    entities: Iterable[tuple[str, str]],
) -> list[dict[str, Any]]:
    """1文の解析結果（``(語, 品詞)`` の並びと ``(語, NERラベル)`` の並び）から項目を作る.

    語は文にそのまま有るものだけ。同じ文の同じ語・kind は1つにまとめる。
    """
    found: dict[tuple[str, str], dict[str, Any]] = {}

    def put(word: str, kind: str, label: str) -> None:
        word = word.strip()
        if (
            not word
            or len(word) > MAX_WORD_CHARS
            or "\n" in word
            or not _HAS_LETTER_RE.search(word)
            or word not in sentence
        ):
            return
        found.setdefault(
            (word, kind), {"word": word, "kind": kind, "label": label, "sentence": number}
        )

    for text, tag in tokens:
        if is_person_tag(tag):
            put(text, KIND_PERSON, tag)
    for text, label in entities:
        kind = kind_for_ner_label(label)
        if kind is None:
            continue
        tails = PERSON_TAILS if kind == KIND_PERSON else ORG_TAILS
        put(strip_tail(text, tails), kind, label)
    return list(found.values())


def dedupe_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """本全体で、同じ語・kind は最初に出た文の1つにする."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        seen.setdefault((item["word"], item["kind"]), item)
    return list(seen.values())


# ---------------------------------------------------------------- 入力・出力

def read_book(path: Path) -> tuple[str, list[str]]:
    """1本を読む（``sentences`` の並びか ``transcript`` の文字列の、どちらの形でもよい）。戻り値は ``(id, 文の並び)``."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"読めません: {Path(path).name}") from exc
    if type(payload) is not dict:
        raise ValueError(f"ルートがオブジェクトではありません: {Path(path).name}")
    book_id = payload.get("id")
    if type(book_id) is not str or not book_id.strip() or "/" in book_id or "\\" in book_id:
        raise ValueError(f"id が正しくありません: {Path(path).name}")
    if isinstance(payload.get("sentences"), list):
        sentences: list[str] = []
        for item in payload["sentences"]:
            text = item.get("text") if type(item) is dict else item
            if type(text) is not str:
                raise ValueError(f"文の形が正しくありません: {Path(path).name}")
            sentences.append(text)
    elif type(payload.get("transcript")) is str:
        sentences = split_sentences(payload["transcript"])
    else:
        raise ValueError(f"sentences も transcript もありません: {Path(path).name}")
    return book_id.strip(), sentences


def find_book_paths(cases_dirs: Sequence[Path], ids: Sequence[str]) -> list[Path]:
    wanted = set(ids)
    paths: list[Path] = []
    for directory in cases_dirs:
        if not Path(directory).is_dir():
            raise ValueError(f"ケースのフォルダがありません: {Path(directory).name}")
        for path in sorted(Path(directory).glob("*.json")):
            if wanted and path.stem not in wanted:
                continue
            paths.append(path)
    if wanted and {p.stem for p in paths} != wanted:
        raise ValueError("--ids のうち、見つからない id があります")
    return paths


def write_private_json(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    """JSON を 0600 で書く（フォルダは無ければ 0700 で作る）."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = directory / name
    temporary = directory / f".{name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, target)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def load_nlp() -> Any:
    """GiNZA を読む（``compound_splitter`` を C 単位にしないと設定の誤りになる）."""
    import spacy  # spaCy は venv-ginza にだけ有る。ここ（main の中）でだけ読む

    return spacy.load(
        "ja_ginza", config={"components": {"compound_splitter": {"split_mode": "C"}}}
    )


def analyze_book(nlp: Any, sentences: Sequence[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for number, doc in enumerate(nlp.pipe(list(sentences), batch_size=32), start=1):
        sentence = sentences[number - 1]
        tokens = [(token.text, token.tag_) for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        items.extend(items_from_analysis(sentence, number, tokens, entities))
    return dedupe_items(items)


MAX_STDIN_SENTENCES = 20000


def parse_stdin_sentences(raw: bytes) -> list[str]:
    """標準入力の JSON（``{"sentences": [文, ...]}``）から文の並びを取り出す."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("標準入力が JSON ではありません") from exc
    if type(payload) is not dict or set(payload) != {"sentences"}:
        raise ValueError("標準入力は sentences だけのオブジェクトにしてください")
    sentences = payload["sentences"]
    if type(sentences) is not list or len(sentences) > MAX_STDIN_SENTENCES:
        raise ValueError("sentences は文の配列にしてください")
    if any(type(item) is not str for item in sentences):
        raise ValueError("sentences の項目は文字列にしてください")
    return list(sentences)


def run_stdin(raw: bytes, nlp: Any) -> bytes:
    """標準入力の中身を解析し、標準出力に書く JSON（UTF-8）を返す（spaCy は ``nlp`` として受け取る）."""
    sentences = parse_stdin_sentences(raw)
    items = analyze_book(nlp, sentences) if sentences else []
    return (json.dumps({"items": items}, ensure_ascii=False) + "\n").encode("utf-8")


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GiNZA で人名・社名の候補を拾う（venv-ginza の Python で動かす）")
    parser.add_argument("--cases", type=Path, nargs="+", action="extend", help="ケースのフォルダ（ファイルの形）")
    parser.add_argument("--ids", default="", help="使う id をカンマ区切りで絞る（既定: 全部）")
    parser.add_argument("--out", type=Path, help="出力先のフォルダ（ファイルの形）")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="標準入力の JSON {sentences: [...]} を読み、標準出力に JSON {items: [...]} を出す",
    )
    args = parser.parse_args(argv)
    if args.stdin:
        if args.cases or args.out or args.ids:
            parser.error("--stdin は --cases・--ids・--out と一緒に使えません")
        try:
            output = run_stdin(sys.stdin.buffer.read(), load_nlp())
        except Exception as exc:
            print(f"続けられません: {type(exc).__name__}", file=sys.stderr)
            return 2
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        return 0
    if not args.cases or args.out is None:
        parser.error("--cases と --out が要ります（または --stdin）")
    try:
        paths = find_book_paths(args.cases, _split(args.ids))
        if not paths:
            raise ValueError("測るケースがありません")
        books = [read_book(path) for path in paths]
        nlp = load_nlp()
        totals: Counter = Counter()
        for book_id, sentences in books:
            items = analyze_book(nlp, sentences)
            write_private_json(args.out, f"{book_id}.json", {"id": book_id, "items": items})
            totals["books"] += 1
            totals["sentences"] += len(sentences)
            totals["items"] += len(items)
            for item in items:
                totals[item["kind"]] += 1
    except Exception as exc:
        print(f"続けられません: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        f"books={totals['books']} sentences={totals['sentences']} items={totals['items']} "
        f"{KIND_PERSON}={totals[KIND_PERSON]} {KIND_ORG}={totals[KIND_ORG]}"
    )
    print(f"出力: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
