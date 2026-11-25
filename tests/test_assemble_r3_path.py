"""道筋（こちらから話した道筋・提案）の欄と、行ごとの話し手の印の試験.

偽の ``MaskedR3`` を渡す。読む役も偽物。文はすべて架空。
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from shokoban.assemble_r3 import (
    F_PATH,
    SHARED_FIELDS,
    assemble_r3,
    empty_assembly,
    speaker_label,
    speaker_mark,
)
from shokoban.pick_bench import FIELDS
from shokoban.reader_r3 import (
    READER_R3_PROMPT_HEAD,
    ReaderR3ShapeError,
    parse_reader_reply_r3,
)
from tests.r3_reader_fakes import ev, make_masked, plain, reply, work

TROUBLE, ASKED, OURS, DECIDED, UNDECIDED = FIELDS

# 打ち合わせの架空の文（番号は1始まり）。
SENTENCES = (
    "まず予約の受け方をそろえましょう。",  # 1 こちら
    "次に集計を自動にします。",  # 2 こちら
    "紙の予約表を毎晩手で書き写しています。",  # 3 お客様
    "最後は振り返りを楽にする仕組みにします。",  # 4 こちら
    "何ステップかに分けて進めましょう。",  # 5 こちら
    "はい、お願いします。",  # 6 お客様
)


def path(sent: int, span: str, *, step: Any = None, final: Any = False, topic: int = 1, rank: int = 1, **extra: Any):
    item: dict[str, Any] = {
        "kind": "道筋",
        "sent": sent,
        "span": span,
        "step": step,
        "final": final,
        "topic": topic,
        "rank": rank,
    }
    item.update(extra)
    return item


def run(payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return assemble_r3(make_masked(SENTENCES), payload, **kwargs)


def lines_of(result: dict[str, Any], field: str) -> list[str]:
    """``shared`` の ``【欄名】`` の下の行."""
    out: list[str] = []
    on = False
    for line in result["shared"].splitlines():
        if line.startswith("【"):
            on = line == f"【{field}】"
            continue
        if on:
            out.append(line)
    return out


class PathColumnTest(unittest.TestCase):
    def test_path_item_goes_only_to_the_path_column(self):
        result = run(reply(path(1, "予約の受け方をそろえましょう", step=1)))
        self.assertEqual([r["span"] for r in result["columns"][F_PATH]], ["予約の受け方をそろえましょう"])
        for field in FIELDS:
            self.assertEqual(result["columns"][field], [])
        self.assertEqual(result["items"], [])  # 5欄だけの平らな一覧には入れない
        self.assertEqual(result["counts"], {})
        self.assertEqual(result["field_counts"][F_PATH], 1)
        self.assertEqual(sum(result["field_counts"][f] for f in FIELDS), 0)

    def test_path_column_is_first_in_columns_and_shared(self):
        result = run(reply())
        self.assertEqual(tuple(result["columns"]), SHARED_FIELDS)
        self.assertEqual(SHARED_FIELDS[0], F_PATH)
        self.assertEqual(result["shared"].splitlines()[0], f"【{F_PATH}】")
        self.assertEqual(lines_of(result, F_PATH), ["なし"])
        self.assertEqual(result["field_counts"], {field: 0 for field in SHARED_FIELDS})
        empty = empty_assembly("失敗")
        self.assertEqual(empty["shared"].splitlines()[0], f"【{F_PATH}】")
        self.assertEqual(empty["field_counts"], {field: 0 for field in SHARED_FIELDS})

    def test_order_is_step_then_no_step_then_final(self):
        result = run(
            reply(
                path(4, "振り返りを楽にする仕組みにします", final=True, rank=1),
                path(5, "何ステップかに分けて進めましょう", rank=2),
                path(2, "集計を自動にします", step=3, rank=3),
                path(1, "予約の受け方をそろえましょう", step=1, rank=4),
            )
        )
        self.assertEqual(
            lines_of(result, F_PATH),
            [
                "- [不明] ステップ1: 予約の受け方をそろえましょう",
                "- [不明] ステップ3: 集計を自動にします",
                "- [不明] 何ステップかに分けて進めましょう",
                "- [不明] 最後: 振り返りを楽にする仕組みにします",
            ],
        )

    def test_same_group_is_ordered_by_sentence_number(self):
        result = run(
            reply(
                path(5, "何ステップかに分けて進めましょう"),
                path(1, "予約の受け方をそろえましょう"),
                path(2, "集計を自動にします", step=2),
                path(1, "予約の受け方", step=2),
            )
        )
        self.assertEqual(
            [(r["step"], r["sent"]) for r in result["columns"][F_PATH]],
            [(2, 1), (2, 2), (None, 1), (None, 5)],
        )

    def test_final_with_step_keeps_the_step_in_the_line(self):
        result = run(reply(path(4, "振り返りを楽にする仕組みにします", step=4, final=True)))
        self.assertEqual(lines_of(result, F_PATH), ["- [不明] 最後（ステップ4）: 振り返りを楽にする仕組みにします"])

    def test_the_same_quote_is_one_row(self):
        item = path(1, "予約の受け方をそろえましょう", step=1)
        result = run(reply(item, dict(item, rank=2)))
        self.assertEqual(len(result["columns"][F_PATH]), 1)

    def test_path_rows_are_not_cut_to_eight(self):
        texts = tuple(f"ステップ{n}で次の作業を進めましょう。" for n in range(1, 11))
        masked = make_masked(texts)
        items = [path(n, f"ステップ{n}で次の作業を進めましょう", step=n, rank=n) for n in range(1, 11)]
        result = assemble_r3(masked, reply(*items))
        self.assertEqual(len(result["columns"][F_PATH]), 10)
        self.assertTrue(any(n["reason"] == "多すぎ" and n["field"] == F_PATH for n in result["notes"]))

    def test_counts_and_items_do_not_change_when_a_path_item_is_added(self):
        base_items = [
            plain("困りごと", 3, "紙の予約表を毎晩手で書き写しています"),
            plain("未決", 6, "はい、お願いします"),
        ]
        before = run(reply(*base_items))
        after = run(reply(*base_items, path(1, "予約の受け方をそろえましょう", step=1)))
        self.assertEqual(before["counts"], after["counts"])
        self.assertEqual(
            [(r["field"], r["sent"], r["span"]) for r in before["items"]],
            [(r["field"], r["sent"], r["span"]) for r in after["items"]],
        )
        self.assertEqual(before["shape_errors"], after["shape_errors"])
        for field in FIELDS:
            self.assertEqual(before["field_counts"][field], after["field_counts"][field])
        self.assertEqual(after["old_rules"], before["old_rules"])


class PathShapeTest(unittest.TestCase):
    def test_bad_step_or_final_is_a_shape_error_and_drops_only_that_item(self):
        for bad in (
            {"step": "1"},
            {"step": True},
            {"step": 0},
            {"step": -1},
            {"step": 1.5},
            {"final": "true"},
            {"final": 1},
            {"final": []},
        ):
            with self.subTest(bad=bad):
                good = path(2, "集計を自動にします", step=3)
                result = run(reply(path(1, "予約の受け方をそろえましょう", **bad), good))
                self.assertEqual(result["counts"], {"形崩れ": 1})
                self.assertEqual(result["shape_errors"], 1)
                self.assertEqual([r["sent"] for r in result["columns"][F_PATH]], [2])

    def test_missing_or_null_step_and_final_are_fine(self):
        item = {"kind": "道筋", "sent": 1, "span": "予約の受け方をそろえましょう", "topic": 1, "rank": 1}
        result = run(reply(item, path(2, "集計を自動にします", step=None, final=None)))
        self.assertEqual(result["counts"], {})
        self.assertEqual([(r["step"], r["final"]) for r in result["columns"][F_PATH]], [(None, False), (None, False)])

    def test_who_state_requested_evidence_are_ignored_on_a_path_item(self):
        item = path(1, "予約の受け方をそろえましょう", step=1, who="支援側", state="提案中", requested=False, evidence=[])
        result = run(reply(item))
        self.assertEqual(result["counts"], {})
        self.assertEqual(len(result["columns"][F_PATH]), 1)

    def test_detail_and_name_are_not_allowed_but_the_item_stays(self):
        item = path(1, "予約の受け方をそろえましょう", step=1, detail=[{"sent": 2, "span": "集計"}], name="予約の整理")
        result = run(reply(item))
        self.assertEqual(len(result["columns"][F_PATH]), 1)
        self.assertEqual(result["columns"][F_PATH][0]["detail"], [])
        self.assertEqual(result["columns"][F_PATH][0]["name"], "")
        self.assertIn("detail不可", result["counts"])

    def test_path_quote_is_checked_like_other_kinds(self):
        masked = make_masked(SENTENCES, hidden=(4,))
        for item, reason in (
            (path(4, "振り返りを楽にする"), "伏せた文"),
            (path(1, "この文に無い言葉"), "部分でない"),
            (path(99, "予約"), None),
        ):
            with self.subTest(sent=item["sent"]):
                result = assemble_r3(masked, reply(item))
                self.assertEqual(result["columns"][F_PATH], [])
                self.assertEqual(len(result["dropped"]), 1)
                if reason:
                    self.assertEqual(result["dropped"][0]["reason"], reason)

    def test_path_with_region_is_a_shape_error(self):
        masked = make_masked(SENTENCES, regions=[(1, 3, 3, "〈伏せた区間：困りごと＝入力〉")])
        item = {"kind": "道筋", "region": 1, "topic": 1, "rank": 1}
        result = assemble_r3(masked, reply(item))
        self.assertEqual(result["counts"], {"形崩れ": 1})

    def test_step_and_final_on_other_kinds_are_ignored(self):
        item = plain("未決", 6, "はい、お願いします")
        item.update({"step": 2, "final": True})
        result = run(reply(item))
        self.assertEqual(result["counts"], {})
        self.assertEqual(len(result["columns"][UNDECIDED]), 1)
        self.assertEqual(result["columns"][F_PATH], [])


class SpeakerMarkTest(unittest.TestCase):
    def test_speaker_mark_and_label(self):
        self.assertEqual(speaker_label("支援側"), "こちら")
        self.assertEqual(speaker_label("お客様"), "お客様")
        self.assertEqual(speaker_label("不明"), "不明")
        self.assertEqual(speaker_label("???"), "不明")
        self.assertEqual(speaker_mark("支援側"), "[こちら] ")

    def test_speakers_range_gives_the_mark_to_every_kind_of_row(self):
        speakers = [
            {"from": 1, "to": 2, "who": "支援側"},
            {"from": 3, "to": 3, "who": "お客様"},
            {"from": 4, "to": 5, "who": "支援側"},
        ]
        result = run(
            reply(
                path(1, "予約の受け方をそろえましょう", step=1),
                plain("困りごと", 3, "紙の予約表を毎晩手で書き写しています"),
                plain("未決", 4, "最後は振り返りを楽にする仕組み"),
                speakers=speakers,
            )
        )
        self.assertEqual(result["columns"][F_PATH][0]["speaker"], "支援側")
        self.assertEqual(result["columns"][TROUBLE][0]["speaker"], "お客様")
        self.assertEqual(result["columns"][UNDECIDED][0]["speaker"], "支援側")
        self.assertEqual(lines_of(result, F_PATH), ["- [こちら] ステップ1: 予約の受け方をそろえましょう"])
        self.assertEqual(lines_of(result, TROUBLE), ["- [お客様] 紙の予約表を毎晩手で書き写しています"])
        self.assertEqual(lines_of(result, UNDECIDED), ["- [こちら] 最後は振り返りを楽にする仕組み"])

    def test_no_speakers_is_unknown(self):
        result = run(reply(plain("困りごと", 3, "紙の予約表を毎晩手で書き写しています")))
        self.assertEqual(result["columns"][TROUBLE][0]["speaker"], "不明")
        self.assertEqual(lines_of(result, TROUBLE), ["- [不明] 紙の予約表を毎晩手で書き写しています"])

    def test_unknown_speaker_range_is_unknown(self):
        result = run(
            reply(plain("困りごと", 3, "紙の予約表を毎晩手で書き写しています"), speakers=[{"from": 1, "to": 6, "who": "不明"}])
        )
        self.assertEqual(result["columns"][TROUBLE][0]["speaker"], "不明")

    def test_evidence_on_the_same_sentence_beats_the_speakers_range(self):
        # speakers は「不明」だが、同じ文を指す根拠の speaker が「支援側」
        item = work(2, "集計を自動にします", evidence=[ev("引き受け", "支援側", 2, "集計を自動にします")])
        result = run(reply(item, speakers=[{"from": 1, "to": 6, "who": "不明"}]))
        self.assertEqual(result["columns"][OURS][0]["speaker"], "支援側")
        self.assertIn("- [こちら] ", lines_of(result, OURS)[0])

    def test_evidence_on_another_sentence_is_not_used(self):
        # 根拠は文1（お客様の依頼）。行の文は文2で、範囲は「支援側」
        item = work(
            2,
            "集計を自動にします",
            evidence=[ev("依頼", "お客様", 1, "まず予約", to="支援側"), ev("引き受け", "支援側", 2, "集計を自動にします")],
        )
        result = run(reply(item, speakers=[{"from": 1, "to": 1, "who": "お客様"}, {"from": 2, "to": 2, "who": "支援側"}]))
        self.assertEqual(result["columns"][OURS][0]["speaker"], "支援側")

        item = work(2, "集計を自動にします", evidence=[ev("依頼", "お客様", 1, "まず予約", to="支援側")])
        result = run(reply(item, speakers=[{"from": 2, "to": 2, "who": "支援側"}]))
        row = next(r for field in FIELDS for r in result["columns"][field] if r["sent"] == 2)
        self.assertEqual(row["speaker"], "支援側")  # 文1の根拠（お客様）ではなく、文2の範囲

    def test_unknown_evidence_speaker_is_skipped(self):
        item = work(2, "集計を自動にします", evidence=[ev("引き受け", "不明", 2, "集計を自動にします")])
        result = run(reply(item, speakers=[{"from": 2, "to": 2, "who": "支援側"}]))
        row = next(r for field in FIELDS for r in result["columns"][field] if r["sent"] == 2)
        self.assertEqual(row["speaker"], "支援側")

    def test_conflicting_evidence_on_the_same_sentence_falls_back_to_the_range(self):
        span = "集計を自動にします"
        item = work(2, span, evidence=[ev("引き受け", "支援側", 2, span), ev("同意", "お客様", 2, span)])
        result = run(reply(item, speakers=[{"from": 2, "to": 2, "who": "支援側"}]))
        row = next(r for field in FIELDS for r in result["columns"][field] if r["sent"] == 2)
        self.assertEqual(row["speaker"], "支援側")
        result = run(reply(item))
        row = next(r for field in FIELDS for r in result["columns"][field] if r["sent"] == 2)
        self.assertEqual(row["speaker"], "不明")

    def test_region_row_uses_the_regions_only_speaker(self):
        tag = "〈伏せた区間：困りごと＝入力〉"
        masked = make_masked(SENTENCES, regions=[(1, 3, 4, tag)])
        item = {"kind": "困りごと", "region": 1, "topic": 1, "rank": 1}
        one = assemble_r3(masked, reply(item, speakers=[{"from": 1, "to": 6, "who": "お客様"}]))
        self.assertEqual(one["columns"][TROUBLE][0]["speaker"], "お客様")
        self.assertIn(f"- [お客様] {tag}", one["shared"])
        mixed = assemble_r3(
            masked,
            reply(item, speakers=[{"from": 1, "to": 3, "who": "お客様"}, {"from": 4, "to": 6, "who": "支援側"}]),
        )
        self.assertEqual(mixed["columns"][TROUBLE][0]["speaker"], "不明")
        with_unknown = assemble_r3(
            masked,
            reply(item, speakers=[{"from": 3, "to": 3, "who": "お客様"}, {"from": 4, "to": 4, "who": "不明"}]),
        )
        self.assertEqual(with_unknown["columns"][TROUBLE][0]["speaker"], "不明")
        none = assemble_r3(masked, reply(item))
        self.assertEqual(none["columns"][TROUBLE][0]["speaker"], "不明")


class ColumnFieldsTest(unittest.TestCase):
    """欄の一覧（``FIELDS``）に道筋は入らず、5欄のままである."""

    def test_pick_bench_fields_do_not_include_the_path_column(self):
        self.assertNotIn(F_PATH, FIELDS)
        self.assertEqual(len(FIELDS), 5)


class ReaderPromptTest(unittest.TestCase):
    def test_prompt_adds_the_path_kind_without_changing_the_old_instructions(self):
        head = READER_R3_PROMPT_HEAD
        self.assertIn("kind は 作業 / 決めごと / 困りごと / 未決 / 道筋 のどれか。", head)
        self.assertIn("  - 道筋: 支援側（こちら）が話した", head)
        self.assertIn('"kind":"作業|決めごと|困りごと|未決|道筋"', head)
        self.assertIn('"step":<整数|null>,"final":<true|false>', head)
        # 未決の説明の直後に道筋の説明
        self.assertIn("  - 未決: まだ決まっていない事柄。\n  - 道筋:", head)
        # 今ある種類の指示の文は残っている
        for sentence in (
            "  - 作業: 支援側かお客様が実際にやる作業。作業は漏らさず全部挙げる。\n",
            "  - 決めごと: 日程・費用・方針など、作業以外で決める・決めた事柄。\n",
            "- 困りごと・未決には who と state を書かない。決めごとには who を書かない。\n",
        ):
            self.assertIn(sentence, head)

    def test_reply_with_a_path_item_is_read(self):
        text = json.dumps(
            {"speakers": [], "items": [{"kind": "道筋", "sent": 1, "span": "x", "step": 1, "final": False, "topic": 1, "rank": 1}]},
            ensure_ascii=False,
        )
        self.assertEqual(parse_reader_reply_r3(text)["items"][0]["kind"], "道筋")
        with self.assertRaises(ReaderR3ShapeError):
            parse_reader_reply_r3("{}")


if __name__ == "__main__":
    unittest.main()
