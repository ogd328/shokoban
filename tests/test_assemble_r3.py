"""読む役の返事の点検と組み立て（``assemble_r3``）の試験.

伏せる部品（``mask_r3``）は使わず、偽の ``MaskedR3`` を渡す。読む役も偽物。文はすべて架空。
"""

from __future__ import annotations

import json
import unittest
from datetime import date
from typing import Any

from shokoban.assemble_r3 import F_PATH, assemble_r3, empty_assembly
from shokoban.pick_bench import FIELDS
from shokoban.stage5_r3_types import MaskedR3
from tests.r3_reader_fakes import BASE, decision, ev, make_masked, plain, reply, work

TROUBLE, ASKED, OURS, DECIDED, UNDECIDED = FIELDS

S1 = "チラシをお願いできますか"
S2 = "チラシは私が作ります"
S3 = "写真は私のほうで用意します"
S4 = "それでお願いします"
S5 = "印刷会社に頼もう"
S6 = "ホームページの更新は来月"
S7 = "一旦保留にしておきましょう"
S8 = "人手が足りない"
S9 = "チラシの話はなしに"

TAG = "〈伏せた区間：困りごと＝人手が足りない〉"


def run(payload: dict[str, Any], masked: MaskedR3 | None = None, **kwargs: Any):
    return assemble_r3(masked or make_masked(), payload, **kwargs)


def spans(result: dict[str, Any], field: str) -> list[str]:
    return [row["span"] for row in result["columns"][field]]


def reasons(result: dict[str, Any], field: str = UNDECIDED) -> list[str]:
    return [row["reason"] for row in result["columns"][field]]


def drop_reasons(result: dict[str, Any], scope: str | None = None) -> list[str]:
    return [
        d["reason"] for d in result["dropped"] if scope is None or d["scope"] == scope
    ]


ASK_US = ev("依頼", "お客様", 1, S1, to="支援側")
OURS_OK = ev("引き受け", "支援側", 2, S2)
CUSTOMER_OFFER = ev("申し出", "お客様", 3, S3, to="支援側")
CUSTOMER_OK = ev("引き受け", "お客様", 3, S3)
OURS_AGREE = ev("同意", "支援側", 4, S4)


class TableRowTest(unittest.TestCase):
    """項目を欄へ振り分ける表の9つの行を、1件ずつ確かめる."""

    def test_row1_trouble(self):
        result = run(reply(plain("困りごと", 8, S8)))
        self.assertEqual(spans(result, TROUBLE), [S8])
        self.assertEqual(result["shape_errors"], 0)

    def test_row2_undecided_kind(self):
        result = run(reply(plain("未決", 6, S6)))
        self.assertEqual(spans(result, UNDECIDED), [S6])
        self.assertEqual(reasons(result), [""])

    def test_row3_hold_with_hold_evidence_is_marked_deferred(self):
        result = run(
            reply(
                work(
                    6,
                    S6,
                    state="保留",
                    evidence=[ev("保留", "支援側", 7, S7)],
                )
            )
        )
        self.assertEqual(spans(result, UNDECIDED), [S6])
        self.assertEqual(reasons(result), ["保留"])
        self.assertTrue(result["columns"][UNDECIDED][0]["deferred"])
        self.assertIn("- [不明] （先送り）" + S6, result["shared"])

    def test_row3_hold_without_hold_evidence_has_no_mark(self):
        result = run(reply(work(6, S6, state="保留")))
        self.assertEqual(spans(result, UNDECIDED), [S6])
        self.assertFalse(result["columns"][UNDECIDED][0]["deferred"])
        self.assertNotIn("先送り", result["shared"])

    def test_row3_applies_to_decisions_too(self):
        result = run(reply(decision(6, S6, state="保留")))
        self.assertEqual(spans(result, UNDECIDED), [S6])
        self.assertEqual(spans(result, DECIDED), [])

    def test_row4_unknown_who_is_unassigned(self):
        result = run(reply(work(2, S2, who="不明", evidence=[OURS_OK])))
        self.assertEqual(spans(result, UNDECIDED), [S2])
        self.assertEqual(spans(result, OURS), [])
        self.assertTrue(result["columns"][UNDECIDED][0]["unassigned"])
        self.assertIn("- [こちら] 〈担当未確定〉" + S2, result["shared"])

    def test_row5_ours_accepted_goes_to_ours_only(self):
        result = run(reply(work(2, S2, evidence=[OURS_OK])))
        self.assertEqual(spans(result, OURS), [S2])
        self.assertEqual(spans(result, ASKED), [])
        self.assertEqual(spans(result, UNDECIDED), [])

    def test_row5_requested_with_ask_also_goes_to_asked(self):
        result = run(
            reply(work(2, S2, requested=True, evidence=[ASK_US, OURS_OK]))
        )
        self.assertEqual(spans(result, OURS), [S2])
        self.assertEqual(spans(result, ASKED), [S2])

    def test_row5_ask_without_requested_does_not_go_to_asked(self):
        result = run(reply(work(2, S2, requested=False, evidence=[ASK_US, OURS_OK])))
        self.assertEqual(spans(result, OURS), [S2])
        self.assertEqual(spans(result, ASKED), [])

    def test_row5_agree_counts_as_acceptance(self):
        result = run(reply(work(2, S2, evidence=[ev("同意", "支援側", 2, S2)])))
        self.assertEqual(spans(result, OURS), [S2])

    def test_row6_only_asked(self):
        result = run(reply(work(1, S1, state="頼まれただけ", evidence=[ASK_US])))
        self.assertEqual(spans(result, ASKED), [S1])
        self.assertEqual(spans(result, OURS), [])

    def test_row7_customer_work_with_mutual_agreement_is_decided(self):
        result = run(
            reply(
                work(
                    3,
                    S3,
                    who="お客様",
                    evidence=[CUSTOMER_OFFER, CUSTOMER_OK, OURS_AGREE],
                )
            )
        )
        self.assertEqual(spans(result, DECIDED), [S3])
        self.assertEqual(spans(result, OURS), [])

    def test_row7_needs_the_customers_own_acceptance(self):
        result = run(
            reply(work(3, S3, who="お客様", evidence=[CUSTOMER_OFFER, OURS_AGREE]))
        )
        self.assertEqual(spans(result, DECIDED), [])
        self.assertEqual(reasons(result), ["根拠なし"])

    def test_row8_decision_with_mutual_agreement_is_decided(self):
        result = run(
            reply(decision(3, S3, evidence=[CUSTOMER_OFFER, OURS_AGREE]))
        )
        self.assertEqual(spans(result, DECIDED), [S3])

    def test_row8_mutual_in_either_direction(self):
        # 支援側が依頼 → お客様が引き受け。
        result = run(
            reply(
                decision(
                    3,
                    S3,
                    evidence=[
                        ev("依頼", "支援側", 2, S2, to="お客様"),
                        ev("同意", "お客様", 3, S3),
                    ],
                )
            )
        )
        self.assertEqual(spans(result, DECIDED), [S3])

    def test_row9_without_evidence(self):
        result = run(reply(work(2, S2, evidence=[])))
        self.assertEqual(spans(result, UNDECIDED), [S2])
        self.assertEqual(reasons(result), ["根拠なし"])

    def test_row9_proposal_states(self):
        result = run(
            reply(
                work(2, S2, state="提案中"),
                decision(6, S6, state="片方の提案"),
            )
        )
        self.assertEqual(reasons(result), ["提案中", "片方の提案"])

    def test_row9_customer_work_only_asked(self):
        result = run(reply(work(3, S3, who="お客様", state="頼まれただけ")))
        self.assertEqual(reasons(result), ["お客様の未確定"])

    def test_row9_unknown_party_is_not_used_to_confirm(self):
        result = run(
            reply(work(2, S2, evidence=[ev("引き受け", "不明", 2, S2)]))
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(reasons(result), ["当事者不明"])

    def test_row9_unknown_to_of_the_ask_is_not_used(self):
        result = run(
            reply(
                decision(
                    3,
                    S3,
                    evidence=[
                        ev("申し出", "お客様", 3, S3, to="不明"),
                        OURS_AGREE,
                    ],
                )
            )
        )
        self.assertEqual(spans(result, DECIDED), [])
        self.assertEqual(reasons(result), ["当事者不明"])

    def test_row9_valid_withdrawal_goes_first(self):
        masked = make_masked(("チラシは私が作ります。", "やっぱりチラシの話はなしにしてください。"))
        result = run(
            reply(
                work(
                    1,
                    S2,
                    evidence=[
                        ev("引き受け", "支援側", 1, S2),
                        ev("撤回", "お客様", 2, S9),
                    ],
                )
            ),
            masked,
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(spans(result, UNDECIDED), [S2])
        self.assertEqual(reasons(result), ["撤回"])


class ConflictTest(unittest.TestCase):
    """ぶつかる例."""

    def test_hold_with_evidence_beats_acceptance(self):
        result = run(
            reply(
                work(
                    2,
                    S2,
                    state="保留",
                    evidence=[OURS_OK, ev("保留", "支援側", 7, S7)],
                )
            )
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(reasons(result), ["保留"])

    def test_requested_without_ask_basis_is_not_listed_as_asked(self):
        result = run(reply(work(2, S2, requested=True, evidence=[OURS_OK])))
        self.assertEqual(spans(result, OURS), [S2])
        self.assertEqual(spans(result, ASKED), [])
        self.assertEqual(
            [n["reason"] for n in result["notes"]], ["依頼の根拠なし"]
        )

    def test_region_reply_cannot_be_evidence(self):
        masked = make_masked(replies=[4])
        result = run(
            reply(work(2, S2, evidence=[ev("引き受け", "支援側", 4, S4)])),
            masked,
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(reasons(result), ["根拠なし"])
        self.assertEqual(drop_reasons(result, "evidence"), ["応答型"])

    def test_speaker_contradicting_speakers_invalidates_evidence(self):
        result = run(
            reply(
                work(2, S2, evidence=[OURS_OK]),
                speakers=[{"from": 1, "to": 3, "who": "お客様"}],
            )
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(drop_reasons(result, "evidence"), ["話し手と矛盾"])

    def test_speaker_unknown_in_speakers_is_not_a_contradiction(self):
        result = run(
            reply(
                work(2, S2, evidence=[OURS_OK]),
                speakers=[{"from": 1, "to": 3, "who": "不明"}],
            )
        )
        self.assertEqual(spans(result, OURS), [S2])

    def test_matching_speakers_keep_the_evidence(self):
        result = run(
            reply(
                work(2, S2, evidence=[OURS_OK]),
                speakers=[{"from": 1, "to": 1, "who": "お客様"}, {"from": 2, "to": 2, "who": "支援側"}],
            )
        )
        self.assertEqual(spans(result, OURS), [S2])


class NegativeEvidenceTest(unittest.TestCase):
    """根拠として認めない例."""

    def test_customer_asked_a_third_party(self):
        result = run(
            reply(
                work(
                    5,
                    S5,
                    state="頼まれただけ",
                    requested=True,
                    evidence=[ev("依頼", "お客様", 5, S5, to="第三者")],
                )
            )
        )
        self.assertEqual(spans(result, ASKED), [])
        self.assertEqual(spans(result, UNDECIDED), [S5])
        self.assertEqual(reasons(result), ["根拠なし"])
        self.assertEqual([n["reason"] for n in result["notes"]], ["依頼の根拠なし"])

    def test_third_party_ask_plus_our_nod_is_not_an_agreement(self):
        result = run(
            reply(
                decision(
                    5,
                    S5,
                    evidence=[
                        ev("依頼", "お客様", 5, S5, to="第三者"),
                        ev("同意", "支援側", 6, S6),
                    ],
                )
            )
        )
        self.assertEqual(spans(result, DECIDED), [])

    def test_we_asked_the_customer_is_not_a_request_to_us(self):
        result = run(
            reply(
                work(
                    3,
                    S3,
                    state="頼まれただけ",
                    evidence=[ev("依頼", "支援側", 2, S2, to="お客様")],
                )
            )
        )
        self.assertEqual(spans(result, ASKED), [])
        self.assertEqual(spans(result, UNDECIDED), [S3])

    def test_one_sided_offer_is_not_an_agreement(self):
        result = run(reply(decision(3, S3, evidence=[CUSTOMER_OFFER])))
        self.assertEqual(spans(result, DECIDED), [])
        self.assertEqual(reasons(result), ["根拠なし"])

    def test_agreement_to_another_matter_out_of_range(self):
        masked = make_masked(tuple(f"チラシの件その{n}です。" for n in range(1, 9)))
        far = ev("同意", "支援側", 8, "チラシの件その8です")
        result = run(reply(work(1, "チラシの件その1です", evidence=[far])), masked)
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(drop_reasons(result, "evidence"), ["範囲外"])

    def test_evidence_five_sentences_away_is_still_valid(self):
        masked = make_masked(tuple(f"チラシの件その{n}です。" for n in range(1, 9)))
        near = ev("引き受け", "支援側", 6, "チラシの件その6です")
        result = run(reply(work(1, "チラシの件その1です", evidence=[near])), masked)
        self.assertEqual(spans(result, OURS), ["チラシの件その1です"])

    def test_agreement_with_another_quote_is_not_part_of_the_sentence(self):
        result = run(
            reply(
                work(2, S2, evidence=[ev("同意", "支援側", 4, "別件の話を進めましょう")])
            )
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(drop_reasons(result, "evidence"), ["部分でない"])

    def test_evidence_in_a_hidden_sentence_is_invalid(self):
        masked = make_masked(hidden=[2])
        result = run(
            reply(work(3, S3, evidence=[ev("引き受け", "支援側", 2, "チラシは私が作ります")])),
            masked,
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(drop_reasons(result, "evidence"), ["伏せた文"])

    def test_label_only_quote_is_not_evidence(self):
        masked = make_masked(("〈内情〉のため、チラシをお願いします。", "はい、作ります。"))
        for bad in ("〈内情〉", "内情", "〈内"):
            with self.subTest(span=bad):
                result = run(
                    reply(
                        work(
                            2,
                            "作ります",
                            evidence=[ev("引き受け", "支援側", 1, bad)],
                        )
                    ),
                    masked,
                )
                self.assertEqual(spans(result, OURS), [])
                self.assertEqual(drop_reasons(result, "evidence"), ["札だけ"])
        ok = run(
            reply(
                work(
                    2,
                    "作ります",
                    evidence=[ev("引き受け", "支援側", 1, "〈内情〉のため、チラシをお願い")],
                )
            ),
            masked,
        )
        self.assertEqual(spans(ok, OURS), ["作ります"])

    def test_later_withdrawal(self):
        masked = make_masked(("チラシは私が作ります。", "やっぱりチラシの話はなしにしてください。"))
        result = run(
            reply(
                decision(
                    1,
                    S2,
                    evidence=[
                        ev("申し出", "支援側", 1, S2, to="お客様"),
                        ev("同意", "お客様", 1, S2),
                        ev("撤回", "お客様", 2, S9),
                    ],
                )
            ),
            masked,
        )
        self.assertEqual(spans(result, DECIDED), [])
        self.assertEqual(reasons(result), ["撤回"])


class ShapeTest(unittest.TestCase):
    """返事の形の点検."""

    def assert_shape_error(self, item: dict[str, Any], count: int = 1, **kwargs: Any):
        result = run(reply(item), **kwargs)
        self.assertEqual(result["shape_errors"], count)
        self.assertEqual(result["items"], [])
        self.assertEqual(result["counts"], {"形崩れ": count})
        return result

    def test_sent_as_bool_is_a_shape_error(self):
        item = plain("困りごと", 1, S1)
        item["sent"] = True
        self.assert_shape_error(item)

    def test_topic_and_rank_as_bool_are_shape_errors(self):
        for key in ("topic", "rank"):
            with self.subTest(key=key):
                item = plain("困りごと", 8, S8)
                item[key] = True
                self.assert_shape_error(item)

    def test_decision_with_who_is_a_shape_error(self):
        item = decision(3, S3)
        item["who"] = "支援側"
        self.assert_shape_error(item)

    def test_trouble_and_undecided_with_who_or_state_are_shape_errors(self):
        for kind in ("困りごと", "未決"):
            for key, value in (("who", "不明"), ("state", "保留")):
                with self.subTest(kind=kind, key=key):
                    item = plain(kind, 8, S8)
                    item[key] = value
                    self.assert_shape_error(item)

    def test_work_without_requested_is_a_shape_error(self):
        item = work(2, S2)
        del item["requested"]
        self.assert_shape_error(item)

    def test_work_with_bad_values_is_a_shape_error(self):
        for key, value in (("who", "社長"), ("state", "完了"), ("requested", "true")):
            with self.subTest(key=key):
                item = work(2, S2)
                item[key] = value
                self.assert_shape_error(item)

    def test_bad_kind_and_non_dict_item(self):
        result = run({"items": [{"kind": "欄", "sent": 1}, "文字列", None]})
        self.assertEqual(result["shape_errors"], 3)

    def test_items_must_be_a_list(self):
        result = run({"items": {"a": 1}})
        self.assertEqual(result["shape_errors"], 1)
        self.assertEqual(result["dropped"][0]["scope"], "reply")

    def test_bad_decision_state(self):
        self.assert_shape_error(decision(3, S3, state="提案中"))

    def test_item_span_must_be_part_of_the_sentence_and_short(self):
        for span, reason in (
            ("", "部分でない"),
            ("   ", "部分でない"),
            ("存在しない一節", "部分でない"),
        ):
            with self.subTest(span=span):
                result = run(reply(plain("困りごと", 8, span)))
                self.assertEqual(drop_reasons(result), [reason])
        long_text = "あ" * 130 + "。"
        masked = make_masked((long_text,))
        result = run(reply(plain("困りごと", 1, "あ" * 121)), masked)
        self.assertEqual(drop_reasons(result), ["長すぎ"])
        ok = run(reply(plain("困りごと", 1, "あ" * 120)), masked)
        self.assertEqual(len(ok["columns"][TROUBLE]), 1)

    def test_sent_must_exist(self):
        for sent in (0, 11, -1):
            with self.subTest(sent=sent):
                result = run(reply(plain("困りごと", sent, S8)))
                self.assertEqual(drop_reasons(result), ["番号違い"])

    def test_hidden_sentence_is_dropped(self):
        masked = make_masked(hidden=[8])
        result = run(reply(plain("困りごと", 8, "この文は伏せました")), masked)
        self.assertEqual(drop_reasons(result), ["伏せた文"])
        self.assertEqual(result["items"], [])

    def test_label_only_item_quote_is_dropped(self):
        masked = make_masked(("〈内情〉のため、チラシをお願いします。",))
        result = run(reply(plain("未決", 1, "〈内情〉")), masked)
        self.assertEqual(drop_reasons(result), ["札だけ"])

    def test_region_trouble_uses_the_whole_tag(self):
        masked = make_masked(regions=[(1, 2, 3, TAG)])
        item = {"kind": "困りごと", "region": 1, "topic": 1, "rank": 1}
        result = run(reply(item), masked)
        self.assertEqual(spans(result, TROUBLE), [TAG])
        self.assertIn("- [不明] " + TAG, result["shared"])
        self.assertIsNone(result["columns"][TROUBLE][0]["sent"])

    def test_region_with_sent_or_span_is_a_shape_error(self):
        masked = make_masked(regions=[(1, 2, 3, TAG)])
        for extra in ({"sent": 2}, {"span": "チラシ"}, {"sent": 2, "span": "チラシ"}):
            with self.subTest(extra=extra):
                item = {"kind": "困りごと", "region": 1, "topic": 1, "rank": 1, **extra}
                result = run(reply(item), masked)
                self.assertEqual(result["shape_errors"], 1)
                self.assertEqual(result["items"], [])

    def test_region_only_for_trouble_and_must_exist(self):
        masked = make_masked(regions=[(1, 2, 3, TAG)])
        undecided = {"kind": "未決", "region": 1, "topic": 1, "rank": 1}
        self.assertEqual(run(reply(undecided), masked)["shape_errors"], 1)
        missing = {"kind": "困りごと", "region": 9, "topic": 1, "rank": 1}
        self.assertEqual(drop_reasons(run(reply(missing), masked)), ["番号違い"])
        bool_region = {"kind": "困りごと", "region": True, "topic": 1, "rank": 1}
        self.assertEqual(run(reply(bool_region), masked)["shape_errors"], 1)

    def test_malformed_evidence_is_dropped_but_the_item_stays(self):
        bad_missing_to = ev("依頼", "お客様", 1, S1, to=None)
        bad_type = ev("相談", "お客様", 1, S1)
        bad_sent = ev("依頼", "お客様", 1, S1, to="支援側")
        bad_sent["sent"] = True
        result = run(
            reply(
                work(
                    2,
                    S2,
                    evidence=[bad_missing_to, bad_type, bad_sent, "文字列", OURS_OK],
                )
            )
        )
        self.assertEqual(spans(result, OURS), [S2])
        self.assertEqual(drop_reasons(result, "evidence"), ["形崩れ"] * 4)
        self.assertEqual(result["shape_errors"], 4)

    def test_evidence_must_be_a_list(self):
        item = work(2, S2)
        item["evidence"] = "なし"
        self.assert_shape_error(item)

    def test_to_is_optional_for_accept_and_agree(self):
        result = run(reply(work(2, S2, evidence=[ev("引き受け", "支援側", 2, S2, to=None)])))
        self.assertEqual(spans(result, OURS), [S2])

    def test_speakers_with_bad_ranges_are_ignored_and_recorded(self):
        result = run(
            reply(
                speakers=[
                    {"from": 1, "to": 3, "who": "お客様"},
                    {"from": 3, "to": 5, "who": "支援側"},  # 重なり
                    {"from": 6, "to": 4, "who": "支援側"},  # 逆順
                    {"from": 9, "to": 12, "who": "支援側"},  # 範囲外
                    {"from": 0, "to": 2, "who": "支援側"},  # 範囲外
                    {"from": 4, "to": 5, "who": "社長"},  # 値違い
                    {"from": True, "to": 5, "who": "支援側"},  # 型違い
                    "文字列",
                    {"from": 4, "to": 6, "who": "支援側"},
                ]
            )
        )
        self.assertEqual(
            result["speakers"],
            [{"from": 1, "to": 3, "who": "お客様"}, {"from": 4, "to": 6, "who": "支援側"}],
        )
        self.assertEqual(drop_reasons(result, "speakers"), ["話し手の形"] * 7)
        self.assertEqual(result["shape_errors"], 0)

    def test_speakers_not_a_list_or_missing(self):
        result = run({"speakers": "なし", "items": []})
        self.assertEqual(drop_reasons(result, "speakers"), ["話し手の形"])
        self.assertEqual(run({"items": []})["dropped"], [])


class LineLimitTest(unittest.TestCase):
    """欄ごとの行数の上限."""

    @staticmethod
    def many(n: int) -> MaskedR3:
        return make_masked(tuple(f"案件その{i}を進めます。" for i in range(1, n + 1)))

    def test_three_columns_are_not_cut_above_eight_rows(self):
        masked = self.many(12)
        items = []
        for i in range(1, 10):
            s = f"案件その{i}を進めます"
            items.append(
                decision(
                    i,
                    s,
                    evidence=[
                        ev("申し出", "お客様", i, s, to="支援側"),
                        ev("同意", "支援側", i, s),
                    ],
                )
            )
        for i in range(10, 13):
            s = f"案件その{i}を進めます"
            items.append(
                work(i, s, requested=True, evidence=[ev("依頼", "お客様", i, s, to="支援側"), ev("引き受け", "支援側", i, s)])
            )
        result = run(reply(*items), masked)
        self.assertEqual(len(result["columns"][DECIDED]), 9)
        self.assertEqual(len(result["columns"][OURS]), 3)
        self.assertEqual(len(result["columns"][ASKED]), 3)
        self.assertEqual(result["dropped"], [])
        notes = [n for n in result["notes"] if n["reason"] == "多すぎ"]
        self.assertEqual([(n["field"], n["count"]) for n in notes], [(DECIDED, 9)])

    def test_nine_work_rows_in_ours_all_come_out_with_a_note(self):
        masked = self.many(9)
        items = [
            work(i, f"案件その{i}を進めます", evidence=[ev("引き受け", "支援側", i, f"案件その{i}を進めます")])
            for i in range(1, 10)
        ]
        result = run(reply(*items), masked)
        self.assertEqual(len(result["columns"][OURS]), 9)
        self.assertEqual(
            [(n["field"], n["count"]) for n in result["notes"]], [(OURS, 9)]
        )

    def test_trouble_rows_are_cut_to_eight_by_rank(self):
        masked = self.many(10)
        items = [
            plain("困りごと", i, f"案件その{i}を進めます", topic=1, rank=11 - i)
            for i in range(1, 11)
        ]
        result = run(reply(*items), masked)
        kept = [row["sent"] for row in result["columns"][TROUBLE]]
        self.assertEqual(kept, [10, 9, 8, 7, 6, 5, 4, 3])  # rank 1..8
        self.assertEqual(
            [(d["field"], d["sent"], d["reason"]) for d in result["dropped"]],
            [(TROUBLE, 2, "行数"), (TROUBLE, 1, "行数")],
        )
        self.assertTrue(all("span" not in d for d in result["dropped"]))

    def test_rows_moved_to_undecided_are_not_cut(self):
        masked = self.many(14)
        items = [
            plain("未決", i, f"案件その{i}を進めます", rank=i) for i in range(1, 11)
        ]
        items += [
            work(i, f"案件その{i}を進めます", state="提案中", rank=1)
            for i in range(11, 15)
        ]
        result = run(reply(*items), masked)
        column = result["columns"][UNDECIDED]
        self.assertEqual(len(column), 8 + 4)
        self.assertEqual(
            [row["sent"] for row in column], [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]
        )
        self.assertEqual([d["sent"] for d in result["dropped"]], [9, 10])

    def test_same_topic_is_not_merged(self):
        masked = self.many(3)
        items = [plain("困りごと", i, f"案件その{i}を進めます", topic=1, rank=i) for i in (1, 2, 3)]
        self.assertEqual(len(run(reply(*items), masked)["columns"][TROUBLE]), 3)

    def test_identical_rows_in_the_same_column_become_one(self):
        result = run(reply(plain("困りごと", 8, S8, rank=1), plain("困りごと", 8, S8, rank=2)))
        self.assertEqual(len(result["columns"][TROUBLE]), 1)

    def test_rows_differing_in_state_are_kept(self):
        result = run(
            reply(
                work(2, S2, state="提案中"),
                work(2, S2, state="頼まれただけ"),
            )
        )
        self.assertEqual(len(result["columns"][UNDECIDED]), 2)

    def test_row_in_both_asked_and_ours_is_not_removed(self):
        result = run(reply(work(2, S2, requested=True, evidence=[ASK_US, OURS_OK])))
        self.assertEqual(len(result["columns"][ASKED]), 1)
        self.assertEqual(len(result["columns"][OURS]), 1)
        self.assertEqual(len(result["items"]), 2)


class OfferAgreedTest(unittest.TestCase):
    """支援側の申し出に、お客様の返事がそろったときの限定的な決まり."""

    SENTS = (
        "写真を撮りましょうか。",  # 1 支援側の申し出
        "はい、お願いします。",  # 2 お客様の返事
        "チラシも作っておきますね。",  # 3 支援側の引き受け
        "それは印刷会社にお願いします。",  # 4 お客様が第三者へ
        "ええ、そうですね。",  # 5
        "一旦保留にしましょう。",  # 6
        "来週までに送ります。",  # 7
    )
    OFFER = ev("申し出", "支援側", 1, "写真を撮りましょうか", to="お客様")

    def masked(self) -> MaskedR3:
        return make_masked(self.SENTS)

    def item(self, *evidence, **kwargs):
        return work(1, "写真を撮りましょうか", evidence=list(evidence), **kwargs)

    def test_positive_offer_with_customer_agree(self):
        result = run(
            reply(self.item(self.OFFER, ev("同意", "お客様", 2, "はい、お願いします"))),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), ["写真を撮りましょうか"])
        self.assertEqual(spans(result, UNDECIDED), [])
        note = [n for n in result["notes"] if n["reason"] == "申し出と同意で確定"]
        self.assertEqual([(n["sent"], n["via"]) for n in note], [(1, "同意")])

    def test_positive_offer_with_customer_accept(self):
        result = run(
            reply(self.item(self.OFFER, ev("引き受け", "お客様", 2, "はい、お願いします"))),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), ["写真を撮りましょうか"])

    def test_positive_customer_please_to_us_is_a_reply(self):
        # 「お願いします」を 依頼（お客様→支援側）で返してきた場合も、返事として数える。
        result = run(
            reply(
                self.item(
                    self.OFFER,
                    ev("依頼", "お客様", 2, "はい、お願いします", to="支援側"),
                    requested=True,
                )
            ),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), ["写真を撮りましょうか"])
        self.assertEqual(spans(result, ASKED), ["写真を撮りましょうか"])  # 頼まれていることにも
        self.assertEqual(result["notes"][0]["via"], "依頼")

    def test_state_proposed_still_goes_to_ours_with_the_note(self):
        result = run(
            reply(
                self.item(
                    self.OFFER,
                    ev("同意", "お客様", 2, "はい、お願いします"),
                    state="提案中",
                )
            ),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), ["写真を撮りましょうか"])
        self.assertEqual(spans(result, UNDECIDED), [])
        self.assertEqual(result["notes"][0]["state"], "提案中")

    def test_negative_offer_without_reply_stays_undecided(self):
        result = run(
            reply(self.item(self.OFFER, state="提案中")),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(spans(result, UNDECIDED), ["写真を撮りましょうか"])
        self.assertEqual(reasons(result), ["提案中"])

    def test_negative_customer_agree_for_another_thing_out_of_range(self):
        # 返事の文が前後5文の外（別件の同意）なら根拠から外れ、未決のまま。
        sents = self.SENTS + ("別の話です。",) * 5 + ("はい、お願いします。",)
        masked = make_masked(sents)
        result = run(
            reply(self.item(self.OFFER, ev("同意", "お客様", 13, "はい、お願いします"), state="提案中")),
            masked,
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertIn("範囲外", drop_reasons(result, "evidence"))
        self.assertEqual(reasons(result), ["提案中"])

    def test_negative_customer_agree_before_the_offer_is_not_a_reply(self):
        offer = ev("申し出", "支援側", 3, "チラシも作っておきますね", to="お客様")
        agree = ev("同意", "お客様", 2, "はい、お願いします")
        item = work(3, "チラシも作っておきますね", state="提案中", evidence=[offer, agree])
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(reasons(result), ["提案中"])

    def test_negative_customer_agrees_with_a_third_party(self):
        result = run(
            reply(
                self.item(
                    self.OFFER,
                    ev("同意", "お客様", 4, "それは印刷会社にお願いします", to="第三者"),
                    state="提案中",
                )
            ),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(reasons(result), ["提案中"])

    def test_negative_customer_asks_a_third_party(self):
        result = run(
            reply(
                self.item(
                    self.OFFER,
                    ev("依頼", "お客様", 4, "それは印刷会社にお願いします", to="第三者"),
                    state="提案中",
                )
            ),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), [])

    def test_negative_supports_own_agree_is_not_the_customers_reply(self):
        result = run(
            reply(
                self.item(self.OFFER, ev("同意", "支援側", 5, "ええ、そうですね"), state="提案中")
            ),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), [])

    def test_negative_offer_by_customer_does_not_count(self):
        offer = ev("申し出", "お客様", 1, "写真を撮りましょうか", to="支援側")
        result = run(
            reply(self.item(offer, ev("同意", "お客様", 2, "はい、お願いします"), state="提案中")),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), [])

    def test_negative_offer_to_a_third_party_does_not_count(self):
        offer = ev("申し出", "支援側", 1, "写真を撮りましょうか", to="第三者")
        result = run(
            reply(self.item(offer, ev("同意", "お客様", 2, "はい、お願いします"), state="提案中")),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), [])

    def test_negative_speaker_conflict_drops_the_reply(self):
        # 話し手の範囲でその文は支援側。お客様の返事は無効になり、未決のまま。
        speakers = [{"from": 1, "to": 2, "who": "支援側"}]
        result = run(
            reply(
                self.item(self.OFFER, ev("同意", "お客様", 2, "はい、お願いします"), state="提案中"),
                speakers=speakers,
            ),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertIn("話し手と矛盾", drop_reasons(result, "evidence"))

    def test_negative_hold_withdrawal_and_unknown_who_keep_their_rows(self):
        agree = ev("同意", "お客様", 2, "はい、お願いします")
        held = run(
            reply(self.item(self.OFFER, agree, ev("保留", "お客様", 6, "一旦保留にしましょう"))),
            self.masked(),
        )
        self.assertEqual(spans(held, OURS), [])
        withdrawn = run(
            reply(self.item(self.OFFER, agree, ev("撤回", "お客様", 6, "一旦保留にしましょう"))),
            self.masked(),
        )
        self.assertEqual(reasons(withdrawn), ["撤回"])
        unknown = run(
            reply(self.item(self.OFFER, agree, who="不明")),
            self.masked(),
        )
        self.assertEqual(spans(unknown, OURS), [])
        self.assertTrue(unknown["columns"][UNDECIDED][0]["unassigned"])

    def test_negative_customer_work_is_not_moved(self):
        # 作業する人がお客様なら、この決まりは当てない（行7の双方の合意の決まりのまま）。
        result = run(
            reply(
                self.item(
                    self.OFFER,
                    ev("同意", "お客様", 2, "はい、お願いします"),
                    who="お客様",
                    state="提案中",
                )
            ),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(reasons(result), ["提案中"])

    def test_existing_row5_is_unchanged(self):
        result = run(
            reply(work(3, "チラシも作っておきますね", evidence=[ev("引き受け", "支援側", 3, "チラシも作っておきますね")])),
            self.masked(),
        )
        self.assertEqual(spans(result, OURS), ["チラシも作っておきますね"])
        self.assertEqual(result["notes"], [])


class FarAskTest(unittest.TestCase):
    """作業の、お客様から支援側への依頼だけは前後5文の外でも根拠にできる."""

    SENTS = (
        ("見積を決まった形にしたいんです。",)  # 1 お客様の希望（遠い）
        + tuple(f"別の話その{n}です。" for n in range(2, 12))
        + (
            "見積の仕組みは私が作ります。",  # 12 支援側の引き受け
            "見積の形はこれで決めましょう。",  # 13 決めごと
            "はい、お願いします。",  # 14
            "見積のひな形も作りましょうか。",  # 15 支援側の申し出
        )
    )
    FAR_ASK = ev("依頼", "お客様", 1, "見積を決まった形にしたいんです", to="支援側")

    def masked(self) -> MaskedR3:
        return make_masked(self.SENTS)

    def test_far_customer_ask_puts_accepted_task_in_both_columns(self):
        item = work(
            12,
            "見積の仕組みは私が作ります",
            requested=True,
            evidence=[self.FAR_ASK, ev("引き受け", "支援側", 12, "見積の仕組みは私が作ります")],
        )
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, OURS), ["見積の仕組みは私が作ります"])
        self.assertEqual(spans(result, ASKED), ["見積の仕組みは私が作ります"])
        self.assertEqual(drop_reasons(result, "evidence"), [])

    def test_far_ask_without_requested_does_not_go_to_asked(self):
        item = work(
            12,
            "見積の仕組みは私が作ります",
            requested=False,
            evidence=[self.FAR_ASK, ev("引き受け", "支援側", 12, "見積の仕組みは私が作ります")],
        )
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, OURS), ["見積の仕組みは私が作ります"])
        self.assertEqual(spans(result, ASKED), [])

    def test_far_ask_from_ours_is_still_out_of_range(self):
        far = ev("依頼", "支援側", 1, "見積を決まった形にしたいんです", to="お客様")
        item = work(
            12, "見積の仕組みは私が作ります", requested=True,
            evidence=[far, ev("引き受け", "支援側", 12, "見積の仕組みは私が作ります")],
        )
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, ASKED), [])
        self.assertEqual(drop_reasons(result, "evidence"), ["範囲外"])

    def test_far_ask_does_not_make_a_decision_mutual(self):
        item = decision(
            13,
            "見積の形はこれで決めましょう",
            evidence=[self.FAR_ASK, ev("同意", "支援側", 13, "見積の形はこれで決めましょう")],
        )
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, DECIDED), [])
        self.assertEqual(drop_reasons(result, "evidence"), ["範囲外"])

    def test_far_ask_is_not_a_reply_to_an_offer(self):
        # 遠い依頼は、申し出への返事にも双方の合意にも使わない。
        offer = ev("申し出", "支援側", 15, "見積のひな形も作りましょうか", to="お客様")
        item = work(
            15, "見積のひな形も作りましょうか", state="提案中", requested=True,
            evidence=[offer, self.FAR_ASK],
        )
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, OURS), [])
        # 頼みに提案で返しただけ → 頼まれていることと、まだ決まっていないこと。
        self.assertEqual(spans(result, ASKED), ["見積のひな形も作りましょうか"])
        self.assertEqual(spans(result, UNDECIDED), ["見積のひな形も作りましょうか"])
        self.assertEqual(reasons(result), ["提案中"])

    def test_offer_answered_with_a_request_goes_to_both_columns(self):
        offer = ev("申し出", "支援側", 13, "見積の形はこれで決めましょう", to="お客様")
        answer = ev("依頼", "お客様", 14, "はい、お願いします", to="支援側")
        item = work(
            13, "見積の形はこれで決めましょう", state="提案中", requested=True,
            evidence=[offer, answer],
        )
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, OURS), ["見積の形はこれで決めましょう"])
        self.assertEqual(spans(result, ASKED), ["見積の形はこれで決めましょう"])

    def test_offer_answered_with_agree_stays_ours_only(self):
        offer = ev("申し出", "支援側", 13, "見積の形はこれで決めましょう", to="お客様")
        answer = ev("同意", "お客様", 14, "はい、お願いします")
        item = work(
            13, "見積の形はこれで決めましょう", state="提案中",
            evidence=[offer, answer],
        )
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, OURS), ["見積の形はこれで決めましょう"])
        self.assertEqual(spans(result, ASKED), [])


class GuardAndNameTest(unittest.TestCase):
    """弱い同意・質問の形を根拠にしない、頼みだけの作業、作業名（name）."""

    SENTS = (
        "ホームページ用の写真を撮りましょうか。",  # 1 支援側の申し出
        "あ、それ、いいかも。",  # 2 弱い同意
        "はい、それでお願いします。",  # 3 受け入れ
        "許可が要りますよね。",  # 4 質問
        "撮り直しもしてもらえますか。",  # 5 頼む言葉のある質問
        "新しいドメインを取りましょうか。",  # 6
    )

    def masked(self) -> MaskedR3:
        return make_masked(self.SENTS)

    OFFER = ev("申し出", "支援側", 1, "ホームページ用の写真を撮りましょうか", to="お客様")

    def test_weak_agree_does_not_confirm_an_offer(self):
        item = work(1, "ホームページ用の写真を撮りましょうか", state="提案中",
                    evidence=[self.OFFER, ev("同意", "お客様", 2, "あ、それ、いいかも")])
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(drop_reasons(result, "evidence"), ["弱い同意"])

    def test_plain_agree_still_confirms(self):
        item = work(1, "ホームページ用の写真を撮りましょうか", state="提案中",
                    evidence=[self.OFFER, ev("同意", "お客様", 3, "はい、それでお願いします")])
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, OURS), ["ホームページ用の写真を撮りましょうか"])

    def test_question_is_not_a_request(self):
        item = work(4, "許可が要りますよね", state="頼まれただけ", requested=True,
                    evidence=[ev("依頼", "お客様", 4, "許可が要りますよね", to="支援側")])
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, ASKED), [])
        self.assertEqual(drop_reasons(result, "evidence"), ["質問の形"])

    def test_question_with_request_words_is_a_request(self):
        item = work(5, "撮り直しもしてもらえますか", state="頼まれただけ", requested=True,
                    evidence=[ev("依頼", "お客様", 5, "撮り直しもしてもらえますか", to="支援側")])
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, ASKED), ["撮り直しもしてもらえますか"])

    def test_customer_proposal_in_question_form_is_kept(self):
        masked = make_masked(("月曜の午前中が空いてるので、そこでどうですか。", "じゃあそこで。"))
        item = decision(1, "そこでどうですか", evidence=[
            ev("申し出", "お客様", 1, "月曜の午前中が空いてるので、そこでどうですか", to="支援側"),
            ev("同意", "支援側", 2, "じゃあそこで"),
        ])
        result = run(reply(item), masked)
        self.assertEqual(spans(result, DECIDED), ["そこでどうですか"])
        self.assertEqual(drop_reasons(result, "evidence"), [])

    def test_support_side_question_offer_is_kept(self):
        # 支援側の「〜しましょうか」は質問の形でも申し出のまま。
        item = work(1, "ホームページ用の写真を撮りましょうか", state="提案中",
                    evidence=[self.OFFER, ev("同意", "お客様", 3, "はい、それでお願いします")])
        result = run(reply(item), self.masked())
        self.assertEqual(drop_reasons(result, "evidence"), [])

    def test_request_without_acceptance_goes_to_asked_and_undecided(self):
        item = work(5, "撮り直しもしてもらえますか", state="引き受けた", requested=True,
                    evidence=[ev("依頼", "お客様", 5, "撮り直しもしてもらえますか", to="支援側")])
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, ASKED), ["撮り直しもしてもらえますか"])
        self.assertEqual(spans(result, OURS), [])
        self.assertEqual(spans(result, UNDECIDED), ["撮り直しもしてもらえますか"])
        self.assertEqual(reasons(result), ["根拠なし"])

    def test_proposed_after_request_goes_to_asked_and_undecided(self):
        ask = ev("依頼", "お客様", 5, "撮り直しもしてもらえますか", to="支援側")
        item = work(6, "新しいドメインを取りましょうか", state="提案中", requested=True,
                    evidence=[ask])
        result = run(reply(item), self.masked())
        self.assertEqual(spans(result, ASKED), ["新しいドメインを取りましょうか"])
        self.assertEqual(reasons(result), ["提案中"])

    def _named(self, name):
        item = work(6, "新しいドメインを取りましょうか", state="提案中",
                    evidence=[ev("申し出", "支援側", 6, "新しいドメインを取りましょうか", to="お客様"),
                              ])
        item["name"] = name
        return run(reply(item), self.masked())

    def test_grounded_name_is_shown(self):
        result = self._named("新しいドメインの取得")
        row = result["columns"][UNDECIDED][0]
        self.assertEqual(row["name"], "新しいドメインの取得")
        self.assertIn("新しいドメインの取得「新しいドメインを取りましょうか」", result["shared"])
        self.assertEqual(drop_reasons(result, "name"), [])

    def test_name_with_words_from_nearby_sentences(self):
        # 「ホームページ」「写真」は前後5文の中にある。
        result = self._named("ホームページ用の写真撮影")
        self.assertEqual(result["columns"][UNDECIDED][0]["name"], "ホームページ用の写真撮影")

    def test_name_with_made_up_words_is_dropped(self):
        for bad_name, reason in (
            ("予約システムの導入", "name言葉"),
            ("ドメインを3個取得", "name言葉"),
            ("〈社名〉のドメイン取得", "name禁止文字"),
            ("あ" * 41, "name形"),
            (5, "name形"),
        ):
            with self.subTest(name=bad_name):
                result = self._named(bad_name)
                self.assertEqual(result["columns"][UNDECIDED][0]["name"], "")
                self.assertEqual(drop_reasons(result, "name"), [reason])
                self.assertNotIn("「", result["shared"].split("【まだ決まっていないこと】")[1])

    def test_name_on_a_decision_is_not_allowed(self):
        item = decision(3, "はい、それでお願いします")
        item["name"] = "写真"
        result = run(reply(item), self.masked())
        self.assertEqual(drop_reasons(result, "name"), ["name不可"])


class TroubleRowsTest(unittest.TestCase):
    """困りごとの8行。区間の札だけの行を後ろに回し、同じ文・同じ区間は1行に."""

    TAG_EMPTY = "〈伏せた区間〉"

    @staticmethod
    def sentences(n: int = 12) -> tuple[str, ...]:
        return tuple(f"困りごとその{i}です。" for i in range(1, n + 1))

    @staticmethod
    def region_item(region: int, rank: int) -> dict[str, Any]:
        return {"kind": "困りごと", "region": region, "topic": 1, "rank": rank}

    def test_region_tag_rows_go_behind_sentence_rows(self):
        regions = [(1, 11, 11, TAG), (2, 12, 12, self.TAG_EMPTY)]
        masked = make_masked(self.sentences(), regions=regions)
        items = [
            self.region_item(1, 1),
            self.region_item(2, 2),
        ] + [plain("困りごと", i, f"困りごとその{i}です", rank=2 + i) for i in range(1, 9)]
        result = run(reply(*items), masked)
        rows = result["columns"][TROUBLE]
        self.assertEqual([r["sent"] for r in rows], list(range(1, 9)))
        self.assertEqual(
            sorted((d["sent"], d["reason"]) for d in result["dropped"] if d["scope"] == "line"),
            [(None, "行数"), (None, "行数")],
        )

    def test_region_rows_still_come_out_when_there_is_room(self):
        regions = [(1, 11, 11, TAG), (2, 12, 12, self.TAG_EMPTY)]
        masked = make_masked(self.sentences(), regions=regions)
        items = [self.region_item(2, 1), self.region_item(1, 2)] + [
            plain("困りごと", i, f"困りごとその{i}です", rank=2 + i) for i in range(1, 5)
        ]
        rows = run(reply(*items), masked)["columns"][TROUBLE]
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["region"] for r in rows[:2]], [2, 1])  # 順位どおりに並べる

    def test_tagged_region_goes_before_the_empty_one_when_cut(self):
        regions = [(1, 11, 11, TAG), (2, 12, 12, self.TAG_EMPTY)]
        masked = make_masked(self.sentences(), regions=regions)
        items = [self.region_item(2, 1), self.region_item(1, 2)] + [
            plain("困りごと", i, f"困りごとその{i}です", rank=2 + i) for i in range(1, 7)
        ]
        rows = run(reply(*items), masked)["columns"][TROUBLE]
        self.assertEqual(len(rows), 8)
        self.assertEqual(sorted(r["region"] for r in rows if r["region"]), [1, 2])
        items = items[:2] + [
            plain("困りごと", i, f"困りごとその{i}です", rank=2 + i) for i in range(1, 8)
        ]
        rows = run(reply(*items), masked)["columns"][TROUBLE]
        self.assertEqual([r["region"] for r in rows if r["region"]], [1])  # 空の札が落ちる

    def test_same_sentence_becomes_one_row_with_the_better_rank(self):
        masked = make_masked(self.sentences())
        items = [
            plain("困りごと", 3, "困りごとその3です", rank=5),
            plain("困りごと", 3, "困りごと", rank=2),
            plain("困りごと", 4, "困りごとその4です", rank=3),
        ]
        result = run(reply(*items), masked)
        rows = result["columns"][TROUBLE]
        self.assertEqual([(r["sent"], r["rank"], r["span"]) for r in rows],
                         [(3, 2, "困りごと"), (4, 3, "困りごとその4です")])
        self.assertEqual(result["dropped"], [])
        self.assertEqual([n["reason"] for n in result["notes"]], ["困りごと統合"])

    def test_same_region_becomes_one_row(self):
        masked = make_masked(self.sentences(), regions=[(1, 11, 12, TAG)])
        result = run(reply(self.region_item(1, 3), self.region_item(1, 1)), masked)
        rows = result["columns"][TROUBLE]
        self.assertEqual([(r["region"], r["rank"]) for r in rows], [(1, 1)])

    def test_region_row_is_folded_into_a_sentence_row_of_the_same_region(self):
        # 区間 1（11〜12）の応答型の文を指す行があれば、同じ区間の札の行は消す。
        masked = make_masked(self.sentences(), regions=[(1, 11, 12, TAG)], replies=(12,))
        items = [self.region_item(1, 1), plain("困りごと", 12, "困りごとその12です", rank=2)]
        rows = run(reply(*items), masked)["columns"][TROUBLE]
        self.assertEqual([(r["sent"], r["region"]) for r in rows], [(12, None)])

    def test_undecided_and_moved_rows_are_not_merged(self):
        masked = make_masked(self.sentences())
        result = run(
            reply(plain("未決", 3, "困りごとその3です", rank=1), plain("未決", 3, "困りごと", rank=2)),
            masked,
        )
        self.assertEqual(len(result["columns"][UNDECIDED]), 2)


class DetailTest(unittest.TestCase):
    """作業・決めごとの任意の ``detail``（対象の引用）."""

    SENTS = (
        "撮影の件ですが、お願いできますか。",  # 1
        "はい、撮っておきますね。",  # 2
        "店先の看板と、メニューの写真をお願いします。",  # 3 対象
        "あと、スタッフ集合写真も。",  # 4 対象
        "ところで今日は寒いですね。",  # 5
        "そうですね。",  # 6
        "来週の天気は晴れです。",  # 7
        "ところで駅前の工事が長引いています。",  # 8
        "近い文です。",  # 9
    ) + tuple(f"雑談の{i}です。" for i in range(10, 20)) + (
        "遠い文です。",  # 20（項目の文 2 から18文。前後15文の外）
    )
    ASK = ev("依頼", "お客様", 1, "撮影の件ですが、お願いできますか", to="支援側")
    OK = ev("引き受け", "支援側", 2, "撮っておきますね")

    def job(self, detail: Any, **kwargs: Any) -> dict[str, Any]:
        item = work(2, "撮っておきますね", requested=True, evidence=[self.ASK, self.OK], **kwargs)
        if detail is not ...:
            item["detail"] = detail
        return item

    def run_one(self, detail: Any, masked: MaskedR3 | None = None, **kwargs: Any):
        return run(reply(self.job(detail, **kwargs)), masked or make_masked(self.SENTS))

    @staticmethod
    def d(sent: int, span: str) -> dict[str, Any]:
        return {"sent": sent, "span": span}

    def test_positive_two_details_are_kept_and_shown(self):
        result = self.run_one([self.d(3, "店先の看板"), self.d(4, "スタッフ集合写真")])
        row = result["columns"][OURS][0]
        self.assertEqual(
            row["detail"],
            [{"sent": 3, "span": "店先の看板"}, {"sent": 4, "span": "スタッフ集合写真"}],
        )
        self.assertEqual(result["dropped"], [])
        self.assertIn("- [こちら] 撮っておきますね（対象: 店先の看板／スタッフ集合写真）", result["shared"])
        # 頼まれていること欄にも同じ行（detail 付き）が出る
        self.assertEqual(result["columns"][ASKED][0]["detail"], row["detail"])
        self.assertIn("（対象: 店先の看板／スタッフ集合写真）", result["shared"].split("【こちらがやること】")[0])

    def test_one_detail_has_no_separator(self):
        result = self.run_one([self.d(3, "メニューの写真")])
        self.assertIn("- [こちら] 撮っておきますね（対象: メニューの写真）", result["shared"])

    def test_no_detail_or_empty_detail_is_plain(self):
        for detail in (..., [], None):
            with self.subTest(detail=detail):
                result = self.run_one(detail)
                self.assertEqual(result["columns"][OURS][0]["detail"], [])
                self.assertIn("- [こちら] 撮っておきますね\n", result["shared"] + "\n")
                self.assertNotIn("対象", result["shared"])
                self.assertEqual(result["dropped"], [])

    def test_detail_on_a_decision(self):
        masked = make_masked(self.SENTS)
        item = decision(
            3,
            "店先の看板",
            evidence=[ev("申し出", "お客様", 3, "店先の看板", to="支援側"), ev("同意", "支援側", 4, "スタッフ集合写真")],
        )
        item["detail"] = [self.d(4, "スタッフ集合写真")]
        result = run(reply(item), masked)
        self.assertEqual(result["columns"][DECIDED][0]["detail"], [{"sent": 4, "span": "スタッフ集合写真"}])
        self.assertIn("- [お客様] 店先の看板（対象: スタッフ集合写真）", result["shared"])

    def test_detail_in_a_row_moved_to_undecided_is_shown(self):
        masked = make_masked(self.SENTS)
        item = work(2, "撮っておきますね", state="提案中")
        item["detail"] = [self.d(3, "メニューの写真")]
        result = run(reply(item), masked)
        self.assertIn("- [不明] 撮っておきますね（対象: メニューの写真）", result["shared"])

    def test_negative_each_bad_detail_is_dropped_alone_and_the_item_stays(self):
        masked = make_masked(self.SENTS, hidden=(8,), replies=(7,))
        cases = {
            "not_part": ([self.d(3, "屋上の看板")], "detail" + "引用が文の一部でない"),
            "too_far": ([self.d(20, "遠い文です")], "detail範囲外"),
            "hidden": ([self.d(8, "駅前の工事")], "detail伏せた文"),
            "reply": ([self.d(7, "来週の天気は晴れです")], "detail応答型"),
            "too_long": ([self.d(3, "店" * 121)], None),
            "number": ([self.d(99, "店先の看板")], None),
            "empty_span": ([self.d(3, "  ")], None),
        }
        from shokoban.pick_bench import REASON_NOT_PART

        cases["not_part"] = ([self.d(3, "屋上の看板")], "detail" + REASON_NOT_PART)
        for name, (detail, reason) in cases.items():
            with self.subTest(name=name):
                result = run(reply(self.job(detail)), masked)
                self.assertEqual(len(result["columns"][OURS]), 1)  # 項目は残る
                self.assertEqual(result["columns"][OURS][0]["detail"], [])
                details = [d for d in result["dropped"] if d["scope"] == "detail"]
                self.assertEqual(len(details), 1)
                self.assertTrue(details[0]["reason"].startswith("detail"))
                if reason:
                    self.assertEqual(details[0]["reason"], reason)
                self.assertEqual(result["shape_errors"], 0)
                self.assertNotIn("対象", result["shared"])

    def test_negative_tag_only_detail_is_dropped(self):
        sents = ("撮影をお願いします。", "撮っておきますね。", "〈内情〉です。", "あと少し。")
        masked = make_masked(sents)
        item = work(2, "撮っておきますね", evidence=[ev("引き受け", "支援側", 2, "撮っておきますね")])
        item["detail"] = [self.d(3, "〈内情〉")]
        result = run(reply(item), masked)
        self.assertEqual(result["columns"][OURS][0]["detail"], [])
        self.assertEqual(drop_reasons(result, "detail"), ["detail札だけ"])

    def test_negative_only_the_bad_one_of_two_is_dropped(self):
        result = self.run_one([self.d(3, "店先の看板"), self.d(20, "遠い文です")])
        self.assertEqual(result["columns"][OURS][0]["detail"], [{"sent": 3, "span": "店先の看板"}])
        self.assertEqual(drop_reasons(result, "detail"), ["detail範囲外"])

    def test_negative_fourth_detail_is_dropped(self):
        # 3個まで。前後15文以内なら離れた文（9）も添えられる。
        result = self.run_one(
            [self.d(3, "店先の看板"), self.d(4, "スタッフ集合写真"), self.d(9, "近い文"), self.d(3, "メニューの写真")]
        )
        self.assertEqual(len(result["columns"][OURS][0]["detail"]), 3)
        self.assertEqual(drop_reasons(result, "detail"), ["detail個数"])

    def test_duplicate_detail_is_shown_once(self):
        result = self.run_one([self.d(3, "店先の看板"), self.d(3, " 店先の看板 ")])
        self.assertEqual(result["columns"][OURS][0]["detail"], [{"sent": 3, "span": "店先の看板"}])

    def test_negative_wrong_types_drop_only_the_detail(self):
        bad_values = [
            "店先の看板",  # list でない
            {"sent": 3, "span": "店先の看板"},  # list でなく辞書
            [{"sent": True, "span": "店先の看板"}],  # sent が真偽値
            [{"sent": "3", "span": "店先の看板"}],
            [{"sent": 3, "span": 5}],
            [{"sent": 3}],
            ["店先の看板"],
            [None],
        ]
        for value in bad_values:
            with self.subTest(value=value):
                result = self.run_one(value)
                self.assertEqual(len(result["columns"][OURS]), 1)
                self.assertEqual(result["columns"][OURS][0]["detail"], [])
                self.assertEqual(drop_reasons(result, "detail"), ["detail形"])
                self.assertEqual(result["shape_errors"], 0)
                self.assertEqual(result["counts"].get("形崩れ", 0), 0)

    def test_negative_trouble_and_undecided_ignore_detail_and_record_it(self):
        masked = make_masked(self.SENTS, regions=[(1, 5, 6, TAG)])
        for kind in ("困りごと", "未決"):
            with self.subTest(kind=kind):
                item = plain(kind, 3, "店先の看板")
                item["detail"] = [self.d(4, "スタッフ集合写真")]
                result = run(reply(item), masked)
                self.assertEqual(result["shape_errors"], 0)
                self.assertEqual(drop_reasons(result, "detail"), ["detail不可"])
                rows = [r for f in FIELDS for r in result["columns"][f]]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["detail"], [])
                self.assertNotIn("対象", result["shared"])
        region_item = {"kind": "困りごと", "region": 1, "topic": 1, "rank": 1, "detail": [self.d(4, "x")]}
        result = run(reply(region_item), masked)
        self.assertEqual(drop_reasons(result, "detail"), ["detail不可"])
        self.assertEqual(len(result["columns"][TROUBLE]), 1)

    def test_detail_is_not_part_of_the_same_column_overlap_key(self):
        # 同じ欄の重なりは文番号・引用・kind・who・state だけ。detail が違っても1行。
        masked = make_masked(self.SENTS)
        a = work(3, "店先の看板", state="提案中")
        a["detail"] = [self.d(4, "スタッフ集合写真")]
        b = work(3, "店先の看板", state="提案中")
        result = run(reply(a, b), masked)
        self.assertEqual(len(result["columns"][UNDECIDED]), 1)

    def test_detail_gets_date_suffix_like_the_line(self):
        sents = ("撮影をお願いします。", "撮っておきますね。", "明日までに撮ります。")
        item = work(2, "撮っておきますね", evidence=[ev("引き受け", "支援側", 2, "撮っておきますね")])
        item["detail"] = [self.d(3, "明日までに")]
        result = run(reply(item), make_masked(sents), meeting_date=date(2026, 11, 2))
        line = [l for l in result["shared"].splitlines() if "撮っておきますね" in l][0]
        self.assertRegex(line, r"対象: 明日までに（11月3日）")


class TroubleTopicTest(unittest.TestCase):
    """困りごとの8行は、topic ごとの最良の1行を先に入れる."""

    TAG_EMPTY = "〈伏せた区間〉"

    @staticmethod
    def sentences(n: int = 14) -> tuple[str, ...]:
        return tuple(f"困りごとその{i}です。" for i in range(1, n + 1))

    @staticmethod
    def t(i: int, topic: int, rank: int) -> dict[str, Any]:
        return plain("困りごと", i, f"困りごとその{i}です", topic=topic, rank=rank)

    def kept(self, items: list[dict[str, Any]], masked: MaskedR3 | None = None):
        result = run(reply(*items), masked or make_masked(self.sentences()))
        return result, [r["sent"] for r in result["columns"][TROUBLE]]

    def test_one_best_row_per_topic_comes_in_before_the_rank_order(self):
        # 話題1が rank 1〜8 を独占し、話題2・3 は rank 9・10 だけ。
        items = [self.t(i, 1, i) for i in range(1, 9)] + [self.t(9, 2, 9), self.t(10, 3, 10)]
        result, kept = self.kept(items)
        self.assertEqual(kept, [1, 2, 3, 4, 5, 6, 9, 10])  # 話題1の7行目・8行目が落ちる
        self.assertEqual(
            sorted(d["sent"] for d in result["dropped"] if d["reason"] == "行数"), [7, 8]
        )

    def test_topic_leader_is_its_best_ranked_row_and_output_is_in_rank_order(self):
        items = [self.t(1, 1, 5), self.t(2, 1, 2), self.t(3, 2, 9), self.t(4, 2, 7)]
        result, kept = self.kept(items)
        self.assertEqual(kept, [2, 1, 4, 3])  # 代表は文2（rank 2）と文4（rank 7）。出力は rank 順
        self.assertEqual([r["rank"] for r in result["columns"][TROUBLE]], [2, 5, 7, 9])

    def test_leaders_alone_exceeding_eight_are_cut_by_rank(self):
        items = [self.t(i, i, 11 - i) for i in range(1, 11)]  # 10話題・各1行
        _, kept = self.kept(items)
        self.assertEqual(kept, [10, 9, 8, 7, 6, 5, 4, 3])

    def test_remaining_slots_are_filled_in_rank_order(self):
        items = [self.t(1, 1, 1), self.t(2, 1, 2), self.t(3, 1, 3), self.t(4, 2, 20),
                 self.t(5, 2, 4), self.t(6, 3, 30)] + [self.t(i, 1, 10 + i) for i in range(7, 12)]
        result, kept = self.kept(items)
        # 代表: 1(rank1)・5(rank4)・6(rank30)。残り5枠は rank 2,3,17(=7),18,19 …の順で埋まる
        self.assertEqual(sorted(kept), [1, 2, 3, 5, 6, 7, 8, 9])
        self.assertEqual(len(kept), 8)

    def test_single_topic_behaves_like_plain_rank_order(self):
        items = [self.t(i, 1, 15 - i) for i in range(1, 11)]
        _, kept = self.kept(items)
        self.assertEqual(kept, [10, 9, 8, 7, 6, 5, 4, 3])

    def test_same_sentence_merge_happens_before_topic_choice(self):
        # 同じ文を指す行は rank の良い1行（その topic）に。悪い方の topic は代表になれない。
        items = [self.t(3, 1, 2), self.t(3, 2, 1)] + [self.t(i, 1, 2 + i) for i in range(4, 14)]
        result, kept = self.kept(items)
        self.assertEqual([r["topic"] for r in result["columns"][TROUBLE] if r["sent"] == 3], [2])
        self.assertEqual(len(kept), 8)

    def test_sentence_row_is_the_topic_leader_over_a_better_ranked_region_tag(self):
        regions = [(1, 13, 13, TAG)]
        masked = make_masked(self.sentences(), regions=regions)
        items = [{"kind": "困りごと", "region": 1, "topic": 2, "rank": 1}, self.t(1, 2, 9)]
        items += [self.t(i, 1, 1 + i) for i in range(2, 10)]  # 話題1: 8行
        result, kept = self.kept(items, masked)
        rows = result["columns"][TROUBLE]
        self.assertIn(1, kept)  # 話題2の代表は文の行（rank 9）
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["region"] is None for r in rows))  # 札の行は落ちた

    def test_tagged_region_is_the_leader_when_its_topic_has_no_sentence_row(self):
        regions = [(1, 13, 13, TAG)]
        masked = make_masked(self.sentences(), regions=regions)
        items = [self.t(i, 1, i) for i in range(1, 11)]
        items.append({"kind": "困りごと", "region": 1, "topic": 2, "rank": 50})
        result, kept = self.kept(items, masked)
        regions_kept = [r["region"] for r in result["columns"][TROUBLE] if r["region"]]
        self.assertEqual(regions_kept, [1])  # rank が最悪でも話題の代表として入る
        self.assertEqual(len(result["columns"][TROUBLE]), 8)

    def test_empty_tag_region_never_leads_a_topic(self):
        regions = [(1, 13, 13, self.TAG_EMPTY)]
        masked = make_masked(self.sentences(), regions=regions)
        items = [self.t(i, 1, i) for i in range(1, 9)]
        items.append({"kind": "困りごと", "region": 1, "topic": 2, "rank": 0})
        result, kept = self.kept(items, masked)
        self.assertEqual(kept, list(range(1, 9)))  # 〈伏せた区間〉だけの話題は代表にならない
        self.assertEqual(len([r for r in result["columns"][TROUBLE] if r["region"]]), 0)

    def test_empty_tag_region_fills_a_free_slot(self):
        regions = [(1, 13, 13, self.TAG_EMPTY)]
        masked = make_masked(self.sentences(), regions=regions)
        items = [self.t(i, 1, i) for i in range(1, 4)]
        items.append({"kind": "困りごと", "region": 1, "topic": 2, "rank": 0})
        rows = run(reply(*items), masked)["columns"][TROUBLE]
        self.assertEqual(len(rows), 4)

    def test_same_region_and_region_vs_sentence_merges_still_apply(self):
        regions = [(1, 11, 12, TAG)]
        masked = make_masked(self.sentences(), regions=regions)
        items = [
            {"kind": "困りごと", "region": 1, "topic": 2, "rank": 1},
            {"kind": "困りごと", "region": 1, "topic": 3, "rank": 2},
            self.t(12, 3, 4),
            self.t(1, 1, 3),
        ]
        rows = run(reply(*items), masked)["columns"][TROUBLE]
        self.assertEqual([(r["sent"], r["region"]) for r in rows], [(1, None), (12, None)])

    def test_undecided_field_is_still_plain_rank_order(self):
        items = [plain("未決", i, f"困りごとその{i}です", topic=i, rank=20 - i) for i in range(1, 11)]
        result = run(reply(*items), make_masked(self.sentences()))
        self.assertEqual([r["sent"] for r in result["columns"][UNDECIDED]], [10, 9, 8, 7, 6, 5, 4, 3])


class OldRulesTest(unittest.TestCase):
    """以前の版の機械の決まりは当てず、``old_rules`` に記録だけする."""

    def test_old_no_agreement_is_recorded_but_not_applied(self):
        masked = make_masked(("写真はお客様が用意します。", "進めてください。"))
        s = "写真はお客様が用意します"
        result = run(
            reply(
                decision(
                    1,
                    s,
                    evidence=[
                        ev("申し出", "お客様", 1, s, to="支援側"),
                        ev("同意", "支援側", 2, "進めてください"),
                    ],
                )
            ),
            masked,
        )
        self.assertEqual(spans(result, DECIDED), [s])
        self.assertEqual(
            result["old_rules"],
            [
                {
                    "sent": 1,
                    "kind": "決めごと",
                    "field": DECIDED,
                    "old_field": UNDECIDED,
                    "rules": ["合意なし"],
                }
            ],
        )

    def test_old_deferral_word_is_recorded_but_not_applied(self):
        masked = make_masked(("次回までにチラシは私が作ります。",))
        s = "チラシは私が作ります"
        result = run(
            reply(work(1, s, evidence=[ev("引き受け", "支援側", 1, s)])), masked
        )
        self.assertEqual(spans(result, OURS), [s])
        self.assertFalse(result["columns"][OURS][0]["deferred"])
        entry = result["old_rules"][0]
        self.assertEqual(entry["field"], OURS)
        self.assertEqual(entry["old_field"], UNDECIDED)
        self.assertIn("先送り", entry["rules"])

    def test_old_rules_hold_no_text(self):
        result = run(reply(work(2, S2, evidence=[OURS_OK])))
        self.assertNotIn(S2, json.dumps(result["old_rules"], ensure_ascii=False))


class DateSuffixTest(unittest.TestCase):
    def test_relative_date_is_added_to_the_line_only(self):
        masked = make_masked(("明日までに見積を送ります。",))
        result = run(
            reply(plain("未決", 1, "明日までに見積を送ります")),
            masked,
            meeting_date=date(2026, 11, 10),
        )
        self.assertIn("明日までに見積を送ります（11月11日）", result["shared"])
        self.assertEqual(spans(result, UNDECIDED), ["明日までに見積を送ります"])


class RunCaseTest(unittest.TestCase):
    """空の組み立て結果（``empty_assembly``）と、試験用の文（``BASE``）."""

    def test_empty_assembly_has_five_empty_columns(self):
        empty = empty_assembly("失敗")
        self.assertEqual(list(empty["columns"]), [F_PATH, *FIELDS])
        self.assertTrue(all(rows == [] for rows in empty["columns"].values()))

    def test_base_fixture_has_ten_sentences(self):
        self.assertEqual(len(BASE), 10)


class DetailDedupeTest(DetailTest):
    """項目の引用や同じ文のほかの detail に含まれる detail は出さない."""

    def test_detail_inside_item_span_or_another_detail_is_hidden(self):
        result = self.run_one(
            [self.d(2, "撮っておきます"), self.d(3, "店先の看板と、メニューの写真"), self.d(3, "店先の看板")]
        )
        self.assertEqual(
            result["columns"][OURS][0]["detail"], [{"sent": 3, "span": "店先の看板と、メニューの写真"}]
        )


class SelfStartedInterimTest(unittest.TestCase):
    """支援側が自分から切り出した途中の成果物（構成案・試作）を、ほかの作業の頼みや
    後のお礼で「頼まれていること」にしない."""

    SENTS = (
        "ホームページに紹介ページを作ってほしいんです。",  # 1 お客様の頼み（本制作）
        "分かりました、紹介ページは作りますね。",  # 2 本制作の引き受け
        "別の話です。",  # 3
        "まず私の方で紹介ページの構成案を作っておきますね。",  # 4 支援側が自分から
        "構成案を作ってもらえるのは助かります。",  # 5 お客様のお礼
        "構成案もお願いします。",  # 6 はっきり頼む
    )
    PAGE_ASK = ev("依頼", "お客様", 1, "ホームページに紹介ページを作ってほしいんです", to="支援側")

    def masked(self) -> MaskedR3:
        return make_masked(self.SENTS)

    def page(self) -> dict[str, Any]:
        return work(
            1, "ホームページに紹介ページを作ってほしいんです", requested=True,
            evidence=[self.PAGE_ASK, ev("引き受け", "支援側", 2, "紹介ページは作りますね")],
        )

    def draft(self, *extra: dict[str, Any]) -> dict[str, Any]:
        return work(
            4, "紹介ページの構成案を作っておきますね", requested=True,
            evidence=[ev("引き受け", "支援側", 4, "紹介ページの構成案を作っておきますね"), *extra],
        )

    def test_borrowed_page_request_does_not_make_draft_requested(self):
        result = run(reply(self.page(), self.draft(self.PAGE_ASK)), self.masked())
        self.assertEqual(spans(result, OURS), [
            "ホームページに紹介ページを作ってほしいんです", "紹介ページの構成案を作っておきますね",
        ])
        self.assertEqual(spans(result, ASKED), ["ホームページに紹介ページを作ってほしいんです"])
        self.assertIn("ほかの作業の頼み", [n["reason"] for n in result["notes"]])

    def test_thanks_after_self_start_is_not_a_request(self):
        thanks = ev("依頼", "お客様", 5, "構成案を作ってもらえるのは助かります", to="支援側")
        result = run(reply(self.draft(thanks)), self.masked())
        self.assertEqual(spans(result, ASKED), [])
        self.assertEqual(spans(result, OURS), ["紹介ページの構成案を作っておきますね"])

    def test_explicit_request_after_self_start_stays(self):
        please = ev("依頼", "お客様", 6, "構成案もお願いします", to="支援側")
        result = run(reply(self.draft(please)), self.masked())
        self.assertEqual(spans(result, ASKED), ["紹介ページの構成案を作っておきますね"])

    def test_final_deliverable_keeps_earlier_request(self):
        # 構成案などでない本制作の言い直しは、前の頼みを外さない。
        item = work(
            4, "紹介ページを作っておきますね", requested=True,
            evidence=[ev("引き受け", "支援側", 4, "紹介ページを作っておきますね"), self.PAGE_ASK],
        )
        sents = list(self.SENTS)
        sents[3] = "まず私の方で紹介ページを作っておきますね。"
        result = run(reply(self.page(), item), make_masked(tuple(sents)))
        self.assertIn("紹介ページを作っておきますね", spans(result, ASKED))


if __name__ == "__main__":
    unittest.main()
