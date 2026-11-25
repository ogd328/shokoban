"""送る前の確認（③）と目立たせ（review_flags）の試験. 文はすべて架空."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from shokoban.mac_app import MacAppPaths
from shokoban.name_candidates import JUDGMENT_UNSURE, SOURCE_RULE, Candidate
from shokoban.name_dict import KIND_PERSON, empty_dict
from shokoban.review_flags import CAP_RATIO, flag_sentences, score_sentence
from shokoban.send_review import (
    SendReviewError,
    approval_path,
    check_approval,
    final_texts,
    load_approval,
    prepare_send_review,
    save_approval,
    texts_digest,
)
from shokoban.stage5_r3_types import MASKED_SENTENCE


class ScoreTest(unittest.TestCase):
    def test_hints(self):
        self.assertGreaterEqual(score_sentence("火曜は〈家族の事情〉、水曜でお願いします。")[0], 2)
        self.assertGreaterEqual(score_sentence("ばあちゃんのことで、来週は出られません。")[0], 2)
        self.assertGreaterEqual(score_sentence("来週、先生に診てもらう予約が午前にあって。")[0], 2)
        self.assertGreaterEqual(score_sentence("写真は三百枚くらいあります。")[0], 2)
        self.assertGreaterEqual(score_sentence("〈名前〉です、たかなし、〈名前〉です。")[0], 5)

    def test_plain_work_sentences_are_not_flagged(self):
        for text in (
            "チラシは来週までにお送りします。",
            "〈名前〉、ですか。",
            "〈社名〉っていうのは、前の店の名前です。",
            "パートさんが手伝いに来てくれてます。",
            MASKED_SENTENCE,
        ):
            self.assertLess(score_sentence(text)[0], 2, text)

    def test_cap_is_one_tenth(self):
        texts = ["ばあちゃんの施設に行きます。"] * 25 + ["チラシを作ります。"] * 5
        flags = flag_sentences(texts)
        self.assertEqual(len(flags), 3)  # 30文の1割
        self.assertEqual(CAP_RATIO, 0.1)

    def test_names_come_first_under_the_cap(self):
        texts = ["ばあちゃんの施設に行きます。"] * 9 + ["〈名前〉です、たかなし、〈名前〉です。"]
        self.assertEqual([f.sent for f in flag_sentences(texts)], [10])

    def test_number_echo(self):
        flags = flag_sentences(["写真は〈数字〉枚くらいあります。", "三百枚、ですね。"] + ["はい。"] * 18)
        self.assertIn(2, [f.sent for f in flags])


def paths_in(root: Path) -> MacAppPaths:
    return MacAppPaths(
        work_dir=root,
        store_root=root / "store",
        inbox=root / "inbox",
        outbox=root / "outbox",
        held=root / "held",
        runtime_config=root / "runtime.json",
    )


class SendReviewTest(unittest.TestCase):
    SENTENCES = ["鷹梨さん、こんにちは。", "チラシは来週までにお送りします。", "ばあちゃんのショートステイの迎えがあって。"]

    def review(self):
        candidate = Candidate("鷹梨", KIND_PERSON, JUDGMENT_UNSURE, (SOURCE_RULE,), (1,))
        return prepare_send_review(self.SENTENCES, name_dict=empty_dict(), candidates=[candidate])

    def test_masks_like_sending_and_flags(self):
        review = self.review()
        self.assertEqual(review.total, 3)
        self.assertNotIn("鷹梨", review.rows[0].text)
        self.assertLessEqual(review.flagged, 1)

    def test_person_can_hide_more(self):
        review = self.review()
        texts = final_texts(review, [2])
        self.assertEqual(texts[1], MASKED_SENTENCE)
        with self.assertRaises(SendReviewError):
            final_texts(review, [9])

    def test_approval_round_trip_and_gate(self):
        review = self.review()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "store-synthetic").mkdir()
            path = approval_path(paths_in(root), "synthetic", "c001", "m001")
            self.assertTrue(str(path).endswith("send-approvals/c001/m001.json"))
            record = save_approval(path, review, [2], reviewed_at=datetime(2026, 11, 22, tzinfo=timezone.utc), read_all=True)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            text = path.read_text(encoding="utf-8")
            for sentence in self.SENTENCES:
                self.assertNotIn(sentence, text)  # 本文は書かない
            self.assertEqual(record["hidden_by_person"], [2])
            self.assertTrue(check_approval(path, final_texts(review, [2])))
            self.assertFalse(check_approval(path, final_texts(review)))  # 確認した後に変わった
            self.assertEqual(load_approval(path)["sha256"], texts_digest(final_texts(review, [2])))

    def test_no_approval_means_no_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(check_approval(Path(tmp) / "none.json", ["a"]))

    def test_real_mode_and_bad_ids_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = paths_in(Path(tmp))
            with self.assertRaises(SendReviewError):
                approval_path(paths, "real", "c001", "m001")
            for bad in ("../x", ".hidden", ""):
                with self.assertRaises(SendReviewError):
                    approval_path(paths, "synthetic", "c001", bad)
            with self.assertRaises(SendReviewError):
                approval_path(paths, "synthetic", "c/01", "m001")

    def test_broken_approval_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            path.write_text(json.dumps({"version": 99}), encoding="utf-8")
            with self.assertRaises(SendReviewError):
                check_approval(path, ["a"])


class ShortFillerTest(unittest.TestCase):
    def test_short_fillers(self):
        from shokoban.send_review import is_short_filler

        self.assertTrue(is_short_filler("はい。", False))
        self.assertTrue(is_short_filler("なるほど、そうですね。", False))
        self.assertFalse(is_short_filler("〈名前〉です。", False))  # 札のある文は畳まない
        self.assertFalse(is_short_filler("ばあちゃんがね。", True))  # 目立たせた文は畳まない
        self.assertFalse(is_short_filler("チラシは来週までにお送りします。", False))

    def test_dictionary_katakana_term_is_masked_inside_a_katakana_word(self):
        from shokoban.name_dict import KIND_ORG, MaskEntry, NameDict
        from shokoban.send_review import final_masked

        name_dict = NameDict([MaskEntry("カクウ", KIND_ORG)], [])
        review = prepare_send_review(["カクウシステムの件です。"], name_dict=name_dict, candidates=[])
        texts = final_texts(review, [])
        self.assertNotIn("カクウ", "".join(texts))
        self.assertNotIn("カクウ", json.dumps(final_masked(review).reader_payload(), ensure_ascii=False))

    def test_review_counts_short_rows(self):
        review = prepare_send_review(
            ["はい。", "チラシは来週までにお送りします。", "なるほど。"], name_dict=empty_dict(), candidates=[]
        )
        self.assertEqual(review.short, 2)
        self.assertEqual(len(final_texts(review, [1])), 3)  # 畳んだ文も「さらに伏せる」を選べる


class OkPhraseTest(unittest.TestCase):
    def test_phrase_window_keeps_context(self):
        from shokoban.review_flags import ok_phrase

        text = "あー、それは痛いですね。"
        start = text.index("痛")
        self.assertEqual(ok_phrase(text, start, start + 1), "それは痛いですね")

    def test_ok_phrase_silences_only_that_wording(self):
        ok = frozenset({"それは痛いですね"})
        self.assertLess(score_sentence("あー、それは痛いですね。", ok)[0], 2)
        self.assertGreaterEqual(score_sentence("最近、腰が痛いですね。", ok)[0], 2)
        # 同じ文に別の手がかりが残っていれば目立たせる
        self.assertGreaterEqual(score_sentence("それは痛いですね、ばあちゃんの施設も。", ok)[0], 2)

    def test_name_reading_cannot_be_silenced(self):
        text = "〈名前〉です、たかなし、〈名前〉です。"
        _, _, phrases = score_sentence(text)
        self.assertEqual(phrases, ())
        self.assertGreaterEqual(score_sentence(text, frozenset({text})) [0], 5)

    def test_store_round_trip_and_screen_rows(self):
        from shokoban.send_review import add_ok_phrases, load_ok_phrases

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review-ok" / "common.json"
            self.assertEqual(load_ok_phrases(path), frozenset())
            self.assertEqual(add_ok_phrases(path, ["それは痛いですね", "それは痛いですね", "あ"]), 1)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            ok = load_ok_phrases(path)
            self.assertEqual(ok, frozenset({"それは痛いですね"}))
            texts = ["あー、それは痛いですね。"] + ["チラシを作ります。"] * 9
            before = prepare_send_review(texts, name_dict=empty_dict(), candidates=[])
            self.assertTrue(before.rows[0].flagged)
            self.assertIn("それは痛いですね", before.rows[0].ok_phrases)
            after = prepare_send_review(texts, name_dict=empty_dict(), candidates=[], ok_phrases=ok)
            self.assertFalse(after.rows[0].flagged)


class HideWordTest(unittest.TestCase):
    def review(self):
        return prepare_send_review(
            ["商工会議所に相談しました。", "チラシは来週までにお送りします。"], name_dict=empty_dict(), candidates=[]
        )

    def test_word_must_be_in_that_sentence_and_not_a_tag(self):
        from shokoban.send_review import HideWord, check_hide_words

        review = self.review()
        self.assertEqual(
            check_hide_words(review, [HideWord(1, " 商工会議所 ", "社名・店名・団体")])[0].term, "商工会議所"
        )
        for bad in (HideWord(2, "商工会議所", "社名・店名・団体"), HideWord(1, "〈名前〉", "人名"),
                    HideWord(1, "商工会議所", "それ以外"), HideWord(9, "商工会議所", "人名")):
            with self.assertRaises(SendReviewError):
                check_hide_words(review, [bad])

    def test_saved_words_mask_next_time(self):
        from shokoban.name_dict import load_name_dict
        from shokoban.send_review import HideWord, save_hide_words

        with tempfile.TemporaryDirectory() as tmp:
            own, common = Path(tmp) / "d" / "c001.json", Path(tmp) / "common" / "common.json"
            counts = save_hide_words(
                own, common,
                [HideWord(1, "商工会議所", "社名・店名・団体"), HideWord(2, "チラシ", "社名・店名・団体", True)],
            )
            self.assertEqual(counts, (1, 1))
            name_dict = load_name_dict(own)
            again = prepare_send_review(["商工会議所に相談しました。"], name_dict=name_dict, candidates=[])
            self.assertNotIn("商工会議所", again.rows[0].text)
            self.assertIn("チラシ", dict(load_name_dict(common).mask_terms()))


class FalsePositiveTrimTest(unittest.TestCase):
    """教室のお客様で空振りしていた言い方を、ヒントにしない."""

    def test_school_words_and_greetings_are_not_hints(self):
        for text in (
            "〈名前〉、こんにちは、どうぞこちらへ。",
            "教室の中は子どもたちの熱気がまだ残ってますね。",
            "理科とプログラミングの教室として打ち出したいんです。",
            "近くの小学校の前で配る分を計算します。",
        ):
            self.assertLess(score_sentence(text)[0], 2, text)

    def test_real_hints_still_work(self):
        self.assertGreaterEqual(score_sentence("熱が下がらなくて、来週は病院です。")[0], 2)
        self.assertGreaterEqual(score_sentence("〈名前〉です、たかなし、〈名前〉です。")[0], 5)


class PayloadApprovalTest(unittest.TestCase):
    """承認は、クラウドに送る中身全体（reader_payload）に効く."""

    SECRET = "ばあちゃんのショートステイの迎えがあって。"

    def review(self):
        from dataclasses import replace

        review = prepare_send_review(
            ["チラシは来週までにお送りします。", self.SECRET, "はい。"], name_dict=empty_dict(), candidates=[]
        )
        # 元の文に由来する手がかり（hint・応答型の印）が付いた記録を作り、伏せたときに消えるかを見る
        records = tuple(
            replace(record, hint=("依頼の型",), region=1, region_reply=True) if record.sent == 2 else record
            for record in review.masked.records
        )
        return replace(review, masked=replace(review.masked, records=records))

    def test_hidden_sentence_leaves_no_trace_in_the_payload(self):
        from shokoban.send_review import final_masked

        review = self.review()
        plain = json.dumps(final_masked(review).reader_payload(), ensure_ascii=False)
        self.assertIn("ショートステイ", plain)
        hidden = final_masked(review, [2])
        dumped = json.dumps(hidden.reader_payload(), ensure_ascii=False)
        self.assertNotIn("ショートステイ", dumped)
        record = hidden.records[1]
        self.assertEqual((record.text, record.state, record.hint, record.region_reply), (MASKED_SENTENCE, "伏せた", (), False))
        # 伏せていない文は変えない
        self.assertEqual(hidden.records[0], review.masked.records[0])
        self.assertEqual(hidden.records[2], review.masked.records[2])
        # 伏せる前の MaskedR3 は書き換えない
        self.assertIn("ショートステイ", json.dumps(review.masked.reader_payload(), ensure_ascii=False))
        # 本文は final_texts と同じ置き換え
        self.assertEqual([r.text for r in hidden.records], final_texts(review, [2]))

    def test_final_masked_rejects_unknown_numbers_and_missing_masked(self):
        from dataclasses import replace

        from shokoban.send_review import final_masked

        review = self.review()
        with self.assertRaises(SendReviewError):
            final_masked(review, [9])
        with self.assertRaises(SendReviewError):
            final_masked(replace(review, masked=None))

    def test_digest_changes_with_hiding_and_is_stable(self):
        from shokoban.send_review import final_masked, payload_digest

        review = self.review()
        self.assertEqual(payload_digest(final_masked(review)), payload_digest(final_masked(review)))
        self.assertNotEqual(payload_digest(final_masked(review)), payload_digest(final_masked(review, [2])))
        self.assertNotEqual(payload_digest(final_masked(review, [1])), payload_digest(final_masked(review, [2])))

    def test_save_approval_refuses_without_read_all(self):
        review = self.review()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "send-approvals" / "c001" / "a.json"
            when = datetime(2026, 11, 24, tzinfo=timezone.utc)
            with self.assertRaises(SendReviewError):
                save_approval(path, review, [2], reviewed_at=when)
            with self.assertRaises(SendReviewError):
                save_approval(path, review, [2], reviewed_at=when, read_all=False)
            self.assertFalse(path.exists())

    def test_check_payload_approval(self):
        from shokoban.send_review import check_payload_approval, final_masked, payload_digest

        review = self.review()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "send-approvals" / "c001" / "a.json"
            self.assertFalse(check_payload_approval(path, final_masked(review)))  # 記録が無い
            record = save_approval(
                path, review, [2], reviewed_at=datetime(2026, 11, 24, tzinfo=timezone.utc), read_all=True
            )
            self.assertTrue(record["read_all"])
            self.assertEqual(record["payload_sha256"], payload_digest(final_masked(review, [2])))
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            text = path.read_text(encoding="utf-8")
            for secret in (self.SECRET, "チラシ", "ショートステイ"):
                self.assertNotIn(secret, text)  # 本文は書かない
            self.assertTrue(check_payload_approval(path, final_masked(review, [2])))
            self.assertFalse(check_payload_approval(path, final_masked(review, [])))  # 伏せ方が変わった
            self.assertFalse(check_payload_approval(path, final_masked(review, [1, 2])))
            # 文字列の関所は今までどおり動く
            self.assertTrue(check_approval(path, final_texts(review, [2])))

    def test_old_approval_without_read_all_or_payload_is_not_enough(self):
        from shokoban.send_review import check_payload_approval, final_masked, payload_digest

        review = self.review()
        digest = payload_digest(final_masked(review))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            for extra in ({}, {"payload_sha256": digest}, {"payload_sha256": digest, "read_all": False}):
                path.write_text(json.dumps({"version": 1, "sha256": "x"} | extra), encoding="utf-8")
                self.assertFalse(check_payload_approval(path, final_masked(review)))
            path.write_text(json.dumps({"version": 1, "sha256": "x", "payload_sha256": digest, "read_all": True}), encoding="utf-8")
            self.assertTrue(check_payload_approval(path, final_masked(review)))


if __name__ == "__main__":
    unittest.main()
