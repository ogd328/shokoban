"""名前・社名の辞書と候補出し（``name_dict``・``name_candidates``・``mask_r3`` の
``extra_terms``）の試験.

本物の12B（llama-server）は一切起動しない。12B は偽の関数・偽の Runtime・偽の http で試す。
文は、この試験が書いた架空の短い文だけを使い、実際の案件データは読まない。
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any

from shokoban.mask_r3 import MaskR3Error, mask_r3
from shokoban.name_candidates import (
    CHUNK_OVERLAP,
    CHUNK_SENTENCES,
    JUDGMENT_SURE,
    JUDGMENT_UNSURE,
    KIND_OTHER,
    NAME_SCHEMA,
    Candidate,
    NameReplyError,
    build_name_prompt,
    chunk_sentences,
    dict_terms_to_mask,
    dry_name_extractor,
    find_candidates,
    llm_candidates,
    make_llama_name_extractor,
    merge_candidates,
    parse_name_reply,
    rule_candidates,
    terms_to_mask,
    validate_items,
)
from shokoban.name_dict import (
    KIND_ORG,
    KIND_PERSON,
    MaskEntry,
    NameDict,
    NameDictError,
    empty_dict,
    load_name_dict,
    save_name_dict,
)


def item(word: str, kind: str = KIND_PERSON, judgment: str = JUDGMENT_SURE) -> dict[str, str]:
    return {"word": word, "kind": kind, "judgment": judgment}


class NameDictTest(unittest.TestCase):
    def test_round_trip_and_file_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dict.json"
            name_dict = NameDict(
                [MaskEntry("山本", KIND_PERSON), MaskEntry("ほしの食堂", KIND_ORG)], ["春日"]
            )
            save_name_dict(path, name_dict)
            loaded = load_name_dict(path)
            self.assertEqual(loaded.mask, name_dict.mask)
            self.assertEqual(loaded.pass_terms, ("春日",))
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["dict.json"])

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "none.json"
            with self.assertRaises(FileNotFoundError):
                load_name_dict(path)
            self.assertEqual(load_name_dict(path, missing_ok=True).mask, ())

    def test_save_does_not_create_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                save_name_dict(Path(tmp) / "nope" / "dict.json", empty_dict())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_failed_save_keeps_old_file_and_no_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dict.json"
            save_name_dict(path, NameDict([MaskEntry("山本", KIND_PERSON)]))
            with self.assertRaises(NameDictError):
                save_name_dict(path, {"mask": []})  # type: ignore[arg-type]
            self.assertEqual(load_name_dict(path).mask_terms(), (("山本", KIND_PERSON),))
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["dict.json"])

    def test_validation(self):
        bad_payloads = [
            [],
            {"mask": {}, "pass": []},
            {"mask": [{"term": "山本"}], "pass": []},
            {"mask": [{"term": "山本", "kind": "それ以外"}], "pass": []},
            {"mask": [{"term": 1, "kind": KIND_PERSON}], "pass": []},
            {"mask": [{"term": "", "kind": KIND_PERSON}], "pass": []},
            {"mask": [{"term": "あ" * 21, "kind": KIND_PERSON}], "pass": []},
            {"mask": [{"term": "山\n本", "kind": KIND_PERSON}], "pass": []},
            {"mask": [], "pass": [3]},
            {"mask": [], "pass": "春日"},
            {"mask": [], "pass": [], "extra": 1},
            {"mask": [{"term": "山本", "kind": KIND_PERSON}], "pass": ["山本"]},
            {"mask": [{"term": "山本", "kind": KIND_PERSON}, {"term": "山本", "kind": KIND_ORG}], "pass": []},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(NameDictError):
                NameDict.from_json(payload)

    def test_unreadable_json_is_a_dict_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dict.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(NameDictError):
                load_name_dict(path)

    def test_add_moves_between_lists(self):
        name_dict = empty_dict()
        name_dict.add_mask("春日", KIND_PERSON)
        name_dict.add_pass("春日")
        self.assertEqual(name_dict.mask_terms(), ())
        self.assertEqual(name_dict.pass_terms, ("春日",))
        name_dict.add_mask("春日", KIND_ORG)
        self.assertEqual(name_dict.mask_terms(), (("春日", KIND_ORG),))
        self.assertEqual(name_dict.pass_terms, ())
        self.assertTrue(name_dict.remove("春日"))
        self.assertFalse(name_dict.knows("春日"))


class ChunkTest(unittest.TestCase):
    def test_chunks_cover_everything_with_overlap(self):
        sentences = [f"文{i}" for i in range(75)]
        chunks = chunk_sentences(sentences)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, 75)
        for before, after in zip(chunks, chunks[1:]):
            self.assertEqual(before.end - after.start, CHUNK_OVERLAP)
        self.assertTrue(all(c.end - c.start <= CHUNK_SENTENCES for c in chunks))

    def test_short_input_is_one_chunk_and_empty_is_none(self):
        self.assertEqual(len(chunk_sentences(["a", "b"])), 1)
        self.assertEqual(chunk_sentences([]), [])

    def test_long_sentences_split_by_chars_and_still_progress(self):
        sentences = ["あ" * 3000] * 4
        chunks = chunk_sentences(sentences)
        self.assertEqual(chunks[-1].end, 4)
        self.assertTrue(all(c.end > c.start for c in chunks))


class ValidateTest(unittest.TestCase):
    SENTENCES = ["山本さんと、ほしの食堂の話をしました。", "社長に聞いてみます。"]

    def check(self, items: list[Any]):
        return validate_items(items, self.SENTENCES)

    def test_valid_items_kept_and_honorific_removed(self):
        kept, dropped = self.check([item("山本さん"), item("ほしの食堂", KIND_ORG, JUDGMENT_UNSURE)])
        self.assertEqual(
            kept, [("山本", KIND_PERSON, JUDGMENT_SURE), ("ほしの食堂", KIND_ORG, JUDGMENT_UNSURE)]
        )
        self.assertEqual(sum(dropped.values()), 0)

    def test_invalid_items_are_dropped_and_counted(self):
        bad = [
            item("田中"),  # 原文に無い
            item("山\n本"),
            item(""),
            item("あ" * 21),
            item("社長"),  # 役割
            item("社長さん"),  # 役割（敬称を外して）
            item("山本", "別の種類"),
            item("山本", KIND_PERSON, "たぶん"),
            {"word": "山本"},
            {"word": 3, "kind": KIND_PERSON, "judgment": JUDGMENT_SURE},
            "山本",
            item("、"),
        ]
        kept, dropped = self.check(bad)
        self.assertEqual(kept, [])
        self.assertEqual(sum(dropped.values()), len(bad))

    def test_company_ending_in_business_word_is_not_a_role_word(self):
        kept, _ = validate_items([item("山田商店", KIND_ORG)], ["山田商店に頼みます。"])
        self.assertEqual(kept, [("山田商店", KIND_ORG, JUDGMENT_SURE)])

    def test_too_many_items_are_dropped(self):
        words = [f"名{i}" for i in range(45)]
        kept, dropped = validate_items([item(w) for w in words], ["".join(words)])
        self.assertEqual(len(kept), 40)
        self.assertEqual(dropped["多すぎ"], 5)

    def test_kinship_and_self_words_are_not_names(self):
        words = ["父", "弟", "親父", "女房", "兄貴", "僕"]
        kept, dropped = validate_items([item(w) for w in words], ["父と弟と親父と女房と兄貴と僕。"])
        self.assertEqual(kept, [])
        self.assertEqual(sum(dropped.values()), len(words))

    def test_place_like_surname_is_kept(self):
        kept, _ = validate_items([item("長野")], ["長野さんに頼みます。"])
        self.assertEqual(kept, [("長野", KIND_PERSON, JUDGMENT_SURE)])

    def test_polite_phrase_is_not_a_person(self):
        kept, dropped = validate_items([item("お疲れ様")], ["お疲れ様です。"])
        self.assertEqual(kept, [])
        self.assertEqual(sum(dropped.values()), 1)


class ParseReplyTest(unittest.TestCase):
    def test_accepts_array_or_single_key_object(self):
        self.assertEqual(parse_name_reply('[{"word": "a"}]'), [{"word": "a"}])
        self.assertEqual(parse_name_reply('{"names": []}'), [])

    def test_rejects_bad_replies(self):
        for raw in ("not json", '"text"', "3", '{"a": 1, "b": 2}', None, 3):
            with self.subTest(raw=raw), self.assertRaises(NameReplyError):
                parse_name_reply(raw)


class LlmCandidatesTest(unittest.TestCase):
    def test_dry_extractor_returns_nothing(self):
        self.assertEqual(dry_name_extractor("文"), [])
        merged, stats = llm_candidates(["山本さんです。"], None)
        self.assertEqual(merged, [])
        self.assertEqual((stats.chunks_ok, stats.chunks_fail), (1, 0))

    def test_failed_chunk_is_counted_and_run_continues(self):
        sentences = [f"文{i}です。" for i in range(60)]
        sentences[50] = "山本さんが来ました。"
        calls: list[str] = []

        def extractor(text: str):
            calls.append(text)
            if len(calls) == 1:
                raise NameReplyError("壊れた返事")
            return [item("山本")]

        merged, stats = llm_candidates(sentences, extractor)
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(stats.chunks_fail, 1)
        self.assertEqual(stats.chunks_ok, len(calls) - 1)
        self.assertEqual(merged, [("山本", KIND_PERSON, JUDGMENT_SURE)])

    def test_non_list_reply_is_a_failed_chunk(self):
        merged, stats = llm_candidates(["山本さんです。"], lambda text: 5)
        self.assertEqual((stats.chunks_ok, stats.chunks_fail), (0, 1))
        self.assertEqual(merged, [])

    def test_extractor_gets_numbered_chunk_text(self):
        seen: list[str] = []
        llm_candidates(["一つ目。", "二つ目。"], lambda text: seen.append(text) or [])
        self.assertEqual(seen, ["[1] 一つ目。\n[2] 二つ目。"])

    def test_overlap_duplicates_merge_and_votes(self):
        sentences = [f"文{i}です。" for i in range(40)]
        sentences[29] = "ひかりさんが来ました。"  # 区切りの重なりに入る
        replies = iter(
            [
                [item("ひかり", KIND_PERSON, JUDGMENT_UNSURE)],
                [item("ひかり", KIND_OTHER, JUDGMENT_SURE)],
            ]
        )
        merged, stats = llm_candidates(sentences, lambda text: next(replies))
        self.assertEqual(stats.chunks_ok, 2)
        self.assertEqual(merged, [("ひかり", KIND_PERSON, JUDGMENT_SURE)])

    def test_validation_drop_counts_reach_stats(self):
        merged, stats = llm_candidates(
            ["山本さんです。"], lambda text: [item("山本"), item("無い語"), item("社長")]
        )
        self.assertEqual(merged, [("山本", KIND_PERSON, JUDGMENT_SURE)])
        self.assertEqual((stats.items_raw, stats.items_dropped), (3, 2))


class RuleCandidatesTest(unittest.TestCase):
    def test_honorific_names_without_roles(self):
        words = [w for w, _ in rule_candidates(["山本さんと社長さんとお客さん。", "ひろしさんにも聞きます。"])]
        self.assertIn("山本", words)
        self.assertIn("ひろし", words)
        self.assertNotIn("社長", words)
        self.assertNotIn("お客", words)

    def test_company_terms_are_not_candidates(self):
        # 登録した社名そのもの・社名の一部の人名は候補にしない。社名の芯（「ほし」）は社名の候補。
        got = rule_candidates(["ほしの食堂さんです。", "山本さんです。"], "ほしの食堂")
        self.assertEqual(got, [("ほし", KIND_ORG), ("山本", KIND_PERSON)])


class MergeTest(unittest.TestCase):
    SENTENCES = ["山本さんと話しました。", "ほしの食堂は春日が好きです。", "山本が来ます。"]

    def test_merge_sources_kind_judgment_and_sentences(self):
        rule = [("山本", KIND_PERSON)]
        llm = [
            ("山本", KIND_PERSON, JUDGMENT_UNSURE),
            ("ほしの食堂", KIND_ORG, JUDGMENT_UNSURE),
            ("春日", KIND_OTHER, JUDGMENT_UNSURE),
        ]
        result = {c.term: c for c in merge_candidates(self.SENTENCES, rule, llm)}
        self.assertEqual(result["山本"], Candidate("山本", KIND_PERSON, JUDGMENT_SURE, ("rule", "12b"), (1, 3)))
        self.assertEqual(result["ほしの食堂"].sources, ("12b",))
        self.assertEqual(result["ほしの食堂"].judgment, JUDGMENT_UNSURE)
        self.assertEqual(result["春日"].kind, KIND_OTHER)

    def test_other_kind_is_upgraded_by_rule(self):
        result = merge_candidates(
            self.SENTENCES, [("山本", KIND_PERSON)], [("山本", KIND_OTHER, JUDGMENT_UNSURE)]
        )
        self.assertEqual((result[0].kind, result[0].judgment), (KIND_PERSON, JUDGMENT_SURE))

    def test_dictionary_and_own_company_are_removed(self):
        name_dict = NameDict([MaskEntry("山本", KIND_PERSON)], ["春日"])
        llm = [
            ("山本", KIND_PERSON, JUDGMENT_SURE),
            ("春日", KIND_PERSON, JUDGMENT_SURE),
            ("ほしの食堂", KIND_ORG, JUDGMENT_SURE),
            ("新顔", KIND_PERSON, JUDGMENT_SURE),
        ]
        result = merge_candidates(
            self.SENTENCES + ["新顔です。"], [], llm, name_dict, company_name="ほしの食堂"
        )
        self.assertEqual([c.term for c in result], ["新顔"])

    def test_find_candidates_end_to_end_with_fake_12b(self):
        found = find_candidates(
            self.SENTENCES,
            extractor=lambda text: [item("ほしの食堂", KIND_ORG, JUDGMENT_UNSURE)],
            name_dict=empty_dict(),
        )
        self.assertEqual({c.term for c in found.candidates}, {"山本", "ほしの食堂"})
        self.assertEqual(found.llm.chunks_ok, 1)


class TermsToMaskTest(unittest.TestCase):
    def test_dictionary_then_unconfirmed_candidates(self):
        name_dict = NameDict([MaskEntry("山本", KIND_PERSON)], ["春日"])
        candidates = [
            Candidate("山本", KIND_PERSON, JUDGMENT_SURE, ("rule",), (1,)),
            Candidate("ほしの食堂", KIND_ORG, JUDGMENT_UNSURE, ("12b",), (2,)),
            Candidate("ひかり", KIND_PERSON, JUDGMENT_UNSURE, ("12b",), (3,)),
            Candidate("春日", KIND_PERSON, JUDGMENT_SURE, ("12b",), (4,)),  # 伏せない語
            Candidate("エコバッグ", KIND_OTHER, JUDGMENT_SURE, ("12b",), (5,)),  # それ以外
        ]
        self.assertEqual(
            terms_to_mask(name_dict, candidates),
            [("山本", KIND_PERSON), ("ほしの食堂", KIND_ORG), ("ひかり", KIND_PERSON)],
        )

    def test_no_dictionary(self):
        self.assertEqual(terms_to_mask(None, []), [])

    def test_dict_terms_are_the_dictionary_part_only(self):
        name_dict = NameDict([MaskEntry("山本", KIND_PERSON), MaskEntry("カクウ", KIND_ORG)], ["春日"])
        self.assertEqual(dict_terms_to_mask(name_dict), frozenset({"山本", "カクウ"}))
        self.assertEqual(dict_terms_to_mask(None), frozenset())


class MaskExtraTermsTest(unittest.TestCase):
    PLAIN = ["チラシは五百部です。", "写真は三枚です。"]

    def texts(self, sentences: list[str], **kwargs: Any) -> list[str]:
        return [r.text for r in mask_r3(sentences, **kwargs).records]

    def test_person_and_company_terms_are_masked_everywhere(self):
        sentences = ["山本が納品します。", "ほしの食堂へ山本が届けます。", "あした山本です。"]
        out = self.texts(
            sentences, extra_terms=[("山本", KIND_PERSON), ("ほしの食堂", KIND_ORG)]
        )
        joined = "".join(out)
        self.assertNotIn("山本", joined)
        self.assertNotIn("ほしの食堂", joined)
        self.assertIn("〈名前〉", joined)
        self.assertIn("〈社名〉", joined)

    def test_longest_first(self):
        out = self.texts(
            ["山本商店が来ます。"], extra_terms=[("山本", KIND_PERSON), ("山本商店", KIND_ORG)]
        )
        self.assertIn("〈社名〉", out[0])
        self.assertNotIn("山本", out[0])
        self.assertNotIn("〈名前〉", out[0])

    def test_kana_term_not_masked_inside_a_word(self):
        out = self.texts(
            ["店は、継がんて、ゆうとるんだわ。", "ゆうとが来ます。", "ゆうとさん、どうも。"],
            extra_terms=[("ゆうと", KIND_PERSON)],
        )
        self.assertIn("ゆうとる", out[0])
        self.assertNotIn("ゆうと", out[1])
        self.assertNotIn("ゆうと", out[2])

    def test_katakana_term_not_masked_inside_a_katakana_word(self):
        out = self.texts(
            ["ケンジが来ます。", "ケンジントンの写真です。"],
            extra_terms=[("ケンジ", KIND_PERSON)],
        )
        self.assertNotIn("ケンジ", out[0])
        self.assertIn("ケンジントン", out[1])

    def test_dictionary_katakana_term_is_masked_inside_a_katakana_word(self):
        # 人が辞書に入れた語は、長いカタカナ語の途中にあっても伏せる。自動の候補は伏せない。
        sentences = ["カクウシステムの件です。", "カクウさんに聞きます。"]
        candidate_only = self.texts(sentences, extra_terms=[("カクウ", KIND_ORG)])
        self.assertIn("カクウシステム", candidate_only[0])
        self.assertNotIn("カクウ", candidate_only[1])
        from_dict = self.texts(sentences, extra_terms=[("カクウ", KIND_ORG)], dict_terms={"カクウ"})
        self.assertNotIn("カクウ", "".join(from_dict))
        self.assertIn("〈社名〉システム", from_dict[0])

    def test_dict_terms_do_not_mask_words_not_in_extra_terms(self):
        out = self.texts(["カクウシステムの件です。"], dict_terms={"カクウ"})
        self.assertIn("カクウ", out[0])  # 伏せる語の並び（extra_terms）に無い語は伏せない
        with self.assertRaises(MaskR3Error):
            mask_r3(["文です。"], extra_terms=[("カクウ", KIND_ORG)], dict_terms="カクウ")

    def test_one_char_term_only_when_standalone(self):
        out = self.texts(
            ["誠実に感謝します。", "誠が来ます。", "修正は三枚です。", "修が来ます。"],
            extra_terms=[("誠", KIND_PERSON), ("修", KIND_PERSON)],
        )
        self.assertIn("誠実", out[0])
        self.assertNotIn("誠", out[1])
        self.assertIn("修正", out[2])
        self.assertNotIn("修", out[3])

    def test_two_char_term_masks_inside_words(self):
        out = self.texts(["山本家に行きます。"], extra_terms=[("山本", KIND_PERSON)])
        self.assertNotIn("山本", out[0])

    def test_other_kind_is_ignored_and_bad_input_raises(self):
        base = self.texts(self.PLAIN + ["エコバッグを作ります。"])
        self.assertEqual(
            self.texts(self.PLAIN + ["エコバッグを作ります。"], extra_terms=[("エコバッグ", KIND_OTHER)]),
            base,
        )
        for bad in ([("山本", "不明")], [("山本",)], [(1, KIND_PERSON)], "山本"):
            with self.subTest(bad=bad), self.assertRaises(MaskR3Error):
                mask_r3(self.PLAIN, extra_terms=bad)  # type: ignore[arg-type]

    def test_existing_nets_win_on_overlap(self):
        out = self.texts(["山本さんに頼みます。"], extra_terms=[("山本", KIND_PERSON)])
        self.assertEqual(out[0].count("〈名前〉"), 1)
        self.assertNotIn("さん", out[0])

    def test_default_output_is_unchanged(self):
        sentences = [
            "ひだまり工房の田中さんに連絡してください。",
            "チラシは五百部です。",
            "山本が来ます。",
            "腰が、もう、あかんくて。",
            "それでお願いします。",
        ]
        kwargs = dict(company_name="ひだまり工房", tagger=lambda text: ["集客"])
        plain = mask_r3(sentences, **kwargs)
        self.assertEqual(plain.to_json(), mask_r3(sentences, extra_terms=(), **kwargs).to_json())
        self.assertEqual(
            plain.to_json(),
            mask_r3(sentences, extra_terms=[("存在しない語", KIND_PERSON)], **kwargs).to_json(),
        )


class FakeRuntime:
    def __init__(self, *, running: bool = False, fail_ensure: bool = False) -> None:
        self.running = running
        self.fail_ensure = fail_ensure
        self.calls: list[str] = []

    def is_running(self) -> bool:
        self.calls.append("is_running")
        return self.running

    def ensure(self, alias: str) -> str:
        self.calls.append(f"ensure:{alias}")
        if self.fail_ensure:
            raise RuntimeError("起動に失敗")
        return alias

    def stop(self) -> bool:
        self.calls.append("stop")
        return True


def http(reply: Any, sent: list[dict[str, Any]] | None = None):
    def caller(method, url, payload, timeout):
        if sent is not None:
            sent.append({"method": method, "url": url, "payload": payload})
        content = reply if type(reply) is str else json.dumps(reply, ensure_ascii=False)
        return {"choices": [{"message": {"content": content}}]}

    return caller


class TwelveBNameTest(unittest.TestCase):


    def test_request_uses_loopback_model_and_name_schema(self):
        sent: list[dict[str, Any]] = []
        extractor = make_llama_name_extractor("gemma-4-12b-ja", http_caller=http([item("山本")], sent))
        extractor("[1] 山本さんです。")
        call = sent[0]
        self.assertTrue(call["url"].startswith("http://127.0.0.1:1234/"))
        self.assertEqual(call["payload"]["model"], "gemma-4-12b-ja")
        self.assertEqual(call["payload"]["response_format"]["json_schema"]["schema"], NAME_SCHEMA)
        self.assertEqual(NAME_SCHEMA["maxItems"], 40)
        prompt = call["payload"]["messages"][0]["content"]
        self.assertIn("UNTRUSTED_JSON_STRING", prompt)
        self.assertEqual(
            json.loads(prompt.split("UNTRUSTED_JSON_STRING = ", 1)[1])["text"], "[1] 山本さんです。"
        )

    def test_bad_reply_makes_that_chunk_fail_without_crashing(self):
        extractor = make_llama_name_extractor("gemma-4-12b-ja", http_caller=http("これは JSON ではない"))
        merged, stats = llm_candidates(["山本さんです。"], extractor)
        self.assertEqual(merged, [])
        self.assertEqual((stats.chunks_ok, stats.chunks_fail), (0, 1))

    def test_prompt_mentions_what_to_pick_and_skip(self):
        prompt = build_name_prompt("[1] 文")
        for word in ("さん", "ひらがな", "屋号", "取引先", "学校", "社長", "お客さん", "パン屋", "工務店", "Instagram", "LINE", "Google"):
            self.assertIn(word, prompt)


class CompanyCoreTest(unittest.TestCase):
    def test_cores(self):
        from shokoban.name_candidates import company_core_terms

        self.assertEqual(company_core_terms("有限会社祖父江木工所", ["祖父江木工所"]), ["祖父江"])
        self.assertEqual(company_core_terms("ヘアサロンはなまる", []), ["はなまる"])
        self.assertEqual(company_core_terms("合同会社ひより灯ケア", ["デイサービスひより灯"]), ["ひより灯"])
        self.assertEqual(company_core_terms("株式会社せとうち急便", []), ["せとうち"])
        self.assertEqual(company_core_terms("山田", []), [])

    def test_core_becomes_org_candidate(self):
        from shokoban.name_candidates import rule_candidates

        got = rule_candidates(["祖父江木工さんに頼みます。"], "有限会社祖父江木工所", ["祖父江木工所"])
        self.assertIn(("祖父江", KIND_ORG), got)


class KanaAndKunTest(unittest.TestCase):
    def texts(self, sentences, extras):
        return [r.text for r in mask_r3(sentences, extra_terms=extras).records]

    def test_kana_name_after_particle_and_before_kanji_is_masked(self):
        out = self.texts(
            ["うちのなゆとと申します。", "ひかり先生が来ます。", "店は、継がんて、ゆうとるんだわ。"],
            [("なゆと", KIND_PERSON), ("ひかり", KIND_PERSON), ("ゆうと", KIND_PERSON)],
        )
        self.assertNotIn("なゆと", out[0])
        self.assertNotIn("ひかり", out[1])
        self.assertIn("ゆうとる", out[2])

    def test_kun_chan_names_are_rule_candidates(self):
        from shokoban.name_candidates import rule_candidates

        got = [w for w, _ in rule_candidates(["大庭くんと愛ちゃんとみのりちゃん。", "お兄ちゃんとじいちゃん。"])]
        self.assertEqual(sorted(got), sorted(["大庭", "愛", "みのり"]))

class FullNameSplitTest(unittest.TestCase):
    def test_kanji_surname_and_kana_given_name_are_split(self):
        from shokoban.name_candidates import llm_candidates

        def fake(_text):
            return [{"word": "千葉ゆかり", "kind": "人名", "judgment": "確か"}]

        got, _ = llm_candidates(["千葉ゆかりです。千葉が来ます。"], fake)
        self.assertEqual({w for w, _, _ in got}, {"千葉ゆかり", "千葉", "ゆかり"})

    def test_kana_chan_does_not_swallow_particle(self):
        from shokoban.name_candidates import rule_candidates

        got = [w for w, _ in rule_candidates(["うちとのおばあちゃんと、ひかりちゃん。"])]
        self.assertEqual(got, ["ひかり"])


# ---------------------------------------------------------------- GiNZA・読みの網・12B の二度見

class GinzaNamesScriptTest(unittest.TestCase):
    """GiNZA で名前を拾うスクリプト（``ginza_names``）の純粋な関数（spaCy は読まない）."""

    def test_import_does_not_pull_spacy(self):
        import sys

        import scripts.ginza_names  # noqa: F401

        self.assertNotIn("spacy", sys.modules)

    def test_split_sentences_agrees_with_pick_bench(self):
        from scripts.ginza_names import split_sentences as mine
        from shokoban.pick_bench import split_sentences as original

        for text in (
            "山本です。よろしく！\n次の話です？？ええ。",
            "  \r\n改行だけ\r\n\r\n終わり",
            "",
            "。。まず。！？そう!?あ",
        ):
            self.assertEqual(mine(text), original(text))

    def test_label_mapping(self):
        from scripts.ginza_names import ORG_LABELS, kind_for_ner_label

        self.assertEqual(kind_for_ner_label("Person"), KIND_PERSON)
        for label in ("Company", "Company_Group", "Corporation_Other", "Organization_Other", "School"):
            self.assertEqual(kind_for_ner_label(label), KIND_ORG)
            self.assertIn(label, ORG_LABELS)
        for label in ("City", "Province", "Date", "Product_Other", "Facility_Other", "Government", "Title_Other"):
            self.assertIsNone(kind_for_ner_label(label))

    def test_tail_stripping(self):
        from scripts.ginza_names import ORG_TAILS, PERSON_TAILS, strip_tail

        self.assertEqual(strip_tail("千葉様", PERSON_TAILS), "千葉")
        self.assertEqual(strip_tail("トミちゃん", PERSON_TAILS), "トミ")
        self.assertEqual(strip_tail("さん", PERSON_TAILS), "さん")  # 空になるときは外さない
        self.assertEqual(strip_tail("丸栄材木さんとこ", ORG_TAILS), "丸栄材木")
        self.assertEqual(strip_tail("丸栄材木のとこ", ORG_TAILS), "丸栄材木")
        self.assertEqual(strip_tail("丸栄さんところ", ORG_TAILS), "丸栄")
        self.assertEqual(strip_tail("丸栄材木さん", ORG_TAILS), "丸栄材木")
        self.assertEqual(strip_tail("丸栄", ORG_TAILS), "丸栄")

    def test_items_from_analysis(self):
        from scripts.ginza_names import items_from_analysis

        sentence = "千葉ゆかりさんと大庭と丸栄材木さんとこに行きました。"
        items = items_from_analysis(
            sentence,
            3,
            [("大庭", "名詞-固有名詞-人名-姓"), ("ナカ", "名詞-固有名詞-人名-名"), ("丸栄", "名詞-固有名詞-一般"), ("行き", "動詞-一般")],
            [("千葉ゆかりさん", "Person"), ("丸栄材木さんとこ", "Company"), ("名古屋", "City"), ("大庭", "Person")],
        )
        got = {(i["word"], i["kind"]) for i in items}
        # 「ナカ」は文に無いので落ちる。地名・一般の固有名詞（品詞だけ）は拾わない。同じ語・kind は1つ。
        self.assertEqual(got, {("大庭", KIND_PERSON), ("千葉ゆかり", KIND_PERSON), ("丸栄材木", KIND_ORG)})
        self.assertEqual(len(items), 3)
        self.assertTrue(all(i["sentence"] == 3 for i in items))
        by_word = {i["word"]: i["label"] for i in items}
        self.assertEqual(by_word["大庭"], "名詞-固有名詞-人名-姓")
        self.assertEqual(by_word["丸栄材木"], "Company")

    def test_write_private_json_is_0600(self):
        from scripts.ginza_names import write_private_json

        with tempfile.TemporaryDirectory() as tmp:
            path = write_private_json(Path(tmp) / "out", "n01.json", {"id": "n01", "items": []})
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"id": "n01", "items": []})

    def test_read_book_two_formats(self):
        from scripts.ginza_names import read_book

        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.json"
            a.write_text(json.dumps({"id": "a", "sentences": [{"text": "一。"}, {"text": "二。"}]}), encoding="utf-8")
            b = Path(tmp) / "b.json"
            b.write_text(json.dumps({"id": "b", "transcript": "一。二。"}), encoding="utf-8")
            self.assertEqual(read_book(a), ("a", ["一。", "二。"]))
            self.assertEqual(read_book(b), ("b", ["一。", "二。"]))


class GinzaCandidatesTest(unittest.TestCase):
    SENTENCES = ["千葉ゆかりさんが来ました。", "丸栄材木に頼みます。", "社長と店長です。"]

    def test_ginza_items_become_sure_candidates_with_source(self):
        items = [
            {"word": "千葉ゆかり", "kind": KIND_PERSON, "label": "Person", "sentence": 1},
            {"word": "丸栄材木", "kind": KIND_ORG, "label": "Company", "sentence": 2},
        ]
        result = find_candidates(self.SENTENCES, ginza_items=items)
        by_term = {c.term: c for c in result.candidates}
        self.assertEqual(by_term["丸栄材木"].sources, ("ginza",))
        self.assertEqual(by_term["丸栄材木"].kind, KIND_ORG)
        self.assertEqual(by_term["丸栄材木"].judgment, JUDGMENT_SURE)
        # 漢字の姓＋かなの名は、姓と名も別に足される（12B と同じ）。
        self.assertEqual(set(by_term), {"千葉ゆかり", "千葉", "ゆかり", "丸栄材木"})
        self.assertEqual((result.ginza.items_raw, result.ginza.items_dropped), (2, 0))

    def test_same_validation_as_12b_items(self):
        items = [
            {"word": "無い語", "kind": KIND_PERSON, "label": "Person", "sentence": 1},  # 原文に無い
            {"word": "社長", "kind": KIND_PERSON, "label": "Person", "sentence": 3},  # 役割語
            {"word": "店長", "kind": KIND_PERSON, "label": "Person", "sentence": 3},
            {"word": "山本", "kind": "それ以外", "label": "X", "sentence": 1},  # kind が違う
            {"word": "千葉ゆかりさん", "kind": "人名", "label": "Person", "sentence": 1},  # 敬称は外す
            "壊れた",
            {"word": 5, "kind": "人名"},
        ]
        result = find_candidates(self.SENTENCES, ginza_items=items)
        self.assertEqual(result.ginza.items_raw, 7)
        self.assertIn("千葉ゆかり", {c.term for c in result.candidates})  # 敬称を外した形
        terms = {c.term for c in result.candidates}
        self.assertNotIn("無い語", terms)
        self.assertNotIn("社長", terms)
        self.assertNotIn("店長", terms)
        self.assertNotIn("山本", terms)
        self.assertGreaterEqual(result.ginza.dropped_reasons.get("原文に無い", 0), 1)
        self.assertGreaterEqual(result.ginza.dropped_reasons.get("役割語", 0), 2)

    def test_none_and_empty_mean_no_ginza(self):
        for ginza_items in (None, []):
            result = find_candidates(self.SENTENCES, ginza_items=ginza_items)
            self.assertNotIn("ginza", {s for c in result.candidates for s in c.sources})
            self.assertEqual(result.ginza.items_raw, 0)

    def test_sources_merge_with_rule_and_12b(self):
        sentences = ["山本さんが来ます。"]
        items = [{"word": "山本", "kind": KIND_PERSON, "label": "Person", "sentence": 1}]
        result = find_candidates(
            sentences, extractor=lambda text: [item("山本", KIND_PERSON, JUDGMENT_UNSURE)], ginza_items=items
        )
        (cand,) = [c for c in result.candidates if c.term == "山本"]
        self.assertEqual(cand.sources, ("rule", "ginza", "12b"))
        self.assertEqual(cand.judgment, JUDGMENT_SURE)

    def test_known_dict_words_are_not_ginza_candidates(self):
        known = empty_dict()
        known.add_pass("丸栄材木")
        items = [{"word": "丸栄材木", "kind": KIND_ORG, "label": "Company", "sentence": 2}]
        result = find_candidates(self.SENTENCES, ginza_items=items, name_dict=known)
        self.assertEqual(result.candidates, [])


class ReadingNetTest(unittest.TestCase):
    def words(self, sentence: str) -> list[str]:
        return [w for w, _ in rule_candidates([sentence])]

    def test_reading_explanations(self):
        self.assertEqual(self.words("畑と書いて、はたって読むんです。"), ["はた"])
        self.assertEqual(self.words("とうかいりん、って読みます。"), ["とうかいりん"])
        self.assertEqual(self.words("とうかいりんって読みます。"), ["とうかいりん"])
        self.assertEqual(self.words("カタカナでハタと書いてください。"), ["ハタ"])
        self.assertEqual(self.words("みのりって呼んでます。"), ["みのり"])
        self.assertEqual(self.words("まことと呼んでください。"), ["まこと"])

    def test_candidates_are_people(self):
        self.assertEqual(rule_candidates(["はたって読む。"]), [("はた", KIND_PERSON)])

    def test_non_names_are_excluded(self):
        for sentence in ("これ、なんて読むんですか。", "どう読むの。", "それって読みにくい。", "こうって書いて。", "あれと読む。"):
            self.assertEqual(self.words(sentence), [], sentence)

    def test_demonstrative_and_verb_endings_are_cut(self):
        self.assertEqual(self.words("これはたって読む。"), ["はた"])
        self.assertEqual(self.words("書いてはたって読む。"), ["はた"])

    def test_leading_particle_gives_both_forms(self):
        # 漢字の後ろの助詞「は」は外す。残りの頭の「に」は名前の頭かもしれないので両方。
        self.assertEqual(sorted(self.words("店はにあぶくまって読む。")), ["あぶくま", "にあぶくま"])

    def test_okurigana_before_the_reading_is_not_a_name(self):
        # 漢字のすぐ後ろのかなは送りがな・助詞。前が助詞1字なら外した残りを見る。
        self.assertEqual(self.words("それが絶対の決まりと勘違いされる。"), [])
        self.assertEqual(self.words("熨斗を付けると書いてある。"), [])
        self.assertEqual(self.words("人によって右と左を間違える。"), [])
        self.assertEqual(self.words("別の方へよけておくと間違いにくい。"), [])
        self.assertIn("たかなし", self.words("名字は小鳥遊でたかなしって読む。"))

    def test_one_char_and_long_words_are_not_taken(self):
        self.assertEqual(self.words("たって読む。"), [])
        self.assertEqual(self.words("漢字って読む。"), [])  # かなではない

    def test_reading_name_is_masked_end_to_end(self):
        sentences = ["はたって読むんです。", "はたさんが来ます。"]
        extras = terms_to_mask(None, find_candidates(sentences).candidates)
        out = [r.text for r in mask_r3(sentences, extra_terms=extras).records]
        self.assertNotIn("はた", out[0])
        self.assertNotIn("はた", out[1])


class HideTermsTest(unittest.TestCase):
    def test_hides_person_and_org_longest_first(self):
        from shokoban.name_candidates import hide_terms

        text = hide_terms("千葉ゆかりさんは丸栄材木の千葉です。", [("千葉", KIND_PERSON), ("千葉ゆかり", KIND_PERSON), ("丸栄材木", KIND_ORG)])
        self.assertEqual(text, "〈名前〉さんは〈社名〉の〈名前〉です。")

    def test_kana_term_inside_a_word_is_not_hidden(self):
        from shokoban.name_candidates import hide_terms

        self.assertEqual(hide_terms("みんなからなかが来る。", [("なか", KIND_PERSON)]), "みんなから〈名前〉が来る。")

    def test_other_kinds_are_not_hidden(self):
        from shokoban.name_candidates import hide_terms

        self.assertEqual(hide_terms("商品です。", [("商品", KIND_OTHER)]), "商品です。")


class SecondPassTest(unittest.TestCase):
    SENTENCES = ["山本さんと光が来ます。", "丸栄材木に頼みます。", "光は元気です。"]

    def fake(self, seen: list[str], reply):
        def extractor(text: str):
            seen.append(text)
            return reply
        return extractor

    def test_second_pass_shows_hidden_text_and_adds_12b2_words(self):
        seen: list[str] = []
        first = [item("山本"), item("丸栄材木", KIND_ORG)]
        calls = {"n": 0}

        def extractor(text: str):
            calls["n"] += 1
            seen.append(text)
            if calls["n"] == 1:
                return first
            return [item("光", KIND_PERSON, JUDGMENT_UNSURE), item("山本"), item("元気")] + [item("ここにない")]

        result = find_candidates(self.SENTENCES, extractor=extractor, second_pass=True)
        self.assertEqual(len(seen), 2)
        self.assertIn("山本", seen[0])
        self.assertNotIn("山本", seen[1])
        self.assertNotIn("丸栄材木", seen[1])
        self.assertIn("〈名前〉さんと光が来ます。", seen[1])
        self.assertIn("〈社名〉に頼みます。", seen[1])
        by_term = {c.term: c for c in result.candidates}
        self.assertEqual(by_term["光"].sources, ("12b2",))
        self.assertEqual(by_term["光"].judgment, JUDGMENT_UNSURE)
        self.assertEqual(by_term["山本"].sources, ("rule", "12b"))  # 既出は二度見で足さない
        # 12B の初回の統計と二度見の統計は別。
        self.assertEqual((result.llm.chunks_ok, result.llm.items_raw), (1, 2))
        self.assertEqual((result.llm2.chunks_ok, result.llm2.items_raw), (1, 4))
        self.assertGreaterEqual(result.llm2.dropped_reasons.get("原文に無い", 0), 1)
        self.assertEqual(result.llm2.dropped_reasons.get("既出"), 1)

    def test_off_by_default_makes_one_pass(self):
        seen: list[str] = []
        result = find_candidates(self.SENTENCES, extractor=self.fake(seen, []))
        self.assertEqual(len(seen), 1)
        self.assertEqual((result.llm2.chunks_ok, result.llm2.items_raw), (0, 0))

    def test_second_pass_words_must_be_in_the_original_sentences(self):
        calls = {"n": 0}

        def extractor(text: str):
            calls["n"] += 1
            return [] if calls["n"] == 1 else [item("〈名前〉"), item("名前〉さ")]

        result = find_candidates(["山本さんです。"], extractor=extractor, second_pass=True)
        self.assertEqual([c.term for c in result.candidates], ["山本"])

    def test_second_pass_failure_is_counted_and_first_results_survive(self):
        calls = {"n": 0}

        def extractor(text: str):
            calls["n"] += 1
            if calls["n"] == 2:
                raise NameReplyError("壊れた")
            return [item("山本")]

        result = find_candidates(["山本さんです。"], extractor=extractor, second_pass=True)
        self.assertEqual((result.llm2.chunks_ok, result.llm2.chunks_fail), (0, 1))
        self.assertEqual([c.term for c in result.candidates], ["山本"])

    def test_dictionary_and_company_terms_are_hidden_too(self):
        known = empty_dict()
        known.add_mask("田中", KIND_PERSON)
        seen: list[str] = []
        find_candidates(
            ["田中さんとほしの食堂です。"],
            extractor=self.fake(seen, []),
            name_dict=known,
            company_name="ほしの食堂",
            second_pass=True,
        )
        self.assertIn("〈名前〉さんと〈社名〉です。", seen[1])

    def test_llama_extractor_second_look_adds_one_sentence_to_prompt(self):
        from shokoban.name_candidates import NAME_PROMPT_SECOND, LlamaNameExtractor

        prompts: list[str] = []

        def generate(prompt: str) -> str:
            prompts.append(prompt)
            return "[]"

        extractor = LlamaNameExtractor("alias", generate)
        extractor("[1] 文")
        extractor.second_look("[1] 〈名前〉")
        self.assertNotIn(NAME_PROMPT_SECOND, prompts[0])
        self.assertIn(NAME_PROMPT_SECOND, prompts[1])
        self.assertEqual(prompts[1].replace(NAME_PROMPT_SECOND, "").count("〈社名〉"), 0)
        self.assertEqual(build_name_prompt("x"), prompts[0].replace("[1] 文", "x"))

    def test_real_extractor_second_pass_uses_second_look(self):
        from shokoban.name_candidates import LlamaNameExtractor, NAME_PROMPT_SECOND

        prompts: list[str] = []
        replies = iter(['[{"word":"山本","kind":"人名","judgment":"確か"}]', '[{"word":"光","kind":"人名","judgment":"迷う"}]'])

        def generate(prompt: str) -> str:
            prompts.append(prompt)
            return next(replies)

        result = find_candidates(
            ["山本さんと光です。"], extractor=LlamaNameExtractor("a", generate), second_pass=True
        )
        self.assertNotIn(NAME_PROMPT_SECOND, prompts[0])
        self.assertIn(NAME_PROMPT_SECOND, prompts[1])
        self.assertEqual({c.term: c.sources for c in result.candidates}["光"], ("12b2",))


class KanaSamaTest(unittest.TestCase):
    def test_kana_name_with_sama(self):
        from shokoban.name_candidates import rule_candidates

        got = [w for w, _ in rule_candidates(["えっ、ひゃくめおに様ですか。", "お疲れ様です。", "ご苦労様。"])]
        self.assertEqual(got, ["ひゃくめおに"])


class OverMaskVerbAndKanaTest(unittest.TestCase):
    """動詞の一部や仮名の並びを名前と取り違えて伏せすぎないことの確かめ."""

    def _masked(self, sentence: str, terms: list[tuple[str, str]]) -> str:
        return mask_r3([sentence], extra_terms=terms).records[0].text

    def test_verb_kun_is_not_a_name_candidate(self):
        got = [w for w, _ in rule_candidates(["誰がどこ行くって書くんです。", "よう働くんですよ。", "そのままにしとくんですか。", "出ていくんですよ。"])]
        self.assertEqual(got, [])
        got = [w for w, _ in rule_candidates(["光くんが来ました。", "大庭くんです。", "ゆうとくんが言ってた。"])]
        self.assertEqual(got, ["光", "大庭", "ゆうと"])

    def test_single_kanji_term_skips_verb_heads_but_keeps_names(self):
        terms = [("来", KIND_PERSON), ("誠", KIND_PERSON)]
        self.assertNotIn("〈名前〉", self._masked("大人の方も、二人、来てます。", terms))
        for sentence in ("誠が来た。", "誠って呼んどる。", "去年入った誠だな。", "誠か、ひかりに頼む。", "誠さんに。", "誠、ちょっと来い。"):
            self.assertIn("〈名前〉", self._masked(sentence, terms), sentence)

    def test_kana_start_term_inside_verb_is_not_masked(self):
        terms = [("てる子", KIND_PERSON)]
        self.assertNotIn("〈名前〉", self._masked("うちに来てる子は。", terms))
        self.assertNotIn("〈名前〉", self._masked("続けてる子は、伸びる。", terms))
        self.assertIn("〈名前〉", self._masked("隣のてる子さんが。", terms))
        self.assertIn("〈社名〉", self._masked("居酒屋まんぷく亭の佐賀さん。", [("まんぷく亭", KIND_ORG)]))

    def test_common_words_are_not_candidates(self):
        from shokoban.name_candidates import is_common_word

        merged = merge_candidates(
            ["LINEとチラシ、ハローワークにも。わんちゃんも。減らせると思う。line で。山本さん。"],
            [("山本", KIND_PERSON)],
            [("チラシ", KIND_PERSON, JUDGMENT_SURE), ("減らせる", KIND_ORG, JUDGMENT_SURE), ("line", KIND_ORG, JUDGMENT_SURE)],
            ginza=[("LINE", KIND_ORG, JUDGMENT_SURE), ("ハローワーク", KIND_ORG, JUDGMENT_SURE), ("わん", KIND_PERSON, JUDGMENT_SURE)],
        )
        self.assertEqual([c.term for c in merged], ["山本"])
        self.assertFalse(is_common_word("田中はる"))
        self.assertFalse(is_common_word("ゆき子"))


class DerivedCandidatesTest(unittest.TestCase):
    """法人の種類を外した取引先名と、漢字の名字に添えた読みから、候補を導く."""

    def masked(self, sentences, extractor=None):
        found = find_candidates(sentences, extractor=extractor, name_dict=empty_dict())
        extras = terms_to_mask(None, found.candidates)
        return found, [r.text for r in mask_r3(sentences, extra_terms=extras).records]

    def test_org_variants(self):
        from shokoban.name_candidates import org_variant_terms

        self.assertEqual(org_variant_terms("株式会社つばめ印刷"), ["つばめ印刷", "つばめ"])
        self.assertEqual(org_variant_terms("有限会社ミドリ精工"), ["ミドリ精工", "ミドリ"])
        self.assertEqual(org_variant_terms("（株）北斗ラベル"), ["北斗ラベル"])
        self.assertEqual(org_variant_terms("つばめ印刷"), [])  # 法人の種類が無いものは広げない

    def test_supplier_short_forms_are_masked(self):
        sentences = [
            "箱は株式会社つばめ印刷にお願いしています。",
            "つばめ印刷の営業部の方が来ます。",
            "つばめ印刷さんには入稿データを送ります。",
            "発注先は、つばめと、もう一社です。",
        ]
        found, out = self.masked(
            sentences, extractor=lambda text: [item("株式会社つばめ印刷", KIND_ORG, JUDGMENT_UNSURE)]
        )
        terms = {c.term: c for c in found.candidates}
        self.assertEqual(terms["つばめ"].judgment, JUDGMENT_UNSURE)
        for text in out:
            self.assertNotIn("つばめ", text)

    def test_reading_after_kanji_surname(self):
        sentences = [
            "いえ、鷹梨です、たかなし、鷹梨健吾です。",
            "鷹梨、タカナシって発音します。",
        ]
        found, out = self.masked(
            sentences, extractor=lambda text: [item("鷹梨", KIND_PERSON), item("鷹梨健吾", KIND_PERSON)]
        )
        self.assertIn("たかなし", {c.term for c in found.candidates})
        self.assertNotIn("たかなし", out[0])
        self.assertNotIn("タカナシ", out[1])

    def test_misreading_and_pronunciation_phrases(self):
        self.assertEqual(rule_candidates(["よくとびなしって間違われます。"]), [("とびなし", KIND_PERSON)])
        self.assertEqual(rule_candidates(["たかなしって発音します。"]), [("たかなし", KIND_PERSON)])

    def test_fillers_after_name_are_not_taken(self):
        from shokoban.name_candidates import appositive_readings

        for sentence in ("鷹梨です、はい、よろしく。", "鷹梨です、えっと、今日は。", "鷹梨は、そう、毎朝来ます。"):
            self.assertEqual(appositive_readings([sentence], ["鷹梨"]), [], sentence)

    def test_only_kanji_person_names_get_readings(self):
        from shokoban.name_candidates import appositive_readings

        self.assertEqual(appositive_readings(["タロウ、たろう、です。"], ["タロウ"]), [])

    def test_phrases_after_name_are_not_readings(self):
        from shokoban.name_candidates import appositive_readings

        for sentence in ("鷹梨さん、いいですね。", "鷹梨、っていう人です。", "鷹梨、すごいよ。", "鷹梨、ありがとうございました。"):
            self.assertEqual(appositive_readings([sentence], ["鷹梨"]), [], sentence)

    def test_readings_ending_in_na_or_wa_are_kept(self):
        from shokoban.name_candidates import appositive_readings

        self.assertEqual(appositive_readings(["石川です、いしかわ、石川陽介です。"], ["石川"]), ["いしかわ"])

    def test_reading_with_words_between(self):
        # 頭の助詞（は）付きと外した形の両方
        self.assertIn(("つるまる", KIND_PERSON), rule_candidates(["鶴丸はつるまるって濁らずに読むんですよ。"]))
        self.assertEqual(rule_candidates(["それって普通に読めますよね。"]), [])


class CompanyHeadTest(unittest.TestCase):
    """社名と別名に共通する頭を、お客様の短い呼び方として伏せる."""

    def test_common_head_of_name_and_aliases(self):
        from shokoban.name_candidates import company_core_terms

        self.assertIn("風継", company_core_terms("風継サイクル舎", ["風継サイクル", "風継輪業"]))

    def test_one_char_head_is_not_used(self):
        from shokoban.name_candidates import company_core_terms

        self.assertNotIn("山", company_core_terms("山田商店", ["山本工房"]))


class OverMaskCommonWordsTest(unittest.TestCase):
    """名前は伏せたまま、一般の語・役職・「ちゃん」付きの語は候補にしない."""

    def rule_words(self, sentence: str) -> list[str]:
        return [word for word, _ in rule_candidates([sentence])]

    def test_honorific_inside_a_word(self):
        self.assertEqual(self.rule_words("権田くんが丁寧に整備している様子を載せたい。"), ["権田"])
        self.assertEqual(self.rule_words("三つ折りを最終仕様にした。"), [])
        self.assertEqual(self.rule_words("借り入れの話まで全部消さんといてほしい。"), [])
        self.assertEqual(self.rule_words("田中様にお送りします。"), ["田中"])

    def test_chanto_and_baby_words(self):
        self.assertEqual(self.rule_words("仕事の話やからちゃんと残してほしい。"), [])
        self.assertEqual(self.rule_words("引き継ぎをちゃんとしなかった。"), [])
        self.assertEqual(self.rule_words("双子の赤ちゃんと長男がいる。"), [])
        self.assertEqual(self.rule_words("月に何頭ぐらいワンちゃんを見ていますか。"), [])
        self.assertEqual(self.rule_words("すみれちゃんと一緒に来ます。"), ["すみれ"])

    def test_roles_and_family_with_honorific(self):
        for sentence in (
            "ドライバーさんの仕事ぶりを撮影します。",
            "看護師さんの募集ページを作りたい。",
            "毎月ケアマネさんに送る資料です。",
            "カメラマンさんを入れて撮影した。",
            "設備担当者さんに聞いてみます。",
            "頼もしいお弟子さんが育っている。",
            "息子さんにも見てもらいます。",
            "オーナーさんの写真を載せます。",
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(self.rule_words(sentence), [])

    def test_common_words_and_ordinals(self):
        from shokoban.name_candidates import is_common_word

        for word in ("せや", "提案書", "四つ目", "百貨店", "ヘッドスパ"):
            with self.subTest(word=word):
                self.assertTrue(is_common_word(word))
        self.assertFalse(is_common_word("四谷"))

    def test_role_words_from_12b_are_dropped(self):
        kept, _ = validate_items(
            [{"word": "商工会議所", "kind": KIND_PERSON, "judgment": JUDGMENT_SURE},
             {"word": "ベテランヘルパー", "kind": KIND_PERSON, "judgment": JUDGMENT_SURE},
             {"word": "山本", "kind": KIND_PERSON, "judgment": JUDGMENT_SURE}],
            ["商工会議所とベテランヘルパーさんと山本さん。"],
        )
        self.assertEqual([word for word, _, _ in kept], ["山本"])
