"""伏せる部品（``mask_r3``）の試験.

本物の12B（llama-server）は一切起動しない。札付け役は偽の関数、``llm_runtime`` は偽の
Runtime で試す。文は、この試験が書いた架空の短い文だけを使う。
"""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path
from typing import Any

from shokoban.mask_r3 import (
    DEFERRAL_RE,
    MONEY_EXCLUDE_UNITS,
    REGION_BEFORE,
    REGION_END_RUN,
    ROUTE_BUSINESS,
    ROUTE_HEALTH,
    ROUTE_HIDDEN,
    ROUTE_FAMILY,
    ROUTE_IN_REGION,
    ROUTE_OUTSIDE,
    ROUTE_REPLY,
    TAGGER_SCHEMA,
    TaggerReplyError,
    build_tagger_prompt,
    classify_clause,
    collect_names,
    compute_regions,
    dry_tagger,
    find_cues,
    hints_for,
    is_business_text,
    is_safe_response,
    is_safe_work_sentence,
    make_llama_tagger,
    mask_r3,
    number_spans,
    parse_tagger_reply,
    spelling_spans,
)
from shokoban.mask_stage5 import (
    MASK_HEALTH,
    MASK_NUMBER,
    MONEY_BANDS,
    money_band_label,
)
from shokoban.stage5_r3_types import (
    MASK_FAMILY,
    MASKED_SENTENCE,
    OMITTED,
    REGION_TAG_EMPTY,
    TROUBLE_MENU,
    MaskedR3,
)


def labels(text: str) -> list[str]:
    return [span.label for span in number_spans(text)]


def cue_texts(text: str) -> list[str]:
    return [text[cue.start : cue.end] for cue in find_cues(text)]


def masked(sentences: list[str], **kwargs: Any) -> MaskedR3:
    return mask_r3(sentences, **kwargs)


class NumberTest(unittest.TestCase):
    """数の途中の結果（金額の札になるか）。最終の出し方は ``FinalOutputTest`` で別に見る."""

    def test_man_with_money_context_becomes_money_label(self):
        self.assertEqual(labels("千五百万、借りて"), ["〈金額：百万円以上〉"])
        spans = number_spans("千五百万、借りて")
        self.assertEqual((spans[0].start, spans[0].end), (0, 4))

    def test_yen_amount_uses_band_table(self):
        self.assertEqual(labels("十五万円"), ["〈金額：10万円台〉"])
        self.assertEqual(labels("五百円"), ["〈金額：1万円未満〉"])
        self.assertEqual(labels("三万円"), ["〈金額：数万円〉"])

    def test_money_band_boundaries(self):
        for value, band in (
            (9_999, "〈金額：1万円未満〉"),
            (10_000, "〈金額：数万円〉"),
            (99_999, "〈金額：数万円〉"),
            (100_000, "〈金額：10万円台〉"),
            (199_999, "〈金額：10万円台〉"),
            (200_000, "〈金額：20〜40万円台〉"),
            (499_999, "〈金額：20〜40万円台〉"),
            (500_000, "〈金額：50〜90万円台〉"),
            (999_999, "〈金額：50〜90万円台〉"),
            (1_000_000, "〈金額：百万円以上〉"),
        ):
            self.assertEqual(money_band_label(value), band)
        self.assertEqual(labels("9999円"), ["〈金額：1万円未満〉"])
        self.assertEqual(labels("10000円"), ["〈金額：数万円〉"])
        self.assertEqual(labels("100000円"), ["〈金額：10万円台〉"])
        self.assertEqual(len(MONEY_BANDS), 6)

    def test_man_without_money_context_is_number(self):
        self.assertEqual(labels("三万"), [MASK_NUMBER])
        self.assertEqual(number_spans("三万")[0].kind, "数字")
        self.assertEqual(labels("あれは五万くらいあるね"), [MASK_NUMBER])

    def test_quantity_unit_after_man_keeps_text(self):
        # 作業の文脈が無い「一万部」は〈数字〉になる（金額の札にはならない）。作業の文なら残す。
        self.assertFalse([l for l in labels("一万部") if l.startswith("〈金額")])
        self.assertEqual(number_spans("チラシを一万部刷りましょう"), ())
        self.assertEqual(number_spans("チラシを一万部刷りたい"), ())
        # 「三万人」のような規模の数は作業の量ではないので〈数字〉（金額の札にはならない）。
        self.assertEqual(labels("三万人が来る"), [MASK_NUMBER])
        # お金の文脈があっても、単位が付く数は金額から外す。
        self.assertEqual(labels("売上で一万件"), [MASK_NUMBER])  # 金額にはならず〈数字〉
        # 円が付くなら単位にかかわらず金額。
        self.assertEqual(labels("一万円"), ["〈金額：数万円〉"])

    def test_money_context_is_judged_on_whole_sentence(self):
        # 隣の節にお金の文脈がある。
        self.assertEqual(labels("借りたのは、三千万で"), ["〈金額：百万円以上〉"])
        self.assertEqual(labels("三千万、ありまして"), [MASK_NUMBER])

    def test_arabic_numbers_are_masked_unless_work_quantity(self):
        self.assertEqual(labels("2026年にします"), [MASK_NUMBER])
        self.assertEqual(number_spans("500部作ります"), ())
        self.assertEqual(labels("1,500人"), [MASK_NUMBER])
        self.assertEqual(number_spans("12ページ"), ())

    def test_business_counts_are_masked_without_work_context(self):
        # 区間の外で節を落とさないので、漢数字の人数・食数・棟数・頭数・割合・所要時間・
        # 大きい枚数などは、その数だけ〈数字〉にする。
        for text in (
            "平日は平均で八人くらいで",
            "昼は四十食くらいですかね",
            "年間十二棟くらいですかね",
            "一日に八匹くらいで",
            "七割くらいは常連さんで",
            "だいたい三時間かかります",
            "三千枚くらいですかね",
            "電話が一日に二十件くらい来る",
            "ひと箱に二十個くらい入れて",
        ):
            self.assertEqual(labels(text), [MASK_NUMBER], text)

    def test_work_quantities_are_kept(self):
        # 作業の語・提案・依頼の文の中の数量、作業の量の単位、1桁の数は残す。
        for text in (
            "チラシは三千枚で出しましょう",
            "施工事例は六件くらい載せたい",
            "ネット予約の枠を一日八人までにしておきましょうか",
            "写真は三枚撮ってください",
            "二回に分けませんか",
            "月に二回くらい",
            "一人で対応している",
        ):
            self.assertEqual(number_spans(text), (), text)

    def test_spoken_phone_number_is_a_contact(self):
        result = masked(["電話は0000の00の0000です。"])
        self.assertIn("〈連絡先〉", result.records[0].text)
        self.assertNotIn("0000", result.records[0].text)

    def test_not_a_number_words_are_left(self):
        self.assertEqual(number_spans("万が一のとき"), ())
        self.assertEqual(number_spans("万全にします"), ())
        self.assertEqual(number_spans("一緒にやります"), ())

    def test_exclude_units_are_separate_from_work_quantity_units(self):
        from shokoban.pick_bench import _QUANTITY_UNITS

        self.assertNotEqual(set(MONEY_EXCLUDE_UNITS), set(_QUANTITY_UNITS))
        self.assertIn("人", MONEY_EXCLUDE_UNITS)
        self.assertNotIn("人", _QUANTITY_UNITS)


class CueTest(unittest.TestCase):
    """危ない手がかり."""

    def test_family_and_health_cues_hit(self):
        # 手がかりは体調と家族だけ。
        for text in (
            "腰が、もう、あかんくて",
            "ぎっくり腰をやりまして",
            "離婚したあと、ほとんど、ひとりで抱えてて",
            "ほんまは、親の介護で、もうあんまり、来れんのですよ",
            "相続の話も、出てきちゃって",
            "実は、うちの、父と、兄が、店のことで、もう、口もきかんくて",
            "お子さんが、熱を、出したとかで、今日、行けません",
        ):
            self.assertTrue(find_cues(text), text)

    def test_family_cues_are_kind_family_and_health_cues_are_kind_health(self):
        self.assertEqual({c.kind for c in find_cues("離婚したあと、ひとりで")}, {"家族"})
        self.assertEqual({c.kind for c in find_cues("腰が、もう、あかんくて")}, {"体調"})

    def test_business_phrases_do_not_hit(self):
        # 事業の話（借入・返済・資金繰り・支払いの遅れ・値下げ要求・取引先との揉め事・
        # 苦情・口コミ・訴訟・退職や独立・物件・後継ぎ）は渡してよいので、手がかりにしない。
        for text in (
            "借りたお金がまだ残ってて",
            "毎月、けっこう、返してるんですよね",
            "三割、下げろと",
            "急に来なくなって",
            "資金繰りが厳しい",
            "弁護士さんに相談している",
            "取引先の、アパレルさんが、工賃を、また、下げたいって",
            "じつは、取次さんから、返品の条件が、きびしくなって",
            "月末の支払いも、一回、遅らせてもろてな",
            "SNSに、悪い、書き込みを、されてしまったんです",
            "実は、先月、お客さんと、ちょっと、トラブルに、なりまして",
            "それとね、材木屋さんと、ちょっと、金額のことで、もめてるんですよ",
            "実は、パティシエの子が、二人、続けて辞めまして",
            "息子は、継ぐ気が、ないみたいで",
            "で、息子もいるにはいるんですけど、東京で、ぜんぜん違う仕事をしてて",
            "それで、あと何年できるか分からないので",
            "あの、あんまり書かんといてほしいんですけど、元請けの会社があるんです",
            "親父が、昭和の終わりに、始めた会社なんですよ",
            "うちは、家族で、やってる、農園です",
        ):
            self.assertEqual(cue_texts(text), [], text)

    def test_safe_overrides_do_not_hit(self):
        for text in (
            "予定を辞めて",
            "メールを返して",
            "値段を下げたセットメニューを作りたい",
        ):
            self.assertEqual(find_cues(text), (), text)

    def test_ordinary_business_phrases_do_not_hit(self):
        # 当たりすぎやすい、普通の業務の言い方。
        for text in (
            "あの、市民の、ホールを、借りて。",
            "訪問介護の、事業所です。",
            "三宅っていう、介護福祉士で。",
            "ハローワークに出しても、介護の、求人って、いっぱいあるでしょ。",
            "決めてへんのは、まず、お金の、払い方、ですね。",
            "なので、あんまりお金をかけられないんです。",
            "誰かに引き継げる形にしないと、まずいなと。",
            "私の頭の中に、だいたい、これくらい、というのがあって。",
            "レッスンの合間に返してるんですけど。",
            "すぐ返さんと、悪いなと思って。",
            "夜中にスマホを見ちゃって、返さないと落ち着かなくて。",
            "字が、つぶれて、読めへんことが、あって。",
        ):
            self.assertEqual(cue_texts(text), [], text)

    def test_added_cues_hit(self):
        # 公開すると漏れてしまう言い方。
        for text in (
            "産んだら、すぐ戻るつもりなんですけど。",
            "なんせ、子どもの送り迎えも、わしが、せなあかんから。",
            "最近、うちの中が、ちょっと、落ち着かなくて。",
            "ちょっとだけ、体のことも、ええですか。",
            "まあ、一身上の都合ということで。",
            "で、兄が、一人、おるんですけど。",
            "で、その人が、去年、ちょっと、腰をやって、二週間休んだんですよ。",
            "毎日、くたくたでよ。",
            "兄は、お金だけ出して、あとは知らん顔で。",
            "嫁にも、さんざん、言われてまって。",
            "でも、辞めたら、親に、何て言えばいいか。",
            "去年の末に、喧嘩して、出ていってしまって。",
            "いまは、もう、話も、してなくて。",
            "あと、バイトの子が、最近、ちょっと、元気ないんすよね。",
            "朝、佐伯さんが、子供の学校の用事で、遅れるって言うて。",
            "女房は、夕方、伝票の整理だけ、手伝いに来てくれてます。",
            "ほんまに、あのときは、しんどかった。",
            "夕方になると、パソコンの前で寝てたりするんですよ。",
        ):
            self.assertTrue(find_cues(text), text)

    def test_override_cancels_only_the_matching_cue(self):
        text = "メールを返して、腰が痛い"
        self.assertEqual(cue_texts(text), ["腰が痛い"])

    def test_override_does_not_cancel_other_cue_in_same_clause(self):
        # 「手が回らない」は打ち消すが、同じ節の「肩が痛い」は打ち消さない。
        text = "手が回らなくて、肩が痛い"
        self.assertEqual([c for c in cue_texts(text) if "肩" in c], ["肩が痛い"])

    def test_health_cue_range_is_decided_by_regex(self):
        text = "腰痛のため投稿をお願いします"
        health = [c for c in find_cues(text) if c.kind == "体調"]
        self.assertEqual([text[c.start : c.end] for c in health], ["腰痛"])
        text2 = "膝が、もう、言うこと聞かんで"
        health2 = [c for c in find_cues(text2) if c.kind == "体調"]
        self.assertEqual(text2[health2[0].start : health2[0].end], text2)

    def test_cue_has_no_original_word_outside_text(self):
        cue = find_cues("離婚した話が残る")[0]
        self.assertEqual({"kind", "name", "start", "end"}, set(vars(cue)))


class ClauseTypeTest(unittest.TestCase):
    """節の型・安全な応答型."""

    def test_money_for_work_is_not_a_work_clause(self):
        self.assertEqual(classify_clause("チラシ代も今月は出せません"), "型なし")

    def test_work_clause(self):
        self.assertEqual(classify_clause("チラシは五百部で"), "作業")

    def test_other_types(self):
        self.assertEqual(classify_clause("写真を撮ってもらえますか"), "作業")  # 作業の語があれば作業
        self.assertEqual(classify_clause("お願いしたいです"), "依頼")
        self.assertEqual(classify_clause("やっておきます"), "引き受け")
        self.assertEqual(classify_clause("それでお願いします"), "依頼")
        self.assertEqual(classify_clause("それでいきましょう"), "同意")
        self.assertEqual(classify_clause("費用の見通しが不安で"), "許された困りごと")
        self.assertEqual(classify_clause("人手が足りなくて"), "許された困りごと")
        self.assertEqual(classify_clause("あの"), "つなぎ")
        self.assertEqual(classify_clause("まあ、"), "つなぎ")

    def test_word_alone_is_not_a_type(self):
        self.assertEqual(classify_clause("息子がチラシを見て"), "型なし")  # 家族の話が混ざる
        self.assertEqual(classify_clause("遅れそうで"), "型なし")  # 「そうで」は同意でない

    def test_safe_response_is_whole_sentence_match(self):
        for text in ("はい。", "あ、それでお願いします。", "そうですね、はい。", "ありがとうございます。"):
            self.assertTrue(is_safe_response(text), text)
        for text in ("はい、チラシをお願いします。", "ありがとうございます、でも不安で。", ""):
            self.assertFalse(is_safe_response(text), text)

    def test_safe_response_and_agreement_is_reply_not_end_candidate(self):
        result = masked(["離婚したあと、ひとりで抱えてて。", "それでお願いします。", "分かりました。"])
        self.assertTrue(result.records[1].region_reply)
        self.assertTrue(result.records[2].region_reply)

    def test_hints_come_from_clause_endings(self):
        self.assertEqual(hints_for("写真をお願いしたい"), ("依頼の型",))
        self.assertEqual(hints_for("やっておきます"), ("引き受けの型",))
        self.assertEqual(hints_for("それでいきましょう"), ("同意の型",))
        self.assertIn("先送りの型", hints_for("それは次回にしましょう"))
        self.assertTrue(DEFERRAL_RE.search("あとで"))


class RegionTest(unittest.TestCase):
    """区間の決め方（入出力例）."""

    @staticmethod
    def regions(pattern: str) -> list[tuple[int, int]]:
        risky = [c == "H" for c in pattern]
        candidate = [c == "S" for c in pattern]  # R（応答型）・x は終わりの候補でない
        return compute_regions(risky, candidate)

    def test_region_shape_constants(self):
        # 始まりは手がかりの文の1文前、終わりは安全な終わりの候補が1文出たらその手前まで。
        self.assertEqual(REGION_BEFORE, 1)
        self.assertEqual(REGION_END_RUN, 1)

    def test_region_examples(self):
        # 入出力例（REGION_BEFORE=1・REGION_END_RUN=1）。H=手がかりの文、S=終わりの候補、x=その他。
        # S S S H x S S S S S: 4 の1つ前から、最初の S（6）の手前の 5 まで。
        self.assertEqual(self.regions("SSSHxSSSSS"), [(3, 5)])
        # H x R x S x S S S S: R は終わりの候補でない。5 の S の手前の 4 まで。
        self.assertEqual(self.regions("HxRxSxSSSS"), [(1, 4)])
        # S S H S S H S S S S: 3 は 4（S）の手前で止まり、
        # 6 は 7（S）の手前で止まる。間の 4 は S なので離れたまま。
        self.assertEqual(self.regions("SSHSSHSSSS"), [(2, 3), (5, 6)])
        # S S S S S S S S H x: 9 の1つ前の 8 から最後まで。
        self.assertEqual(self.regions("SSSSSSSSHx"), [(8, 10)])

    def test_wider_values_give_wider_regions(self):
        # 値を引数で渡せば、2文前から・S が2文続くまでの広い区間も出せる。
        def old(pattern: str) -> list[tuple[int, int]]:
            return compute_regions(
                [c == "H" for c in pattern], [c == "S" for c in pattern], before=2, end_run=2
            )

        self.assertEqual(old("SSSHxSSSSS"), [(2, 5)])
        self.assertEqual(old("HxRxSxSSSS"), [(1, 6)])
        self.assertEqual(old("SSHSSHSSSS"), [(1, 6)])
        self.assertEqual(old("SSSSSSSSHx"), [(7, 10)])

    def test_edges_and_extension(self):
        self.assertEqual(self.regions("Hxxxxxxxxx"), [(1, 10)])
        self.assertEqual(self.regions("xxxxxxxxxx"), [])
        # 手がかりが続けば区間は延びる: 3 の区間は 7（S）の手前まで、5 の区間とまとまる。
        self.assertEqual(self.regions("SSHxHxSSSS"), [(2, 6)])
        with self.assertRaises(ValueError):
            compute_regions([True], [True, False])
        with self.assertRaises(ValueError):
            compute_regions([True], [True], end_run=0)

    def test_two_regions_stay_apart(self):
        self.assertEqual(self.regions("HSSSSSSSSH"), [(1, 1), (9, 10)])

    def test_replies_do_not_end_region_end_to_end(self):
        sentences = [
            "離婚したあと、ひとりで抱えてて。",
            "それでお願いします。",
            "分かりました。",
            "それでお願いします。",
            "分かりました。",
        ]
        result = masked(sentences)
        self.assertEqual([(g.start, g.end) for g in result.regions], [(1, 5)])
        self.assertEqual([r.region for r in result.records], [1] * 5)

    def test_two_safe_candidates_end_region_end_to_end(self):
        sentences = [
            "離婚したあと、ひとりで抱えてて。",
            "チラシは五百部です。",
            "写真は三枚です。",
            "ホームページは一ページです。",
        ]
        result = masked(sentences)
        self.assertEqual([(g.start, g.end) for g in result.regions], [(1, 1)])
        self.assertEqual([r.region for r in result.records], [1, None, None, None])


class FinalOutputTest(unittest.TestCase):
    """金額の最終の出し方、1文の出し方."""

    def test_man_money_final_output(self):
        # 借入そのものは渡してよい。金額だけ幅の札になる（区間は開かない）。
        result = masked(["千五百万、借りて。"])
        self.assertEqual(result.records[0].text, "〈金額：百万円以上〉、借りて。")
        self.assertEqual(result.regions, ())

    def test_band_label_in_published_clause(self):
        result = masked(["チラシの予算は十五万円です。"])
        self.assertEqual(result.records[0].text, "チラシの予算は〈金額：10万円台〉です。")
        self.assertEqual(result.records[0].state, "公開")

    def test_number_in_work_clause_is_masked(self):
        result = masked(["チラシは2026年に出します。"])
        self.assertIn(MASK_NUMBER, result.records[0].text)

    def test_health_range_is_replaced_and_work_clause_stays(self):
        result = masked(["腰痛のため投稿をお願いします。"])
        record = result.records[0]
        self.assertEqual(record.text, f"{MASK_HEALTH}のため投稿をお願いします。")
        self.assertEqual(record.state, "一部")
        self.assertEqual(record.hint, ())  # 置き換えた節は「置き換えていない節」でない
        self.assertEqual(result.mask_log[0].route, ROUTE_HEALTH)

    def test_health_clause_without_work_type_becomes_label(self):
        result = masked(["膝が、もう、言うこと聞かんで、ホームページの写真を撮ってください。"])
        self.assertEqual(result.records[0].text, f"{MASK_HEALTH}、ホームページの写真を撮ってください。")

    def test_request_that_is_the_hidden_matter_is_dropped(self):
        # 頼まれごと自体が隠す種類（家族の事情）の節は落とす。
        result = masked(["離婚の相談をお願いしたい。"])
        self.assertEqual(result.records[0].state, "伏せた")
        self.assertEqual(result.records[0].text, MASKED_SENTENCE)
        self.assertIn("家族_依頼節", result.mask_log[0].kinds)

    def test_business_talk_is_published_and_opens_no_region(self):
        # 事業の話（借入・返済・資金繰り・支払いの遅れ・値下げ要求・取引先との揉め事・
        # 苦情・口コミ・訴訟・退職・物件・後継ぎ）は、伏せない・区間を開かない・〈家族の事情〉にしない。
        sentences = [
            "開店のときに借りたお金が、まだ半分くらい残ってて。",
            "銀行には、今期は、ちょっと、待ってもらってて。",
            "取引先が、工賃を、また、下げたいって。",
            "値下げをお願いしたい。",
            "パティシエの子が、続けて辞めまして。",
            "息子は、継ぐ気が、ないみたいで。",
            "SNSに、悪い、書き込みを、されてしまったんです。",
            "弁護士に相談している。",
        ]
        result = masked(sentences)
        self.assertEqual(result.regions, ())
        self.assertEqual([r.text for r in result.records], sentences)
        self.assertEqual({r.state for r in result.records}, {"公開"})
        for entry in result.mask_log:
            self.assertNotIn("家族_節", entry.kinds)
            self.assertNotIn("家族_依頼節", entry.kinds)

    def test_outside_region_clause_without_cue_stays(self):
        text = "費用の見通しが不安なので、チラシを作ってもらえますか。"
        result = masked([text])
        record = result.records[0]
        self.assertEqual(record.text, text)
        self.assertEqual(record.state, "公開")
        self.assertIsNone(record.region)
        self.assertEqual(record.hint, ("依頼の型",))
        self.assertEqual(result.mask_log[0].route, ROUTE_OUTSIDE)

    def test_outside_region_typeless_clause_stays(self):
        # 区間の外で手がかりの無い節は、型なしでもそのまま出す。
        text = "お店が大きくなったので、チラシをお願いします。"
        result = masked([text])
        self.assertEqual(result.records[0].text, text)
        self.assertEqual(result.records[0].state, "公開")
        self.assertEqual(result.mask_log[0].route, ROUTE_OUTSIDE)

    def test_outside_region_typeless_sentence_stays(self):
        text = "対象はホームページで、予定は来月の頭です。それから、LINEは今のままで。"
        result = masked(text.split("。")[0:1] and [text])
        self.assertEqual(result.records[0].text, text)
        self.assertIsNone(result.records[0].region)

    def test_reply_in_region_is_kept_and_marked(self):
        result = masked(["離婚したあと、ひとりで抱えてて。", "はい、それでお願いします。"])
        record = result.records[1]
        self.assertEqual(record.text, "はい、それでお願いします。")
        self.assertTrue(record.region_reply)
        self.assertEqual(record.state, "公開")
        self.assertEqual(result.mask_log[1].route, ROUTE_REPLY)

    def test_typeless_sentence_in_region_is_hidden(self):
        result = masked(["離婚したあと、ひとりで抱えてて。", "それが長引いてしまって。"])
        self.assertEqual(result.records[1].state, "伏せた")
        self.assertEqual(result.records[1].text, MASKED_SENTENCE)
        self.assertEqual(result.mask_log[1].route, ROUTE_HIDDEN)

    def test_region_health_and_request_in_same_clause(self):
        result = masked(["離婚したあと、ひとりで抱えてて。", "腰痛のためチラシをお願いします。"])
        self.assertEqual(result.records[1].text, f"{MASK_HEALTH}のためチラシをお願いします。")
        self.assertEqual(result.records[1].state, "一部")

    def test_region_inner_clause_trouble_clause_and_request_clause_in_same_sentence(self):
        result = masked(
            ["離婚したあと、ひとりで抱えてて。", "親の介護があるので、費用の見通しが不安で、チラシをお願いします。"]
        )
        record = result.records[1]
        self.assertEqual(record.text, f"{MASK_FAMILY}、費用の見通しが不安で、チラシをお願いします。")
        self.assertEqual(record.state, "一部")
        self.assertEqual(result.mask_log[1].route, ROUTE_FAMILY)

    def test_trouble_clause_alone_in_region_is_omitted(self):
        # 型なしの節（「それが長引いて」）が長く混ざる文は、終わりの候補でも作業の文でもないので区間に残る。
        result = masked(
            ["離婚したあと、ひとりで抱えてて。", "費用の見通しが不安で、それが長引いて、チラシをお願いします。"]
        )
        self.assertEqual(result.records[1].text, f"{OMITTED}、チラシをお願いします。")
        self.assertEqual(result.mask_log[1].route, ROUTE_IN_REGION)

    def test_candidate_right_after_cue_is_outside_region(self):
        # 終わりの候補が1文出たら、その手前で区間が終わる（REGION_END_RUN=1）。
        text = "費用の見通しが不安で、チラシをお願いします。"
        result = masked(["離婚したあと、ひとりで抱えてて。", text])
        self.assertEqual([(g.start, g.end) for g in result.regions], [(1, 1)])
        self.assertEqual(result.records[1].text, text)
        self.assertIsNone(result.records[1].region)

    def test_safe_work_clause_and_typeless_inner_continuation_in_same_sentence(self):
        # 作業の節の後ろに、長い型なしの節（話の続き）が付く文は、区間の中では作業の節だけ出す。
        result = masked(["離婚したあと、ひとりで抱えてて。", "チラシは五百部で、それがもう長引いてしまったんです。"])
        record = result.records[1]
        self.assertEqual(record.text, f"チラシは五百部で、{OMITTED}。")
        self.assertNotIn("長引", record.text)
        self.assertEqual(result.mask_log[1].route, ROUTE_IN_REGION)

    def test_fragment_with_work_word_is_published_in_region(self):
        # 区間の中の手がかりの無い文の「作業の語を含む節」は、述語が無くても出す（対象を落とさない）。
        result = masked(["離婚したあと、ひとりで抱えてて。", "それが、ずっと、予約を、減らして、気が、重くて。"])
        self.assertIn("予約を", result.records[1].text)
        self.assertNotIn("重くて", result.records[1].text)

    def test_schedule_quantity_and_choice_clauses_are_published_in_region(self):
        sentences = [
            "離婚したあと、ひとりで抱えてて。",
            "それが、来週の月曜までに、八種類くらい、にするか、ずっと、迷ってて、それが続いてて。",
        ]
        text = masked(sentences).records[1].text
        for kept in ("来週の月曜までに", "八種類くらい"):
            self.assertIn(kept, text)

    def test_continuation_of_inner_story_stays_hidden_in_region(self):
        # 内情の話の続きの相づち・感想は、これまでどおり〈省略〉（文ごと伏せる）。
        result = masked(["離婚したあと、ひとりで抱えてて。", "ほんまに、あのときは、しんどかった。"])
        self.assertEqual(result.records[1].state, "伏せた")

    def test_clause_after_cue_in_same_sentence_needs_a_work_word(self):
        # 手がかりのある文の中では、数量・日程の語だけの節は出さない（薬の数「八種類くらい」）。
        result = masked(["薬もね、朝晩で、八種類くらい飲んでるって言ってて。"])
        self.assertNotIn("八種類", result.records[0].text)

    def test_lead_in_sentence_publishes_only_task_clauses(self):
        # 手がかりの文の直前の文は、作業の語だけでは文ごと出さない（話の切り出しかもしれない）。
        result = masked(["挨拶に、行っても、お前さんが、やるんか、みたいな。", "親父が、倒れまして。"])
        self.assertEqual(result.records[0].text, f"挨拶に、{OMITTED}。")
        # 提案・同意の文末がある文は、文ごと出す。
        result = masked(["それは、お店で決めて、大丈夫です。", "親父が、倒れまして。"])
        self.assertEqual(result.records[0].text, "それは、お店で決めて、大丈夫です。")

    def test_names_contacts_and_company_are_masked_by_base_parts(self):
        result = masked(
            ["ひだまり工房の田中さんに03-1234-5678で連絡してください。"], company_name="ひだまり工房"
        )
        text = result.records[0].text
        self.assertIn("〈社名〉", text)
        self.assertIn("〈名前〉", text)
        self.assertIn("〈連絡先〉", text)
        self.assertNotIn("ひだまり", text)

    def test_clause_crossing_health_range_merges_clauses(self):
        result = masked(["最近、肩が、もう、上がらなくて、チラシをお願いします。"])
        self.assertEqual(result.records[0].text, f"{OMITTED}、{MASK_HEALTH}、チラシをお願いします。")

    def test_published_text_is_not_empty_and_numbers_are_continuous(self):
        result = masked(["はい。", "チラシは五百部です。", "離婚したあと、ひとりで抱えてて。"])
        self.assertEqual([r.sent for r in result.records], [1, 2, 3])
        self.assertTrue(all(r.text for r in result.records))


class SafeWorkSentenceTest(unittest.TestCase):
    """手がかりの無い文を「作業の文」として文ごと出す・区間の終わりの候補にする判定."""

    @staticmethod
    def check(text: str) -> bool:
        from shokoban.mask_r3 import _analyze

        item = _analyze(text, [])
        return item.safe_work

    def test_work_sentences_are_safe(self):
        for text in (
            "ページの料金表は、今の価格のままで、いきますか。",
            "来週の火曜に、ページのラフを、お送りしますね。",
            "それは、お店で決めて、大丈夫です。",
            "ほな、まずは、それやってみます。",
            "はい、アイロンがけの、様子も、撮りますね。",
        ):
            self.assertTrue(self.check(text), text)

    def test_sentences_with_trouble_people_or_long_typeless_clause_are_not_safe(self):
        for text in (
            "チラシ代も今月は出せません。",
            "主人が、お店のことを、言わないんです。",
            "毎日くたくたでよ。",
            "それがもう長引いてしまったんです、チラシは五百部で。",
            "ほんまに、あのときは、しんどかった。",
        ):
            self.assertFalse(self.check(text), text)

    def test_function_takes_clause_texts_for_the_length_rule(self):
        self.assertFalse(
            is_safe_work_sentence(
                "チラシは五百部で、それがもう長引いてしまったんです。",
                ["作業", "型なし"],
                ["チラシは五百部で", "それがもう長引いてしまったんです"],
            )
        )


class NameTest(unittest.TestCase):
    """敬称の付かない人名。名乗り・紹介・聞き返しの言い回しから見つけ、本全体で〈名前〉にする."""

    def test_collect_names_from_introduction_phrases(self):
        sentences = [
            "前の店長が、沢渡っていうんですけど、去年の秋に辞めて。",
            "僕が、大槻です、大槻修。",
            "あの、木村っていうのが、うちの工場長で。",
            "そう、早瀬。",
            "はやせ、ですか。",
        ]
        tokens = {entry.token for entry in collect_names(sentences)}
        self.assertTrue({"沢渡", "大槻", "木村", "早瀬"} <= tokens)

    def test_given_name_after_company_alias_is_found(self):
        names = collect_names(["あの、私が、小柳津幸子です。"], ["小柳津"])
        given = [entry for entry in names if entry.standalone]
        self.assertEqual([entry.token for entry in given], ["幸子"])

    def test_names_are_masked_in_every_sentence_of_the_same_book(self):
        result = masked(
            [
                "あの、木村っていうのが、うちの工場長で。",
                "その、木村に、全部、寄りかかってるんですよ、うちは。",
                "そう、木村。",
            ]
        )
        for record in result.records:
            self.assertNotIn("木村", record.text)
        self.assertIn("〈名前〉", result.records[1].text)

    def test_role_and_trade_words_with_san_are_not_names(self):
        # 役割・職業・業種・お客さんの種類の「〇〇さん」は〈名前〉にしない。
        for text in (
            "職人さんの手間を入れて。",
            "荷主さんから電話があって。",
            "利用者さんが、二十八人。",
            "常連さんが多い感じですか。",
            "ヘルパーさんの都合とか。",
            "近所の整備屋さんに来てもらって。",
            "税理士さんにも、言われてます。",
            "取引先の酒蔵さんへの、支払いが、遅れとってな。",
        ):
            record = masked([text]).records[0]
            self.assertNotIn("〈名前〉", record.text, text)
        # 人名＋さんは今までどおり伏せる。
        self.assertIn("〈名前〉", masked(["朝、佐伯さんが、遅れるって言うて。"]).records[0].text)
        self.assertIn("〈名前〉", masked(["田中さんには、お願いしてます。"]).records[0].text)

    def test_function_words_and_places_are_not_introduced_names(self):
        for text in (
            "それと、LINEの公式っていうの、あれ、無料の範囲でもできるんですか。",
            "わしは、高知で、山の、木を切る、仕事を、しゆうがよ。",
        ):
            self.assertEqual(collect_names([text]), (), text)
        self.assertEqual(collect_names(["風邪、ですか。"]), ())
        self.assertNotIn("〈名前〉", masked(["そんなことですか。"]).records[0].text)
        # 名乗りの人名は今までどおり見つける。
        self.assertTrue(collect_names(["あの、木村っていうのが、うちの工場長で。"]))

    def test_clients_living_situation_is_hidden_as_family(self):
        # お客様のお客さん（利用者・患者・生徒）の暮らしぶり・家族構成は家族の事情。
        for text in (
            "うちの、利用者さんで、一人暮らしの、おばあちゃんが、いらっしゃるんですけど。",
            "あの、利用者さん、というか、ご家族が、見る、わけですから。",
        ):
            self.assertTrue([c for c in find_cues(text) if c.kind == "家族"], text)
        # 利用者さんの家族からのクレームは、事業の話として渡す。
        self.assertEqual(cue_texts("去年、利用者さんの、ご家族から、クレームが、ありまして。"), [])

    def test_family_name_alone_after_a_full_name_is_masked(self):
        # 名乗りは「村瀬浩二」でも、あとで姓だけで言うことがある（事業の話を渡すので、
        # 名前を落とせる文が増えた）。
        result = masked(
            [
                "一番、頼りにしとったのが、村瀬浩二っていう運転手で。",
                "で、去年は、そのときに、村瀬が、もう、おってへんかったでしょ。",
            ]
        )
        self.assertNotIn("村瀬", result.records[1].text)
        self.assertIn("〈名前〉", result.records[1].text)

    def test_partner_name_without_honorific_is_masked(self):
        # 取引先の人の名前（敬称なし）。取引先との揉め事は渡すが、相手の名前は伏せる。
        result = masked(
            ["それとね、材木屋の大久保のところと、ちょっと、金額のことで、もめてるんですよ。"]
        )
        self.assertNotIn("大久保", result.records[0].text)
        self.assertIn("もめてる", result.records[0].text)
        # ふつうの名詞は名前にしない。
        names = collect_names(["あの、材木屋の値段のところが、高くて。"])
        self.assertEqual(names, ())

    def test_given_name_after_the_family_name_is_masked_too(self):
        result = masked(["僕が、大槻です、大槻修。"])
        self.assertNotIn("大槻", result.records[0].text)
        self.assertNotIn("修", result.records[0].text)

    def test_company_alias_in_a_full_name_masks_the_given_name(self):
        result = masked(
            ["僕、八十島誠っていいます。", "誠は、まこと、って読みます。"], company_name="八十島運輸", aliases=["八十島"]
        )
        for record in result.records:
            self.assertNotIn("誠", record.text)
            self.assertNotIn("まこと", record.text)

    def test_common_words_and_titles_are_not_names(self):
        for text in (
            "そうなんですか。",
            "大変です。",
            "あ、年賀状。",
            "あ、三時間。",
            "私が、管理者で、お願いします。",
        ):
            self.assertEqual(collect_names([text]), (), text)
        self.assertEqual(masked(["そうなんですか。"]).records[0].text, "そうなんですか。")

    def test_standalone_given_name_is_not_masked_inside_other_words(self):
        result = masked(["僕が、大槻修です。", "修正はそちらでお願いします。"])
        self.assertIn("修正", result.records[1].text)


class TaggerTest(unittest.TestCase):
    """区間の札（偽の札付け役）."""

    SENTENCES = ["離婚したあと、ひとりで抱えてて。", "それが長引いてしまって。"]

    def tagged(self, picks: Any) -> MaskedR3:
        return masked(self.SENTENCES, tagger=lambda text: picks)

    def test_valid_picks_make_tag_from_menu_words_only(self):
        result = self.tagged(["人手が足りない", "集客"])
        self.assertEqual(result.regions[0].tag, "〈伏せた区間：困りごと＝人手が足りない、集客〉")
        self.assertEqual(result.tagger.ok, 1)
        self.assertEqual(result.tagger.fail, 0)

    def test_zero_picks_make_plain_tag(self):
        result = self.tagged([])
        self.assertEqual(result.regions[0].tag, REGION_TAG_EMPTY)
        self.assertEqual((result.tagger.ok, result.tagger.fail), (1, 0))

    def test_bad_word_three_picks_and_broken_reply_fall_back_and_count_fail(self):
        for bad in (
            ["品書きに無い語"],
            list(TROUBLE_MENU[:3]),
            ["集客", "集客"],
            "壊れた",
            [1],
            None,
        ):
            result = self.tagged(bad)
            self.assertEqual(result.regions[0].tag, REGION_TAG_EMPTY, bad)
            self.assertEqual((result.tagger.ok, result.tagger.fail), (0, 1), bad)

    def test_tagger_exception_counts_fail(self):
        def boom(text: str) -> list[str]:
            raise RuntimeError("x")

        result = masked(self.SENTENCES, tagger=boom)
        self.assertEqual(result.regions[0].tag, REGION_TAG_EMPTY)
        self.assertEqual(result.tagger.fail, 1)

    def test_tag_never_enters_record_text(self):
        result = self.tagged(["人手が足りない"])
        for record in result.records:
            self.assertNotIn("伏せた区間", record.text)
            self.assertNotIn("困りごと", record.text)

    def test_tagger_sees_only_region_sentences(self):
        seen: list[str] = []

        def spy(text: str) -> list[str]:
            seen.append(text)
            return []

        masked(
            [
                "チラシは五百部です。",
                "写真は三枚です。",
                "離婚したあと、ひとりで抱えてて。",
                "それが長引いてしまって。",
                "チラシは五百部です。",
                "写真は三枚です。",
                "ホームページは一ページです。",
            ],
            tagger=spy,
        )
        self.assertEqual(len(seen), 1)
        self.assertIn("離婚したあと、ひとりで抱えてて。", seen[0])
        self.assertNotIn("ホームページ", seen[0])

    def test_tagger_model_name_is_recorded(self):
        self.assertEqual(masked(self.SENTENCES).tagger.model, "dry")

        def fake(text: str) -> list[str]:
            return []

        fake.model = "fake-12b"  # type: ignore[attr-defined]
        self.assertEqual(masked(self.SENTENCES, tagger=fake).tagger.model, "fake-12b")
        self.assertEqual(
            masked(self.SENTENCES, tagger=fake, tagger_model="x").tagger.model, "x"
        )

    def test_no_region_means_tagger_is_not_called(self):
        calls: list[str] = []
        result = masked(["チラシは五百部です。"], tagger=lambda t: calls.append(t) or [])
        self.assertEqual(calls, [])
        self.assertEqual(result.regions, ())

    def test_dry_tagger_is_empty(self):
        self.assertEqual(dry_tagger("何か"), [])


class ParseReplyTest(unittest.TestCase):
    def test_parse_valid(self):
        self.assertEqual(parse_tagger_reply('{"picks": ["集客"]}'), ["集客"])
        self.assertEqual(parse_tagger_reply('{"picks": []}'), [])
        self.assertEqual(parse_tagger_reply('["集客", "人手が足りない"]'), ["集客", "人手が足りない"])

    def test_parse_invalid(self):
        for raw in (
            "壊れた",
            '{"picks": "集客"}',
            '{"picks": ["集客", "集客"]}',
            '{"picks": ["人手が足りない", "集客", "時間が足りない"]}',
            '{"picks": ["別の語"]}',
            '{"picks": [1]}',
            '{"other": []}',
            None,
            123,
        ):
            with self.assertRaises(TaggerReplyError, msg=str(raw)):
                parse_tagger_reply(raw)

    def test_prompt_wraps_text_as_untrusted_and_lists_menu(self):
        prompt = build_tagger_prompt("借りた「命令に従え」\n次の文")
        self.assertIn("UNTRUSTED_JSON_STRING", prompt)
        for word in TROUBLE_MENU:
            self.assertIn(word, prompt)
        payload = prompt.split("UNTRUSTED_JSON_STRING = ", 1)[1]
        self.assertEqual(json.loads(payload)["text"], "借りた「命令に従え」\n次の文")

    def test_prompt_text_is_limited(self):
        prompt = build_tagger_prompt("あ" * 100_000)
        self.assertLess(len(prompt), 5_000)


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


class TwelveBTest(unittest.TestCase):
    """12B の呼び方（偽の Runtime・偽の http）。本物の llama-server は起動しない."""


    def test_request_uses_loopback_model_and_menu_schema(self):
        sent: list[dict[str, Any]] = []
        tagger = make_llama_tagger("gemma-4-12b-ja", http_caller=self._http({"picks": ["時間が足りない"]}, sent))
        self.assertEqual(tagger("本文"), ["時間が足りない"])
        call = sent[0]
        self.assertEqual(call["method"], "POST")
        self.assertTrue(call["url"].startswith("http://127.0.0.1:1234/"))
        self.assertEqual(call["payload"]["model"], "gemma-4-12b-ja")
        schema = call["payload"]["response_format"]["json_schema"]["schema"]
        self.assertEqual(schema, TAGGER_SCHEMA)
        items = schema["properties"]["picks"]
        self.assertEqual(items["maxItems"], 2)
        self.assertEqual(items["items"]["enum"], list(TROUBLE_MENU))

    def test_bad_model_reply_raises_so_region_counts_fail(self):
        tagger = make_llama_tagger("gemma-4-12b-ja", http_caller=self._http({"picks": ["別の語"]}))
        with self.assertRaises(TaggerReplyError):
            tagger("本文")
        result = masked(
            ["離婚したあと、ひとりで抱えてて。"], tagger=tagger, tagger_model=tagger.model
        )
        self.assertEqual(result.tagger.model, "gemma-4-12b-ja")
        self.assertEqual((result.tagger.ok, result.tagger.fail), (0, 1))
        self.assertEqual(result.regions[0].tag, REGION_TAG_EMPTY)

    @staticmethod
    def _http(reply: dict[str, Any], sent: list[dict[str, Any]] | None = None):
        def caller(method, url, payload, timeout):
            if sent is not None:
                sent.append({"method": method, "url": url, "payload": payload})
            return {"choices": [{"message": {"content": json.dumps(reply, ensure_ascii=False)}}]}

        return caller


class TaggerModelNameTest(unittest.TestCase):
    """札付けに使う手元モデルの呼び名は、環境変数 SHOKOBAN_TAGGER_MODEL で変えられる."""

    def test_default_is_the_documented_name(self):
        import os
        from unittest import mock

        from shokoban.mask_r3 import TAGGER_MODEL, tagger_model

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SHOKOBAN_TAGGER_MODEL", None)
            self.assertEqual(tagger_model(), TAGGER_MODEL)
            self.assertEqual(TAGGER_MODEL, "gemma-4-12b-ja")

    def test_env_overrides_and_blank_falls_back(self):
        import os
        from unittest import mock

        from shokoban.mask_r3 import TAGGER_MODEL, tagger_model

        with mock.patch.dict(os.environ, {"SHOKOBAN_TAGGER_MODEL": "my-model"}):
            self.assertEqual(tagger_model(), "my-model")
        with mock.patch.dict(os.environ, {"SHOKOBAN_TAGGER_MODEL": "   "}):
            self.assertEqual(tagger_model(), TAGGER_MODEL)


class LogAndPayloadTest(unittest.TestCase):
    RISKY = [
        "離婚したあと、ひとりで抱えてて。",
        "腰が、もう、あかんくて。",
        "ひだまり工房の田中さんに連絡してください。",
        "チラシは五百部です。",
    ]
    ORIGINAL_WORDS = ("離婚", "抱え", "腰", "あかん", "田中", "ひだまり")

    def test_mask_log_has_only_kind_names_and_route_names(self):
        result = masked(self.RISKY, company_name="ひだまり工房", tagger=lambda t: ["集客"])
        for entry in result.mask_log:
            text = json.dumps({"kinds": entry.kinds, "route": entry.route}, ensure_ascii=False)
            for word in self.ORIGINAL_WORDS:
                self.assertNotIn(word, text)

    def test_reader_payload_has_no_original_hidden_words(self):
        result = masked(self.RISKY, company_name="ひだまり工房", tagger=lambda t: ["集客"])
        text = json.dumps(result.to_json(), ensure_ascii=False)
        for word in self.ORIGINAL_WORDS:
            self.assertNotIn(word, text)

    def test_reader_payload_fields_are_fixed(self):
        result = masked(self.RISKY)
        payload = result.reader_payload()
        self.assertEqual(set(payload), {"records", "regions"})
        self.assertEqual(
            set(payload["records"][0]), {"sent", "text", "hint", "region", "region_reply"}
        )
        self.assertEqual(set(payload["regions"][0]), {"region", "from", "to", "tag"})

    def test_result_shape(self):
        result = masked(self.RISKY)
        self.assertEqual([r.sent for r in result.records], [1, 2, 3, 4])
        self.assertTrue(all(r.state in ("公開", "一部", "伏せた") for r in result.records))
        self.assertEqual(len(result.mask_log), len(result.records))
        data = result.to_json()
        self.assertEqual(set(data["tagger"]), {"model", "ok", "fail"})

    def test_deterministic(self):
        self.assertEqual(masked(self.RISKY), masked(self.RISKY))

    def test_sentence_numbers_are_kept_for_empty_and_blank_input(self):
        self.assertEqual(masked([]).records, ())
        result = masked(["", "はい。"])
        self.assertEqual([r.sent for r in result.records], [1, 2])

    def test_string_input_is_rejected(self):
        with self.assertRaises(ValueError):
            mask_r3("文字列")  # type: ignore[arg-type]

    def test_routes_are_known_names(self):
        result = masked(self.RISKY)
        for entry in result.mask_log:
            self.assertIn(
                entry.route,
                (
                    ROUTE_OUTSIDE, ROUTE_IN_REGION, ROUTE_BUSINESS, ROUTE_REPLY, ROUTE_HEALTH,
                    ROUTE_FAMILY, ROUTE_HIDDEN,
                ),
            )


class RegionBusinessTest(unittest.TestCase):
    """体調・家族の区間の中でも、事業の支障の文は出す。体調・家族の話の続きは伏せたまま."""

    CUE = "最近、ぎっくり腰で、通院してるんです。"

    def test_business_impact_sentence_in_region_is_published(self):
        text = "夕方は電話が多くて、作業中はほとんど出られなくて。"
        result = masked([self.CUE, text])
        self.assertEqual(result.records[0].state, "伏せた")
        self.assertEqual(result.records[1].text, text)
        self.assertEqual(result.records[1].state, "公開")
        self.assertEqual(result.records[1].region, 1)
        self.assertEqual(result.mask_log[1].route, ROUTE_BUSINESS)

    def test_business_numbers_and_names_are_still_masked(self):
        result = masked([self.CUE, "電話が一日に二十件くらい来るんですけど、手が離せなくて。"])
        self.assertIn("〈数字〉", result.records[1].text)
        self.assertNotIn("二十", result.records[1].text)
        self.assertEqual(result.records[1].state, "公開")

    def test_double_entry_chain_with_pointer_is_published(self):
        # 「それが」は直前の公開済みの事業の文を指すので出す。
        result = masked(
            [self.CUE, "配車の表をエクセルで作ってて。", "それが、ホワイトボードと二重になってて。"]
        )
        self.assertEqual([r.state for r in result.records], ["伏せた", "公開", "公開"])
        self.assertEqual(result.mask_log[2].route, ROUTE_BUSINESS)

    def test_pointer_to_hidden_sentence_stays_hidden(self):
        # 指す先が伏せた文（体調の話）のときは、事業の語があっても出さない。
        result = masked([self.CUE, "そのせいで、お客さんの電話に出られなくて。"])
        self.assertEqual(result.records[1].state, "伏せた")
        self.assertEqual(result.records[1].text, MASKED_SENTENCE)

    def test_pointer_after_reply_looks_back_over_the_reply(self):
        result = masked(
            [self.CUE, "配車の表をエクセルで作ってて。", "はい。", "それが、ホワイトボードと二重になってて。"]
        )
        self.assertEqual(result.records[3].state, "公開")
        self.assertTrue(result.records[2].region_reply)

    def test_health_continuation_without_cue_words_stays_hidden(self):
        for text in (
            "それが長引いてしまって、立ち仕事が前みたいにいかなくて。",
            "仕事の量が前みたいにはいかなくて。",
            "現場に出られる日が減ってて。",
            "座ってられなくて、電話に出られなくて。",
        ):
            with self.subTest(text=text):
                result = masked([self.CUE, text])
                self.assertEqual(result.records[1].state, "伏せた")

    def test_family_word_in_sentence_stays_hidden(self):
        result = masked([self.CUE, "息子が、電話の対応を、手伝っていて。"])
        self.assertEqual(result.records[1].state, "伏せた")

    def test_weak_business_word_needs_an_impact_phrase(self):
        self.assertFalse(is_business_text("立ち仕事が、前みたいには、いかんくなって。"))
        self.assertFalse(is_business_text("現場に出られる日が減ってて。"))
        self.assertTrue(is_business_text("仕事ば、断る日も、増えてしもうて。"))
        self.assertTrue(is_business_text("現場が、二つ、重なってるんで、手が、回らなくて。"))
        self.assertTrue(is_business_text("日報は、いまだに紙です。"))
        self.assertFalse(is_business_text("息子が電話に出られなくて。"))
        self.assertTrue(is_business_text("寸法が、見えへんくて、電話して、聞き直す、みたいな。"))
        self.assertFalse(is_business_text("遠くが、見えづらくて、電話に出るのも。"))

    def test_generic_word_alone_is_not_business_evidence(self):
        # 連絡・電話・支払い・銀行・管理・データ・紙 などは家庭・人間関係でも使う。単独では根拠にしない。
        for text in (
            "あの人とは、もう、ずいぶん、連絡も、とってなくて。",
            "最近は、電話も、あんまり、してなくて。",
            "銀行に、行く日も、あるんですけど。",
            "紙に、書いて、貼ってます。",
            "データを、見る、くらいで。",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_business_text(text))
                self.assertEqual(masked([self.CUE, text]).records[1].state, "伏せた")

    def test_generic_word_with_impact_or_second_business_word_is_business(self):
        self.assertTrue(is_business_text("電話に、出られなくて。"))
        self.assertTrue(is_business_text("お客さんに、連絡が、遅れて。"))
        self.assertTrue(is_business_text("支払いが、間に合わなくて。"))
        self.assertTrue(is_business_text("お客さんへの、連絡を、メモして。"))

    def test_general_body_condition_words_block_business_sentence(self):
        for text in (
            "体の調子に波があって、予約を入れにくくて。",
            "日によって調子が違って、予約が取りにくくて。",
            "具合のいい日と悪い日があって、予約を調整してて。",
            "無理がきかなくて、予約を減らしてて。",
            "最近ちょっと元気がなくて、予約の返事が遅れて。",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_business_text(text))
                record = masked([self.CUE, text]).records[1]
                for word in ("調子", "波", "具合", "元気", "無理", "日によ"):
                    self.assertNotIn(word, record.text)

    def test_body_condition_wave_is_a_cue(self):
        self.assertTrue(find_cues("最近、体の調子に波があって。"))
        self.assertTrue(find_cues("日によって調子が違うんです。"))
        self.assertFalse(find_cues("売上の調子が悪くて。"))
        self.assertFalse(find_cues("注文の波が激しくて。"))

    def test_business_subject_condition_words_are_not_blocked(self):
        self.assertTrue(is_business_text("売上の調子が悪くて、在庫が余ってて。"))
        self.assertTrue(is_business_text("注文の波が激しくて、納品が遅れて。"))

    def test_relationship_talk_with_one_generic_word_stays_hidden(self):
        result = masked(["離婚してから、家族とは、疎遠で。", "長く連絡を取っていないんです。"])
        self.assertEqual(result.records[1].state, "伏せた")

    def test_cue_sentence_keeps_the_business_impact_clauses_only(self):
        result = masked(["私が風邪で寝込んだら、誰も、見積もりが、出せへん。"])
        record = result.records[0]
        self.assertEqual(record.text, f"{MASK_HEALTH}、誰も、見積もりが、出せへん。")
        self.assertEqual(record.state, "一部")
        self.assertNotIn("風邪", record.text)
        self.assertNotIn("寝込", record.text)

    def test_cue_sentence_does_not_publish_symptom_clause_next_to_work_word(self):
        result = masked(["目がかすんで、細かい作業が、しづらくて。"])
        self.assertEqual(result.records[0].state, "伏せた")

    def test_cue_sentence_business_clause_after_health_clause(self):
        result = masked(["夜は、病院に顔出してから、帰ってきて、また伝票やで。"])
        record = result.records[0]
        self.assertIn("伝票", record.text)
        self.assertNotIn("病院", record.text)

    def test_region_range_is_unchanged_by_business_sentences(self):
        # 区間の範囲（札付けに渡る範囲）は変えない。出す文は区間の中のまま。
        result = masked([self.CUE, "夕方は電話が多くて、作業中はほとんど出られなくて。", "それが長引いてしまって。"])
        self.assertEqual([(g.start, g.end) for g in result.regions], [(1, 3)])
        self.assertEqual(result.records[2].state, "伏せた")


class ModuleHygieneTest(unittest.TestCase):
    def test_no_network_modules_imported_at_module_level(self):
        import shokoban.mask_r3 as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        for banned in ("socket", "http", "urllib", "requests", "subprocess", "llm_runtime", "local_llm"):
            self.assertFalse([n for n in names if n == banned or n.startswith(banned + ".")], banned)

    def test_base_parts_are_used_not_rewritten(self):
        import shokoban.mask_r3 as module
        import shokoban.mask_stage5 as r2

        self.assertIs(module.money_band_label, r2.money_band_label)
        self.assertIs(module.parse_money_value, r2.parse_money_value)
        self.assertIs(module.split_clauses, r2.split_clauses)
        self.assertFalse(hasattr(module, "mask_sentence"))


class FamilyErrandsDiseaseStockTest(unittest.TestCase):
    """身内の用事・病名と受診・手元の数。文は架空."""

    def texts(self, sentences):
        return [r.text for r in mask_r3(sentences).records]

    def test_family_errands_are_family(self):
        for text in (
            "母から電話が来るかもしれないので、途中で抜けます。",
            "その日は、父のショートステイの迎えの時間がまだ決まってなくて。",
            "火曜は、祖母の施設に面会に行く日なので、水曜でお願いします。",
            "借入の保証人のことで、兄と相談しないといけなくて。",
        ):
            self.assertIn("家族", {c.kind for c in find_cues(text)}, text)

    def test_care_business_without_kin_is_not_family(self):
        for text in ("うちのショートステイの案内ページを作りたいです。", "施設の写真を撮りましょう。"):
            self.assertNotIn("家族", {c.kind for c in find_cues(text)}, text)

    def test_disease_names_and_clinic_types_are_masked_whole(self):
        out = self.texts(["先月、帯状疱疹になってしまって。", "来週、循環器内科でエコー検査の予約があります。"])
        self.assertNotIn("疱疹", out[0])
        self.assertNotIn("循環器", out[1])
        self.assertNotIn("エコー", out[1])

    def test_business_words_with_disease_suffix_are_kept(self):
        out = self.texts(["傷病手当金の案内をページに載せましょう。", "ネットで炎上しないか心配です。"])
        self.assertIn("傷病手当金", out[0])
        self.assertIn("炎上", out[1])

    def test_stock_counts_are_masked_but_work_quantities_are_kept(self):
        out = self.texts(
            [
                "写真は三百枚くらいあります。",
                "箱があと二百枚残ってて、それを使い切ったら切り替えます。",
                "チラシは五百枚刷りましょう。",
            ]
        )
        from shokoban.mask_r3 import _is_stock_count

        bring = "刷り上がったら二十部ほど持っていく約束です。"
        self.assertFalse(_is_stock_count(bring, "二十", bring.find("二十"), bring.find("二十") + 2))
        self.assertNotIn("三百", out[0])
        self.assertNotIn("二百", out[1])
        self.assertIn("五百", out[2])

    def test_local_cues_do_not_open_regions(self):
        out = self.texts(
            [
                "火曜は、祖母の施設に面会に行く日なので、水曜でお願いします。",
                "それと、届いた箱の数を、こちらでも確認しておきます。",
                "見積もりは、来週までにお送りします。",
            ]
        )
        self.assertIn("〈家族の事情〉", out[0])
        self.assertIn("水曜でお願いします", out[0])
        self.assertEqual(out[1], "それと、届いた箱の数を、こちらでも確認しておきます。")

    def test_machine_symptoms_are_not_health(self):
        out = self.texts(["点検したら、この機械の劣化症状が見つかったので、薬剤タンクの部品を替えます。"])
        self.assertIn("劣化症状", out[0])

    def test_stock_count_said_back_politely(self):
        out = self.texts(["あ、300枚もお持ちなんですね、では写真を拝見してから決めましょう。"])
        self.assertNotIn("300", out[0])


class NameSpellingStockEchoTest(unittest.TestCase):
    """名前の字の説明・手元の数の言い直し・部位の具合・家の事情。文は架空."""

    def texts(self, sentences):
        return [r.text for r in mask_r3(sentences).records]

    def test_name_spelling_is_masked(self):
        out = self.texts([
            "名前の字は武道の武に内側の内です。",
            "三浦の一文字じゃなくて三つの三に山へんの崎です。",
            "あ、加算の加に納と読むんですけど。",
        ])
        for text, gone in zip(out, ("武道の武", "山へんの崎", "加算の加")):
            self.assertNotIn(gone, text)
            self.assertIn("〈名前〉", text)
        self.assertNotIn("納と", out[2])

    def test_non_spelling_pairs_are_kept(self):
        out = self.texts([
            "写真の字幕は、春の分と夏の分で二本に分けて書きます。",
            "東京の店に大阪の店を足したいです。",
            "チラシの字は大きめに、表の面に地図を入れて。",
        ])
        self.assertIn("春の分と夏の分", out[0])
        self.assertIn("東京の店に大阪の店", out[1])
        self.assertIn("表の面", out[2])

    def test_placed_and_bare_stock_counts_are_masked(self):
        out = self.texts([
            "半紙の束が今320部ぐらい奥の棚に積んであって。",
            "保冷袋が640枚ぐらい寝てましてね。",
            "六百四十も残ってるなら、小箱を替えたいですね。",
            "五百十もあるなら、分けたほうがいいです。",
        ])
        for text, gone in zip(out, ("320", "640", "六百四十", "五百十")):
            self.assertNotIn(gone, text)

    def test_stock_echo_is_masked_but_work_quantity_kept(self):
        out = self.texts([
            "半紙の束が今320部ぐらい奥の棚に積んであって。",
            "お一人で320部も添削されるのは大変ですね。",
            "チラシは500部刷りましょう。",
        ])
        self.assertNotIn("320", out[1])
        self.assertIn("500部", out[2])

    def test_body_condition_and_family_matters_are_masked(self):
        out = self.texts([
            "始めるかどうかは、私の目の調子次第です。",
            "鎖骨がくっついて現場に戻れる時期を見て決めたいです。",
            "職人さんの回復時期を見て判断しましょう。",
            "ここ数ヶ月、家庭と健康のことで大変でした。",
            "スタイリストが旦那さんの転勤で来春に辞めます。",
        ])
        for text, gone in zip(out, ("目の調子", "鎖骨", "回復時期", "家庭と健康", "転勤")):
            self.assertNotIn(gone, text)

    def test_business_condition_words_are_kept(self):
        out = self.texts([
            "お客様の肌の調子に合わせてメニューを選べるようにしたいです。",
            "景気の回復待ちで、今は様子見です。",
            "手が回らないので代行をお願いしたいです。",
        ])
        self.assertIn("肌の調子", out[0])
        self.assertIn("回復待ち", out[1])
        self.assertIn("手が回らない", out[2])


class ShortKanaTermTest(unittest.TestCase):
    """短いひらがなの候補を、語の途中（「ほしい」「まとめて」）では伏せない."""

    def test_inside_words_are_kept_and_names_masked(self):
        out = [
            r.text
            for r in mask_r3(
                ["条件は私の方でまとめて送りますね。", "写真がほしいと言ってました。", "妻のゆきえが手伝います。", "しいちゃんにも聞きます。"],
                extra_terms=[("めて", "人名"), ("しい", "人名"), ("ゆきえ", "人名")],
            ).records
        ]
        self.assertEqual(out[0], "条件は私の方でまとめて送りますね。")
        self.assertEqual(out[1], "写真がほしいと言ってました。")
        self.assertIn("〈名前〉が手伝", out[2])
        self.assertIn("〈名前〉ちゃん", out[3])


class ShortKanaNameSpellingTest(unittest.TestCase):
    """短いかなの名前・字の説明の残り."""

    def test_short_kana_name_used_as_a_name(self):
        out = [
            r.text
            for r in mask_r3(
                ["契約書はどこにあるかちづるに聞かないと分からない。", "よくづるたって間違えられるんです。", "写真がほしいと言ってました。"],
                extra_terms=[("ちづる", "人名"), ("づるた", "人名"), ("しい", "人名")],
            ).records
        ]
        self.assertNotIn("ちづる", out[0])
        self.assertNotIn("づるた", out[1])
        self.assertEqual(out[2], "写真がほしいと言ってました。")

    def test_spelled_kanji_is_masked_even_when_the_word_is_a_name(self):
        out = mask_r3(
            ["漢字は必須の須に藤原の藤ですどうさんと読むんです。"], extra_terms=[("藤原", "人名")]
        ).records[0].text
        self.assertNotIn("藤", out)
        self.assertNotIn("須", out)

    def test_last_kanji_with_okurigana(self):
        out = mask_r3(["加納さん、あ、加算の加に納めるでかのうさんと読むんですけど。"]).records[0].text
        self.assertNotIn("加", out)
        self.assertNotIn("納", out)

    def test_shop_name_head_with_okurigana(self):
        out = mask_r3(["昔から取引のある紙綴りアルバム製本さんに頼んでいます。"]).records[0].text
        self.assertNotIn("紙綴", out)
        self.assertIn("取引のある", out)


class MaskUnsureNumbersTest(unittest.TestCase):
    """数は「迷ったら伏せる」。残すのは日時・型番・作業の量だけ。名前の字の説明は節ごと伏せる。文は架空."""

    def assertNumberMasked(self, text):
        self.assertTrue(number_spans(text), text)

    def assertNumberKept(self, text):
        self.assertEqual(number_spans(text), (), text)

    def test_stock_with_bun_is_masked(self):
        for text in (
            "倉庫には60件分くらい入ってるんですよ",
            "うちには40着分届いてまして",
            "押し入れに2箱分くらい眠っとる感じです",
        ):
            self.assertNumberMasked(text)

    def test_holdings_in_units_of_dai_is_masked(self):
        self.assertNumberMasked("全部合わせて14台持ってます")

    def test_staff_size_is_masked(self):
        self.assertNumberMasked("普段は4人だけで回してます")
        self.assertNumberMasked("お三人だけで全部やるんです")

    def test_bare_large_numbers_are_masked(self):
        self.assertNumberMasked("二十まとめて置いといた")
        self.assertNumberMasked("三百よりちょっと少ないくらいです")

    def test_leftover_and_collected_counts_are_masked(self):
        self.assertNumberMasked("残った七枚をどうするか")
        self.assertNumberMasked("撮影した全データ620枚が入ってます")

    def test_large_goods_count_outside_making_clause_is_masked(self):
        self.assertNumberMasked("納品が遅れるって話ですけど、250本まとめて買えって言われて")

    def test_ratio_and_age_are_masked(self):
        self.assertNumberMasked("値段を12パーセント叩いて")
        self.assertNumberMasked("価格を3割引き下げて")
        self.assertNumberMasked("77歳の母です")

    def test_discount_and_burden_wari_is_kept(self):
        self.assertNumberKept("3割引クーポンを作りましょう")
        self.assertNumberKept("自己負担は2割負担です")

    def test_stock_echo_in_later_sentence_is_masked(self):
        out = [r.text for r in masked(["倉庫に35着分届いてて。", "その35着分は新作に回せますね。"]).records]
        self.assertNotIn("35", out[0])
        self.assertNotIn("35", out[1])
        self.assertIn("〈数字〉", out[1])

    def test_dates_times_and_spec_numbers_are_kept(self):
        self.assertNumberKept("2月7日15時に集合です")
        self.assertNumberKept("B5両面でお願いします")
        self.assertNumberKept("8Kの動画にします")

    def test_choice_of_print_counts_is_kept(self):
        self.assertNumberKept("刷る部数は三百か八百かで考えてる")

    def test_work_quantity_is_kept(self):
        self.assertNumberKept("チラシを480枚、仮で刷ってみます")
        self.assertNumberKept("カメラ役3人の手配をお願いします")
        self.assertNumberKept("メニュー4つぐらいにしましょう")
        self.assertNumberKept("3トントラックの写真を載せます")
        # 刷る・手配する だけの文も作業の文。
        self.assertNumberKept("370枚を仮で刷って")
        self.assertNumberKept("撮影役2人の手配")

    def test_volume_per_period_is_masked(self):
        self.assertNumberMasked("扉の発注が年間で390本から減ってしまって")
        self.assertNumberMasked("月720本の注文が切り替わってしまい")
        self.assertNumberKept("月2回の投稿にしましょう")
        self.assertNumberKept("パンフレット900部の印刷をお願いします")

    def test_age_decade_is_masked_unless_work(self):
        self.assertNumberMasked("60代の父が来ます")
        self.assertNumberKept("30代の女性向けのチラシを作ります")

    def test_kanji_numeral_inside_words_is_kept(self):
        self.assertNumberKept("五十嵐くんが来ます")
        self.assertNumberKept("千春さんにお願いします")
        self.assertNumberKept("百貨店の催事に出ます")

    def test_spelling_explanation_without_cue_words_is_masked_whole(self):
        out = [r.text for r in masked([
            "あ、ヤマダは山脈の山に田んぼの田で山田さんですけど",
        ]).records][0]
        for gone in ("山脈", "田んぼ", "山田"):
            self.assertNotIn(gone, out)
        self.assertIn("〈名前〉", out)

    def test_spelling_with_bare_first_kanji(self):
        out = masked(["あ、ハヤシは林に田んぼの田で林田さんですけど"]).records[0].text
        self.assertNotIn("田んぼ", out)
        self.assertNotIn("林田", out)

    def test_spelling_with_kakite_form(self):
        out = masked(["あ、ソウは曽に我と書いて部屋の部で曽部さんですけど"]).records[0].text
        self.assertNotIn("部屋", out)
        self.assertNotIn("曽部", out)

    def test_spelling_with_negated_form(self):
        out = masked(["あ、ミサキは山ではない方の岬に崎玉の崎で岬崎さんですけど"]).records[0].text
        self.assertNotIn("崎玉", out)
        self.assertNotIn("岬崎", out)

    def test_non_spelling_sentences_are_not_hit(self):
        for text in (
            "何かあれば携帯の方に掛けてください",
            "今のホームページには全然書けてへんのですよ",
            "棚の札に「貸出9」と書いた束",
            "体験が1人2500円と書いてあるだけで",
        ):
            self.assertEqual(spelling_spans(text), [], text)


if __name__ == "__main__":
    unittest.main()
