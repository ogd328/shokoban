"""``mask_stage5`` の部品（節の分け方・残った数の検査・日付の言い換え）の試験.

文はすべて架空の短い文。本物の読む役（``claude``）は起動しない。
"""

from __future__ import annotations

import ast
import re
import unittest
from datetime import date
from pathlib import Path

from shokoban.mask_stage5 import (
    add_date_suffixes,
    leftover_types,
    relative_date_suffixes,
    split_clauses,
)

MEETING = date(2026, 11, 10)  # 火曜日


class ClauseSplitTest(unittest.TestCase):
    """節の切り方（「が」「て」では切らない）."""

    def test_ga_and_te_do_not_split(self):
        self.assertEqual(len(split_clauses("私が作ります。")), 1)
        self.assertEqual(len(split_clauses("写真を撮って送ります。")), 1)
        self.assertEqual(len(split_clauses("案を出して、確認します。")), 2)

    def test_ends_after_node_and_kara(self):
        clauses = [clause for _, _, clause in split_clauses("安いので、写真を撮ります。")]
        self.assertEqual(clauses, ["安いので", "写真を撮ります。"])


class NetTest(unittest.TestCase):
    """置き換え漏れの型（``leftover_types``）."""

    def test_leftover_types(self):
        self.assertEqual(leftover_types("参加者は300人"), ("数字",))
        self.assertEqual(leftover_types("参加者は300部"), ())
        self.assertEqual(leftover_types("田中さんに渡す"), ("名前",))
        self.assertEqual(leftover_types("info@example.com"), ("連絡先",))
        self.assertEqual(
            leftover_types("ひだまり工房の話", company_name="ひだまり工房"),
            ("社名",),
        )
        self.assertEqual(leftover_types("問い合わせが少ない"), ())


class DateSuffixTest(unittest.TestCase):
    """相対の日付（年を書かない）."""

    def test_next_week_tuesday_within_the_year(self):
        self.assertEqual(relative_date_suffixes("来週火曜", MEETING), ("（11月17日）",))
        self.assertEqual(
            add_date_suffixes("来週火曜までに写真を3枚撮る", MEETING),
            "来週火曜までに写真を3枚撮る（11月17日）",
        )

    def test_year_changes_to_next_year(self):
        self.assertEqual(
            relative_date_suffixes("来週火曜", date(2026, 12, 28)),
            ("（翌年1月5日）",),
        )
        self.assertEqual(
            relative_date_suffixes("来月", date(2026, 12, 3)), ("（翌年1月）",)
        )
        self.assertEqual(
            relative_date_suffixes("月末", MEETING), ("（11月）",)
        )

    def test_no_four_digit_year_in_suffix(self):
        for text, meeting in (("来週火曜", MEETING), ("来週火曜", date(2026, 12, 28))):
            for suffix in relative_date_suffixes(text, meeting):
                self.assertIsNone(re.search(r"\d{4}", suffix))


class NoNetworkImportTest(unittest.TestCase):
    """伏せる部品がネットワーク・subprocess を import していないこと."""

    def test_component_does_not_import_banned_modules(self):
        import shokoban.mask_stage5 as module

        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        for banned in ("socket", "http", "urllib", "requests", "subprocess"):
            self.assertFalse(
                [name for name in names
                 if name == banned or name.startswith(banned + ".")],
                f"{banned} を import しています",
            )


if __name__ == "__main__":
    unittest.main()
