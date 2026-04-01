"""Tests du parsing PDF et du fallback chapitre."""

import unittest

from processor import _select_toc_level, detect_chapters


class ProcessorTests(unittest.TestCase):
    def test_select_toc_level_prefers_chapter_level_over_subsections(self):
        entries = [
            {"level": 1, "title": "Chapter 1: Futures Markets", "start_page": 10},
            {"level": 1, "title": "Chapter 2: Interest Rates", "start_page": 40},
            {"level": 1, "title": "Chapter 3: Swaps", "start_page": 75},
            {"level": 2, "title": "1.1 Futures Contracts", "start_page": 12},
            {"level": 2, "title": "1.2 History of Futures Markets", "start_page": 18},
            {"level": 2, "title": "2.1 Zero Rates", "start_page": 42},
        ]

        self.assertEqual(_select_toc_level(entries), 1)

    def test_detect_chapters_regex_fallback_extracts_clean_titles(self):
        pages = [
            {
                "text": "Chapter 1: Futures Markets\nDefinition of futures contracts.\n",
                "page": 1,
                "source": "book.pdf",
                "has_visuals": False,
            },
            {
                "text": "More details about margin calls.\n",
                "page": 2,
                "source": "book.pdf",
                "has_visuals": False,
            },
            {
                "text": "Chapter 2: Options Markets\nDefinition of calls and puts.\n",
                "page": 3,
                "source": "book.pdf",
                "has_visuals": False,
            },
        ]

        chapters = detect_chapters(pages)

        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "Chapter 1: Futures Markets")
        self.assertEqual(chapters[1]["title"], "Chapter 2: Options Markets")
        self.assertEqual(chapters[0]["chapter_origin"], "regex")


if __name__ == "__main__":
    unittest.main()
