"""Tests ciblés sur les helpers du moteur sans appel aux APIs externes."""

import unittest

from engine import JeSuisCoachEngine


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = JeSuisCoachEngine.__new__(JeSuisCoachEngine)
        self.matches = [
            {
                "text": "The delta of an option measures the sensitivity of the option price to the underlying.",
                "metadata": {"has_visuals": False},
            }
        ]

    def test_parse_structured_evaluation_json(self):
        result = self.engine._parse_structured_evaluation(
            """
            {
              "score": 0.8,
              "feedback": "Bonne intuition",
              "correction": "Correct",
              "strengths": ["idee juste"],
              "mistakes": ["notation incomplete"],
              "source_used": "john-hull - p.381"
            }
            """
        )

        self.assertEqual(result["score"], 0.8)
        self.assertEqual(result["strengths"], ["idee juste"])
        self.assertEqual(result["mistakes"], ["notation incomplete"])
        self.assertEqual(result["source_used"], "john-hull - p.381")

    def test_parse_structured_evaluation_fallback(self):
        result = self.engine._parse_structured_evaluation(
            "SCORE: 0.4\nFEEDBACK: incomplet\nCORRECTION: $$d_1$$\nSOURCE_USED: ref"
        )

        self.assertEqual(result["score"], 0.4)
        self.assertEqual(result["feedback"], "incomplet")
        self.assertEqual(result["correction"], "$$d_1$$")
        self.assertEqual(result["source_used"], "ref")

    def test_select_question_type_avoids_mental_math_outside_dedicated_topic(self):
        question_type = self.engine._select_question_type(
            topic="Black-Scholes",
            difficulty="Fondamental",
            selected_matches=self.matches,
            recent_question_types=["definition", "application"],
        )

        self.assertEqual(question_type, "definition")

    def test_select_question_type_keeps_mental_math_for_dedicated_topic(self):
        question_type = self.engine._select_question_type(
            topic="Calcul mental / Approximations",
            difficulty="Fondamental",
            selected_matches=self.matches,
            recent_question_types=["definition", "application"],
        )

        self.assertEqual(question_type, "mental_math")


if __name__ == "__main__":
    unittest.main()
