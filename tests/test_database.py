"""Tests SQLite pour la persistance structurée des réponses."""

import os
import tempfile
import unittest

import database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = database.DB_PATH
        database.DB_PATH = os.path.join(self.temp_dir.name, "quant_coach_test.db")
        database.init_db()

    def tearDown(self):
        database.DB_PATH = self.original_db_path
        self.temp_dir.cleanup()

    def test_save_answer_persists_structured_fields(self):
        session_id = database.create_session("Black-Scholes")
        database.save_answer(
            session_id=session_id,
            question="Expliquez le delta.",
            user_answer="C'est une sensibilite.",
            score=0.8,
            response_time_s=12.5,
            feedback="Bonne intuition",
            source_ref="john-hull - p.381",
            correction="Correct",
            mistakes=["notation incomplete"],
            strengths=["idee juste"],
            question_type="Compréhension / définition",
            difficulty="Fondamental",
            source_used="john-hull - p.381",
        )

        history = database.get_answer_history(10)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["mistakes"], ["notation incomplete"])
        self.assertEqual(history[0]["strengths"], ["idee juste"])
        self.assertEqual(history[0]["question_type"], "Compréhension / définition")
        self.assertEqual(history[0]["difficulty"], "Fondamental")
        self.assertEqual(history[0]["source_used"], "john-hull - p.381")


if __name__ == "__main__":
    unittest.main()
