import os
import sys
import unittest

# Ensure pdf_rag_bot directory is in Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory import recent_turns, condense_question, same_question


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return FakeResponse(self.reply)


class FakeRegistry:
    def __init__(self, llm, tokenizer=None):
        self.llm = llm
        self.tokenizer = tokenizer

    def load_llm(self):
        return self.llm, self.tokenizer


TURNS = [{"question": "What is BM25?", "answer": "BM25 is a keyword ranking algorithm."}]


class TestSessionMemory(unittest.TestCase):
    def test_same_question_ignores_case_spacing_and_punctuation(self):
        self.assertTrue(same_question("What is BM25?", "what is bm25"))
        self.assertFalse(same_question("What is BM25?", "What is TF-IDF?"))

    def test_recent_turns_pairs_and_limits(self):
        messages = []
        for i in range(5):
            messages.append({"role": "user", "content": f"q{i}"})
            messages.append({"role": "assistant", "content": f"a{i}"})
        turns = recent_turns(messages, max_turns=3)
        self.assertEqual([t["question"] for t in turns], ["q2", "q3", "q4"])
        self.assertEqual(turns[-1]["answer"], "a4")

    def test_recent_turns_uses_resolved_question_and_skips_unanswered(self):
        messages = [
            {"role": "user", "content": "how is he?", "standalone": "How is Ramo?"},
            {"role": "assistant", "content": "He is fine."},
            {"role": "user", "content": "still waiting for an answer"}
        ]
        turns = recent_turns(messages)
        self.assertEqual(turns, [{"question": "How is Ramo?", "answer": "He is fine."}])

    def test_no_history_makes_no_llm_call(self):
        llm = FakeLLM(reply="should not be used")
        self.assertEqual(condense_question("What is BM25?", [], FakeRegistry(llm)), "What is BM25?")
        self.assertEqual(llm.calls, [])

    def test_follow_up_is_rewritten(self):
        llm = FakeLLM(reply='"What are the default parameters of BM25?"\nextra line')
        result = condense_question("What are its defaults?", TURNS, FakeRegistry(llm))
        self.assertEqual(result, "What are the default parameters of BM25?")
        prompt = llm.calls[0][1][1]
        self.assertIn("What is BM25?", prompt)
        self.assertIn("What are its defaults?", prompt)

    def test_falls_back_to_question_on_failure_or_bad_output(self):
        question = "What are its defaults?"
        self.assertEqual(condense_question(question, TURNS, FakeRegistry(FakeLLM(error=RuntimeError("down")))), question)
        self.assertEqual(condense_question(question, TURNS, FakeRegistry(FakeLLM(reply="   "))), question)
        self.assertEqual(condense_question(question, TURNS, FakeRegistry(FakeLLM(reply="x" * 5000))), question)

    def test_non_chat_model_keeps_question(self):
        llm = FakeLLM(reply="rewritten")
        self.assertEqual(condense_question("What are its defaults?", TURNS, FakeRegistry(llm, tokenizer=object())),
                         "What are its defaults?")
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
