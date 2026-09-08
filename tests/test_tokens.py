from __future__ import annotations

import json
import sqlite3
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl
from ollama import ChatResponse, Message

from drc_names_tagging.config import ExperimentConfig, ExperimentSettings
from drc_names_tagging.experiments.checkpoint import signature
from drc_names_tagging.experiments.tokens import (
    TokenRunner,
    compare_token_runs,
    run_tokens,
)
from drc_names_tagging.models import Name, Tag, Token
from drc_names_tagging.taggers.ollama import Ollama


@dataclass
class Fake:
    name: str = "fake"
    calls: list[str] = field(default_factory=list, repr=False)
    broken: bool = field(default=False, repr=False)

    def tag(self, name: str) -> Name:
        self.calls.append(name)
        if self.broken and name == "beta":
            raise ValueError("invalid response")
        return Name(name, (Token(0, name, Tag.NATIVE, 0.9),))


class TokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.checkpoint = self.directory / "annotations.sqlite3"
        self.words = pl.DataFrame(
            {
                "token_key": ["alpha", "beta", "gamma", "ndjondo"],
                "token": ["alpha", "beta", "gamma", "ndjondo"],
                "frequency": [1, 2, 3, 4],
            }
        )

    def test_resume_after_failure_and_changed_sample(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.sqlite3"
            fake = Fake(broken=True)
            with self.assertRaisesRegex(RuntimeError, "tokens failed"):
                TokenRunner().run(
                    fake, self.words, batch_size=1, checkpoint=path, retries=1
                )
            self.assertEqual(fake.calls, ["alpha", "beta", "beta"])
            fake.broken = False
            fake.calls.clear()
            run = TokenRunner().run(fake, self.words, batch_size=1, checkpoint=path)
            self.assertEqual(fake.calls, ["beta", "gamma", "ndjondo"])
            self.assertEqual((run.cached_tokens, run.processed_tokens), (1, 3))
            fake.calls.clear()
            sampled = TokenRunner().run(
                fake, self.words, batch_size=1, checkpoint=path, sample_fraction=0.5
            )
            self.assertEqual(fake.calls, [])
            self.assertEqual(sampled.cached_tokens, 2)
            self.assertEqual(
                compare_token_runs([sampled])[1]["tokens_per_second"][0], 0
            )
            changed = TokenRunner().run(
                Fake(name="different"), self.words, batch_size=1, checkpoint=path
            )
            self.assertEqual(changed.cached_tokens, 0)

    def test_cpu_processes_match_serial(self) -> None:
        serial = TokenRunner().run(
            Fake(), self.words, batch_size=1, checkpoint=self.checkpoint
        )
        parallel = TokenRunner().run(
            Fake(),
            self.words,
            cpu_workers=2,
            batch_size=1,
            checkpoint=self.checkpoint,
            resume=False,
        )
        self.assertEqual(parallel.processed_tokens, 4)
        self.assertTrue(serial.table.equals(parallel.table))

    def test_failure_drains_inflight_successes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.sqlite3"
            fake = Fake(broken=True)
            with self.assertRaises(RuntimeError):
                TokenRunner().run(
                    fake,
                    self.words,
                    checkpoint=path,
                    concurrency=2,
                    batch_size=1,
                    retries=0,
                )
            successful = set(fake.calls) - {"beta"}
            fake.broken = False
            fake.calls.clear()
            TokenRunner().run(
                fake, self.words, checkpoint=path, concurrency=2, batch_size=1
            )
            self.assertFalse(successful.intersection(fake.calls))

    def test_concurrency_is_bounded_and_results_are_ordered(self) -> None:
        lock = threading.Lock()
        active = maximum = 0

        def tag(name: str) -> Name:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.02 if name == "alpha" else 0.01)
            with lock:
                active -= 1
            return Name(name, (Token(0, name, Tag.NATIVE, 0.9),))

        with patch.object(Fake, "tag", side_effect=tag):
            result = TokenRunner().run(
                Fake(),
                self.words,
                concurrency=2,
                batch_size=1,
                checkpoint=self.checkpoint,
            )
        self.assertEqual(maximum, 2)
        self.assertEqual(result.table["token"].to_list(), self.words["token"].to_list())

    def test_ollama_batch_schema_and_validation(self) -> None:
        words = ["bope", "ndjondo", "ndjondo"]
        components = [
            {"token": token, "tag": "native", "confidence": 0.9} for token in words
        ]
        with patch("drc_names_tagging.taggers.ollama.Client") as client:
            client.return_value.chat.return_value = ChatResponse(
                message=Message(
                    role="assistant",
                    content=json.dumps(
                        {
                            "components": {
                                str(i): {"tag": "native", "confidence": 0.9}
                                for i in range(3)
                            }
                        }
                    ),
                )
            )
            result = Ollama().tag_batch(words)
            self.assertEqual([token.text for token in result], words)
            schema = client.return_value.chat.call_args.kwargs["format"]["properties"][
                "components"
            ]
            self.assertEqual(schema["required"], ["0", "1", "2"])
            self.assertFalse(schema["additionalProperties"])
            self.assertIn(
                "independent",
                client.return_value.chat.call_args.kwargs["messages"][1]["content"],
            )
        for invalid in (components[:2], list(reversed(components))):
            with self.assertRaises(ValueError):
                Ollama._parse({"components": invalid}, words)

    def test_batch_rejects_missing_ids_and_bad_confidence(self) -> None:
        with patch("drc_names_tagging.taggers.ollama.Client") as client:
            for values in (
                {},
                {"2": {"tag": "native", "confidence": 0.9}},
                {"0": {"tag": "native", "confidence": 2}},
            ):
                client.return_value.chat.return_value = ChatResponse(
                    message=Message(
                        role="assistant", content=json.dumps({"components": values})
                    )
                )
                with self.assertRaises(ValueError):
                    Ollama().tag_batch(["saint-plus"])

    def test_empty_vocabulary(self) -> None:
        run = TokenRunner().run(Fake(), self.words.head(0), checkpoint=self.checkpoint)
        self.assertEqual(run.table.height, 0)
        self.assertIn("tag", run.table.columns)

    def test_fresh_runs_are_persisted_without_replacing_resume_data(self) -> None:
        fake = Fake()
        runs = [TokenRunner().run(fake, self.words, checkpoint=self.checkpoint)]
        for _ in range(2):
            runs.append(
                TokenRunner().run(
                    fake, self.words, checkpoint=self.checkpoint, resume=False
                )
            )
        self.assertTrue(self.checkpoint.is_file())
        with sqlite3.connect(self.checkpoint) as connection:
            self.assertEqual(
                connection.execute("SELECT count(*) FROM annotations").fetchone()[0], 12
            )
            self.assertEqual(
                connection.execute(
                    "SELECT count(DISTINCT identity) FROM annotations"
                ).fetchone()[0],
                3,
            )
            location = connection.execute("PRAGMA database_list").fetchone()[2]
            self.assertEqual(Path(location), self.checkpoint.resolve())
        resumed = TokenRunner().run(fake, self.words, checkpoint=self.checkpoint)
        self.assertEqual(resumed.cached_tokens, 4)
        self.assertEqual(len({run.checkpoint_identity for run in runs}), 3)

    def test_checkpoint_location_is_independent_of_csv_output(self) -> None:
        settings = ExperimentSettings(dataset_path=self.directory / "names.csv")
        fake = Fake()
        experiment = ExperimentConfig(name="fake")
        with (
            patch(
                "drc_names_tagging.experiments.tokens.Vocabulary.load",
                return_value=self.words,
            ),
            patch(
                "drc_names_tagging.experiments.tokens.Registry.create",
                return_value=fake,
            ),
        ):
            run_tokens(experiment, settings, output=self.directory / "first.csv")
            fake.calls.clear()
            run_tokens(
                experiment, settings, output=self.directory / "elsewhere" / "second.csv"
            )
        self.assertEqual(fake.calls, [])
        self.assertTrue(settings.checkpoint_path.is_file())
        self.assertFalse((self.directory / "first.sqlite3").exists())

    def test_execution_config_validation(self) -> None:
        for settings in (
            {"batch_size": 0},
            {"cpu_workers": 0},
            {"retries": -1},
            {"cpu_workers": 2, "concurrency": 2},
        ):
            with self.assertRaises(ValueError):
                ExperimentConfig.from_dict({"name": "invalid", **settings})

    def test_checkpoint_key_is_manual(self) -> None:
        self.assertEqual(signature("manual-v2"), "manual-v2")
        self.assertEqual(
            ExperimentConfig(name="manual", checkpoint_key="  run-3  ").checkpoint_key,
            "run-3",
        )
        with self.assertRaises(ValueError):
            signature(" ")


if __name__ == "__main__":
    unittest.main()
