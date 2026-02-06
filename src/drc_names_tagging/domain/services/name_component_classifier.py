from __future__ import annotations

from pathlib import Path

import polars as pl
from tqdm import tqdm

from drc_names_tagging.core import assert_file_exists
from drc_names_tagging.core import get_dataset_path
from drc_names_tagging.domain.model import TokenClassification
from drc_names_tagging.domain.model import TransitionModel

NATIVE_LABEL = "probable_native"
SURNAME_LABEL = "probable_surname"
TAG_NATIVE = "native"
TAG_SURNAME = "surname"


class NameComponentClassifier:
    def __init__(
        self,
        *,
        dataset_filename: str = "names.csv",
        transition_stage: str = "sliver",
        output_stage: str = "sliver",
        epsilon: float = 1e-6,
        output_filename: str = "name_component_classification.csv",
        output_json_filename: str = "name_component_classification.json",
    ) -> None:
        self._dataset_filename = dataset_filename
        self._transition_stage = transition_stage
        self._output_stage = output_stage
        self._epsilon = epsilon
        self._output_filename = output_filename
        self._output_json_filename = output_json_filename

        self._native_model = self._load_model(NATIVE_LABEL, "native_transition.csv")
        self._surname_model = self._load_model(SURNAME_LABEL, "surname_transition.csv")

    def classify_dataset(self) -> pl.DataFrame:
        table = self._classify_token_table()
        return (
            table.sort(["row_index", "token_index"])
            .select(
                [
                    "name",
                    "token_index",
                    "component",
                    "tag",
                    "native_score",
                    "surname_score",
                    "score_margin",
                ]
            )
        )

    def save_classification(self, table: pl.DataFrame) -> Path:
        output_path = get_dataset_path(self._output_stage, self._output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(output_path, float_precision=8)
        return output_path

    def save_classification_json(self, table: pl.DataFrame) -> Path:
        json_table = self._build_json_dataset(table)
        output_path = get_dataset_path(self._output_stage, self._output_json_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_table.write_json(output_path)
        return output_path

    def run(self) -> Path:
        table = self.classify_dataset()
        return self.save_classification(table)

    def run_with_json(self) -> tuple[Path, Path]:
        token_table = self._classify_token_table()
        csv_path = self.save_classification(
            token_table.sort(["row_index", "token_index"]).select(
                [
                    "name",
                    "token_index",
                    "component",
                    "tag",
                    "native_score",
                    "surname_score",
                    "score_margin",
                ]
            )
        )
        json_path = self.save_classification_json(token_table)
        return csv_path, json_path

    def classify_name(self, name: str) -> list[dict[str, float | int | str]]:
        tokens = self._tokenize_name(name)
        classifications = self._classify_tokens(tokens)
        return [
            {
                "token_index": index,
                "component": token,
                "tag": self._tag_from_component(item.predicted_component),
                "native_score": item.native_score,
                "surname_score": item.surname_score,
                "score_margin": item.score_margin,
            }
            for index, (token, item) in enumerate(zip(tokens, classifications))
        ]

    def _load_dataset(self) -> pl.DataFrame:
        dataset_path = assert_file_exists(
            get_dataset_path("bronze", self._dataset_filename)
        )
        return pl.read_csv(dataset_path)

    def _load_model(self, component: str, filename: str) -> TransitionModel:
        path = assert_file_exists(get_dataset_path(self._transition_stage, filename))
        return TransitionModel.from_csv(component, path, epsilon=self._epsilon)

    def _explode_tokens(self, dataset: pl.DataFrame) -> pl.DataFrame:
        cleaned_name = (
            pl.col("name")
            .fill_null("")
            .cast(pl.String)
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")
        )
        tokens = cleaned_name.str.split(" ")
        table = dataset.with_row_index("row_index").with_columns(
            [
                cleaned_name.alias("name_clean"),
                tokens.alias("tokens"),
            ]
        )
        table = table.with_columns(
            pl.int_ranges(0, pl.col("tokens").list.len()).alias("token_index")
        ).explode(["tokens", "token_index"])

        return (
            table.rename({"tokens": "token_raw"})
            .with_columns(
                [
                    pl.col("token_raw")
                    .fill_null("")
                    .cast(pl.String)
                    .str.strip_chars()
                    .alias("token_raw"),
                    pl.col("token_raw")
                    .fill_null("")
                    .cast(pl.String)
                    .str.strip_chars()
                    .str.to_lowercase()
                    .alias("token_normalized"),
                ]
            )
            .filter(pl.col("token_raw").str.len_chars() > 0)
        )

    def _classify_tokens(self, tokens: list[str]) -> list[TokenClassification]:
        return [
            self._classify_token(token)
            for token in tqdm(tokens, desc="Classifying tokens", unit="token")
        ]

    def _classify_token(self, token: str) -> TokenClassification:
        normalized = self._normalize_token(token)
        native_score = self._native_model.average_log_likelihood(normalized)
        surname_score = self._surname_model.average_log_likelihood(normalized)
        if native_score >= surname_score:
            predicted_component = NATIVE_LABEL
        else:
            predicted_component = SURNAME_LABEL

        return TokenClassification(
            token_normalized=normalized,
            native_score=native_score,
            surname_score=surname_score,
            predicted_component=predicted_component,
            score_margin=native_score - surname_score,
        )

    def _normalize_token(self, token: str) -> str:
        return " ".join(token.strip().split()).lower()

    def _tokenize_name(self, name: str) -> list[str]:
        normalized = " ".join(name.strip().split())
        if not normalized:
            return []
        return normalized.split(" ")

    def _tag_from_component(self, component: str) -> str:
        if component == NATIVE_LABEL:
            return TAG_NATIVE
        return TAG_SURNAME

    def _build_json_dataset(self, table: pl.DataFrame) -> pl.DataFrame:
        return (
            table.with_columns(
                pl.struct(
                    [
                        pl.col("token_index"),
                        pl.col("component"),
                        pl.col("tag"),
                        pl.col("native_score"),
                        pl.col("surname_score"),
                        pl.col("score_margin"),
                    ]
                ).alias("token_struct")
            )
            .group_by(["row_index", "name"])
            .agg(pl.col("token_struct").sort_by("token_index").alias("components"))
            .sort("row_index")
            .select(["name", "components"])
        )

    def _classify_token_table(self) -> pl.DataFrame:
        dataset = self._load_dataset()
        tokens = self._explode_tokens(dataset)
        classifications = self._classify_tokens(
            tokens.get_column("token_normalized").to_list()
        )
        return tokens.with_columns(
            [
                pl.Series(
                    "predicted_component",
                    [item.predicted_component for item in classifications],
                ),
                pl.Series(
                    "tag",
                    [
                        self._tag_from_component(item.predicted_component)
                        for item in classifications
                    ],
                ),
                pl.Series(
                    "native_score", [item.native_score for item in classifications]
                ),
                pl.Series(
                    "surname_score", [item.surname_score for item in classifications]
                ),
                pl.Series(
                    "score_margin", [item.score_margin for item in classifications]
                ),
                pl.col("token_raw").alias("component"),
            ]
        )
