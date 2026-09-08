from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from drc_names_tagging.models import Model, Name, Tag, Token, tokens
from drc_names_tagging.utils.paths import assert_file


# Methodology: the versioned matrices are derived from the corpus transition
# reports, using probable native and probable surname name groups as the
# native and foreign character distributions, respectively. See:
# https://github.com/bernard-ng/drc-names-corpus/blob/main/reports/name_analysis/probable_native_transition_matrix.csv
# https://github.com/bernard-ng/drc-names-corpus/blob/main/reports/name_analysis/probable_surname_transition_matrix.csv
@dataclass(frozen=True, slots=True)
class Markov:
    """Character-transition baseline for native/foreign token tagging."""

    native: Model
    foreign: Model

    @classmethod
    def from_paths(cls, native: str | Path, foreign: str | Path) -> Markov:
        return cls(
            native=Model.from_csv("native", assert_file(Path(native))),
            foreign=Model.from_csv("foreign", assert_file(Path(foreign))),
        )

    @property
    def name(self) -> str:
        return "markov"

    def tag(self, name: str) -> Name:
        annotations: list[Token] = []
        for index, token in enumerate(tokens(name)):
            native_score = self.native.average_log_likelihood(token.lower())
            foreign_score = self.foreign.average_log_likelihood(token.lower())
            tag = Tag.NATIVE if native_score >= foreign_score else Tag.FOREIGN
            annotations.append(Token(index, token, tag, native_score - foreign_score))
        return Name(name, tuple(annotations))
