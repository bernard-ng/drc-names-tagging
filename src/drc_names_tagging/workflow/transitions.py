from __future__ import annotations

import logging

from drc_names_tagging.domain.services import TransitionMatrixGenerator

logger = logging.getLogger(__name__)


def transitions() -> None:
    logger.info("Generating Markov transitions...")
    service = TransitionMatrixGenerator()
    results = service.create_transition_matrices()
    for result in results:
        logger.info("Saved %s", result.summary())


if __name__ == "__main__":
    transitions()
