from __future__ import annotations

import logging

from drc_names_tagging.domain.services import NameComponentClassifier

logger = logging.getLogger(__name__)


def classify() -> None:
    logger.info("Classifying name components...")
    classifier = NameComponentClassifier()
    output_path = classifier.run()
    logger.info("Saved classification dataset to %s", output_path)


if __name__ == "__main__":
    classify()
