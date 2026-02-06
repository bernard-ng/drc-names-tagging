from __future__ import annotations

import logging

from drc_names_tagging.domain.services import NameComponentClassifier

logger = logging.getLogger(__name__)


def classify() -> None:
    logger.info("Classifying name components...")
    classifier = NameComponentClassifier()
    csv_path, json_path = classifier.run_with_json()
    logger.info("Saved classification dataset to %s", csv_path)
    logger.info("Saved classification json to %s", json_path)


if __name__ == "__main__":
    classify()
