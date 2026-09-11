#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Select the 600-sentence SICK subset used to build the corpus."""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd
from datasets import DatasetDict, load_dataset


HF_DATASET_ID = "RobZamp/sick"

HF_DATASET_REVISION = "refs/convert/parquet"

SPLITS_TO_USE = (
    "train",
    "validation",
    "test",
)

OUTPUT_DIR = Path("./sick_filtered_subset")

RANDOM_SEED = 42

PAIRS_PER_GROUP = {
    "low": 100,
    "medium": 100,
    "high": 100,
}


LOW_MAX = 2.3

MEDIUM_MIN = 3.0
MEDIUM_MAX = 3.7

HIGH_MIN = 4.2


MIN_WORDS = 4
MAX_WORDS = 30

REMOVE_IDENTICAL_PAIRS = True
REMOVE_REVERSED_DUPLICATES = True

REMOVE_QUESTIONS = True
REMOVE_EXCLAMATIONS = True

REMOVE_URLS = True
REMOVE_HTML = True


ENFORCE_SENTENCE_DISJOINT_PAIRS = True

DISJOINT_SELECTION_ATTEMPTS = 1_000


VALID_ENTAILMENT_LABELS = {
    "ENTAILMENT",
    "NEUTRAL",
    "CONTRADICTION",
}

LABEL_ID_TO_NAME = {
    0: "ENTAILMENT",
    1: "NEUTRAL",
    2: "CONTRADICTION",
}


def decode_entailment_label(
    value: object,
    label_feature: Optional[object],
) -> str:
    if pd.isna(value):
        return ""

    if label_feature is not None and hasattr(label_feature, "int2str"):
        try:
            return label_feature.int2str(int(value)).strip().upper()
        except (TypeError, ValueError, KeyError):
            pass

    if isinstance(value, str):
        normalized = value.strip().upper()

        if normalized in VALID_ENTAILMENT_LABELS:
            return normalized

        try:
            value = int(normalized)
        except ValueError:
            return normalized

    try:
        return LABEL_ID_TO_NAME[int(value)]
    except (TypeError, ValueError, KeyError):
        return str(value).strip().upper()


def find_column(
    dataframe: pd.DataFrame,
    possible_names: Tuple[str, ...],
) -> Optional[str]:
    """Find the first available column from a list of possible names."""
    for name in possible_names:
        if name in dataframe.columns:
            return name

    return None


def normalize_huggingface_split(
    split_dataset,
    split_name: str,
) -> pd.DataFrame:
    dataframe = split_dataset.to_pandas()

    sentence_a_column = find_column(
        dataframe,
        (
            "sentence_A",
            "sentence_a",
            "sentence1",
            "sentence_1",
        ),
    )

    sentence_b_column = find_column(
        dataframe,
        (
            "sentence_B",
            "sentence_b",
            "sentence2",
            "sentence_2",
        ),
    )

    score_column = find_column(
        dataframe,
        (
            "relatedness_score",
            "relatedness",
            "similarity_score",
        ),
    )

    label_column = find_column(
        dataframe,
        (
            "label",
            "entailment_judgment",
            "entailment_label",
            "entailment",
        ),
    )

    pair_id_column = find_column(
        dataframe,
        (
            "id",
            "pair_ID",
            "pair_id",
        ),
    )

    missing = []

    if sentence_a_column is None:
        missing.append("sentence A")

    if sentence_b_column is None:
        missing.append("sentence B")

    if score_column is None:
        missing.append("relatedness score")

    if label_column is None:
        missing.append("entailment label")

    if missing:
        raise ValueError(
            f"Could not identify {missing} in split '{split_name}'.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    if pair_id_column is None:
        pair_ids = [
            f"{split_name}_{index:06d}"
            for index in range(len(dataframe))
        ]
    else:
        pair_ids = dataframe[pair_id_column].astype(str)

    label_feature = split_dataset.features.get(label_column)

    canonical = pd.DataFrame(
        {
            "pair_id": pair_ids,
            "sentence_a": dataframe[sentence_a_column],
            "sentence_b": dataframe[sentence_b_column],
            "relatedness_score": dataframe[score_column],
            "entailment_judgment": dataframe[label_column].map(
                lambda value: decode_entailment_label(
                    value=value,
                    label_feature=label_feature,
                )
            ),
            "split": split_name,
        }
    )

    return canonical


def load_sick_from_huggingface() -> pd.DataFrame:
    """Download and combine the configured SICK splits."""
    print("=" * 80)
    print("LOADING SICK FROM HUGGING FACE")
    print("=" * 80)
    print(f"Dataset:  {HF_DATASET_ID}")
    print(f"Revision: {HF_DATASET_REVISION}")

    dataset = load_dataset(
        HF_DATASET_ID,
        revision=HF_DATASET_REVISION,
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            "Expected a DatasetDict, but received "
            f"{type(dataset).__name__}."
        )

    missing_splits = [
        split_name
        for split_name in SPLITS_TO_USE
        if split_name not in dataset
    ]

    if missing_splits:
        raise ValueError(
            f"Missing dataset splits: {missing_splits}\n"
            f"Available splits: {list(dataset.keys())}"
        )

    split_frames = []

    for split_name in SPLITS_TO_USE:
        split_frame = normalize_huggingface_split(
            split_dataset=dataset[split_name],
            split_name=split_name,
        )

        split_frames.append(split_frame)

        print(
            f"Loaded {split_name:12s}: "
            f"{len(split_frame):,} pairs"
        )

    dataframe = pd.concat(
        split_frames,
        ignore_index=True,
    )

    print(f"\nCombined dataset: {len(dataframe):,} pairs")

    return dataframe


def normalize_text(value: object) -> str:
    """Normalize whitespace while preserving the sentence content."""
    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\u00a0", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalized_sentence_key(text: str) -> str:
    """Create a case-insensitive sentence key."""
    return normalize_text(text).casefold()


def word_count(text: str) -> int:
    """Count word-like units in a sentence."""
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
            flags=re.UNICODE,
        )
    )


def contains_url(text: str) -> bool:
    """Return True when a sentence contains a URL."""
    return bool(
        re.search(
            r"(https?://|ftp://|www\.)",
            text,
            flags=re.IGNORECASE,
        )
    )


def contains_html(text: str) -> bool:
    """Return True when a sentence contains HTML-like markup."""
    return bool(re.search(r"<[^>]+>", text))


def sentence_is_valid(text: str) -> bool:
    """Apply the configured sentence-level filters."""
    if not text:
        return False

    number_of_words = word_count(text)

    if number_of_words < MIN_WORDS:
        return False

    if number_of_words > MAX_WORDS:
        return False

    if REMOVE_QUESTIONS and "?" in text:
        return False

    if REMOVE_EXCLAMATIONS and "!" in text:
        return False

    if REMOVE_URLS and contains_url(text):
        return False

    if REMOVE_HTML and contains_html(text):
        return False

    return True


def create_pair_key(
    sentence_a: str,
    sentence_b: str,
) -> Tuple[str, str]:
    """Create a key for detecting duplicate and reversed pairs."""
    key_a = normalized_sentence_key(sentence_a)
    key_b = normalized_sentence_key(sentence_b)

    if REMOVE_REVERSED_DUPLICATES:
        return tuple(sorted((key_a, key_b)))

    return key_a, key_b


def clean_and_filter_pairs(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the complete SICK dataframe."""
    dataframe = dataframe.copy()

    initial_size = len(dataframe)

    dataframe["sentence_a"] = dataframe["sentence_a"].map(
        normalize_text
    )

    dataframe["sentence_b"] = dataframe["sentence_b"].map(
        normalize_text
    )

    dataframe["relatedness_score"] = pd.to_numeric(
        dataframe["relatedness_score"],
        errors="coerce",
    )

    dataframe["entailment_judgment"] = (
        dataframe["entailment_judgment"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe = dataframe.dropna(
        subset=[
            "sentence_a",
            "sentence_b",
            "relatedness_score",
            "entailment_judgment",
        ]
    ).copy()

    dataframe = dataframe[
        dataframe["relatedness_score"].between(1.0, 5.0)
    ].copy()

    dataframe = dataframe[
        dataframe["entailment_judgment"].isin(
            VALID_ENTAILMENT_LABELS
        )
    ].copy()

    dataframe["sentence_a_word_count"] = (
        dataframe["sentence_a"].map(word_count)
    )

    dataframe["sentence_b_word_count"] = (
        dataframe["sentence_b"].map(word_count)
    )

    valid_sentence_a = dataframe["sentence_a"].map(
        sentence_is_valid
    )

    valid_sentence_b = dataframe["sentence_b"].map(
        sentence_is_valid
    )

    dataframe = dataframe[
        valid_sentence_a & valid_sentence_b
    ].copy()

    if REMOVE_IDENTICAL_PAIRS:
        identical_mask = (
            dataframe["sentence_a"].map(normalized_sentence_key)
            == dataframe["sentence_b"].map(normalized_sentence_key)
        )

        dataframe = dataframe[~identical_mask].copy()

    dataframe["_pair_key"] = dataframe.apply(
        lambda row: create_pair_key(
            row["sentence_a"],
            row["sentence_b"],
        ),
        axis=1,
    )

    dataframe = dataframe.drop_duplicates(
        subset="_pair_key",
        keep="first",
    ).copy()

    dataframe = dataframe.drop(columns="_pair_key")
    dataframe = dataframe.reset_index(drop=True)

    removed = initial_size - len(dataframe)

    print("\n" + "=" * 80)
    print("CLEANING")
    print("=" * 80)
    print(f"Initial pairs:   {initial_size:,}")
    print(f"Removed pairs:   {removed:,}")
    print(f"Remaining pairs: {len(dataframe):,}")

    return dataframe


def assign_similarity_group(score: float) -> str:
    if score <= LOW_MAX:
        return "low"

    if MEDIUM_MIN <= score <= MEDIUM_MAX:
        return "medium"

    if score >= HIGH_MIN:
        return "high"

    return "excluded"


def add_similarity_groups(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add the similarity-group column."""
    dataframe = dataframe.copy()

    dataframe["similarity_group"] = (
        dataframe["relatedness_score"].map(
            assign_similarity_group
        )
    )

    return dataframe


def select_group_without_disjoint_constraint(
    candidates: pd.DataFrame,
    requested_count: int,
    seed: int,
) -> pd.DataFrame:
    """Randomly sample pairs without enforcing sentence uniqueness."""
    if len(candidates) < requested_count:
        raise ValueError(
            f"Only {len(candidates):,} candidate pairs are available, "
            f"but {requested_count:,} were requested."
        )

    return candidates.sample(
        n=requested_count,
        random_state=seed,
    ).copy()


def greedy_disjoint_selection(
    candidates: pd.DataFrame,
    requested_count: int,
    used_sentences: Set[str],
    seed: int,
) -> Tuple[pd.DataFrame, Set[str]]:
    shuffled = candidates.sample(
        frac=1.0,
        random_state=seed,
    )

    local_used_sentences = set(used_sentences)
    selected_indices = []

    for index, row in shuffled.iterrows():
        key_a = normalized_sentence_key(row["sentence_a"])
        key_b = normalized_sentence_key(row["sentence_b"])

        if key_a in local_used_sentences:
            continue

        if key_b in local_used_sentences:
            continue

        selected_indices.append(index)

        local_used_sentences.add(key_a)
        local_used_sentences.add(key_b)

        if len(selected_indices) >= requested_count:
            break

    selected = candidates.loc[selected_indices].copy()

    return selected, local_used_sentences


def select_sentence_disjoint_pairs(
    eligible_pairs: pd.DataFrame,
) -> pd.DataFrame:
    group_names = list(PAIRS_PER_GROUP.keys())

    best_total = 0
    best_counts: Dict[str, int] = {}
    best_parts: Dict[str, pd.DataFrame] = {}

    for attempt in range(DISJOINT_SELECTION_ATTEMPTS):
        attempt_seed = RANDOM_SEED + attempt * 10_000

        offset = attempt % len(group_names)

        group_order = (
            group_names[offset:]
            + group_names[:offset]
        )

        used_sentences: Set[str] = set()
        selected_parts: Dict[str, pd.DataFrame] = {}

        successful = True

        for group_index, group_name in enumerate(group_order):
            requested_count = PAIRS_PER_GROUP[group_name]

            candidates = eligible_pairs[
                eligible_pairs["similarity_group"] == group_name
            ].copy()

            selected_group, updated_used_sentences = (
                greedy_disjoint_selection(
                    candidates=candidates,
                    requested_count=requested_count,
                    used_sentences=used_sentences,
                    seed=attempt_seed + group_index * 1_000,
                )
            )

            selected_parts[group_name] = selected_group
            used_sentences = updated_used_sentences

            if len(selected_group) < requested_count:
                successful = False
                break

        current_counts = {
            group_name: len(
                selected_parts.get(
                    group_name,
                    eligible_pairs.iloc[0:0],
                )
            )
            for group_name in group_names
        }

        current_total = sum(current_counts.values())

        if current_total > best_total:
            best_total = current_total
            best_counts = current_counts
            best_parts = selected_parts

        if successful:
            selected = pd.concat(
                [
                    selected_parts[group_name]
                    for group_name in group_names
                ],
                ignore_index=True,
            )

            print(
                "\nSentence-disjoint selection succeeded "
                f"on attempt {attempt + 1:,}."
            )

            return selected

    raise RuntimeError(
        "Could not create a complete sentence-disjoint selection.\n\n"
        f"Best total selected: {best_total:,}\n"
        f"Best group counts: {best_counts}\n\n"
        "Possible solutions:\n"
        "1. Increase DISJOINT_SELECTION_ATTEMPTS.\n"
        "2. Reduce PAIRS_PER_GROUP.\n"
        "3. Set ENFORCE_SENTENCE_DISJOINT_PAIRS = False."
    )


def select_pairs(
    eligible_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Select the final low-, medium-, and high-relatedness pairs."""
    print("\n" + "=" * 80)
    print("PAIR SELECTION")
    print("=" * 80)

    for group_name, requested_count in PAIRS_PER_GROUP.items():
        available_count = int(
            (
                eligible_pairs["similarity_group"]
                == group_name
            ).sum()
        )

        print(
            f"{group_name:7s}: "
            f"{available_count:5,d} available, "
            f"{requested_count:3,d} requested"
        )

        if available_count < requested_count:
            raise ValueError(
                f"Group '{group_name}' contains only "
                f"{available_count:,} eligible pairs, but "
                f"{requested_count:,} were requested."
            )

    if ENFORCE_SENTENCE_DISJOINT_PAIRS:
        selected = select_sentence_disjoint_pairs(
            eligible_pairs=eligible_pairs,
        )

    else:
        selected_parts = []

        for group_index, group_name in enumerate(
            PAIRS_PER_GROUP
        ):
            requested_count = PAIRS_PER_GROUP[group_name]

            candidates = eligible_pairs[
                eligible_pairs["similarity_group"] == group_name
            ].copy()

            selected_group = (
                select_group_without_disjoint_constraint(
                    candidates=candidates,
                    requested_count=requested_count,
                    seed=RANDOM_SEED + group_index * 10_000,
                )
            )

            selected_parts.append(selected_group)

        selected = pd.concat(
            selected_parts,
            ignore_index=True,
        )

    group_order = pd.CategoricalDtype(
        categories=["low", "medium", "high"],
        ordered=True,
    )

    selected["similarity_group"] = (
        selected["similarity_group"].astype(group_order)
    )

    selected = selected.sort_values(
        by=[
            "similarity_group",
            "relatedness_score",
            "pair_id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    selected.insert(
        0,
        "selected_pair_id",
        [
            f"sick_pair_{index:06d}"
            for index in range(1, len(selected) + 1)
        ],
    )

    selected["similarity_group"] = (
        selected["similarity_group"].astype(str)
    )

    return selected


def build_unique_sentence_table(
    selected_pairs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sentence_to_id: Dict[str, str] = {}
    sentence_records = []

    def register_sentence(sentence: str) -> str:
        key = normalized_sentence_key(sentence)

        if key not in sentence_to_id:
            sentence_id = (
                f"sick_sentence_"
                f"{len(sentence_to_id) + 1:06d}"
            )

            sentence_to_id[key] = sentence_id

            sentence_records.append(
                {
                    "sentence_id": sentence_id,
                    "text": sentence,
                    "word_count": word_count(sentence),
                    "language": "en",
                }
            )

        return sentence_to_id[key]

    selected_pairs = selected_pairs.copy()

    selected_pairs["sentence_a_id"] = (
        selected_pairs["sentence_a"].map(
            register_sentence
        )
    )

    selected_pairs["sentence_b_id"] = (
        selected_pairs["sentence_b"].map(
            register_sentence
        )
    )

    unique_sentences = pd.DataFrame(
        sentence_records
    )

    preferred_pair_columns = [
        "selected_pair_id",
        "pair_id",
        "sentence_a_id",
        "sentence_b_id",
        "sentence_a",
        "sentence_b",
        "relatedness_score",
        "similarity_group",
        "entailment_judgment",
        "split",
        "sentence_a_word_count",
        "sentence_b_word_count",
    ]

    selected_pairs = selected_pairs[
        preferred_pair_columns
    ].copy()

    return selected_pairs, unique_sentences


def write_generation_metadata(
    unique_sentences: pd.DataFrame,
    output_path: Path,
) -> None:
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in unique_sentences.itertuples(
            index=False
        ):
            record = {
                "sentence_id": row.sentence_id,
                "text": row.text,
                "language": row.language,
                "speaker": None,
                "emotion": None,
                "seed": None,
                "output_audio": None,
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def create_selection_summary(
    selected_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Create group-level statistics with entailment counts."""
    group_statistics = (
        selected_pairs
        .groupby(
            "similarity_group",
            sort=False,
        )
        .agg(
            number_of_pairs=(
                "selected_pair_id",
                "count",
            ),
            mean_relatedness=(
                "relatedness_score",
                "mean",
            ),
            standard_deviation=(
                "relatedness_score",
                "std",
            ),
            minimum_relatedness=(
                "relatedness_score",
                "min",
            ),
            maximum_relatedness=(
                "relatedness_score",
                "max",
            ),
        )
    )

    entailment_counts = pd.crosstab(
        selected_pairs["similarity_group"],
        selected_pairs["entailment_judgment"],
    )

    entailment_counts = entailment_counts.rename(
        columns={
            label: f"{label.lower()}_count"
            for label in entailment_counts.columns
        }
    )

    summary = group_statistics.join(
        entailment_counts,
        how="left",
    )

    summary = summary.reset_index()

    return summary


def save_configuration(
    output_path: Path,
) -> None:
    """Save the experiment configuration for reproducibility."""
    configuration = {
        "huggingface_dataset_id": HF_DATASET_ID,
        "huggingface_revision": HF_DATASET_REVISION,
        "splits": list(SPLITS_TO_USE),
        "random_seed": RANDOM_SEED,
        "pairs_per_group": PAIRS_PER_GROUP,
        "thresholds": {
            "low": {
                "maximum": LOW_MAX,
            },
            "medium": {
                "minimum": MEDIUM_MIN,
                "maximum": MEDIUM_MAX,
            },
            "high": {
                "minimum": HIGH_MIN,
            },
            "excluded_regions": [
                {
                    "minimum_exclusive": LOW_MAX,
                    "maximum_exclusive": MEDIUM_MIN,
                },
                {
                    "minimum_exclusive": MEDIUM_MAX,
                    "maximum_exclusive": HIGH_MIN,
                },
            ],
        },
        "text_filters": {
            "minimum_words": MIN_WORDS,
            "maximum_words": MAX_WORDS,
            "remove_identical_pairs": REMOVE_IDENTICAL_PAIRS,
            "remove_reversed_duplicates": REMOVE_REVERSED_DUPLICATES,
            "remove_questions": REMOVE_QUESTIONS,
            "remove_exclamations": REMOVE_EXCLAMATIONS,
            "remove_urls": REMOVE_URLS,
            "remove_html": REMOVE_HTML,
        },
        "sampling": {
            "preserve_natural_entailment_distribution": True,
            "enforce_sentence_disjoint_pairs": (
                ENFORCE_SENTENCE_DISJOINT_PAIRS
            ),
            "disjoint_selection_attempts": (
                DISJOINT_SELECTION_ATTEMPTS
            ),
        },
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_available_distribution(
    dataframe: pd.DataFrame,
) -> None:
    """Print candidate counts before sampling."""
    print("\n" + "=" * 80)
    print("RELATEDNESS GROUP DISTRIBUTION")
    print("=" * 80)

    group_counts = (
        dataframe["similarity_group"]
        .value_counts()
        .reindex(
            [
                "low",
                "medium",
                "high",
                "excluded",
            ],
            fill_value=0,
        )
    )

    for group_name, count in group_counts.items():
        percentage = count / len(dataframe) * 100

        print(
            f"{group_name:8s}: "
            f"{count:5,d} pairs "
            f"({percentage:6.2f}%)"
        )

    eligible = dataframe[
        dataframe["similarity_group"] != "excluded"
    ]

    print("\nEligible group × entailment distribution:")

    print(
        pd.crosstab(
            eligible["similarity_group"],
            eligible["entailment_judgment"],
        )
        .reindex(
            index=["low", "medium", "high"],
            fill_value=0,
        )
        .to_string()
    )


def print_final_summary(
    selected_pairs: pd.DataFrame,
    unique_sentences: pd.DataFrame,
) -> None:
    """Print final subset statistics."""
    print("\n" + "=" * 80)
    print("FINAL SELECTION")
    print("=" * 80)

    print(f"Selected pairs:   {len(selected_pairs):,}")
    print(f"Unique sentences: {len(unique_sentences):,}")

    expected_unique_sentences = len(selected_pairs) * 2

    if len(unique_sentences) == expected_unique_sentences:
        print("Sentence-disjoint: yes")
    else:
        print("Sentence-disjoint: no")

    print("\nPairs by group:")

    print(
        selected_pairs["similarity_group"]
        .value_counts()
        .reindex(
            ["low", "medium", "high"],
            fill_value=0,
        )
        .to_string()
    )

    print("\nSelected group × entailment distribution:")

    print(
        pd.crosstab(
            selected_pairs["similarity_group"],
            selected_pairs["entailment_judgment"],
        )
        .reindex(
            index=["low", "medium", "high"],
            fill_value=0,
        )
        .to_string()
    )

    print("\nSelected relatedness statistics:")

    statistics = (
        selected_pairs
        .groupby(
            "similarity_group",
            sort=False,
        )["relatedness_score"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .reindex(
            [
                "low",
                "medium",
                "high",
            ]
        )
    )

    print(
        statistics.to_string(
            float_format=lambda value: f"{value:.3f}"
        )
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_pairs_path = (
        OUTPUT_DIR / "sick_selected_pairs.csv"
    )

    unique_sentences_path = (
        OUTPUT_DIR / "sick_unique_sentences.csv"
    )

    generation_metadata_path = (
        OUTPUT_DIR / "sick_generation_metadata.jsonl"
    )

    summary_path = (
        OUTPUT_DIR / "sick_selection_summary.csv"
    )

    eligible_pairs_path = (
        OUTPUT_DIR / "sick_eligible_pairs.csv"
    )

    excluded_pairs_path = (
        OUTPUT_DIR
        / "sick_excluded_transition_pairs.csv"
    )

    configuration_path = (
        OUTPUT_DIR
        / "sick_selection_configuration.json"
    )

    dataframe = load_sick_from_huggingface()

    dataframe = clean_and_filter_pairs(
        dataframe
    )

    dataframe = add_similarity_groups(
        dataframe
    )

    print_available_distribution(
        dataframe
    )

    eligible_pairs = dataframe[
        dataframe["similarity_group"] != "excluded"
    ].copy()

    excluded_pairs = dataframe[
        dataframe["similarity_group"] == "excluded"
    ].copy()

    eligible_pairs.to_csv(
        eligible_pairs_path,
        index=False,
        encoding="utf-8",
    )

    excluded_pairs.to_csv(
        excluded_pairs_path,
        index=False,
        encoding="utf-8",
    )

    selected_pairs = select_pairs(
        eligible_pairs=eligible_pairs,
    )

    selected_pairs, unique_sentences = (
        build_unique_sentence_table(
            selected_pairs=selected_pairs,
        )
    )

    selection_summary = create_selection_summary(
        selected_pairs=selected_pairs,
    )

    selected_pairs.to_csv(
        selected_pairs_path,
        index=False,
        encoding="utf-8",
    )

    unique_sentences.to_csv(
        unique_sentences_path,
        index=False,
        encoding="utf-8",
    )

    selection_summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    write_generation_metadata(
        unique_sentences=unique_sentences,
        output_path=generation_metadata_path,
    )

    save_configuration(
        output_path=configuration_path,
    )

    print_final_summary(
        selected_pairs=selected_pairs,
        unique_sentences=unique_sentences,
    )

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(f"Selected pairs:       {selected_pairs_path}")
    print(f"Unique sentences:     {unique_sentences_path}")
    print(f"TTS metadata:         {generation_metadata_path}")
    print(f"Selection summary:    {summary_path}")
    print(f"Eligible pairs:       {eligible_pairs_path}")
    print(f"Excluded transitions: {excluded_pairs_path}")
    print(f"Configuration:        {configuration_path}")


if __name__ == "__main__":
    main()
