import argparse
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm


EXPECTED_NDIM = 3


def validate_file(file_path: Path) -> tuple[int, int, int]:
    with safe_open(
        str(file_path),
        framework="pt",
        device="cpu",
    ) as features:
        keys = set(features.keys())

        if "hidden_states" not in keys:
            raise KeyError(
                f"{file_path}: missing hidden_states"
            )

        hidden_shape = tuple(
            features.get_slice(
                "hidden_states"
            ).get_shape()
        )

        if len(hidden_shape) != 3:
            raise ValueError(
                f"{file_path}: expected [L,T,d], "
                f"received {hidden_shape}"
            )

        number_of_layers, sequence_length, hidden_size = (
            hidden_shape
        )

        if "special_tokens_mask" in keys:
            mask_shape = tuple(
                features.get_slice(
                    "special_tokens_mask"
                ).get_shape()
            )

            if mask_shape != (sequence_length,):
                raise ValueError(
                    f"{file_path}: mask shape {mask_shape} "
                    f"does not match T={sequence_length}"
                )

        if "input_ids" in keys:
            input_shape = tuple(
                features.get_slice(
                    "input_ids"
                ).get_shape()
            )

            if input_shape != (sequence_length,):
                raise ValueError(
                    f"{file_path}: input_ids shape "
                    f"{input_shape} does not match T="
                    f"{sequence_length}"
                )

    return (
        number_of_layers,
        sequence_length,
        hidden_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", type=Path, help="Root directory of extracted features")
    BASE_DIR = parser.parse_args().features_dir
    files = list(BASE_DIR.rglob("*.safetensors"))

    if not files:
        print(f"No .safetensors files found under {BASE_DIR}")
        return

    for file_path in tqdm(files, desc="Checking files", unit="file"):
        validate_file(file_path)

    print(f"Successfully checked {len(files)} files.")


if __name__ == "__main__":
    main()
