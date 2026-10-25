from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def load_embedding(
    analyzer: FaceAnalysis,
    image_path: Path,
) -> np.ndarray:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = analyzer.get(image)

    if len(faces) != 1:
        raise ValueError(
            f"Expected exactly one face in {image_path}, " f"but detected {len(faces)}."
        )

    embedding = faces[0].normed_embedding

    if embedding is None:
        raise ValueError(f"No embedding produced for: {image_path}")

    return np.asarray(
        embedding,
        dtype=np.float32,
    )


def similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(
        np.dot(
            first,
            second,
        )
    )


def collect_images(
    dataset_path: Path,
) -> dict[str, list[Path]]:
    identities = {}

    for identity_dir in sorted(dataset_path.iterdir()):
        if not identity_dir.is_dir():
            continue

        images = [
            path
            for path in sorted(identity_dir.iterdir())
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]

        if images:
            identities[identity_dir.name] = images

    return identities


def print_summary(
    name: str,
    scores: list[float],
) -> None:
    values = np.asarray(
        scores,
        dtype=np.float32,
    )

    print(f"\n{name}")
    print(f"  count: {len(values)}")
    print(f"  minimum: {values.min():.4f}")
    print(f"  maximum: {values.max():.4f}")
    print(f"  mean: {values.mean():.4f}")


def main() -> None:
    parser = ArgumentParser(
        description=("Evaluate genuine and impostor " "face-embedding similarities.")
    )

    parser.add_argument(
        "dataset",
        type=Path,
    )

    args = parser.parse_args()

    analyzer = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CPUExecutionProvider",
        ],
    )

    analyzer.prepare(
        ctx_id=-1,
        det_size=(640, 640),
    )

    identities = collect_images(args.dataset)

    embeddings = {}

    for identity, image_paths in identities.items():
        print(f"\nLoading identity: {identity}")

        embeddings[identity] = []

        for image_path in image_paths:
            embedding = load_embedding(
                analyzer,
                image_path,
            )

            embeddings[identity].append(
                (
                    image_path,
                    embedding,
                )
            )

            print(f"  loaded: {image_path.name}")

    genuine_scores = []
    impostor_scores = []

    print("\nGENUINE COMPARISONS")

    for identity, samples in embeddings.items():
        for first, second in combinations(
            samples,
            2,
        ):
            first_path, first_embedding = first
            second_path, second_embedding = second

            score = similarity(
                first_embedding,
                second_embedding,
            )

            genuine_scores.append(score)

            print(
                f"  {identity}: "
                f"{first_path.name} vs "
                f"{second_path.name} "
                f"= {score:.4f}"
            )

    print("\nIMPOSTOR COMPARISONS")

    identity_names = list(embeddings.keys())

    for first_name, second_name in combinations(
        identity_names,
        2,
    ):
        first_samples = embeddings[first_name]
        second_samples = embeddings[second_name]

        for first_path, first_embedding in first_samples:
            for second_path, second_embedding in second_samples:
                score = similarity(
                    first_embedding,
                    second_embedding,
                )

                impostor_scores.append(score)

                print(
                    f"  {first_name}/{first_path.name} "
                    f"vs "
                    f"{second_name}/{second_path.name} "
                    f"= {score:.4f}"
                )

    if genuine_scores:
        print_summary(
            "GENUINE SUMMARY",
            genuine_scores,
        )

    if impostor_scores:
        print_summary(
            "IMPOSTOR SUMMARY",
            impostor_scores,
        )

    print("\nTHRESHOLD EVALUATION")

    print("Threshold |     FAR |     FRR |     TAR")

    print("----------+---------+---------+---------")

    for threshold in (
        0.20,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ):
        evaluate_threshold(
            genuine_scores,
            impostor_scores,
            threshold,
        )


def evaluate_threshold(
    genuine_scores: list[float],
    impostor_scores: list[float],
    threshold: float,
) -> None:
    false_rejects = sum(score < threshold for score in genuine_scores)

    false_accepts = sum(score >= threshold for score in impostor_scores)

    genuine_count = len(genuine_scores)
    impostor_count = len(impostor_scores)

    false_reject_rate = false_rejects / genuine_count

    false_accept_rate = false_accepts / impostor_count

    true_accept_rate = 1.0 - false_reject_rate

    print(
        f"{threshold:>9.2f} | "
        f"{false_accept_rate:>7.2%} | "
        f"{false_reject_rate:>7.2%} | "
        f"{true_accept_rate:>7.2%}"
    )


if __name__ == "__main__":
    main()
