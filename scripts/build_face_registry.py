from argparse import ArgumentParser
from pathlib import Path

from app.recognition.insightface_embedder import (
    InsightFaceEmbedder,
)
from app.recognition.registry import FaceRegistry

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def main() -> None:
    parser = ArgumentParser(
        description=("Build a persistent Smart Vision " "face registry.")
    )

    parser.add_argument(
        "source",
        type=Path,
    )

    parser.add_argument(
        "output",
        type=Path,
    )

    args = parser.parse_args()

    embedder = InsightFaceEmbedder()

    registry = FaceRegistry()

    sample_count = 0

    for identity_dir in sorted(args.source.iterdir()):
        if not identity_dir.is_dir():
            continue

        identity = identity_dir.name

        print(f"Registering: {identity}")

        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            embedding = embedder.embed_image(image_path)

            registry.add_sample(
                identity,
                embedding,
            )

            sample_count += 1

            print(f"  added {image_path.name}")

    registry.save(args.output)

    print(f"\nSaved {sample_count} face samples")

    print(f"Registry: {args.output}")


if __name__ == "__main__":
    main()
