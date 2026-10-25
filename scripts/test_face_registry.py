from argparse import ArgumentParser
from pathlib import Path

from app.recognition.insightface_embedder import InsightFaceEmbedder
from app.recognition.registry import FaceRegistry

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


def load_registry(
    registry_path: Path,
    embedder: InsightFaceEmbedder,
) -> FaceRegistry:
    registry = FaceRegistry()

    for identity_dir in sorted(registry_path.iterdir()):
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

            print(f"  added {image_path.name}")

    return registry


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument(
        "registry",
        type=Path,
    )

    parser.add_argument(
        "query",
        type=Path,
    )

    args = parser.parse_args()

    embedder = InsightFaceEmbedder()

    registry = load_registry(
        args.registry,
        embedder,
    )

    query_embedding = embedder.embed_image(args.query)

    match = registry.match(query_embedding)

    print("\nMATCH RESULT")

    print(f"Identity: {match.identity}")

    print(f"Similarity: {match.similarity}")

    print(f"Second identity: {match.second_identity}")

    print(f"Second similarity: {match.second_similarity}")

    print(f"Margin: {match.margin}")

    print(f"State: {match.state}")


if __name__ == "__main__":
    main()
