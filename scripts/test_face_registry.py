from argparse import ArgumentParser
from pathlib import Path

from app.recognition.insightface_embedder import InsightFaceEmbedder
from app.recognition.registry import FaceRegistry

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
}


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

    registry = FaceRegistry.load(args.registry)

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
