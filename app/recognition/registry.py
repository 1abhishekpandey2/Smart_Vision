from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np


class MatchState(StrEnum):
    NO_MATCH = "no_match"
    CANDIDATE = "candidate"
    STRONG_MATCH = "strong_match"


@dataclass(frozen=True)
class FaceMatch:
    identity: str | None
    similarity: float | None
    second_identity: str | None
    second_similarity: float | None
    margin: float | None
    state: MatchState


class FaceRegistry:
    def __init__(
        self,
        candidate_threshold: float = 0.35,
        strong_threshold: float = 0.50,
    ) -> None:
        if not 0.0 <= candidate_threshold < strong_threshold <= 1.0:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= candidate_threshold < strong_threshold <= 1"
            )

        self._candidate_threshold = candidate_threshold
        self._strong_threshold = strong_threshold

        self._embeddings: dict[
            str,
            list[np.ndarray],
        ] = {}

    @classmethod
    def load(
        cls,
        path: Path,
        candidate_threshold: float = 0.35,
        strong_threshold: float = 0.50,
    ) -> "FaceRegistry":
        if not path.exists():
            raise FileNotFoundError(f"Face registry not found: {path}")

        registry = cls(
            candidate_threshold=candidate_threshold,
            strong_threshold=strong_threshold,
        )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            identities = data["identities"]
            embeddings = data["embeddings"]

            if len(identities) != len(embeddings):
                raise ValueError("Face registry data is inconsistent")

            for identity, embedding in zip(
                identities,
                embeddings,
                strict=True,
            ):
                registry.add_sample(
                    identity=str(identity),
                    embedding=embedding,
                )

        return registry

    def save(
        self,
        path: Path,
    ) -> None:
        identities: list[str] = []
        embeddings: list[np.ndarray] = []

        for identity, samples in self._embeddings.items():
            for sample in samples:
                identities.append(identity)
                embeddings.append(sample)

        if not embeddings:
            raise ValueError("Cannot save an empty face registry")

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            identities=np.asarray(
                identities,
                dtype=np.str_,
            ),
            embeddings=np.stack(embeddings).astype(np.float32),
        )

    def add_sample(
        self,
        identity: str,
        embedding: np.ndarray,
    ) -> str:
        normalized = self._normalize(embedding)
        identity = identity.strip()
        identity_key = identity.casefold()

        stored_identity = next(
            (
                existing_identity
                for existing_identity in self._embeddings
                if existing_identity.casefold() == identity_key
            ),
            identity,
        )

        self._embeddings.setdefault(
            stored_identity,
            [],
        ).append(normalized)

        return stored_identity

    def match(
        self,
        query_embedding: np.ndarray,
    ) -> FaceMatch:
        if not self._embeddings:
            return FaceMatch(
                identity=None,
                similarity=None,
                second_identity=None,
                second_similarity=None,
                margin=None,
                state=MatchState.NO_MATCH,
            )

        query = self._normalize(query_embedding)

        identity_scores: list[tuple[str, float]] = []

        for identity, samples in self._embeddings.items():
            identity_similarity = max(
                float(np.dot(query, sample)) for sample in samples
            )

            identity_scores.append(
                (
                    identity,
                    identity_similarity,
                )
            )

        identity_scores.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        best_identity, best_similarity = identity_scores[0]

        second_identity = None
        second_similarity = None
        margin = None

        if len(identity_scores) >= 2:
            (
                second_identity,
                second_similarity,
            ) = identity_scores[1]

            margin = best_similarity - second_similarity

        if best_similarity < self._candidate_threshold:
            return FaceMatch(
                identity=None,
                similarity=best_similarity,
                second_identity=second_identity,
                second_similarity=second_similarity,
                margin=margin,
                state=MatchState.NO_MATCH,
            )

        if best_similarity < self._strong_threshold:
            return FaceMatch(
                identity=best_identity,
                similarity=best_similarity,
                second_identity=second_identity,
                second_similarity=second_similarity,
                margin=margin,
                state=MatchState.CANDIDATE,
            )

        return FaceMatch(
            identity=best_identity,
            similarity=best_similarity,
            second_identity=second_identity,
            second_similarity=second_similarity,
            margin=margin,
            state=MatchState.STRONG_MATCH,
        )

    @staticmethod
    def _normalize(
        embedding: np.ndarray,
    ) -> np.ndarray:
        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Embedding cannot have zero magnitude")

        return vector / norm
