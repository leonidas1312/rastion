from __future__ import annotations

from dataclasses import dataclass


TSPLIB_REFERENCE_URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf"


@dataclass(frozen=True, slots=True)
class TspReference:
    instance_id: str
    display_name: str
    best_known_distance: float
    reference_label: str
    reference_url: str

    def payload(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "display_name": self.display_name,
            "label": self.reference_label,
            "distance": float(self.best_known_distance),
            "url": self.reference_url,
        }


TSPLIB_REFERENCES: dict[str, TspReference] = {
    "berlin52": TspReference(
        instance_id="berlin52",
        display_name="Berlin52",
        best_known_distance=7542.0,
        reference_label="Best known",
        reference_url=TSPLIB_REFERENCE_URL,
    ),
    "ch150": TspReference(
        instance_id="ch150",
        display_name="ch150",
        best_known_distance=6528.0,
        reference_label="Best known",
        reference_url=TSPLIB_REFERENCE_URL,
    ),
    "a280": TspReference(
        instance_id="a280",
        display_name="a280",
        best_known_distance=2579.0,
        reference_label="Best known",
        reference_url=TSPLIB_REFERENCE_URL,
    ),
}


def get_tsplib_reference(instance_id: str | None) -> TspReference | None:
    if instance_id is None:
        return None
    return TSPLIB_REFERENCES.get(instance_id.strip().lower())


def gap_to_reference(score: float | None, reference_distance: float | None) -> float | None:
    if score is None or reference_distance is None or reference_distance <= 0:
        return None
    return ((float(score) - float(reference_distance)) / float(reference_distance)) * 100.0
