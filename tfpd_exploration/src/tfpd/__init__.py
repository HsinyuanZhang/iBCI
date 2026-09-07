"""Task-Frame Population Decoder package (Stage-0 synthetic scope)."""

from src.tfpd.bilinear_readin import BilinearTaskFrameDecoder
from src.tfpd.population_vector import LearnedPopulationVectorDecoder
from src.tfpd.synth import SyntheticSession, generate_session

__all__ = [
    "BilinearTaskFrameDecoder",
    "LearnedPopulationVectorDecoder",
    "SyntheticSession",
    "generate_session",
]
