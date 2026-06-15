import logging
from pathlib import Path
from pdgen import generate_diagram

from dm_detector.extraction import CandidateExtraction

from dm_detector.location import LFinderDetector
from dm_detector.location import LineSegment
from dm_detector.location import LPattern

from dm_detector.location import DataMatrixValidator
from dm_detector.location import ValidationResult

from dm_detector.location import DashedBorderDetector
from dm_detector.location import DataMatrixLocation

from dm_detector.geometry import BorderFitter
from dm_detector.geometry import PreciseLocation

from dm_decoder.grid_estimation import GridEstimator

from dm_decoder.sampling import ModuleSampler

from dm_decoder.mapping import UtahMapper

from dm_decoder.decoding import DataMatrixDecoder

from dm_detector.pipeline import DetectionResult
from dm_detector.pipeline import DataMatrixPipeline

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    poza_output = Path("diagrama_clase_dmc.png")
    text_output = Path("diagrama_clase_dmc.txt")

    generate_diagram(poza_output, text_output)