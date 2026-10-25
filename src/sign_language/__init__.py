"""Isolated American Sign Language word recognition on WLASL100.

Pipeline: download WLASL videos -> MediaPipe Holistic landmarks per frame ->
normalise (shoulder-centred, shoulder-width scaled, 32-frame resample) ->
cache as (32, 75, 3) arrays -> train a 2-layer bidirectional GRU -> evaluate
top-1/top-5 against two non-deep baselines.
"""

__version__ = "0.1.0"
