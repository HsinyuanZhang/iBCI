"""Submit-time CPU budget. Short random-input profiles are not release evidence."""

from __future__ import annotations

# Official falcon-challenge evaluator runs held-out then held-in sequentially.
# S2 581938 failed with "Execution time limit exceeded" after ~119 minutes and
# empty stdout. Do not reuse the old single-slot 13 ms/bin * 2-wave formula as
# a pass. Measure under container CPU quota, official batch size, and the full
# evaluator path before registering any later M1/H1/M2 image.
# https://github.com/snel-repo/falcon-challenge/blob/main/falcon_challenge/evaluator.py

OFFICIAL_SEQUENTIAL_SPLITS = ("heldout", "heldin")
PROFILE_BATCH_IS_NOT_TRAIN_BATCH = True
TRAIN_MICROBATCH = 8
PROFILE_USED_BATCH = 32
S2_TIMEOUT_SUBMISSION = 581938
S2_STDERR = "Execution time limit exceeded"
