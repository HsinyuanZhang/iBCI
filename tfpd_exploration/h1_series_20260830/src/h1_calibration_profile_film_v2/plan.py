"""V2 route identity and exact V1 incident authority."""

SCHEMA = "h1_calibration_profile_film_v2"
RESULT_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v2"
V1_ROOT_RELATIVE = "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v1"
V1_BODIES = {
    "attempt.json": "1dd84e148fd03d3da472f3340662adcfbf36e7d3db366e7477075595cacf9d8f",
    "training_19250108.json": "e4691925908f8e7663155bee780cfeda3c2058da854bb7f36804239030af87a5",
    "checkpoint_19250108_ep-film.pt": "25851d58844486126f67e97b5b84ef15728e8ca571dc515dd7e2fdc0b388920b",
    "checkpoint_19250108_lp-film.pt": "298abb04f2018a09daf0aafbc5a845c7427a1bda774649f21ad8a8f14270e165",
    "failure.json": "51265f15650c6f4044d80966df94a87011c0bee79a2a9d19712feaf507b93796",
}
DESIGN_RELATIVE = "tfpd_exploration/h1_series_20260830/docs/DESIGN_H1_CALIBRATION_PROFILE_FILM_V2_20260904.md"
WORKORDER_RELATIVE = "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CALIBRATION_PROFILE_FILM_V2_20260904.md"
INCIDENT_RELATIVE = "tfpd_exploration/h1_series_20260830/docs/RESULT_H1_CALIBRATION_PROFILE_FILM_V1_FAILURE_20260904.md"
ANCHOR_R2_TOLERANCE = 1.0e-7
INCIDENT_SHA256 = "34e3389d4fb7add13defa739b9cedbb49c2bfb7bb93a170f8902def2e6118275"
V1_OUTER_DATE = "19250108"
V1_FAILURE_ERROR = "ses-19250108T110520: LP-ZERO prediction anchor drift"
V1_FILM_STATE_SHA256 = {
    "EP-FILM": "60eb1161d9621b0830d226e0eaf07aefb0ee1f56eb72d42880e95aa518bc6b58",
    "LP-FILM": "fba420f403b248ee0a6fe6c1898f704dc6ae4be0be03d3a92324771afe979736",
}
