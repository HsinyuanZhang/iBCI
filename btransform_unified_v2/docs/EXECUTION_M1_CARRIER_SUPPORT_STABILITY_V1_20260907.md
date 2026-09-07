# M1 M10 carrier support stability

The diagnostic reconstructs each visible held-out-calibration carrier from its
NWB with the production public-calibration reader, frozen all-source NNMF
basis, source normalizer, NNLS activation projection, and production per-unit
ridge encoding. It never reads query values for the carrier. The three
chronological whole-trial M10 supports are `20121004`, `20121017`, and
`20121024`.

For each session it requires a byte-exact/allclose reproduction of
`m1_projadd_depth2_series.encode_heldout_calib_carrier`, then holds its M10
carrier as reference. It draws ten without-replacement whole-trial supports at
M5 and M8 with seeds 101--110. Full M10 is evaluated once. The receipt stores
trial indices, normalized Frobenius error, cosine similarity, NNLS/ridge design
coverage and condition number, carrier-solve time, and separately measured
frozen B3 Sfix E0 construction time.

The result is CPU-only and the carrier code reads calibration EMG and spikes
only. Query window counts are retained solely to bind the exact development
surface and to demonstrate that M10 support is separate from query arrays.
