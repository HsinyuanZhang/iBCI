# Frozen protocol: exploratory H1 M=4 EB-carrier fold-0 matched system pilot

This is a single-fold exploratory system pilot, not a decoding claim or an
override of the failed M=4 empirical-Bayes absolute attachment CPU gate. The
CPU gate failed its >=4/6 split-pair cosine condition (2/6); this pilot cannot
retroactively call that gate passed and cannot support a standalone result.

The sole target date is lexicographic fold 0, `19250101`, with both same-date
recordings as target and all other five dates as source. Both matched arms use
seed 42, released learning rate `5e-5`, batch 32, the original H1 architecture
and window 700, `calibration_n_trials=4`, and fixed terminal epoch 50 without
checkpoint/validation selection. Source windows and the ordered random
contiguous M=4 calibration-start schedule are one shared manifest. The same
four chronological eval-valid TrialNum trials are used for SPINT identity and
the carrier in every sample.

The wrapper is shared: a RNG-neutral literal-zero bias-free `nn.Parameter`
mapping 4 to 700 is inserted exactly after original `fc_id_out` and before the
original source-plus-identity combination. The matched base keeps it frozen and
literal zero; the joint arm trains it. No concat, FiLM, gate, nonlinear side
network, extra attention, or topology difference is permitted. The shared
initial-state hash must match before training. Original H1 `dynamic_dropout=true`
is retained exactly: the wrapper executes the original Python `random.uniform`
and torch dropout sequence. Base/joint parity is assessed only after resetting
both RNG states; later optimization trajectories are expected to diverge.

The carrier is the fixed M=4 empirical-Bayes estimator bound to the immutable
raw-M4 and confidence-shrinkage receipts. Source plans are date-excluded;
legal source contiguous M=4 blocks are precomputed/cached under the global
non-electrode prior and analytic shrinkage. No additional z-score, learned
estimator, or strength choice is added. Target query is only later held-in
calib trials; every 700-bin query history starts at/after the first bin of the
fifth selected valid TrialNum trial. Minival is forbidden.

On one terminal joint checkpoint only, `Full`, literal `Zero4`, deterministic
complete row-shuffle, and deterministic within-each-support-trial
label-rotation-refit carriers are inference-only interventions. They must not
mutate checkpoint/model state. The fixed exploratory expansion gate is the
conjunction: Full R2 > 0, Full-matched-base > 0, Full-Zero4 > 0,
Full-RowShuffle > 0, and Full-LabelShuffle > 0. One failed clause means STOP;
do not expand dates or tune the pilot.

This preparation step may run CPU tests/preflight only. It must not launch GPU,
create fake terminal checkpoints, or open formal held-out/minival/EvalAI.
