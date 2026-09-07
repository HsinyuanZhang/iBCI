# PACD P0-to-V4 terminal handoff checklist

Date: 2026-08-31

Status: pre-terminal review template. This document is outside the active P0
execution closure. It authorizes no source edit, capability, root reservation,
data/checkpoint open, CUDA initialization, or launch.

## 1. Purpose

This checklist turns the P0-to-V4 transition into a descriptor-only audit.
Every box must be supported by immutable current-state evidence. A missing,
indirect, reconstructed-from-memory, or merely plausible fact is a failure to
admit V4.

The governing authorities remain:

| Authority | SHA-256 |
|---|---|
| `WORKORDER_PACD_P1_P2_ADMISSION_MULTIGPU_V4_20260831.md` | `34a67357d66c36816457252e82d5ac7dcecf34340dafff1f372c67571dc09cd3` |
| `AUDIT_PACD_V4_PARALLEL_ROUTE_ISOLATION_20260831.md` | `7c2c27697329e13f53fc2dacd7981cec2f24b88f550ab1af38c266c4cc2dcc52` |
| `AUDIT_PACD_V4_SHARED_LIFECYCLE_SEAM_20260831.md` | `10bc3e81385e43bb6f9f79ebbdf13db8f5100a37de532d471f5792ecc83c9f8c` |

Historical P0 root:

`tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42`

Historical P0 closure:

`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`

## 2. Pre-terminal prohibition

Until a valid terminal exists, all of the following remain required:

- [ ] no file in the active 48-file P0 closure has been edited;
- [ ] no additive V4 producer package exists;
- [ ] no V4 authority or result root exists;
- [ ] no P1/P2 capability has been issued;
- [ ] no P1/P2 data, checkpoint, CUDA, forward, backward or update has occurred;
- [ ] monitoring remains read-only and does not alter process affinity,
  priority, GPU binding, workers, root or bytes.

The first five boxes deliberately remain unchecked while P0 is active.

## 3. Exact successful topology

The accepted terminal root must contain exactly 58 bodies and their 58
canonical sidecars: 116 leaves total, with no failure body and no extra leaf.

```text
attempt.json
launch.json
source_authority.json
epoch000.json ... epoch047.json
epoch044.pt ... epoch047.pt
swa_final4.pt
manifest.json
terminal.json
```

For every body and sidecar:

- [ ] the leaf is a regular file and not a symlink;
- [ ] mode is exactly `0444`;
- [ ] the body SHA-256 equals the literal in its canonical basename sidecar;
- [ ] the sidecar contains the exact body basename;
- [ ] held-directory identity is stable from first validation through handoff;
- [ ] no missing, extra, renamed, replaced-parent or replaced-root leaf exists.

## 4. Frozen prefix already observed

The following bodies were independently rehashed before terminal. They are
prefix evidence only; every value must be revalidated through held descriptors
after terminal.

| Body | Observed SHA-256 |
|---|---|
| `attempt.json` | `d5beac402022a32f25de641a27b6fe3cd9bb5aa7accc65a0bc2ab7a5488a92ff` |
| `launch.json` | `1594056712ee76c54fea89b92ccabb945c0d3d00201180a279fe1c27756cd2f9` |
| `source_authority.json` | `3379b0423cfaad9082aa8594b3b443351aa9c9d051f0e1aed6713768f3bdca85` |
| `epoch000.json` | `664880e1af5505d622f2c698cc30619514940a7c7034c5d37f2130d244c9d10c` |
| `epoch001.json` | `d6fdd384b4d689ddfd15565564ea654b560046e217e00c353e0b48ca2129634a` |
| `epoch002.json` | `19d2e3b7d697a17eac71484571cb70d40e9cd0be535d4af357598b3ae93a7e13` |
| `epoch003.json` | `a8f7b327266100553864e185c82a181e4e92c46a4161ce25a5ee2829f52e5b16` |
| `epoch004.json` | `3c87f251cf9b05e7d37979bf0be37a94a2ad68bd8939d7aadc083fa50ecc8193` |
| `epoch005.json` | `8606e609548a7e28a75111ad81decc0e36962ca4bba6ee5fc5df557c3a22266d` |
| `epoch006.json` | `c784feda333db280c065373e7b2e3c857f9d331738bdbb9a96929b2e409dc424` |
| `epoch007.json` | `9d57394264be9843e5904dee35636dccb03219065a5d9ea90f824fd99fef884c` |
| `epoch008.json` | `44fd689eec395d2219d5b67411770002aa13be98ede38fa4a8fba032ad9a9eb3` |
| `epoch009.json` | `a15c50443222544f915a8a0d3e19c89596006752f4dce7a08457a66d5f6f20c1` |
| `epoch010.json` | `52162245de5c4509eb8b65cecba4cebfc5e1f975b3a7854d906b0f06f1d2438b` |
| `epoch011.json` | `308d48c4eb1e34e8dbdccddbe5cb3acdd36540437fc7333e6b7b5594d56c3470` |
| `epoch012.json` | `33ec3b47b8950380ae87ea65e26852c3413b77dff12b6e8c0d3257541602d00a` |
| `epoch013.json` | `ecb9e6b245e27d40d9d2f063736b97f15b8b11e981fb83bc59b27b6857663ea4` |
| `epoch014.json` | `6688d88fca7ec860a656bb6f30e6d088128da8a4fda1d54022914b8c88833493` |
| `epoch015.json` | `1f75e4993da29173e985317f23f2cc777a37965feba86e0ae8059863ec80ba9c` |
| `epoch016.json` | `8e2edf79a8a28620294e15369a150905186d560f90ac6afe75529c779fc348c3` |
| `epoch017.json` | `2474cf4c76b4c2e3a734995ed4be5c841c13c417ea9802a3edeaeff1887d1062` |
| `epoch018.json` | `8608a4eeae3675ddd9603fd877cabdbcf3eb6d3945e2edd8ed204b4462bd5d21` |
| `epoch019.json` | `e935df4aef8c8106dec09a7f6a66ad15a60026f70067777b4a1c1aff5decb646` |
| `epoch020.json` | `1dfa4cd8561ef2846798d0fa6f7c8c96316838a24361e2a74ed1372eab4630c4` |
| `epoch021.json` | `cb5fd3c9e475f17506d31db784a6a09f21e802d9ec78a2ba90b7b30ded139fa1` |
| `epoch022.json` | `417d04deb9de6c6fc99aafd8334ba645099d9cc3a775d1fac6bc78eecd8dd625` |
| `epoch023.json` | `471cb788abc26fc5717f2d255ce9945e01daf872b3863086efe687603fe92a06` |
| `epoch024.json` | `e70438ebee18c199c4f2dd2a87708faad3fa559f30ad1c65af329acc9925d9ae` |
| `epoch025.json` | `72f2b91fe93f0620889aa67936e6606f4864374b099b385639dcd66d9dccb271` |
| `epoch026.json` | `822839fd6c47a8f64d05a577c3f491785ea2cd5004de54fe9afbd2a706cddca6` |
| `epoch027.json` | `619f72cc396f26776d532cc50adf0194ed12628089cb1abf41573ea7a8be57bb` |
| `epoch028.json` | `70ad0191efef379d0d540f7ba872868503013c31210f437dabb6f632b4a07824` |
| `epoch029.json` | `64e094e21d8c5c5b1ac0510e0abe016b47b8a1e7e966d360523b183d49c8b3c2` |
| `epoch030.json` | `2d1768f278325cd4efd1aa712738495808ba042397d79350c2178ebc246bc903` |
| `epoch031.json` | `70e183a511d71806435bb6c52f5a62f12c9da3c6c264ad6679c57c8ada3a4f8d` |
| `epoch032.json` | `355062a43de031be56baa0df118c1ece9e155fa1984aade69e8e3b8f266adaca` |
| `epoch033.json` | `0e4c2426ea5da6acb8245fc2c47d5db76b8cba9f916976401f026a0488f93c7a` |
| `epoch034.json` | `65d9fc22deee46ea2df64a2278dc3549b8501c160fdf64feadfd8698ab57e450` |
| `epoch035.json` | `c366a1265b2bea203141b2dd25c883849ecf24581de816de8b96e11520cc1891` |
| `epoch036.json` | `9062a1e79c8a616c791d6fe0d6f393be545eafd7be30a99df2df08201de16ef1` |
| `epoch037.json` | `f6e26afe88151b7d76981a50fbeca63c466329b53d3f015b50d1352df257b982` |
| `epoch038.json` | `26f7e0ddd89cc373046e58f56ba2209b87f12c7665ecdd876a4f2e26e3a5cce6` |
| `epoch039.json` | `31f4863ce9d7b49bb9785b2cdb0624baa91b30b091a8d57bb016e6985050f5fa` |
| `epoch040.json` | `ec060eab89239c084d6bf88c3a1512881fb96e7e2935da48ea6862c4056e3683` |
| `epoch041.json` | `e470c040e4ee7b490ff0348f6dd9d71da1241408692f540a6bb4d35c46c6389a` |
| `epoch042.json` | `80a69557cabcb2deb3b531c8ebfdebe2a2a2ce6f0afe9a6b07560ab131621d52` |
| `epoch043.json` | `452902fb61179f149e52a2b535507f5f49f6b73861fc0a5815d9ae9dcebc8485` |
| `epoch044.json` | `16ba92799d0e0c9b54fca68256df28fa0e5f2a2673ee0aedc12f8b607364a55f` |
| `epoch045.json` | `d1f5d0e903508d3834ab7305edde4ecde487e2247c4c6afec00eaa8e88a4bd1b` |
| `epoch046.json` | `50bd62d0949acb4d900f84bf2f9b8e2d128b7022e205e7095b0e81f19085dea0` |

The first final-window checkpoint is now independently descriptor-verified:

| Checkpoint body | SHA-256 |
|---|---|
| `epoch044.pt` | `f01bd4bc83e72f12ab1430a01dbfdc45b2732ee0506a5239a89761336ebb0139` |
| `epoch045.pt` | `efaa259bb47b7c1a5503373f5b377c457ba17184059563d78bbd90b933619ed7` |
| `epoch046.pt` | `1c6e753bd9c92a3b0b97a73edd3b72e25c10a6344c21787db5fba385e5848998` |

The checkpoint audit covered regular-file type, single link, mode `0444`,
canonical sidecar basename and body SHA only. No model tensor was loaded or
interpreted. Checkpoint 47, SWA, manifest and terminal remain pending.

## 5. Attempt, launch and source authority

- [ ] attempt schema is `pacd_matched_full_training_v3_attempt`, status is
  `ATTEMPT_PUBLISHED`, cell/arm is exact `p0`, seed is 42, batch is 32,
  workers are zero, and the historical root is exact;
- [ ] attempt closure contains exactly 48 files and equals the historical
  closure above;
- [ ] launch links the exact attempt and source authority body digests;
- [ ] launch/final implementation closures are identical;
- [ ] source authority binds the exact 27-session source-only roster;
- [ ] target access is false and val/test authorities are empty;
- [ ] behavior, side-feature, normalizer, T4 and initial-state authorities are
  exact;
- [ ] sampler window-index and batched-index digests are exact and immutable;
- [ ] lower V2 failure predecessor and its exact held topology are unchanged,
  including failure body SHA-256
  `c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0`.

## 6. Epoch contract

For every epoch `e` in `0..47`:

- [ ] `epoch == e` and steps equal `33,925`;
- [ ] cumulative steps equal `(e + 1) * 33,925`;
- [ ] learning-rate endpoints and inherited 48-epoch law are exact;
- [ ] loss summaries and finite min/mean/max fields are finite;
- [ ] Adam/model/materialized-parameter finiteness is true;
- [ ] P0 prediction mismatch and identity mismatch are zero;
- [ ] RNG, paired-mask, prefix, finiteness and sampler violations are zero;
- [ ] decoder zero-gradient steps are zero;
- [ ] encoder accepted-zero count equals encoder zero count;
- [ ] every accepted encoder zero uses
  `all_units_dropped_valid_zero` and retained-unit count zero;
- [ ] positive encoder and decoder coverage is nonzero;
- [ ] sentinel coordinates are exactly `0`, `1`, `16,962`, `33,924`;
- [ ] sampler-order evidence equals source authority before and after epoch;
- [ ] epoch descriptors form the exact ordered terminal chain.

Final arithmetic:

- [ ] epoch count is 48;
- [ ] optimizer-step total is `1,628,400`;
- [ ] terminal `target_access` is false, and the source authority retains the
  exact source-only roster with empty validation/test lists; the historical
  V3 schema does not invent separate target backward/optimizer/update
  counters.

## 7. Checkpoint, SWA and manifest contract

- [ ] checkpoint bodies are exactly epochs 44, 45, 46 and 47, in order;
- [ ] every checkpoint descriptor links the matching epoch model and optimizer
  state digest without deserializing its tensor body during audit;
- [ ] terminal and manifest carry the same four checkpoint body digests;
- [ ] SWA uses exactly the final-four checkpoint window `[44,45,46,47]`;
- [ ] SWA arithmetic and inherited builder identity are exact;
- [ ] SWA strict fresh load succeeds under the producer proof;
- [ ] repeated actual-shaped eval/no-grad forward outputs are bitwise equal;
- [ ] SWA output is finite;
- [ ] dynamic dropout call count is zero during the proof;
- [ ] strict-loaded, state-before and state-after SHA-256 values are equal;
- [ ] manifest checkpoint and SWA links are exact;
- [ ] no SWA, manifest or checkpoint path is inferred from a glob.

## 8. Terminal and process release

- [ ] terminal schema is `pacd_matched_full_training_v3_terminal`;
- [ ] status is `PACD_FULL_TRAINING_COMPLETE` and no `failure.json` exists;
- [ ] terminal arm/identity is exact P0 and terminal target access is false;
- [ ] terminal links exact attempt, launch, source authority, all 48 epochs,
  four checkpoints, SWA and manifest body digests;
- [ ] terminal progress is exact:
  `epochs_published == 48`, `checkpoints_published == 4`, and
  `swa_published is true`;
- [ ] launch/final closure equality is true;
- [ ] producer PID and its shell/tmux owner have exited naturally;
- [ ] physical GPU0 has no stale producer compute PID;
- [ ] physical GPU1 has no conflicting compute PID;
- [ ] fixed P1 CPUs `0-3,16-19` and P2 CPUs `4-15,20-31` have no in-scope
  owner;
- [ ] the frozen host-memory gate passes at capability issuance: available
  memory is at least 16 GiB, memory PSI `some avg10 <= 0.1`, and memory PSI
  `full avg10 == 0.0`; CPU and IO PSI are recorded separately as descriptive
  coexistence observations and are not invented as V4 receipt fields;
- [ ] the two canonical prospective P1/P2 result roots are absent, and their
  parent/root identities are held before reservation; V4 defines no separate
  authority root, and authorization remains an opaque in-process capability.

Historical SUA release is not sufficient for these boxes; resources must be
measured freshly after P0 terminal.

## 9. Handoff payload

Only after Sections 3--8 pass, freeze one literal payload containing:

- P0 held root and parent device/inode/name identities;
- attempt, launch, source-authority, terminal, manifest and SWA body SHA-256;
- ordered 48 epoch body SHA-256 values;
- ordered four checkpoint body SHA-256 values;
- historical P0 closure and exact V2 failure witness;
- terminal resource and process-release observations;
- this audit result and its own source SHA-256.

The payload is review evidence for V4 implementation. It is not itself an
execution capability.

## 10. Admission decision

```text
P0 terminal graph valid:        PENDING
P0 process/resources released:  PENDING
V4 shared edit authorized:      NO
V4 code implementation:         FORBIDDEN PRE-TERMINAL
P1/P2 capability issuance:      FORBIDDEN
P1/P2 launch:                   FORBIDDEN
```

After every checklist item passes, change only the decision block in a
successor completed audit. Do not rewrite this pre-terminal record or relax a
failed criterion.
