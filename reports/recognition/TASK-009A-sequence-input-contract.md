# TASK-009A — Core-28 Sequence-Recognition Input Contract

## Task

Design, implement, audit and **freeze** the exact transformation from the
finalized TASK-008 virtual-glove dataset to trustworthy variable-length ML
sequences with explicit masks, and to signer-independent LOSO PyTorch batches.

No classifier is built here. No LSTM, no optimizer, no hyperparameter search, no
training run. The deliverable is a contract, its implementation, and the evidence
that it holds over all 4,222 production sequences.

## Purpose

Every downstream experiment inherits whatever this layer decides. If the meaning
of a tensor slot is ambiguous — is that `0.0` a straight finger or a hand that
was never seen? — then every accuracy number computed on top of it is
uninterpretable. TASK-009A exists so that **no future model has to guess what any
tensor slot means**, and so that TASK-009B can change one config field instead of
re-deciding tensor semantics.

## Final TASK-008 dataset evidence

Carried forward from the TASK-008C finalization (commit `db36038`) and re-derived
here through the new code path rather than quoted:

| Property | Value |
| --- | --- |
| Sequences | 4,222 requested, 4,222 complete, 0 failed |
| Frames | 83,659 |
| Classes | 28 (SignIDs 0032–0059, `label_index` 0–27) |
| Signers | 01 (1,402), 02 (1,401), 03 (1,419) |
| Bend validity | 98.316% overall, 100.000% given a present hand |
| Spread validity | 77.842% overall, 79.175% given a present hand |
| IMU validity | 98.316% overall, 100.000% given a present hand |
| Tracking states present | OBSERVED 164,501 / MISSING 2,817 only |
| Neither-hand frames | 0 |
| Sensor layout | one identical layout across all 4,222 samples |

The production tree is **frozen and read-only** (backed up and verified under
TASK-008D, archive SHA-256 `6d1fa3cd…`). Nothing in TASK-009A writes to it.

## Input dataset

| Item | Path |
| --- | --- |
| Index | `datasets/manifests/karsl_core28_virtual_glove.csv` (4,222 rows) |
| Splits | `datasets/splits/karsl_core28_loso_s{01,02,03}.csv` |
| Sequences | `<run_root>/virtual_glove/<sample_id>/virtual_glove.npz` |
| Layout | `<run_root>/virtual_glove/<sample_id>/sensor_layout.json` |

`sample_id`, `sign_id`, `label_ar`, `label_index`, `signer_id`,
`official_partition` and the sequence path all come from **authoritative index
columns**. Nothing is parsed out of a directory name and nothing is inferred from
file ordering. The index is checked for duplicate sample IDs and for
`label_index` outside 0–27 at load time.

### Phase A inventory — what the finalized artifacts actually contain

Read from production, not assumed from the brief. `virtual_glove.npz` holds 19
arrays; the nine this contract consumes are:

| Array | Shape | dtype |
| --- | --- | --- |
| `frame_index` | `[T]` | int32 |
| `timestamp_seconds` | `[T]` | float64 |
| `bend_normalized` | `[T, 2, 5, 3]` | float32 |
| `bend_valid` | `[T, 2, 5, 3]` | bool |
| `spread_normalized` | `[T, 2, 4]` | float32 |
| `spread_valid` | `[T, 2, 4]` | bool |
| `imu_quaternion_wxyz` | `[T, 2, 4]` | float32 |
| `palm_imu_valid` | `[T, 2]` | bool |
| `tracking_state_code` | `[T, 2]` | int32 |

Metadata confirmed from `virtual_glove_meta.json`: `schema_version =
virtual_glove_v1`, `track_order = ["LEFT", "RIGHT"]`, `quaternion_order = wxyz`,
`normalized_ideal_sensor.authoritative_for_ml = true`, `fitted_to_dataset =
false`, `orientation_transform = "none - copied verbatim"`, and a validity policy
of `no_interpolation / no_forward_fill / no_invented_values /
no_cross_hand_copying`.

Not consumed by the primary contract, and why: `bend_angle_deg` /
`spread_angle_deg` (degree twins of the normalized arrays),
`bend_adc_12bit` / `spread_adc_12bit` (`authoritative_for_ml = false`),
`imu_rotation_matrix` (redundant with the quaternion),
`imu_angular_velocity_rad_s` (a derivative TASK-009B may add deliberately),
`source_valid_kinematics` / `source_valid_palm_frame` (upstream provenance).

**Measured on 60 random production sequences:** `palm_imu_valid` and
"all 15 bend channels valid" are each *exactly* equal to the pose-bearing
tracking state, 0 disagreements in 2,532 hand instances. The contract still keeps
hand presence and channel validity as separate masks, because that equality is a
property of this corpus and not of the schema.

## 4,222-sequence audit

Every indexed sequence was loaded through the production dataset class, with
`verify_layout="all"` so all 4,222 `sensor_layout.json` files were checked.

| Metric | Result |
| --- | --- |
| Total indexed | **4,222** |
| Successfully loaded | **4,222** |
| Rejected | **0** |
| All tensor values finite | **True** |
| Wall time | 4.4 s (whole corpus, single process) |

### Validity reconciliation with TASK-008C

| Channel group | TASK-009A tensor-level | TASK-008C dataset-level | Difference |
| --- | ---: | ---: | ---: |
| Bend | 2,467,515 / 2,509,770 = **0.9831638** | 0.9831638 | **0** |
| Spread | 520,974 / 669,272 = **0.7784189** | 0.7784189 | **0** |
| IMU | 164,501 / 167,318 = **0.9831638** | 0.9831638 | **0** |

Exact agreement to all reported digits. One denominator note: the IMU is **one
package per hand**, and the contract broadcasts its single validity flag across
the four quaternion columns. The audit therefore counts IMU *packages*
(`T × 2`), matching TASK-008C's `palm_imu_valid` denominator. Had it counted the
four broadcast columns instead, the numerator and denominator would both have
quadrupled and the fraction would be unchanged — the reconciliation is not
sensitive to that choice, but the reported counts are, so the package-level
denominator is the one used.

### Hand availability through the TASK-009A representation

| Case | Frames | Share |
| --- | ---: | ---: |
| Both hands present | 80,842 | 96.63% |
| LEFT present | 81,574 | 97.51% |
| RIGHT present | 82,927 | 99.13% |
| LEFT only | 732 | 0.87% |
| RIGHT only | 2,085 | 2.49% |
| Neither present | **0** | 0.00% |

Tracking states seen through the contract: code 1 (OBSERVED) 164,501; code 0
(MISSING) 2,817. No other code occurs.

### Sequence lengths through the representation

| Scope | count | min | p5 | p25 | median | p75 | p95 | max | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 4,222 | 9 | 12 | 15 | 20 | 24 | 29 | 70 | 19.82 |
| Signer 01 | 1,402 | 14 | 17 | 19 | 21 | 23 | 29 | 70 | 21.36 |
| Signer 02 | 1,401 | 10 | 11 | 13 | 13 | 15 | 17 | 45 | 13.74 |
| Signer 03 | 1,419 | 9 | 18 | 22 | 24 | 26 | 30 | 67 | 24.29 |
| train partition | 3,550 | 9 | — | — | 20 | — | — | 67 | 19.78 |
| test partition | 672 | 11 | — | — | 20 | — | — | 70 | 19.98 |

Per-class statistics are in the audit JSON. Lengths are preserved exactly: the
loader raises if a stored sequence's length disagrees with the index.

### Proof that a valid zero stays distinguishable

| Population | Count |
| --- | ---: |
| Valid observations in the corpus | 3,646,493 |
| Valid observations **exactly** equal to 0.0 | **0** |
| Valid observations within 1e-6 of 0.0 | **5** |
| Smallest \|value\| among valid observations | 3.81e-07 |
| Invalid slots holding the 0.0 placeholder | 201,821 |

Two things follow, and they are different claims:

1. **Architecturally**, the two are always separable, because separation is done
   by `feature_valid` and never by the numeric value. `value=0, valid=True` is a
   real zero-degree reading; `value=0, valid=False` is *no measurement*. This is
   enforced by unit test, using a synthetic sequence whose bend values genuinely
   are 0.0 while a hand is missing in the middle of it.
2. **Empirically**, in *this* corpus the collision never actually arises: not one
   of 3.6 million valid observations is exactly 0.0. But five of them sit within
   1e-6 of the fill value, which is exactly why the distinction is carried by a
   mask rather than by a sentinel value — a sentinel would have to be chosen
   outside the data range, and this data range comes within 4e-7 of zero.

## Feature families

| Family | Channels / hand | Source array | Validity source |
| --- | ---: | --- | --- |
| bend | 15 | `bend_normalized` | `bend_valid` (per channel) |
| spread | 4 | `spread_normalized` | `spread_valid` (per channel) |
| quaternion | 4 | `imu_quaternion_wxyz` | `palm_imu_valid` (per hand, broadcast ×4) |

| Feature set | Families | Channels / hand | **D** (both hands) |
| --- | --- | ---: | ---: |
| `bend_only` | bend | 15 | **30** |
| `bend_spread` | bend + spread | 19 | **38** |
| `full` | bend + spread + quaternion | 23 | **46** |

One configurable implementation serves all three. There is no duplicated loader
per ablation; `feature_set` selects which family blocks are concatenated.

## Exact feature order

Channel layout is **hand-major, then family, then the frozen within-family
order**. For `full` (D = 46):

| Index range | Meaning |
| --- | --- |
| 0–14 | LEFT bend, finger-major × chain-minor |
| 15–18 | LEFT spread, adjacent pairs |
| 19–22 | LEFT palm quaternion, w x y z |
| 23–37 | RIGHT bend |
| 38–41 | RIGHT spread |
| 42–45 | RIGHT palm quaternion |

Finger order `thumb, index, middle, ring, pinky`; chain order `proximal, middle,
distal`; spread pairs `thumb-index, index-middle, middle-ring, ring-pinky`;
quaternion `w, x, y, z`.

`channel_names(feature_set)` returns the full list, so `X[..., i]` is answerable
for every `i` — for example `X[..., 0]` is `LEFT/bend/thumb/proximal`,
`X[..., 15]` is `LEFT/spread/thumb-index`, `X[..., 19]` is
`LEFT/palm_quaternion/w`, `X[..., 45]` is `RIGHT/palm_quaternion/z`. Helpers
`hand_slice`, `family_slice` and `channel_index` expose the same layout
programmatically, and each is tested against the name list so the two can never
disagree.

For the reduced sets the same hand-major rule applies, so `bend_only` is
LEFT 0–14 / RIGHT 15–29 and `bend_spread` is LEFT 0–18 / RIGHT 19–37. A test
asserts each reduced set is a **prefix-consistent** subset of `full` within each
hand, so an ablation never silently permutes channels.

The order is not merely declared: `verify_sensor_layout` re-reads each sample's
`sensor_layout.json` and asserts the finger order, chain order, spread pairs and
every `array_index` still match. All 4,222 production layouts passed.

## Hand representation

`HAND_ORDER = ("LEFT", "RIGHT")` — physical identity, frozen. Slot 0 is always
the left hand and slot 1 always the right, and the order is **never** rearranged
by image x-position, detector confidence, which hand is present, or how many
channels are valid. Three separate tests cover those cases, including the
adversarial one where RIGHT carries strictly more valid channels than LEFT.

Hands are never collapsed into an unordered set: missing-LEFT and missing-RIGHT
are different states and stay in different tensor blocks.

## Missing-data representation

A timestep with one hand missing keeps the other hand's real measurements and
masks only the absent hand. **The timestep is never dropped**, because dropping
it would silently shorten the sequence and destroy frame alignment.

Neither-hand frames do not occur in this corpus (0 of 83,659), but the contract
handles them: all channels masked, all values filled, the frame retained, and
`hand_present` False for both hands. This is covered by a synthetic test rather
than left untested because the corpus happens not to exercise it.

## Mask representation

Masks are **separate tensors**, not concatenated into `values` by default.

Rationale: keeping them separate means `feature_dim` continues to mean exactly
what the contract says, ablation dimensions stay clean (30 / 38 / 46, not 60 / 76
/ 92), and a model that wants mask channels can obtain them with one tested
helper, `concat_features_and_masks(batch) -> [B, T, 2D]`. Concatenating by
default would have baked one architecture's preference into the data contract.

| Mask | Shape | dtype | True means |
| --- | --- | --- | --- |
| `feature_valid` | `[B, T, D]` | bool | this exact channel carries a real measurement |
| `hand_present` | `[B, T, 2]` | bool | this physical hand was reconstructed |
| `frame_valid` | `[B, T]` | bool | a real source frame, not batch padding |

**Every mask in this contract uses the same polarity: True = present/valid/real.**
No inverted twin of any mask is stored. The attention-style
`key_padding_mask(batch)` (True = ignore) is provided as a derivation, `~frame_valid`,
so the contract cannot ship two masks that disagree about which way round they are.

The three states the brief requires to stay distinct:

| State | frame_valid | feature_valid | value |
| --- | --- | --- | --- |
| Real observation | True | True | the measurement (may legitimately be 0.0) |
| Invalid / missing sensor | True | False | 0.0 placeholder |
| Batch padding | False | False | 0.0 placeholder |

A test asserts all three coexist in one batch and remain separable, and another
asserts that at a padded step `feature_valid`, `hand_present` and `frame_valid`
are *simultaneously* False — so a model consulting any one of them reaches the
same conclusion.

## Tracking-state decision

**Chosen: option A for the model input, option C for diagnostics.**

Hand presence is exposed as a 2-channel `hand_present` mask derived from the
frozen pose-bearing codes `{OBSERVED, AMBIGUOUS}`. The raw
`tracking_state_code [B, T, 2]` is carried in the batch as **metadata only** and
is never part of `values`.

Rejected: a 5-way one-hot per hand. It would add 10 channels per timestep of
which **three are identically zero across all 167,318 hand instances** in this
corpus — dead capacity, plus a silent representation change the moment a future
corpus does produce AMBIGUOUS or LIKELY_OCCLUDED. The full enum is still frozen
in the contract module and the raw codes still ride along, so nothing is lost;
the model simply does not receive constant-zero inputs.

Leakage note: `hand_present` is derived from the same tracking state that
determines channel validity, so it is largely redundant with `feature_valid` in
this corpus. It is kept because it is the semantically correct signal for
"physical hand absent" and is 2 channels rather than 46.

## Quaternion decision

**Default: `absolute`.** The verbatim TASK-008 palm quaternion — WXYZ, camera
frame, `w >= 0`, max unit-norm error 4.7e-08, zero `w < 0` violations across
164,501 valid observations. It is the most conservative default because it
applies no transformation at all, leaving the provenance chain from WiLoR through
TASK-005 unbroken.

**Alternative: `relative_first_valid`**, offered as a clean, tested ablation:

```
q_rel(t) = canonicalize( conjugate(q_ref) ⊗ q(t) )
```

where `q_ref` is the **first valid palm quaternion of that physical hand in that
sequence**, `⊗` is the Hamilton product on WXYZ quaternions, and `canonicalize`
negates the result when `w < 0`.

Properties, each covered by a test:

* **Strictly causal.** Frames earlier than `t_ref` have no valid orientation by
  construction, so no earlier timestep is ever defined using later information.
* **Per hand.** LEFT and RIGHT choose their references independently; a LEFT hand
  that only appears at t=2 does not shift RIGHT's reference away from t=0.
* **Identity at the reference.** `q_rel(t_ref) = [1, 0, 0, 0]` exactly.
* **Rotation-preserving.** The angle between `q_ref` and `q(t)` equals the angle
  between identity and `q_rel(t)`, verified numerically.
* **Never valid hand.** If a hand has no valid orientation anywhere in the
  sequence there is no reference; its quaternion channels stay masked and the
  stored values are left untouched rather than invented.
* **Sign.** The result is re-canonicalized to `w >= 0`, preserving the TASK-008
  convention. Since `q` and `-q` are the same rotation this is lossless as a
  rotation, but it does introduce a discontinuity when `w` crosses zero — stated
  here rather than hidden, and the same discontinuity already exists in the
  absolute representation.

No Euler-angle conversion is performed anywhere, for any reason.

TASK-009A does **not** claim which representation classifies better; that is a
controlled TASK-009B experiment. It provides a defensible default and a correct
alternative behind one config field.

## Normalization policy

**Identity.** TASK-008's physical normalization is used verbatim:
`bend_normalized = bend_angle_deg / 180.0`, `spread_normalized =
spread_angle_deg / 180.0`, both with a fixed a-priori divisor and
`fitted_to_dataset = false`. Quaternions are already unit-norm and dimensionless.

No global min/max scaling, no dataset-wide statistics, no per-signer or
per-subject offsets. **Nothing is fitted, therefore nothing can leak across a
LOSO boundary** — the strongest available guarantee, and the reason to prefer it
over a "safe" fold-wise standardization that would still need to be audited.

If TASK-009B later introduces standardization it must fit on train only, per
fold, persist the parameters, apply the frozen values to validation and test, and
test for leakage. The contract records `normalization: "task008_physical"` so an
experiment that changes it cannot silently disagree with this report.

## Variable-length policy

**Original length preserved.** No fixed window, no cropping, no temporal
resampling, no interpolation. Lengths span 9–70 frames and every one of them is
carried through unchanged; the loader raises if a stored length disagrees with
the index.

Batching is right-padded dense tensors plus `lengths` (int64, CPU, unsorted) and
`frame_valid`. This is directly compatible with
`pack_padded_sequence(..., enforce_sorted=False)`; a round-trip test packs a
batch, unpacks it, and asserts both the values and the lengths come back
identical.

Rejected alternatives and why: a fixed window would truncate signer 03 far more
than signer 02 (means 24.3 vs 13.7 frames); resampling to a common length would
destroy exactly the duration information the Phase J audit shows is
signer-correlated, replacing an honest confound with a hidden one.

## Padding policy

Padding is written as `padding_fill_value = 0.0` and marked False in
`frame_valid`, `feature_valid` and `hand_present` at once. `frame_index` is
filled with `-1` at padded positions so a padded step cannot be mistaken for
frame 0.

A test collates a sequence alone and again inside a longer batch and asserts the
real region is bit-identical, so padding never perturbs a real frame.

Measured padding overhead on representative 32-sample training batches (natural
index order, no length bucketing):

| Batch | `values` shape | lengths | real steps | padded steps | padding |
| --- | --- | --- | ---: | ---: | ---: |
| S01 train, first 32 | `[32, 17, 46]` | 10–17 | 426 | 118 | 21.7% |
| S02 train, first 32 | `[32, 36, 46]` | 23–36 | 940 | 212 | 18.4% |
| S03 train, first 32 | `[32, 36, 46]` | 23–36 | 940 | 212 | 18.4% |

Around 20% of cells are padding at batch size 32 without bucketing. That is an
efficiency figure, not a correctness one — the mask makes it harmless — and
TASK-009B can reduce it with length-bucketed sampling if it wants to.

## LOSO contract

Verified against the frozen split files, then re-verified through the loader:

| Fold | Held-out | train | validation | test | Total | Train/val signers | Classes per role | Leakage |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| S01 | 01 | 2,372 | 448 | 1,402 | 4,222 | 02, 03 | 28 / 28 / 28 | none |
| S02 | 02 | 2,373 | 448 | 1,401 | 4,222 | 01, 03 | 28 / 28 / 28 | none |
| S03 | 03 | 2,355 | 448 | 1,419 | 4,222 | 01, 02 | 28 / 28 / 28 | none |

Counts match the frozen TASK-008B expectations exactly. `load_fold` **raises**
rather than warns on: a held-out sample in train or validation, a foreign signer
in test, a sample assigned two roles, a sample in the split but not the index, a
`fold` column naming a different fold, or incomplete coverage of the index. A
fold with leakage is not loadable, because a warning would still produce a
number that someone would eventually quote.

Fold contents are sorted by `sample_id`, so they are deterministic regardless of
file or dict ordering — tested by loading the same fold from a reversed record
list and comparing.

Validation is signer-mixed by design: each fold's 448 validation samples come
from the two non-held-out signers, so validation measures in-signer fit while
test measures cross-signer generalization. That asymmetry is inherited from the
frozen TASK-008B splits, not introduced here.

## Signer-duration analysis

Duration differs materially by signer, so this was audited directly rather than
assumed benign.

**Can sequence length alone identify the signer?**

| Target | Oracle accuracy from length alone | Majority baseline |
| --- | ---: | ---: |
| Signer (3-way) | **78.07%** | 33.61% |
| Letter class (28-way) | **7.60%** | 3.79% |

The oracle is an optimistic in-sample upper bound: it gives every distinct length
its own majority label. Read together, these two rows are the finding — **length
carries a great deal of signer information and almost no letter information**. A
recognizer that keys on duration is therefore learning signer identity, not the
sign. This is a genuine confound in the corpus and it is reported rather than
engineered away.

**Per-fold duration shift:**

| Fold | Train mean | Test mean | Test/train ratio | Train p5–p95 | Test inside that band |
| --- | ---: | ---: | ---: | --- | ---: |
| S01 | 19.05 | 21.36 | 1.12 | 12–29 | 96.2% |
| S02 | 22.83 | 13.74 | **0.60** | 17–29 | **5.2%** |
| S03 | 17.55 | 24.29 | **1.38** | 12–26 | 75.0% |

**Fold S02 is the severe case.** Only 5.2% of its test sequences fall inside the
central 90% of the training length distribution — the durations are close to
disjoint. Any S02 result must be read as cross-signer generalization *under
substantial duration shift*, and it would be wrong to average S02 with S01 as if
the three folds posed the same problem.

What the contract does about it: nothing to the data, deliberately. Sequences are
not altered to hide the effect. What it does guarantee is that the confound
cannot leak in through the plumbing:

* length is not a feature — `values` contains no duration channel;
* padded steps are masked in all three masks simultaneously, so padding cannot
  act as a length proxy through an unmasked sum;
* nothing is normalized using held-out signer statistics, because nothing is
  normalized at all;
* `lengths`, `signer_ids` and per-sample `sample_ids` ride along in every batch,
  so TASK-009B can stratify, bucket, or run a duration-ablation without touching
  the dataset code.

Residual risk TASK-009B must handle: an RNN's final hidden state is implicitly
length-dependent even with correct packing. Masked mean-pooling, or a
duration-matched control experiment, is the way to test whether the recognizer is
reading the letter or the clock.

## Data-loader architecture

```
recognition/data/
  contract.py           frozen orders, feature sets, config, contract document
  sequence_dataset.py   index loader, NPZ loader, layout check, feature assembly
  collate.py            variable-length batching and mask helpers
  loso.py               fold loading with enforced signer independence
```

* **Index loader** — `load_index` reads the finalized CSV, rejects duplicate
  sample IDs and out-of-range `label_index`.
* **Safe NPZ loader** — `load_sequence_arrays` uses `allow_pickle=False`
  throughout, checks the nine required arrays exist, checks every shape against
  `T`, checks the three masks are boolean, and converts any corrupt-file
  exception into a clear `SequenceContractError`.
* **Feature selector** — one implementation; `feature_set` picks the family
  blocks.
* **Mask construction** — per-channel validity, per-hand presence, per-frame
  padding.
* **Quaternion policy** — `absolute` or `relative_first_valid`, config-selected.
* **LOSO selection** — frozen split files, leakage enforced by exception.
* **Collation** — right-padded batches, `pack_padded_sequence`-ready.
* **Metadata** — `sample_ids`, `sign_ids`, `labels_ar`, `signer_ids`, `labels`,
  `lengths`, `frame_index` travel with every batch, outside the numeric tensor.
* **Efficiency** — sequences are read lazily per item; `preload=True` caches the
  whole corpus in RAM (the glove tree is 133.7 MiB). Runtime validation is
  *lightweight*: shapes, dtypes and masks. **No SHA-256 re-hashing per epoch** —
  TASK-008C already established integrity over all 4,222 sequences with a full
  hash pass, and repeating it every epoch would cost hours for no information.
  Layout verification defaults to the first sample (`verify_layout="first"`);
  the full audit runs it on all 4,222.

`VirtualGloveSequenceDataset` implements the mapping protocol rather than
subclassing `torch.utils.data.Dataset`, which is all a `DataLoader` needs and
keeps the contract importable and testable without torch.

## Tensor schema

For `feature_set="full"`, batch size `B`, longest sequence `T_max` in the batch:

| Key | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `values` | `[B, T_max, 46]` | float32 | features; 0.0 where not a real measurement |
| `feature_valid` | `[B, T_max, 46]` | bool | True = this channel is a real measurement |
| `hand_present` | `[B, T_max, 2]` | bool | True = that physical hand was reconstructed |
| `frame_valid` | `[B, T_max]` | bool | True = real frame, False = batch padding |
| `lengths` | `[B]` | int64 (CPU) | original sequence lengths, unsorted |
| `labels` | `[B]` | int64 | `label_index` in 0–27 |
| `tracking_state_code` | `[B, T_max, 2]` | int16 | diagnostics only, never a feature |
| `frame_index` | `[B, T_max]` | int64 | source frame number; −1 at padding |
| `sample_ids`, `sign_ids`, `labels_ar`, `signer_ids` | `[B]` | list[str] | metadata |
| `feature_set`, `quaternion_policy` | — | str | the contract this batch was built under |

`D` is 30 for `bend_only`, 38 for `bend_spread`, 46 for `full`; `feature_valid`
always matches `values` exactly in shape.

### Worked example

Real batch from the S01 fold's training split, `feature_set="full"`, batch size 8
(reproduced by the smoke script):

```
values          : (8, 17, 46)  torch.float32
feature_valid   : (8, 17, 46)  torch.bool
hand_present    : (8, 17, 2)   torch.bool
frame_valid     : (8, 17)      torch.bool
lengths         : (8,)         torch.int64   [17, 11, 14, 11, 13, 13, 12, 12]
labels          : (8,)         torch.int64   [0, 0, 0, 0, 0, 0, 0, 0]
signers         : ['02']       (held-out signer 01 is correctly absent from train)

padded timesteps            : 33          = 8*17 - sum(lengths)
zeros that ARE measurements : 0           (feature_valid = True)
zeros that are placeholders : 1809        (feature_valid = False)
```

Row 1 has length 11, so `frame_valid[1, 11:]` is False, `values[1, 11:]` is all
0.0, and `frame_index[1, 11:]` is −1. Reading `values[1, 3, 15]` gives
`LEFT/spread/thumb-index` at source frame 3 of that sequence — a real number if
`feature_valid[1, 3, 15]` is True, and a placeholder to be ignored otherwise.

## Ablation support

TASK-009B changes one field, not any dataset code:

```python
SequenceInputConfig(feature_set="bend_only")                                  # D = 30
SequenceInputConfig(feature_set="bend_spread")                                # D = 38
SequenceInputConfig(feature_set="full")                                       # D = 46
SequenceInputConfig(feature_set="full", quaternion_policy="relative_first_valid")
```

Nonsensical combinations are rejected at construction: `bend_only` with a
non-default quaternion policy raises, because that config would silently do
nothing. Mixing items built under different feature sets in one batch raises
`BatchError` rather than producing a wrongly-shaped tensor.

## Tests

`tests/test_task009a_sequence_input.py` — **53 tests**, all synthetic, no GPU, no
production reads, no model. Coverage against the required list:

| # | Requirement | Covered by |
| --- | --- | --- |
| 1 | Exact feature ordering | `test_exact_feature_order_and_dimensions`, `test_hand_and_family_slices_agree_with_names` |
| 2 | LEFT/RIGHT never swaps | 3 tests incl. the unequal-valid-channel case |
| 3 | `label_index` stays 0–27 | `test_label_index_outside_the_frozen_range_is_rejected` |
| 4 | Sequence length preserved | `test_index_length_disagreement_is_rejected`, round-trip test |
| 5 | 9-frame sequence | `test_extreme_lengths_nine_and_seventy` |
| 6 | 70-frame sequence | same |
| 7 | Variable-length batch | `test_padding_mask_matches_lengths_exactly` |
| 8 | Padding-mask correctness | same + `key_padding_mask` equality |
| 9 | Padding does not alter real frames | `test_padding_never_alters_real_frames` |
| 10 | Valid zero vs invalid fill | `test_real_zero_measurement_is_distinguishable_from_invalid_fill` |
| 11 | Missing LEFT only | `test_left_right_never_swaps_when_only_right_is_present` |
| 12 | Missing RIGHT only | `test_left_right_never_swaps_when_only_left_is_present` |
| 13 | Neither-hand frame | `test_neither_hand_frame_is_handled_not_rejected` |
| 14 | Spread invalid, bend valid | `test_spread_invalid_while_bend_stays_valid` |
| 15 | Quaternion valid/invalid | `test_relative_policy_on_a_never_valid_hand_has_no_reference` |
| 16 | WXYZ ordering | `test_wxyz_ordering_is_preserved` |
| 17 | Relative identity at reference | `test_relative_policy_makes_the_reference_frame_the_identity` + 4 more |
| 18 | No held-out signer leakage | `test_held_out_signer_leaking_into_train_is_rejected` |
| 19 | Deterministic fold contents | `test_fold_contents_are_deterministic` |
| 20 | Malformed NPZ rejected clearly | `test_malformed_npz_is_rejected_clearly` |
| 21 | Wrong schema rejected | missing-array, wrong-shape and non-boolean-mask tests |
| 22 | Sensor-layout reorder detected | bend-reorder and spread-pair-reorder tests |
| 23 | Ablation dimensions correct | `test_ablation_batches_have_the_declared_dimension` |
| 24 | Batch metadata aligned | `test_metadata_stays_aligned_with_tensor_rows` |
| 25 | No silent NaN→zero without a mask | `test_nan_is_never_silently_converted_without_a_mask` |

Full suite: **548 passed, 0 failed** (TASK-008C baseline 495 + 53 new; no existing
test weakened or modified). `compileall`: 0 errors.

## Known limitations

1. **Duration is a strong signer cue** (78.1% oracle vs 33.6% chance) and carries
   almost no class information (7.6% vs 3.8%). The contract exposes this rather
   than hiding it; testing whether a recognizer exploits it is TASK-009B's job.
2. **Fold S02 is a near-disjoint duration regime** — 5.2% of its test lengths lie
   inside the training p5–p95 band. Its result is not comparable with S01's
   without saying so.
3. **Only two tracking states occur.** The pipeline handles all five, but the
   corpus cannot show whether AMBIGUOUS or LIKELY_OCCLUDED are handled sensibly
   *in practice*; only synthetic tests cover them.
4. **`hand_present` is nearly redundant with `feature_valid`** in this corpus,
   because palm/bend validity coincides exactly with the pose-bearing state. It
   is retained for semantic clarity, at a cost of 2 channels of metadata.
5. **Spread missingness is class-dependent** (55%–100% by class). A model that
   ignores `feature_valid` on the spread block will silently train on 20.8%
   placeholder zeros for a present hand.
6. **Padding overhead is ~20%** at batch size 32 without length bucketing.
   Correctness is unaffected; throughput is not optimized here.
7. **Three signers only.** LOSO over three folds gives a high-variance estimate;
   a per-fold number is not a confidence interval.
8. **The run root is an absolute external path.** The index stores only
   run-root-relative paths, and the run root must be supplied by the caller.
9. **No angular-velocity, ADC, rotation-matrix or degree channels** are in the
   primary contract. They remain in the frozen NPZ files and can be added by a
   documented contract revision, not by an undeclared reinterpretation.

## TASK-009B interface

```python
from recognition.data import (
    SequenceInputConfig, VirtualGloveSequenceDataset,
    load_index, load_fold, make_collate_fn,
)

records = load_index("datasets/manifests/karsl_core28_virtual_glove.csv")
fold    = load_fold("datasets/splits", held_out_signer="01", records=records)
config  = SequenceInputConfig(feature_set="full", quaternion_policy="absolute")

train = VirtualGloveSequenceDataset(fold.roles["train"], RUN_ROOT, config)
loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=make_collate_fn(config))
```

TASK-009B consumes `configs/recognition/task009a_sequence_input_v1.json` for
tensor semantics instead of re-deriving them, and must not change feature order,
hand order, mask polarity or normalization without a new contract version.

## Reproducibility commands

```bash
cd /home/hatim/Graduation-Project-Simulation-task009a

# Regenerate the frozen contract JSON (a test asserts it matches the code).
PYTHONPATH=. python scripts/write_task009a_contract.py

# Full 4,222-sequence audit through the new data contract. No training.
PYTHONPATH=. python scripts/run_task009a_audit.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --verify-layout all \
  --output reports/recognition/TASK-009A-audit.json

# Non-training smoke example: one fold, one DataLoader, one batch.
PYTHONPATH=. python scripts/task009a_smoke_example.py \
  --run-root /home/hatim/graduation-project-runs/task008-core28-full \
  --held-out-signer 01 --feature-set full --quaternion-policy absolute

# Tests.
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q recognition scripts tests
```

Machine-readable statistics: `reports/recognition/TASK-009A-audit.json`.
Frozen contract: `configs/recognition/task009a_sequence_input_v1.json`.

## Final verdict

| Item | Result |
| --- | --- |
| Contract version | `task009a_sequence_input_v1` |
| Sequences audited | 4,222 |
| Rejected | **0** |
| Feature dimensions | bend_only 30, bend_spread 38, full 46 |
| Tensor schema | `values [B,T,D] f32`, `feature_valid [B,T,D] bool`, `hand_present [B,T,2] bool`, `frame_valid [B,T] bool`, `lengths [B] i64`, `labels [B] i64` |
| Quaternion policy | `absolute` default; `relative_first_valid` available and tested |
| Variable-length strategy | original lengths preserved; right-padded + masks + `pack_padded_sequence` |
| Validity reconciliation | bend/spread/IMU match TASK-008C exactly, difference 0 |
| LOSO | 3 folds, counts exact, no leakage, 28 classes per role |
| Sequence lengths | 9–70, median 20, mean 19.82 |
| Signer-duration finding | length identifies signer at 78.1% vs 33.6% chance; S02 fold nearly duration-disjoint |
| Valid-zero distinction | proven; 0 valid observations equal 0.0, 5 within 1e-6 |
| Tests | 548 passed, 0 failed |
| compileall | 0 errors |

**TASK-009A SEQUENCE INPUT CONTRACT FROZEN — READY FOR BASELINE TRAINING**
