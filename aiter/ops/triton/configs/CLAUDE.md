# Triton kernel configs — rules for automated edits

Scope: **every tuned JSON file under `aiter/ops/triton/configs/`.** Read this
before adding, moving, renaming, or tuning one, or before touching a loader
under `utils/` (`gemm_config_utils.py`, `moe_config_utils.py`,
`conv_config_utils.py`, `mhc_config_utils.py`, `tuned_config_utils.py`).

Out of scope, do not touch without an explicit request: the two AOT directories
built at runtime — see the end of §6. Every checked-in family (GEMM, MOE, conv,
attention, GMM, MHC) is covered here.

There is one layout. Every family lives in the nested
`<arch>/<backend>/<op>/<d_type>/` tree, every loader resolves through the same
probe, and the flat, arch-prefixed layout is gone: `configs/` now holds this
file and the `<arch>/` directories, nothing else. A config outside the nested
layout does not resolve.

Two non-negotiables:

1. **Tuning values live in JSON, never in Python.** No `setdefault`, no inline
   dict literals, no arch-conditional constants, no hardcoded fallback configs.
   If a value is missing, fix the JSON.
2. **New configs go in the nested layout.** It is the only layout any resolver
   reads.

`GEMM-AFP4WFP4` (gfx950 triton, gfx950/gfx1250 gluon) is the worked reference
— copy its shape when in doubt.

---

## 1. Layouts

### The layout — every op, every arch

```
configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json
configs/<arch>/<backend>/<op>/<d_type>/<CONFIG_NAME>-<suffix>.json
```

| Segment     | Values                                                    |
| ----------- | --------------------------------------------------------- |
| `<arch>`    | `gfx942`, `gfx950`, `gfx1100`, `gfx1151`, `gfx1200`, `gfx1201`, `gfx1250` |
| `<backend>` | `triton` or `gluon`                                        |
| `<op>`      | `gemm`, `moe`, `conv`, `attention`, `gmm`, `mhc`, `fusions` — the op family, which is not always the wrapper's category folder (MHC's wrapper lives in `fusions/`, its configs in `mhc/`) |
| `<d_type>`  | `config_name.lower().replace("-", "_")` — `GEMM-AFP4WFP4` → `gemm_afp4wfp4`. The transform lives in `gemm_config_utils._dtype_dir()` |
| filename    | **no arch prefix** — the arch is the directory. The default is literally `DEFAULT.json`; specialized files keep the `<CONFIG_NAME>-` stem |

```
configs/gfx950/triton/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx950/triton/gemm/gemm_afp4wfp4/GEMM-AFP4WFP4-N=8192-K=8192.json
configs/gfx950/gluon/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx1250/gluon/gemm/gemm_afp4wfp4/DEFAULT.json
configs/gfx950/triton/moe/moe_fp8_w8a8/DEFAULT.json
configs/gfx1201/triton/conv/conv_3x3_nhwc/DEFAULT.json
configs/gfx950/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=7168.json
```

A handful of `.gitkeep` files hold directories open: `<arch>/gluon/moe/` on
gfx950 and gfx1250 is genuinely empty, and the placeholders under
`gfx1250/triton/gemm/`, `gfx950/triton/moe/` and `gfx1250/triton/moe/` outlived
the directories filling up. Keep them all — do not rename or delete a
`.gitkeep`.

### The flat layout is gone

Configs used to live flat and arch-prefixed: `configs/gemm/<arch>-<NAME>.json`
(plus `configs/gemm/gluon/`), `configs/moe/<arch>-MOE-<dtype_str>.json`,
`configs/conv/<arch>-CONV-<KERNEL>.json`, `configs/hstu_attn/`, and loose
`<arch>-<NAME>.json` files for attention, GMM and MHC at the top of `configs/`.
None of those directories or files exist any more and no resolver probes for
them, so a file dropped there is dead weight — it will not be found and nothing
will warn you.

Enumerate what actually ships rather than trusting a listing in this document:
`git ls-tree -r --name-only HEAD aiter/ops/triton/configs/`

---

## 2. Resolution order — `resolve_config_dir()`

Every op resolves through one shared probe,
`gemm_config_utils.py::resolve_config_dir()`. It picks a directory by probing
candidates for the family's *default* config file (`DEFAULT.json`) in order
and taking the first hit; specialized files are then read from that same
directory. `get_gemm_config()` wraps it for GEMM; conv, attention, GMM, MHC
and MOE call it directly and parse their own schemas (§5).

**`backend=None`**:

1. `configs/<arch>/triton/<op>/<d_type>/DEFAULT.json`
2. `configs/<arch>/gluon/<op>/<d_type>/DEFAULT.json`

**`backend="triton"|"gluon"`** (what every non-GEMM caller passes):

1. `configs/<arch>/<backend>/<op>/<d_type>/DEFAULT.json`

If nothing matches, the last candidate is used and the missing-default
assertion fires there, naming the nested path the file should have had.

Two parameters exist for the edges. `arch=` overrides the running
architecture, for a loader that retries under another arch when the running
one ships no tuned configs — MHC's gfx942 fallback is the only user.
`legacy_dir=` appends flat, arch-prefixed candidates; nothing passes it any
more and no flat directory survives for it to find.

Consequences to keep in mind:

- **A directory is chosen as a unit.** The unit is the family's `<d_type>/`
  directory. Splitting a family across two candidate directories silently
  drops the specialized files in whichever one loses the probe — in particular,
  adding a `DEFAULT.json` for a family whose specialized files live elsewhere
  hides them. Move a family wholesale or not at all. Worse: a
  `<d_type>/` directory with specialized files but **no `DEFAULT.json` is
  invisible** — the probe keys only on `DEFAULT.json`, so it falls through to
  the next candidate and ignores everything in the directory.
- **`backend=None` prefers `triton` over `gluon`.** On an arch with only a
  gluon default (currently gfx1250 `GEMM-AFP4WFP4`), lookup falls through to
  gluon. Adding `configs/gfx1250/triton/gemm/gemm_afp4wfp4/DEFAULT.json` later
  would change which file gfx1250 resolves to — verify that is intended.
- Results are cached twice: `functools.lru_cache` on the full argument
  tuple, plus a per-path cache of parsed JSON
  (`utils/core.py::load_config_json`) that also caches negative results
  (missing files). Adding a config file at runtime therefore has no effect;
  restart the process (tooling may call `load_config_json.cache_clear()`
  instead).

Direct-path loaders bypass the resolver's directory probe. Grep for
`f"{AITER_TRITON_CONFIGS_PATH}/..."` before moving anything —
`gluon/gemm_a8w8_blockscale.py` needs the whole config dict for its tile
filtering, so it calls `resolve_config_dir()` directly instead of
`get_gemm_config()` — it is on the shared probe and needs no path edits.
`gluon/gemm_a8w8.py` and `gluon/gemm_afp4wfp4.py` go through
`get_gemm_config(backend="gluon")`. `utils/tuned_config_utils.py` builds its
own nested path rather than probing — it takes `op` and `backend` from the
caller, so there is only ever one candidate — but the layout it builds is the
same one.

---

## 3. GEMM config file contents

Required top-level shape:

```json
{
  "M_LEQ_64":   { "...": "..." },
  "M_GEQ_4096": { "...": "..." },
  "any":        { "...": "..." }
}
```

- `M_LEQ_x` is searched ascending over `STANDARD_M_BOUNDS =
  (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)`, then
  `M_GEQ_x` descending, then `any`. A caller may override with
  `bounds=(...)`, which must be strictly increasing positive ints.
- `any` must exist unless every reachable `M` is covered by an explicit bound.
- The deprecated `{"large": ..., "small": ...}` shape must not be introduced.
- A `KeyError` at lookup time means no bound matched — usually a missing `any`.

Each `M_*` entry carries at minimum:

```
BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim,
cache_modifier, NUM_KSPLIT
```

`add_default_gemm_config_params()` backfills `NUM_KSPLIT=1` and
`cache_modifier=None` as a last resort, and `compute_splitk_params()` derives
`SPLITK_BLOCK_SIZE` and may clamp `BLOCK_SIZE_K` / `NUM_KSPLIT`. Neither is a
license to omit keys.

`get_gemm_config()` returns `(config, is_tuned)`. `is_tuned` is `True` only when
a specialized (N/K-, B-, or `specialized_filename`-keyed) file was hit, `False`
for the default file or `any`. Do not discard it.

### The JSON is the only place tuning values live

A `_get_config()` should do nothing but call `get_gemm_config()` and return:

```python
def _get_config(M: int, N: int, K: int):
    return get_gemm_config("GEMM-AFP4WFP4", M, N, K)
```

`_triton_kernels/gemm/basic/gemm_afp4wfp4.py` carried a block of `setdefault`
calls and it was deleted — it masked incomplete config files with values nobody
had tuned, and made the effective config un-inspectable from the JSON.

---

## 4. GEMM naming

| Kind              | Path under `<arch>/<backend>/gemm/`                              |
| ----------------- | ---------------------------------------------------------------- |
| Default           | `gemm_a16w16/DEFAULT.json`                                        |
| N/K specialized   | `gemm_a16w16/GEMM-A16W16-N=256-K=7168.json`                       |
| Batched (B, N, K) | `batched_gemm_a16w16/BATCHED_GEMM-A16W16-B=4-N=1024-K=4096.json`  |
| Custom suffix     | `fused_gemm_afp4wfp4_a16w16/FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json` |

The `<d_type>` directory name is `config_name.lower().replace("-", "_")`.
Dashes, underscores, and case all fold together, so new config names must stay
distinct under that transform — `GEMM-FOO-BAR` and `GEMM-FOO_BAR` would collide.

Config-name patterns: `GEMM-A{x}W{y}`, `BATCHED_GEMM-A{x}W{y}`,
`FUSED-GEMM-{op}`, `FF-A{x}W{y}-fused`; variant suffixes
`_PRESHUFFLED`, `_BLOCKSCALE`.

**`K` in AFP4WFP4 filenames is the logical K, i.e. `2 * K_bytes`.** The kernel
does `K = 2 * K` before calling `get_gemm_config`. Tuning output that names
files by the packed byte width will never be found.

---

## 5. The non-GEMM ops

Everything outside GEMM finds its directory the same way — `resolve_config_dir()`
(§2), or the identical nested path built by `get_tuned_kernel_config()` — and
then parses its own file. Path resolution is all these loaders share with
`get_gemm_config()`: no common schema, and no common `is_tuned` signal.

| Op | Loader | `<d_type>` directories | Selection within the file |
| -- | ------ | ---------------------- | ------------------------- |
| `conv` | `utils/conv_config_utils.py::get_conv_config` | `conv_1x1`, `conv_3x3_nhwc`, `conv_3x3_cblocked`, `conv_general`, `conv_wino_f4x3_{input,gemm,output}` | `shapes[<shape_key>]` → `M_LEQ_x` → `any` |
| `attention` | seven modules under `_triton_kernels/attention/`; `chunk_delta_attn` instead goes through `utils/tuned_config_utils.py::get_tuned_kernel_config` | `mha`, `extend_attention`, `mla_decode_rope`, `leanattn`, `hstu_attn_fwd`, `hstu_attn_bwd`, `chunk_delta_attn` | per module |
| `gmm` | `_triton_kernels/gmm.py::get_config` | `gmm` | variant (`gmm`/`ptgmm`/`nptgmm`) → `default` / `accumulate` |
| `mhc` | `utils/mhc_config_utils.py` | `mhc_fused_sinkhorn`, `mhc_post` | C-specialized file, then `M_LEQ_x` → `any` |
| `moe` | four loaders, below | `moe_<dtype_str>`, `a8w4`, `a4w4`, `moe_routing_sigmoid_topk1` | one schema per loader, below |

Mind the directory names: the redundant `-DEFAULT` the flat filenames carried
(`gfx942-MHA-DEFAULT.json`) is not part of `<d_type>` — the file inside is
already `DEFAULT.json`, so the directory is `mha`, not `mha_default`.

MHC is the one loader that resolves a *foreign* arch's directory: an arch with
no tuned MHC configs retries under gfx942 through the `arch=` parameter of §2,
so both `mhc_fused_sinkhorn` directories have to stay complete.

### MOE

Four independent loaders resolve `op="moe"` with `backend="triton"` and read
`DEFAULT.json` out of the directory they get back, each parsing its own schema:

| Loader | Config directory | Schema |
| ------ | ---------------- | ------ |
| `utils/moe_config_utils.py::get_moe_configs` | `<arch>/triton/moe/moe_<dtype_str>/` | `small_M` / `medium_M` / `large_M` |
| `moe/moe_op_gemm_a8w4.py::_get_a8w4_dispatch` | `<arch>/triton/moe/a8w4/` | `bm<block_m>_n<N>_k<K>` |
| `moe/moe_op_gemm_a4w4.py::_get_a4w4_dispatch` | `<arch>/triton/moe/a4w4/` | `bm<block_m>_n<N>_k<K>_<bucket>` |
| `_triton_kernels/moe/moe_routing_sigmoid_top1_fused.py` | `<arch>/triton/moe/moe_routing_sigmoid_topk1/` | `N16` → `small` / `medium` / … |

`<dtype_str>` comes from `get_config_dtype_str()`: `DEFAULT`, `FP8_W8A8`,
`INT8_W8A16`, `INT8_W8A8`, `INT4_W4A16`, `MX_FP4`; lowercased into the
directory name, so `MOE-FP8_W8A8` resolves to `moe_fp8_w8a8/`.

`small_M` / `medium_M` / `large_M` split on `M_THRESHOLD_SMALL = 256` and
`M_THRESHOLD_MEDIUM = 1024`, both module constants in `moe_config_utils.py`.
This is **not** the GEMM `M_LEQ_x` / `M_GEQ_y` scheme — do not mix them.

`A4W4` feeds the **gluon path only** (`get_kernel_config_gluon`); a4w4's triton
path still computes its config in Python. Its `<bucket>` is a *third* M scheme —
`m2bucket()` in `moe_op_gemm_a4w4.py`, splitting on 8 / 32 / 128 / 256 / 512 into
`tiny` / `small` / `medium` / `medium2` / `large` / `xlarge`. Lookup is two tiers:
`bm<block_m>_n<N>_k<K>_<bucket>`, then `bm<block_m>_any`. Since a missing bucket
falls all the way through to `_any` and loses the shape's tuning, a tuned shape
must supply **all six** buckets, even where the values repeat.

Both `a4w4/` and `a8w4/` sit under `<arch>/triton/` even though A4W4 serves the
gluon kernel: their loaders pass `backend="triton"`, so that is the only
candidate the probe looks at.

### MOE is the main offender for tuning values in Python

Fix these as you touch them; do not add more:

- `get_optimal_moe_config()` returns a hardcoded dict (`BLOCK_SIZE_M: 256`,
  `BLOCK_SIZE_N: 256`, …) when no config file exists, behind a
  `warnings.warn`. A missing config silently runs untuned values.
- `moe_op_gemm_a8w4.py` has a three-tier Python fallback: exact
  `bm_n_k` hit → any-`block_m` proxy with matching `(N, K)` → a gfx942-gated
  shape heuristic → a conservative default. Only the first tier reads tuned
  numbers from JSON.

### There is no `get_moe_config()`

An earlier revision of this file specified one. It was not written. The four
loaders reuse `resolve_config_dir()` — the shared probe that already existed —
rather than getting a wrapper of their own, and MOE's unification was
**path resolution only**: the schemas above were left exactly as they are, each
caller still parses its own, and converging them stays a separate decision that
would touch every MOE config file and require re-validating dispatch on every
arch.

The probe they call:

```python
def resolve_config_dir(op: str, config_name: str, backend: str | None = None,
                       legacy_dir: str | None = None,
                       arch: str | None = None) -> tuple[str, str]:
    """Return (cfg_dir, name_prefix) for the first candidate whose default
    file exists: DEFAULT.json when name_prefix is empty (the nested layout,
    dir from _dtype_dir()), else <name_prefix><config_name>.json. Falls back
    to the last candidate so the missing-file assertion names a legacy path.
    ``arch`` overrides the running architecture, for loaders that retry under
    another arch when the running one has no tuned configs."""
```

With `backend="triton"` and no `legacy_dir` there is exactly one candidate and
the returned name prefix is always `""`, so the file is always `DEFAULT.json`.

What is still open is the Python fallback tiers above, not the resolution path.
They were deliberately left in place through the migration so that a resolution
regression could not be swallowed by them. Retiring them is blocked on shipping
a `moe_default/DEFAULT.json` for every supported arch — today only gfx942,
gfx950 and gfx1250 ship any MOE config at all, and only gfx942 and gfx950 ship
`moe_default`, so everything else lands on the hardcoded dict in
`get_optimal_moe_config()`.

---

## 6. Adding a config

One family = one `<arch>` × `<backend>` × `<CONFIG_NAME>`, including every
specialized file. `GEMM-AFP4WFP4` is the worked example — copy its shape if a
step is ambiguous.

1. **Pick the directory.** `configs/<arch>/<backend>/<op>/<d_type>/`, with
   `<d_type>` = `config_name.lower().replace("-", "_")`. If the family already
   has a directory, the new file goes in that one — never a second directory
   for the same family.
2. **Name the file.** No arch prefix; the arch is the directory. The default is
   literally `DEFAULT.json`, and a specialized file keeps the `<CONFIG_NAME>-`
   stem plus its suffix (`GEMM-A16W16-N=256-K=7168.json`).
3. **Ship the `DEFAULT.json` first.** A `<d_type>/` directory holding only
   specialized files is invisible: the probe keys on `DEFAULT.json`, falls
   through to the next candidate, and ignores everything in the directory.
   `mkdir` the `<d_type>/` directory as part of adding its first file; it needs
   no `.gitkeep` (it is created populated).
4. **Find every reader.** Grep for the config name and for
   `AITER_TRITON_CONFIGS_PATH` in `aiter/ops/triton/`. `get_gemm_config()` and
   `resolve_config_dir()` callers need no change; hand-built paths do.
5. **If the file is moving, move it with `git mv`** so the change reviews as a
   rename, and do not edit contents in the same commit — renames stay at 100%
   similarity, content changes go in a follow-up.
6. **Pull any tuning values still hardcoded in Python into the JSON.** A family
   must be fully described by its config files.
7. **Verify** on the target arch: the config resolves, `is_tuned` is `True` for
   a shape that has a specialized file, and numerics are unchanged.
8. **Update the docs** if the change touches a convention:
   `aiter/ops/triton/README.md` ("Tuned configs"),
   `.github/instructions/aiter-ops-triton.instructions.md` (the Copilot review
   rules, expected to stay in sync with that README), and
   `aiter/ops/triton/utils/_triton/tunning/README.md` (the copy step under
   "Verify performance").

### A new op family

A family whose `<op>` segment does not exist yet needs the directory and a
loader that calls `resolve_config_dir("<op>", "<CONFIG_NAME>",
backend="triton")` — nothing else. Do not write a second probe: `conv`,
`attention`, `gmm`, `mhc` and `moe` were all wired up this way, and the shared
probe is what keeps every family findable under one set of rules. A kernel that
only needs one pinned tile per arch, rather than a per-shape lookup, uses
`utils/tuned_config_utils.py::get_tuned_kernel_config()` instead — same layout,
same `DEFAULT.json`.

### Do not

- Rename or delete `.gitkeep` placeholder directories.
- Put an arch prefix on a file inside `<arch>/...`, or name a nested default
  anything other than `DEFAULT.json`.
- Recreate a flat, arch-prefixed config directory — `configs/gemm/`,
  `configs/moe/`, `configs/conv/`, `configs/hstu_attn/`, or loose
  `<arch>-<NAME>.json` files at the top of `configs/`. Nothing probes them any
  more. (The runtime AOT caches below write under `configs/gemm/aot/` and
  `configs/paged_mqa_logits/aot/`; those are not config directories and are
  never checked in.)
- Put tuning values in `.py` files — no `setdefault`, no inline dicts, no
  arch-conditional constants, no hardcoded fallback configs.
- Mix the GEMM `M_LEQ_x`/`M_GEQ_y` scheme with the MOE
  `small_M`/`medium_M`/`large_M` scheme.

### Not tuning configs

Two AOT code paths build directories under this tree at runtime that are **not
checked in and out of scope**:

- `configs/gemm/aot/<kernel>_M=…-N=…-K=…` — `gemm/fused/fused_gemm_afp4wfp4_a16w16.py`,
  `gemm/fused/fused_gemm_afp4wfp4_mul_add.py`
- `configs/paged_mqa_logits/aot/<kernel>` — `attention/pa_mqa_logits.py`

Both are guarded by `use_aot and os.path.exists(...)` and hold compiled-kernel
metadata, not tuning parameters. Do not create, migrate, or document them as
config directories.
