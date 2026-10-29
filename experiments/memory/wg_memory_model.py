# ruff: noqa: T201
# (command-line report: printing is the output)
"""Analytical model of WeatherGenerator's per-GPU training memory.

Predicts the two peaks of a training step (A: decoder backward of the last forecast
step, B: embedding recompute at the end of the backward) from the config, so we can
ask "what fits in 80 GiB?" before spending node-hours. Every term is calibrated on
Matteo's baseline snapshot (run huw6isc4: lowres operan config, HEALPix level 5,
D = 2048, Sf = 2, 1 node, 4 GPUs, spatial parallel size 1) and then scaled by what
it depends on in the code.

Scaling assumptions (from reading the code; each is a hypothesis to check by a run):
  cells    C = 12 * 4**level            -> latent, adapter, gather, FA3 padding
  width    D                             -> every latent-sized tensor
  tokens   source tokens ~ cells * data -> batch on GPU, embedding (observations are
                                           binned into per-cell tokens, so finer cells
                                           mean more, emptier tokens)
  targets  target points per step       -> decoder activations, predictions
  Sf       forecast steps               -> forecast inputs, gather, per-step outputs
  P        spatial parallel size        -> cell-local stages / P (encoder, decoder);
                                           global and forecast engines replicated

v2 (27 Oct): v1 scaled the batch, the embedding and the remainders with raw data and
underpredicted level 6 by 17% (68.10 vs 82.16 GiB, u1xasv71). Snapshot diffs by call
site (snapdiff.py) showed the batch growing 3.8-3.9x per level and peak B (embedding
recompute) ~3.1x. v2 scales tokens with cells, splits each remainder into a per-cell
and a fixed part fitted on levels 5 and 6, keeps level 4 (ijubn2gz) out of the fit,
and takes the per-step slope (5.50 GiB at level 5) from tensor counts at the Sf = 2/4/6 peaks.
Per-step saved memory is linear; the peak carries a 1.7-4.2 GiB backward burst whose captured
size varies, so expect about +-2 GiB around any rollout prediction.
Decoder K/V over the 9x-gathered latent (attention.py:414/415 in TargetPredictionEngineClassic)
is recomputed in the backward: 2 x 432 MiB at level 5, 2 x 1728 MiB at level 6.

v4 (29 Oct): v3 overpredicted level 6 with Andres's 4 exact options by 13 GiB (66.55 vs 53.45,
qjbmx499). Diffing that snapshot against u1xasv71 by call site: fe_pairs removes the attention
residual (attention.py:656, 10.50 GiB) but keeps the MLP one, i.e. 46% of the forecast inputs,
not 50%; local_pairs removes 2.70 of 3.41 GiB local-engine residuals (attention.py:128), which v3
ignored; gather_after_kv also removes the decoder K/V recompute (-3.14 GiB), which v3 had inside
the per-cell remainder; and the "pair burst" v3 added (+6.4 GiB at level 6) is not there. It had
been inferred from Andres's level-5 run, where the peak had moved to the embedding (peak B), so
the saving looked smaller. Peak B now includes the bf16 batch and the per-stream checkpoint.

Usage:  python wg_memory_model.py            (prints validation and projections)
"""

from dataclasses import dataclass, replace

# nvidia-smi on a JUPITER booster node (job 2541772): "NVIDIA GH200 120GB", 97871 MiB HBM.
# Minus ~2 GiB for the CUDA context and NCCL, which the allocator's "reserved" does not include
# (an assumption, not measured). The 80 GiB used before came from the 15 Oct meeting.
HBM_GIB = 97871 / 1024
GIB_LIMIT = HBM_GIB - 2.0


@dataclass(frozen=True)
class Setup:
    level: int = 5  # HEALPix level
    dim: int = 2048  # latent width D
    fe_blocks: int = 16  # forecast-engine blocks
    sf: int = 2  # forecast steps
    p: int = 1  # spatial parallel size
    n_gpus: int = 4  # FSDP group size
    data: float = 1.0  # raw source data per sample, relative to lowres
    targets: float = 1.0  # target points per step, relative to lowres (O96 + stations)
    params_b: float = 0.9  # parameters, billions
    # Matteo's changes
    ckpt_adapter: bool = False  # change #1
    embed_chunks: int = 1  # change E (k)
    decoder_groups: int = 1  # change #3 (k)
    # Andres's switches (config keys in brackets)
    fe_pairs: bool = False  # [fe_checkpoint_per_block] one checkpoint per attention+MLP pair
    local_pairs: bool = False  # [ae_local_checkpoint_per_block] same for the local engine
    gather_after_kv: bool = False  # [decoder_gather_after_kv_projection] no 9x latent copy
    fe_bf16: bool = False  # [fe_checkpoint_bf16] forecast inputs saved in bf16
    source_bf16: bool = False  # [source_tokens_in_mixed_precision] batch copied to GPU in bf16
    # Alejandro's prototype (not in this repo; see README.md)
    embed_per_stream: bool = False  # [embed_checkpoint_per_stream] one checkpoint per stream
    # candidate changes (nobody's code yet)
    fe_offload: bool = False  # forecast inputs saved to host memory
    gather_in_ckpt: bool = False  # Matteo's #2: rebuild the 9-cell gather in backward


BASE = Setup()

# Calibration constants, GiB, at BASE (Matteo's baseline snapshot huw6isc4).
STATIC = dict(optimizer=1.72, fsdp_gather=1.17, pre_recording=1.85)  # 4.74 total
FE_PER_STEP = 5.72 / 2  # saved block inputs, measured 5.72 at Sf = 2
GATHER_PER_STEP = 0.84375  # [C, 9, D] fp32 = 864 MiB
# Peak A per forecast step at level 5, from counting the tensors live at the peaks of
# Sf = 2 / 4 / 6 (huw6isc4, vkyfae3p, szok2vom): each step adds 31 x 96 MiB forecast inputs,
# 7 x 236 MiB decoder, 1 x 864 MiB gather, 105 + 37 MiB small = 5.50 GiB, strictly linear.
# The measured slopes (5.03 for 2->4, 7.40 for 4->6) scatter around it because a backward
# burst of 16-32 tensors of ~100 MiB (plus 48 MiB ones) is caught to a different degree at
# each peak: 2.70 / 1.73 / 4.19 GiB at Sf = 2 / 4 / 6. Expect about +-2 GiB from that burst.
SLOPE_PER_STEP = 5.50
PER_STEP_OTHER = SLOPE_PER_STEP - FE_PER_STEP - GATHER_PER_STEP
ADAPTER = 2.40
ADAPTER_CKPT = 0.07
DEC_TRANSIENT_PAD = 2.39  # FA3 fp32 dQ scratch, padding per cell sequence
DEC_TRANSIENT_ACT = 2.85  # rest of the decoder backward burst (fit to #3: -3.93 GiB at k = 4)
DEC_PERSISTENT = 0.45  # 3.30 decoder group - 2.85
BATCH_ON_GPU = 9.7 / 4 / 1.0737  # 2.4 GB source tensors per sample (nyue0k4y), in GiB
GLOBAL_SAVED = 8 * 0.09375  # 4 blocks x (attention + MLP) inputs, fp32 C x D
REST_A = 29.26 - (  # unattributed remainder at peak A
    sum(STATIC.values())
    + 2 * (FE_PER_STEP + GATHER_PER_STEP + PER_STEP_OTHER)
    + ADAPTER
    + DEC_TRANSIENT_PAD
    + DEC_TRANSIENT_ACT
    + DEC_PERSISTENT
    + BATCH_ON_GPU
    + GLOBAL_SAVED
)
# Split of REST_A (2.42 at level 5) into a per-cell and a fixed part: fitting level 6 Sf = 2
# (u1xasv71, 82.16) gives per-cell 2.427 and fixed -0.006, i.e. all of it scales with cells
# (it includes the decoder K/V recompute over the 9x gather). Level 4 is the held-out check.
REST_A_CELL = REST_A
REST_A_FIX = 0.0
EMBED_BURST = 8.24 / (1 - 1 / 4)  # 10.99 GiB, implied by change E
REST_B = 26.84 - sum(STATIC.values()) - EMBED_BURST  # live at the end of the backward
# Same split for REST_B (11.11 at level 5), fitted to peak B at level 6 Sf = 1 (xgd5y6vp,
# 73.67, embedding recompute). No held-out point for B yet: the level-4 peak was A.
REST_B_CELL = 4.62
REST_B_FIX = REST_B - REST_B_CELL
# reserved/allocated: 1.07-1.11 at level 5 (huw6isc4, Matteo after changes, a09k1cvv, nyue0k4y),
# 1.06 at level 6 Sf = 2 but 1.15 at level 6 Sf = 1; "fits" uses the worst case seen
RESERVED_OVER_ALLOCATED = 1.15
# Andres's options, measured at level 6 Sf = 2 (u1xasv71 -> qjbmx499, peak A, by call site).
# fe_pairs: the attention residual (attention.py:656) is no longer saved, the MLP one is:
# 2.62 of 5.72 GiB at level 5, 10.50 of 22.9 at level 6.
FE_PAIRS_KEEP = 1 - 2.62 / 5.72
# local_pairs: local-engine residuals (attention.py:128) 1.49 GiB at level 5, 3.41 at level 6
# (2.3x per level, so exponent 0.6 on cells, fitted to these two); 2.70 of 3.41 removed.
LOCAL_SAVED = 1.49
LOCAL_EXP = 0.6
LOCAL_PAIRS_REMOVED = 2.70 / 3.41
# gather_after_kv: decoder K/V recomputed over the 9x gather, 2 x 432 MiB at level 5; measured
# -3.14 GiB at level 6 (modelled 3.38). At level 5 it is not live at peak A in huw6isc4, so the
# level-5 saving is untested. Still counted inside REST_A_CELL for the baseline.
DEC_KV_RECOMPUTE = 2 * 432 / 1024
# embed_per_stream: peak B -3.31 GiB at level 6 Sf = 1 (xgd5y6vp 73.67 -> rliwt2zh 70.36) =
# 7.5% of the embedding burst; ERA5_in alone is 30.3 of the recompute (g2z5vwaj).
PER_STREAM_KEEP = 1 - 3.31 / (EMBED_BURST * 4)


def scales(s: Setup) -> dict:
    cells = 4 ** (s.level - BASE.level)
    width = s.dim / BASE.dim
    return dict(
        cd=cells * width,  # one latent tensor
        cd_local=cells * width / s.p,  # latent tensor, cell-local stage
        c_local=cells / s.p,  # per-cell padding, cell-local
        data_local=s.data / s.p,
        tok_local=cells * s.data / s.p,  # source tokens, cell-local
        tgt_local=s.targets / s.p,
        fe=cells * width * s.fe_blocks / BASE.fe_blocks,
        params=s.params_b / BASE.params_b,
    )


def peak_a(s: Setup) -> dict:
    k = scales(s)
    fe = FE_PER_STEP * k["fe"] * s.sf
    if s.fe_pairs:
        fe *= FE_PAIRS_KEEP
    if s.fe_bf16:
        fe *= (32 * 48) / (24 * 96 + 8 * 48)  # all inputs at 48 MiB instead of mixed fp32/bf16
    if s.fe_offload:
        fe = FE_PER_STEP * k["fe"] / 16  # about one block's input in flight on the GPU
    gather = s.sf * GATHER_PER_STEP * k["cd_local"]
    if s.gather_in_ckpt:
        gather = GATHER_PER_STEP * k["cd_local"]  # one step's copy, rebuilt in the backward
    if s.gather_after_kv:
        gather /= 9  # keys/values of each cell once instead of a 9x copy of the latent
    # forecast-engine parameters stay gathered (reshard_after_forward=False): ~ blocks * D^2
    fe_params = (s.fe_blocks / BASE.fe_blocks) * (s.dim / BASE.dim) ** 2
    terms = {
        "static": (STATIC["optimizer"] + STATIC["pre_recording"])
        * k["params"]
        * BASE.n_gpus
        / s.n_gpus
        + STATIC["fsdp_gather"] * fe_params,
        "forecast inputs": fe,
        "decoder gather": gather,
        "per-step outputs": s.sf * PER_STEP_OTHER * k["tgt_local"],
        "adapter": (ADAPTER_CKPT if s.ckpt_adapter else ADAPTER) * k["cd_local"],
        "decoder burst": (DEC_TRANSIENT_PAD * k["c_local"] + DEC_TRANSIENT_ACT * k["tgt_local"])
        / s.decoder_groups,
        "decoder persistent": DEC_PERSISTENT * k["tgt_local"],
        "batch on GPU": BATCH_ON_GPU * k["tok_local"] * (0.5 if s.source_bf16 else 1.0),
        "global engine": GLOBAL_SAVED * k["cd"],
        "other, per cell (fit)": REST_A_CELL * k["tok_local"],
        "other, fixed (fit)": REST_A_FIX * k["data_local"],
        "local pairs saving": -LOCAL_PAIRS_REMOVED * LOCAL_SAVED * k["c_local"] ** LOCAL_EXP
        if s.local_pairs
        else 0.0,
        "K/V recompute saving": -DEC_KV_RECOMPUTE * k["cd_local"] if s.gather_after_kv else 0.0,
    }
    return terms


def peak_b(s: Setup) -> dict:
    k = scales(s)
    return {
        "static": peak_a(s)["static"],
        "embedding burst": EMBED_BURST
        * k["tok_local"]
        / s.embed_chunks
        * (PER_STREAM_KEEP if s.embed_per_stream else 1.0),
        "live at end, per cell (fit)": REST_B_CELL * k["tok_local"],
        "live at end, fixed (fit)": REST_B_FIX * k["data_local"],
        "bf16 batch saving": -0.5 * BATCH_ON_GPU * k["tok_local"] if s.source_bf16 else 0.0,
    }


def total(terms: dict) -> float:
    return sum(terms.values())


def report(name: str, s: Setup, measured: str = "") -> None:
    a, b = total(peak_a(s)), total(peak_b(s))
    peak = max(a, b)
    fits = "fits" if peak * RESERVED_OVER_ALLOCATED <= GIB_LIMIT else "DOES NOT FIT"
    print(f"{name:<44} A {a:6.2f}  B {b:6.2f}  peak {peak:6.2f} GiB  {fits:<12} {measured}")


def max_sf(s: Setup, limit: float = GIB_LIMIT) -> int:
    sf = 1
    while (
        sf < 64
        and max(total(peak_a(replace(s, sf=sf + 1))), total(peak_b(s))) * RESERVED_OVER_ALLOCATED
        <= limit
    ):
        sf += 1
    return sf


def main() -> None:
    changes = dict(ckpt_adapter=True, embed_chunks=4, decoder_groups=4)
    print("Validation against Matteo's snapshots (GiB, max over 4 GPUs)")
    report("baseline Sf=2", BASE, "measured A 29.26, B 26.84")
    report("baseline Sf=1", replace(BASE, sf=1), "measured peak 26.52 (B)")
    report(
        "baseline Sf=4", replace(BASE, sf=4), "measured 39.32 (vkyfae3p; slope from tensor counts)"
    )
    report("#1 adapter checkpoint", replace(BASE, ckpt_adapter=True), "measured A 26.95")
    report("#1 + E", replace(BASE, ckpt_adapter=True, embed_chunks=4), "measured B 18.60")
    report("#1 + E + #3", replace(BASE, **changes), "measured 23.02")
    print("Calibration of the v2 terms (Alejandro's snapshots)")
    report("baseline Sf=6", replace(BASE, sf=6), "measured 54.12 (szok2vom; burst +2.9 above line)")
    report("level 6, Sf=2", replace(BASE, level=6), "measured 82.16 (u1xasv71, sets REST_A_CELL)")
    report(
        "level 6, Sf=1",
        replace(BASE, level=6, sf=1),
        "measured 73.67 B (xgd5y6vp, sets REST_B_CELL)",
    )

    print("\nOut-of-sample checks (not used for calibration)")
    report(
        "#1 + E + #3 with 2 decoder groups",
        replace(BASE, ckpt_adapter=True, embed_chunks=4, decoder_groups=2),
        "measured 24.70 (hzwxpsw8)",
    )
    report("level 4, Sf=2", replace(BASE, level=4), "measured 17.00 A (ijubn2gz)")

    print("\nLevel 6 with Andres's options (v4; 'fit' = used to set a v4 constant)")
    opts4 = dict(level=6, fe_pairs=True, gather_after_kv=True, local_pairs=True, source_bf16=True)
    l6 = replace(BASE, **opts4)
    report(
        "gather_after_kv only",
        replace(BASE, level=6, gather_after_kv=True),
        "measured 74.20 = B 73.85 (f1mgt06z; A not isolated)",
    )
    report("4 options", l6, "measured 69.78 B (jdg1lutw, held out)")
    report(
        "4 options + per-stream",
        replace(l6, embed_per_stream=True),
        "measured 66.52 B (g2z5vwaj, held out)",
    )
    report(
        "Sf=1, per-stream only",
        replace(BASE, level=6, sf=1, embed_per_stream=True),
        "measured 70.36 B (rliwt2zh, fit)",
    )
    ch = replace(l6, embed_per_stream=True, embed_chunks=4)
    report("4 options + per-stream + 4 chunks", ch, "measured 53.45 A (qjbmx499, fit)")
    report(
        "  same, 2 chunks",
        replace(ch, embed_chunks=2),
        "measured 53.45 A (v2sceodc, predicted before the run)",
    )
    report("  same, Sf=4", replace(ch, sf=4), "measured 70.93 (xokjem39, held out)")
    report("  same, Sf=6", replace(ch, sf=6), "measured 88.68 (tdaw4mhu, held out)")
    print(
        "  Andres's level-5 runs (his reference config) are peak-B limited after the options, "
        "so they do not test the A terms"
    )

    print("\nComposition of peak A")
    team = dict(
        ckpt_adapter=True,
        embed_chunks=4,
        decoder_groups=2,
        gather_after_kv=True,
        fe_pairs=True,
        local_pairs=True,
        fe_bf16=True,
        source_bf16=True,
    )
    for name, s in (
        ("baseline", BASE),
        ("after #1+E+#3", replace(BASE, **changes)),
        ("all team switches", replace(BASE, **team)),
    ):
        terms = peak_a(s)
        print(f"  {name}: " + ", ".join(f"{k} {v:.2f}" for k, v in terms.items() if v))

    print("\nProjections (estimates; each scaling rule is an assumption to test)")
    after = replace(BASE, **changes)
    allsw = replace(BASE, **team)
    scenarios = [
        ("level 5, all team switches", allsw),
        ("level 6, baseline", replace(BASE, level=6)),
        ("level 6, after #1+E+#3", replace(after, level=6)),
        ("level 6, all team switches", replace(allsw, level=6)),
        ("level 6, all switches + FE offload", replace(allsw, level=6, fe_offload=True)),
        ("level 6, all switches, P=4", replace(allsw, level=6, p=4)),
        ("D = 3072, level 5, all switches", replace(allsw, dim=3072, params_b=0.9 * 2.25)),
        # tokens ~ cells * data is an upper bound at fixed cells: denser cells fill tokens up
        ("4x data (finer inputs), all switches", replace(allsw, data=4.0)),
        ("4x data, all switches, P=4", replace(allsw, data=4.0, p=4)),
    ]
    for name, s in scenarios:
        report(name, s)

    for level in (5, 6):
        print(f"\nLongest rollout that fits (Sf), level {level}")
        for name, s in (
            ("baseline", BASE),
            ("after #1+E+#3", after),
            ("all team switches", allsw),
            ("all switches + FE offload", replace(allsw, fe_offload=True)),
        ):
            print(f"  {name:<28} Sf <= {max_sf(replace(s, level=level))}")

    print("\nHost memory hypothesis (to test with a num_workers / prefetch sweep)")
    sample_gib = 2.4 / 1.0737
    for workers, prefetch in ((12, 2), (12, 1), (6, 2), (6, 1)):
        queued = workers * prefetch * sample_gib
        print(
            f"  {workers:>2} workers x prefetch {prefetch}: up to {queued:5.1f} GiB of queued "
            f"samples per rank (measured MaxRSS 90.8 GiB at 12 x 2)"
        )
    print(
        "  level 6 (samples ~4x): 12 workers ran out of /dev/shm before step 1, MaxRSS 131 GiB "
        "(js5hwonm); 2 workers ran, MaxRSS 15 GiB (u1xasv71)"
    )


if __name__ == "__main__":
    main()
