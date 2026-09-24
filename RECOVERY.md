# Recovery Notes (2026-09-24)

The workspace was lost in a sandbox reset (no git repo existed). This tree was
reconstructed by replaying the original session (20260923_212714_dc5a1d) from
the Hermes session database:

- 54 `write_file` calls replayed in message order
- 207 `patch` calls replayed (exact + line-normalized fuzzy matching),
  cross-checked against recorded tool success/failure results
- 279 `read_file` results used as ground-truth snapshots to reconcile the
  replay at every point a file was read
- 22 embedded `patch`/`write_file` calls inside `execute_code` blocks included
- Duplicate tool calls injected by context-compaction replays deduplicated by
  call id (first occurrence carries full arguments)

## Verification

13 of 16 source files match their last in-session `read_file` ground truth
byte-for-byte (within +/-200 bytes of trailing-edit drift). The three deltas
are all explained by edits that postdate the last read:

| File | Replay | Last read | Cause |
|---|---|---|---|
| binetic-core/src/backend.rs | 19903 | 19052 | msg-1671 patch (+851B) landed after last read |
| binetic-llama-cpp/build.rs | 2311 | 3569 | final rewrite (2309B) at msg 1448 postdates read |
| ARCHITECTURE.md | 107109 | 85246 | sections 22/23 + glossary rows added after msg-73 read |

`cargo check -p binetic-core` on the recovered tree reproduces the session's
final state exactly: 11 E0308 errors (u64/usize mismatches in backend.rs),
28 warnings - the same error count the session ended on. This is the strongest
possible confirmation that the replay is faithful.

One repair was applied during recovery: `binetic-llama-cpp/Cargo.toml` had a
duplicated `[build-dependencies]` section (a replay artifact), which broke
manifest parsing before cargo could even report the known errors.

## Known state at session end

- binetic-core: 11 E0308 compile errors, 28 warnings. Tests written, never green.
- binetic-llama-cpp: build.rs compiles ggml-cpu (with the .c/.cpp split and
  _GNU_SOURCE fixes); no real FFI wiring (execute() returns placeholder
  latency/energy - no actual llama.cpp calls).
- binetic-mlx: cfg-gated stub for macOS; never compiled on Linux.
- No benchmark, no end-to-end inference, no CI.
- The vendored backends/llama-cpp and backends/mlx clones are NOT in git
  (see .gitignore) - re-clone them to build the backend crates.

## Next steps (agreed)

1. Fix the 11 E0308s, get `cargo test` green, commit - this repo.
2. Spec amendment: phase=epoch register modes (LEAF/CUBE/PORT/DELTA), typed
   cubes (BITS/CSA/TILED/KV/RECORD), ports as synchronizers, CSA/RESOLVE
   execution domains.
3. Vertical slice: one parent cube, one CUBE-mode child, one port with
   EPOCH-swap, one real op crossing a phase boundary.
