# Spec: `--pause-at <step>` — freeze the sim at a fixed step for LOD-comparison screenshots

Date: 2026-07-23
Status: approved (design)
Component: remote-rendering (rr-server / libmimir serveRemote)

## Motivation

For the benchmark paper we need side-by-side figures of the *same* simulation moment
rendered under different LOD settings — LOD native (`--lod-cells 0`), 32, 64, 128, 256, …
The particle simulation is seed-deterministic, so a given step `N` is the identical particle
state in every run; LOD only changes how that state is *rendered*. To capture equal images we
therefore need to freeze the sim at a fixed, reproducible step and grab a frame, repeating per
LOD config.

Today the sim free-runs, so the existing headless capture (`rr-client … <frames>` → `rr-client.ppm`)
lands on an *uncontrolled* step that differs per config. `--max-steps N` does not help: it makes
the server **exit** at `N` rather than freeze-and-hold, so no client can connect afterward to
capture. This spec adds a freeze-and-hold control.

## Behavior

New server option `--pause-at N` (default `0` = disabled).

- When the deterministic step counter (`total_iter`) reaches exactly `N`, the server sets
  `sim_paused = true` and **holds**: the sim stops advancing while the server keeps rendering,
  streaming, and accepting client connections. This reuses the existing viewer-pause machinery
  (`sim_paused` atomic + the "consumer keeps streaming the frozen state; the path tracer
  converges while paused" behavior).
- The freeze happens **with or without a client connected**:
  - Decoupled mode (default, `--steps-per-frame 0`): the background sim thread runs to `N` and
    freezes on its own; a client may connect any time afterward and see the frozen step-`N` state.
  - Lockstep mode (`--steps-per-frame ≥ 1`): the same check runs in the inline stepping path.
- Because the sim state at `N` is identical across runs and the headless client uses the fixed
  default camera pose, every LOD config yields identically-framed images differing only by the
  LOD representation.

### Resume

The interactive client's pause toggle continues to work: toggling resumes the sim from the
auto-pause. The consumer's local `paused` flag is initialized/synced to the auto-pause state so
the *first* toggle actually resumes rather than no-op'ing. (Resume is not needed for the capture
workflow but must not be broken.)

### Interaction with `--max-steps`

Independent. `--max-steps` exits at its step; `--pause-at` holds at its step. If both are set and
`pause_at <= max_steps`, the sim freezes at `pause_at` and never reaches `max_steps`. No special
coupling logic is added.

## Capture workflow (per config)

```bash
rr-server 9000 1920 1080 <N> 1 tcp "" ... --lod-cells 64 --pause-at 5000 &
rr-client <host> 9000 ... 1          # headless: connect, grab the frozen frame -> rr-client.ppm
mv rr-client.ppm lod64-step5000.ppm
```

Repeat with `--lod-cells 0` (native), 32, 128, 256. Same frozen step + same default camera ⇒
equal framing across all images.

## Camera

Uses the existing default deterministic camera pose. The headless client never moves the camera,
so framing is identical across configs automatically. No new camera control is added (confirmed
acceptable for the figures).

## Scope / non-goals

- **No** server-side image writing — reuse the existing `rr-client` headless PPM capture.
- **No** new camera control / angle presets.
- **No** change to `--max-steps` semantics.
- Not a benchmarking feature; purely a capture aid.

## Implementation outline

Files touched:

1. `samples/remote-rendering/rr-server.cu`
   - Add `size_t pause_at = 0;` to the parsed options (near `max_steps` at line ~242).
   - Parse `else if (a == "--pause-at") pause_at = (size_t)std::stoull(v);` (near line ~285).
   - Add a `--pause-at N` help line in the usage block (near the `--max-steps` help at line ~145).
   - Pass `pause_at` into the `serveRemote(...)` call (line ~595–596).

2. `lib/include/private/mimir/engine.hpp` (line ~359–363) and
   `lib/include/public/mimir/mimir.hpp` (line ~174–178)
   - Add a trailing `size_t pause_at = 0` parameter to both `serveRemote` declarations (default
     keeps existing callers source-compatible).

3. `lib/src/remote.cpp`
   - Add `size_t pause_at` to the `MimirInstance::serveRemote` definition (line ~364–366) and to
     the public free-function wrapper.
   - There are **three** places the sim advances; each needs the same post-advance check
     `if (pause_at != 0 && total_iter.load() >= pause_at) sim_paused.store(true)`:
     1. Decoupled background sim thread (`timed_compute(); total_iter.fetch_add(...)`, ~line 508–509).
     2. Lockstep no-viewer inline advance (`if (!decoupled …) { timed_compute(); total_iter.fetch_add(...) }`, ~line 539–540).
     3. Lockstep with-viewer inline stepping (`for (int s = 0; s < steps_per_frame; ++s)`, ~line 851) — apply after the loop.
   - Factor the check into a small local helper/lambda to avoid triplicating the guard, and use a
     `std::atomic<bool>` (or a one-shot flag) so the "paused at step N" info line logs exactly once.
   - Initialize the consumer's local `paused` (line ~662) consistently: when a session starts and
     `sim_paused` is already true due to auto-pause, set local `paused = true` so the pause-toggle
     round-trips correctly.
   - Emit a single info log on first reaching the step:
     `remote: paused at step {} (frozen; connect a client to capture)`.

## Testing

- `--pause-at 200 --steps-per-frame 1`: the `[sim]` heartbeat stops at step 200 and holds; server
  stays alive and accepts connections.
- Two consecutive headless captures against the frozen server produce **byte-identical** PPMs
  (proves the state is truly frozen).
- Native (`--lod-cells 0`) vs `--lod-cells 64` at the same `--pause-at`: the two PPMs differ only
  by the LOD representation (same framing / same underlying step).
- `--pause-at 0` (default): behavior unchanged from today (no freeze).
