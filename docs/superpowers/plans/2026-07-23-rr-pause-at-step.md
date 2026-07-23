# `--pause-at <step>` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server option `--pause-at N` that freezes the simulation at exactly step N and holds (keeps serving), so the same simulation moment can be captured under different LOD settings for equal side-by-side screenshots.

**Architecture:** Thread a new `size_t pause_at` parameter through the `serveRemote` API (public wrapper → engine method). Inside `serveRemote`, a single helper lambda checks `total_iter >= pause_at` after each sim advance and sets the existing `sim_paused` atomic once, reusing the viewer-pause "freeze and keep streaming" machinery. `rr-server` parses `--pause-at` and passes it through.

**Tech Stack:** C++20, CUDA, Vulkan, spdlog. Built via CMake into `libmimir.a` (`./build`) which `rr-server` (`samples/remote-rendering/build`) links.

## Global Constraints

- C++ standard: C++20 (match surrounding code; no new dependencies).
- No unit-test framework exists in this repo. "Tests" are **integration checks**: build, run `rr-server`, observe logs, capture PPMs with the headless `rr-client`, and byte-compare. Every verification step below is a concrete command with expected output.
- The build has two trees and a known clock-skew quirk: after editing `lib/src/*.cpp`, run `touch lib/src/remote.cpp`, rebuild `libmimir.a`, then relink `rr-server`. Exact commands are in each task.
- New parameter defaults to `0` (disabled) everywhere so all existing callers are source-compatible and behavior is unchanged when unused.
- Commit message trailer (every commit):
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01CQfpwcA5mm3RDPqc27Kw3W
  ```
- Requires a working Vulkan/GPU box to run the integration checks (e.g. the current working RunPod pod with the libEGL ICD fix, or Patagon). Building does not require a GPU.

## File Structure

- `lib/include/public/mimir/mimir.hpp` — public free-function `serveRemote` declaration (add `pause_at`).
- `lib/include/private/mimir/engine.hpp` — `MimirInstance::serveRemote` method declaration (add `pause_at`).
- `lib/src/mimir.cpp` — public wrapper definition (add `pause_at`, forward it).
- `lib/src/remote.cpp` — method definition: add `pause_at`, the freeze helper, the three stepping-site calls, and the session `paused` sync. This is where the behavior lives.
- `samples/remote-rendering/rr-server.cu` — parse `--pause-at`, help text, pass through.

---

### Task 1: Thread `pause_at` through libmimir and add the freeze logic

**Files:**
- Modify: `lib/include/public/mimir/mimir.hpp:174-179`
- Modify: `lib/include/private/mimir/engine.hpp:359-363`
- Modify: `lib/src/mimir.cpp:280-287`
- Modify: `lib/src/remote.cpp:364-366` (signature), `:494-495` (helper), `:508-509`, `:539-540`, `:851-857` (stepping sites), `:662` (session sync)

**Interfaces:**
- Produces: `mimir::serveRemote(..., int steps_per_frame = 0, size_t pause_at = 0)` (public free function) and `MimirInstance::serveRemote(..., int steps_per_frame = 0, size_t pause_at = 0)` (method). `pause_at == 0` disables; `pause_at > 0` freezes the sim at that step.

- [ ] **Step 1: Add `pause_at` to the public declaration**

In `lib/include/public/mimir/mimir.hpp`, extend the `serveRemote` declaration (currently ending `int fps = 0, int steps_per_frame = 0`) and add a doc line. Replace lines 174-179:

```cpp
// 'pause_at' > 0 freezes the simulation once it reaches that step and holds (the server keeps
// streaming the frozen frame and accepting clients), for capturing the same step across configs;
// 0 disables. Unlike max_iters, which returns/exits, pause_at holds the run alive.
void serveRemote(InstanceHandle engine, unsigned short port,
    std::function<void(void)> func, size_t max_iters, bool use_h264 = false,
    remote::TransportKind kind = remote::TransportKind::Tcp,
    const char *token = "", int bitrate_kbps = 8000, const char *stats_csv = nullptr,
    int fps = 0, int steps_per_frame = 0, size_t pause_at = 0
);
```

- [ ] **Step 2: Add `pause_at` to the method declaration**

In `lib/include/private/mimir/engine.hpp`, replace lines 359-363:

```cpp
    void serveRemote(uint16_t port, std::function<void(void)> compute, size_t max_iters,
        bool use_h264 = false,
        remote::TransportKind kind = remote::TransportKind::Tcp,
        std::string token = {}, int bitrate_kbps = 8000, std::string stats_csv = {},
        int target_fps = 0, int steps_per_frame = 0, size_t pause_at = 0);
```

- [ ] **Step 3: Forward `pause_at` in the public wrapper**

In `lib/src/mimir.cpp`, replace lines 280-287:

```cpp
void serveRemote(InstanceHandle engine, unsigned short port,
    std::function<void(void)> func, size_t max_iters, bool use_h264,
    remote::TransportKind kind, const char *token, int bitrate_kbps, const char *stats_csv,
    int fps, int steps_per_frame, size_t pause_at)
{
    engine->serveRemote(port, func, max_iters, use_h264, kind, token ? token : "", bitrate_kbps,
        stats_csv ? stats_csv : "", fps, steps_per_frame, pause_at);
}
```

- [ ] **Step 4: Add `pause_at` to the method definition signature**

In `lib/src/remote.cpp`, replace the signature at lines 364-366:

```cpp
void MimirInstance::serveRemote(uint16_t port, std::function<void(void)> compute,
    size_t max_iters, bool use_h264, remote::TransportKind kind, std::string token,
    int bitrate_kbps, std::string stats_csv, int target_fps, int steps_per_frame, size_t pause_at)
```

- [ ] **Step 5: Add the freeze helper right after `sim_paused`**

In `lib/src/remote.cpp`, the block currently reads (lines 493-495):

```cpp
    std::atomic<bool> sim_stop{false};
    std::atomic<bool> sim_paused{false};
    std::thread sim_thread;
```

Insert the helper between `sim_paused` and `sim_thread` so the decoupled thread lambda (which captures by `[&]`) can see it:

```cpp
    std::atomic<bool> sim_stop{false};
    std::atomic<bool> sim_paused{false};
    // --pause-at: freeze the sim exactly when it reaches step `pause_at` (0 = disabled), holding the
    // frozen state so a client can connect and capture an identical frame across LOD configs. Called
    // after each sim advance at the three stepping sites below; sets sim_paused and logs once.
    std::atomic<bool> pause_at_hit{false};
    auto apply_pause_at = [&]()
    {
        if (pause_at == 0 || pause_at_hit.load(std::memory_order_relaxed)) { return; }
        if (total_iter.load(std::memory_order_acquire) >= pause_at)
        {
            sim_paused.store(true, std::memory_order_release);
            pause_at_hit.store(true, std::memory_order_relaxed);
            spdlog::info("remote: paused at step {} (frozen; connect a client to capture)", pause_at);
        }
    };
    std::thread sim_thread;
```

- [ ] **Step 6: Call the helper in the decoupled sim thread**

In `lib/src/remote.cpp`, the decoupled thread body currently reads (lines 508-509):

```cpp
                timed_compute();
                total_iter.fetch_add(1, std::memory_order_release);
```

Add the call:

```cpp
                timed_compute();
                total_iter.fetch_add(1, std::memory_order_release);
                apply_pause_at();
```

- [ ] **Step 7: Call the helper in the lockstep no-viewer advance**

In `lib/src/remote.cpp`, inside the `if (!decoupled && !(max_iters ...))` block (lines 539-540):

```cpp
                timed_compute();
                total_iter.fetch_add(1, std::memory_order_release);
```

Add the call:

```cpp
                timed_compute();
                total_iter.fetch_add(1, std::memory_order_release);
                apply_pause_at();
```

Note: there are two occurrences of `timed_compute(); total_iter.fetch_add(1, std::memory_order_release);` in the file (Steps 6 and 7). Apply Step 6 to the one inside the `sim_thread` lambda (~line 508, indented inside `while (!sim_stop...)`), and Step 7 to the one inside the main `while (!stop)` loop's no-viewer branch (~line 539, guarded by `if (!decoupled ...)`). After both edits each of those two `fetch_add` lines is followed by `apply_pause_at();`.

- [ ] **Step 8: Call the helper after the lockstep with-viewer step loop**

In `lib/src/remote.cpp`, the lockstep with-viewer loop (lines 851-857) currently reads:

```cpp
                for (int s = 0; s < steps_per_frame; ++s)
                {
                    if (max_iters != 0 && total_iter.load() >= max_iters) { stop = true; break; }
                    timed_compute();
                    total_iter.fetch_add(1, std::memory_order_relaxed);
                }
                pt_scene_dirty = true; // the sim moved, so reset the path-trace accumulator
```

Insert the call after the loop:

```cpp
                for (int s = 0; s < steps_per_frame; ++s)
                {
                    if (max_iters != 0 && total_iter.load() >= max_iters) { stop = true; break; }
                    timed_compute();
                    total_iter.fetch_add(1, std::memory_order_relaxed);
                }
                apply_pause_at();
                pt_scene_dirty = true; // the sim moved, so reset the path-trace accumulator
```

- [ ] **Step 9: Sync the session's local `paused` to the auto-pause**

In `lib/src/remote.cpp` line 662, replace:

```cpp
        bool paused = false;
```

with:

```cpp
        // Start the session's pause state from the shared flag so an auto-pause (--pause-at) already
        // in effect is reflected here, and the client's first pause-toggle resumes instead of no-op'ing.
        bool paused = sim_paused.load(std::memory_order_acquire);
```

- [ ] **Step 10: Build libmimir**

Run:
```bash
cd /home/cnavarro/temporal/mimir && touch lib/src/remote.cpp && cmake --build build --target mimir -j
```
Expected: compiles and links `libmimir.a` with no errors or warnings (the new param is used, so no `-Wunused-parameter`).

- [ ] **Step 11: Commit**

```bash
cd /home/cnavarro/temporal/mimir
git add lib/include/public/mimir/mimir.hpp lib/include/private/mimir/engine.hpp lib/src/mimir.cpp lib/src/remote.cpp
git commit -m "$(printf 'feat(rr): serveRemote pause_at param -- freeze sim at a fixed step and hold\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01CQfpwcA5mm3RDPqc27Kw3W')"
```

---

### Task 2: Wire `--pause-at` in rr-server and verify end-to-end

**Files:**
- Modify: `samples/remote-rendering/rr-server.cu:242` (option field), `:285` (parse), `:148` (help), `:595-596` (call)

**Interfaces:**
- Consumes: `serveRemote(..., int steps_per_frame, size_t pause_at)` from Task 1.

- [ ] **Step 1: Add the option field**

In `samples/remote-rendering/rr-server.cu`, after line 242 (`size_t max_steps = 0;`):

```cpp
    size_t max_steps        = 0;
    size_t pause_at         = 0;
```

- [ ] **Step 2: Parse `--pause-at`**

In `samples/remote-rendering/rr-server.cu`, after the `--max-steps` parse (line 285):

```cpp
            else if (a == "--max-steps")   max_steps = (size_t)std::stoull(v);
            else if (a == "--pause-at")    pause_at = (size_t)std::stoull(v);
```

- [ ] **Step 3: Add help text**

In `samples/remote-rendering/rr-server.cu`, after the `--max-steps` help block (after line 148, before the `--fly` line):

```cpp
        "  --pause-at N       Freeze the simulation when it reaches step N and hold (default: 0 =\n"
        "                     disabled). Unlike --max-steps (which exits), the server keeps serving\n"
        "                     the frozen frame so a client can connect and capture. Use to grab\n"
        "                     identical screenshots of the same step across LOD configs.\n"
```

- [ ] **Step 4: Pass `pause_at` into the serveRemote call**

In `samples/remote-rendering/rr-server.cu`, replace lines 595-596:

```cpp
    }, max_steps, use_h264, transport, token.c_str(), bitrate_kbps,
        bench_csv.empty() ? nullptr : bench_csv.c_str(), fps_cap, steps_per_frame, pause_at);
```

- [ ] **Step 5: Build rr-server**

Run:
```bash
cd /home/cnavarro/temporal/mimir && cmake --build samples/remote-rendering/build --target rr-server -j
```
Expected: compiles and links `samples/remote-rendering/build/rr-server` with no errors. (If it skips relinking against the freshly built `libmimir.a` due to clock skew, run `touch lib/src/remote.cpp && cmake --build build --target mimir -j` again, then rebuild rr-server.)

- [ ] **Step 6: Verify `--pause-at` appears in help**

Run:
```bash
samples/remote-rendering/build/rr-server --help 2>&1 | grep -A3 'pause-at'
```
Expected: the four `--pause-at` help lines print.

- [ ] **Step 7: Integration test — freeze holds at N (primary proof), raster mode**

Start the server in decoupled default mode with a raster light model (deterministic frozen frame) and a small step:
```bash
cd /home/cnavarro/temporal/mimir
pkill -f rr-server 2>/dev/null; sleep 1
nohup samples/remote-rendering/build/rr-server 9000 640 480 $((1*10**6)) 1 tcp "" \
  --seed 1 --light-model none --pause-at 200 > /tmp/pa.log 2>&1 &
sleep 4
grep -E "paused at step 200|\[sim\] step" /tmp/pa.log | tail -6
```
Expected: a `remote: paused at step 200 (frozen; connect a client to capture)` line, and the `[sim]` heartbeat step count reaches 200 and then stays at 200 on subsequent lines (frozen — not climbing).

- [ ] **Step 8: Integration test — two captures are byte-identical (frozen)**

With the frozen server from Step 7 still running, capture twice with the headless client (the last positional arg is the frame count that triggers headless "grab → save rr-client.ppm → exit"):
```bash
cd /home/cnavarro/temporal/mimir
samples/remote-rendering/build/rr-client 127.0.0.1 9000 3 && cp rr-client.ppm /tmp/shotA.ppm
samples/remote-rendering/build/rr-client 127.0.0.1 9000 3 && cp rr-client.ppm /tmp/shotB.ppm
cmp /tmp/shotA.ppm /tmp/shotB.ppm && echo "IDENTICAL (frozen OK)"
```
Expected: `cmp` prints nothing and `IDENTICAL (frozen OK)` — the two frames match because the sim is frozen and `--light-model none` is deterministic (no path-trace convergence between captures). If the exact `rr-client` positional syntax differs, consult `rr-client --help`; the required behavior is "headless: receive N frames, save rr-client.ppm, exit".

- [ ] **Step 9: Integration test — same step, different LOD → equal framing, different content**

Capture native vs LOD-64 at the same paused step (use a lit model so LOD applies; PT/phong both fine — use phong for speed):
```bash
cd /home/cnavarro/temporal/mimir
pkill -f rr-server 2>/dev/null; sleep 1
nohup samples/remote-rendering/build/rr-server 9000 640 480 $((1*10**6)) 1 tcp "" \
  --seed 1 --light-model phong --lod-cells 0 --pause-at 200 > /tmp/pa0.log 2>&1 &
sleep 4
samples/remote-rendering/build/rr-client 127.0.0.1 9000 3 && cp rr-client.ppm /tmp/native.ppm
pkill -f rr-server 2>/dev/null; sleep 1
nohup samples/remote-rendering/build/rr-server 9000 640 480 $((1*10**6)) 1 tcp "" \
  --seed 1 --light-model phong --lod-cells 64 --pause-at 200 > /tmp/pa64.log 2>&1 &
sleep 4
samples/remote-rendering/build/rr-client 127.0.0.1 9000 3 && cp rr-client.ppm /tmp/lod64.ppm
head -c 20 /tmp/native.ppm | head -1   # same "P6 640 480" header => same framing/resolution
head -c 20 /tmp/lod64.ppm  | head -1
cmp -s /tmp/native.ppm /tmp/lod64.ppm && echo "UNEXPECTED: identical" || echo "OK: differ by LOD"
```
Expected: both PPMs share the same `P6 640 480` header (identical framing), and `cmp` reports they differ (`OK: differ by LOD`) — same underlying step, different LOD representation.

- [ ] **Step 10: Regression — `--pause-at 0` never freezes**

```bash
cd /home/cnavarro/temporal/mimir
pkill -f rr-server 2>/dev/null; sleep 1
nohup samples/remote-rendering/build/rr-server 9000 640 480 $((1*10**6)) 1 tcp "" \
  --seed 1 --light-model none --pause-at 0 > /tmp/pa_off.log 2>&1 &
sleep 4
grep -c "paused at step" /tmp/pa_off.log   # expect 0
grep -E "\[sim\] step" /tmp/pa_off.log | tail -2   # step count keeps climbing
pkill -f rr-server 2>/dev/null
```
Expected: `0` "paused at step" lines and the `[sim]` step count still increasing — default behavior unchanged.

- [ ] **Step 11: Commit**

```bash
cd /home/cnavarro/temporal/mimir
git add samples/remote-rendering/rr-server.cu
git commit -m "$(printf 'feat(rr): --pause-at <step> CLI to freeze the sim for LOD-comparison screenshots\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01CQfpwcA5mm3RDPqc27Kw3W')"
```

---

## Self-Review

**Spec coverage:**
- `--pause-at N` option, default 0 → Task 2 Steps 1-4. ✓
- Freeze at exact step, hold, keep serving → Task 1 Steps 5-8 (helper + three stepping sites) reusing `sim_paused`. ✓
- Works with/without client (decoupled + both lockstep paths) → Task 1 Steps 6, 7, 8 cover all three advance sites. ✓
- Resume via client toggle (local `paused` sync) → Task 1 Step 9. ✓
- Independent of `--max-steps` (holds vs exits) → no coupling added; `max_iters` checks untouched. ✓
- Default camera / no server-side image write / no camera control → nothing added for these (non-goals honored). ✓
- Capture workflow + equal-framing + differs-by-LOD → Task 2 Steps 8, 9. ✓
- Log line once → Task 1 Step 5 (`pause_at_hit` guard). ✓

**Placeholder scan:** none — every code and command step is concrete.

**Type consistency:** `pause_at` is `size_t` in all five sites (public decl, method decl, wrapper def, method def, rr-server field/parse); helper named `apply_pause_at` and used identically at all three call sites; `pause_at_hit` is `std::atomic<bool>`. ✓

**Note on PT vs raster in tests:** the byte-identical determinism check (Task 2 Step 8) deliberately uses `--light-model none` because a path-traced frozen scene keeps converging between captures; the LOD-difference check (Step 9) uses `--light-model phong`. This matches the spec's testing intent (freeze proof via the held step count + a deterministic raster byte-compare).
