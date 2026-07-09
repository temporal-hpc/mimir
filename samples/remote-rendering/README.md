# Remote rendering sample

A headless GPU server renders a live CUDA workload and streams it to a thin native client over the
network; the client displays the frames and sends interaction (camera, pause) back. This models a
researcher on a laptop driving a visualization that runs on a remote GPU box (cluster node,
workstation) over the internet.

Two binaries:

- **`rr-server`** — renders headless with mimir, encodes (H.264 via NVENC, or raw), and streams to
  one client at a time. Links the `mimir` library + CUDA, so it needs a full build.
- **`rr-client`** — a thin viewer: receives the stream, decodes it, shows it in a window, and sends
  control back. Depends only on the wire protocol + ngtcp2 + OpenSSL + ffmpeg + GLFW/OpenGL — **no
  mimir, CUDA, or Vulkan**, so it builds and runs on a laptop with no NVIDIA stack at all (see
  [Viewer only](#viewer-only-no-nvidia-gpu--cuda-needed)).

`rr-client` is **workload-agnostic** — it knows nothing about the point-cloud server, so the *same*
client views any mimir server built with `serveRemote()`. The build also installs this exact program
as a standalone tool, **`mimir-client`** (top-level `client/`), so you can `cmake --install` it and
run `mimir-client <host> <port>` against your own server without rebuilding the samples.

## What it renders

The **k-modal 3D point cloud** — the same simulation the
[`particles-kmodal-3d`](../particles-kmodal-3d/) benchmark renders locally, so the remote client
sees an identical view. `point_count` particles start in a k-cluster gaussian mixture in the
`[-1,1]³` cube and each frame take a mean-reverting (Ornstein–Uhlenbeck) random-walk step drawn with
cuRAND — the clusters keep their blobby "cheese" shape indefinitely while the points jiggle. It's
continuously moving, which keeps the encoder honest, and the particle buffer lives on the GPU via
mimir's CUDA interop (no per-frame host round-trip).

The **render path is selectable** with `--light-model`, so the client can view the simulation as:

| `--light-model` | What the client sees                                            |
|-----------------|-----------------------------------------------------------------|
| `none` / `point`| Unlit pixel-sized discs (cheapest)                              |
| `phong`         | Lit sphere impostors (default)                                  |
| `phong-mesh`    | Lit instanced icosphere meshes (`--subdiv` tessellation)        |
| `path-tracing`  | Vulkan ray tracing (`--spp`, `--bounces`)                       |

The lighting (sun direction, `--pcolor`, `--background`) and camera framing match
`particles-kmodal-3d/benchmark_mimir`, so a phong / phong-mesh / path-traced remote frame looks the
same as its local counterpart.

## Building

Pick the path that matches the machine you are on:

- **Viewer only** — the machine will just *view* a remote server (laptop, ultrabook, anything
  with Intel/AMD graphics). No CUDA toolkit, Vulkan, NVIDIA hardware, or mimir library needed.
  → [Viewer only](#viewer-only-no-nvidia-gpu--cuda-needed), one command.
- **Full build** — the machine has an NVIDIA GPU + CUDA and will run `rr-server` (and optionally
  the client too). → [Full build](#full-build-rr-server--rr-client), two steps.

**Extra system packages** (install before building):
- ffmpeg — H.264 encoding/decoding (`libavcodec`, `libavutil`, `libswscale`)
- ngtcp2 + OpenSSL — QUIC transport
- GLFW — the client's window (viewer-only builds; full builds compile their own)
- On Arch: `pacman -S ffmpeg libngtcp2 openssl glfw`

### Viewer only (no NVIDIA GPU / CUDA needed)

`rr-client` depends only on the wire protocol + the packages above, so it builds anywhere:

```sh
./samples-build-from-zero.sh --rr-client-only    # from the repository root
```

**Or, manually:**
```sh
cmake -B samples/remote-rendering/build -S samples/remote-rendering/ -DMIMIR_RR_CLIENT_ONLY=ON
cmake --build samples/remote-rendering/build -j
```

That's it — skip ahead to [Running](#running). The same flag also works at the repository root
(`./mimir-build-from-zero.sh --rr-client-only`), where it builds/installs the identical
standalone `mimir-client` instead. H.264 decoding falls back to ffmpeg's software decoder when
there is no NVDEC, so integrated graphics are enough.

### Full build (rr-server + rr-client)

> **New here?** Start from the repository root [`README.md`](../../README.md) to build the
> mimir library first, then come back here. `rr-server` cannot be built without it.

**Step 1 — build the mimir library** from the repository root with the remote rendering flags:

```sh
./mimir-build-from-zero.sh --remote --quic   # add --gcc 14 on Arch Linux / GCC 16 systems
```

**Or, manually:**
```sh
cmake -B build -DMIMIR_ENABLE_REMOTE=ON -DMIMIR_ENABLE_QUIC=ON
cmake --build build -j
```

- `--remote` / `-DMIMIR_ENABLE_REMOTE=ON` enables H.264 encoding via ffmpeg/NVENC. Without it the server streams raw frames.
- `--quic` / `-DMIMIR_ENABLE_QUIC=ON` enables the QUIC transport via ngtcp2. Without it the server is TCP-only.

**Step 2 — build this sample:**

```sh
./samples-build-from-zero.sh --sample remote-rendering   # add --gcc 14 if used in step 1
```

**Or, manually:**
```sh
cmake -B samples/remote-rendering/build -S samples/remote-rendering/ -Dmimir_DIR=$(pwd)/build/lib/mimir
cmake --build samples/remote-rendering/build -j
```

- `rr-client` is built automatically when ngtcp2 + ffmpeg + OpenGL + GLFW are all found; otherwise
  it is skipped and only `rr-server` is built.

Binaries land in `samples/remote-rendering/build/`. Run them from there.

## Running

```
rr-server [port] [width] [height] [point_count] [h264] [transport] [token] [options]
rr-client [host] [port] [token] [auto|quic|tcp] [frames]
```

`rr-server` also takes named options after the positional args — `--light-model`, `--spp`,
`--bounces`, `--subdiv`, `--size`, `--pcolor`, `--background`, `--seed`, `--k`, `--epsilon`,
`--max-steps` (run `./rr-server --help` for the full list).

### Local, raw frames (simplest, no optional deps)

```sh
./rr-server 9000 1280 720 10000 0          # raw, TCP, phong (default)
./rr-client 127.0.0.1 9000                 # interactive window
```

### Pick a render path

```sh
./rr-server 9000 1280 720 50000 1 --light-model phong-mesh       # instanced meshes
./rr-server 9000 1280 720 50000 1 --light-model path-tracing --spp 2   # Vulkan RT
```

### Local, H.264 over TCP

```sh
./rr-server 9000 1280 720 50000 1          # h264=1, TCP
./rr-client 127.0.0.1 9000                 # auto transport (QUIC then TCP)
```

### H.264 over QUIC (preferred for a direct internet connection)

```sh
./rr-server 9000 1920 1080 50000 1 quic
./rr-client <server-ip> 9000 "" quic
```

### With an auth token (recommended for any exposed port)

```sh
./rr-server 9000 1280 720 10000 1 quic mysecret
./rr-client <server-ip> 9000 mysecret quic
```

A client with a wrong/missing token is rejected; the server keeps serving others.

### Through an SSH tunnel (no open firewall port)

SSH tunnels are TCP, so force the TCP transport:

```sh
ssh -L 9000:localhost:9000 user@cluster        # on the laptop
./rr-server 9000 1280 720 10000 1 tcp          # on the cluster
./rr-client 127.0.0.1 9000 "" tcp              # on the laptop
```

(`auto` also works — it will just time out on QUIC after ~3 s and fall back to TCP. Passing `tcp`
skips that wait.)

### Headless test mode (no display)

Receive N frames, save the last one to `rr-client.ppm`, and exit — handy for CI or a server with no
X:

```sh
./rr-client 127.0.0.1 9000 "" auto 60
```

## Controls (interactive window)

| Input              | Action          |
|--------------------|-----------------|
| Left-drag          | Rotate camera   |
| Right-drag         | Zoom            |
| Middle-drag        | Pan             |
| `P`                | Pause/resume sim|
| `H`                | Toggle the HUD  |
| `Q` / `Esc`        | Quit            |

A minimal HUD overlay in the top-left corner shows where the simulation is running
(`user@host:port`, transport, codec, resolution), the end-to-end latency, the stream fps, and the
simulation progress — `step x of y` when the server was started with `--max-steps N`, or
`step x of unlimited` for an endless run.

The window is freely resizable; the frame is **stretched** to fill it (it may look softer when
enlarged). The client never asks the server to re-render at a new resolution — the server's render
cost stays fixed. (The server *can* resize on request — that path is wired and tested — but no
bundled client uses it.)

## Notes

- **Client decoding** auto-selects **NVDEC** (`h264_cuvid`) when the client has an NVIDIA GPU and it
  initializes; otherwise it uses ffmpeg's **software** H.264 decoder. Either way it's decoding the
  same standard bitstream the server's NVENC produced. The chosen decoder is printed at startup.
- **Bandwidth:** H.264 is ~150–180× smaller than raw BGRA (e.g. ~20 KB vs 3.6 MB per 1280×720
  frame). Use `h264=1` for anything beyond localhost/LAN.
- **Encryption:** QUIC uses TLS 1.3 with an ephemeral self-signed certificate (encryption without
  authentication — pair it with a token, or rely on the SSH tunnel's encryption for that path).
- **Live stats:** the server reports fps / bitrate / encode latency ~once per second; the client
  prints them as `[stats] …`.
