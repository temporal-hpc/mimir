# Remote rendering sample

A headless GPU server renders a live CUDA workload and streams it to a thin native client over the
network; the client displays the frames and sends interaction (camera, pause) back. This models a
researcher on a laptop driving a visualization that runs on a remote GPU box (cluster node,
workstation) over the internet.

Two binaries, built from this directory:

- **`rr-server`** — renders headless with mimir, encodes (H.264 via NVENC, or raw), and streams to
  one client at a time. Links the `mimir` library + CUDA.
- **`rr-client`** — a thin viewer: receives the stream, decodes it, shows it in a window, and sends
  control back. Depends only on the wire protocol + ngtcp2 + OpenSSL + ffmpeg + GLFW/OpenGL — **no
  mimir, CUDA, or Vulkan**, so it runs on a GPU-less laptop.

`rr-client` is **workload-agnostic** — it knows nothing about the point-cloud server, so the *same*
client views any mimir server built with `serveRemote()`. The build also installs this exact program
as a standalone tool, **`mimir-client`** (top-level `client/`), so you can `cmake --install` it and
run `mimir-client <host> <port>` against your own server without rebuilding the samples.

## What it renders

A **3D Brownian-motion point cloud**: `point_count` particles start at random positions in the unit
cube and each frame take a small Gaussian random-walk step (drawn with cuRAND, clamped to the cube),
drawn as shaded sphere impostors. It's continuously moving, which keeps the encoder honest, and the
particle buffer lives on the GPU via mimir's CUDA interop.

## Building

The client (and the server's H.264/QUIC paths) need optional dependencies, enabled at configure
time:

```sh
cmake -S . -B build -G Ninja -DMIMIR_ENABLE_REMOTE=ON -DMIMIR_ENABLE_QUIC=ON
cmake --build build --target rr-server rr-client
```

- `-DMIMIR_ENABLE_REMOTE=ON` — H.264 encoding via ffmpeg/NVENC (pkg-config `libavcodec` …). Without
  it the server streams raw frames.
- `-DMIMIR_ENABLE_QUIC=ON` — QUIC transport via ngtcp2. Without it the server is TCP-only.
- `rr-client` builds whenever ngtcp2 + ffmpeg + OpenGL are found (independent of the flags above);
  otherwise it's skipped and only `rr-server` is built.

Binaries land in `build/samples/` and expect the slang shaders there (the `copy_shaders` target
handles this for the server). Run them from `build/samples/`.

## Running

```
rr-server [port] [width] [height] [point_count] [h264] [transport] [token]
rr-client [host] [port] [token] [auto|quic|tcp] [frames]
```

### Local, raw frames (simplest, no optional deps)

```sh
./rr-server 9000 1280 720 10000 0          # raw, TCP
./rr-client 127.0.0.1 9000                 # interactive window
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
| `Q` / `Esc`        | Quit            |

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
