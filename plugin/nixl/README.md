# FlagCX → NIXL v1.1.0 Integration Patch

This directory ships `flagcx_p2p_on_nixl_v1.1.0.patch`, a patch that integrates the FlagCX P2P engine into [NIXL](https://github.com/ai-dynamo/nixl) v1.1.0 as a new backend plugin named **`FLAGCX`**.

---

## 1. What the Patch Does

The patch adds a NIXL backend plugin that wraps FlagCX's P2P engine, allowing any NIXL agent to perform one-sided RDMA `READ` between GPUs (or host buffers) using FlagCX as the transport. It is delivered as a single drop-in patch against the v1.1.0 tag.

**Files added** (under `nixl/`):

| Path | Purpose |
|------|---------|
| `src/plugins/flagcx/flagcx_backend.h` / `.cpp` | `nixlFlagcxEngine` — implements the `nixlBackendEngine` virtual interface (`registerMem`, `getConnInfo`, `loadRemoteConnInfo`, `prepXfer`, `postXfer`, `checkXfer`, `getNotifs`, …). |
| `src/plugins/flagcx/flagcx_plugin.cpp` | Plugin entry point (`nixl_plugin_init` / `nixl_plugin_fini`) that the NIXL plugin manager dlopens. |
| `src/plugins/flagcx/meson.build` | Build rules producing `libplugin_FLAGCX.so`, linked against `libflagcx`. |
| `src/plugins/flagcx/README.md` | Plugin-level usage notes inside the NIXL tree. |
| `test/gtest/plugins/flagcx/flagcx_test.cpp` + `meson.build` | gtest-based correctness tests (`BasicXfer` for `READ`). |

**Files modified:**

| Path | Change |
|------|--------|
| `meson.build` (top level) | Adds `'FLAGCX'` to `all_plugins`. |
| `src/plugins/meson.build` | Probes for `libflagcx` via `cc.find_library('flagcx', ...)`; if found and enabled, descends into `src/plugins/flagcx/`. |
| `test/gtest/plugins/meson.build` | Wires the FlagCX gtest into the test build. |
| `examples/python/expanded_two_peers.py` | Tightens the metadata-exchange handshake so the example works correctly with FlagCX. |

**Capabilities (preview release):**

- Internode GPU-to-GPU transfers over RDMA (one-sided `READ`).
- Vectorized transfers via `flagcxP2pEngineReadVector`.
- Auto NIC selection based on PCIe topology at memory-registration time.
- Out-of-band completion notifications via FlagCX's own notif channel.

---

## 2. Applying the Patch

The patch is authored against NIXL **v1.1.0**. Apply it on a fresh checkout of that tag.

```bash
# Clone upstream NIXL and check out v1.1.0
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
git checkout v1.1.0

# Apply the patch (adjust the path to where your FlagCX checkout lives)
git apply /path/to/FlagCX/plugin/nixl/flagcx_p2p_on_nixl_v1.1.0.patch

# Sanity check
git status      # should show 11 changed/added files under src/plugins/flagcx, test/gtest/plugins/flagcx, etc.
```

If you prefer a non-`git` workflow, `patch -p1 < flagcx_p2p_on_nixl_v1.1.0.patch` from the NIXL repo root works as well.

---

## 3. Installation

Two stages: build & install **FlagCX** first, then build & install **NIXL** with the FlagCX plugin enabled.

### 3.1 Build FlagCX

```bash
git clone --recursive https://github.com/FlagOpen/FlagCX.git
cd FlagCX
make USE_NVIDIA=1 -j32 # nvidia for example
```

The build produces `libflagcx.so` and the public headers under `FlagCX/build/{lib,include}/`.

(Optional) Verify the FlagCX build with the P2P unit tests:

```bash
cd test/unittest/p2p && make -j32 && ./build/bin/p2p_unit_tests
```

### 3.2 Configure environment

Create an `env.sh` and `source` it before each build/run. Adjust the prefixes to match your layout; do **not** commit credentials.

```bash
#!/bin/bash
# --- FlagCX ---
export FLAGCX_PREFIX=/path/to/FlagCX/build
export LIBRARY_PATH=$FLAGCX_PREFIX/lib:$LIBRARY_PATH
export CPATH=$FLAGCX_PREFIX/include:$CPATH
export LDFLAGS="-L$FLAGCX_PREFIX/lib -Wl,-rpath,$FLAGCX_PREFIX/lib"
export CPPFLAGS="-I$FLAGCX_PREFIX/include"

# --- NIXL install destination ---
export NIXL_PREFIX=/path/to/install_nixl
export LD_LIBRARY_PATH=$NIXL_PREFIX/lib:$FLAGCX_PREFIX/lib:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=$NIXL_PREFIX/lib/x86_64-linux-gnu/plugins
```

```bash
source env.sh
```

### 3.3 Build & install NIXL with the FlagCX plugin

```bash
cd /path/to/nixl       # the patched v1.1.0 checkout from §2

meson setup --wipe build \
  --prefix=$NIXL_PREFIX \
  -Denable_plugins=FLAGCX \
  -Dcudapath_inc=/usr/local/cuda/include \
  -Dcudapath_lib=/usr/local/cuda/lib64

ninja -C build
ninja -C build install   # installs into $NIXL_PREFIX (no sudo required)
```

Verify the plugin shared library landed in the expected location:

```bash
ls $NIXL_PREFIX/lib/x86_64-linux-gnu/plugins/libplugin_FLAGCX.so
```

### 3.4 Install the NIXL Python bindings

> Run this on every machine that will host a NIXL agent (target and initiator).

```bash
cd /path/to/nixl
pip install .
# or, equivalently:
pip install build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
```

---

## 4. Sanity Check

### 4.1 Agent instantiation smoke test

Confirms that the FLAGCX plugin loads and a NIXL agent can be created.

```bash
python3 -c "
from nixl._api import nixl_agent, nixl_agent_config
a = nixl_agent('test', nixl_agent_config(True, True, 0, backends=['FLAGCX']))
print('FLAGCX agent OK')
"
```

Expected output ends with `FLAGCX agent OK`. Logs should also show `Backend FLAGCX was instantiated`.

### 4.2 Two-peer end-to-end transfer

Run the bundled example on two hosts that share an RDMA-capable network. Replace `<HOST_A_IP>` with the target's IP.

```bash
# --- Host A (target) ---
cd /path/to/nixl
python3 examples/python/expanded_two_peers.py \
    --mode=target --backend=FLAGCX --use_cuda=true \
    --ip=<HOST_A_IP> --port=4242
```

```bash
# --- Host B (initiator) ---
cd /path/to/nixl
python3 examples/python/expanded_two_peers.py \
    --mode=initiator --backend=FLAGCX --use_cuda=true \
    --ip=<HOST_A_IP> --port=4242
```

A successful run prints, on **both** hosts:

```
NIXL INFO _api.py:369  Backend FLAGCX was instantiated
NIXL INFO _api.py:247  Initialized NIXL agent: <target|initiator>
...
NIXL INFO expanded_two_peers.py:148  Target data verification passed       # target only
NIXL INFO expanded_two_peers.py:342  Initiator final data verification passed  # initiator only
NIXL INFO expanded_two_peers.py:360  Test Complete.
```

### 4.3 (Optional) P2P throughput benchmark

For a quick perf gut-check, run the FlagCX-backed NIXL benchmark from the FlagCX repo.

```bash
source env.sh
cd /path/to/FlagCX/test/perf/p2p

# --- Server ---
python3 benchmark_flagcx.py --role=server --remote-ip=<SERVER_IP> \
    --device=gpu --local-gpu-idx=1 --op-type=read \
    --sizes=1048576,4194304,16777216,104857600,1073741824 --iters=20

# --- Client ---
python3 benchmark_flagcx.py --role=client --remote-ip=<SERVER_IP> \
    --device=gpu --local-gpu-idx=1 --op-type=read \
    --sizes=1048576,4194304,16777216,104857600,1073741824 --iters=20
```

---

## Troubleshooting

- **`flagcx library not found` during `meson setup`** — the plugin's meson rule probes `cc.find_library('flagcx')`. Make sure `LIBRARY_PATH` and `CPATH` from `env.sh` are exported in the same shell that runs `meson setup`.
- **`libplugin_FLAGCX.so: cannot open shared object file`** at runtime — confirm `LD_LIBRARY_PATH` includes both `$NIXL_PREFIX/lib` and `$FLAGCX_PREFIX/lib`, and that `NIXL_PLUGIN_DIR` points at `$NIXL_PREFIX/lib/x86_64-linux-gnu/plugins`.
- **Handshake hangs in the two-peer example** — verify the target's IP is reachable from the initiator and that the chosen `--port` is not blocked by a firewall. FlagCX uses TCP for bootstrap and the configured RDMA fabric for the data path.
