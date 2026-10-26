# G0.5 model Fork and RTC adaptation memory

Maintained Fork: doulei741/GalaxeaVLA. Upstream: OpenGalaxea/GalaxeaVLA.
Canonical management instructions:
https://github.com/doulei741/A1Z_Lerobot/blob/main/docs/G05_PROJECT_MEMORY.md

Baseline branch baseline/a1z-fm-only contains the original local A1Z training,
loader, reader and inference compatibility changes preserved at 5ea8c7f.
They are historical baseline changes, not new RTC training. Current branch exp/rtc
adds **inference-time** FM guidance only; training-time RTC is out of scope.

Runtime calls PolicyInferencer.infer_rtc with an absolute grouped action prefix.
The trained processor re-anchors it against the NEW observation state, then applies
the original normalization, merger and padding. ContextVar scopes the normalized
prefix to one locked inference transaction; exceptions clear the context.

FMHelper delegates ordinary inference exactly as before when no RTC context is
active. RTC enables autograd only on the current action latent, BEFORE calling the
velocity network, and computes a full VJP including its Jacobian. Model parameters
are not optimized and .grad is not accumulated. The committed prefix follows its
noise-to-data path and is restored exactly at the final step; a soft suffix uses
bounded guidance (default5). Both pi and galaxea time conventions are handled.

Existing LeRobot RTC code was read as a reference, not copied blindly: its current
ordering sets requires_grad after evaluating v_t, so the new implementation has an
analytical test that would fail if only the identity part of the endpoint were
differentiated. Tests also cover padding, conventions, context cleanup and prefix
projection. These checks do not establish physical task success.

G0.5 inference/server is pinned by the standalone runtime Fork. Push model commits
before runtime references. Preserve upstream licenses and do not push to upstream.
No robot/camera/CAN execution or model training is authorized by this document.
