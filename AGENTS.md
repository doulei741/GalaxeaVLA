# G0.5 maintained model Fork

Read PROJECT_MEMORY.md before continuing. Canonical cross-project memory is
doulei741/A1Z_Lerobot/docs/G05_PROJECT_MEMORY.md. Branch exp/rtc is inference-time
RTC only. No training-time conditioning or automatic training/hardware tests.
Preserve upstream license headers and mark modified files. Latent autograd is not
parameter optimization. Absolute prefixes must pass the real trained processor
against the new observation. Never bypass normalization or add state twice.
Verify analytical gradients, baseline equivalence with RTC absent, and recorded
GPU inference; hardware experiments belong to the user. Push only to the user's
Fork, then update the runtime submodule commit. Do not modify the old source tree.
