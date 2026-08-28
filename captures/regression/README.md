# PBR regression baseline

`damaged_helmet_hangar.png` is the initial deterministic raster-PBR baseline.
It was captured at 1280×720 on Intel Iris Xe (ADL GT2), frame 8, validation
required, Hangar Interior 4K, 0° environment rotation, and −1 EV exposure.

Run `tools/capture_pbr_regressions.sh` under a Wayland session to generate
`captures/current/damaged_helmet_hangar.png` and compare it using normalized
RMSE. `PBR_MAX_RMSE` overrides the default 0.025 tolerance for a certified GPU
configuration. Expected cross-vendor differences must be reviewed visually
before establishing vendor-specific thresholds or baselines.

Baseline SHA-256:

```text
9b60d30df5bb93b63acb03f0f1d1342ff9f371e5c05b27f9c8a5802084f9c476
```
