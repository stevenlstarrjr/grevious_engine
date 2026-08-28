# Reference Repositories

These pinned submodules support renderer validation and research. They are not
linked runtime dependencies and must not dictate the renderer's architecture.

| Path | Role |
|---|---|
| `glTF-Sample-Assets` | Khronos integration-test models and textures |
| `pbr-book` | Physically Based Rendering theory and equation reference |
| `pbrt-v4` | PBRT implementation reference and provenance source |
| `pbrt-v4-scenes` | Offline PBRT scene configurations and future conversion/stress resources |

Initialize renderer dependencies and the currently required glTF test assets:

```bash
git submodule update --init --recursive thirdparty reference/glTF-Sample-Assets
```

Initialize the PBRT references when doing rendering research or provenance
work:

```bash
git submodule update --init reference/pbr-book reference/pbrt-v4
```

`pbrt-v4-scenes` can be large, especially when Git LFS content is downloaded.
Initialize it only when those scene resources are needed:

```bash
git submodule update --init reference/pbrt-v4-scenes
```

Source format is not runtime format. PBRT scenes are not loaded directly by the
real-time renderer. glTF sample assets are test inputs, not content shipped by
the renderer.
