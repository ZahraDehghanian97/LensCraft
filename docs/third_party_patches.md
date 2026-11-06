# Local third-party patches

LensCraft keeps its upstream submodules at the commits below. The patch files
under [`patches/third_party/`](../patches/third_party/) preserve the local baseline
compatibility fixes in the main repository. A normal LensCraft commit includes
these patch files; it does not include edits inside a nested submodule checkout.

| Submodule | Pinned upstream commit | Patch |
| --- | --- | --- |
| DIRECTOR | `62f805d9aeaf411f39b4a5aec19c15f67654ec26` | [`director-clatr-checkpoint-loading.patch`](../patches/third_party/director-clatr-checkpoint-loading.patch) |
| GenDoP | `26099d0b5bdb5307701fd81cb41ab2ff0acea0c8` | [`gendop-text-conditioning-and-eos.patch`](../patches/third_party/gendop-text-conditioning-and-eos.patch) |

The DIRECTOR patch changes `clatr/src/models/clatr.py` to pass
`weights_only=False` explicitly when loading the existing CLaTr checkpoint.

The GenDoP patch changes `core/models.py` to load its tokenizer and text encoder
directly, use the `truncation` tokenizer argument, and remove a final EOS token
before decoding coordinate tokens. The text and image-plus-text branches use
the existing local model cache. It also preserves the current file's removal
of trailing whitespace.

The deleted `third_parties/GenDoP/Blender_visualization/demo/.DS_Store` is Finder
metadata. Its deletion is excluded from both code patches.

## Check and apply

Run these commands from the LensCraft repository root after initializing the
submodules. First confirm that their current commits match the table:

```sh
git -C third_parties/DIRECTOR rev-parse HEAD
git -C third_parties/GenDoP rev-parse HEAD
lenscraft_patch_root="$(pwd)/patches/third_party"
```

For DIRECTOR, a successful reverse check means the patch is already present:

```sh
git -C third_parties/DIRECTOR apply --reverse --check \
  "$lenscraft_patch_root/director-clatr-checkpoint-loading.patch"
```

If the reverse check fails, check the forward patch before applying it:

```sh
git -C third_parties/DIRECTOR apply --check \
  "$lenscraft_patch_root/director-clatr-checkpoint-loading.patch" &&
git -C third_parties/DIRECTOR apply \
  "$lenscraft_patch_root/director-clatr-checkpoint-loading.patch"
```

Repeat the same checks for GenDoP:

```sh
git -C third_parties/GenDoP apply --reverse --check \
  "$lenscraft_patch_root/gendop-text-conditioning-and-eos.patch"
```

If the reverse check fails:

```sh
git -C third_parties/GenDoP apply --check \
  "$lenscraft_patch_root/gendop-text-conditioning-and-eos.patch" &&
git -C third_parties/GenDoP apply \
  "$lenscraft_patch_root/gendop-text-conditioning-and-eos.patch"
```

If both forward and reverse checks fail, inspect the local submodule changes
and base commit before proceeding. Applying a patch changes only the submodule
worktree; it leaves its commit and index unchanged. Git will continue to mark
that submodule as modified.

## Integrity and verification

The patch SHA-256 values are:

| Patch | SHA-256 |
| --- | --- |
| `director-clatr-checkpoint-loading.patch` | `2e0bbb2c4403abb3224d46265cdf943314f05a6a9ed1e750242b75b05ea9a1c7` |
| `gendop-text-conditioning-and-eos.patch` | `f141ae2920de14a0c4e14f73a3e90fb0d186cb200fbc3d4634b994702dc84311` |

Packaging verification checks that each patch:

- passes `git apply --reverse --check` against the current submodule worktree;
- passes `git apply --check` against its upstream file in an isolated temporary
  directory;
- reproduces the current code file byte for byte when applied there;
- passes a Python syntax check and restores the upstream file when reversed.

These checks verify patch portability and syntax. They do not replace baseline
inference tests with the relevant model checkpoints.
