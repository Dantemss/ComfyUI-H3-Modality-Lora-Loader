"""H3 Modality LoRA Loader.

Applies multiple MiniMax H3 LoRAs as masked additive deltas with per-row LoRA
strength plus global audio/video/text strengths.

Each supported Linear target gets a forward wrapper that adds the batched
LoRA delta: attention/MLP projections (``qkv_proj``/``out_proj``/``fc1``/
``fc2``) are scaled per packed-sequence row by a modality mask, block and
final-layer ``adaln_proj.linear`` targets by a mask over the unique-
timestep rows of their ``t_emb`` input, token-refiner targets by the
scalar text strength, and the modality-dedicated projections (``video_patch_proj``/
``audio_patch_proj``/``video_out``/``audio_out``/``condition_proj``) by
their single global strength.  A single top-level ``diffusion_model.forward``
wrapper publishes the row mask and the timestep mask by re-deriving the same
packed-sequence layout and unique timesteps the model embeds, and shares them
with the Linear wrappers through a per-node ``MaskContext`` that keeps each
mask on the device it is applied on.  Strengths that are equal across every
modality a mask would separate are constant for that target, so they fold into
the scalar scale and skip publishing a mask for it.  Chained loaders compose
additively because every wrapper wraps the previous object patch instead of
overwriting it.
"""

import json
import logging

import torch
import folder_paths
import comfy.utils
from comfy.ldm.minimax.model import (
    AUDIO_COND_TIMESTEP,
    PackedLayout,
    VISUAL_COND_TIMESTEP,
    time_shift_sigma,
)


# mod_segments adaln_row tag -> global strength slot order
_TAG_VIDEO, _TAG_TEXT, _TAG_AUDIO = 0, 1, 2

# packed-sequence segment kind -> global strength slot order
_SEG_TAG = {
    "cond": _TAG_VIDEO,
    "ref_img": _TAG_VIDEO,
    "video": _TAG_VIDEO,
    "text": _TAG_TEXT,
    "cond_audio": _TAG_AUDIO,
    "ref_audio": _TAG_AUDIO,
    "audio": _TAG_AUDIO,
}

_LORA_DOWN_SUFFIX = ".lora_down.weight"
_PEFT_DOWN_SUFFIX = ".lora_A.weight"
_LORA_UP_SUFFIX = ".lora_up.weight"
_PEFT_UP_SUFFIX = ".lora_B.weight"
_ALPHA_SUFFIX = ".alpha"

_SUPPORTED_LINEAR_NAMES = {"qkv_proj", "out_proj", "fc1", "fc2", "linear",
                           "video_patch_proj", "audio_patch_proj", "video_out",
                           "audio_out", "condition_proj"}

# LoRA target kinds
_KIND_ROWS = "rows"
_KIND_T_EMB = "t_emb"
_KIND_REFINER = "refiner"
_KIND_VIDEO_UNIFORM = "video_uniform"
_KIND_AUDIO_UNIFORM = "audio_uniform"
_KIND_TEXT_UNIFORM = "text_uniform"
# target whose global strengths are all equal, so the mask is a constant that
# folds into the delta and no mask is published for it
_KIND_UNIFORM = "uniform"

# kinds that scale per row of their input and therefore need a published mask
_MASKED_KINDS = (_KIND_ROWS, _KIND_T_EMB)


def _mask_runtime(cached, mask, x):
    """Return ``(mask, key, device_mask)``, reusing ``cached`` when applicable.

    ``Tensor.to`` is a no-op that returns the same tensor when device and dtype
    already match, so the lookup stays cheap while a real device or dtype change
    costs exactly one copy.  The entry is keyed on the identity of the published
    mask tensor as well, which is what drops a cache entry once the mask is
    refilled, and the device and dtype come from ``x`` -- the tensor the delta is
    added to -- so the mask can never be prepared for a device the add is not
    happening on.
    """
    key = (x.device.type, x.device.index, x.dtype)
    if cached is not None and cached[0] is mask and cached[1] == key:
        return cached
    return (mask, key, mask.to(device=x.device, dtype=x.dtype))


class MaskContext:
    """Per-node holder for the LoRA scaling masks.

    Written by the top-level ``diffusion_model.forward`` wrapper and read by
    the Linear-forward wrappers.  ``mask`` is a 1-D float tensor with one
    entry per packed-sequence row (attention/MLP targets) and
    ``timestep_mask`` is a 1-D float tensor with one entry per unique
    timestep row of the ``t_emb`` input consumed by the block and final-layer
    adaln, in the sorted order the model embeds them; each entry holds the
    global strength of the modality family that row modulates.  Both stay
    ``None`` until the first wrapped forward runs, and stay unset for targets
    whose strengths folded into the delta.

    ``rows`` and ``timestep`` hand the Linear wrappers the mask matched to the
    tensor they are scaling and keep the one device-resident copy those
    wrappers share, so the copy to the compute device happens once per device
    or dtype change instead of once per masked Linear per step.
    """

    def __init__(self):
        self.mask = None
        self.timestep_mask = None
        self._rows = None
        self._timestep = None

    def rows(self, x):
        if self.mask is None:
            return None
        self._rows = _mask_runtime(self._rows, self.mask, x)
        return self._rows[2]

    def timestep(self, x):
        if self.timestep_mask is None:
            return None
        self._timestep = _mask_runtime(self._timestep, self.timestep_mask, x)
        return self._timestep[2]


def _parse_stack_data(stack_data):
    """Parse the ``stack_data`` JSON into enabled LoRA slot specs."""
    if not stack_data:
        return []
    try:
        raw = json.loads(stack_data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    slots = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if not entry.get("on", False):
            continue
        lora = entry.get("lora")
        if not lora or lora == "None":
            continue
        try:
            strength = float(entry.get("str", 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        slots.append({"lora": lora, "str": strength})
    return slots


def _lora_module_paths(sd):
    """Group LoRA down/up tensors by their target Linear module path.

    Accepts both the ``lora_down``/``lora_up`` and ``lora_A``/``lora_B`` key
    conventions.  Returns a dict mapping module path (e.g.
    ``diffusion_model.blocks.0.attn.qkv_proj``) to a ``(down, up, alpha)``
    tuple.  ``down`` has shape ``[rank, in]`` and ``up`` has shape
    ``[out, rank]``.
    """
    grouped = {}
    for key, value in sd.items():
        if key.endswith(_LORA_DOWN_SUFFIX):
            path = key[: -len(_LORA_DOWN_SUFFIX)]
            grouped.setdefault(path, {})["down"] = value
        elif key.endswith(_PEFT_DOWN_SUFFIX):
            path = key[: -len(_PEFT_DOWN_SUFFIX)]
            grouped.setdefault(path, {})["down"] = value
        elif key.endswith(_LORA_UP_SUFFIX):
            path = key[: -len(_LORA_UP_SUFFIX)]
            grouped.setdefault(path, {})["up"] = value
        elif key.endswith(_PEFT_UP_SUFFIX):
            path = key[: -len(_PEFT_UP_SUFFIX)]
            grouped.setdefault(path, {})["up"] = value
        elif key.endswith(_ALPHA_SUFFIX):
            path = key[: -len(_ALPHA_SUFFIX)]
            grouped.setdefault(path, {})["alpha"] = value
    specs = {}
    for path, parts in grouped.items():
        if "down" not in parts or "up" not in parts:
            continue
        down = parts["down"]
        up = parts["up"]
        if not torch.is_tensor(down) or not torch.is_tensor(up):
            continue
        if down.ndim != 2 or up.ndim != 2:
            continue
        alpha = parts.get("alpha")
        if torch.is_tensor(alpha):
            alpha = float(alpha.detach().flatten()[0].item())
        elif alpha is not None:
            alpha = float(alpha)
        else:
            alpha = float(down.shape[0])
        specs[path] = (down, up, alpha)
    return specs


def _build_rows_mask(layout, video, text, audio):
    """Build the per-position modality mask from the packed-sequence layout.

    ``layout.segments`` is a list of ``(start, end, kind)`` tuples covering
    ``[0, seq_len)``.  Every segment's rows take the global strength of the
    modality mapped from its kind.
    """
    strengths = (video, text, audio)
    mask = torch.zeros(layout.seq_len, dtype=torch.float32)
    for start, end, kind in layout.segments:
        mask[start:end] = strengths[_SEG_TAG[kind]]
    return mask


def _build_timestep_mask(dit, timestep, layout, video, audio, transformer_options=None, payload=None):
    """Build the mask over the unique-timestep rows of the block and final adaln.

    The block and final ``adaln_proj.linear`` consume one row per unique timestep, in
    the sorted order the model embeds them (row i <-> ``unique_t[i]``).  Each
    row modulates exactly one modality family: the video target rows at
    ``t_v``, the audio target rows at ``t_a``, the visual condition rows at
    ``max(t_v, vis_aug)`` and the audio condition rows at ``max(t_a,
    aud_aug)``, so each row is scaled by that family's global strength.
    """
    if timestep is None:
        return None
    transformer_options = transformer_options or {}
    payload = payload or {}
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", dit.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", dit.sigma_shift_audio))
    sigma_v = max(float(timestep.flatten()[0]) / 1000.0, 1e-6)
    t_v = 1.0 - sigma_v
    t_a = 1.0 - time_shift_sigma(sigma_v, shift_v, shift_a)
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    has_vis_cond = layout is not None and any(k in ("cond", "ref_img") for _, _, k in layout.segments)
    has_aud_cond = layout is not None and any(k in ("cond_audio", "ref_audio") for _, _, k in layout.segments)
    per_value = [(t_v, video), (t_a, audio)]
    if has_vis_cond:
        per_value.append((max(t_v, vis_aug), video))
    if has_aud_cond:
        per_value.append((max(t_a, aud_aug), audio))
    rows = dict(per_value)
    return torch.tensor([rows[v] for v in sorted(rows)], dtype=torch.float32)


def _make_forward_wrapper(prev_forward, mask_context, dit, video, text, audio,
                          need_rows=True, need_timestep=True):
    """Wrap ``diffusion_model.forward`` to publish the LoRA scaling masks.

    Re-derives the packed-sequence layout exactly the way the model does, so
    the row mask and the final-layer timestep mask always match the tensors
    the wrapped Linear targets consume.  Masks with no remaining target are
    never built, the row mask is only rebuilt when the layout it was built
    from is no longer the one being embedded, and the timestep mask is
    rebuilt every step because which unique timesteps it holds depends on the
    step's sigma.
    """
    layout_used = []

    def wrapper(x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        payload = minimax_payload or {}
        if need_rows or need_timestep:
            video_x, audio_x = x[0], x[1]
            # round t/h/w up the way the model's patch padding does, so the layout
            # signature matches the one the model embeds
            pt, ph, pw = dit.patch_size
            dims = tuple(int(d + (p - d % p) % p) for d, p in zip(video_x.shape[2:5], (pt, ph, pw)))
            signature = (context.shape[1], *dims, audio_x.shape[-1])
            layout = payload.get("layout")
            if layout is None or layout.signature != signature:
                layout = PackedLayout(*signature, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
            if need_rows and (not layout_used or layout_used[0] is not layout):
                mask_context.mask = _build_rows_mask(layout, video, text, audio)
                layout_used[:] = [layout]
            if need_timestep:
                mask_context.timestep_mask = _build_timestep_mask(dit, timestep, layout, video, audio, transformer_options, payload)
        return prev_forward(x, timestep, context, transformer_options=transformer_options,
                            minimax_payload=minimax_payload, **kwargs)
    wrapper._h3_modality_lora = True
    return wrapper


def _make_linear_wrapper(prev_forward, mask_context, a_cat, b_cat, kind, scale):
    """Wrap a supported Linear forward to add the scaled batched LoRA delta.

    ``a_cat`` has shape ``[sum(rank), in]`` (down matrices scaled by
    ``alpha/rank * row_strength``) and ``b_cat`` has shape ``[out, sum(rank)]``.
    The scaling happens on the ``[rows, sum(rank)]`` bottleneck and the result
    is accumulated straight into the Linear's own output with ``addmm_``, which
    keeps the mask multiply off the ``[rows, out]`` output and folds scalar
    strengths into the gemm instead of a second elementwise pass:

    - ``_KIND_ROWS`` scales per packed-sequence row by ``mask_context.mask``.
    - ``_KIND_T_EMB`` scales per unique-timestep row of the ``t_emb`` input
      (block and final-layer adaln) by ``mask_context.timestep_mask``.
    - every other kind (``_KIND_REFINER``, the modality-dedicated
      ``video_patch_proj``/``audio_patch_proj``/``video_out``/``audio_out``/
      ``condition_proj`` targets, and ``_KIND_UNIFORM`` once the global
      strengths are equal for the target) scales the whole delta by the single
      global ``scale`` of the modality whose rows the Linear consumes.

    ``prev_forward`` is always a ``comfy.ops.Linear`` forward or another one of
    these wrappers, so ``base`` is a freshly allocated tensor that is safe to
    accumulate into in place.
    """
    cache = {}

    def wrapper(x):
        base = prev_forward(x)
        if a_cat is None:
            return base
        key = (x.device.type, x.device.index, x.dtype)
        runtime = cache.get(key)
        if runtime is None:
            runtime = (a_cat.to(device=x.device, dtype=x.dtype).T,
                       b_cat.to(device=x.device, dtype=x.dtype).T)
            cache[key] = runtime
        a_t, b_t = runtime
        if kind not in _MASKED_KINDS:
            return base.addmm_(x @ a_t, b_t, alpha=scale)
        if kind == _KIND_ROWS:
            mask = mask_context.rows(x)
        else:
            mask = mask_context.timestep(x)
        if mask is None or mask.shape[0] != x.shape[0]:
            return base
        return base.addmm_((x @ a_t).mul_(mask[:, None]), b_t)
    wrapper._h3_modality_lora = True
    return wrapper


def _get_lora_path(lora_name):
    if hasattr(folder_paths, "get_full_path_or_raise"):
        return folder_paths.get_full_path_or_raise("loras", lora_name)
    path = folder_paths.get_full_path("loras", lora_name)
    if path is None:
        raise FileNotFoundError(f"LoRA not found in ComfyUI loras folder: {lora_name}")
    return path


class H3ModalityLoraLoader:
    """H3 Modality LoRA Loader node."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"description": "Base H3 model that will receive the modality LoRA stack."}),
                "stack_data": ("STRING", {"default": "[]", "multiline": False, "description": "JSON-encoded LoRA slot data managed by the custom UI."}),
                "audio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "description": "Global audio modality strength."}),
                "video": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "description": "Global video modality strength."}),
                "text": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "description": "Global text modality strength."}),
            },
            "hidden": {
                "available_loras": (lora_list, {"description": "Internal list of LoRA files used by the custom slot picker."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_stack"
    CATEGORY = "loaders/lora"

    def apply_stack(self, model, stack_data="[]", audio=1.0, video=1.0, text=1.0, available_loras=None):
        slots = _parse_stack_data(stack_data)
        if not slots:
            return (model,)
        diffusion_model = model.model.diffusion_model
        if diffusion_model.__class__.__name__ != "MiniMaxH3Model":
            raise ValueError("H3 Modality LoRA Loader requires a MiniMax H3 diffusion model.")

        # Build a key_map to convert LoRA file keys (underscore format) to actual model paths (dot format)
        key_map = self._build_key_map(diffusion_model)

        # Group LoRA specs across all slots by target Linear module path.
        layer_specs = {}
        for slot in slots:
            sd = comfy.utils.load_torch_file(_get_lora_path(slot["lora"]), safe_load=True)
            for path, (down, up, alpha) in _lora_module_paths(sd).items():
                # Convert path using key_map (handles both underscore and dot formats)
                actual_path = key_map.get(path, path)
                rank = int(down.shape[0])
                scale = (alpha / rank) if rank else 0.0
                layer_specs.setdefault(actual_path, []).append((down, up, scale, slot["str"]))

        supported, unsupported = {}, set()
        for path, specs in layer_specs.items():
            if path.rsplit(".", 1)[-1] in _SUPPORTED_LINEAR_NAMES:
                supported[path] = specs
            else:
                unsupported.add(path.rsplit(".", 1)[-1])
        if unsupported:
            logging.warning(
                "H3 Modality LoRA Loader: skipping unsupported LoRA targets: "
                + ", ".join(sorted(unsupported))
            )
        if not supported:
            raise ValueError("No supported H3 LoRA Linear layers found in the provided LoRAs.")

        patched = model.clone()
        mask_context = MaskContext()
        # A strength that is equal across every modality a mask would separate is
        # constant for that target, so it rides the scalar scale instead of a mask:
        # equal audio/video/text drops both masks, equal audio/video drops the
        # timestep mask, whose rows only ever hold those two families.
        all_equal = video == text == audio
        video_eq_audio = video == audio
        need_rows = need_timestep = False
        for path, specs in supported.items():
            a_cat = torch.cat([down * (scale * row_str) for down, up, scale, row_str in specs], dim=0)
            b_cat = torch.cat([up for _, up, _, _ in specs], dim=1)
            kind, scale = self._resolve_target_kind(path, video, text, audio)
            if all_equal:
                kind, scale = _KIND_UNIFORM, video
            elif kind == _KIND_T_EMB and video_eq_audio:
                kind, scale = _KIND_UNIFORM, video
            need_rows = need_rows or kind == _KIND_ROWS
            need_timestep = need_timestep or kind == _KIND_T_EMB
            existing = patched.object_patches.get(path + ".forward")
            prev_forward = existing if getattr(existing, "_h3_modality_lora", False) else self._resolve_linear_forward(diffusion_model, path)
            patched.add_object_patch(path + ".forward", _make_linear_wrapper(
                prev_forward, mask_context, a_cat, b_cat, kind, scale))

        # The row mask and the timestep mask are published by a single
        # top-level wrapper so every Linear target shares one mask source;
        # stacks of only scalar-scaled targets need no mask and no wrapper.
        if need_rows or need_timestep:
            existing = patched.object_patches.get("diffusion_model.forward")
            prev_forward = existing if getattr(existing, "_h3_modality_lora", False) else diffusion_model.forward
            patched.add_object_patch("diffusion_model.forward", _make_forward_wrapper(
                prev_forward, mask_context, diffusion_model, video, text, audio,
                need_rows=need_rows, need_timestep=need_timestep))

        return (patched,)

    @staticmethod
    def _build_key_map(diffusion_model):
        """Build a key_map to convert LoRA file keys to actual model paths.
        
        Maps underscore format keys (e.g. ``lora_unet_blocks_0_attn_out_proj``)
        to dot format paths (e.g. ``diffusion_model.blocks.0.attn.out_proj``).
        """
        key_map = {}
        for k in diffusion_model.state_dict().keys():
            if k.endswith(".weight"):
                # Build the actual model path with diffusion_model prefix
                path = "diffusion_model." + k[:-len(".weight")]
                # Build the lora_unet key from the module path
                key_lora = k[:-len(".weight")].replace(".", "_")
                # Map lora_unet_ prefix format to actual path
                key_map["lora_unet_{}".format(key_lora)] = path
        return key_map

    @staticmethod
    def _resolve_target_kind(path, video, text, audio):
        parts = path.split(".")
        if ".token_refiner." in path:
            return _KIND_REFINER, text
        dedicated = {
            "video_patch_proj": (_KIND_VIDEO_UNIFORM, video),
            "audio_patch_proj": (_KIND_AUDIO_UNIFORM, audio),
            "video_out": (_KIND_VIDEO_UNIFORM, video),
            "audio_out": (_KIND_AUDIO_UNIFORM, audio),
            "condition_proj": (_KIND_TEXT_UNIFORM, text),
        }
        if parts[-1] in dedicated:
            return dedicated[parts[-1]]
        if path.endswith(".adaln_proj.linear"):
            return _KIND_T_EMB, None
        return _KIND_ROWS, None

    @staticmethod
    def _resolve_linear_forward(diffusion_model, path):
        obj = diffusion_model
        for part in path.split(".")[1:]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj.forward
