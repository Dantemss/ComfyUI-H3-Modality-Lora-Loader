import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

import comfy.ldm.minimax.model as h3_model

MODULE_PATH = Path(__file__).parents[1] / "nodes" / "h3_modality_lora_loader.py"

def load_module():
    spec = importlib.util.spec_from_file_location("h3_modality_lora_loader_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class MockLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x):
        return x @ self.weight.T


class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = MockLinear(8, 8)
        self.out_proj = MockLinear(8, 8)

    def forward(self, x, rope_freqs=None, transformer_options={}):
        return self.out_proj(self.qkv_proj(x))


class MockMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = MockLinear(8, 16)
        self.fc2 = MockLinear(16, 8)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class MockAdaLNProj(nn.Module):
    def __init__(self, out_dim=24):
        super().__init__()
        self.linear = MockLinear(8, out_dim)


class MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MockAttention()
        self.mlp = MockMLP()
        self.adaln_proj = MockAdaLNProj(24)

    def forward(self, x, t_emb, rope_freqs, transformer_options={}):
        return x + self.attn(x) + self.mlp(x)


class MiniMaxH3Model(nn.Module):
    """Forward signature, module paths and input shapes mirroring
    ``comfy.ldm.minimax.model.MiniMaxH3Model``."""

    def __init__(self, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([MockBlock() for _ in range(num_blocks)])
        self.token_refiner = nn.ModuleList([MockBlock()])
        self.video_patch_proj = MockLinear(8, 8)
        self.audio_patch_proj = MockLinear(8, 8)
        self.condition_proj = MockLinear(8, 8)
        self.final_layer = nn.Module()
        self.final_layer.adaln_proj = MockAdaLNProj(24)
        self.final_layer.video_out = MockLinear(8, 8)
        self.final_layer.audio_out = MockLinear(8, 8)
        self.sigma_shift_video = 12.0
        self.sigma_shift_audio = 3.0
        self.patch_size = (1, 2, 2)

    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        layout = (minimax_payload or {}).get("layout")
        rows = torch.ones(layout.seq_len, 8)
        for block in self.blocks:
            rows = block(rows, None, None, transformer_options)
        return rows


class MockModelPatcher:
    def __init__(self, model):
        self.model = model
        self.object_patches = {}

    def clone(self):
        n = MockModelPatcher(self.model)
        n.object_patches = self.object_patches.copy()
        return n

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj


class MockModel:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


def make_model(num_blocks=2):
    return MockModelPatcher(MockModel(MiniMaxH3Model(num_blocks)))


def apply_object_patches(model):
    diffusion_model = model.model.diffusion_model
    for name, fn in model.object_patches.items():
        parts = name.split(".")
        obj = diffusion_model
        for part in parts[1:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        setattr(obj, parts[-1], fn)


def lora_sd(path, down, up, alpha=None):
    sd = {
        f"{path}.lora_down.weight": down,
        f"{path}.lora_up.weight": up,
    }
    if alpha is not None:
        sd[f"{path}.alpha"] = alpha
    return sd


def stack_json(*slots):
    return json.dumps([{"on": True, "lora": name, "str": strength} for name, strength in slots])


def make_layout(text_len=3, latent_t=2, latent_h=2, latent_w=2, audio_t=1, keyframes=None, refs=None):
    return h3_model.PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t,
                                 keyframes=keyframes, refs=refs)


KEYFRAME = {"resolved_frame_index": 0,
            "latent": torch.zeros(1, 24, 1, 2, 2),
            "audio_latent": torch.zeros(1, 32, 2, 1)}

# text(0,3) audio(3,5) video(5,7)
LAYOUT = make_layout()
# adds cond/cond_audio keyframe rows
COND_LAYOUT = make_layout(keyframes=[KEYFRAME])
# every segment kind: text/cond/cond_audio/ref_img/ref_audio/audio/video
ALL_KINDS_LAYOUT = make_layout(keyframes=[KEYFRAME],
                               refs=[{"kind": "image", "latent_h": 2, "latent_w": 2},
                                     {"kind": "audio", "ref_audio_t": 1}])

VIDEO_X = torch.ones(1, 24, 2, 2, 2)
AUDIO_X = torch.ones(1, 32, 2, 1)
CONTEXT = torch.ones(1, 3, 8)
TIMESTEP = torch.tensor([500.0])
DIT = SimpleNamespace(sigma_shift_video=12.0, sigma_shift_audio=3.0, patch_size=(1, 2, 2))

PAYLOAD = {"layout": LAYOUT}

def test_node_class_exists():
    module = load_module()
    assert hasattr(module, "H3ModalityLoraLoader")


def test_node_io_contract():
    module = load_module()
    with mock.patch("folder_paths.get_filename_list", return_value=["a.safetensors"]):
        spec = module.H3ModalityLoraLoader.INPUT_TYPES()
    assert set(spec["required"]) == {"model", "stack_data", "audio", "video", "text"}
    assert "available_loras" in spec["hidden"]
    assert module.H3ModalityLoraLoader.RETURN_TYPES == ("MODEL",)
    assert module.H3ModalityLoraLoader.FUNCTION == "apply_stack"


def test_build_rows_mask_from_layout():
    module = load_module()
    mask = module._build_rows_mask(LAYOUT, video=1.0, text=2.0, audio=3.0)
    assert mask.dtype == torch.float32
    assert torch.equal(mask, torch.tensor([2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0]))


def test_build_rows_mask_maps_all_segment_kinds():
    module = load_module()
    mask = module._build_rows_mask(ALL_KINDS_LAYOUT, video=1.0, text=2.0, audio=3.0)
    assert torch.equal(mask, torch.tensor([2.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 3.0, 3.0, 3.0, 3.0, 1.0, 1.0]))


def test_build_timestep_mask_none_timestep():
    module = load_module()
    assert module._build_timestep_mask(DIT, None, LAYOUT, video=1.0, audio=3.0) is None


def test_build_timestep_mask_one_row_per_unique_timestep():
    module = load_module()
    # no cond rows: one row per target stream, sorted by timestep
    mask = module._build_timestep_mask(DIT, TIMESTEP, LAYOUT, video=1.0, audio=3.0)
    assert torch.allclose(mask, torch.tensor([1.0, 3.0]))
    # cond rows pin the vis/aud aug timesteps: four rows
    mask = module._build_timestep_mask(DIT, TIMESTEP, COND_LAYOUT, video=1.0, audio=3.0)
    assert mask.shape == (4,)
    assert torch.allclose(mask, torch.tensor([1.0, 3.0, 1.0, 3.0]))
    # ref blocks set the same cond flags
    mask = module._build_timestep_mask(DIT, TIMESTEP, ALL_KINDS_LAYOUT, video=1.0, audio=3.0)
    assert torch.allclose(mask, torch.tensor([1.0, 3.0, 1.0, 3.0]))


def test_build_timestep_mask_honors_shift_and_aug_overrides():
    module = load_module()
    transformer_options = {"minimax_h3_sigma_shift_video": 3.0, "minimax_h3_sigma_shift_audio": 12.0}
    payload = {"visual_cond_noise_aug": 0.9, "audio_cond_noise_aug": 0.95}
    mask = module._build_timestep_mask(DIT, TIMESTEP, COND_LAYOUT, video=1.0, audio=3.0,
                                       transformer_options=transformer_options, payload=payload)
    t_a = 1.0 - h3_model.time_shift_sigma(0.5, 3.0, 12.0)
    per_value = {0.5: 1.0, t_a: 3.0, 0.9: 1.0, 0.95: 3.0}
    assert torch.allclose(mask, torch.tensor([per_value[v] for v in sorted(per_value)], dtype=torch.float32))


def test_build_timestep_mask_dedupes_cond_values():
    module = load_module()
    # audio aug below t_a collapses the audio cond row onto the audio target row
    payload = {"audio_cond_noise_aug": 0.7}
    mask = module._build_timestep_mask(DIT, TIMESTEP, COND_LAYOUT, video=1.0, audio=3.0, payload=payload)
    assert mask.shape == (3,)
    assert torch.allclose(mask, torch.tensor([1.0, 3.0, 1.0]))


def test_linear_wrapper_a_cat_none_passthrough():
    module = load_module()
    prev = MockLinear(8, 8)
    wrapper = module._make_linear_wrapper(prev.forward, module.MaskContext(), None, None, module._KIND_ROWS, None)
    x = torch.zeros(4, 8)
    out = wrapper(x)
    assert torch.equal(out, prev(x))


def test_linear_wrapper_rows_without_mask_passthrough():
    module = load_module()
    prev = MockLinear(8, 8)
    ctx = module.MaskContext()
    wrapper = module._make_linear_wrapper(prev.forward, ctx, torch.ones(2, 8), torch.ones(8, 2), module._KIND_ROWS, None)
    x = torch.ones(4, 8)
    out = wrapper(x)
    assert torch.equal(out, prev.forward(x))


def test_linear_wrapper_rows_shape_mismatch_passthrough():
    module = load_module()
    prev = MockLinear(8, 8)
    ctx = module.MaskContext()
    ctx.mask = torch.ones(3)
    wrapper = module._make_linear_wrapper(prev.forward, ctx, torch.ones(2, 8), torch.ones(8, 2), module._KIND_ROWS, None)
    x = torch.ones(4, 8)
    out = wrapper(x)
    assert torch.equal(out, prev.forward(x))


def test_linear_wrapper_rows_applies_mask():
    module = load_module()
    prev = MockLinear(8, 8)
    ctx = module.MaskContext()
    ctx.mask = torch.tensor([1.0, 0.0, 2.0, 0.5])
    down, up = torch.ones(2, 8), torch.ones(8, 2)
    wrapper = module._make_linear_wrapper(prev.forward, ctx, down, up, module._KIND_ROWS, None)
    x = torch.ones(4, 8)
    base = prev.forward(x)
    out = wrapper(x)
    expected = base + (x @ down.T) @ up.T * torch.tensor([1.0, 0.0, 2.0, 0.5])[:, None]
    assert torch.allclose(out, expected)


def test_linear_wrapper_block_adaln_timestep_rows():
    module = load_module()
    prev = MockLinear(8, 24)
    ctx = module.MaskContext()
    ctx.timestep_mask = torch.tensor([1.0, 3.0])
    down, up = torch.ones(2, 8), torch.ones(24, 2)
    wrapper = module._make_linear_wrapper(prev.forward, ctx, down, up, module._KIND_T_EMB, None)
    x = torch.ones(2, 8)
    base = prev.forward(x)
    out = wrapper(x)
    expected = base + (x @ down.T) @ up.T * torch.tensor([1.0, 3.0])[:, None]
    assert torch.allclose(out, expected)


def test_linear_wrapper_final_adaln_timestep_rows():
    module = load_module()
    prev = MockLinear(8, 24)
    ctx = module.MaskContext()
    ctx.timestep_mask = torch.tensor([1.0, 3.0, 1.0, 3.0])
    down, up = torch.ones(2, 8), torch.ones(24, 2)
    wrapper = module._make_linear_wrapper(prev.forward, ctx, down, up, module._KIND_T_EMB, None)
    x = torch.ones(4, 8)
    base = prev.forward(x)
    out = wrapper(x)
    expected = base + (x @ down.T) @ up.T * torch.tensor([1.0, 3.0, 1.0, 3.0])[:, None]
    assert torch.allclose(out, expected)


def test_linear_wrapper_final_adaln_without_mask_passthrough():
    module = load_module()
    prev = MockLinear(8, 24)
    ctx = module.MaskContext()
    wrapper = module._make_linear_wrapper(prev.forward, ctx, torch.ones(2, 8), torch.ones(24, 2), module._KIND_T_EMB, None)
    x = torch.ones(3, 8)
    assert torch.equal(wrapper(x), prev.forward(x))


def test_linear_wrapper_uniform_kinds_scale_by_single_strength():
    module = load_module()
    for kind, strength in ((module._KIND_VIDEO_UNIFORM, 1.0),
                           (module._KIND_AUDIO_UNIFORM, 3.0),
                           (module._KIND_TEXT_UNIFORM, 2.0)):
        prev = MockLinear(8, 8)
        down, up = torch.ones(2, 8), torch.ones(8, 2)
        wrapper = module._make_linear_wrapper(prev.forward, module.MaskContext(), down, up, kind, strength)
        x = torch.ones(4, 8)
        base = prev.forward(x)
        out = wrapper(x)
        expected = base + (x @ down.T) @ up.T * strength
        assert torch.allclose(out, expected)


def test_linear_wrapper_refiner_uses_text_strength():
    module = load_module()
    prev = MockLinear(8, 8)
    ctx = module.MaskContext()
    down, up = torch.ones(2, 8), torch.ones(8, 2)
    wrapper = module._make_linear_wrapper(prev.forward, ctx, down, up, module._KIND_REFINER, 2.0)
    x = torch.ones(3, 8)
    base = prev.forward(x)
    out = wrapper(x)
    expected = base + ((x @ down.T) @ up.T) * 2.0
    assert torch.allclose(out, expected)


def test_forward_wrapper_publishes_masks_and_calls_prev():
    module = load_module()
    ctx = module.MaskContext()
    seen = {}

    def prev_forward(x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        seen.update({"x": x, "timestep": timestep, "context": context,
                     "transformer_options": transformer_options, "minimax_payload": minimax_payload,
                     "kwargs": kwargs})
        return "called"

    wrapper = module._make_forward_wrapper(prev_forward, ctx, DIT, video=1.0, text=2.0, audio=3.0)
    assert wrapper._h3_modality_lora is True
    out = wrapper((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT, transformer_options={"a": 1},
                  minimax_payload=PAYLOAD, extra=42)
    assert out == "called"
    assert seen["x"][0] is VIDEO_X and seen["x"][1] is AUDIO_X
    assert seen["timestep"] is TIMESTEP and seen["context"] is CONTEXT
    assert seen["transformer_options"] == {"a": 1}
    assert seen["minimax_payload"] is PAYLOAD
    assert seen["kwargs"] == {"extra": 42}
    assert torch.equal(ctx.mask, torch.tensor([2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0]))
    assert ctx.timestep_mask is not None and ctx.timestep_mask.shape == (2,)


def test_forward_wrapper_rebuilds_stale_or_missing_layout():
    module = load_module()
    ctx = module.MaskContext()
    wrapper = module._make_forward_wrapper(lambda *a, **k: None, ctx, DIT, video=1.0, text=2.0, audio=3.0)
    expected = module._build_rows_mask(make_layout(), video=1.0, text=2.0, audio=3.0)
    # no prebuilt layout: rebuilt from the forward inputs
    wrapper((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT, minimax_payload={})
    assert torch.equal(ctx.mask, expected)
    assert ctx.timestep_mask is not None and ctx.timestep_mask.shape == (2,)
    # a layout whose signature no longer matches is ignored, not trusted
    wrapper((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT, minimax_payload={"layout": make_layout(text_len=5)})
    assert torch.equal(ctx.mask, expected)


def test_resolve_target_kind_routing():
    module = load_module()
    cases = [
        ("diffusion_model.blocks.0.attn.qkv_proj", module._KIND_ROWS),
        ("diffusion_model.blocks.0.attn.out_proj", module._KIND_ROWS),
        ("diffusion_model.blocks.0.mlp.fc1", module._KIND_ROWS),
        ("diffusion_model.blocks.1.mlp.fc2", module._KIND_ROWS),
        ("diffusion_model.token_refiner.0.attn.qkv_proj", module._KIND_REFINER),
        ("diffusion_model.blocks.0.adaln_proj.linear", module._KIND_T_EMB),
        ("diffusion_model.final_layer.adaln_proj.linear", module._KIND_T_EMB),
        ("diffusion_model.video_patch_proj", module._KIND_VIDEO_UNIFORM),
        ("diffusion_model.audio_patch_proj", module._KIND_AUDIO_UNIFORM),
        ("diffusion_model.final_layer.video_out", module._KIND_VIDEO_UNIFORM),
        ("diffusion_model.final_layer.audio_out", module._KIND_AUDIO_UNIFORM),
        ("diffusion_model.condition_proj", module._KIND_TEXT_UNIFORM),
    ]
    for path, kind in cases:
        assert module.H3ModalityLoraLoader._resolve_target_kind(path, 1.0, 2.0, 3.0)[0] == kind
    assert module.H3ModalityLoraLoader._resolve_target_kind("diffusion_model.blocks.0.adaln_proj.linear", 1.0, 2.0, 3.0) == (module._KIND_T_EMB, None)
    assert module.H3ModalityLoraLoader._resolve_target_kind("diffusion_model.token_refiner.0.attn.qkv_proj", 1.0, 2.0, 3.0) == (module._KIND_REFINER, 2.0)
    assert module.H3ModalityLoraLoader._resolve_target_kind("diffusion_model.final_layer.video_out", 1.0, 2.0, 3.0) == (module._KIND_VIDEO_UNIFORM, 1.0)
    assert module.H3ModalityLoraLoader._resolve_target_kind("diffusion_model.condition_proj", 1.0, 2.0, 3.0) == (module._KIND_TEXT_UNIFORM, 2.0)


def test_lora_module_paths_groups_both_key_conventions():
    module = load_module()
    down = torch.ones(2, 8)
    up = torch.ones(8, 2)
    sd = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_down.weight": down,
        "diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight": up,
        "diffusion_model.blocks.0.attn.qkv_proj.alpha": 4.0,
        "diffusion_model.token_refiner.0.attn.qkv_proj.lora_A.weight": down,
        "diffusion_model.token_refiner.0.attn.qkv_proj.lora_B.weight": up,
        "diffusion_model.blocks.0.attn.out_proj.lora_down.weight": down,
    }
    specs = module._lora_module_paths(sd)
    assert set(specs) == {
        "diffusion_model.blocks.0.attn.qkv_proj",
        "diffusion_model.token_refiner.0.attn.qkv_proj",
    }
    d, u, a = specs["diffusion_model.blocks.0.attn.qkv_proj"]
    assert torch.equal(d, down) and torch.equal(u, up) and a == 4.0
    d, u, a = specs["diffusion_model.token_refiner.0.attn.qkv_proj"]
    assert torch.equal(d, down) and torch.equal(u, up) and a == 2.0


E2E_DOWN = torch.ones(2, 8)
E2E_UP = torch.ones(8, 2)
E2E_DOWN2 = torch.full((2, 8), 2.0)
E2E_UP2 = torch.full((24, 2), 3.0)


def e2e_sd():
    return {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_down.weight": E2E_DOWN,
        "diffusion_model.blocks.0.attn.qkv_proj.lora_up.weight": E2E_UP,
        "diffusion_model.blocks.0.attn.qkv_proj.alpha": 4.0,
        "diffusion_model.blocks.0.adaln_proj.linear.lora_down.weight": E2E_DOWN2,
        "diffusion_model.blocks.0.adaln_proj.linear.lora_up.weight": E2E_UP2,
        "diffusion_model.video_patch_proj.lora_down.weight": E2E_DOWN,
        "diffusion_model.video_patch_proj.lora_up.weight": E2E_UP,
        "diffusion_model.video_patch_proj.alpha": 4.0,
    }


def apply_stack(module, model, stack, loras={}, **strengths):
    with mock.patch.object(module, "_get_lora_path", side_effect=lambda name: f"fake/{name}"), \
            mock.patch.object(module.comfy.utils, "load_torch_file",
                             side_effect=lambda path, **kwargs: loras[Path(path).stem]):
        return module.H3ModalityLoraLoader().apply_stack(model, stack_json(*stack), **strengths)


class OtherModel(nn.Module):
    pass


def test_apply_stack_empty_stack_returns_model_untouched():
    module = load_module()
    model = make_model()
    out = apply_stack(module, model, [])
    assert out == (model,)
    assert not model.object_patches


def test_apply_stack_rejects_non_h3_model():
    module = load_module()
    model = MockModelPatcher(MockModel(OtherModel()))
    with pytest.raises(ValueError):
        apply_stack(module, model, [("a.safetensors", 1.0)], {"a": e2e_sd()})


DOWN_B = torch.full((2, 8), 2.0)
UP_B = torch.full((8, 2), 3.0)
MASK_EXPECTED = torch.tensor([2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 1.0])


def test_apply_stack_e2e_forward_publishes_masks():
    module = load_module()
    model = make_model()
    lora = lora_sd("diffusion_model.blocks.0.attn.qkv_proj", E2E_DOWN, E2E_UP, alpha=4.0)
    patched, = apply_stack(module, model, [("a.safetensors", 1.0)], {"a": lora},
                           video=1.0, text=2.0, audio=3.0)
    assert "diffusion_model.forward" in patched.object_patches
    apply_object_patches(patched)
    dit = patched.model.diffusion_model
    out = dit.forward((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT,
                      transformer_options={}, minimax_payload=PAYLOAD)
    assert out.shape == (LAYOUT.seq_len, 8)
    x = torch.ones(LAYOUT.seq_len, 8)
    qkv = dit.blocks[0].attn.qkv_proj
    expected = x @ qkv.weight.T + (x @ (E2E_DOWN * 2.0).T) @ E2E_UP.T * MASK_EXPECTED[:, None]
    assert torch.allclose(qkv(x), expected)


def test_apply_stack_e2e_t_emb_uses_timestep_mask():
    module = load_module()
    model = make_model()
    lora = lora_sd("diffusion_model.blocks.0.adaln_proj.linear", E2E_DOWN2, E2E_UP2)
    patched, = apply_stack(module, model, [("a.safetensors", 1.0)], {"a": lora},
                           video=1.0, text=2.0, audio=3.0)
    apply_object_patches(patched)
    dit = patched.model.diffusion_model
    dit.forward((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT,
                transformer_options={}, minimax_payload=PAYLOAD)
    x = torch.ones(2, 8)
    adaln = dit.blocks[0].adaln_proj.linear
    base = x @ adaln.weight.T
    expected = base + (x @ E2E_DOWN2.T) @ E2E_UP2.T * torch.tensor([1.0, 3.0])[:, None]
    assert torch.allclose(adaln(x), expected)


def test_apply_stack_e2e_chained_loaders_compose():
    module = load_module()
    model = make_model()
    path = "diffusion_model.blocks.0.attn.qkv_proj"
    lora_a = lora_sd(path, E2E_DOWN, E2E_UP, alpha=4.0)
    lora_b = lora_sd(path, DOWN_B, UP_B, alpha=2.0)
    first, = apply_stack(module, model, [("a.safetensors", 1.0)], {"a": lora_a}, video=1.0, text=2.0, audio=3.0)
    second, = apply_stack(module, first, [("b.safetensors", 1.0)], {"b": lora_b}, video=1.0, text=2.0, audio=3.0)
    apply_object_patches(second)
    dit = second.model.diffusion_model
    dit.forward((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT,
                transformer_options={}, minimax_payload=PAYLOAD)
    x = torch.ones(LAYOUT.seq_len, 8)
    qkv = dit.blocks[0].attn.qkv_proj
    base = x @ qkv.weight.T
    expected = base + ((x @ (E2E_DOWN * 2.0).T) @ E2E_UP.T + (x @ DOWN_B.T) @ UP_B.T) * MASK_EXPECTED[:, None]
    assert torch.allclose(qkv(x), expected)


def test_apply_stack_e2e_uniform_targets_get_no_forward_wrapper():
    module = load_module()
    model = make_model()
    lora = lora_sd("diffusion_model.video_patch_proj", E2E_DOWN, E2E_UP, alpha=4.0)
    patched, = apply_stack(module, model, [("a.safetensors", 1.0)], {"a": lora},
                           video=1.5, text=2.0, audio=3.0)
    assert "diffusion_model.forward" not in patched.object_patches
    apply_object_patches(patched)
    dit = patched.model.diffusion_model
    x = torch.ones(5, 8)
    vpp = dit.video_patch_proj
    base = x @ vpp.weight.T
    expected = base + (x @ E2E_DOWN.T) @ E2E_UP.T * 2.0 * 1.5
    assert torch.allclose(vpp(x), expected)


def test_build_key_map_maps_underscore_to_dot():
    module = load_module()
    model = make_model()
    key_map = module.H3ModalityLoraLoader._build_key_map(model.model.diffusion_model)
    # Test that underscore keys map to dot keys
    assert "lora_unet_blocks_0_attn_qkv_proj" in key_map
    assert key_map["lora_unet_blocks_0_attn_qkv_proj"] == "diffusion_model.blocks.0.attn.qkv_proj"
    assert "lora_unet_blocks_0_attn_out_proj" in key_map
    assert key_map["lora_unet_blocks_0_attn_out_proj"] == "diffusion_model.blocks.0.attn.out_proj"
    assert "lora_unet_blocks_0_mlp_fc1" in key_map
    assert key_map["lora_unet_blocks_0_mlp_fc1"] == "diffusion_model.blocks.0.mlp.fc1"
    assert "lora_unet_blocks_0_mlp_fc2" in key_map
    assert key_map["lora_unet_blocks_0_mlp_fc2"] == "diffusion_model.blocks.0.mlp.fc2"
    assert "lora_unet_blocks_0_adaln_proj_linear" in key_map
    assert key_map["lora_unet_blocks_0_adaln_proj_linear"] == "diffusion_model.blocks.0.adaln_proj.linear"
    assert "lora_unet_video_patch_proj" in key_map
    assert key_map["lora_unet_video_patch_proj"] == "diffusion_model.video_patch_proj"
    assert "lora_unet_audio_patch_proj" in key_map
    assert key_map["lora_unet_audio_patch_proj"] == "diffusion_model.audio_patch_proj"
    assert "lora_unet_condition_proj" in key_map
    assert key_map["lora_unet_condition_proj"] == "diffusion_model.condition_proj"
    assert "lora_unet_final_layer_video_out" in key_map
    assert key_map["lora_unet_final_layer_video_out"] == "diffusion_model.final_layer.video_out"
    assert "lora_unet_final_layer_audio_out" in key_map
    assert key_map["lora_unet_final_layer_audio_out"] == "diffusion_model.final_layer.audio_out"


def test_apply_stack_e2e_underscore_keys_apply_correctly():
    module = load_module()
    model = make_model()
    # Create LoRA with underscore keys
    lora = {
        "lora_unet_blocks_0_attn_qkv_proj.lora_down.weight": E2E_DOWN,
        "lora_unet_blocks_0_attn_qkv_proj.lora_up.weight": E2E_UP,
        "lora_unet_blocks_0_attn_qkv_proj.alpha": 4.0,
    }
    patched, = apply_stack(module, model, [("a.safetensors", 1.0)], {"a": lora},
                           video=1.0, text=2.0, audio=3.0)
    # Verify the LoRA was applied to the correct module
    apply_object_patches(patched)
    dit = patched.model.diffusion_model
    out = dit.forward((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT,
                      transformer_options={}, minimax_payload=PAYLOAD)
    x = torch.ones(LAYOUT.seq_len, 8)
    qkv = dit.blocks[0].attn.qkv_proj
    base = x @ qkv.weight.T
    expected = base + (x @ (E2E_DOWN * 2.0).T) @ E2E_UP.T * MASK_EXPECTED[:, None]
    assert torch.allclose(qkv(x), expected)



def test_linear_wrapper_uniform_kind_scales_by_global_strength():
    module = load_module()
    prev = MockLinear(8, 8)
    down, up = torch.ones(2, 8), torch.ones(8, 2)
    wrapper = module._make_linear_wrapper(prev.forward, module.MaskContext(), down, up, module._KIND_UNIFORM, 2.5)
    x = torch.ones(4, 8)
    expected = prev.forward(x) + (x @ down.T) @ up.T * 2.5
    assert torch.allclose(wrapper(x), expected)


def test_mask_context_reuses_device_mask_until_refilled():
    module = load_module()
    ctx = module.MaskContext()
    assert ctx.rows(torch.ones(2, 4)) is None
    assert ctx.timestep(torch.ones(2, 4)) is None
    ctx.mask = torch.tensor([1.0, 2.0])
    x = torch.ones(2, 4)
    # one shared copy per device/dtype, not one per masked Linear
    assert ctx.rows(x) is ctx.rows(x)
    assert ctx.rows(x).dtype == x.dtype
    assert ctx.rows(torch.ones(2, 4, dtype=torch.bfloat16)).dtype == torch.bfloat16
    ctx.mask = torch.tensor([3.0, 4.0])
    assert torch.equal(ctx.rows(x), torch.tensor([3.0, 4.0]))


def test_forward_wrapper_reuses_row_mask_for_unchanged_layout():
    module = load_module()
    ctx = module.MaskContext()
    wrapper = module._make_forward_wrapper(lambda *a, **k: None, ctx, DIT, video=1.0, text=2.0, audio=3.0)
    wrapper((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT, minimax_payload=PAYLOAD)
    rows, timesteps = ctx.mask, ctx.timestep_mask
    wrapper((VIDEO_X, AUDIO_X), torch.tensor([300.0]), CONTEXT, minimax_payload=PAYLOAD)
    assert ctx.mask is rows
    assert ctx.timestep_mask is not timesteps


def test_forward_wrapper_skips_masks_without_targets():
    module = load_module()
    timestep_only = module.MaskContext()
    wrapper = module._make_forward_wrapper(lambda *a, **k: None, timestep_only, DIT,
                                           video=1.0, text=2.0, audio=3.0,
                                           need_rows=False, need_timestep=True)
    wrapper((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT, minimax_payload=PAYLOAD)
    assert timestep_only.mask is None and timestep_only.timestep_mask is not None

    rows_only = module.MaskContext()
    wrapper = module._make_forward_wrapper(lambda *a, **k: None, rows_only, DIT,
                                           video=1.0, text=2.0, audio=3.0,
                                           need_rows=True, need_timestep=False)
    wrapper((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT, minimax_payload=PAYLOAD)
    assert rows_only.mask is not None and rows_only.timestep_mask is None


def test_apply_stack_e2e_equal_strengths_fold_rows_without_mask():
    module = load_module()
    model = make_model()
    lora = lora_sd("diffusion_model.blocks.0.attn.qkv_proj", E2E_DOWN, E2E_UP, alpha=4.0)
    patched, = apply_stack(module, model, [("a.safetensors", 1.0)], {"a": lora},
                           video=1.5, text=1.5, audio=1.5)
    assert "diffusion_model.forward" not in patched.object_patches
    apply_object_patches(patched)
    dit = patched.model.diffusion_model
    x = torch.ones(LAYOUT.seq_len, 8)
    qkv = dit.blocks[0].attn.qkv_proj
    expected = x @ qkv.weight.T + (x @ (E2E_DOWN * 2.0).T) @ E2E_UP.T * 1.5
    assert torch.allclose(qkv(x), expected)


def test_apply_stack_e2e_equal_audio_video_folds_adaln_without_forward_wrapper():
    module = load_module()
    model = make_model()
    lora = lora_sd("diffusion_model.blocks.0.adaln_proj.linear", E2E_DOWN2, E2E_UP2)
    patched, = apply_stack(module, model, [("a.safetensors", 1.0)], {"a": lora},
                           video=2.0, text=1.0, audio=2.0)
    assert "diffusion_model.forward" not in patched.object_patches
    apply_object_patches(patched)
    adaln = patched.model.diffusion_model.blocks[0].adaln_proj.linear
    x = torch.ones(2, 8)
    expected = x @ adaln.weight.T + (x @ E2E_DOWN2.T) @ E2E_UP2.T * 2.0
    assert torch.allclose(adaln(x), expected)


def test_apply_stack_e2e_equal_audio_video_keeps_row_mask_only():
    module = load_module()
    model = make_model()
    lora = {**lora_sd("diffusion_model.blocks.0.attn.qkv_proj", E2E_DOWN, E2E_UP, alpha=4.0),
            **lora_sd("diffusion_model.blocks.0.adaln_proj.linear", E2E_DOWN2, E2E_UP2)}
    patched, = apply_stack(module, model, [("a.safetensors", 1.0)], {"a": lora},
                           video=2.0, text=1.0, audio=2.0)
    assert "diffusion_model.forward" in patched.object_patches
    apply_object_patches(patched)
    dit = patched.model.diffusion_model
    dit.forward((VIDEO_X, AUDIO_X), TIMESTEP, CONTEXT,
                transformer_options={}, minimax_payload=PAYLOAD)
    x = torch.ones(LAYOUT.seq_len, 8)
    qkv = dit.blocks[0].attn.qkv_proj
    row_mask = module._build_rows_mask(LAYOUT, video=2.0, text=1.0, audio=2.0)
    expected = x @ qkv.weight.T + (x @ (E2E_DOWN * 2.0).T) @ E2E_UP.T * row_mask[:, None]
    assert torch.allclose(qkv(x), expected)
    t = torch.ones(2, 8)
    adaln = dit.blocks[0].adaln_proj.linear
    assert torch.allclose(adaln(t), t @ adaln.weight.T + (t @ E2E_DOWN2.T) @ E2E_UP2.T * 2.0)

