import io
import os
import json
import logging
import numpy as np
from PIL import Image
from .client import get_client, ModelManagerAuthError

logger = logging.getLogger("comfyui-model-manager")

# ---------------------------------------------------------------------------
# Flexible input type helpers (for dynamic LoRA inputs)
# ---------------------------------------------------------------------------

class AnyType(str):
    """Matches any ComfyUI type for flexible inputs."""
    def __ne__(self, __value):
        return False

any_type = AnyType("*")

class FlexibleOptionalInputType(dict):
    """Dict that accepts any key, returning (any_type,) for unknowns."""
    def __contains__(self, key):
        return True
    def __getitem__(self, key):
        return (any_type,)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_model_list(folder):
    """Fetch model list from Model Manager, formatted as 'modelId@versionId:name' strings."""
    try:
        client = get_client()
        if not client.authenticated:
            return ["(not connected)"]
        models = client.list_models(folder)
        if not models:
            return ["(no models found)"]
        return [_model_combo_value(m) for m in models]
    except ModelManagerAuthError:
        return ["(not connected)"]
    except Exception as e:
        logger.warning(f"Failed to list models in {folder}: {e}")
        return ["(error loading models)"]


def _model_combo_value(m):
    """Build a combo string from a model dict: 'modelId@versionId:name'."""
    mid = m["id"]
    vid = m.get("versionId")
    name = m["name"]
    return f"{mid}@{vid}:{name}" if vid else f"{mid}:{name}"


def _parse_model_value(value):
    """Extract (model_id, version_id) from a 'modelId@versionId:name' string.

    Returns (int, int|None) or (None, None) on failure.
    """
    if not value or value.startswith("("):
        return None, None
    id_part = value.split(":", 1)[0]
    if "@" in id_part:
        mid_str, vid_str = id_part.split("@", 1)
        try:
            mid = int(mid_str)
            vid = int(vid_str) if vid_str and vid_str != "None" else None
            return mid, vid
        except ValueError:
            return None, None
    try:
        return int(id_part), None
    except ValueError:
        return None, None

# ---------------------------------------------------------------------------
# Checkpoint Loader
# ---------------------------------------------------------------------------

class ModelManagerCheckpointLoader:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load"
    CATEGORY = "loaders/model-manager"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_get_model_list("checkpoints"),),
            }
        }

    @classmethod
    def IS_CHANGED(cls, ckpt_name):
        return get_client().version

    def load(self, ckpt_name):
        import comfy.sd
        import comfy.utils

        model_id, version_id = _parse_model_value(ckpt_name)
        if model_id is None:
            raise ValueError(f"Invalid model selection: {ckpt_name}")

        client = get_client()
        pbar = comfy.utils.ProgressBar(100)
        def on_progress(downloaded, total):
            pbar.update_absolute(int(downloaded * 100 / total), 100)
        local_path = client.download_model(model_id, "checkpoints", version_id=version_id, progress_callback=on_progress)
        return comfy.sd.load_checkpoint_guess_config(local_path)[:3]

# ---------------------------------------------------------------------------
# LoRA Loader (single)
# ---------------------------------------------------------------------------

class ModelManagerLoRALoader:
    RETURN_TYPES = ("MODEL", "CLIP", "MM_LORA_INFO")
    RETURN_NAMES = ("model", "clip", "lora_info")
    FUNCTION = "load"
    CATEGORY = "loaders/model-manager"

    def __init__(self):
        self.loaded_lora = None
        self.loaded_lora_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (_get_model_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.05}),
            },
            "optional": {
                "clip": ("CLIP",),
                "lora_info": ("MM_LORA_INFO",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return get_client().version

    def load(self, model, lora_name, strength_model, strength_clip, clip=None, lora_info=None):
        import comfy.utils
        import comfy.sd

        model_id, version_id = _parse_model_value(lora_name)
        if model_id is None:
            raise ValueError(f"Invalid LoRA selection: {lora_name}")

        if self.loaded_lora_name != lora_name:
            client = get_client()
            pbar = comfy.utils.ProgressBar(100)
            def on_progress(downloaded, total):
                pbar.update_absolute(int(downloaded * 100 / total), 100)
            local_path = client.download_model(model_id, "loras", version_id=version_id, progress_callback=on_progress)
            self.loaded_lora = comfy.utils.load_torch_file(local_path, safe_load=True)
            self.loaded_lora_name = lora_name

        model_out, clip_out = comfy.sd.load_lora_for_models(model, clip, self.loaded_lora, strength_model, strength_clip)

        info = list(lora_info or [])
        display_name = lora_name.split(":", 1)[1] if ":" in lora_name else lora_name
        info.append({
            "modelId": model_id,
            "versionId": version_id,
            "name": display_name,
            "strength": strength_model,
        })

        return (model_out, clip_out, info)

# ---------------------------------------------------------------------------
# Multi-LoRA Loader (Power LoRA style)
# ---------------------------------------------------------------------------

class ModelManagerMultiLoRALoader:
    RETURN_TYPES = ("MODEL", "CLIP", "MM_LORA_INFO")
    RETURN_NAMES = ("model", "clip", "lora_info")
    FUNCTION = "load"
    CATEGORY = "loaders/model-manager"

    def __init__(self):
        self.loaded_loras = {}  # "id:name" -> loaded lora data

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": FlexibleOptionalInputType({
                "clip": ("CLIP",),
            }),
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return get_client().version

    def load(self, model, clip=None, **kwargs):
        import comfy.utils
        import comfy.sd

        lora_info = []

        for key in sorted(kwargs.keys()):
            if not key.startswith("lora_"):
                continue

            value = kwargs[key]

            # Accept structured {on, lora, strength, strengthTwo} dicts from the frontend
            if isinstance(value, dict):
                on = value.get("on", True)
                lora_name = value.get("lora", "None")
                strength_model = value.get("strength", 1.0)
                strength_clip = value.get("strengthTwo") if value.get("strengthTwo") is not None else strength_model
            else:
                on = True
                lora_name = str(value)
                strength_model = 1.0
                strength_clip = 1.0

            if not on or lora_name == "None":
                continue
            if clip is None:
                strength_clip = 0

            model_id, version_id = _parse_model_value(lora_name)
            if model_id is None:
                logger.warning(f"Skipping invalid LoRA value: {lora_name}")
                continue

            if lora_name not in self.loaded_loras:
                client = get_client()
                pbar = comfy.utils.ProgressBar(100)
                def on_progress(downloaded, total):
                    pbar.update_absolute(int(downloaded * 100 / total), 100)
                local_path = client.download_model(model_id, "loras", version_id=version_id, progress_callback=on_progress)
                self.loaded_loras[lora_name] = comfy.utils.load_torch_file(local_path, safe_load=True)

            if strength_model != 0 or strength_clip != 0:
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, self.loaded_loras[lora_name], strength_model, strength_clip,
                )
                display_name = lora_name.split(":", 1)[1] if ":" in lora_name else lora_name
                lora_info.append({
                    "modelId": model_id,
                    "versionId": version_id,
                    "name": display_name,
                    "strength": strength_model,
                })

        return (model, clip, lora_info)

# ---------------------------------------------------------------------------
# VAE Loader
# ---------------------------------------------------------------------------

class ModelManagerVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "loaders/model-manager"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (_get_model_list("vae"),),
            }
        }

    @classmethod
    def IS_CHANGED(cls, vae_name):
        return get_client().version

    def load(self, vae_name):
        import comfy.utils
        import comfy.sd

        model_id, version_id = _parse_model_value(vae_name)
        if model_id is None:
            raise ValueError(f"Invalid VAE selection: {vae_name}")

        client = get_client()
        pbar = comfy.utils.ProgressBar(100)
        def on_progress(downloaded, total):
            pbar.update_absolute(int(downloaded * 100 / total), 100)
        local_path = client.download_model(model_id, "vae", version_id=version_id, progress_callback=on_progress)
        sd = comfy.utils.load_torch_file(local_path)
        return (comfy.sd.VAE(sd=sd),)

# ---------------------------------------------------------------------------
# Clear Cache
# ---------------------------------------------------------------------------

class ModelManagerClearCache:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear"
    CATEGORY = "loaders/model-manager"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confirm": ("BOOLEAN", {"default": False}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def clear(self, confirm):
        if not confirm:
            client = get_client()
            cache_dir = client.cache_dir
            total = 0
            if os.path.isdir(cache_dir):
                for dirpath, _dirnames, filenames in os.walk(cache_dir):
                    for f in filenames:
                        total += os.path.getsize(os.path.join(dirpath, f))
            size_mb = total / (1024 * 1024)
            return (f"Cache: {size_mb:.1f} MB — set confirm to true to clear",)

        client = get_client()
        freed = client.clear_cache()
        freed_mb = freed / (1024 * 1024)
        return (f"Cleared {freed_mb:.1f} MB",)

# ---------------------------------------------------------------------------
# Image Upload
# ---------------------------------------------------------------------------

class ModelManagerImageUpload:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "upload"
    CATEGORY = "loaders/model-manager"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model_name": (_get_model_list("checkpoints"),),
            },
            "optional": {
                "lora_info": ("MM_LORA_INFO",),
                "prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000, "forceInput": True}),
                "cfg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "forceInput": True}),
                "sampler": ("STRING", {"default": "", "forceInput": True}),
                "scheduler": ("STRING", {"default": "", "forceInput": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def upload(self, images, model_name, lora_info=None, prompt="",
               negative_prompt="", seed=0, steps=0, cfg_scale=0.0,
               sampler="", scheduler="", extra_pnginfo=None):
        model_id, _ = _parse_model_value(model_name)
        if model_id is None:
            return ("Error: invalid model selection",)

        client = get_client()
        if not client.authenticated:
            return ("Error: not connected to Model Manager",)

        # Build workflow JSON from extra_pnginfo
        workflow_json = None
        if extra_pnginfo and isinstance(extra_pnginfo, dict):
            workflow_json = extra_pnginfo.get("workflow")

        results = []
        for i in range(images.shape[0]):
            # Convert IMAGE tensor (B,H,W,C float32 0-1) to PNG bytes
            img_array = (images[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            metadata = {}
            if prompt:
                metadata["prompt"] = prompt
            if negative_prompt:
                metadata["negativePrompt"] = negative_prompt
            if seed:
                metadata["seed"] = seed
            if steps:
                metadata["steps"] = steps
            if cfg_scale:
                metadata["cfgScale"] = cfg_scale
            if sampler:
                metadata["sampler"] = sampler
            if scheduler:
                metadata["scheduler"] = scheduler
            if lora_info:
                metadata["loras"] = lora_info
            if workflow_json:
                metadata["comfyWorkflow"] = workflow_json

            try:
                result = client.upload_image(
                    model_id, image_bytes, f"comfyui_{i:05d}.png", metadata,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to upload image {i} to model {model_id}: {e}")
                return (f"Error uploading image {i}: {e}",)

        count = len(results)
        return (f"Uploaded {count} image{'s' if count != 1 else ''} to model {model_id}",)
