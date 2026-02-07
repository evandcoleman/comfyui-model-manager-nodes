import asyncio
import logging
from aiohttp import web

from .nodes import (
    ModelManagerDiffusionModelLoader,
    ModelManagerCheckpointLoader,
    ModelManagerLoRALoader,
    ModelManagerMultiLoRALoader,
    ModelManagerVAELoader,
    ModelManagerClearCache,
    ModelManagerMergeLoRAInfo,
    ModelManagerImageUpload,
)
from .client import get_client, ModelManagerAuthError, ModelManagerError

logger = logging.getLogger("comfyui-model-manager")

# ---------------------------------------------------------------------------
# Node Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "ModelManagerDiffusionModelLoader": ModelManagerDiffusionModelLoader,
    "ModelManagerCheckpointLoader": ModelManagerCheckpointLoader,
    "ModelManagerLoRALoader": ModelManagerLoRALoader,
    "ModelManagerMultiLoRALoader": ModelManagerMultiLoRALoader,
    "ModelManagerVAELoader": ModelManagerVAELoader,
    "ModelManagerClearCache": ModelManagerClearCache,
    "ModelManagerMergeLoRAInfo": ModelManagerMergeLoRAInfo,
    "ModelManagerImageUpload": ModelManagerImageUpload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelManagerDiffusionModelLoader": "Load Diffusion Model (Model Manager)",
    "ModelManagerCheckpointLoader": "Load Checkpoint (Model Manager)",
    "ModelManagerLoRALoader": "Load LoRA (Model Manager)",
    "ModelManagerMultiLoRALoader": "Load LoRA Multi (Model Manager)",
    "ModelManagerVAELoader": "Load VAE (Model Manager)",
    "ModelManagerClearCache": "Clear Cache (Model Manager)",
    "ModelManagerMergeLoRAInfo": "Merge LoRA Info (Model Manager)",
    "ModelManagerImageUpload": "Upload Image (Model Manager)",
}

WEB_DIRECTORY = "./js"

# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

ALLOWED_FOLDERS = {"diffusion_models", "checkpoints", "loras", "vae"}

try:
    from server import PromptServer

    @PromptServer.instance.routes.get("/model-manager/status")
    async def model_manager_status(request):
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(None, get_client)
        return web.json_response({
            "connected": client.authenticated,
            "api_url": client.api_url,
        })

    @PromptServer.instance.routes.post("/model-manager/connect")
    async def model_manager_connect(request):
        data = await request.json()
        api_url = data.get("api_url", "").strip()
        api_key = data.get("api_key", "").strip()

        if not api_url or not api_key:
            return web.json_response(
                {"error": "api_url and api_key are required"},
                status=400,
            )

        loop = asyncio.get_event_loop()
        try:
            client = await loop.run_in_executor(None, get_client)
            await loop.run_in_executor(None, lambda: client.connect(api_url, api_key, persist=True))
            return web.json_response({
                "connected": True,
                "api_url": client.api_url,
            })
        except ModelManagerAuthError as e:
            return web.json_response({"error": str(e)}, status=401)
        except Exception as e:
            logger.error(f"Connect error: {e}")
            return web.json_response({"error": f"Connection failed: {e}"}, status=502)

    @PromptServer.instance.routes.post("/model-manager/disconnect")
    async def model_manager_disconnect(request):
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(None, get_client)
        await loop.run_in_executor(None, client.disconnect)
        return web.json_response({"connected": False})

    @PromptServer.instance.routes.post("/model-manager/refresh-models")
    async def model_manager_refresh_models(request):
        loop = asyncio.get_event_loop()
        client = await loop.run_in_executor(None, get_client)
        await loop.run_in_executor(None, client.refresh_models)
        return web.json_response({"ok": True})

    @PromptServer.instance.routes.get("/model-manager/models/{folder}")
    async def model_manager_models(request):
        folder = request.match_info["folder"]
        if folder not in ALLOWED_FOLDERS:
            return web.json_response(
                {"error": f"Invalid folder. Allowed: {', '.join(sorted(ALLOWED_FOLDERS))}"},
                status=400,
            )

        loop = asyncio.get_event_loop()
        try:
            client = await loop.run_in_executor(None, get_client)
            models = await loop.run_in_executor(None, client.list_models, folder)
            return web.json_response({"models": models})
        except ModelManagerAuthError as e:
            return web.json_response({"error": str(e)}, status=401)
        except ModelManagerError as e:
            return web.json_response({"error": str(e)}, status=500)
        except Exception as e:
            logger.error(f"Models list error for {folder}: {e}")
            return web.json_response({"error": str(e)}, status=500)

except ImportError:
    logger.warning("PromptServer not available — API routes not registered")
