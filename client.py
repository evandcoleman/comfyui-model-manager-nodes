import os
import re
import threading
import logging
import yaml
import requests

logger = logging.getLogger("comfyui-model-manager")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ModelManagerError(Exception):
    """Base exception for Model Manager operations."""

class ModelManagerAuthError(ModelManagerError):
    """Authentication failed (bad API key or missing credentials)."""

class ModelManagerAPIError(ModelManagerError):
    """Model Manager API returned an error response."""
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _config_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

def load_config():
    """Load config from config.yaml, then let env vars override."""
    config = {
        "api_url": "",
        "api_key": "",
        "cache_dir": "",
    }
    path = _config_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            for key in config:
                if key in file_config and file_config[key]:
                    config[key] = file_config[key]
        except Exception as e:
            logger.warning(f"Failed to read config.yaml: {e}")

    env_map = {
        "MODEL_MANAGER_API_URL": "api_url",
        "MODEL_MANAGER_API_KEY": "api_key",
    }
    for env_key, config_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            config[config_key] = val

    return config


def save_config(config):
    """Write config dict back to config.yaml."""
    path = _config_path()
    try:
        with open(path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Saved config to {path}")
    except Exception as e:
        logger.warning(f"Failed to save config.yaml: {e}")

# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

FOLDER_TO_CATEGORY = {
    "diffusion_models": "Diffusion Model",
    "checkpoints": "Checkpoint",
    "loras": "LoRA",
    "vae": "VAE",
}

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

def _default_cache_dir():
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, "model_manager_cache")
    except ImportError:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# ---------------------------------------------------------------------------
# Version expansion
# ---------------------------------------------------------------------------

def _expand_versions(raw_models):
    """Expand a list of models into one entry per local version.

    For models with one local version the display name is just the model name.
    For models with multiple local versions: "ModelName - VersionName".
    Models without a versions array fall back to a single entry.
    """
    expanded = []
    for model in raw_models:
        versions = model.get("versions")
        if not versions:
            # No version data — use model as-is (single entry, no versionId)
            expanded.append({
                "id": model["id"],
                "versionId": None,
                "name": model["name"],
                "modelName": model["name"],
                "versionName": None,
                "baseModel": model.get("baseModel"),
            })
            continue

        local_versions = [v for v in versions if v.get("isLocal")]
        if not local_versions:
            continue

        multi = len(local_versions) > 1
        for v in local_versions:
            display_name = (
                f"{model['name']} - {v['name']}" if multi
                else model["name"]
            )
            expanded.append({
                "id": model["id"],
                "versionId": v["id"],
                "name": display_name,
                "modelName": model["name"],
                "versionName": v["name"],
                "baseModel": v.get("baseModel") or model.get("baseModel"),
            })

    return expanded


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ModelManagerClient:
    def __init__(self, cache_dir=None):
        self._lock = threading.Lock()
        self._api_url = None
        self._api_key = None
        self._cache_dir = cache_dir or _default_cache_dir()
        self._validated = False
        self._model_cache = {}  # folder -> [model dicts]
        self._version = 0

    # -- public properties --------------------------------------------------

    @property
    def version(self):
        return self._version

    @property
    def authenticated(self):
        return bool(self._api_url and self._api_key and self._validated)

    @property
    def api_url(self):
        return self._api_url

    @property
    def cache_dir(self):
        return self._cache_dir

    # -- config persistence -------------------------------------------------

    def _persist_config(self):
        """Save current connection info to config.yaml."""
        config = load_config()
        config["api_url"] = self._api_url or ""
        config["api_key"] = self._api_key or ""
        save_config(config)

    # -- connection ---------------------------------------------------------

    def connect(self, api_url, api_key, persist=False):
        """Validate credentials with a test request, then store them."""
        with self._lock:
            api_url = api_url.rstrip("/")
            if not api_url:
                raise ModelManagerAuthError("API URL is required")
            if not api_key:
                raise ModelManagerAuthError("API key is required")

            # Test the connection
            try:
                resp = requests.get(
                    f"{api_url}/api/v1/models",
                    params={"limit": 1},
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15,
                )
            except requests.ConnectionError as e:
                raise ModelManagerAPIError(f"Could not connect to {api_url}: {e}")
            except requests.Timeout:
                raise ModelManagerAPIError(f"Connection to {api_url} timed out")

            if resp.status_code == 401:
                raise ModelManagerAuthError("Invalid API key")
            if resp.status_code == 403:
                raise ModelManagerAuthError("Access denied")
            resp.raise_for_status()

            self._api_url = api_url
            self._api_key = api_key
            self._validated = True
            self._model_cache.clear()
            self._version += 1
            logger.info(f"Connected to Model Manager at {api_url}")

            if persist:
                self._persist_config()

    def disconnect(self):
        """Clear connection info."""
        with self._lock:
            self._api_url = None
            self._api_key = None
            self._validated = False
            self._model_cache.clear()
            self._version += 1
            self._persist_config()
            logger.info("Disconnected from Model Manager")

    # -- HTTP ---------------------------------------------------------------

    def _request(self, method, path, params=None, stream=False, timeout=30):
        """Central HTTP method with auth header and error handling."""
        if not self._api_url or not self._api_key:
            raise ModelManagerAuthError("Not connected — please connect first")

        url = f"{self._api_url}{path}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        resp = requests.request(
            method, url,
            params=params,
            headers=headers,
            stream=stream,
            timeout=timeout,
        )

        if resp.status_code == 401:
            self._validated = False
            raise ModelManagerAuthError("Invalid or expired API key")
        if resp.status_code == 403:
            raise ModelManagerAuthError("Access denied")
        if not resp.ok and not stream:
            try:
                data = resp.json()
                msg = data.get("error", resp.text)
            except Exception:
                msg = resp.text
            raise ModelManagerAPIError(f"API error ({resp.status_code}): {msg}", code=resp.status_code)

        return resp

    # -- model operations ---------------------------------------------------

    def list_models(self, folder):
        """List models for a folder category, expanded by version.

        Returns a list of dicts, one per local version:
            {id, versionId, name, modelName, versionName, baseModel, ...}
        """
        if folder in self._model_cache:
            return self._model_cache[folder]

        category = FOLDER_TO_CATEGORY.get(folder)
        if not category:
            raise ModelManagerError(f"Unknown folder: {folder}")

        raw_models = []
        page = 1
        while True:
            resp = self._request("GET", "/api/v1/models", params={
                "category": category,
                "include": "versions",
                "limit": 100,
                "page": page,
            })
            data = resp.json()
            raw_models.extend(data.get("items", []))

            if not data.get("hasMore", False):
                break
            page += 1

        expanded = _expand_versions(raw_models)
        self._model_cache[folder] = expanded
        return expanded

    def get_model(self, model_id):
        """Fetch full details for a single model."""
        resp = self._request("GET", f"/api/v1/models/{model_id}")
        return resp.json()

    def download_model(self, model_id, folder, version_id=None, progress_callback=None):
        """Download a model file. Returns the local cache path.

        Uses "{model_id}_{version_id}_" as the cache prefix so each version
        is cached independently.
        """
        cache_folder = os.path.join(self._cache_dir, folder)
        os.makedirs(cache_folder, exist_ok=True)

        # Build cache prefix including version when available
        prefix = f"{model_id}_{version_id}_" if version_id else f"{model_id}_"
        for existing in os.listdir(cache_folder):
            if existing.startswith(prefix):
                local_path = os.path.join(cache_folder, existing)
                logger.info(f"Cache hit: {local_path}")
                return local_path

        # Stream download — pass versionId when available
        params = {"versionId": version_id} if version_id else None
        resp = self._request(
            "GET", f"/api/v1/models/{model_id}/download",
            params=params, stream=True, timeout=600,
        )
        resp.raise_for_status()

        # Extract filename from content-disposition header
        cd = resp.headers.get("content-disposition", "")
        filename = None
        if "filename=" in cd:
            # Parse filename="value" or filename=value
            match = re.search(r'filename="?([^";\n]+)"?', cd)
            if match:
                filename = match.group(1).strip()
        if not filename:
            filename = f"model_{model_id}.safetensors"
        safe_filename = re.sub(r'[^\w\-.]', '_', filename)
        cache_name = (
            f"{model_id}_{version_id}_{safe_filename}" if version_id
            else f"{model_id}_{safe_filename}"
        )
        local_path = os.path.join(cache_folder, cache_name)

        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0

        tmp_path = local_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress_callback(downloaded, total_size)
            os.rename(tmp_path, local_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        logger.info(f"Downloaded model {model_id} -> {local_path}")
        return local_path

    def upload_image(self, model_id, image_data, filename, metadata=None, version_id=None):
        """Upload an image with generation metadata to a model.

        Args:
            model_id: integer model ID
            image_data: bytes of the image file (PNG/JPEG)
            filename: display filename (e.g. "comfyui_00001.png")
            metadata: dict with optional keys — prompt, negativePrompt, seed,
                      steps, cfgScale, sampler, scheduler, nsfwLevel, loras,
                      comfyWorkflow
            version_id: optional version ID to associate the image with
        Returns:
            The created image dict from the API.
        """
        import json as _json

        if not self._api_url or not self._api_key:
            raise ModelManagerAuthError("Not connected — please connect first")

        url = f"{self._api_url}/api/v1/models/{model_id}/images"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        files = {"file": (filename, image_data, "image/png")}
        data = {}
        if version_id:
            data["versionId"] = str(version_id)
        if metadata:
            for key, value in metadata.items():
                if value is None:
                    continue
                if isinstance(value, (dict, list)):
                    data[key] = _json.dumps(value)
                elif isinstance(value, bool):
                    data[key] = str(value).lower()
                else:
                    data[key] = str(value)

        resp = requests.post(url, headers=headers, files=files, data=data, timeout=120)

        if resp.status_code == 401:
            self._validated = False
            raise ModelManagerAuthError("Invalid or expired API key")
        if not resp.ok:
            try:
                msg = resp.json().get("error", resp.text)
            except Exception:
                msg = resp.text
            raise ModelManagerAPIError(f"Upload failed ({resp.status_code}): {msg}", code=resp.status_code)

        logger.info(f"Uploaded image to model {model_id}")
        return resp.json()

    def refresh_models(self):
        """Clear the in-memory model list cache so the next list_models call re-fetches."""
        self._model_cache.clear()
        self._version += 1
        logger.info("Refreshed model list cache")

    def clear_cache(self):
        """Delete all cached model files and return the number of bytes freed."""
        import shutil
        freed = 0
        if os.path.isdir(self._cache_dir):
            for dirpath, _dirnames, filenames in os.walk(self._cache_dir):
                for f in filenames:
                    freed += os.path.getsize(os.path.join(dirpath, f))
            shutil.rmtree(self._cache_dir)
            os.makedirs(self._cache_dir, exist_ok=True)
            logger.info(f"Cleared cache: freed {freed / (1024 * 1024):.1f} MB")
        return freed

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client = None
_client_lock = threading.Lock()

def get_client():
    """Lazy singleton factory for the Model Manager client."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                config = load_config()
                cache_dir = config.get("cache_dir") or _default_cache_dir()
                _client = ModelManagerClient(cache_dir=cache_dir)

                # Auto-connect if credentials are available from config/env
                api_url = config.get("api_url")
                api_key = config.get("api_key")
                if api_url and api_key:
                    try:
                        _client.connect(api_url, api_key)
                        logger.info("Auto-connect from config/environment succeeded")
                    except Exception as e:
                        logger.warning(f"Auto-connect failed: {e}")
    return _client
