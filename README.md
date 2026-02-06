# ComfyUI Model Manager

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that load models from a Model Manager REST API. Models are downloaded on demand and cached locally.

## Nodes

- **Load Checkpoint (Model Manager)** - Load checkpoint models
- **Load LoRA (Model Manager)** - Load a single LoRA with strength controls
- **Load LoRA Multi (Model Manager)** - Load multiple LoRAs with per-slot toggles and strength controls
- **Load VAE (Model Manager)** - Load VAE models
- **Clear Cache (Model Manager)** - View cache size and clear downloaded models

## Features

- Connect to any Model Manager API with URL + API key
- Automatic local caching (models are only downloaded once)
- In-node connect/disconnect for API authentication
- **LoRA multi-loader**: add/remove slots dynamically, per-slot toggle switches, inline strength adjustment with arrow controls, baseModel-grouped dropdown, right-click context menu for move/remove, Toggle All switch

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/evandcoleman/comfyui-model-manager.git
cd comfyui-model-manager
pip install -r requirements.txt
```

## Configuration

You can configure the connection in three ways:

### 1. In-node connect (recommended)

Click the **Connect to Model Manager** button on any loader node, enter your API URL and API key. Connection info is saved to `config.yaml` for future sessions.

### 2. Config file

Copy `config.yaml.example` to `config.yaml` and fill in your values:

```yaml
api_url: "http://localhost:3000"
api_key: "your-api-key"
```

### 3. Environment variables

```bash
export MODEL_MANAGER_API_URL="http://localhost:3000"
export MODEL_MANAGER_API_KEY="your-api-key"
```

Environment variables take priority over `config.yaml` values.

## License

MIT
