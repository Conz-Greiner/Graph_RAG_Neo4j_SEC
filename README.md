## 🚀 Quick Start — Infrastructure

### Prerequisites
- Docker Engine **24+** with **Compose v2**
- NVIDIA driver **535+** and **NVIDIA Container Toolkit** (for vLLM)
- At least **48 GB RAM**, **1× GPU with ≥24 GB VRAM** (A10G/L4/A100 recommended)
- HuggingFace account + token for gated models (optional)

### 1. Clone & configure
```bash
git clone https://github.com/your-org/agentic-graph-rag.git
cd agentic-graph-rag
cp .env.example .env
# Edit .env — at minimum set NEO4J_PASSWORD and HF_TOKEN
```

### 2. Validate settings
```bash
docker compose run --rm app python -c "from config.settings import settings; print(settings.model_dump_json(indent=2))"
```

### 3. Bring up the stack
```bash
docker compose up -d --build
docker compose ps
```

### 4. Wait for healthy state
```bash
docker compose logs -f vllm        # first boot downloads ~30 GB of weights
docker compose logs -f neo4j       # APOC plugin installation
```

### 5. Verify endpoints
| Service     | URL                                   |
|-------------|---------------------------------------|
| FastAPI     | http://localhost:8080/docs            |
| Streamlit   | http://localhost:8501                 |
| Qdrant UI   | http://localhost:6333/dashboard       |
| Neo4j       | http://localhost:7474 (neo4j / ****)  |
| vLLM        | http://localhost:8000/v1/models       |

### 6. Tear down
```bash
docker compose down          # keep data
docker compose down -v       # wipe volumes (destructive)
```
 
### Troubleshooting
- **vLLM fails to start** → ensure `nvidia-smi` works on host and `docker run --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` succeeds.
- **Neo4j health check fails** → give it up to 90 s on first boot; APOC is installed at runtime.
- **OOM during ingestion** → lower `AGR_EMBEDDING__BATCH_SIZE` and `AGR_INGESTION__BATCH_SIZE` in `.env`.