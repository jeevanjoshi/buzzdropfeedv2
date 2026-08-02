# Autonomous YouTube Storytelling Video Generation (CSVG) Pipeline

Production-grade, zero-human-intervention **Customized Storytelling Video Generation (CSVG)** pipeline designed for high-retention 10–15 minute 16:9 widescreen Infotainment YouTube channels targeting **$2,000+ USD / Month** ad revenue.

---

## 🏗️ Architecture & Node Execution Roles (OCI vs. Raspberry Pi 5)

The pipeline uses a distributed architecture splitting heavy cloud media rendering/agent orchestration from local edge audio synthesis and speech alignment:

```
               OCI CLOUD NODE (/opt/csvg_pipeline)
   [2 OCPUs, 12 GB RAM, Ubuntu 22.04 LTS - Cron: 0 4 * * *]
   ├── main.py                          <-- Pipeline Orchestrator & CLI Entrypoint
   ├── src/                             <-- Core Agent Framework & Engines
   │   ├── agents/                      (Orchestrator, FactRetriever, StoryDesigner, Observer, MediaProducer, Publisher)
   │   ├── engine/                      (TOPSIS Math, EMA Velocity, Cosine RPM, RSS Ingestion, Logger, Tracer, APINinjas, ExternalAPIs, SpaceCinema, GIFRetriever)
   │   └── schemas/                     (GlobalState, A2AMessage, ScriptData, AssetPaths)
   ├── mcp_servers/media_cloud/         <-- Cloud Media MCP Server (Port 8001)
   │   └── server.py                    (Flux.1 16:9, Ken Burns, Dynamic Charts, Playwright SVG, GIPHY Reactions, Thumbnails, FFmpeg Timeline)
   └── mcp_servers/youtube_cloud/       <-- YouTube Publishing MCP Server (Port 8002)
       └── server.py                    (Headless OAuth2 Upload & Quota Guard)

                               │
                       HTTP / SSE REST Bridge
                    (http://172.198.1.30:8000/tools/...)
                               │
                               v

          RASPBERRY PI 5 EDGE NODE (/opt/csvg_edge)
         [4 GB RAM, Raspberry Pi OS - Systemd Service]
   ├── mcp_servers/audio_edge/          <-- Edge Audio MCP Server (Port 8000)
   │   └── server.py                    (Kokoro-ONNX TTS Synthesis & Faster-Whisper .ass Subtitles)
   ├── kokoro-v0.19.onnx                <-- Local Speech Weights (82M)
   └── voices.bin                       <-- Multi-Voice Intonation Blend Weights
```

### Hardware Rules & Allocation
- **OCI Cloud Node (Primary Controller & Renderer)**:
  - **Role**: Runs Orchestrator, multi-agent logic, RAG fact grounding, image rendering (Flux.1 / Playwright dynamic charts), video timeline assembly via FFmpeg, and YouTube OAuth publishing.
  - **Local LLM**: Ollama running `qwen2.5:7b-instruct` (or `mistral-7b`) using 12 GB RAM.
- **Raspberry Pi 5 Edge Node (Audio & Subtitle Processor)**:
  - **Role**: Offloads speech synthesis and subtitle alignment over HTTP/REST bridge to keep cloud OCPUs dedicated to media assembly.
  - **Local LLM & Models**: Ollama running `llama3.2:3b` + ONNX runtime executing `Kokoro-ONNX` (82M parameters) and `Faster-Whisper` for frame-accurate `.ass` subtitle alignment.

---

## 🔄 Detailed Pipeline Phase Breakdown

### Phase 1: Topic Discovery & TOPSIS Selection
- **How It Works**: Ingests breaking news from RSS feeds (Google News, BBC, Economic Times, Moneycontrol) and external APIs. Computes velocity metrics and selects optimal viral topics using a 7-criteria TOPSIS decision matrix.
- **Tooling Used**: `feedparser`, `requests`, `numpy`, `scipy`.
- **LLM Used & Interaction**: Zero LLM requirement for math; optional Local `qwen2.5:7b-instruct` / Cloud Gemini to summarize multi-source headline clusters.
- **Node Location**: OCI Cloud Node.
- **Cost**: **$0.00 (Free)** via RSS & free public APIs.

### Phase 2: RAG Fact Retrieval & Verification
- **How It Works**: `FactRetrieverAgent` queries multi-domain factual APIs (World Bank, NASA Open APIs, Alpha Vantage, Marketaux, API-Ninjas) to fetch verified statistics, box office numbers, stock charts, and historical milestones.
- **Tooling Used**: `requests`, Python async IO.
- **LLM Used & Interaction**: Local LLM or Gemini extracts structured JSON facts (`FactItem` schema) and calculates fact-grounding density scores.
- **Node Location**: OCI Cloud Node.
- **Cost**: **$0.00 (Free Tier APIs)** or optional paid API keys for high-volume market feeds.

### Phase 3: Storyboard & Script Generation
- **How It Works**: `StoryDesignerAgent` transforms grounded facts into a high-retention 10-15 minute, shot-by-shot script (`ScriptData`). Each shot contains narration text, visual prompts, chart specifications, and duration estimates.
- **Tooling Used**: `LLMClient` (with schema enforcement and automatic retry).
- **LLM Used & Interaction**: `qwen2.5:7b-instruct` (Ollama local $0 cost) or `google/gemini-2.5-flash` / `openai/gpt-4o-mini`. Generates structured JSON adhering strictly to `ScriptData` Pydantic schemas.
- **Node Location**: OCI Cloud Node.
- **Cost**: **$0.00** (Local Ollama) or **~$0.01 - $0.03 / script** (Cloud OpenRouter/Gemini API).

### Phase 4: Edge Audio & Subtitle Generation
- **How It Works**: `MediaProducerAgent` dispatches narration text per shot over HTTP to the Raspberry Pi 5 Edge Node.
  - Pi 5 generates high-fidelity neural voice audio clips (`.wav`).
  - Pi 5 runs Faster-Whisper to generate time-aligned subtitle files (`.ass`).
- **Tooling Used**: `kokoro-onnx` (82M model), `faster-whisper`, `ffmpeg`.
- **LLM Used & Interaction**: None (Uses dedicated neural ONNX speech models).
- **Node Location**: Raspberry Pi 5 Edge Node (`mcp_servers/audio_edge`).
- **Cost**: **$0.00 (100% Free & Unlimited Local Generation)**.

### Phase 5: Cloud Media Production & Motion Graphics
- **How It Works**: `MediaProducerAgent` invokes `mcp_servers/media_cloud`:
  - **Flux.1**: Renders cinematic 16:9 images per shot based on `visual_prompt`.
  - **Ken Burns Motion**: Applies smooth pan/zoom effects to convert static images into `.mp4` video clips.
  - **Playwright SVG & Dynamic Charts**: Generates stock/macro economic trend chart overlays using Alpha Vantage / BLS data.
  - **Thumbnail Generator**: Renders high-CTR 16:9 thumbnail artwork.
- **Tooling Used**: `playwright`, `pillow`, `ffmpeg`, `replicate`/Flux local API bridge.
- **LLM Used & Interaction**: None (Image generation & SVG rendering pipeline).
- **Node Location**: OCI Cloud Node.
- **Cost**: **$0.00** (Local SD/Flux / Playwright charts) or **~$0.01 - $0.03 / image** if using cloud GPU APIs.

### Phase 6: Timeline Assembly & Video Rendering
- **How It Works**: Assembles all shot clips (`.mp4`), audio tracks (`.wav`), background music (`bgm.mp3`), and burned `.ass` subtitles into a single 1080p widescreen master video (`final_video_1080p.mp4`).
- **Tooling Used**: `ffmpeg` CLI.
- **LLM Used & Interaction**: None.
- **Node Location**: OCI Cloud Node.
- **Cost**: **$0.00 (Free)**.

### Phase 7: Automated Publishing & Compliance Guard
- **How It Works**: `PublisherAgent` interacts with `mcp_servers/youtube_cloud` to upload the video, thumbnail, tags, and description via Google YouTube Data API v3. Enforces compliance: auto-sets `syntheticContent: true` (EU AI Act compliance) and ensures daily quota limits (< 4 uploads/day).
- **Tooling Used**: `google-api-python-client`, `google-auth-oauthlib`.
- **LLM Used & Interaction**: Local LLM or Gemini generates optimized YouTube title, description, and hashtag metadata.
- **Node Location**: OCI Cloud Node.
- **Cost**: **$0.00 (Free YouTube Data API v3)**.

---

## 💰 Cost Matrix: Free vs. Paid API Requirements

| Feature / Resource | Free Tier / Zero-Cost Option | Paid API Requirement (Optional) | API Key Needed? |
|---|---|---|---|
| **Script & Story Logic** | **Ollama** (`qwen2.5:7b-instruct` on OCI / `llama3.2:3b` on Pi 5) | OpenRouter / Gemini 2.5 Flash / OpenAI GPT-4o | Optional (`OPENROUTER_API_KEY` / `GEMINI_API_KEY`) |
| **Speech Synthesis (TTS)** | **Kokoro-ONNX** (Raspberry Pi 5 local neural voice) | ElevenLabs / Google Cloud TTS | **No** (100% Free on Pi 5) |
| **Subtitle Alignment** | **Faster-Whisper** (Raspberry Pi 5 local alignment) | OpenAI Whisper Cloud API | **No** (100% Free on Pi 5) |
| **Visual Images (16:9)** | **Local Stable Diffusion / Flux** or Playwright SVG | Replicate / Fal.ai Flux API | Optional (`REPLICATE_API_TOKEN`) |
| **Fact Grounding** | **Data.gov.in**, **World Bank**, **NASA Open APIs**, **Wikipedia** | Alpha Vantage Premium, Marketaux Pro | Optional (`ALPHAVANTAGE_API_KEY`, `MARKETAUX_API_KEY`) |
| **YouTube Publishing** | **Google Cloud YouTube Data API v3** (10,000 free daily units) | None | **Yes** (`client_secret.json`) |

---

## ⚙️ Stage-by-Stage Installation & Setup Requirements

### Stage A: Prerequisites & System Libraries (Both Nodes)
```bash
# Install core build and audio/video processing tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv ffmpeg git curl build-essential
```

### Stage B: Raspberry Pi 5 Edge Node Setup (`audio_edge`)
```bash
# 1. Install Ollama for local Edge LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# 2. Clone repository & create virtualenv
git clone https://github.com/jeevanjoshi/buzzdropfeedv2.git /opt/csvg_edge
cd /opt/csvg_edge
python3 -m venv venv
source venv/bin/activate

# 3. Install speech and neural runtime dependencies
pip install kokoro-onnx soundfile faster-whisper fastapi uvicorn requests

# 4. Download speech weights
curl -L -o kokoro-v0.19.onnx https://github.com/thewhitetail/kokoro-onnx/releases/download/v0.19/kokoro-v0.19.onnx
curl -L -o voices.bin https://github.com/thewhitetail/kokoro-onnx/releases/download/v0.19/voices.bin
```

### Stage C: OCI Cloud Node Setup (Pipeline & Cloud MCP Servers)
```bash
# 1. Install Ollama for local Cloud LLM (12 GB RAM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct

# 2. Clone repository & create virtualenv
git clone https://github.com/jeevanjoshi/buzzdropfeedv2.git /opt/csvg_pipeline
cd /opt/csvg_pipeline
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements and Playwright browser engine for SVG chart rendering
pip install -r requirements.txt
playwright install chromium --with-deps
```

### Stage D: One-Command Multi-Node Remote Deployment (`deploy.sh`)
Once both machines have SSH access configured, deploy code and start background services across both nodes automatically:

```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 🚀 Running the Autonomous Pipeline

```bash
# Run End-to-End Autonomous Pipeline (Live RSS & RAG Feeds)
python3 main.py

# Target Indian English Business & Tech News Feeds
python3 main.py --india

# Target Global Finance & Tech Feeds
python3 main.py --global

# Offline Dry-Run Mode (Uses Local Templates & Grounded Stubs)
python3 main.py --offline
```

### Automated Testing Suite
Run all 16 unit and integration test suites:
```bash
python3 run_tests.py
```

---

## 📊 TOPSIS 7-Criteria Topic Selection Math

Given candidate stories $i \in \mathcal{S}$, TOPSIS calculates relative closeness score $C_i^* \in [0, 1.0]$:

$$V_i = w_1 \text{TVS}_i + w_2 \text{RPM}_i + w_3 \text{IDI}_i + w_4 \text{SDI}_i + w_5 \text{SHM}_i + w_6 \text{VPH}_i - w_7 \text{SAT}_i$$

---

## ⚖️ EU AI Act & YouTube Policy Compliance

- Metadata tag `syntheticContent: true` and `bInformed: true` injected automatically during YouTube Data API ingestion.
- 100% freshly rendered AI visuals (Flux.1) + CC0 ambient background tracks to guarantee zero copyright strikes.
- Daily upload quota safety limit: Max 4 uploads per day (6,400 / 10,000 daily API units).
