# Autonomous YouTube Storytelling Video Generation (CSVG) Pipeline

Production-grade, zero-human-intervention **Customized Storytelling Video Generation (CSVG)** pipeline designed for high-retention 10–15 minute 16:9 widescreen Infotainment YouTube channels targeting **$2,000+ USD / Month** ad revenue.

---

## 🏗️ Codebase Split & Deployment Mapping (OCI vs. Raspberry Pi 5)

```
               OCI CLOUD NODE (/opt/csvg_pipeline)
   [2 OCPUs, 12 GB RAM, Ubuntu 22.04 LTS - Cron: 0 4 * * *]
   ├── main.py                          <-- Main Pipeline Entrypoint
   ├── src/                             <-- All Agents & Engine Modules
   │   ├── agents/                      (Orchestrator, FactRetriever, StoryDesigner, Observer, MediaProducer, Publisher)
   │   ├── engine/                      (TOPSIS Math, EMA Velocity, Cosine RPM, RSS Ingestion, Logger, Tracer)
   │   └── schemas/                     (GlobalState, A2AMessage)
   ├── mcp_servers/media_cloud/         <-- Cloud Media MCP Server (Port 8001)
   │   └── server.py                    (Flux.1 16:9, Ken Burns, Dynamic Charts, Thumbnails, FFmpeg Timeline)
   └── mcp_servers/youtube_cloud/       <-- YouTube Publishing MCP Server (Port 8002)
       └── server.py                    (Headless OAuth2 Upload & Quota Guard)

                               │
                       HTTP / SSE REST Bridge
                    (http://<PI5_IP>:8000/tools/...)
                               │
                               v

          RASPBERRY PI 5 EDGE NODE (/opt/csvg_edge)
         [4 GB RAM, Raspberry Pi OS - Systemd Service]
   ├── mcp_servers/audio_edge/          <-- Edge Audio MCP Server (Port 8000)
   │   └── server.py                    (Kokoro-ONNX TTS Synthesis & Faster-Whisper .ass Subtitles)
   ├── kokoro-v0.19.onnx                <-- Local Speech Weights (82M)
   └── voices.bin                       <-- Multi-Voice Intonation Blend Weights
```

---

## 💻 Hardware Sizing & Local LLM Compatibility

| Host Node | Hardware Profile | Local Models Supported | Zero-API Cost Setup |
|---|---|---|---|
| **OCI Cloud Instance** | 2 OCPUs, 12 GB RAM (Ubuntu 22.04) | **Ollama**: `qwen2.5:7b-instruct` / `mistral-7b` | **$0.00** (Full local 7B LLM scripting) |
| **Raspberry Pi 5** | 4 GB RAM (Raspberry Pi OS 64-bit) | **Ollama**: `llama3.2:3b` / `qwen2.5:3b`<br>**Speech**: `Kokoro-ONNX` (82M) & `Faster-Whisper` | **$0.00** (Full local 3B LLM & neural audio) |

---

## 📥 Installation & Setup Guide

### 1. System Dependencies (Linux / Ubuntu / Raspberry Pi OS)
Install FFmpeg, Python 3.9+, git, and build tools:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv ffmpeg git curl -y
```

### 2. Setting Up Local LLM (Ollama)

#### On OCI Cloud Node (12 GB RAM):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct
```

#### On Raspberry Pi 5 Node (4 GB RAM):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

### 3. Repository Installation & Python Virtual Environment
```bash
git clone https://github.com/organization/buzzdropfeedv2.git
cd buzzdropfeedv2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Environment Configuration (`.env`)
Create a `.env` file in the project root on OCI:

```ini
# Local LLM Endpoint (Ollama on OCI or Raspberry Pi 5)
OLLAMA_URL=http://localhost:11434

# Cloud Fallback API Keys (Optional)
OPENROUTER_API_KEY=your_openrouter_key_here
FAL_KEY=your_fal_ai_key_here

# YouTube OAuth & Hardware Endpoints
YOUTUBE_CLIENT_SECRET=client_secret.json
YOUTUBE_TOKEN_FILE=token.json
PI5_IP=192.168.1.150
```

---

## 🚀 Running the Autonomous Pipeline

### Run Full End-to-End Autonomous Pipeline (Live RSS Feeds):
```bash
python3 main.py
```

### Run Pipeline for Specific Target Regions:
```bash
# Target Indian English Business & Tech News Feeds (ET, Livemint, Moneycontrol, YourStory, Inc42)
python3 main.py --india

# Target Global Finance & Tech Feeds (CNBC, TechCrunch, NYT)
python3 main.py --global

# Offline Dry-Run Mode (Testing without live API calls)
python3 main.py --offline
```

### Run Comprehensive Automated Test Suite (12 Unit & Integration Tests):
```bash
python3 run_tests.py
```

---

## ⚙️ Systemd Persistent Services & Cron Automation

### 1. Raspberry Pi 5 Edge Audio MCP Service (`/etc/systemd/system/kokoro_tts.service`)
```ini
[Unit]
Description=Raspberry Pi 5 Edge Audio & Local Model Service
After=network.target

[Service]
User=pi
WorkingDirectory=/opt/csvg_edge
ExecStart=/opt/csvg_edge/venv/bin/python3 /opt/csvg_edge/mcp_servers/audio_edge/server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service on Pi 5:
```bash
sudo systemctl daemon-reload
sudo systemctl enable kokoro_tts.service
sudo systemctl start kokoro_tts.service
```

### 2. OCI Cron Job Automation
Schedule the pipeline on OCI to run daily at 04:00 AM server time (ahead of peak Tier-1 viewing hours):

```bash
crontab -e
# Add the following job:
0 4 * * * /opt/csvg_pipeline/venv/bin/python3 /opt/csvg_pipeline/main.py >> /var/log/csvg.log 2>&1
```

---

## 📊 TOPSIS 7-Criteria Topic Selection Math

Given candidate stories $i \in \mathcal{S}$, TOPSIS calculates relative closeness score $C_i^* \in [0, 1.0]$:

$$V_i = w_1 \text{TVS}_i + w_2 \text{RPM}_i + w_3 \text{IDI}_i + w_4 \text{SDI}_i + w_5 \text{SHM}_i + w_6 \text{VPH}_i - w_7 \text{SAT}_i$$

- **TVS**: 7-Day Exponential Moving Average search query acceleration.
- **RPM**: Cosine similarity against High-RPM Finance/Tech advertiser taxonomy.
- **IDI**: Semantic novelty distance vs previously published channel history.
- **SDI**: Sentiment variance & conflict index for comment velocity.
- **SHM**: X/Twitter & Reddit post momentum multiplier.
- **VPH**: YouTube competitor Views-Per-Hour recommendation wave.
- **SAT**: Competition saturation penalty.

---

## ⚖️ EU AI Act & YouTube Policy Compliance

- Metadata tag `syntheticContent: true` and `bInformed: true` injected automatically during YouTube Data API ingestion.
- 100% freshly rendered AI visuals (Flux.1) + CC0 ambient background tracks to guarantee zero copyright strikes.
- Daily upload quota safety limit: Max 4 uploads per day (6,400 / 10,000 daily API units).
