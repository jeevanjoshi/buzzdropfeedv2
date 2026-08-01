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
   │   ├── engine/                      (TOPSIS Math, EMA Velocity, Cosine RPM, RSS Ingestion, Logger, Tracer, APINinjas)
   │   └── schemas/                     (GlobalState, A2AMessage)
   ├── mcp_servers/media_cloud/         <-- Cloud Media MCP Server (Port 8001)
   │   └── server.py                    (Flux.1 16:9, Ken Burns, Dynamic Charts, Playwright SVG, Thumbnails, FFmpeg Timeline)
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

---

## 🚀 One-Command Multi-Node Remote Deployment (`deploy.sh`)

Deploy all code, dependencies, systemd services, and cron jobs to both **Raspberry Pi 5 (`172.198.1.30`)** and **OCI Cloud (`oci-prod`)** with a single command:

```bash
chmod +x deploy.sh
./deploy.sh
```

---

## 🌐 Master API Resource Directory & Evaluation Matrix

Below is the curated directory of external APIs, evaluated for integration into our CSVG Topic Engine, RAG Fact Grounding, and Dynamic Chart Generator:

### 1. Facts & Figures APIs
Structural data, public datasets, economics, demographics, and trivia.

*   **[Data.gov.in OGD Platform API](https://data.gov.in)**: *India-Specific / Free (HIGH PRIORITY).* Open data initiative by the Govt of India (230,000+ datasets). Excellent for Indian macroeconomic, agricultural, and infrastructure Infotainment videos.
*   **[API Setu India](https://apisetu.gov.in)**: *India-Specific / Free.* Official Open API platform from MeitY for public data utilities.
*   **[World Bank Data API](https://data.worldbank.org)**: *Free (HIGH PRIORITY).* Global macroeconomic indicators, GDP growth, inflation, and demographic statistics.
*   **[API-Ninjas Facts & Business API](https://api-ninjas.com)**: *Integrated in `src/engine/api_ninjas.py`.* Business news and interesting facts data provider.
*   **[Bureau of Labor Statistics (BLS) API](https://www.bls.gov/developers/)**: Structured US economic data, inflation, employment, and wage indicators.
*   **[Public APIs Repository](https://github.com/public-apis/public-apis)**: Community-curated collection of free public APIs across various niches.

---

### 2. Research Papers APIs
Programmatic discovery of scientific literature, citations, and open-access metadata.

*   **[OpenAlex API](https://openalex.org)**: *Free CC0.* Catalog aggregating Crossref, arXiv, and PubMed. Perfect for deep-dive AI / Science documentaries.
*   **[Semantic Scholar Academic Graph API](https://semanticscholar.org)**: AI-driven API for retrieving paper metadata and influential citations.
*   **[arXiv API](https://arxiv.org)**: Non-commercial access to preprints in physics, AI, computer science, and mathematics.
*   **[Springer Nature Open Access API](https://springernature.com)**: Access for open-access articles from thousands of scientific journals.
*   **[Unpaywall API](https://unpaywall.org)**: Tracking for over 30 million open-access scholarly articles.
*   **[CORE API](https://core.ac.uk)**: The world's largest aggregator of open-access research papers.

---

### 3. News APIs
Live breaking headlines, historical articles, and text-mining global media.

*   **[Marketaux API](https://marketaux.com)**: *Generous Free Tier (HIGH PRIORITY).* Dedicated stock market & global financial news API with integrated AI ticker sentiment analysis.
*   **[NewsAPI.org (India Endpoint)](https://newsapi.org)**: *HIGH PRIORITY.* Dedicated `country=in` parameter to isolate real-time breaking headlines across India.
*   **[World News API (India Track)](https://worldnewsapi.com)**: Monitors 300+ primary regional and national Indian media sources.
*   **[Firecrawl News Search](https://firecrawl.dev)**: Converts live web search results into clean Markdown for LLMs.
*   **[NewsAPI.ai (Event Registry)](https://newsapi.ai)**: Multilingual indexer tracking 150,000+ global publishers.
*   **[Currents News API](https://currentsapi.services)**: Free tier offering regional and category-filtered news feeds.

---

### 4. Trends & Financial APIs
Market momentum, search interest, real-time public curiosity, and stock metrics.

*   **[Alpha Vantage API](https://www.alphavantage.co/)**: *Generous Free Tier (HIGH PRIORITY).* Global stock quotes, forex, crypto, and 50+ technical indicators. Directly powers `/tools/generate_dynamic_chart`.
*   **[Indian Stock Market API](https://indianapi.in)**: *India-Specific (HIGH PRIORITY).* Granular NSE/BSE stock quotes, market trends, and technical metrics for listed Indian corporations.
*   **[Database on Indian Economy (DBIE - RBI)](https://rbi.org.in)**: *India-Specific / Free.* Reserve Bank of India macroeconomic trends across public finance and financial markets.
*   **[Exa Search API](https://exa.ai)**: *Built for AI Agents ($10/mo free credit).* Semantic web search and real-time trend discovery.
*   **[Google Trends API / SerpAPI](https://serpapi.com)**: scaled search interest data and "Interest Over Time" tables.

---

## 💻 Hardware Sizing & Local LLM Compatibility

| Host Node | Hardware Profile | Local Models Supported | Zero-API Cost Setup |
|---|---|---|---|
| **OCI Cloud Instance** | 2 OCPUs, 12 GB RAM (Ubuntu 22.04) | **Ollama**: `qwen2.5:7b-instruct` / `mistral-7b` | **$0.00** (Full local 7B LLM scripting) |
| **Raspberry Pi 5** | 4 GB RAM (Raspberry Pi OS 64-bit) | **Ollama**: `llama3.2:3b` / `qwen2.5:3b`<br>**Speech**: `Kokoro-ONNX` (82M) & `Faster-Whisper` | **$0.00** (Full local 3B LLM & neural audio) |

---

## 📥 Installation & Setup Guide

### 1. System Dependencies (Linux / Ubuntu / Raspberry Pi OS)
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
git clone https://github.com/jeevanjoshi/buzzdropfeedv2.git
cd buzzdropfeedv2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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

# Offline Dry-Run Mode
python3 main.py --offline
```

### Run Comprehensive Automated Test Suite (12 Unit & Integration Tests):
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
