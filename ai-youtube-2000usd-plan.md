# Plan: $2,000/Month from a Fully AI-Generated YouTube Channel

## 0. The Math, Upfront

- Sourced CPM for AI/Tech content (2026): **$10–25 CPM** (vidIQ, "Tier 2 Rising"). RPM typically runs 30–50% below CPM after YouTube's 45% cut → roughly **$5–14 RPM** — but only if your audience skews US/UK/Canada/Australia.
- **Geography matters as much as niche:** Tier-3 regions (India, Brazil, SE Asia) generate CPMs **3–5x lower** than Tier-1 (US/UK/Canada/Australia) on identical content (vidIQ). This is why topic selection dynamically favors Tier-1 search demand rather than any fixed region — see Section 1.
- At a validated cadence (dynamically set per Section 6 — starts around 3-4 videos/week, scales toward 2-3/day only if retention holds), clearing $2,000/month requires roughly **1,700–6,700 views/video**, depending on how much cadence actually scales and how close your audience mix lands to Tier-1 geography.
- AdSense alone gets you there slowly. Affiliate links and sponsorships (once you have ~10K subs) can cut the timeline significantly — build for both from day one.

**Realistic timeline: 4–8 months to first $2,000 month**, assuming consistent publishing, Tier-1-leaning audience capture, and no monetization strikes.

---

## 1. Niche & Dynamic Geographic Targeting

**Primary niche: AI/Tech only** — no crossover into finance, career, or industry content. Not because those don't pay more (they do), but because they need domain expertise and fact-checking outside tech, which is exactly what an unattended pipeline can't safely verify. Pure AI/Tech content is checkable against official docs, changelogs, and pricing pages — that's what actually makes automation viable at higher cadence.

Target specific, narrow questions with real search demand, evaluated dynamically per candidate topic:
- "[Tool A] vs [Tool B] for [specific task]" — evergreen until a tool is discontinued
- "How to [specific technical task] using [AI tool]" — tutorial-style, high search volume
- Individual tool deep-dive reviews
- Enterprise/developer tooling comparisons — see Section 2 for concrete examples

**Geographic targeting is dynamic, not fixed to any one region:** every candidate topic gets checked against Google Trends' "interest by region" data before scripting (Section 4, step 1). Topics are prioritized when search interest concentrates in Tier-1 CPM countries (US/UK/Canada/Australia), regardless of where you're producing from. Scripts and titles use globally-neutral English phrasing by default.

Why this fits you specifically: you already work with Salesforce/Apex/JS/cloud infra, so you can sanity-check scripts fast, spot AI hallucinations before publishing, and cover dev/SaaS-tool content with real domain credibility — content policy explicitly rewards this kind of genuine expertise over generic narration.

---

## 2. Starter Video List: 4 Pure AI/Tech Templates

Four templates, all staying inside AI/Tech, each fully AI-producible (screen recordings/stock B-roll + AI voiceover + on-screen text/diagrams — no real footage needed) and fact-checkable against official product docs, changelogs, or pricing pages. Treat these as templates: the pipeline should dynamically fill in the bracketed specifics per Section 4's validation workflow, not lock to the exact examples below.

| # | Template | Example | CPM |
|---|---|---|---|
| 1 | "[AI Tool A] vs [AI Tool B]: Which Is Better for [Specific Task]?" | "Claude vs ChatGPT for Coding: Which Is Actually Better in 2026?" | $10–30 |
| 2 | "How to [Specific Technical Task] Using [AI Tool]" | "How to Automate Your Inbox Using [AI Tool]" | $8–20 |
| 3 | "[AI Tool] Review: Is It Worth It in 2026?" | "Cursor AI Review: Is It Worth It in 2026?" | $8–25 |
| 4 | "[Category] AI Tools Compared for Enterprise/Developer Teams" | "AI Coding Assistants for Enterprise Teams: GitHub Copilot vs Cursor vs Claude Code" | $20–30 (top of range) |

Running all 4 templates in rotation, rather than one repeated format, also directly helps with the "inauthentic content" originality requirement in Section 3 — each template requires different research, structure, and visuals, while staying inside a niche narrow enough for the pipeline to fact-check unattended.

> **Note (aspirational templates):** in the current pipeline (`buzzdropfeedv2`) these review/"best-of"/"vs" templates are largely **filtered out at ingestion** by the promotional/listicle block in `rss_ingestion.py` (`_is_promotional_listicle`), which deliberately drops "Top N", "Best X", and "X vs Y" clickbait from the candidate pool. The channel therefore leans on high-RPM *news/explainer* automation rather than these review formats — treat Section 2's list as aspirational direction, not what the pipeline currently auto-produces.

---

## 2a. Other Niches That Also Fit Full Automation (Optional Expansion)

AI/Tech isn't the *only* niche safe to automate — it's just the strongest fit for you specifically (Section 1). These clear the same bar (facts verifiable against an authoritative source, no regulatory/liability risk, no real footage needed):

| Niche | Why it's automatable | CPM |
|---|---|---|
| **History / documentary-style** | Facts checkable against established historical record; archival images + maps + AI narration; evergreen — doesn't decay like news | $6–14 |
| **Science / space explainers** | Facts checkable against scientific consensus; same epistemic profile as tech | $5–15 |
| **Personal-finance *education*** (concepts only — "how compound interest works," "how a 401k works" — never stock picks or buy/sell calls) | Explaining established financial mechanics is as verifiable as explaining a tech tool; highest CPM of this group | $15–40+ |

**If you want to expand:** run these as a **separate channel**, not folded into the AI/Tech one — mixing niches on one channel dilutes the algorithm's understanding of what your channel is "about," which hurts suggested/browse traffic (Section 6). Treat them as a second pipeline instance later, or a fallback if AI/Tech saturates, not a day-one parallel track.

**Niches to avoid for full automation**, despite being common "faceless channel" picks:
- **Health/wellness** — YMYL (Your Money or Your Life) content gets heavy scrutiny for source authority; high misinformation risk unattended.
- **True crime** — needs editorial judgment on sourcing/sensitivity that's hard to automate safely.
- **Stock/investment picks** (as opposed to finance education) — regulatory and legal-liability risk, not just accuracy risk.

---

## 3. Non-Negotiable: YouTube Monetization Policy Compliance

The single biggest risk to a high-frequency automated pipeline. As of 2026, YouTube's **inauthentic content policy** explicitly targets channels using AI/templates to mass-produce near-identical videos with no original insight — this can demonetize your *entire channel*, not just individual videos.

**Build these into the pipeline from day one, not after a strike:**
1. Every video must add a genuine angle, opinion, or synthesis — not just "here's what happened." Script prompts should require a take, not a summary.
2. Rotate across the 4 templates in Section 2 rather than repeating one format — identical templates repeated at scale is exactly what triggers review.
3. Enable the **"altered or synthetic content" disclosure** toggle in YouTube Studio for every video.
4. Avoid pure TTS-over-stock-slideshow with zero editing — add b-roll relevance, on-screen text that matches narration timing, pacing cuts.
5. Don't republish/rephrase material already covered elsewhere without substantive original commentary — this trips the separate "reused content" policy.

**Practical implication:** higher cadence is workable, but only if template rotation + human review checkpoints (Section 4) are non-negotiable parts of every cycle — not corners cut to hit volume. If review capacity is ever the bottleneck, drop cadence before dropping originality (this is also what Section 6's dynamic rule enforces automatically).

---

## 4. Topic Validation & Launch Workflow (Per Video)

Goal: confirm real demand, underserved angle, and Tier-1 geographic pull — before any script is written.

1. **Trend detection** (existing 6h scraper) surfaces raw candidate topics.
2. **Google Trends "interest by region"** (free, no quota) — check where search interest concentrates. Deprioritize topics skewed to Tier-3-only regions; prioritize Tier-1 (US/UK/Canada/Australia) or broad global interest.
3. **Google Trends trajectory** — confirm interest is rising or stable, not past its peak.
4. **YouTube `search.list`** (100 units/call, 100 calls/day free tier) — pull top 20-50 existing videos for the angle.
5. **YouTube `videos.list`** (1 unit per batch of 50 IDs) — get view counts for those results.
6. **Opportunity Score** — the implemented `opportunity_score` (views-per-competitor, `competitor_30d_avg_views ÷ max(1, competing_video_count)`, log-tamed to [0, 1]) computed at RSS ingestion, plus a first-pass **B1 shortlist precise check** on the TOPSIS top-3 (one on-topic `search.list` + one `videos.list` batch to get true average views/count). A measured-but-low opportunity (below `OPPORTUNITY_MIN_SCORE`) is a hard gate in REVENUE/SCALE phases; unmeasured topics fall through to TOPSIS. YouTube quota budget is **10k units/day** (free tier): `search.list` = 100 units/call (budgeted calls/day), `videos.list` batch = 1 unit, `videos.update/insert` (upload) = 1600 units.
7. **Template assignment** — map the greenlit topic to whichever of the 4 templates (Section 2) fits, rotating so no template repeats back-to-back.
8. **Script generation** — prompt requires a genuine angle/take (Section 3, rule 1), not a summary.
9. **Fact-check pass** — flag anything ambiguous for a closer human look before TTS.
10. **TTS voiceover** — free Kokoro TTS served on the Raspberry Pi edge node (`audio_edge`, port 8000). No ElevenLabs subscription; Kokoro is commercially usable in monetized content.
11. **Visual assembly** — handled by the existing `src/agents/` A2A orchestrator (Flux visuals + ffmpeg assembly + sidechain-ducked audio mix), not an external CLI.
12. **Render** — 1080p on OCI Ampere free-tier compute.
13. **Human review checkpoint** — compliance (originality, disclosure-worthy AI use) + quality, before publish. Non-negotiable for at least the first 2–3 months.
14. **Metadata** — globally-neutral English title/description, "altered or synthetic content" disclosure toggle on, affiliate links for any tools covered.
15. **Schedule** — **out of scope for now.** The pipeline publishes immediately (till-upload / auto-publish); the WHEN algorithm (Section 6) is deferred. Keep "synthetic content" disclosure-on (step 14).
16. **Feedback loop** — after 48–72h, check YouTube Analytics' audience geography + retention (AVD) + CTR for that video. Feed this into both future topic scoring (Section 1) and the cadence rule (Section 6).

Budget your 100 daily `search.list` calls across cycles (e.g., ~15-25 candidate checks per 6h cycle) so you're validating enough candidates without wasting quota on obviously weak topics.

---

## 5. Production Setup (Using Your Existing Infrastructure)

- **Trend detection:** existing job scraping trending searches every 6h → feeds Section 4's workflow.
- **Script + editing orchestration:** the existing `src/agents/` A2A orchestrator (fact_retriever → story_designer → observer → media_producer → publisher); no external CLI.
- **Compute:** Raspberry Pi 5 edge node handles TTS (Kokoro) and light tasks; OCI Ampere free tier (2 OCPU/12GB) runs the master pipeline (agents, media_cloud/ffmpeg, youtube_cloud).
- **TTS:** free Kokoro TTS on the Pi (`audio_edge`, port 8000) — see Section 4, step 10.
- **Audio mix:** the ffmpeg renderer sidechain-ducks BGM under narration (`BGM_VOLUME`, `BGM_SIDECHAIN_THRESHOLD` env-tunable); the moviepy fallback uses a flat reduced `BGM_VOLUME`.
- **Source policy:** social/community platforms (Reddit, X/Twitter, Facebook, Instagram, LinkedIn, TikTok, Quora, Pinterest, Snapchat, Threads, Discord, Medium, Substack, forums) are **excluded from the RAG fact corpus** in BOTH the scraper path and Google grounding (`--rag grounded`/`hybrid`).
- **Publishing:** dedicated YouTube channel account, separate from personal accounts.
- **Long-form only, not Shorts, for revenue:** Shorts CPM runs roughly 95% lower than long-form in the same niche — pooled ad revenue split caps payout regardless of niche CPM. Use Shorts only as a discovery funnel into long-form, never as primary revenue.

---

## 6. Publishing Algorithm: When and How Much

> **Out of scope for now.** The current pipeline does **not** schedule uploads to audience-local timezones or auto-adjust cadence by AVD/CTR feedback — it publishes on demand (till-upload / auto-publish). Keep the research below as a design reference for a future scheduling layer.

### WHEN — timing rule
YouTube tests every new upload against a small slice of your audience first; strong early watch time + CTR in that window is what earns wider distribution. So the goal is to be **live and indexed 2–4 hours before your audience's peak viewing time — in their local timezone, not yours.**

- **Target windows (audience-local time):** weekdays 12 PM–4 PM (index before the evening surge), Thursday/Friday are historically the strongest long-form days; weekends 9–11 AM.
- **Your specific wrinkle:** your target audience is Tier-1 (US/UK/Canada/Australia) while you're producing from India — don't do this math by hand. Set each video's **Schedule card timezone directly to the target audience's timezone** (e.g., US Eastern or Pacific) and pick a time inside the window above; YouTube handles the conversion, so you're not guessing IST offsets.
- **Consistency beats a "perfect" slot:** channels with zero upload gaps over a 12-month period get better algorithm distribution than channels with the same total video count but with gaps — so once you pick a cadence, don't pause it; scale it down instead (see below).

### HOW MUCH — frequency rule (dynamic, not fixed)
2026 research is genuinely mixed on raw volume, but converges on one point: **the algorithm rewards retention and consistency, not upload count by itself** — daily posting only helps if quality holds, and YouTube throttles distribution fast when it detects declining engagement per video.

Use a feedback rule instead of a fixed number:
1. **Start** at 3-4 videos/week (Validate phase, Section 2a's rotation gives you enough template variety to sustain this).
2. **Scale up** by ~1 video/week only if your trailing 5-video average AVD% and CTR are flat or improving vs. the previous 5-video window.
3. **Scale down immediately** if trailing average AVD drops more than ~15% or CTR drops meaningfully — treat this as the algorithm's throttle signal, don't wait for it to compound.
4. **Hard ceiling: 3/day**, even if metrics allow more — human review capacity (Section 3) is the real bottleneck, not the pipeline.
5. **Never fully stop** — if cadence needs to drop, drop to a sustainable weekly minimum rather than pausing, to protect the no-gaps consistency signal above.

| Phase | Timeline | Starting Cadence | Adjust via the rule above |
|---|---|---|---|
| 1. Validate | Weeks 1–4 | 3-4 videos/week | Test all 4 templates, establish baseline AVD/CTR |
| 2. Commit | Months 2–3 | Scale toward 1-2/day | Apply for YPP (1,000 subs + 4,000 watch hours) once eligible |
| 3. Scale | Months 4–6 | Scale toward 2-3/day, if retention holds | Add affiliate links, pitch small sponsors |
| 4. Compound | Months 6–8+ | Whatever cadence retention supports | Evergreen backfill; older videos now contribute a steady baseline |

**Seasonality tip:** CPMs peak in **Q4 (Oct–Dec)** — sometimes 2–3x the annual average — and crash in the **first weeks of January** (down ~50%) as advertiser budgets reset. Save your most polished uploads for Q4; use the January dip to backfill evergreen content rather than chasing view count.

---

## 7. Revenue Diversification (Don't Rely on AdSense Alone)

- **AdSense:** baseline, scales with views.
- **Affiliate links:** for a tools/tutorials niche, this is often bigger than AdSense — link the tools you review/explain in every relevant video description.
- **Sponsorships:** realistic once you cross ~10K subs in a tech/tools niche — reach out to smaller SaaS tools directly, they're often cheaper to work with than expected and eager for faceless-channel placements.

---

## 8. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Demonetization for "inauthentic content" | Genuine angle per video, rotate templates, enable AI disclosure toggle |
| YouTube API quota exhaustion | Google Trends first (free), search.list only for shortlisted candidates |
| Topic decays before video finishes rendering | Favor evergreen-framed angles over pure breaking-news |
| Channel sounds identical to competitors | Distinct voice/persona choice, consistent thumbnail branding, 4-template rotation |
| Cadence outpaces quality, algorithm throttles distribution | Dynamic cadence rule (Section 6) — scale down on retention/CTR drop, never scale on a fixed schedule alone |
| Burnout / review bottleneck at high cadence | Human review checkpoint stays non-negotiable; drop cadence before dropping originality |
| Audience geography suppresses RPM | Topic selection dynamically weights Tier-1 search interest (Section 4); track actual audience geography in YouTube Analytics and feed it back into topic scoring |

---

## Sources
- CPM/RPM tiers, Shorts vs long-form gap, and seasonality data: vidIQ, ["Highest-Paying YouTube Niches in 2026"](https://vidiq.com/blog/post/most-profitable-youtube-niches/), Apr 2026.
- Upload timing / index-before-peak logic: SocialPilot, ["Best Time to Post on YouTube in 2026"](https://www.socialpilot.co/insights/best-time-to-post-on-youtube) (301K videos analyzed); PostFast, ["Best Time to Post on YouTube in 2026"](https://postfa.st/blog/best-time-to-post-on-youtube).
- Upload frequency vs. retention/algorithm trust: Theme Circle, ["YouTube Upload Frequency Explained: What Works Best in 2026"](https://www.themecircle.net/youtube-upload-frequency-explained-what-works-best-in-2026/) — note: some datasets (e.g. FluxNote) report daily/2x-weekly cadences outperforming weekly *if* quality holds, so this is treated as a feedback rule rather than a fixed number.
- TTS: local **Kokoro** on the Raspberry Pi edge node (`audio_edge`) — no external subscription (ElevenLabs line dropped as out of scope).
