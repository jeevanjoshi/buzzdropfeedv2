#!/usr/bin/env python3
import os
import sys
import json
import argparse
from dotenv import load_dotenv

# Ensure local packages are loadable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.engine.youtube_video_verifier import YouTubeVideoVerifier

def generate_markdown_report(report: dict, output_path: str):
    """Generates a beautiful markdown report of the audit findings."""
    video_id = report["video_id"]
    video_title = report["video_title"]
    topic = report["target_topic"]
    description = report.get("description", "")
    seo_tags = report.get("seo_tags", [])
    
    sync_sec = report["sync_analysis"]
    content_sec = report["content_analysis"]
    
    # Calculate sync score or status
    sync_issues = sync_sec["issues"]
    total_sync_issues = sync_sec["total_issues"]
    
    # Coherence and Relevance
    coherence_score = content_sec.get("coherence_score", 0.0)
    relevance_score = content_sec.get("relevance_score", 0.0)
    
    tags_formatted = ", ".join([f"`{t}`" for t in seo_tags]) if seo_tags else "*None*"
    desc_snippet = description[:300] + "..." if len(description) > 300 else description
    
    md = f"""# YouTube Video Quality & Sync Audit Report

This report evaluates the subtitle synchronization, speech accuracy, script coherence, and RAG-based relevance of the published YouTube video.

## Video Metadata
* **Video Title**: `{video_title}`
* **YouTube Video ID**: `{video_id}`
* **Target Topic / Domain**: `{topic}`
* **Video URL**: [https://youtube.com/watch?v={video_id}](https://youtube.com/watch?v={video_id})
* **Subtitles Checked**: {report["subtitle_count"]} segments
* **SEO Tags / Keywords**: {tags_formatted}
* **Description Snippet**:
  > {desc_snippet or "*No description found*"}

---

## Executive Summary
| Category | Score / Status | Description |
| :--- | :--- | :--- |
| **Subtitle & Speech Sync** | { "⚠️ WARN" if total_sync_issues > 0 else "✅ PASS" } | {f"Found {total_sync_issues} synchronization or text mismatch issues." if total_sync_issues > 0 else "All subtitles are perfectly synced."} |
| **Script Coherence** | `{coherence_score}/10` | Overall structure, transitions, and flow of script. |
| **Topic Relevance (RAG)** | `{relevance_score}/10` | Alignment with topic facts and anti-slop check. |

---

## 1. Speech vs Subtitle Sync Audit
"""
    if total_sync_issues == 0:
        md += "✅ No sync drift or word mismatches detected between spoken audio and subtitles.\n"
    else:
        md += f"Found **{total_sync_issues}** issue(s):\n\n"
        md += "| Type | Timestamp / Index | Description | Severity |\n"
        md += "| :--- | :--- | :--- | :--- |\n"
        for issue in sync_issues:
            itype = issue.get("type", "unknown").upper()
            t_start = issue.get("subtitle_start", "-")
            desc = issue.get("description", "")
            sev = issue.get("severity", "MEDIUM")
            md += f"| {itype} | {t_start}s (Sub #{issue.get('subtitle_index')}) | {desc} | {sev} |\n"
            
    md += f"""

---

## 2. Script Coherence & Flow Findings
* **Coherence Score**: `{coherence_score}/10`

### Findings
"""
    for finding in content_sec.get("coherence_findings", []):
        md += f"- {finding}\n"
        
    md += f"""

---

## 3. RAG Relevance & "AI Slop" Audit
* **Relevance Score**: `{relevance_score}/10`
* Grounding search context fetched using Exa.

### Irrelevant or Tangent Segments Detected
"""
    irrelevant = content_sec.get("irrelevant_segments", [])
    if not irrelevant:
        md += "✅ No irrelevant, repetitive, or off-topic segments detected in the narration.\n"
    else:
        for idx, seg in enumerate(irrelevant):
            md += f"#### Tangent #{idx+1}\n"
            md += f"> **Segment Text**: \"{seg.get('text', '')}\"\n\n"
            md += f"- **Reason**: {seg.get('reason', '')}\n"
            md += f"- **Suggested Fix**: {seg.get('suggested_fix', '')}\n\n"
            
    md += """

---

## 4. Metadata & SEO Optimization Audit
"""
    seo_issues = content_sec.get("metadata_seo_issues", [])
    if not seo_issues:
        md += "✅ Title, description, and SEO tags are perfectly aligned with transcript content.\n"
    else:
        md += "| Field | Issue Found | Suggested Fix |\n"
        md += "| :--- | :--- | :--- |\n"
        for issue in seo_issues:
            md += f"| {issue.get('field', '')} | {issue.get('issue', '')} | {issue.get('suggested_fix', '')} |\n"
            
    md += """

---

## 5. Actionable Fix Suggestions & Structural Recommendations
"""
    for fix in content_sec.get("structural_fixes", []):
        md += f"- [ ] {fix}\n"
        
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[Verifier] Markdown report written to: {output_path}")

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Audit YouTube video subtitles, speech synchronization, and relevance.")
    parser.add_argument("video_url", help="YouTube video URL or 11-char Video ID")
    parser.add_argument("--topic", help="Optional target topic to evaluate relevance (RAG)")
    parser.add_argument("--model", default="google/gemini-2.5-flash", help="LLM model name to use for evaluation")
    parser.add_argument("--out", help="Optional output markdown file path")
    args = parser.parse_args()

    # Determine artifact output path
    artifact_dir = "/home/ubuntu/.gemini/antigravity-cli/brain/feae1fc1-65ff-4a5d-b964-938ba667f598"
    os.makedirs(artifact_dir, exist_ok=True)
    out_path = args.out or os.path.join(artifact_dir, "youtube_audit_report.md")

    print("=== Starting YouTube Quality Audit ===")
    print(f"Video URL/ID: {args.video_url}")
    if args.topic:
        print(f"Target Topic (RAG): {args.topic}")
        
    verifier = YouTubeVideoVerifier(model_name=args.model)
    try:
        report = verifier.verify_video(args.video_url, target_topic=args.topic)
        
        # Save JSON raw report
        json_out = out_path.replace(".md", ".json")
        with open(json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Verifier] Raw JSON report saved to: {json_out}")
        
        # Generate beautiful markdown report
        generate_markdown_report(report, out_path)
        print("=== Audit Completed Successfully ===")
        
    except Exception as e:
        print(f"ERROR running YouTube Video Verifier: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
