#!/usr/bin/env python3
import os
import sys
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv

# Load env variables
load_dotenv()

NOTIFY_EMAIL_FROM = os.getenv("NOTIFY_EMAIL_FROM", "buzzdropfeed@gmail.com")
NOTIFY_EMAIL_TO = os.getenv("NOTIFY_EMAIL_TO", "jeevan.z.joshi@gmail.com")
NOTIFY_EMAIL_PASSWORD = os.getenv("NOTIFY_EMAIL_PASSWORD", "")
NOTIFY_SMTP_HOST = os.getenv("NOTIFY_SMTP_HOST", "smtp.gmail.com")
NOTIFY_SMTP_PORT = int(os.getenv("NOTIFY_SMTP_PORT", "587"))

def _ypp_progress_html() -> str:
    """GROWTH-phase YPP progress block (subs/watch-hrs toward unlock), so the
    completion email surfaces how close the channel is to monetization."""
    try:
        import json
        stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "channel_stats.json")
        if not os.path.exists(stats_path):
            return ""
        with open(stats_path, "r") as f:
            stats = json.load(f)
        if stats.get("channel_phase", "GROWTH") not in ("GROWTH",):
            return ""
        from src.engine.channel_phase_manager import get_ypp_progress_report
        from src.schemas.state import ChannelStats
        prog = get_ypp_progress_report(ChannelStats(
            subscribers=int(stats.get("subscribers", 0)),
            total_watch_hours=int(stats.get("total_watch_hours", 0)),
            channel_phase=stats.get("channel_phase", "GROWTH"),
            ypp_unlocked=bool(stats.get("ypp_unlocked", False)),
        ))
        return (
            "<h3>YPP Progress (GROWTH phase)</h3>"
            "<p>"
            f"Subscribers: {prog['subs_progress_pct']:.1f}% ({prog['subs_remaining']} to go) &nbsp;|&nbsp; "
            f"Watch Hours: {prog['watch_hours_progress_pct']:.1f}% ({prog['watch_hours_remaining']} hrs to go) &nbsp;|&nbsp; "
            f"Est. days to YPP: ~{prog['estimated_days_to_ypp']}"
            "</p>"
        )
    except Exception as e:
        print(f"Warning: could not build YPP progress block: {e}")
        return ""


def send_status_email(status: str, log_file_path: str):
    if not NOTIFY_EMAIL_PASSWORD:
        print("Error: NOTIFY_EMAIL_PASSWORD is not set in .env. Skipping email notification.")
        sys.exit(1)

    subject = f"CSVG Pipeline Execution: {status.upper()}"
    ypp_block = _ypp_progress_html()
    
    # Read last 100 lines of log file for body snippet
    log_snippet = ""
    if os.path.exists(log_file_path):
        with open(log_file_path, "r", errors="ignore") as f:
            lines = f.readlines()
            log_snippet = "".join(lines[-100:])

    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
      <h2 style="color: {'#4CAF50' if status.lower() == 'success' else '#F44336'};">
        CSVG Pipeline Run: {status.upper()}
      </h2>
      <p>The autonomous YouTube generation pipeline has finished execution.</p>
      {ypp_block}
      <h3>Last 100 Lines of Log Output:</h3>
      <pre style="background: #f4f4f4; padding: 15px; border: 1px solid #ddd; overflow-x: auto; max-height: 400px; font-size: 12px;">
{log_snippet}
      </pre>
      <p style="color: #666; font-size: 11px;">This is an automated notification from your CSVG deployment.</p>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = f"CSVG Pipeline <{NOTIFY_EMAIL_FROM}>"
    msg["To"] = NOTIFY_EMAIL_TO
    msg.attach(MIMEText(html_content, "html"))

    # Attach log file
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, "rb") as f:
                attachment = MIMEApplication(f.read(), _subtype="txt")
                attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(log_file_path))
                msg.attach(attachment)
        except Exception as e:
            print(f"Warning: Failed to attach log file: {e}")

    try:
        with smtplib.SMTP(NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(NOTIFY_EMAIL_FROM, NOTIFY_EMAIL_PASSWORD)
            server.sendmail(NOTIFY_EMAIL_FROM, NOTIFY_EMAIL_TO, msg.as_string())
        print("Notification email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True, choices=["success", "failure"])
    parser.add_argument("--log_file", required=True)
    args = parser.parse_args()

    send_status_email(args.status, args.log_file)
