"""
Email Notifier — sends pipeline success/failure alerts via SMTP (Gmail / any SMTP).

Setup (Gmail):
    1. Enable 2-Factor Auth on your Google account
    2. Go to: https://myaccount.google.com/apppasswords
    3. Create an App Password for "Mail"
    4. Add to your .env file:
       NOTIFY_EMAIL_FROM=you@gmail.com
       NOTIFY_EMAIL_TO=you@gmail.com          # can be same or different
       NOTIFY_EMAIL_PASSWORD=xxxx xxxx xxxx xxxx   # 16-char app password
       NOTIFY_SMTP_HOST=smtp.gmail.com        # optional, defaults to Gmail
       NOTIFY_SMTP_PORT=587                   # optional, defaults to 587
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional


NOTIFY_EMAIL_FROM     = os.getenv("NOTIFY_EMAIL_FROM", "")
NOTIFY_EMAIL_TO       = os.getenv("NOTIFY_EMAIL_TO", "")
NOTIFY_EMAIL_PASSWORD = os.getenv("NOTIFY_EMAIL_PASSWORD", "")
NOTIFY_SMTP_HOST      = os.getenv("NOTIFY_SMTP_HOST", "smtp.gmail.com")
NOTIFY_SMTP_PORT      = int(os.getenv("NOTIFY_SMTP_PORT", "587"))


def _send_email(subject: str, html_body: str) -> bool:
    """
    Sends an HTML email via SMTP with TLS.
    Returns True on success, False on failure (never raises).
    """
    if not NOTIFY_EMAIL_FROM or not NOTIFY_EMAIL_TO or not NOTIFY_EMAIL_PASSWORD:
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"CSVG Pipeline <{NOTIFY_EMAIL_FROM}>"
        msg["To"]      = NOTIFY_EMAIL_TO
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(NOTIFY_EMAIL_FROM, NOTIFY_EMAIL_PASSWORD)
            server.sendmail(NOTIFY_EMAIL_FROM, NOTIFY_EMAIL_TO, msg.as_string())
        return True
    except Exception:
        return False


def notify_success(
    pipeline_id: str,
    topic: str,
    video_id: str,
    runtime_mins: float,
    topsis_score: float,
    dedup_sync: Optional[dict] = None
) -> None:
    """Sends a ✅ success email after a successful YouTube upload."""
    yt_url = f"https://www.youtube.com/watch?v={video_id}"
    dedup_row = ""
    if dedup_sync and dedup_sync.get("source") == "youtube_api":
        dedup_row = f"""
        <tr>
          <td style="padding:6px 12px;color:#888;">Dedup Sync</td>
          <td style="padding:6px 12px;">{dedup_sync.get('total_yt_titles', 0)} channel videos checked
          ({dedup_sync.get('synced', 0)} newly seeded)</td>
        </tr>"""

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
      <div style="background:#1db954;padding:20px 24px;">
        <h2 style="color:#fff;margin:0;">✅ Pipeline Succeeded</h2>
        <p style="color:#d4f5e0;margin:4px 0 0;">CSVG Autonomous YouTube Pipeline</p>
      </div>
      <div style="padding:24px;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f9f9f9;">
            <td style="padding:6px 12px;color:#888;width:140px;">Topic</td>
            <td style="padding:6px 12px;font-weight:bold;">{topic}</td>
          </tr>
          <tr>
            <td style="padding:6px 12px;color:#888;">Runtime</td>
            <td style="padding:6px 12px;">{runtime_mins:.1f} minutes</td>
          </tr>
          <tr style="background:#f9f9f9;">
            <td style="padding:6px 12px;color:#888;">TOPSIS Score</td>
            <td style="padding:6px 12px;">{topsis_score:.4f}</td>
          </tr>
          <tr>
            <td style="padding:6px 12px;color:#888;">Pipeline ID</td>
            <td style="padding:6px 12px;font-family:monospace;font-size:13px;">{pipeline_id}</td>
          </tr>
          <tr style="background:#f9f9f9;">
            <td style="padding:6px 12px;color:#888;">Video ID</td>
            <td style="padding:6px 12px;font-family:monospace;font-size:13px;">{video_id}</td>
          </tr>
          {dedup_row}
        </table>
        <div style="margin-top:20px;text-align:center;">
          <a href="{yt_url}"
             style="background:#ff0000;color:#fff;padding:12px 28px;border-radius:6px;
                    text-decoration:none;font-weight:bold;display:inline-block;">
            ▶ Watch on YouTube
          </a>
        </div>
      </div>
      <div style="background:#f5f5f5;padding:12px 24px;font-size:12px;color:#aaa;text-align:center;">
        CSVG Pipeline · Automated notification
      </div>
    </div>
    """
    _send_email(subject=f"✅ Pipeline Success — {topic[:60]}", html_body=html)


def notify_failure(
    pipeline_id: str,
    stage: str,
    error: str,
    fix_hint: str = ""
) -> None:
    """Sends a ❌ failure email when the pipeline crashes."""
    fix_row = ""
    if fix_hint:
        fix_row = f"""
        <tr style="background:#fff8e1;">
          <td style="padding:6px 12px;color:#888;">Fix Hint</td>
          <td style="padding:6px 12px;color:#e65100;">{fix_hint}</td>
        </tr>"""

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;">
      <div style="background:#d32f2f;padding:20px 24px;">
        <h2 style="color:#fff;margin:0;">❌ Pipeline Failed</h2>
        <p style="color:#ffcdd2;margin:4px 0 0;">CSVG Autonomous YouTube Pipeline</p>
      </div>
      <div style="padding:24px;">
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f9f9f9;">
            <td style="padding:6px 12px;color:#888;width:140px;">Failed Stage</td>
            <td style="padding:6px 12px;font-weight:bold;">{stage}</td>
          </tr>
          <tr>
            <td style="padding:6px 12px;color:#888;">Pipeline ID</td>
            <td style="padding:6px 12px;font-family:monospace;font-size:13px;">{pipeline_id}</td>
          </tr>
          <tr style="background:#fff3f3;">
            <td style="padding:6px 12px;color:#888;vertical-align:top;">Error</td>
            <td style="padding:6px 12px;font-family:monospace;font-size:12px;color:#c62828;">{error[:500]}</td>
          </tr>
          {fix_row}
        </table>
      </div>
      <div style="background:#f5f5f5;padding:12px 24px;font-size:12px;color:#aaa;text-align:center;">
        CSVG Pipeline · Automated notification
      </div>
    </div>
    """
    _send_email(subject=f"❌ Pipeline Failed — {stage}", html_body=html)


def notify_test() -> bool:
    """Sends a test email to verify SMTP config is working."""
    html = """
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;padding:24px;
                border:1px solid #e0e0e0;border-radius:8px;text-align:center;">
      <h2>🤖 CSVG Notifier — Test Email</h2>
      <p>Email notifications are configured correctly! ✅</p>
    </div>
    """
    return _send_email(subject="🤖 CSVG Pipeline — Test Email", html_body=html)
