import asyncio

import markdown
import resend

from .settings import settings


async def send_report_email(report_markdown: str, report_date: str) -> None:
    """Send the daily research report as a formatted HTML email via Resend."""
    if not settings.email_enabled:
        print("  Email disabled, skipping.")
        return

    if not settings.email_to:
        print("  EMAIL_TO not set, skipping.")
        return

    resend.api_key = settings.resend_api_key.get_secret_value()

    html_body = markdown.markdown(report_markdown, extensions=["tables"])
    html = f"""\
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; \
max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6;">
<style>
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background-color: #f5f5f5; }}
a {{ color: #0366d6; }}
</style>
{html_body}
</div>"""

    try:
        await asyncio.to_thread(
            resend.Emails.send,
            {
                "from": settings.email_from,
                "to": [settings.email_to],
                "subject": f"Research Report — {report_date}",
                "html": html,
            },
        )
        print(f"  Email sent to {settings.email_to}")
    except Exception as e:
        print(f"  Email failed (report is saved to disk): {e}")
