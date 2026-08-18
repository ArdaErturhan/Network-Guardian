"""
alerting.py
-----------
Sends an email alert via Gmail SMTP (smtplib) when an anomaly is detected.

Credentials are read from config/config.yaml -> alerting section.
Use a Gmail *App Password*, NOT your real account password.
"""

import smtplib
import time
from email.mime.text import MIMEText


class EmailAlerter:
    def __init__(self, sender, app_password, recipient,
                 smtp_host="smtp.gmail.com", smtp_port=587,
                 cooldown_seconds=60):
        self.sender = sender
        self.app_password = app_password
        self.recipient = recipient
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.cooldown_seconds = cooldown_seconds
        self._last_sent = 0.0

    def send(self, score, threshold, feature_summary=""):
        # rate-limit to avoid alert storms
        now = time.time()
        if now - self._last_sent < self.cooldown_seconds:
            return False

        body = (
            "NetworkGuardian detected anomalous network traffic.\n\n"
            f"Anomaly score : {score:.6f}\n"
            f"Threshold      : {threshold:.6f}\n"
            f"Time           : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"{feature_summary}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = "[NetworkGuardian] Anomaly Detected"
        msg["From"] = self.sender
        msg["To"] = self.recipient

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.app_password)
                server.sendmail(self.sender, [self.recipient], msg.as_string())
            self._last_sent = now
            return True
        except Exception as exc:  # keep the monitor loop alive on SMTP failure
            print(f"[alerting] email send failed: {exc}")
            return False
