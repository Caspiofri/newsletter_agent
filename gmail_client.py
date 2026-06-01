import os.path
import base64
from email.message import EmailMessage

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class GmailClient():

    def __init__(self):
        self.service = None
        self.authenticate()

    def authenticate(self):
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0, prompt='select_account')
                with open("token.json", "w") as token:
                    token.write(creds.to_json())
        try:
            self.service = build("gmail", "v1", credentials=creds)
        except HttpError as error:
            raise RuntimeError(f"Gmail authentication failed: {error}") from error

    def gmail_read_messages(self, label: str = "INBOX", days_back: int = 2):
        from datetime import date, timedelta
        since = date.today() - timedelta(days=days_back)
        query = f"after:{since.strftime('%Y/%m/%d')}"

        results = self.service.users().messages().list(
            userId="me", labelIds=[label], q=query
        ).execute()
        raw_messages = results.get("messages", [])
        messages = []
        for raw_message in raw_messages:
            msg = self.service.users().messages().get(userId="me", id=raw_message["id"]).execute()
            messages.append(msg)
        return messages

    def gmail_send_message(self, to, sender, subject, html_content):
        message = EmailMessage()
        message.set_content(html_content, subtype='html')
        message["Subject"] = subject
        message["To"] = to
        message["From"] = sender

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded_message}
        send_message = (
            self.service.users()
            .messages()
            .send(userId="me", body=create_message)
            .execute()
        )
        return send_message
