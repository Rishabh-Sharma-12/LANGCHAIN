from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="Llama3-8b-8192",
    temperature=0.3
)

class EmailInput(BaseModel):
    receiver_email: str = Field(..., description="Recipient's Gmail address")
    subject: str = Field(..., description="Subject of the mail")
    body: str = Field(..., description="Main content of the email")

def send_gmail(receiver_email: str, subject: str, body: str) -> str:
    sender_email = os.getenv("GMAIL_SENDER")
    password = os.getenv("GMAIL_APP_PASSWORD")

    if not sender_email or not password:
        return "Gmail credentials are not set in the .env file"

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return f"✅ Email sent to {receiver_email} with subject: '{subject}'"
    except Exception as e:
        return f"❌ FAILED: {str(e)}"

gmail_tool = StructuredTool.from_function(
    func=send_gmail,
    description="Send a Gmail message to someone. Requires subject, body, and recipient email.",
    args_schema=EmailInput
)

agent = initialize_agent(
    tools=[gmail_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# result = agent.invoke("""
# Send an email to XXXXXXXX.com with subject 'LangChain Test' and body 'This is a test sent via LangChain Gmail tool.'
# """)
# print(result)

result = gmail_tool.invoke({
    "receiver_email": "---mail-----",
    "subject": "LangChain Test",
    "body": "Hello sir rimsaab this side."
})
print(result)
