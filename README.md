
AI Appointment Booking Agent
Project Overview
This is an AI-powered appointment booking system that uses a conversational agent to help users book, reschedule, cancel, and view appointments. Built with LangGraph, Gemini AI, and integrated with Google Calendar.
Technologies
- Backend: Python, FastAPI
- AI/Agent: LangGraph, Google Gemini
- Calendar Integration: Google Calendar API (service account)
- Frontend: Streamlit
- Deployment: Docker, Caddy (reverse proxy with HTTPS), AWS EC2
- Domain/DNS: sslip.io (for dynamic IPs)
How It Works
![AI Appointment Flow](flow.png)

The agent uses LangGraph to manage conversation state and tool calls. When a user message comes in, the LLM determines if tools are needed (checking availability, booking, etc.). After tool execution, the result loops back to the LLM for response generation.

The AI agent follows a structured flow with built-in validation rules to ensure all bookings comply with business constraints.
Business Rules
Rule	Details
Services	consultation, follow-up, demo
Slot Duration	60 minutes
Max Advance Booking	60 days ahead
Working Days	Monday - Friday
Working Hours	10:00 AM - 6:00 PM
Cancellation/Reschedule Notice	At least 60 minutes before appointment
System Flow
AI Appointment Flow
The agent uses LangGraph to manage conversation state and tool calls. When a user message comes in, the LLM determines if tools are needed (checking availability, booking, etc.). After tool execution, the result loops back to the LLM for response generation.
Setup Instructions
1. Clone the Repository
git clone https://github.com/SoftCr8ors/AI-Appointment-Booking-Agent.git
cd AI-Appointment-Booking-Agent
2. Create a Virtual Environment
For Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
For macOS/Linux:
python -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
4. Add the Gemini API Key
Create a .env file in the project root with the following variables:
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
CALENDAR_ID=your_google_calendar_id
TIMEZONE=Asia/Karachi
API_KEY=your_random_api_key
Also add your Google Calendar service account credentials as secrets/service_account.json in the root directory.
5. Run the Application
Run the Backend (FastAPI):
cd backend
python main.py
Run the Frontend (Streamlit):
cd frontend
streamlit run app.py
Examples
Example 1: Check availability for Monday
User Question:
"i want to book appointment on monday"
Resulting Output:
For Monday, October 5, we have the following free times available:

10:00 AM
12:00 PM
1:00 PM
3:00 PM
5:00 PM

Which service would you like (consultation, follow-up, or demo), and which time would you prefer?
Example 2: Book an appointment
User Question:
"book consultation at 10 AM Monday"
Resulting Output:
I can book a consultation for Monday, October 5, 2026 at 10:00 AM.

Do you confirm?
Example 3: View existing appointments
User Question:
"show my appointments"
Resulting Output:
You have 1 upcoming appointment:

- Consultation on Monday, October 5, 2026 at 10:00 AM
