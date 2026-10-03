# AI Job Hunt Autopilot

Daily n8n workflow finds AI/ML jobs, a FastAPI + CrewAI service scores them against a resume and drafts outreach, and a top-5 digest goes to Discord.

Work in progress. Full setup and architecture docs land in Phase 7.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r api\requirements.txt
Copy-Item .env.example .env
```
