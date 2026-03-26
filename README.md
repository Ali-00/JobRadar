# JobRadar 🔍

AI-powered job search across LinkedIn, Indeed, StepStone, XING, Glassdoor and more.

## Project Structure

```
jobradar/
├── main.py            # FastAPI backend
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
└── static/
    └── index.html     # Frontend UI
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
```bash
cp .env.example .env
# Edit .env and paste your API key
```

Or export it directly:
```bash
export OPENAI_API_KEY=your_api_key_here
```

Get your API key at: https://platform.openai.com/

### 3. Start the server
```bash
uvicorn main:app --reload
```

### 4. Open in browser
Visit: http://localhost:8000

## API Endpoints

| Method | Endpoint      | Description              |
|--------|---------------|--------------------------|
| GET    | `/`           | Serves the frontend UI   |
| POST   | `/api/search` | Search for jobs (AI)     |
| GET    | `/health`     | Health check             |

### POST /api/search

**Request body:**
```json
{
  "job_title": "Data Scientist",
  "location": "Berlin, Germany",
  "date_range": "1w",
  "date_label": "Last Week"
}
```

**date_range values:** `24h`, `2d`, `3d`, `1w`, `2w`, `1m`

**Response:**
```json
{
  "jobs": [
    {
      "title": "Senior Data Scientist",
      "company": "Some Company GmbH",
      "location": "Berlin, Germany",
      "posted": "2 days ago",
      "source": "LinkedIn",
      "url": "https://linkedin.com/jobs/...",
      "type": "Full-time",
      "description": "Short role summary"
    }
  ],
  "total": 12
}
```
