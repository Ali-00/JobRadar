from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import re
import os
import asyncio
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

app = FastAPI(title="JobRadar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


class SearchRequest(BaseModel):
    job_title: str
    location: str = "Germany"
    date_range: str = "1w"
    date_label: str = "Last Week"


DATE_RANGE_MAP = {
    "24h": "last 24 hours",
    "2d":  "last 2 days",
    "3d":  "last 3 days",
    "1w":  "last week",
    "2w":  "last 2 weeks",
    "1m":  "last month",
}

DATE_RANGE_MAX_DAYS = {
    "24h": 1,
    "2d": 2,
    "3d": 3,
    "1w": 7,
    "2w": 14,
    "1m": 31,
}

PLATFORMS = [
    {
        "source": "LinkedIn",
        "url_substrings": ["linkedin.com/jobs"],
        "search_site": 'site:linkedin.com/jobs',
    },
    {
        "source": "Indeed",
        "url_substrings": ["indeed.com/viewjob", "indeed.com/jobs", "indeed.com/q-"],
        "search_site": 'site:indeed.com/jobs',
    },
    {
        "source": "StepStone",
        "url_substrings": ["stepstone.de"],
        "search_site": 'site:stepstone.de',
    },
    {
        "source": "XING",
        "url_substrings": ["xing.com/jobs"],
        "search_site": 'site:xing.com/jobs',
    },
    {
        "source": "Glassdoor",
        "url_substrings": ["glassdoor.com", "glassdoor.de"],
        "search_site": 'site:glassdoor.com',
    },
    {
        "source": "Monster",
        "url_substrings": ["monster.de", "monster.com"],
        "search_site": 'site:monster.* jobs',
    },
    {
        "source": "Bundesagentur für Arbeit",
        "url_substrings": ["arbeitsagentur.de/jobs", "arbeitsagentur.de/arbeitsplatz"],
        "search_site": 'site:arbeitsagentur.de/jobs',
    },
]


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.post("/api/search")
async def search_jobs(req: SearchRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment.")

    client = AsyncOpenAI(api_key=api_key)

    date_str = DATE_RANGE_MAP.get(req.date_range, req.date_label.lower())
    max_days = DATE_RANGE_MAX_DAYS.get(req.date_range, 7)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini-search-preview")

    def estimate_days_ago(posted: str) -> Optional[int]:
        """
        Parse common relative timestamps like:
        - "2 hours ago"
        - "1 day ago"
        - "3 days ago"
        - "2 weeks ago"
        - "1 month ago"
        """
        if not posted:
            return None
        s = posted.strip().lower()
        if "just now" in s or "today" in s:
            return 0

        m = re.search(r"(\d+)\s*hour", s)
        if m:
            hours = int(m.group(1))
            # Keep it conservative: 23 hours -> 1 day
            return max(0, (hours + 23) // 24)

        m = re.search(r"(\d+)\s*day", s)
        if m:
            return int(m.group(1))

        m = re.search(r"(\d+)\s*week", s)
        if m:
            return int(m.group(1)) * 7

        m = re.search(r"(\d+)\s*month", s)
        if m:
            return int(m.group(1)) * 31

        return None

    job_item_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "company": {"type": "string"},
            "location": {"type": "string"},
            "posted": {"type": "string"},
            "source": {"type": "string"},
            "url": {"type": "string"},
            "type": {"type": "string"},
            "description": {"type": "string"},
            "posted_days_ago": {"type": "integer", "minimum": 0},
        },
        # OpenAI's strict JSON schema mode requires that every key in `properties`
        # also appears in `required`.
        "required": [
            "title",
            "company",
            "location",
            "posted",
            "source",
            "url",
            "type",
            "description",
            "posted_days_ago",
        ],
        "additionalProperties": False,
    }

    result_schema = {
        "type": "object",
        "properties": {
            "jobs": {"type": "array", "items": job_item_schema, "maxItems": 15}
        },
        "required": ["jobs"],
        "additionalProperties": False,
    }

    def url_matches_platform(url: str, url_substrings: List[str]) -> bool:
        if not url:
            return False
        u = url.lower()
        return any(s.lower() in u for s in url_substrings)

    def filter_and_normalize_jobs(
        jobs: List[Dict[str, Any]],
        platform: Dict[str, Any],
        max_days_local: int,
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for job in jobs or []:
            if not isinstance(job, dict):
                continue
            url = job.get("url", "")
            if not url_matches_platform(url, platform["url_substrings"]):
                continue

            posted_days_model = job.get("posted_days_ago")
            if not isinstance(posted_days_model, int):
                continue

            posted = job.get("posted", "")
            posted_days_est = estimate_days_ago(posted)
            # Prefer parsing from the `posted` text. If the estimate exists, treat it as the source of truth,
            # otherwise fall back to the model-provided number.
            days_to_use = posted_days_est if posted_days_est is not None else posted_days_model

            if days_to_use <= max_days_local:
                # Ensure required UI fields exist
                if not isinstance(job.get("description"), str):
                    job["description"] = ""
                if not isinstance(job.get("source"), str):
                    job["source"] = platform["source"]
                job["posted_days_ago"] = days_to_use
                filtered.append(job)
        return filtered

    async def run_platform_agent(platform: Dict[str, Any], cap: int) -> List[Dict[str, Any]]:
        prompt = f"""You are a job search agent for ONLY this platform: {platform['source']}.

Task: Find real, currently open jobs for "{req.job_title}" in "{req.location}" that were posted within {date_str}.

Hard constraints:
- When searching the web, prioritize this query first:
  {platform['search_site']} "{req.job_title}" "{req.location}" "{date_str}"
- Return jobs ONLY from URLs that match at least one of these domain substrings:
  {platform['url_substrings']}
- Every job URL must be a direct application URL to the job posting page.
- Every job must be for the correct timeframe.

Return ONLY JSON in the following schema:
{{
  "jobs": [
    {{
      "title": string,
      "company": string,
      "location": string,
      "posted": string,
      "source": string,   // must be exactly "{platform['source']}"
      "url": string,
      "type": string,
      "description": string, // one sentence summary; use "" if unknown
      "posted_days_ago": integer // must be computed from "posted". For items outside timeframe, still compute the real value.
    }}
  ]
}}

If you cannot find jobs, return an empty list [] in "jobs".
Maximum jobs: {cap}
"""
        response = await client.chat.completions.create(
            model=model,
            max_tokens=1200,
            messages=[
                {
                    "role": "system",
                    "content": "Return only JSON that matches the provided schema.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "job_search_result",
                    "schema": result_schema,
                    "strict": True,
                },
            },
        )

        choice = response.choices[0].message
        content = choice.content or ""
        data = json.loads(content)
        return data.get("jobs", [])

    async def run_parallel_platforms(cap: int) -> Dict[str, List[Dict[str, Any]]]:
        concurrency = int(os.getenv("OPENAI_PLATFORM_CONCURRENCY", "4"))
        sem = asyncio.Semaphore(concurrency)

        async def wrapped(p: Dict[str, Any]) -> List[Dict[str, Any]]:
            async with sem:
                return await run_platform_agent(p, cap)

        tasks = [wrapped(p) for p in PLATFORMS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: Dict[str, List[Dict[str, Any]]] = {}
        for p, r in zip(PLATFORMS, results):
            if isinstance(r, Exception):
                out[p["source"]] = []
            else:
                out[p["source"]] = r
        return out

    try:
        # 1) First pass: parallel by platform, then deterministic filtering by `posted_days_ago`.
        cap1 = int(os.getenv("PLATFORM_JOB_CAP", "3"))
        platform_results = await run_parallel_platforms(cap1)

        filtered_by_source: Dict[str, List[Dict[str, Any]]] = {}
        all_jobs: List[Dict[str, Any]] = []
        for p in PLATFORMS:
            jobs = filter_and_normalize_jobs(platform_results.get(p["source"], []), p, max_days)
            filtered_by_source[p["source"]] = jobs
            all_jobs.extend(jobs)

        # 2) Optional second pass when distribution is weak.
        # Even if total jobs are already OK, we prefer not to return 100% LinkedIn.
        counts_by_source = {p["source"]: len(filtered_by_source.get(p["source"], [])) for p in PLATFORMS}
        non_linkedin_present = sum(1 for s, c in counts_by_source.items() if s != "LinkedIn" and c > 0)
        total_jobs = len(all_jobs)

        min_jobs_total = int(os.getenv("MIN_JOBS_TOTAL", "10"))
        second_pass_trigger = total_jobs < min_jobs_total or non_linkedin_present == 0
        if second_pass_trigger:
            missing_sources = [
                p
                for p in PLATFORMS
                if p["source"] != "LinkedIn" and len(filtered_by_source.get(p["source"], [])) == 0
            ]
            if missing_sources:
                os_cap2 = int(os.getenv("PLATFORM_JOB_CAP_SECOND", "5"))
                # Run second pass only for the first N missing non-LinkedIn platforms.
                max_missing = int(os.getenv("SECOND_PASS_MAX_PLATFORMS", "3"))
                missing_sources = missing_sources[:max_missing]
                concurrency = int(os.getenv("OPENAI_PLATFORM_CONCURRENCY", "4"))
                sem = asyncio.Semaphore(concurrency)

                async def wrapped_missing(p: Dict[str, Any]) -> List[Dict[str, Any]]:
                    async with sem:
                        return await run_platform_agent(p, os_cap2)

                tasks = [wrapped_missing(p) for p in missing_sources]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for p, r in zip(missing_sources, results):
                    if isinstance(r, Exception):
                        continue
                    jobs2 = filter_and_normalize_jobs(r, p, max_days)
                    filtered_by_source[p["source"]] = jobs2
                    all_jobs.extend(jobs2)

        if os.getenv("DEBUG_PLATFORM_COUNTS") == "1":
            counts = {s: len(filtered_by_source.get(s, [])) for s in filtered_by_source}
            print(f"[DEBUG] platform job counts (post-filter): {counts}")

        # Dedupe by URL + rerank by freshness.
        dedup: Dict[str, Dict[str, Any]] = {}
        for j in all_jobs:
            url = j.get("url")
            if not url:
                continue
            dedup[url] = j

        jobs_sorted = sorted(dedup.values(), key=lambda x: x.get("posted_days_ago", 10**9))
        jobs_final = jobs_sorted[:15]

        # Strip internal fields before returning (frontend ignores them, but this keeps response clean).
        for j in jobs_final:
            j.pop("posted_days_ago", None)

        return {"jobs": jobs_final, "total": len(jobs_final)}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse job results: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
