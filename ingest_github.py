"""
Fetch README files from a GitHub user's public repositories and save them
as .txt files, feeds them to a gemini model for distillation and then
stores the distilled descriptions in data/ for ingestion by ingest.py.
"""

import os
import time
import sys
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
from openai import OpenAI
import requests
from dotenv import load_dotenv

OWNER = os.environ.get("GITHUB_OWNER", "ajbax-cmd")
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "projects"
GITHUB_API = "https://api.github.com"

script_dir = Path(__file__).resolve().parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

CLIENT = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)


#print(response.model_dump_json(
    #exclude_none=True, indent=4))

def get_headers(*, raw: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_repos(owner: str) -> list[dict]:
    """List public repositories for a GitHub user, following pagination."""
    repos: list[dict] = []
    url = f"{GITHUB_API}/users/{owner}/repos"
    params = {"per_page": 100, "sort": "updated", "type": "owner"}

    while url:
        response = requests.get(
            url,
            headers=get_headers(),
            params=params,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list repos for {owner}: "
                f"{response.status_code} - {response.text}"
            )

        repos.extend(response.json())
        url = response.links.get("next", {}).get("url")
        params = None  # pagination URL already includes query params

    return repos


def fetch_readme(owner: str, repo_name: str) -> str | None:
    """Return raw README text, or None if the repo has no README."""
    url = f"{GITHUB_API}/repos/{owner}/{repo_name}/readme"
    response = requests.get(url, headers=get_headers(raw=True), timeout=30)

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch README for {repo_name}: "
            f"{response.status_code} - {response.text}"
        )

    text = response.text.strip()
    return text or None

def generate_GROK_descriptions(repo:dict, readme_text:str) -> str:
    prompt = f"""You are an expert technical writer helping craft strong resume bullet points and project descriptions.

    Repository: {repo['name']}
    Description: {repo.get('description', 'No description provided')}
    Stars: {repo.get('stargazers_count', 0)}
    Primary Language: {repo.get('language', 'Unknown')}
    Topics: {', '.join(repo.get('topics', []))}

    README Content:
    {readme_text}

    Task: Create a concise, achievement-oriented project summary optimized for resumes.

    Output format (strictly follow this structure):

    **Project Title:** {repo['name']} - Short one-line description

    **Overview:** 2-3 sentence summary of what the project does and its goal.

    **Key Achievements / Contributions:**
    - Bullet point 1 (quantified if possible)
    - Bullet point 2
    - Bullet point 3

    **Technologies & Skills:**
    - Tech1, Tech2, Tech3, etc.

    Make the language professional, impactful, and suitable for a senior-level resume. Focus on impact, technologies, and problem-solving."""

    response = CLIENT.chat.completions.create(
            model="grok-4.3",           # or grok-4
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=1200
        )

    return response.choices[0].message.content


def main() -> int:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    repos = fetch_repos(OWNER)
    print(f"Found {len(repos)} repositories for {OWNER}")

    saved = 0
    skipped = 0

    for repo in repos:
        repo_name = repo["name"]
        stars = repo.get("stargazers_count", 0)
        print(f"- {repo_name} (stars: {stars})")

        try:
            readme_text = fetch_readme(OWNER, repo_name)
        except RuntimeError as exc:
            print(f"  ! {exc}")
            skipped += 1
            continue

        if readme_text is None:
            print("  - no README found, skipping")
            skipped += 1
            continue
        # have gemin model synthesize descriptions
        gemini_distilled_text = generate_GROK_descriptions(repo=repo, readme_text=readme_text)
        file_path = OUTPUT_PATH / f"readme_{repo_name}.txt"
        file_path.write_text(gemini_distilled_text, encoding="utf-8")
        print(f"  - saved {file_path}")
        saved += 1

    print(f"Done. Saved {saved} README file(s), skipped {skipped}.")
    print(f"Run ingest.py to embed files from {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())