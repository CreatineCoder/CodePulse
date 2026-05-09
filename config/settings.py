import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DB_PATH = os.getenv("DB_PATH", "database/codepulse.db")
CLONE_DIR = os.getenv("CLONE_DIR", "data/repos")
