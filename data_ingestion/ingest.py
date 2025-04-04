import os
import json
import shutil
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REPOS_CONFIG_PATH = "data_ingestion/repositories.json"
REPOS_STORAGE_PATH = "/data/repositories"
GIT_EXECUTABLE = "/usr/bin/git"  # Default path

def load_repository_config():
    """Load repository configuration from JSON file."""
    with open(REPOS_CONFIG_PATH, 'r') as f:
        return json.load(f)

def clone_repository(repo_url, repo_name, target_dir):
    """Clone a GitHub repository to the target directory."""
    repo_path = os.path.join(target_dir, repo_name)
    
    # Remove existing repository if it exists
    if os.path.exists(repo_path):
        logger.info(f"Removing existing repository: {repo_path}")
        shutil.rmtree(repo_path)
    
    logger.info(f"Cloning repository: {repo_url} to {repo_path}")
    
    try:
        subprocess.run(
            [GIT_EXECUTABLE, "clone", repo_url, repo_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return repo_path
    except Exception as e:
        logger.error(f"Git clone failed: {str(e)}")
        raise

def main():
    """Main function to download repositories."""
    global GIT_EXECUTABLE  # This is the correct place for global
    
    # Create storage directory if it doesn't exist
    os.makedirs(REPOS_STORAGE_PATH, exist_ok=True)
    
    # Try to find git
    try:
        git_path = subprocess.check_output(["which", "git"], text=True).strip()
        logger.info(f"Found git at: {git_path}")
        GIT_EXECUTABLE = git_path
    except Exception:
        logger.warning(f"Using default git path: {GIT_EXECUTABLE}")
    
    # Load repository configuration
    repos_config = load_repository_config()
    
    # Clone each repository
    for repo in repos_config:
        try:
            repo_path = clone_repository(
                repo["url"],
                repo["name"],
                REPOS_STORAGE_PATH
            )
            logger.info(f"Successfully cloned {repo['name']} to {repo_path}")
        except Exception as e:
            logger.error(f"Failed to clone {repo['name']}: {str(e)}")

if __name__ == "__main__":
    main() 