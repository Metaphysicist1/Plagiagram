#!/bin/bash

# Set your GitHub token (optional but recommended to avoid rate limits)
GITHUB_TOKEN="$GIT_TOKEN"

# Directory to save repositories
DOWNLOAD_DIR="github_repos"

# Create download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# List of repositories (format: "owner/repo")
REPOS=(
  "facebook/react"
  "tensorflow/tensorflow"
  "microsoft/vscode"
)

# Function to clone a repository
clone_repo() {
  local repo=$1
  echo "Cloning $repo..."
  
  if [ -n "$GITHUB_TOKEN" ]; then
    # Clone with token for private repos or higher rate limits
    git clone "https://$GITHUB_TOKEN@github.com/$repo.git"
  else
    # Clone without token
    git clone "https://github.com/$repo.git"
  fi
}

# Function to download repository as ZIP (alternative to cloning)
download_zip() {
  local repo=$1
  local owner=$(echo $repo | cut -d '/' -f1)
  local repo_name=$(echo $repo | cut -d '/' -f2)
  
  echo "Downloading $repo as ZIP..."
  
  if [ -n "$GITHUB_TOKEN" ]; then
    curl -s -L -H "Authorization: token $GITHUB_TOKEN" \
         -H "Accept: application/vnd.github.v3+json" \
         "https://api.github.com/repos/$repo/zipball" \
         -o "$repo_name.zip"
  else
    curl -s -L "https://github.com/$repo/archive/refs/heads/main.zip" -o "$repo_name.zip"
  fi
  
  # Uncomment to automatically extract
  # mkdir -p "$repo_name"
  # unzip -q "$repo_name.zip" -d "$repo_name"
  # rm "$repo_name.zip"
}

# Main execution
echo "Starting download of ${#REPOS[@]} repositories..."

for repo in "${REPOS[@]}"; do
  # Choose one of these methods:
  clone_repo "$repo"
  # OR
  # download_zip "$repo"
done

echo "All repositories downloaded to $DOWNLOAD_DIR"