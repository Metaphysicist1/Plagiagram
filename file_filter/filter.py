import os
from typing import List, Optional

def filter_files(
    root_dir: str,
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,

) -> List[str]:
    """
    Filter files in a directory based on various criteria.
    
    Args:
        root_dir: Root directory to start searching from
        extensions: List of file extensions to include (e.g., ['.py', '.txt'])
        exclude_dirs: List of directory names to exclude
    
    Returns:
        List of filtered file paths
    """
    # Initialize default values
    extensions = extensions or []
    exclude_dirs = exclude_dirs or ['.git', '__pycache__', 'node_modules']

    filtered_files = []
    
    for root, dirs, files in os.walk(root_dir):

        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
    
            if extensions:
                if not any(file.endswith(ext) for ext in extensions):
                    continue
            
            filtered_files.append(file_path)
    
    return filtered_files


if __name__ == "__main__":

    files = filter_files(
        root_dir="./repo_fetching/Repositories_path/",
        extensions=['.py', '.sh','.js'],
        exclude_dirs=['.git', '__pycache__', 'tests']
        )
    
    for file in files:
        print(file)
