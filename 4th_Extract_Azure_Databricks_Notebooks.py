import os
import requests
from azure.identity import DefaultAzureCredential
from pathlib import Path
import time
import base64

class DatabricksNotebookDownloader:
    def __init__(self, databricks_instance):
        """
        Initialize Databricks client with Azure AD authentication
        
        Args:
            databricks_instance (str): Your Databricks workspace URL 
                                     (e.g., "https://adb-123456789.azuredatabricks.net")
        """
        self.databricks_instance = databricks_instance.rstrip('/')
        self.credential = DefaultAzureCredential()
        self.token = self._get_access_token()
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
    
    def _get_access_token(self):
        """Get Azure AD access token for Databricks"""
        try:
            # The scope for Databricks is always this specific URL
            token = self.credential.get_token("2ff814a6-3304-4ab8-85cb-cd0e6f879c1d/.default")
            return token.token
        except Exception as e:
            print(f"Error getting Azure AD token: {e}")
            raise
    
    def list_notebooks(self, path="/", recursive=True):
        """
        List all notebooks in the workspace
        
        Args:
            path (str): Starting path in workspace (default: root)
            recursive (bool): Whether to search recursively
            
        Returns:
            list: List of notebook objects with path, language, and object_type
        """
        notebooks = []
        try:
            url = f"{self.databricks_instance}/api/2.0/workspace/list"
            params = {"path": path}
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'objects' in data:
                    for obj in data['objects']:
                        if obj['object_type'] == 'NOTEBOOK':
                            notebooks.append(obj)
                        elif obj['object_type'] == 'DIRECTORY' and recursive:
                            # Recursively search subdirectories
                            sub_notebooks = self.list_notebooks(obj['path'], recursive)
                            notebooks.extend(sub_notebooks)
            elif response.status_code == 404:
                print(f"Path not found: {path}")
            else:
                print(f"Error listing notebooks: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error in list_notebooks: {e}")
            
        return notebooks
    
    def export_notebook(self, notebook_path, format_type="SOURCE"):
        """
        Export a notebook from Databricks
        
        Args:
            notebook_path (str): Path to the notebook in Databricks workspace
            format_type (str): Export format - "SOURCE", "HTML", "JUPYTER", "DBC"
            
        Returns:
            str: Notebook content (base64 decoded if SOURCE format)
        """
        try:
            url = f"{self.databricks_instance}/api/2.0/workspace/export"
            params = {
                "path": notebook_path,
                "format": format_type
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                content = data.get('content', '')
                
                # Decode base64 content for SOURCE format
                if format_type == "SOURCE" and content:
                    try:
                        return base64.b64decode(content).decode('utf-8')
                    except Exception as e:
                        print(f"Error decoding notebook content: {e}")
                        return content
                return content
            else:
                print(f"Error exporting notebook {notebook_path}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error in export_notebook: {e}")
            return None
    
    def download_all_notebooks(self, local_directory, notebook_filter=None, export_format="SOURCE"):
        """
        Download all notebooks from Databricks to local directory
        
        Args:
            local_directory (str): Local directory to save notebooks
            notebook_filter (function): Optional filter function for notebooks
            export_format (str): Export format - "SOURCE", "JUPYTER", "HTML", "DBC"
            
        Returns:
            list: List of downloaded notebook file paths
        """
        print("🔍 Discovering notebooks in Databricks workspace...")
        notebooks = self.list_notebooks()
        
        if notebook_filter:
            notebooks = [nb for nb in notebooks if notebook_filter(nb)]
        
        print(f"📝 Found {len(notebooks)} notebooks to download")
        
        downloaded_files = []
        local_path = Path(local_directory)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension based on export format
        extensions = {
            "SOURCE": ".py",
            "JUPYTER": ".ipynb", 
            "HTML": ".html",
            "DBC": ".dbc"
        }
        file_extension = extensions.get(export_format, ".py")
        
        for i, notebook in enumerate(notebooks):
            try:
                print(f"⬇️  Downloading ({i+1}/{len(notebooks)}): {notebook['path']}")
                
                # Get notebook content
                content = self.export_notebook(notebook['path'], export_format)
                
                if content is not None:
                    # Create local file path, maintaining directory structure
                    relative_path = notebook['path'].lstrip('/')
                    local_file_path = local_path / f"{relative_path}{file_extension}"
                    
                    # Create subdirectories if needed
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write content to file
                    with open(local_file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    downloaded_files.append(str(local_file_path))
                    print(f"✅ Downloaded: {local_file_path}")
                else:
                    print(f"❌ Failed to download: {notebook['path']}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error downloading {notebook['path']}: {e}")
        
        print(f"\n🎉 Successfully downloaded {len(downloaded_files)} notebooks to {local_directory}")
        return downloaded_files


def main():
    """Main function to download notebooks from Databricks"""
    # --- Configuration ---
    DATABRICKS_INSTANCE = ""  # Replace with your workspace URL
    LOCAL_NOTEBOOKS_DIRECTORY = 'C:\\Users\\v-subchandra\\Desktop\\NGPO\\metadata\\5th_ADLS_Notebooks_Output'  # Local folder to save notebooks

    # Optional: Filter notebooks (uncomment and modify as needed)
    def python_notebook_filter(notebook):
        """Only download Python notebooks"""
        return notebook.get('language') == 'PYTHON'
    
    def sql_notebook_filter(notebook):
        """Only download SQL notebooks"""
        return notebook.get('language') == 'SQL'
    
    # --- Download notebooks ---
    try:
        print("🚀 Connecting to Azure Databricks with Azure AD authentication...")
        downloader = DatabricksNotebookDownloader(DATABRICKS_INSTANCE)
        
        print(f"📥 Downloading notebooks to: {LOCAL_NOTEBOOKS_DIRECTORY}")
        
        # Download all notebooks (remove notebook_filter parameter to get all)
        downloaded_files = downloader.download_all_notebooks(
            local_directory=LOCAL_NOTEBOOKS_DIRECTORY,
            # notebook_filter=python_notebook_filter,  # Uncomment to filter
            export_format="SOURCE"  # Options: "SOURCE", "JUPYTER", "HTML", "DBC"
        )
        
        print(f"\n✨ Download complete! {len(downloaded_files)} notebooks saved.")
        
    except Exception as e:
        print(f"❌ Script failed: {e}")


if __name__ == '__main__':
    main()