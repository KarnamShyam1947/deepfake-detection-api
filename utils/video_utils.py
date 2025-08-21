import requests

import os

def download_video(url: str, output_path: str = None, chunk_size: int = 8192) -> str:
    print("Downloading video.....................................")
    filename = output_path if output_path else "temp_video.mp4"
   

    with requests.get(url, stream=True) as response:
        response.raise_for_status() 
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  
                    file.write(chunk)

    return filename

def delete_file(file_path: str) -> bool:
    print("Deleting video.....................................")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        
        except Exception as e:
            print(f"Error deleting file '{file_path}': {e}")
            return False
    else:
        print(f"File '{file_path}' does not exist.")
        return False

