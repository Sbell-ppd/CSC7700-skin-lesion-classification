import requests
import zipfile
from pathlib import Path
from tqdm import tqdm


# Setup path to data folder
data_path = Path("data/")

# Download and extract the dataset
def download_train_dataset(url: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    
    dataset_folder = dest / "ISIC2018_Task3_Training_Input"
    if dataset_folder.exists() and dataset_folder.is_dir():
        print("Dataset already exists. Skipping download.")
        return
    
    try:
        # Stream the download to handle large files
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            zip_path = dest / "dataset.zip"
            
            with open(zip_path, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            
        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest)
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error extracting the zip file: {e}")
    finally:
        if zip_path.exists():
            zip_path.unlink()


download_train_dataset("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip", data_path)

# Verify the dataset folder exists and print its contents
dataset_folder = data_path / "ISIC2018_Task3_Training_Input"
if dataset_folder.exists() and dataset_folder.is_dir():
    print(f"Dataset folder '{dataset_folder}' created with contents ({len(list(dataset_folder.iterdir()))} items):")
else:
    print(f"Dataset folder '{dataset_folder}' does not exist.")
