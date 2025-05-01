import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Dict, Tuple # Added List, Dict, Tuple for clarity

import requests
from requests import Session

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))


def download_dataset_file(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    filename: str,
    directory: str,
    overwrite: bool,
) -> Tuple[bool, str]: # Added Tuple type hint
    """Downloads a single dataset file after retrieving its temporary URL."""
    # if a file from this dataset already exists, skip downloading it.
    file_path = Path(directory, filename).resolve()
    if not overwrite and file_path.exists():
        logger.info(f"Dataset file '{filename}' already exists. Skipping.")
        return True, filename

    endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files/{filename}/url"
    try:
        get_file_response = session.get(endpoint)
        get_file_response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        logger.warning(f"Unable to get download URL for file: {filename}. Error: {e}")
        logger.warning(f"Response content: {get_file_response.content if 'get_file_response' in locals() else 'No response'}")
        return False, filename


    # retrieve download URL for dataset file
    try:
        download_url = get_file_response.json().get("temporaryDownloadUrl")
        if not download_url:
             logger.warning(f"No 'temporaryDownloadUrl' found in response for file: {filename}")
             logger.warning(f"Response JSON: {get_file_response.json()}")
             return False, filename
    except Exception as e:
        logger.exception(f"Error parsing JSON response for file {filename}: {e}")
        logger.warning(f"Response content: {get_file_response.content}")
        return False, filename


    # use download URL to GET dataset file. We don't need to set the 'Authorization' header,
    # The presigned download URL already has permissions to GET the file contents
    return download_file_from_temporary_download_url(download_url, directory, filename)


def download_file_from_temporary_download_url(
    download_url: str, directory: str, filename: str
) -> Tuple[bool, str]: # Added Tuple type hint
    """Downloads a file from the provided temporary URL."""
    file_path = Path(directory, filename)
    try:
        # Use a new requests session for the download URL as it might be a different domain
        # and doesn't need the auth header. Stream=True is important for large files.
        with requests.get(download_url, stream=True, timeout=300) as r: # Added timeout
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192 * 16): # Increased chunk size potentially
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        logger.exception(f"Unable to download file '{filename}' using download URL. Error: {e}")
        # Clean up partially downloaded file on error
        if file_path.exists():
             try:
                 file_path.unlink()
                 logger.info(f"Removed partially downloaded file: {filename}")
             except OSError as unlink_error:
                 logger.error(f"Could not remove partial file {filename}: {unlink_error}")
        return False, filename
    except Exception as e:
        logger.exception(f"An unexpected error occurred downloading file '{filename}': {e}")
        # Clean up partially downloaded file on error
        if file_path.exists():
             try:
                 file_path.unlink()
                 logger.info(f"Removed partially downloaded file: {filename}")
             except OSError as unlink_error:
                 logger.error(f"Could not remove partial file {filename}: {unlink_error}")
        return False, filename


    logger.info(f"Successfully downloaded dataset file '{filename}' to '{file_path}'")
    return True, filename


def list_dataset_files(
    session: Session,
    base_url: str,
    dataset_name: str,
    dataset_version: str,
    params: Dict[str, str], # Changed dict to Dict
) -> Tuple[List[str], Dict[str, Any]]: # Added Tuple, List, Dict type hints
    """Lists dataset files from the API, handling pagination."""
    logger.info(f"Retrieving dataset files with query params: {params}")

    list_files_endpoint = f"{base_url}/datasets/{dataset_name}/versions/{dataset_version}/files"

    try:
        list_files_response = session.get(list_files_endpoint, params=params)
        list_files_response.raise_for_status() # Check for HTTP errors
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed when listing files: {e}")
        # Consider how to handle this - retry? Raise exception?
        raise Exception(f"Unable to list dataset files. Error: {e}") from e

    try:
        list_files_response_json = list_files_response.json()
        dataset_files = list_files_response_json.get("files", []) # Default to empty list
        dataset_filenames = [f.get("filename") for f in dataset_files if f.get("filename")]
        return dataset_filenames, list_files_response_json
    except Exception as e: # Catch broader exceptions during JSON parsing or processing
        logger.exception(f"Error processing API response for listing files: {e}")
        logger.error(f"Response text: {list_files_response.text}")
        raise Exception("Error processing file list response.") from e


def get_max_worker_count(filesizes: List[int]) -> int: # Added List type hint
    """Determines the number of download threads based on average file size."""
    if not filesizes: # Handle case with no files
        return 1
    size_for_threading = 10_000_000  # 10 MB threshold
    average = sum(filesizes) / len(filesizes)
    # Use more threads for smaller average file sizes, fewer for very large files
    if average > size_for_threading * 10: # Significantly large files
         threads = 2 # Maybe slightly more than 1 is okay
    elif average > size_for_threading: # Large files
         threads = 5
    else: # Smaller files
         threads = 10 # Or even os.cpu_count() * 2 or similar
    logger.info(f"Average file size: {average/1_000_000:.2f} MB. Using {threads} download workers.")
    return threads


async def main():
    # --- Configuration ---
    # WARNING: Hardcoding API keys is a security risk. Consider environment variables.
    api_key = "eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6ImRmYmRkMGQ2NzFjZTRmMDU4ZDBlMGEyZGUzNDE2OTcyIiwiaCI6Im11cm11cjEyOCJ9"
    dataset_name = "EV24"
    dataset_version = "2"
    base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
    download_directory = "./dataset-download-EV24-v2" # Make directory name more specific
    # When set to True, if a file with the same name exists the output is written over the file.
    # To prevent unnecessary bandwidth usage, leave it set to False.
    overwrite = False
    max_keys_per_page = 500 # Max files listed per API call
    # --- End Configuration ---


    # Create the download directory if it doesn't exist
    download_path = Path(download_directory)
    try:
        download_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured download directory exists: {download_path.resolve()}")
    except OSError as e:
         logger.error(f"Could not create download directory '{download_path}'. Error: {e}")
         return # Exit if directory cannot be created


    # Set up the shared requests session with the API key
    session = requests.Session()
    session.headers.update({"Authorization": api_key})

    all_filenames = []
    all_file_sizes: List[int] = [] # Explicitly type hint
    next_page_token = None
    page_num = 1
    # --- Define sleep duration ---
    LISTING_DELAY_SECONDS = 1.0 # Start with 1 second, adjust if needed

    # Use the API to get a list of all dataset filenames, handling pagination
    logger.info(f"Starting retrieval of file list for {dataset_name} v{dataset_version}...")
    while True:
        logger.info(f"Fetching page {page_num} of file list...")
        params = {"maxKeys": str(max_keys_per_page)}
        if next_page_token:
            params["nextPageToken"] = next_page_token

        try:
             current_page_filenames, response_json = list_dataset_files(
                session,
                base_url,
                dataset_name,
                dataset_version,
                params,
             )
             if not current_page_filenames:
                 logger.warning(f"Received empty file list on page {page_num}. Response: {response_json}")
                 # Decide if this is an error or just the end

             # Store filenames and sizes
             all_filenames.extend(current_page_filenames)
             all_file_sizes.extend(file.get("size", 0) for file in response_json.get("files", []) if file.get("filename") in current_page_filenames)

             # Check if this is the last page
             next_page_token = response_json.get("nextPageToken")
             logger.info(f"Retrieved {len(current_page_filenames)} filenames from page {page_num}.")

             if not next_page_token:
                 logger.info("Retrieved names of all dataset files.")
                 break # Exit loop if no more pages
             else:
                 page_num += 1
                 # ********* ADDED DELAY HERE *********
                 logger.info(f"Waiting for {LISTING_DELAY_SECONDS} seconds before fetching next page...")
                 await asyncio.sleep(LISTING_DELAY_SECONDS)
                 # *************************************

        except Exception as e:
             logger.error(f"Failed to retrieve page {page_num} of file list: {e}. Stopping.")
             session.close() # Close the session on failure
             return

    logger.info(f"Total number of files to potentially download: {len(all_filenames)}")
    if not all_filenames:
        logger.info("No files found for this dataset version. Exiting.")
        session.close()
        return

    # Determine worker count and set up thread pool
    worker_count = get_max_worker_count(all_file_sizes)
    loop = asyncio.get_running_loop() # Use get_running_loop in async context
    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures = []

    logger.info(f"Starting download process with {worker_count} workers...")
    # Create tasks that download the dataset files concurrently
    for dataset_filename in all_filenames:
        # Create future for dataset file download
        future = loop.run_in_executor(
            executor,
            download_dataset_file, # Function to run
            # Arguments for download_dataset_file:
            session,
            base_url,
            dataset_name,
            dataset_version,
            dataset_filename,
            str(download_path.resolve()), # Pass resolved path as string
            overwrite,
        )
        futures.append(future)

    # Wait for all download tasks to complete and gather the results
    # Use return_exceptions=True to get results even if some downloads failed
    future_results = await asyncio.gather(*futures, return_exceptions=True)
    logger.info(f"Finished '{dataset_name}' v{dataset_version} dataset download processing.")

    # Process results and report failures
    successful_downloads = 0
    failed_downloads = []
    exceptions_caught = []

    for i, result in enumerate(future_results):
        original_filename = all_filenames[i] # Get corresponding filename
        if isinstance(result, Exception):
            logger.warning(f"Download task for '{original_filename}' raised an exception: {result}")
            exceptions_caught.append((original_filename, result))
        elif isinstance(result, tuple) and len(result) == 2:
             success, filename = result
             if success:
                 successful_downloads += 1
             else:
                 logger.warning(f"Download task reported failure for: {filename}")
                 failed_downloads.append(filename)
        else:
             # This case should ideally not happen if download_dataset_file always returns tuple or raises Exception
             logger.error(f"Unexpected result type for file '{original_filename}': {result}")
             exceptions_caught.append((original_filename, "Unexpected return value"))


    logger.info(f"Summary: {successful_downloads} files downloaded successfully.")
    if failed_downloads:
        logger.warning(f"Failed to download {len(failed_downloads)} files (API/Download level):")
        # Log first 10 failed filenames for brevity
        logger.warning(failed_downloads[:10])
        if len(failed_downloads) > 10:
            logger.warning(f"... and {len(failed_downloads) - 10} more.")
    if exceptions_caught:
        logger.error(f"{len(exceptions_caught)} download tasks raised exceptions:")
        # Log first 10 exceptions for brevity
        for fname, exc in exceptions_caught[:10]:
             logger.error(f" - {fname}: {exc}")
        if len(exceptions_caught) > 10:
            logger.error(f"... and {len(exceptions_caught) - 10} more.")


    # Clean up the session and executor
    session.close()
    executor.shutdown(wait=False) # Don't wait for threads to finish here, asyncio.gather already did
    logger.info("Session closed and thread pool shutdown initiated.")


if __name__ == "__main__":
    # To run the async main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Download process interrupted by user.")
    except Exception as e:
        logger.exception(f"An unhandled exception occurred in the main execution: {e}")