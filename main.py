import os
import sys
import subprocess
import json
from datetime import datetime
import sqlite3
import base64
import asyncio
import httpx
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from email_validator import validate_email, EmailNotValidError
from fastapi import HTTPException
from dateutil import parser


# Constants for API calls
COMPLETIONS_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
EMBEDDINGS_URL  = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
AIPROXY_TOKEN   = os.getenv("AIPROXY_TOKEN")


def is_valid_email_address(email: str) -> bool:
    """
    Checks if the given email address is valid.
    
    Args:
        email (str): The email address to validate.

    Returns:
        bool: True if the email address is valid, False otherwise
    """
    try:
        valid = validate_email(email)
        return True
    except EmailNotValidError as e:
        return False


def convert_date_format(date_str: str) -> str:
    """
    Converts any given date string to ISO format (YYYY-MM-DD).

    This function attempts to parse the given date string using 
    dateutil's robust parser. It uses fuzzy parsing to handle extra content 
    (such as time information) and returns only the date part formatted in 'YYYY-MM-DD'.

    Args:
        date_str (str): A date string in any common format.
        
    Returns:
        str: The date in ISO format (YYYY-MM-DD).
        
    Raises:
        HTTPException: If the date string cannot be parsed.
    """
    try:
        # Parse the date string; fuzzy=True allows ignoring extra text.
        dt = parser.parse(date_str, fuzzy=True)
        # Return formatted date.
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse the date string '{date_str}': {e}")


# ---------------------------------------------------------------------------
# Task A1: Install "uv" (if required) and run the data generator script
# ---------------------------------------------------------------------------
async def task_a1(user_email: str, script_url: str) -> str:
    """
    Asynchronously performs a series of tasks including email verification, package installation, 
    and script execution using the 'uv' package.
    Args:
        user_email (str): The email address of the user to be verified and used in the script execution.
        script_url (str): The URL of the data generator script to be executed.
    Returns:
        str: A message indicating the successful execution of the data generator script.
    Raises:
        HTTPException: If the user email is not provided or is invalid, if the 'uv' package installation fails,
                       or if the script execution fails or times out.
    """

    print("Verifying email address")
    if not user_email:
        raise HTTPException(status_code=400, detail="User email must be provided.")
    if not is_valid_email_address(user_email):
        raise HTTPException(status_code=400, detail="Invalid email address.")
    print("Email address verified")

    print("Checking for 'uv' package")
    try:
        # Check if 'uv' can be imported; if not, install it.
        try:
            import uv  # Attempt to import the module.
        except ImportError:
            print("Installing 'uv' package")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to install 'uv' package.") from e
    print("'uv' package is installed")

    # Execute the script using 'uv run' with user_email as argument and --root './data'.
    try:
        print("Executing data generator script")
        process = subprocess.run(["uv", "run", script_url, user_email, "--root", "./data"],
                                 capture_output=True, text=True, timeout=60)
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Script execution failed: {process.stderr}")
    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail="Script execution timed out.") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to execute the data generator script.") from e
    print("Data generator script executed successfully")
    return "Data generator script executed successfully"
    

# ---------------------------------------------------------------------------
# Task A2: Format the contents of ```target_file``` using prettier@3.4.2
# ---------------------------------------------------------------------------
async def task_a2(target_file: str) -> str:
    """
    Task A2.
    
    Formats the specified file in-place using prettier version 3.4.2.
    
    Args:
        target_file (str): The path to the file to be formatted.
    
    Returns:
        str: A success message indicating the file was formatted.
    
    Raises:
        HTTPException: If the specified file is missing or if prettier fails or times out.
    """
    if not os.path.isfile(target_file):
        raise HTTPException(status_code=404, detail=f"File {target_file} not found.")
    
    try:
        print("Executing prettier formatting")
        # Execute prettier using npx with the specific version.
        try:
            process = subprocess.run(
                ["npx", "prettier@3.4.2", "--write", target_file],
                capture_output=True, text=True, timeout=30
            )
            if process.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Prettier formatting failed: {process.stderr}")
        except FileNotFoundError:
            print("Installing prettier package")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "prettier"])
            process = subprocess.run(
                ["npx", "prettier@3.4.2", "--write", target_file],
                capture_output=True, text=True, timeout=30
            )
            if process.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Prettier formatting failed: {process.stderr}")
    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail="Prettier formatting timed out.") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to execute prettier formatting.") from e
    print("Prettier formatting completed successfully")
    return f"Formatted {target_file} successfully."


# ---------------------------------------------------------------------------
# Task A3: Count the number of ```Days``` in ```source_file``` and write to ```target_file```
# ---------------------------------------------------------------------------
async def task_a3(source_file: str, target_file: str, day: str) -> str:
    """
    Task A3.
    
    Reads the specified source file (which contains a date per line in any common format),
    counts how many of those dates fall on the specified day of the week, and writes the count 
    (as a number) to the specified target file.
    
    Args:
        source_file (str): The path to the input file containing dates.
        target_file (str): The path to the output file where the count will be written.
        day (str): The day of the week to count (e.g., 'Monday', 'Tuesday', etc.).
    
    Returns:
        str: A message indicating how many of the specified day were counted and the output file path.
    
    Raises:
        HTTPException: If the source file does not exist, if the day is invalid, or for file read/write issues.
    """
    if not os.path.isfile(source_file):
        raise HTTPException(status_code=404, detail=f"{source_file} not found.")
    
    try:
        day_index = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"].index(day.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid day of the week: {day}.")
    
    day_count = 0
    try:
        print(f"Counting {day}s in {source_file}")
        with open(source_file, "r") as f:
            for line in f:
                date_str = line.strip()
                if not date_str:
                    continue
                date_str = convert_date_format(date_str)
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if date.weekday() == day_index:
                    day_count += 1
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing dates file.") from e
    
    try:
        print(f"Writing day count to {target_file}")
        with open(target_file, "w") as f:
            f.write(str(day_count))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error writing day count to file.") from e
    
    print(f"Counted {day_count} {day}s and wrote to {target_file}.")
    return f"Counted {day_count} {day}s and wrote to {target_file}."


# ---------------------------------------------------------------------------
# Task A4: Sort contacts by last_name then first_name in /data/contacts.json
# ---------------------------------------------------------------------------
async def task_a4(source_file: str, target_file: str, sort_fields: List[str]) -> str:
    """
    Task A4.
    
    Reads the specified source file, which should contain a list of contact objects.
    The contacts are sorted by the specified fields in the given order.
    The sorted list is written to the specified target file.
    
    Args:
        source_file (str): The path to the input file containing contacts.
        target_file (str): The path to the output file where the sorted contacts will be written.
        sort_fields (List[str]): The list of fields to sort by, in order of priority.
    
    Returns:
        str: A message indicating the sorted file was successfully written.
    
    Raises:
        HTTPException: If the source file is missing, the JSON structure is unexpected, or file operations fail.
    """
    if not os.path.isfile(source_file):
        raise HTTPException(status_code=404, detail=f"{source_file} not found.")
    
    try:
        print(f"Sorting contacts by {sort_fields}")
        with open(source_file, "r") as f:
            contacts = json.load(f)
        
        if not isinstance(contacts, list):
            raise HTTPException(status_code=400, detail="Expected a list of contacts in the JSON file.")
        
        sorted_contacts = sorted(contacts, key=lambda x: tuple(x[f'{field}'] for field in sort_fields))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing contacts file.") from e
    
    try:
        with open(target_file, "w") as f:
            json.dump(sorted_contacts, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error writing sorted contacts to file.") from e
    print(f"Sorted contacts by {sort_fields} and wrote to {target_file}.")
    return f"Sorted contacts by {sort_fields} and wrote to {target_file}."


# ---------------------------------------------------------------------------
# Task A5: Write the first line of the 10 most recent .log files in /data/logs/
# ---------------------------------------------------------------------------
async def task_a5(source_dir: str, target_file: str, file_extension: str) -> str:
    """
    Task A5.
    
    In the directory specified by `source_file`, identifies the 10 most recent files 
    with the given `file_extension` (based on file modification time), reads the first line from each file, and 
    writes these lines (most recent first) to `target_file`.
    
    Args:
        source_file (str): The path to the directory containing log files.
        target_file (str): The path to the output file where the first lines will be written.
        file_extension (str): The file extension to filter files (e.g., '.log').
    
    Returns:
        str: A message indicating that the recent log lines have been successfully written.
    
    Raises:
        HTTPException: If the source directory does not exist, no files with the given extension are found, or for file read/write issues.
    """
    print(f"Checking if directory {source_dir} exists")
    if not os.path.isdir(source_dir):
        raise HTTPException(status_code=404, detail=f"Directory {source_dir} not found.")
    
    try:
        print(f"Gathering all files with extension '{file_extension}' from {source_dir}")
        # Gather all files with the given extension from the directory.
        log_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(file_extension)]
        if not log_files:
            raise HTTPException(status_code=404, detail=f"No files with extension '{file_extension}' found in the directory.")
        
        print(f"Sorting files by last modification time")
        # Sort the files by last modification time (most recent first).
        log_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        selected_logs = log_files[:10]
        
        lines = []
        for log_file in selected_logs:
            try:
                print(f"Reading first line from {log_file}")
                with open(log_file, "r") as f:
                    first_line = f.readline().strip()
                    lines.append(first_line)
            except Exception:
                lines.append(f"Error reading {os.path.basename(log_file)}")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing log files.") from e
    
    try:
        print(f"Writing first lines to {target_file}")
        with open(target_file, "w") as f:
            for line in lines:
                f.write(line + "\n")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error writing recent logs to file.") from e
    
    print(f"First lines of the 10 most recent files with extension '{file_extension}' written to {target_file}")
    return f"First lines of the 10 most recent files with extension '{file_extension}' written to {target_file}."


# ---------------------------------------------------------------------------
# Task A6: Index Markdown Files from /data/docs/
# ---------------------------------------------------------------------------
def task_a6() -> str:
    """
    Task A6:
    - Scans the directory '/data/docs/' for all Markdown (.md) files.
    - Extracts the first occurrence of an H1 header (a line starting with "# ") in each file.
    - Builds an index dictionary mapping each filename (relative to /data/docs/) to its title.
    - Writes the index to '/data/docs/index.json'.
    
    Returns:
        str: A success message with details of the generated index.
    
    Raises:
        FileNotFoundError: If the /data/docs/ directory is not found.
        RuntimeError: For issues reading files or writing the index.
    """
    docs_dir = "/data/docs/"
    index_output = os.path.join(docs_dir, "index.json")
    
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Directory {docs_dir} not found.")
    
    index_dict = {}
    
    try:
        # Walk through the docs directory; includes subdirectories.
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith(".md"):
                    target_file = os.path.join(root, file)
                    try:
                        with open(target_file, "r", encoding="utf-8") as f:
                            title = ""
                            for line in f:
                                line_strip = line.strip()
                                if line_strip.startswith("# "):
                                    title = line_strip.lstrip("# ").strip()
                                    break
                            # Save relative path (as key) and title.
                            rel_path = os.path.relpath(target_file, docs_dir)
                            index_dict[rel_path] = title
                    except Exception as e:
                        raise RuntimeError(f"Error processing file {target_file}") from e
    except Exception as e:
        raise RuntimeError("Error scanning markdown documents.") from e
    
    try:
        with open(index_output, "w", encoding="utf-8") as f:
            json.dump(index_dict, f, indent=2)
    except Exception as e:
        raise RuntimeError("Error writing index file.") from e
    
    return f"Created markdown index with {len(index_dict)} entries at {index_output}."


# ---------------------------------------------------------------------------
# Task A7: Extract Email Sender using LLM (GPT-4o-Mini via httpx)
# ---------------------------------------------------------------------------
async def task_a7() -> str:
    """
    Task A7:
    - Reads the plain-text email from '/data/email.txt'.
    - Sends the content to GPT-4o-Mini (via an HTTP POST using httpx) to extract the sender's email address.
    - Writes the extracted email address to '/data/email-sender.txt'.
    
    Returns:
        str: A message indicating the extracted sender email.
    
    Raises:
        FileNotFoundError: If the email input file does not exist.
        RuntimeError: For LLM API, network, or file writing errors.
    """
    input_file = "/data/email.txt"
    output_file = "/data/email-sender.txt"
    
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Email file {input_file} not found.")
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            email_content = f.read()
    except Exception as e:
        raise RuntimeError("Failed to read email file.") from e
    
    prompt = (
        "Extract the sender's email address from the following email content:\n\n" 
        + email_content
    )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.example.com/v1/llm",  # Replace with your actual LLM endpoint.
                json={
                    "model": "gpt-4o-mini",
                    "prompt": prompt,
                    "max_tokens": 50
                },
                headers={"Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"}
            )
            response.raise_for_status()
            result_text = response.json()["choices"][0]["text"].strip()
    except Exception as e:
        raise RuntimeError("LLM API call failed in task A7.") from e
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)
    except Exception as e:
        raise RuntimeError("Failed to write email sender output.") from e
    
    return f"Extracted sender email: {result_text}"


# ---------------------------------------------------------------------------
# Task A8: OCR Credit Card Image using LLM (GPT-4o-Mini via httpx)
# ---------------------------------------------------------------------------
async def task_a8() -> str:
    """
    Task A8:
    - Reads '/data/credit-card.png' which contains a credit card image.
    - Encodes the image in base64 and submits it (with an instructive prompt) to GPT-4o-Mini via httpx.
    - The LLM extracts the credit card number (without spaces).
    - Writes the extracted card number to '/data/credit-card.txt'.
    
    Returns:
        str: A message indicating the extracted credit card number.
    
    Raises:
        FileNotFoundError: If the credit card image file is not found.
        RuntimeError: For errors in reading the image, LLM API call, or file writing.
    """
    input_image = "/data/credit-card.png"
    output_file = "/data/credit-card.txt"
    
    if not os.path.isfile(input_image):
        raise FileNotFoundError(f"Credit card image file {input_image} not found.")
    
    try:
        with open(input_image, "rb") as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode("utf-8")
    except Exception as e:
        raise RuntimeError("Failed to read or encode credit card image.") from e
    
    prompt = (
        "Extract the credit card number from the provided image. "
        "Return the number without any spaces."
    )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.example.com/v1/llm/image",  # Replace with your actual image API endpoint.
                json={
                    "model": "gpt-4o-mini",
                    "prompt": prompt,
                    "image": img_base64,
                    "max_tokens": 50
                },
                headers={"Authorization": f"Bearer {os.getenv('AIPROXY_TOKEN')}"}
            )
            response.raise_for_status()
            card_number = response.json()["choices"][0]["text"].strip().replace(" ", "")
    except Exception as e:
        raise RuntimeError("LLM API call failed in task A8.") from e
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(card_number)
    except Exception as e:
        raise RuntimeError("Failed to write credit card number output.") from e
    
    return f"Extracted credit card number: {card_number}"


# Function to compute text embeddings
async def get_embedding(text: str) -> np.ndarray:
    """
    Generates an embedding vector for a given text using OpenAI's embeddings API.

    Args:
        text (str): The text to embed.

    Returns:
        np.ndarray: The embedding vector as a NumPy array.

    Raises:
        RuntimeError: If the API call fails or the response is invalid.
    """
    if not AIPROXY_TOKEN:
        raise ValueError("AIPROXY_TOKEN environment variable is not set.")

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "text-embedding-3-small", # "text-embedding-ada-002",  # Use a suitable embedding model
        "input": text
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(EMBEDDINGS_URL, json=payload, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if "data" not in data or not isinstance(data["data"], list) or len(data["data"]) == 0:
                raise ValueError("Invalid response format: Missing or empty 'data' field.")

            embedding = data["data"][0]["embedding"]
            return np.array(embedding)
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"API Error: {e}")
    except httpx.TimeoutException as e:
        raise RuntimeError(f"API Timeout: {e}")
    except httpx.RequestError as e:
        raise RuntimeError(f"API Request Failed: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


# Function to compute cosine similarity
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity score.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Handle zero vector case to avoid division by zero

    return dot_product / (norm_vec1 * norm_vec2)


# ---------------------------------------------------------------------------
# Task A9: Find Most Similar Comment Pair using Embeddings
# ---------------------------------------------------------------------------
async def task_a9() -> str:
    """
    Task A9:
    - Reads the file '/data/comments.txt' which contains one comment per line.
    - Uses OpenAI's embedding API to compute embeddings for each comment.
    - Calculates cosine similarity between all pairs of comments.
    - Identifies the pair of comments with the highest similarity (excluding self-similarity).
    - Writes the two similar comments, one per line, to '/data/comments-similar.txt'.

    Returns:
        str: A message indicating the similar pair has been written and the similarity score.

    Raises:
        FileNotFoundError: If '/data/comments.txt' is not found.
        RuntimeError: For errors during reading, embedding computation, or file writing.
    """
    input_file = "/data/comments.txt"
    output_file = "/data/comments-similar.txt"

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Comments file {input_file} not found.")

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            comments = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise RuntimeError("Failed to read comments file.") from e

    if len(comments) < 2:
        raise RuntimeError("Not enough comments to compute similarity.")

    # Get embeddings for all comments
    embeddings: List[np.ndarray] = []  # Explicit typing for clarity
    try:
        for comment in comments:
            embedding = await get_embedding(comment)
            embeddings.append(embedding)
    except Exception as e:
        raise RuntimeError(f"Error generating embeddings: {e}")

    # Find the pair with maximum similarity (ignoring self similarity)
    max_sim = -1.0
    pair_indices = (None, None)
    num_comments = len(comments)
    for i in range(num_comments):
        for j in range(i + 1, num_comments):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > max_sim:
                max_sim = sim
                pair_indices = (i, j)

    if pair_indices[0] is None or pair_indices[1] is None:
        raise RuntimeError("Failed to find a similar pair of comments.")

    similar_pair = (comments[pair_indices[0]], comments[pair_indices[1]])

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(similar_pair[0] + "\n" + similar_pair[1])
    except Exception as e:
        raise RuntimeError("Failed to write similar comments output.") from e

    return f"Most similar comments written to {output_file} with similarity score {max_sim:.4f}."


# ---------------------------------------------------------------------------
# Task A10: Compute Gold Ticket Total Sales from SQLite Database
# ---------------------------------------------------------------------------
def task_a10() -> str:
    """
    Task A10:
    - Connects to the SQLite database '/data/ticket-sales.db' which contains a table 'tickets'
      with columns: type, units, and price.
    - Computes the total sales (units multiplied by price) for rows where type is 'Gold'.
    - Writes the computed sum to '/data/ticket-sales-gold.txt'.
    
    Returns:
        str: A message indicating the total sales for Gold tickets.
    
    Raises:
        FileNotFoundError: If the database file is not found.
        RuntimeError: For database query failures or file writing issues.
    """
    db_file = "/data/ticket-sales.db"
    output_file = "/data/ticket-sales-gold.txt"
    
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f"SQLite database file {db_file} not found.")
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        query = "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold';"
        cursor.execute(query)
        result = cursor.fetchone()
        total_sales = result[0] if result[0] is not None else 0
        conn.close()
    except Exception as e:
        raise RuntimeError("Database query failed in task A10.") from e
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(total_sales))
    except Exception as e:
        raise RuntimeError("Failed to write ticket sales output.") from e
    
    return f"Total sales for 'Gold' tickets: {total_sales} written to {output_file}."


tools = [
    {
        "type": "function",
        "function": {
            "name": "task_a1",
            "description": "Installs 'uv' (if required) and runs the data generator script with a user email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_email": {
                        "type": "string",
                        "description": "The user's email address to pass to the data generator."
                    }
                },
                "required": ["user_email"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a2",
            "description": "Formats the contents of /data/format.md using prettier@3.4.2, updating the file in-place.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a3",
            "description": "Counts the number of Wednesdays in /data/dates.txt and writes the count to /data/dates-wednesdays.txt.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a4",
            "description": "Sorts the array of contacts in /data/contacts.json by last_name, then first_name, and writes the result to /data/contacts-sorted.json.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a5",
            "description": "Writes the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a6",
            "description": "Finds all Markdown (.md) files in /data/docs/, extracts H1 headers, and creates an index file /data/docs/index.json.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a7",
            "description": "Extracts the sender's email address from /data/email.txt using an LLM and writes it to /data/email-sender.txt.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a8",
            "description": "Extracts the credit card number from /data/credit-card.png using an LLM and writes it without spaces to /data/credit-card.txt.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a9",
            "description": "Finds the most similar pair of comments in /data/comments.txt using embeddings and writes them to /data/comments-similar.txt.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "task_a10",
            "description": "Computes total sales of 'Gold' tickets from /data/ticket-sales.db and writes the number to /data/ticket-sales-gold.txt.",
            "parameters": {
                "type": "object",
                "properties": {}
                ,
                "additionalProperties": False
            },
            "strict": True
        }
    }
]


async def query_gpt(user_input: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Queries the GPT-4o-Mini LLM with a user input, using the provided tools to guide the response.
    This function handles potential API errors gracefully.

    Args:
        user_input (str): The plain-English task description from the user.
        tools (List[Dict[str, Any]]): A list of function definitions that GPT-4o-Mini can use.

    Returns:
        Dict[str, Any]: The LLM's response in JSON format, potentially including a tool call.

    Raises:
        RuntimeError: If the API call fails or returns an unexpected result.
    """
    EMBEDDINGS_URL = "https://api.openai.com/v1/chat/completions"  # Or the proper endpoint
    api_key = os.getenv("AIPROXY_TOKEN")

    if not api_key:
        raise ValueError("Environment variable AIPROXY_TOKEN not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": user_input}],
        "tools": tools,
        "tool_choice": "auto",  # Let the LLM decide whether to use a tool
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:  # Set a reasonable timeout
            response = await client.post(EMBEDDINGS_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            response_data = response.json()

            if "choices" not in response_data or not isinstance(response_data["choices"], list) or len(response_data["choices"]) == 0:
                raise ValueError("Unexpected response format: Missing or empty 'choices' in LLM response.")

            message = response_data["choices"][0]["message"]  # Extract the message
            return message
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"LLM API Error: {e}")
    except httpx.TimeoutException as e:
        raise RuntimeError(f"LLM API Timeout: {e}")
    except httpx.RequestError as e:
        raise RuntimeError(f"LLM API Request Failed: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM API Response Decode Error: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    # asyncio.run(task_a1(os.getenv("EMAIL"), "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"))
    # asyncio.run(task_a2('./data/format.md'))
    # asyncio.run(task_a3('./data/dates.txt', './data/dates-wednesdays.txt', 'Wednesday'))
    # asyncio.run(task_a4('./data/contacts.json', './data/contacts-sorted.json', ['last_name', 'first_name']))
    asyncio.run(task_a5('./data/logs/', './data/logs-recent.txt', '.log'))
    pass