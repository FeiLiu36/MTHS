
from llm4ad.tools.llm.llm_api_https import HttpsApi
import sys
sys.path.append('../')  # This is for finding all the modules
import re  # Import the regular expression module to parse the LLM response
import os  # Import the os module for robust path handling

# --- Configuration ---
# Define the path to the source code file we want to load.
# os.path.join is used to create a path that works on any operating system.
SOURCE_CODE_FILENAME = "eop_fssp.py"
SOURCE_CODE_PATH = os.path.join('.', SOURCE_CODE_FILENAME) # Assumes the file is in the parent directory

# --- Load the source code from the file ---
# Instead of importing a variable, we will read the entire file content into a string.
try:
    with open(SOURCE_CODE_PATH, 'r', encoding='utf-8') as f:
        program = f.read()
    print(f"Successfully loaded source code from '{SOURCE_CODE_PATH}'")
except FileNotFoundError:
    print(f"Error: The source code file was not found at '{SOURCE_CODE_PATH}'.")
    print("Please ensure the file exists and the path is correct.")
    sys.exit(1) # Exit the script if the source code can't be found
except Exception as e:
    print(f"An unexpected error occurred while reading the file: {e}")
    sys.exit(1)

# --- LLM API Setup ---
# It's recommended to load secrets like API keys from environment variables
# or a config file instead of hardcoding them for better security.
llm = HttpsApi(host='api.bltcy.ai',
               key='sk-dKMuUHsISnfTuYGPDb78437190Db4bC19f968bC796260230',
               model='gemini-2.5-pro',
               timeout=360)

# --- Prompt and API Call ---
prompt = "Please carefully read the following code. 1) add numba to the code to significantly speed up the efficiency and 2) only give me the code and do not change any functionality and implementation:\n" + program
print(prompt)
response = llm.draw_sample(prompt)

print("\n--- LLM Response ---")
print(response)
print("--------------------")


# --- Process and Write the LLM Response to a File ---

# The LLM response often contains the Python code within a markdown block (```python...```).
# This part extracts the code and saves it to a new file.
output_filename = "eop_cvrp_numba.py"
code_to_write = response

# Use regex to find code inside ```python ... ```, which is a common LLM output format.
# re.DOTALL allows '.' to match newlines, which is crucial for multi-line code blocks.
match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
if match:
    # If a markdown code block is found, extract only the code inside it.
    code_to_write = match.group(1).strip()
    print(f"\nExtracted Python code from the markdown block.")
else:
    # If no block is found, assume the whole response is code, but warn the user.
    print(f"\nWarning: Could not find a '```python' markdown block. Writing the entire response to the file.")

# Write the (potentially extracted) code to the output file
try:
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(code_to_write)
    print(f"Successfully wrote optimized code to '{output_filename}'")
except IOError as e:
    print(f"\nError: Could not write to file '{output_filename}'. Reason: {e}")

