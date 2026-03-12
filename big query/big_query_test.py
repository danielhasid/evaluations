import os                                # Standard library for interacting with the operating system
import json                              # Standard library for parsing and formatting JSON data
from google.cloud import bigquery        # Google Cloud BigQuery client library

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\work\PythonProject\big query\daniel-489817-b2bf1671163c.json"  # Set the path to the service account key file for authentication

client = bigquery.Client(project='daniel-489817')  # Create a BigQuery client tied to the specified GCP project

QUERY = """SELECT TO_JSON_STRING(t) AS json_result   -- Convert each row to a JSON string, aliased as json_result
FROM `daniel-489817`.`Daniel_test`.`workers` t       -- Query the workers table, aliased as t
LIMIT 2"""

query_job = client.query(QUERY)          # Submit the SQL query and return a QueryJob object
rows = query_job.result()                # Wait for the query to complete and return a RowIterator

json_result = [json.loads(row["json_result"]) for row in rows]  # Iterate over rows, parse each JSON string into a Python dict

OUTPUT_FILE = r"C:\work\PythonProject\big query\output.json"  # Define the output file path for saving results

with open(OUTPUT_FILE, "w") as f:        # Open the file in write mode (creates it if it doesn't exist)
    json.dump(json_result, f, indent=2)  # Write the list of dicts to the file as formatted JSON
    print(f"Results saved to {OUTPUT_FILE}")  # Confirm the file was saved successfully

print(json.dumps(json_result, indent=2)) # Also print the formatted JSON to the console