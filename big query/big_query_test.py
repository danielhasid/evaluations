import os                              # Used to set environment variables
import json                            # Used to parse and pretty-print JSON
from google.cloud import bigquery      # BigQuery client library

# Authenticate using a service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\work\PythonProject\big query\daniel-489817-b2bf1671163c.json"

# Create a BigQuery client tied to the specified GCP project
client = bigquery.Client(project='daniel-489817')

# Submit the SQL query and get back a QueryJob object
QUERY = "SELECT * FROM `daniel-489817`.`Daniel_test`.`workers` LIMIT 2"
query_job = client.query(QUERY)

# Wait for the query to finish and convert results to a DataFrame once (REST endpoint, no Storage API needed)
df = query_job.result().to_dataframe(create_bqstorage_client=False)

# Save results to a JSON file
df.to_json("worker_output.json", orient="records", indent=2)

# Parse the DataFrame into a Python list of dicts
json_results = json.loads(df.to_json(orient="records", indent=2))

# Print the value of 'first_name' from the first row
print(json_results[0].get('first_name'))