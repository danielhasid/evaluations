import os                              # Used to set environment variables
import json                            # Used to parse and pretty-print JSON
from google.cloud import bigquery      # BigQuery client library

# Authenticate using a service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\work\PythonProject\big query\daniel-489817-b2bf1671163c.json"

# Create a BigQuery client tied to the specified GCP project
client = bigquery.Client(project='daniel-489817')

# Submit the SQL query and get back a QueryJob object
QUERY = "SELECT * FROM `daniel-489817`.`Daniel_test`.`daniel` LIMIT 1"
query_job = client.query(QUERY)

# Wait for the query to finish and retrieve the results
results = query_job.result()

# Convert results to a DataFrame, serialize to JSON string, then parse into a Python list
json_results = json.loads(results.to_dataframe().to_json(orient="records", indent=2))

# Print the value of 'string_field_3' from the first row
print(json_results[0].get('string_field_3'))