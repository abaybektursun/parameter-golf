"""One-time setup: create DynamoDB tables for multi-agent coordination."""

import boto3

session = boto3.Session(profile_name="fuelos")
ddb = session.client("dynamodb", region_name="us-east-1")

TABLES = [
    {
        "TableName": "pg_sota",
        "KeySchema": [{"AttributeName": "pr_number", "KeyType": "HASH"}],
        "AttributeDefinitions": [{"AttributeName": "pr_number", "AttributeType": "N"}],
        "BillingMode": "PAY_PER_REQUEST",
    },
    {
        "TableName": "pg_experiments",
        "KeySchema": [{"AttributeName": "experiment_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "experiment_id", "AttributeType": "S"},
            {"AttributeName": "state", "AttributeType": "S"},
            {"AttributeName": "updated_at", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "state-index",
                "KeySchema": [
                    {"AttributeName": "state", "KeyType": "HASH"},
                    {"AttributeName": "updated_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
    },
    {
        "TableName": "pg_runs",
        "KeySchema": [{"AttributeName": "run_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "experiment_id", "AttributeType": "S"},
            {"AttributeName": "created_at", "AttributeType": "S"},
            {"AttributeName": "machine_id", "AttributeType": "S"},
            {"AttributeName": "sliding_val_bpb", "AttributeType": "N"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "experiment-index",
                "KeySchema": [
                    {"AttributeName": "experiment_id", "KeyType": "HASH"},
                    {"AttributeName": "created_at", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "machine-index",
                "KeySchema": [
                    {"AttributeName": "machine_id", "KeyType": "HASH"},
                    {"AttributeName": "sliding_val_bpb", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
    },
    {
        "TableName": "pg_agents",
        "KeySchema": [{"AttributeName": "agent_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [{"AttributeName": "agent_id", "AttributeType": "S"}],
        "BillingMode": "PAY_PER_REQUEST",
    },
]

existing = ddb.list_tables()["TableNames"]
for table_def in TABLES:
    name = table_def["TableName"]
    if name in existing:
        print(f"  {name}: already exists, skipping")
        continue
    ddb.create_table(**table_def)
    print(f"  {name}: created")

print("Done.")
