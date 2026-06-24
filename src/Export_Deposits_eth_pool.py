from dune_client.client import DuneClient
import pandas as pd

dune = DuneClient("4LMa73t2JMPn128D6NcmJBBHnGwf8bd7")
query_result = dune.get_latest_result(7748441)  # replace with your query ID

df = pd.DataFrame(query_result.result.rows)
df.to_csv("privacypools_eth_pool_deposits.csv", index=False)

print("Done! Saved", len(df), "rows.")
