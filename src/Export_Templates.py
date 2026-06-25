# if you didn't install thoses package here is the command : pip install dune-client pandas
from dune_client.client import DuneClient
import pandas as pd 

dune = DuneClient("API_KEY")
query_result = dune.get_latest_result(7748441)  # replace with your query ID

df = pd.DataFrame(query_result.result.rows)
df.to_csv("privacypools_eth_pool_deposits.csv", index=False) # replace with the file name

print("Done! Saved", len(df), "rows.")
