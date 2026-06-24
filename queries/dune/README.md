# Dune Analytics Queries

SQL queries used to extract PrivacyPools on-chain deposit/withdrawal data for entropy analysis.

## Queries

### `Deposit on ethereum pool of eth in privacypools`
- **Purpose**: Pulls all deposit events in ETH pool with timestamp, amount, depositor address
- **Dune link**: https://dune.com/queries/7748441
- **Output columns**: `time, depositor, amount_eth`

### `withdrawals`
- **Purpose**: Pulls withdrawal events with recipient and nullifier hash
- **Dune link**: 
- **Output columns**: 

## Notes
- All queries run against Ethereum mainnet, Dune's PrivacyPools contract tables.
- Re-run periodically since new deposits/withdrawals accrue — exports in 
  `data/raw/` are dated snapshots, not live.
