# privacypools-deanonymization
Measuring effective anonymity in PrivacyPools via entropy analysis of on-chain deposit/withdrawal data

## Project Structure
.

├── data/

│   ├── raw/              # Raw exports from Dune (deposits, withdrawals, templates)

│   └── processed/        # Cleaned/derived datasets ready for entropy analysis

├── queries/

│   └── dune/             # SQL queries used on Dune Analytics, with their own README

├── results/

│   ├── figures/          # Plots and charts generated from the analysis

│   └── tables/           # Summary statistics, output CSVs

├── src/                  # Python modules (entropy calculation, heuristics, data loading)

├── LICENSE

└── README.md

### Folder details

- **`data/raw/`** — Untouched exports straight from Dune (e.g. `privacypools_eth_pool_deposits...`, plus the export scripts `Export_Deposits_eth_pool.py` and `Export_Templates.py` used to pull them are in **`src/`**). Not meant to be edited by hand.
- **`data/processed/`** — Cleaned, merged, or reshaped data derived from `raw/`, ready to feed into the entropy analysis in `src/`.
- **`queries/dune/`** — All SQL queries run on Dune Analytics to extract on-chain deposit/withdrawal data. See [`queries/dune/README.md`](queries/dune/README.md) for a catalog of each query and what it returns.
- **`results/figures/`** — Generated plots (e.g. entropy distributions, n_eff/N gap visualizations).
- **`results/tables/`** — Numeric outputs and summary tables from the analysis.
- **`src/`** — Core Python code: entropy/anonymity metric functions, address-clustering heuristics, data loading utilities.
