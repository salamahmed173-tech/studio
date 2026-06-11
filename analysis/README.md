# GAC Motors -> UAE import analysis

This folder contains a script to extract GAC Motors import units to UAE from a CAAM CSV (local file or URL), visualize the last two years, and forecast the next 6 months using Prophet.

Quick steps:

1. Install dependencies into your workspace virtual environment:

```powershell
cd /d C:\Users\user\Studio
.venv\Scripts\Activate.ps1
pip install -r analysis\requirements.txt
```

2. Run the analysis (example using local CSV):

```powershell
python analysis\analyze_gac_uae.py --input caam_data.csv --outdir analysis\outputs
```

3. Outputs:
- `analysis/outputs/monthly_units.csv` - aggregated monthly units
- `analysis/outputs/monthly_history.png` - historical monthly plot
- `analysis/outputs/forecast.csv` - full Prophet forecast table
- `analysis/outputs/forecast.png` - combined history+forecast plot

Notes:
- The script tries to auto-detect columns (date, maker/manufacturer, country/destination, units). If detection fails, provide a cleaned CSV with these columns.
- Prophet installation can take time on Windows; if `prophet` fails to install, consider using WSL or installing prebuilt wheels.

If you want, I can attempt to run the script here if you provide either a CAAM CSV file in the workspace or a public CSV URL.
