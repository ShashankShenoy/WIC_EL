# Original Streamlit Application

This folder contains the original Streamlit-based carbon-aware scheduler application.

## 🚀 Running the Streamlit Version

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app will open at **http://localhost:8501**

## ⚠️ Note

This is the **legacy version**. The new FastAPI + React version is in the `backend/` and `frontend/` folders with better performance and modern UI.

### Comparison

| Feature | Streamlit (this folder) | FastAPI + React (new) |
|---------|------------------------|----------------------|
| Performance | Slower (full reloads) | Fast (client-side) |
| UI | Basic | Modern, professional |
| API | No | Full REST API |
| Mobile | Poor | Responsive |
| Deployment | Single process | Scalable |

## 📁 Files

- `app.py` - Main Streamlit application (1358 lines)
- `requirements.txt` - Python dependencies for Streamlit version
- `generate_sample_data.py` - Script to generate synthetic test data
- `optimization_presets.json` - Pre-configured optimization settings

## 📊 Sample Data

Sample data files are located in the `../data/` folder:
- `energy_data_historical.csv` - Historical carbon intensity data
- `energy_forecast_models.pkl` - Pre-trained ML models
- `cloud_schedule_future_optimal.csv` - Example output

To use them, upload through the Streamlit sidebar or copy them to this folder.

## 🔄 Migration

To use the new version, see the main project README and run `.\setup.ps1` then `.\run.ps1` from the root directory.
