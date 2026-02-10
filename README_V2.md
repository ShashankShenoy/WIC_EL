# Carbon-Aware Cloud Scheduler v2.0

Modern full-stack application for sustainable cloud computing using FastAPI + React.

## 🏗️ Architecture

### Backend (FastAPI)
- **Location**: `backend/`
- **Tech Stack**: Python, FastAPI, Pandas, Scikit-learn, Pymoo
- **Features**:
  - RESTful API for optimization
  - Multi-objective NSGA-II algorithm
  - Carbon intensity forecasting
  - Session-based data management

### Frontend (React)
- **Location**: `frontend/`
- **Tech Stack**: TypeScript, React, Vite, TailwindCSS
- **Features**:
  - Enterprise dashboard UI
  - File upload with drag-and-drop
  - Real-time optimization progress
  - Interactive charts and metrics
  - Export to JSON/CSV

## 🚀 Quick Start

### Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
# Or: uvicorn app:app --reload
```

Backend will run on: **http://localhost:8000**

API documentation: **http://localhost:8000/docs**

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will run on: **http://localhost:3000**

## 📁 Project Structure

```
├── backend/                      # FastAPI server
│   ├── app.py                   # API routes & endpoints
│   ├── core/
│   │   ├── forecasting.py       # Forecast generation
│   │   ├── optimization.py      # NSGA-II optimization
│   │   ├── analytics.py         # Metrics calculation
│   │   └── utils.py             # Helper functions
│   └── requirements.txt
│
├── frontend/                     # React application
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── ui/              # Base UI primitives
│   │   │   ├── FileUpload.tsx
│   │   │   └── MetricCard.tsx
│   │   ├── pages/
│   │   │   └── Dashboard.tsx    # Main dashboard page
│   │   ├── services/
│   │   │   └── api.ts           # API client
│   │   ├── lib/
│   │   │   └── utils.ts         # Utilities
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── streamlit/                    # Original Streamlit app (legacy)
│   ├── app.py                   # Original implementation
│   ├── requirements.txt         # Streamlit dependencies
│   ├── generate_sample_data.py  # Data generator
│   └── optimization_presets.json
│
├── data/                         # Sample datasets
│   ├── energy_data_historical.csv
│   ├── energy_forecast_models.pkl
│   └── cloud_schedule_future_optimal.csv
│
├── setup.ps1                     # One-time setup script
├── run.ps1                       # Start both servers
├── README_V2.md                  # This file
├── MIGRATION_GUIDE.md            # Migration documentation
└── UI_IMPROVEMENTS.md            # UI enhancement details
```

## 🔌 API Endpoints

### `POST /api/upload`
Upload historical CSV and model PKL files.
- **Returns**: `session_id` for subsequent requests

### `POST /api/forecast/{session_id}`
Generate carbon intensity forecasts.
- **Body**: `OptimizationConfig`
- **Returns**: Forecasted slots

### `POST /api/optimize/{session_id}`
Run multi-objective optimization.
- **Body**: `OptimizationConfig`
- **Returns**: Pareto solutions, schedule, convergence data

### `GET /api/analytics/{session_id}`
Get detailed analytics and insights.
- **Returns**: Regional stats, hourly patterns, recommendations

### `GET /api/export/{session_id}?format=json|csv`
Export optimized schedule.
- **Returns**: Schedule in requested format

## 🎨 UI Features

### Dashboard Components
1. **File Upload** - Drag & drop CSV and PKL files
2. **Configuration Panel** - Adjust optimization parameters
3. **KPI Metrics** - Real-time carbon savings visualization
4. **Results Table** - Interactive schedule preview
5. **Export Options** - Download as JSON or CSV

### Design System
- **Colors**: Green-focused palette for sustainability
- **Typography**: Modern sans-serif (system fonts)
- **Components**: Tailwind CSS + custom components
- **Responsiveness**: Mobile-first design

## 🔧 Configuration Options

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_jobs` | Number of jobs to schedule | 20 | 1-500 |
| `forecast_days` | Days to forecast ahead | 3 | 1-30 |
| `avg_job_power_kw` | Job power consumption | 0.5 | 0.1-10 |
| `avg_job_duration_min` | Job duration | 5 | 1-120 |
| `weight_carbon` | Carbon priority | 0.5 | 0-1 |
| `weight_renewable` | Renewable priority | 0.3 | 0-1 |
| `weight_cost` | Cost priority | 0.2 | 0-1 |
| `pop_size` | NSGA-II population | 100 | 20-500 |
| `n_gen` | NSGA-II generations | 200 | 10-500 |

## 🆚 Streamlit vs FastAPI+React

### Why Migrate?

| Feature | Streamlit | FastAPI+React |
|---------|-----------|---------------|
| **Performance** | Slow, full page reloads | Fast, client-side rendering |
| **Customization** | Limited | Full control |
| **Scalability** | Single-threaded | Multi-threaded API |
| **UI Quality** | Basic | Professional |
| **API Support** | None | Full REST API |
| **Mobile** | Poor | Responsive |
| **Deployment** | Challenging | Standard practices |

### Advantages of New Architecture

✅ **Separation of Concerns**: Backend logic separate from UI  
✅ **API-First**: Can integrate with other services  
✅ **Better UX**: No page reloads, instant feedback  
✅ **Modern Stack**: Industry-standard technologies  
✅ **Extensible**: Easy to add new features  
✅ **Professional**: Enterprise-grade dashboard  

## 📊 Example Workflow

1. **Upload Data**: Drop CSV and PKL files
2. **Configure**: Set job parameters (power, duration)
3. **Forecast**: Generate future carbon intensity predictions
4. **Optimize**: Run NSGA-II to find optimal schedule
5. **Analyze**: View metrics, carbon savings, regional distribution
6. **Export**: Download schedule as JSON or CSV

## 🐛 Troubleshooting

### Backend Issues
- **Port conflict**: Change port in `app.py`: `uvicorn.run(app, port=8001)`
- **Module errors**: Ensure all dependencies installed: `pip install -r requirements.txt`
- **CORS errors**: Check frontend URL in CORS settings

### Frontend Issues
- **Build errors**: Delete `node_modules` and run `npm install` again
- **API connection**: Verify backend is running on port 8000
- **TypeScript errors**: Run `npm run build` to check for issues

## 🚢 Deployment

### Backend (Production)
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### Frontend (Production)
```bash
npm run build
# Serve the 'dist' folder with nginx, Apache, or static hosting
```

### Docker (Coming Soon)
Docker Compose configuration for one-command deployment.

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Built with ❤️ for a sustainable future**
