# Run both backend and frontend in separate terminals

Write-Host "🚀 Starting Carbon-Aware Cloud Scheduler..." -ForegroundColor Green
Write-Host ""

# Check and install backend dependencies if needed
Write-Host "Checking backend dependencies..." -ForegroundColor Yellow
cd backend
if (Test-Path "venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
    pip install -q -r requirements.txt 2>$null
} else {
    Write-Host "⚠️  Backend virtual environment not found. Run setup.ps1 first." -ForegroundColor Red
    cd ..
    exit 1
}
cd ..

# Start Backend
Write-Host "✓ Starting Backend API on http://localhost:8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; .\venv\Scripts\Activate.ps1; python app.py"

Start-Sleep -Seconds 3

# Start Frontend  
Write-Host "✓ Starting Frontend UI on http://localhost:3000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"

Start-Sleep -Seconds 2

Write-Host ""
Write-Host "🎉 Application started successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Yellow
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor Yellow
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C in each terminal window to stop the servers" -ForegroundColor Gray
Write-Host ""

