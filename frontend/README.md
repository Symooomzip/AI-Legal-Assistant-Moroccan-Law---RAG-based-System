# Frontend (React + Vite)

## 1) Start the Python API backend

From project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn legal_api_server:app --host 0.0.0.0 --port 8000 --reload
```

## 2) Start the React frontend

From `frontend/`:

```powershell
npm install
npm run dev
```

The app runs at:

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`

## Optional environment variable

You can override API URL:

```powershell
$env:VITE_API_BASE_URL="http://127.0.0.1:8000"
```
