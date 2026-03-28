# Intelligent Hardware Thermal Failure Prediction System 🖥️🔥

An advanced machine learning pipeline designed to predict hardware failure risks using live telemetry data. This project replaces static BIOS temperature thresholds with a proactive, hybrid ML approach.

## 🚀 Overview
Hardware degrades non-linearly. A CPU at 85°C with a failing fan and erratic voltage is in critical danger, whereas a GPU at 85°C under stable voltage is operating normally. This system uses **Decision Trees** for interpretable rule generation and **Artificial Neural Networks (Backpropagation)** to detect complex, non-linear failure states.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Frontend/UI:** HTML5, CSS3, Bootstrap 5

## 📊 Dataset Features
The model ingests the following telemetry parameters:
* `CPU_Temp` (Celsius)
* `GPU_Temp` (Celsius)
* `Fan_RPM` (Rotations Per Minute)
* `Voltage_Variance` (Volts)
* `RAM_Usage` (%)

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/hardware-failure-prediction.git](https://github.com/yourusername/hardware-failure-prediction.git)
   ```
2. Install dependencies:
   pip install -r requirements.txt
   
3. Run the prediction script:
   python main.py

🧠 Algorithms
Decision Tree (ID3 concept): Provides human-readable conditional logic for hardware technicians.

Multi-Layer Perceptron (ANN): Utilizes backpropagation to map complex interactions between minor voltage drops and thermal limits.

4. UI Idea: Web Dashboard (HTML/CSS)
Since you are familiar with HTML and hosting projects on Vercel, building a clean web dashboard is the best way to make this project look advanced for your viva. 

You can use a Python framework like Flask or FastAPI to serve your ML model, and connect it to this simple, professional frontend. 

**`index.html`**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hardware Telemetry ML Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; }
        .card { border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .status-safe { color: #198754; font-weight: bold; }
        .status-critical { color: #dc3545; font-weight: bold; }
    </style>
</head>
<body>

<div class="container mt-5">
    <h2 class="text-center mb-4">Hardware Thermal Prediction System</h2>
    
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card p-4">
                <form id="telemetryForm">
                    <div class="mb-3">
                        <label class="form-label">CPU Temperature (°C)</label>
                        <input type="number" class="form-control" id="cpuTemp" placeholder="e.g. 75" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Fan Speed (RPM)</label>
                        <input type="number" class="form-control" id="fanRpm" placeholder="e.g. 1200" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Voltage Variance (V)</label>
                        <input type="number" step="0.01" class="form-control" id="voltage" placeholder="e.g. 0.05" required>
                    </div>
                    <button type="button" class="btn btn-dark w-100" onclick="runPrediction()">Analyze Hardware Risk</button>
                </form>
            </div>
        </div>
        
        <div class="col-md-4">
            <div class="card p-4 text-center h-100 d-flex justify-content-center">
                <h4>System Status</h4>
                <hr>
                <h1 id="resultDisplay" class="text-muted">Awaiting Data...</h1>
                <p class="mt-3 text-muted" id="modelInfo">ML Model: ANN (Backpropagation)</p>
            </div>
        </div>
    </div>
</div>

<script>
    // In a real application, this function would send an AJAX/Fetch request to your Python backend.
    // This is a simulation for your frontend demo.
    function runPrediction() {
        const cpu = document.getElementById('cpuTemp').value;
        const fan = document.getElementById('fanRpm').value;
        const resultDisplay = document.getElementById('resultDisplay');

        if(cpu > 85 && fan < 1500) {
            resultDisplay.innerHTML = "CRITICAL FAILURE RISK";
            resultDisplay.className = "status-critical";
        } else {
            resultDisplay.innerHTML = "SYSTEM SAFE";
            resultDisplay.className = "status-safe";
        }
    }
</script>

</body>
</html>
