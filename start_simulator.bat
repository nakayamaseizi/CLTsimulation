@echo off
powershell -NoExit -Command "Set-Location '%~dp0'; & '%~dp0.venv\Scripts\python.exe' -m streamlit run '%~dp0src\clt_fire_sim\web\app.py' --browser.gatherUsageStats false --server.address localhost --server.fileWatcherType poll"
