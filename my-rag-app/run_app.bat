@echo off
echo Dang khoi chay Ung dung RAG Tra cuu Tai lieu Offline...
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run ui\main.py
pause
