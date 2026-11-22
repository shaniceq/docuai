DocuAI - complete project with login

Folder layout
  app.py                  Flask backend (accounts, per-user documents, /api/chat)
  templates/index.html    main app page
  templates/login.html    login + sign-up page
  static/                 style.css, auth.css, app.js, doctypes.js, auth.js, templates.json

Run
  1. pip install -r requirements.txt
  2. (optional, real AI answers)  set GROQ_API_KEY in your terminal
       PowerShell:   $env:GROQ_API_KEY="gsk_..."
       macOS/Linux:  export GROQ_API_KEY="gsk_..."
  3. python app.py   ->  http://127.0.0.1:5000

Your existing docuai.db keeps working. Documents saved before accounts existed
go to the first account that signs up.

Easier key setup: create a file named .env next to app.py containing
  GROQ_API_KEY=gsk_yourkey
