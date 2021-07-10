mkdir -p ~/.streamlit/

echo "[theme]
primaryColor='#e61e4e'
backgroundColor='#3e3e43'
secondaryBackgroundColor='#868c98'
textColor='#f1f2f5'
font='serif'
[server]
port = $PORT
enableCORS = false
headless = true

" > ~/.streamlit/config.toml
