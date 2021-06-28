mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
[theme]\n\
primaryColor="#e82453"\n\
backgroundColor="#3e3e43"\n\
secondaryBackgroundColor="#868c98"\n\
textColor="#f1f2f5"\n\
font="monospace"\n\
\n\

" > ~/.streamlit/config.toml
