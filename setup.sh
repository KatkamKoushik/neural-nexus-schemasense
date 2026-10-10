#!/bin/bash

# Setup script for neural-nexus-schemasense project
echo "🚀 Setting up neural-nexus-schemasense project..."

# Navigate to project directory (portable)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install project dependencies
echo "📦 Installing project dependencies..."
if [ -f "constraints.txt" ]; then
    pip install -r requirements.txt -c constraints.txt
else
    pip install -r requirements.txt
fi

# Verify critical packages
echo "🔍 Verifying critical packages..."
python -c "import pymysql; print('✅ PyMySQL installed successfully')" || echo "❌ PyMySQL installation failed"
python -c "import psycopg2; print('✅ psycopg2 installed successfully')" || echo "❌ psycopg2 installation failed"
python -c "import streamlit; print('✅ Streamlit installed successfully')" || echo "❌ Streamlit installation failed"
python -c "import google.genai; print('✅ Google GenAI installed successfully')" || echo "❌ Google GenAI installation failed"

# Create secrets file if it doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "📝 Creating secrets.toml from template..."
    cp .streamlit/secrets.toml.template .streamlit/secrets.toml
    echo "⚠️  Please edit .streamlit/secrets.toml with your actual credentials"
fi

echo "✅ Setup complete! Run 'streamlit run app.py' to start the application."