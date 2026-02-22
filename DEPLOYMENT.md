# Deployment Notes

## PyMySQL Installation Fix

If you encounter "No module named 'pymysql'" error:

```bash
pip install --force-reinstall pymysql==1.1.1
sudo pip install pymysql==1.1.1  # For system-wide access
```

## Database Connections

- **PostgreSQL**: ✅ Working
- **MySQL**: Credentials configured (check network access)

## Quick Start

```bash
./setup.sh
streamlit run app.py
```