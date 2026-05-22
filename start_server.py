#!/usr/bin/env python
"""Simple starter script for the Django ML service server."""
import os
import sys
import subprocess

def main():
    # Change to the house_price_project directory (UI project)
    server_dir = os.path.join(os.path.dirname(__file__), 'house_price_project')
    os.chdir(server_dir)
    
    print("=" * 60)
    print("Starting Django ML Service UI Server...")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print("Server will be available at: http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Ensure database tables exist (runs migrations automatically)
    try:
        subprocess.run([sys.executable, 'manage.py', 'migrate', '--noinput'], check=True)
        subprocess.run([sys.executable, 'manage.py', 'runserver'], check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nError starting server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
