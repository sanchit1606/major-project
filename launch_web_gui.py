#!/usr/bin/env python3
"""
Major Project Web GUI Launcher
This script launches the beautiful web-based Major Project CT Scan Reconstruction GUI.
"""

import os
import sys
import subprocess
import webbrowser
import time

def main():
    print("🚀 Major Project Web GUI Launcher")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('web_gui/app.py'):
        print("❌ Error: web_gui/app.py not found in current directory")
        print("Please run this script from the Major Project root directory")
        return
    
    # Change to web_gui directory
    os.chdir('web_gui')
    
    # Check if Flask is installed
    try:
        import flask
        print("✅ Flask is available")
    except ImportError:
        print("📦 Installing Flask...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Flask"])
            print("✅ Flask installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Flask")
            return
    
    # Check other dependencies
    dependencies = ['torch', 'numpy', 'matplotlib', 'skimage', 'tqdm', 'PIL', 'imageio', 'psutil']
    missing_deps = []
    
    for dep in dependencies:
        try:
            if dep == 'PIL':
                import PIL
            elif dep == 'skimage':
                import skimage
            else:
                __import__(dep)
        except ImportError:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("Some features may not work properly")
        print("Install with: pip install " + " ".join(missing_deps))
        print()
    
    print("🎨 Starting beautiful web-based Major Project GUI...")
    print("📱 The GUI will open in your default browser")
    print("🌐 You can also manually navigate to: http://localhost:5000")
    print()
    print("✨ Features:")
    print("   • Beautiful, modern interface with animations")
    print("   • Real-time console output and progress tracking")
    print("   • System monitoring and GPU detection")
    print("   • All Major Project functionality in a web interface")
    print()
    
    # Auto-open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
        except:
            pass
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Launch Flask app
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Web GUI stopped by user")
    except Exception as e:
        print(f"❌ Error starting web GUI: {e}")
        print("Please check the error message above")

if __name__ == "__main__":
    main()
