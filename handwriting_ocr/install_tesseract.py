import os
import sys
import subprocess
import requests
import winreg

def download_tesseract():
    """Download Tesseract installer"""
    url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.1.20230401.exe"
    installer = "tesseract_installer.exe"
    print("Downloading Tesseract...")
    
    # Download with requests
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Save the installer
    with open(installer, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return installer

def install_tesseract(installer):
    """Install Tesseract silently"""
    install_cmd = [installer, "/S", "/D=C:\\Program Files\\Tesseract-OCR"]
    print("Installing Tesseract...")
    subprocess.run(install_cmd, shell=True)

def add_to_path():
    """Add Tesseract to system PATH"""
    tesseract_path = "C:\\Program Files\\Tesseract-OCR"
    
    try:
        # Open the registry key for system PATH
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment", 0, winreg.KEY_ALL_ACCESS)
        
        # Get current PATH
        path = winreg.QueryValueEx(key, "Path")[0]
        
        # Add Tesseract path if not already present
        if tesseract_path not in path:
            new_path = path + ";" + tesseract_path
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
            print("Added Tesseract to PATH")
        
        winreg.CloseKey(key)
    except Exception as e:
        print(f"Error updating PATH: {str(e)}")

def main():
    try:
        # Download and install Tesseract
        installer = download_tesseract()
        install_tesseract(installer)
        
        # Add to PATH
        add_to_path()
        
        # Clean up installer
        if os.path.exists(installer):
            os.remove(installer)
        
        print("\nTesseract installation completed!")
        print("Please restart your terminal/IDE for the PATH changes to take effect.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 