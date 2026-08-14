$ErrorActionPreference = "Stop"
py -3.12 -m pip install -e . pyinstaller
py -3.12 -m PyInstaller --noconfirm packaging/autocapcut.spec
Write-Host "Portable build written to dist\AutoCapCut"

