from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

project_root = Path(SPECPATH).parent

datas = collect_data_files("pycapcut")
datas += collect_data_files("imageio_ffmpeg")
datas += [(str(project_root / "src" / "auto_capcut" / "assets"), "auto_capcut/assets")]

a = Analysis(
    [str(project_root / "src" / "auto_capcut" / "main.py")],
    pathex=[str(project_root / "src")],
    datas=datas,
    hiddenimports=["pymediainfo", "pycapcut", "imageio_ffmpeg", "cv2", "PyQt6.QtMultimedia", "PyQt6.QtMultimediaWidgets"],
    excludes=["uiautomation"],
    name="AutoCapCut",
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="AutoCapCut", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AutoCapCut")
