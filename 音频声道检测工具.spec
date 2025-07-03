# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['audio_analyzer.py'],
    pathex=[],
    binaries=[],
    datas=[('requirements.txt', '.')],
    hiddenimports=['numpy', 'pandas', 'librosa', 'soundfile', 'PyQt5', 'openpyxl', 'concurrent.futures', 'multiprocessing'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='音频声道检测工具',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon_256x256.png'],
)
