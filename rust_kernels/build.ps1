# Build the Rust GPU-kernel extension into the project venv.
# Requires: rustup (windows-gnu toolchain) + WinLibs mingw-w64 (for dlltool/as).
$mingw = "C:\Users\Ken\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
$env:Path = "$mingw;$env:USERPROFILE\.cargo\bin;$env:Path"
$env:VIRTUAL_ENV = "D:\uq\TinyVLM\.venv"
Set-Location "D:\uq\TinyVLM\rust_kernels"
& "D:\uq\TinyVLM\.venv\Scripts\python.exe" -m maturin develop --release
