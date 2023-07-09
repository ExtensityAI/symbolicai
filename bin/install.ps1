# Check if Git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    # Download Git from the official site and install
    Invoke-WebRequest -Uri "https://github.com/git-for-windows/git/releases/download/v2.31.0.windows.1/Git-2.31.0-64-bit.exe" -OutFile "$env:TEMP\Git-2.31.0-64-bit.exe"
    Start-Process "$env:TEMP\Git-2.31.0-64-bit.exe" -ArgumentList "/VERYSILENT" -Wait
}

# Check if Anaconda installed
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    # Download Anaconda from the official site and install
    Invoke-WebRequest -Uri "https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Windows-x86_64.exe" -OutFile "$env:TEMP\Anaconda3-2023.03-1-Windows-x86_64.exe"
    Start-Process "$env:TEMP\Anaconda3-2023.03-1-Windows-x86_64.exe" -ArgumentList "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S" -Wait
}

# Source conda
$env:Path = "$env:USERPROFILE\Anaconda3\Scripts;" + $env:Path

# Create conda environment named symenv
conda create --name symenv -y

# Install Jupyter Notebook using conda
conda install -n symenv jupyter -y

# Activate the conda environment
conda activate symenv

# Install symbolicai using pip that comes with the Anaconda environment
& "$env:USERPROFILE\Anaconda3\envs\symenv\Scripts\pip" install symbolicai

# Set up an ipython configuration
if (!(Test-Path -Path "$env:USERPROFILE\.ipython\profile_default\startup")) {
    New-Item -ItemType directory -Path "$env:USERPROFILE\.ipython\profile_default\startup"
}
Add-Content "$env:USERPROFILE\.ipython\profile_default\startup\startup.py" "from symai import *; from symai.extended.conversation import Conversation; q = Conversation(auto_print=True)"

# Create a shortcut for the ipython console on Desktop
$Shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\SymbolicAI.lnk")
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.WorkingDirectory = "" # set this to the intended starting directory
$command = '-ExecutionPolicy Bypass -NoExit -Command . "$env:USERPROFILE\Anaconda3\envs\symenv\Scripts\ipython.exe"'
$Shortcut.Arguments = $command
$Shortcut.Save()
