#!/bin/bash
# Check if git is installed
if ! command -v git &> /dev/null
then
    # Install git
    if [ "$(uname)" == "Darwin" ]; then
        # MacOS
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
        brew install git
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Linux
        sudo apt-get install git -y
    fi
fi

# Check if Anaconda and git are installed
if ! command -v conda &> /dev/null
then
    # Check the OS to download the appropriate version of Anaconda
    if [ "$(uname)" == "Darwin" ]; then
        # MacOS
        if [ "$(uname -m)" == "x86_64" ]; then
            # Intel Mac
            curl -O https://repo.anaconda.com/archive/Anaconda3-2023.03-1-MacOSX-x86_64.sh
            bash Anaconda3-2023.03-1-MacOSX-x86_64.sh -b
        else
            # Apple Silicon Mac
            curl -O https://repo.anaconda.com/archive/Anaconda3-2023.03-1-MacOSX-arm64.sh
            bash Anaconda3-2023.03-1-MacOSX-arm64.sh -b
        fi
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Linux
        wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
        bash Anaconda3-2023.03-1-Linux-x86_64.sh -b
    fi
fi

# Initialize Anaconda3
# Skip initialization if already initialized
if [ -z "$(conda info --envs)" ]
then
    eval "$(conda shell.bash hook)"
fi

# Source conda
source ~/anaconda3/etc/profile.d/conda.sh

# Create conda environment named symai
conda create --name symenv -y

# Install Jupyter Notebook using conda
conda install -n symenv jupyter -y

# Activate the conda environment
conda activate symenv

# Install symbolicai using pip that comes with the Anaconda environment
~/anaconda3/envs/symenv/bin/pip install symbolicai

# Install ipython configuration
mkdir -p ~/.ipython/profile_default/startup
echo "from symai import *; from symai.extended.conversation import Conversation; q = Conversation(auto_print=True)" > ~/.ipython/profile_default/startup/startup.py

# Create a desktop icon on Linux
if [ "$(uname)" == "Darwin" ]; then
    # MacOS
    # Generate open-terminal-and-run script
    echo "#!/bin/bash
    open -a Terminal.app ~/anaconda3/envs/symenv/bin/ipython" > SymbolicAI.app

    # Make the generated script executable
    chmod +x SymbolicAI.app

    # Move the script to the Applications directory
    mv SymbolicAI.app /Applications/SymbolicAI.app

    # Generate an Automator application that runs the script
    echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">
    <plist version=\"1.0\">
    <dict>
        <key>Label</key>
        <string>com.user.loginscript</string>
        <key>ProgramArguments</key>
        <array>
            <string>/Applications/SymbolicAI.app</string>
        </array>
        <key>RunAtLoad</key>
        <true/>
    </dict>
    </plist>" > ~/Library/LaunchAgents/com.user.loginscript.plist
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    echo "[Desktop Entry]
    Version=1.0
    Type=Application
    Name=Sym AI
    Icon=terminal
    Exec=bash -c 'source ~/anaconda3/bin/activate symenv && ipython'
    Terminal=true" > ~/Desktop/symai.desktop

    # Make the desktop file executable
    chmod +x ~/Desktop/symai.desktop
fi