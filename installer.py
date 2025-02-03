import os
import platform
import queue
import customtkinter
import threading
import subprocess
from customtkinter import CTkButton, CTkEntry, CTk, CTkProgressBar, CTkLabel, CTkTextbox


def get_anaconda_url():
    # Check if anaconda is already installed
    if platform.system() == 'Windows':
        if os.path.exists(os.path.expanduser('~\\Anaconda3')):
            return None
    else:
        if os.path.exists(os.path.expanduser('~/anaconda3')):
            return None

    # Get Anaconda download URL based on OS
    system = platform.system()
    if system == 'Linux':
        return 'https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh'
    elif system == 'Windows':
        return 'https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Windows-x86_64.exe'
    elif system == 'Darwin':
        # distinguish between Apple Silicon and Intel Macs
        if 'arm' in platform.processor():
            return 'https://repo.anaconda.com/archive/Anaconda3-2024.10-1-MacOSX-arm64.sh'
        else:
            return 'https://repo.anaconda.com/archive/Anaconda3-2024.10-1-MacOSX-x86_64.sh'


def run_command(command):
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )

    output = []
    # Read output and error streams line by line
    while True:
        line = process.stdout.readline()
        error = process.stderr.readline()

        if not line and not error and process.poll() is not None:
            break

        if line:
            output.append(('stdout', line.strip()))
        if error:
            output.append(('stderr', error.strip()))

    returncode = process.wait()
    return returncode, output


class InstallerGUI:
    def __init__(self):
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        self.root = CTk()
        self.root.geometry("800x700")  # Increased window size
        self.root.title("SymbolicAI Installer")

        self.is_installing = False
        self.log_queue = queue.Queue()
        self.setup_gui()

        # Start the log consumer
        self.root.after(100, self.process_log_queue)

    def setup_gui(self):
        # Welcome Label
        welcome_label = CTkLabel(self.root, text="Welcome to the SymbolicAI Installer!")
        welcome_label.pack(pady=20)

        # API Key Entry
        api_key_label = CTkLabel(self.root, text="Enter your API Key:")
        api_key_label.pack(pady=5)

        self.api_key_entry = CTkEntry(self.root, width=300, placeholder_text="Your API Key here")
        self.api_key_entry.pack(pady=10)
        # Add validation for API key entry
        self.api_key_entry.bind('<KeyRelease>', self.validate_api_key)
        # Add focus out event
        self.api_key_entry.bind('<FocusOut>', lambda e: self.root.focus())

        # Progress Components
        self.log_text = CTkLabel(self.root, text="")
        self.log_text.pack(pady=10)

        self.progress_bar = CTkProgressBar(self.root, width=300)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)

        # Output Text Box
        self.output_text = CTkTextbox(self.root, width=700, height=300)
        self.output_text.pack(pady=10, padx=20)

        # Install Button (modified to start as disabled)
        self.install_button = CTkButton(
            self.root,
            text='Install Anaconda & SymbolicAI Package',
            command=self.start_installation,
            state="disabled"  # Start with disabled button
        )
        self.install_button.pack(pady=20)

    def log(self, message, message_type="info"):
        self.log_queue.put((message, message_type))

    def validate_api_key(self, event=None):
        """Validate API key and enable/disable install button accordingly"""
        api_key = self.api_key_entry.get().strip()
        self.install_button.configure(state="normal" if api_key else "disabled")

    def process_log_queue(self):
        try:
            while True:
                message, message_type = self.log_queue.get_nowait()
                self.output_text.configure(state="normal")

                # Add color coding based on message type
                if message_type == "error":
                    message = f"ERROR: {message}\n"
                    tag = "error"
                elif message_type == "stderr":
                    message = f"STDERR: {message}\n"
                    tag = "error"
                elif message_type == "stdout":
                    message = f"OUTPUT: {message}\n"
                    tag = "stdout"
                else:
                    message = f"INFO: {message}\n"
                    tag = "info"

                self.output_text.insert("end", message)
                self.output_text.see("end")
                self.output_text.configure(state="disabled")

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_log_queue)

    def update_gui(self, message, progress=None):
        def update():
            self.log_text.configure(text=message)
            if progress is not None:
                self.progress_bar.set(progress)
        self.root.after(0, update)

    def install_anaconda(self):
        try:
            self.update_gui('Installing Anaconda...', 0.2)
            self.log("Starting Anaconda installation")
            url = get_anaconda_url()

            if url is None:
                self.log("Anaconda already installed")
                self.update_gui('Anaconda already installed.', 0.4)
                return True

            if platform.system() == 'Windows':
                download_cmd = f'powershell -Command "Invoke-WebRequest -Uri {url} -OutFile anaconda_installer.exe"'
                self.log(f"Downloading Anaconda with command: {download_cmd}")
                returncode, output = run_command(download_cmd)
                for stream, line in output:
                    self.log(line, stream)
                if returncode != 0:
                    raise Exception("Failed to download Anaconda installer")

                install_cmd = 'start /wait "" anaconda_installer.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\\Anaconda3'
                self.log(f"Installing Anaconda with command: {install_cmd}")
                returncode, output = run_command(install_cmd)
                for stream, line in output:
                    self.log(line, stream)
                if returncode != 0:
                    raise Exception("Failed to install Anaconda")
            else:
                self.log(f"Downloading Anaconda from: {url}")
                returncode, output = run_command(f'curl -LO {url}')
                for stream, line in output:
                    self.log(line, stream)
                if returncode != 0:
                    raise Exception("Failed to download Anaconda installer")

                file_name = url.split('/')[-1]
                self.log(f"Installing Anaconda with file: {file_name}")
                returncode, output = run_command(f'bash {file_name} -b')
                for stream, line in output:
                    self.log(line, stream)
                if returncode != 0:
                    raise Exception("Failed to install Anaconda")

            self.update_gui('Anaconda installed.', 0.4)
            return True
        except Exception as e:
            self.log(str(e), "error")
            self.update_gui(f'Error installing Anaconda: {str(e)}')
            return False

    def create_environment(self):
        try:
            self.update_gui('Creating environment...', 0.6)
            self.log("Creating conda environment")

            if platform.system() == 'Windows':
                conda_path = os.path.expanduser('~\\Anaconda3\\Scripts\\conda.exe')
            else:
                conda_path = os.path.expanduser('~/anaconda3/bin/conda')

            if not os.path.exists(conda_path):
                raise Exception("Conda not found. Please ensure Anaconda is installed correctly.")

            cmd = f'"{conda_path}" create -y --name symaiprod python=3.10'
            self.log(f"Running command: {cmd}")
            returncode, output = run_command(cmd)
            for stream, line in output:
                self.log(line, stream)
            if returncode != 0:
                raise Exception("Failed to create conda environment")

            self.update_gui('Environment created.', 0.7)
            return True
        except Exception as e:
            self.log(str(e), "error")
            self.update_gui(f'Error creating environment: {str(e)}')
            return False

    def install_symbolicai(self):
        try:
            self.update_gui('Installing SymbolicAI...', 0.8)
            self.log("Installing SymbolicAI package")

            if platform.system() == 'Windows':
                conda_path = os.path.expanduser('~\\Anaconda3\\Scripts\\conda.exe')
                activate_cmd = f'"{conda_path}" activate symaiprod'
                pip_install_cmd = 'pip install symbolicai'
                full_cmd = f'{activate_cmd} && {pip_install_cmd}'
            else:
                # For macOS and Linux
                conda_path = os.path.expanduser('~/anaconda3/bin/conda')
                # Use the full path to pip in the conda environment
                pip_path = os.path.expanduser('~/anaconda3/envs/symaiprod/bin/pip')
                full_cmd = f'"{pip_path}" install symbolicai'

            self.log(f"Running command: {full_cmd}")
            returncode, output = run_command(full_cmd)
            for stream, line in output:
                self.log(line, stream)
            if returncode != 0:
                raise Exception("Failed to install SymbolicAI")

            self.update_gui('SymbolicAI installed.', 0.9)
            return True
        except Exception as e:
            self.log(str(e), "error")
            self.update_gui(f'Error installing SymbolicAI: {str(e)}')
            return False

    def installation_process(self):
        try:
            api_key = self.api_key_entry.get().strip()
            self.update_gui('Starting installation...', 0.1)

            if self.install_anaconda():
                if self.create_environment():
                    if self.install_symbolicai():
                        # Set API Key
                        os.environ['NEUROSYMBOLIC_ENGINE_API_KEY'] = api_key

                        # Test Installation using the correct Python path
                        self.update_gui('Testing installation...', 0.9)

                        # Use the correct Python path based on the OS
                        if platform.system() == 'Windows':
                            python_path = os.path.expanduser('~\\Anaconda3\\envs\\symaiprod\\python.exe')
                            test_cmd = f'"{python_path}" -c "from symai import *"'
                        else:
                            python_path = os.path.expanduser('~/anaconda3/envs/symaiprod/bin/python')
                            test_cmd = f'"{python_path}" -c "from symai import *"'

                        self.log(f"Testing installation with command: {test_cmd}")
                        returncode, output = run_command(test_cmd)

                        if returncode == 0:
                            self.update_gui('Installation completed successfully!', 1.0)
                        else:
                            error_message = "\n".join([line[1] for line in output if line[0] == 'stderr'])
                            self.update_gui(f'Error testing installation: {error_message}')

            self.is_installing = False
            self.install_button.configure(state="normal")

        except Exception as e:
            self.log(str(e), "error")
            self.update_gui(f'Installation failed: {str(e)}')
            self.is_installing = False
            self.install_button.configure(state="normal")

    def start_installation(self):
        if self.is_installing:
            return

        api_key = self.api_key_entry.get().strip()
        if not api_key:
            self.update_gui("Please enter a valid API key!")
            return

        self.is_installing = True
        self.install_button.configure(state="disabled")
        self.progress_bar.set(0)

        # Start installation in a separate thread
        threading.Thread(target=self.installation_process, daemon=True).start()

    def run(self):
        self.root.mainloop()


def main():
    installer = InstallerGUI()
    installer.run()


if __name__ == '__main__':
    main()
