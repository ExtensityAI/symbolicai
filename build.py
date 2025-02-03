# build.py
import os
import platform
import shutil
import subprocess
import sys
from typing import Optional


class BuildConfig:
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self.arch = self._determine_arch()

    def _determine_arch(self) -> str:
        if self.machine == 'arm64' or self.machine == 'aarch64':
            return 'arm64'
        elif self.machine == 'x86_64' or self.machine == 'AMD64':
            return 'x86_64'
        else:
            raise ValueError(f"Unsupported architecture: {self.machine}")

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == 'Darwin' and self.arch == 'arm64'


# Add this function to build.py
def create_info_plist(app_dir: str, target_arch: str):
    """Create Info.plist file for macOS app bundle."""
    info_plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>SymbolicAI Installer</string>
    <key>CFBundleExecutable</key>
    <string>SymbolicAI_Installer</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.symbolicai.installer</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>SymbolicAI Installer</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
</dict>
</plist>
"""
    plist_path = os.path.join(app_dir, 'Contents', 'Info.plist')
    with open(plist_path, 'w') as f:
        f.write(info_plist)


def create_spec_file(target_arch: str) -> str:
    """Create a spec file for the target architecture."""
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-
import sys
import os
import customtkinter
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect all necessary packages
cts = collect_all('customtkinter')

a = Analysis(
    ['installer.py'],
    pathex=[],
    binaries=[],
    datas=[
        *cts[0],  # Binaries
        *cts[1],  # Datas
    ],
    hiddenimports=[*cts[2]],  # Hidden imports
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    target_arch='{target_arch}'
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='SymbolicAI_Installer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        target_arch='{target_arch}',
        codesign_identity=None,
        entitlements_file=None,
        icon='icon.icns'
    )

    # Bundle everything into the app
    app = BUNDLE(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name='SymbolicAI_Installer-{target_arch}.app',
        icon='icon.icns',
        bundle_identifier='com.symbolicai.installer',
        info_plist={{
            'CFBundleShortVersionString': '1.0.0',
            'LSMinimumSystemVersion': '10.15.0',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
        }},
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='SymbolicAI_Installer',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        target_arch='{target_arch}',
        codesign_identity=None,
        entitlements_file=None,
        icon='icon.ico' if sys.platform == 'win32' else None
    )
"""
    spec_file = f'installer_{target_arch}.spec'
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    return spec_file


def clean_build_dirs():
    """Clean up build directories."""
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)


def ensure_icons():
    """Ensure icons are available before building."""
    if not os.path.exists('icon_converter.py'):
        raise FileNotFoundError("icon_converter.py not found!")
    subprocess.check_call(['python', 'icon_converter.py'])


def verify_executable_format(executable_path: str, target_arch: str):
    """Verify the format of the executable."""
    try:
        output = subprocess.check_output(['file', executable_path], text=True)
        print(f"Executable format: {output.strip()}")

        if target_arch == 'x86_64':
            if 'x86_64' not in output:
                print(f"Warning: Executable {executable_path} is not in x86_64 format")
                return False
        elif target_arch == 'arm64':
            if 'arm64' not in output:
                print(f"Warning: Executable {executable_path} is not in arm64 format")
                return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking executable format: {str(e)}")
        return False


def verify_binary_architecture(binary_path: str, expected_arch: str):
    """Verify the architecture of a binary."""
    try:
        output = subprocess.check_output(['lipo', '-info', binary_path], text=True)
        if expected_arch not in output:
            print(f"Warning: Binary {binary_path} does not contain {expected_arch} architecture")
            print(f"Architecture info: {output.strip()}")
            return False
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error checking architecture: {str(e)}")
        return False


def verify_mac_app(app_path: str):
    """Verify the macOS app bundle."""
    try:
        # Verify bundle structure
        required_paths = [
            'Contents/Info.plist',
            'Contents/MacOS/SymbolicAI_Installer',
            'Contents/Resources/icon.icns'
        ]

        for path in required_paths:
            full_path = os.path.join(app_path, path)
            if not os.path.exists(full_path):
                print(f"Warning: Missing {path}")

        # Verify signing of all components
        try:
            print("\nVerifying code signing:")
            subprocess.check_call(['codesign', '--verify', '--verbose=4', app_path])
            print("Main bundle verification successful")

            # Verify all binaries in the bundle
            for root, _, files in os.walk(app_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        output = subprocess.check_output(['file', file_path], text=True)
                        if 'Mach-O' in output:
                            subprocess.check_call(['codesign', '--verify', file_path])
                            print(f"Verified: {os.path.relpath(file_path, app_path)}")
                    except subprocess.CalledProcessError:
                        print(f"Warning: Signature verification failed for {file_path}")
                    except Exception:
                        continue

            # Check if app can be opened
            subprocess.check_call(['spctl', '--assess', '--verbose=4', app_path])
            print("App passed system security assessment")

        except subprocess.CalledProcessError as e:
            print(f"Warning: Verification failed: {str(e)}")
        except FileNotFoundError:
            print("Warning: Verification tools not found")

    except Exception as e:
        print(f"Error verifying app bundle: {str(e)}")


def build_installer(target_arch: Optional[str] = None):
    """Build the installer for the current platform and specified architecture."""
    config = BuildConfig()

    # Clean previous builds
    clean_build_dirs()

    # Ensure correct Python architecture
    if config.system == 'Darwin':
        subprocess.run(['arch', f'-{target_arch}' if target_arch else '',
                       sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.run(['arch', f'-{target_arch}' if target_arch else '',
                       sys.executable, '-m', 'pip', 'install', 'pyinstaller', 'customtkinter'])

    # Create spec file for target architecture
    spec_file = create_spec_file(target_arch or config.arch)

    # Build command
    build_cmd = ['pyinstaller', '--clean', '--noconfirm', spec_file]

    if config.system == 'Darwin':
        build_cmd = ['arch', f'-{target_arch}'] + build_cmd if target_arch else build_cmd

    try:
        # Execute build
        subprocess.run(build_cmd, check=True)

        # Create platform-specific release folder
        release_dir = f'release_{config.system.lower()}_{target_arch}'
        if os.path.exists(release_dir):
            shutil.rmtree(release_dir)
        os.makedirs(release_dir)

        if config.system == 'Darwin':
            app_name = f'SymbolicAI_Installer-{target_arch}.app'
            final_app = f'{release_dir}/{app_name}'

            # Create proper bundle structure
            os.makedirs(f'{final_app}/Contents/MacOS', exist_ok=True)
            os.makedirs(f'{final_app}/Contents/Resources', exist_ok=True)
            os.makedirs(f'{final_app}/Contents/Frameworks', exist_ok=True)

            # Find the built app in dist
            dist_app = None

            # Look for the .app in dist directory
            for item in os.listdir('dist'):
                if item.endswith('.app'):
                    dist_app = os.path.join('dist', item)
                    break

            if not dist_app:
                raise FileNotFoundError("Built .app not found in dist directory")

            # Copy the entire contents preserving structure
            shutil.copytree(dist_app, final_app, dirs_exist_ok=True)

            # Ensure executable is in the correct location
            executable_path = f'{final_app}/Contents/MacOS/SymbolicAI_Installer'
            if not os.path.exists(executable_path):
                # Look for the executable in the dist directory
                for root, _, files in os.walk(dist_app):
                    for file in files:
                        if file == 'SymbolicAI_Installer':
                            src = os.path.join(root, file)
                            shutil.copy2(src, executable_path)
                            break

            # Copy icon
            if os.path.exists('icon.icns'):
                shutil.copy2('icon.icns', f'{final_app}/Contents/Resources/')

            # Create Info.plist
            create_info_plist(final_app, target_arch)

            # Create entitlements file
            create_entitlements()

            # Set permissions
            os.chmod(executable_path, 0o755)

            # Set proper permissions
            subprocess.run(['chmod', '-R', '+x', final_app], check=True)

            # Remove any existing signatures and extended attributes
            subprocess.run(['xattr', '-cr', final_app], check=True)

            try:
                # First, sign all frameworks and libraries
                print("Signing frameworks and libraries...")
                for root, _, files in os.walk(final_app):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith(('.so', '.dylib')) or 'Python' in file:
                            try:
                                print(f"Signing {file_path}")
                                subprocess.run([
                                    'codesign',
                                    '--force',
                                    '--sign', '-',
                                    '--timestamp',
                                    '--verbose',
                                    '--preserve-metadata=identifier,entitlements,requirements,flags,runtime',
                                    file_path
                                ], check=True, capture_output=True, text=True)
                            except subprocess.CalledProcessError as e:
                                print(f"Warning: Failed to sign {file_path}")
                                print(f"Error: {e.stderr}")

                # Sign the main executable
                print("Signing main executable...")
                subprocess.run([
                    'codesign',
                    '--force',
                    '--sign', '-',
                    '--timestamp',
                    '--verbose',
                    '--options', 'runtime',
                    executable_path
                ], check=True, capture_output=True, text=True)

                # Finally sign the entire bundle
                print("Signing complete bundle...")
                subprocess.run([
                    'codesign',
                    '--force',
                    '--sign', '-',
                    '--timestamp',
                    '--verbose',
                    '--deep',
                    '--options', 'runtime',
                    '--entitlements', 'entitlements.plist',
                    final_app
                ], check=True, capture_output=True, text=True)

            except subprocess.CalledProcessError as e:
                print(f"Code signing failed: {e.stderr}")

            # Verify the build
            verify_mac_app(final_app)

            # Verify the executable format
            verify_executable_format(executable_path, target_arch)

    except subprocess.CalledProcessError as e:
        print(f"Build process failed: {str(e)}")
        raise
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error during build: {str(e)}")
        raise


def create_release_zip(target_arch: str):
    config = BuildConfig()
    release_dir = f'release_{config.system.lower()}_{target_arch}'
    archive_name = f'SymbolicAI_Installer_{config.system.lower()}_{target_arch}'
    shutil.make_archive(archive_name, 'zip', release_dir)


def sign_mac_app(app_path: str):
    """Sign all binaries in the macOS app bundle."""
    try:
        # Remove any existing signatures first
        subprocess.run(['xattr', '-cr', app_path], check=True)

        # Sign all binaries and libraries recursively
        for root, _, files in os.walk(app_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    output = subprocess.check_output(['file', file_path], text=True)
                    if 'Mach-O' in output:
                        subprocess.run([
                            'codesign',
                            '--force',
                            '--sign',
                            '-',
                            '--timestamp',
                            '--deep',
                            '--preserve-metadata=entitlements,requirements,flags,runtime',
                            file_path
                        ], check=True)
                        print(f"Signed: {file_path}")
                except (subprocess.CalledProcessError, Exception) as e:
                    print(f"Warning: Could not sign {file_path}: {str(e)}")
                    continue

        # Finally sign the main bundle
        subprocess.run([
            'codesign',
            '--force',
            '--sign',
            '-',
            '--timestamp',
            '--deep',
            '--options', 'runtime',
            '--entitlements', 'entitlements.plist',
            app_path
        ], check=True)
        print(f"Successfully signed {app_path}")
    except Exception as e:
        print(f"Warning: Signing failed: {str(e)}")


def create_entitlements(entitlements_path: str = 'entitlements.plist'):
    """Create entitlements file for macOS app."""
    entitlements = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key>
    <true/>
    <key>com.apple.security.cs.allow-jit</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
    <key>com.apple.security.cs.debugger</key>
    <true/>
    <key>com.apple.security.get-task-allow</key>
    <true/>
    <key>com.apple.security.automation.apple-events</key>
    <true/>
    <key>com.apple.security.files.user-selected.read-write</key>
    <true/>
    <key>com.apple.security.files.downloads.read-write</key>
    <true/>
</dict>
</plist>"""
    with open(entitlements_path, 'w') as f:
        f.write(entitlements)


def build_universal_mac():
    """Build universal binary for macOS (Apple Silicon only)."""
    config = BuildConfig()
    if not config.is_apple_silicon:
        print("Universal builds are only supported on Apple Silicon Macs")
        return

    # Build for both architectures
    build_installer('arm64')
    build_installer('x86_64')

    # Verify the builds
    verify_mac_app(f'release_darwin_arm64/SymbolicAI_Installer-arm64.app')
    verify_mac_app(f'release_darwin_x86_64/SymbolicAI_Installer-x86_64.app')

    # Create universal build directory
    universal_dir = 'release_darwin_universal'
    if os.path.exists(universal_dir):
        shutil.rmtree(universal_dir)
    os.makedirs(f'{universal_dir}/SymbolicAI_Installer.app/Contents/MacOS')

    # Use lipo to create universal binary
    subprocess.check_call([
        'lipo',
        'release_darwin_arm64/SymbolicAI_Installer-arm64.app/Contents/MacOS/SymbolicAI_Installer',
        'release_darwin_x86_64/SymbolicAI_Installer-x86_64.app/Contents/MacOS/SymbolicAI_Installer',
        '-create',
        '-output',
        f'{universal_dir}/SymbolicAI_Installer.app/Contents/MacOS/SymbolicAI_Installer'
    ])

    # Copy resources and Info.plist from arm64 build
    shutil.copytree(
        'release_darwin_arm64/SymbolicAI_Installer-arm64.app/Contents/Resources',
        f'{universal_dir}/SymbolicAI_Installer.app/Contents/Resources'
    )
    shutil.copy2(
        'release_darwin_arm64/SymbolicAI_Installer-arm64.app/Contents/Info.plist',
        f'{universal_dir}/SymbolicAI_Installer.app/Contents/Info.plist'
    )

    # Verify universal build
    verify_mac_app(f'{universal_dir}/SymbolicAI_Installer.app')

    # Create zip for universal build
    create_release_zip('universal')


if __name__ == '__main__':
    config = BuildConfig()

    # Ensure icons are available
    ensure_icons()

    if config.system == 'Darwin':
        if config.is_apple_silicon:
            # On Apple Silicon, build universal binary
            build_universal_mac()
        else:
            # On Intel Mac, build only x86_64
            build_installer('x86_64')
            create_release_zip('x86_64')
    else:
        # For other platforms, build for current architecture
        build_installer()
        create_release_zip(config.arch)