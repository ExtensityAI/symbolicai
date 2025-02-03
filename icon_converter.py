import os
import platform
import shutil
import cairosvg
from PIL import Image


def create_ico(png_path, ico_path):
    """Create a high-quality ICO file with multiple sizes."""
    # Define the sizes we want to include
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

    # Open original PNG
    img = Image.open(png_path)

    # Create a list to store all the images
    img_list = []

    # Create each size with high quality
    for size in sizes:
        resized_img = img.copy()
        resized_img.thumbnail(size, Image.Resampling.LANCZOS)
        # Convert to RGBA if not already
        if resized_img.mode != 'RGBA':
            resized_img = resized_img.convert('RGBA')
        img_list.append(resized_img)

    # Save as ICO with multiple sizes
    img_list[0].save(
        ico_path,
        format='ICO',
        sizes=[(img.width, img.height) for img in img_list],
        append_images=img_list[1:]
    )


def create_icns(png_path, icns_path):
    """Create an ICNS file for macOS."""
    if not os.path.exists('iconset'):
        os.makedirs('iconset')

    # Define the required sizes for ICNS
    icon_sizes = {
        (16, 16): "16x16",
        (32, 32): "32x32",
        (64, 64): "64x64",
        (128, 128): "128x128",
        (256, 256): "256x256",
        (512, 512): "512x512",
        (1024, 1024): "1024x1024"
    }

    img = Image.open(png_path)

    # Create each size
    for size, name in icon_sizes.items():
        resized_img = img.copy()
        resized_img.thumbnail(size, Image.Resampling.LANCZOS)

        # Convert to RGBA if not already
        if resized_img.mode != 'RGBA':
            resized_img = resized_img.convert('RGBA')

        # Save normal and @2x versions
        resized_img.save(f'iconset/icon_{name}.png')
        if size[0] <= 512:  # Don't create @2x for 1024
            resized_img.save(f'iconset/icon_{name}@2x.png')

    # Use iconutil to create icns (macOS only)
    if os.path.exists('iconset'):
        os.system(f'iconutil -c icns iconset -o {icns_path}')
        # Clean up
        import shutil
        shutil.rmtree('iconset')


def create_favicon(svg_path, output_path):
    """Create a high-quality favicon.ico with optimal sizes."""
    # Favicon-specific sizes (common favicon sizes)
    favicon_sizes = [96, 144, 192]
    temp_pngs = []
    images = []

    try:
        for size in favicon_sizes:
            temp_png = f'temp_favicon_{size}.png'
            # Convert SVG to PNG with higher quality settings
            cairosvg.svg2png(
                url=svg_path,
                write_to=temp_png,
                output_width=size,
                output_height=size,
                scale=2.0  # Increase internal resolution for better quality
            )

            # Open and optimize the PNG
            img = Image.open(temp_png)

            # Ensure RGBA mode
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # Apply minimal smoothing
            img = img.resize((size, size), Image.Resampling.LANCZOS)

            images.append(img.copy())
            temp_pngs.append(temp_png)
            img.close()

        # Save as ICO with all sizes
        images[0].save(
            output_path,
            format='ICO',
            sizes=[(size, size) for size in favicon_sizes],
            append_images=images[1:],
            optimize=True
        )

    finally:
        # Clean up temporary files
        for temp_png in temp_pngs:
            if os.path.exists(temp_png):
                os.remove(temp_png)


def convert_svg_to_ico(svg_path, output_path, sizes=[16, 32, 48, 64, 128, 256]):
    """Convert SVG to ICO with multiple sizes."""
    # Create temporary PNG files for each size
    temp_pngs = []
    images = []

    try:
        for size in sizes:
            temp_png = f'temp_{size}.png'
            # Convert SVG to PNG
            cairosvg.svg2png(
                url=svg_path,
                write_to=temp_png,
                output_width=size,
                output_height=size
            )
            # Open PNG and append to images list
            img = Image.open(temp_png)
            images.append(img.copy())
            temp_pngs.append(temp_png)
            img.close()

        # Save as ICO
        images[0].save(
            output_path,
            format='ICO',
            sizes=[(size, size) for size in sizes],
            append_images=images[1:]
        )

    finally:
        # Clean up temporary files
        for temp_png in temp_pngs:
            if os.path.exists(temp_png):
                os.remove(temp_png)


def convert_svg_to_icns(svg_path, output_path):
    """Convert SVG to ICNS for macOS."""
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    temp_pngs = []
    iconset_path = 'icon.iconset'

    try:
        # Create iconset directory
        if not os.path.exists(iconset_path):
            os.makedirs(iconset_path)

        # Generate PNG files at different sizes
        for size in sizes:
            temp_png = f'{iconset_path}/icon_{size}x{size}.png'
            temp_png2x = f'{iconset_path}/icon_{size//2}x{size//2}@2x.png'

            # Convert SVG to PNG
            cairosvg.svg2png(
                url=svg_path,
                write_to=temp_png,
                output_width=size,
                output_height=size
            )

            # Also save as @2x version if applicable
            if size > 16:
                os.rename(temp_png, temp_png2x)

            temp_pngs.append(temp_png)

        # Use iconutil to create icns (macOS only)
        os.system(f'iconutil -c icns {iconset_path} -o {output_path}')

    finally:
        # Clean up
        if os.path.exists(iconset_path):
            shutil.rmtree(iconset_path)


def main(svg_path='public/eai.svg'):
    try:
        import cairosvg
    except ImportError:
        os.system('pip install cairosvg pillow')
        import cairosvg

    # Convert for Windows
    convert_svg_to_ico(svg_path, 'icon.ico', sizes=[128])

    # Create favicon
    create_favicon(svg_path, 'favicon.ico')

    # Convert for macOS (only on macOS systems)
    if platform.system() == 'Darwin':
        convert_svg_to_icns(svg_path, 'icon.icns')


if __name__ == '__main__':
    main()