from PIL import Image

def pngs_to_gif(png_files, output_path, duration=100, loop=0):
    """
    Convert a list of PNG files to an animated GIF.

    Args:
        png_files (list of str): List of file paths to PNG images (in order).
        output_path (str): Path where the output GIF should be saved.
        duration (int): Duration between frames in milliseconds.
        loop (int): Number of times the GIF should loop (0 means infinite).
    """
    if not png_files:
        raise ValueError("The list of PNG files is empty.")

    frames = [Image.open(png).convert("RGBA") for png in png_files]
    frames[0].save(
        output_path,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    print(f"GIF saved to {output_path}")