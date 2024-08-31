import os
import sys
from PIL import Image, ImageDraw, ImageOps, ImageStat

class HalftoneGenerator:
    def __init__(self, image_path):
        """
        image_path is the file path to the image to be halftoned.
        """
        self.image_path = image_path

    def generate_halftone(
        self,
        sample_size=10,
        scaling_factor=1,
        gray_component_ratio=0,
        filename_suffix="_halftoned",
        rotation_angles=[0, 15, 30, 45],
        mode="color",
        enable_antialiasing=False,
        output_file_format="default",
        quality=75,
        save_individual_channels=False,
        channels_file_format="default",
        channels_mode="color",
    ):
        """
        Generates a halftone version of the image. Optional parameters allow customization.
        Arguments:
            sample_size: Size of sample box from the original image, in pixels.
            scaling_factor: Max output dot diameter is sample_size * scaling_factor.
            gray_component_ratio: Percentage of gray component to shift from CMY to K.
            filename_suffix: Suffix to add to the filename (before the extension).
            rotation_angles: List of angles for each channel's screen rotation.
            mode: 'color' or 'grayscale'.
            enable_antialiasing: Boolean for enabling antialiasing.
            output_file_format: "default", "jpeg", "png", "tiff".
            quality: Quality level for saving images, default 75.
            save_individual_channels: Boolean, whether to save separate channel images.
            channels_file_format: "default", "jpeg", "png", "tiff".
            channels_mode: 'color' or 'grayscale'.
        """

        # Validate input parameters
        self.validate_params(
            angles=rotation_angles,
            antialias=enable_antialiasing,
            output_format=output_file_format,
            quality=quality,
            percentage=gray_component_ratio,
            sample=sample_size,
            save_channels=save_individual_channels,
            channels_format=channels_file_format,
            channels_style=channels_mode,
            scale=scaling_factor,
            style=mode,
        )

        base_name, extension = os.path.splitext(self.image_path)

        # Determine the correct extension based on the output format
        if output_file_format == "jpeg":
            extension = ".jpg"
        elif output_file_format == "png":
            extension = ".png"
        elif output_file_format == "tiff":
            extension = ".tiff"

        output_file_name = f"{base_name}{filename_suffix}{extension}"

        try:
            original_image = Image.open(self.image_path)
        except IOError as e:
            raise Exception(f"Failed to open the image file '{self.image_path}'") from e

        if mode == "grayscale":
            rotation_angles = rotation_angles[:1]
            grayscale_image = original_image.convert("L")
            channel_images = self.create_halftone(
                original_image, grayscale_image, sample_size, scaling_factor, rotation_angles, enable_antialiasing
            )
            combined_image = channel_images[0]
        else:
            cmyk_image = self.apply_gcr(original_image, gray_component_ratio)
            channel_images = self.create_halftone(original_image, cmyk_image, sample_size, scaling_factor, rotation_angles, enable_antialiasing)

            if save_individual_channels:
                self.export_channel_images(
                    channel_images,
                    mode=channels_mode,
                    format=channels_file_format,
                    base_filename=output_file_name,
                    quality=quality,
                )

            combined_image = Image.merge("CMYK", channel_images)

        # Save the final combined image
        if extension == ".jpg":
            combined_image.save(output_file_name, "JPEG", subsampling=0, quality=quality)
        elif extension == ".png":
            combined_image.convert("RGB").save(output_file_name, "PNG")
        elif extension == ".tiff":
            combined_image.save(output_file_name, "TIFF", quality=quality)

    def validate_params(
        self,
        angles,
        antialias,
        output_format,
        quality,
        percentage,
        sample,
        save_channels,
        channels_format,
        channels_style,
        scale,
        style,
    ):
        """
        Ensures all input parameters are valid.
        Raises TypeError or ValueError if any parameter is invalid.
        """

        if not isinstance(angles, list):
            raise TypeError("The angles parameter must be a list of integers.")

        if style == "grayscale":
            if len(angles) < 1:
                raise ValueError("For grayscale, angles list must contain at least 1 integer.")
        else:
            if len(angles) != 4:
                raise ValueError("For color, angles list must contain exactly 4 integers.")

        for angle in angles:
            if not isinstance(angle, int):
                raise ValueError("All elements of the angles list must be integers.")

        if not isinstance(antialias, bool):
            raise TypeError("The antialias parameter must be a boolean.")

        if output_format not in ["default", "jpeg", "png", "tiff"]:
            raise ValueError("The output_format parameter must be 'default', 'jpeg', 'png', or 'tiff'.")

        if not isinstance(quality, int):
            raise TypeError("The quality parameter must be an integer.")
        if quality < 0 or quality > 100:
            raise ValueError("The quality parameter must be between 0 and 100.")

        if not isinstance(percentage, (float, int)):
            raise TypeError("The percentage parameter must be an integer or float.")

        if not isinstance(sample, int):
            raise TypeError("The sample parameter must be an integer.")

        if not isinstance(save_channels, bool):
            raise TypeError("The save_channels parameter must be a boolean.")

        if channels_format not in ["default", "jpeg", "png", "tiff"]:
            raise ValueError("The channels_format parameter must be 'default', 'jpeg', 'png', or 'tiff'.")

        if channels_style not in ["color", "grayscale"]:
            raise ValueError("The channels_style parameter must be 'color' or 'grayscale'.")

        if not isinstance(scale, int):
            raise TypeError("The scale parameter must be an integer.")

        if style not in ["color", "grayscale"]:
            raise ValueError("The style parameter must be 'color' or 'grayscale'.")

        return True

    def apply_gcr(self, image, percentage):
        """
        Applies Gray Component Replacement to the image and returns a CMYK image.
        """
        cmyk_image = image.convert("CMYK")
        if not percentage:
            return cmyk_image
        cmyk_channels = cmyk_image.split()
        cmyk = []
        for i in range(4):
            cmyk.append(cmyk_channels[i].load())
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                gray_value = int(min(cmyk[0][x, y], cmyk[1][x, y], cmyk[2][x, y]) * percentage / 100)
                for i in range(3):
                    cmyk[i][x, y] = cmyk[i][x, y] - gray_value
                cmyk[3][x, y] = gray_value
        return Image.merge("CMYK", cmyk_channels)

    def create_halftone(self, original_image, cmyk_image, sample_size, scaling_factor, rotation_angles, enable_antialiasing):
        """
        Creates a list of halftone images for each CMYK channel.
        """

        if enable_antialiasing:
            scaling_factor *= 4

        cmyk_channels = cmyk_image.split()
        halftone_images = []

        for channel, angle in zip(cmyk_channels, rotation_angles):
            rotated_channel = channel.rotate(angle, expand=1)
            new_size = rotated_channel.size[0] * scaling_factor, rotated_channel.size[1] * scaling_factor
            halftone_image = Image.new("L", new_size)
            draw = ImageDraw.Draw(halftone_image)

            # Generate the halftone pattern
            for x in range(0, rotated_channel.size[0], sample_size):
                for y in range(0, rotated_channel.size[1], sample_size):
                    sample_box = rotated_channel.crop((x, y, x + sample_size, y + sample_size))
                    average_value = ImageStat.Stat(sample_box).mean[0]
                    circle_diameter = (average_value / 255) ** 0.5
                    draw_box_size = sample_size * scaling_factor
                    circle_draw_diameter = circle_diameter * draw_box_size
                    box_x, box_y = (x * scaling_factor), (y * scaling_factor)
                    x1 = box_x + ((draw_box_size - circle_draw_diameter) / 2)
                    y1 = box_y + ((draw_box_size - circle_draw_diameter) / 2)
                    x2 = x1 + circle_draw_diameter
                    y2 = y1 + circle_draw_diameter

                    draw.ellipse([(x1, y1), (x2, y2)], fill=255)

            halftone_image = halftone_image.rotate(-angle, expand=1)
            width_half, height_half = halftone_image.size
            xx1 = (width_half - original_image.size[0] * scaling_factor) / 2
            yy1 = (height_half - original_image.size[1] * scaling_factor) / 2
            xx2 = xx1 + original_image.size[0] * scaling_factor
            yy2 = yy1 + original_image.size[1] * scaling_factor

            halftone_image = halftone_image.crop((xx1, yy1, xx2, yy2))

            if enable_antialiasing:
                halftone_image = halftone_image.resize(
                    (int((xx2 - xx1) / 4), int((yy2 - yy1) / 4)), resample=Image.LANCZOS
                )

            halftone_images.append(halftone_image)
        return halftone_images

    def export_channel_images(
        self,
        channel_images,
        mode,
        format,
        base_filename,
        quality,
    ):
        """
        Save the separate CMYK channel images.
        """

        channel_labels = (
            ("c", "cyan"),
            ("m", "magenta"),
            ("y", "yellow"),
            ("k", "black"),
        )

        base_name, extension = os.path.splitext(base_filename)

        if format == "jpeg":
            extension = ".jpg"
        elif format == "png":
            extension = ".png"
        elif format == "tiff":
            extension = ".tiff"

        for idx, channel_img in enumerate(channel_images):
            channel_filename = f"{base_name}_{channel_labels[idx][0]}{extension}"
            processed_img = ImageOps.invert(channel_img)

            if mode == "color" and idx < 3:
                processed_img = ImageOps.colorize(
                    processed_img, black=channel_labels[idx][1], white="white"
                )

            if extension == ".jpg":
                processed_img.convert("CMYK").save(
                    channel_filename, "JPEG", subsampling=0, quality=quality
                )
            elif extension == ".png":
                processed_img.save(channel_filename, "PNG")
            elif extension == ".tiff":
                processed_img.save(channel_filename, "TIFF", quality=quality)


if __name__ == "__main__":
    # Example usage
    input_image_path = 'Y.png'
    halftone_gen = HalftoneGenerator(input_image_path)
    halftone_gen.generate_halftone(
        mode="grayscale",
        rotation_angles=[0],
        output_file_format="tiff",
        gray_component_ratio=20,
        sample_size=2,
        scaling_factor=5
    )
