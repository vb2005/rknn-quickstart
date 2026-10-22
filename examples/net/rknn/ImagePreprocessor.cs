using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace RknnPInvokeNet9;

public sealed record PreparedImage(
    byte[] Rgb, int Width, int Height,
    int OriginalWidth, int OriginalHeight,
    double Scale, int Left, int Top);

public static class ImagePreprocessor
{
    /// <summary>
    /// Python-equivalent stages: imread -> letterbox (bilinear, fill=114)
    /// -> BGR-to-RGB -> contiguous NHWC uint8 input.
    /// ImageSharp interpolation may differ by ~1 intensity level from OpenCV.
    /// </summary>
    public static PreparedImage Letterbox(string imagePath, int width = 640, int height = 640)
    {
        using Image<Rgb24> image = Image.Load<Rgb24>(imagePath);
        double scale = Math.Min((double)width / image.Width, (double)height / image.Height);
        int newWidth = Math.Clamp((int)Math.Round(image.Width * scale), 1, width);
        int newHeight = Math.Clamp((int)Math.Round(image.Height * scale), 1, height);
        int left = (width - newWidth) / 2;
        int top = (height - newHeight) / 2;

        using Image<Rgb24> resized = image.Clone(ctx =>
            ctx.Resize(newWidth, newHeight, KnownResamplers.Bilinear));

        // All pixels have RGB = (114,114,114) until replaced by image pixels.
        byte[] dst = new byte[checked(width * height * 3)];
        Array.Fill(dst, (byte)114);

        resized.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < newHeight; y++)
            {
                Span<Rgb24> row = accessor.GetRowSpan(y);
                int offset = checked(((top + y) * width + left) * 3);
                for (int x = 0; x < newWidth; x++)
                {
                    Rgb24 c = row[x];
                    dst[offset++] = c.R;
                    dst[offset++] = c.G;
                    dst[offset++] = c.B;
                }
            }
        });

        return new PreparedImage(dst, width, height,
            image.Width, image.Height, scale, left, top);
    }
}
