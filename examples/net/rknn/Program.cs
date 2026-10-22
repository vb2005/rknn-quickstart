using System.Diagnostics;
using System.Globalization;
using RknnPInvokeNet9;

const int InputW = 640;
const int InputH = 640;
const int WarmupCount = 5;
const int DefaultCount = 20;

if (args.Length > 0 && args[0] is "--help" or "-h")
{
    Console.WriteLine("Usage: dotnet run -- <model.rknn> <image.jpg> [iterations]");
    return;
}

string modelPath = args.ElementAtOrDefault(0) ?? "yolo26s-rk3588.rknn";
string imagePath = args.ElementAtOrDefault(1) ?? "test.jpg";
int count = args.Length > 2
    ? int.Parse(args[2], CultureInfo.InvariantCulture)
    : DefaultCount;
if (count < 1) throw new ArgumentOutOfRangeException(nameof(count));

try
{
    Console.WriteLine("Create RKNN context and load model...");
    using var rknn = new RknnSession(modelPath);
    Console.WriteLine($"API: {rknn.ApiVersion}; NPU driver: {rknn.DriverVersion}");
    Console.WriteLine("Core mask: 0|1|2 (7)");
    PrintInfo("Input", rknn.InputInfo);
    foreach (RknnTensorInfo output in rknn.OutputInfos)
        PrintInfo("Output", output);

    Console.WriteLine("Load and preprocess image...");
    PreparedImage frame = ImagePreprocessor.Letterbox(imagePath, InputW, InputH);
    Console.WriteLine($"Original image: {frame.OriginalWidth}x{frame.OriginalHeight}");
    Console.WriteLine($"Input: [1,{InputH},{InputW},3], UINT8, NHWC, RGB");
    Console.WriteLine($"Letterbox scale={frame.Scale:F6} padding=({frame.Left},{frame.Top})");

    Console.WriteLine($"Warm-up: {WarmupCount} iterations...");
    for (int i = 0; i < WarmupCount; i++)
        _ = rknn.InferRgb24(frame.Rgb, InputW, InputH);

    // Mirrors Python MVP: one inference, then count timed inferences.
    RknnTensorOutput[] outputs = rknn.InferRgb24(frame.Rgb, InputW, InputH);
    Stopwatch timer = Stopwatch.StartNew();
    for (int i = 0; i < count; i++)
        outputs = rknn.InferRgb24(frame.Rgb, InputW, InputH);
    timer.Stop();

    Console.WriteLine();
    Console.WriteLine("===========================");
    Console.WriteLine("RKNN inference test / C#");
    Console.WriteLine("===========================");
    Console.WriteLine($"Iterations: {count}");
    Console.WriteLine($"Total time: {timer.Elapsed.TotalSeconds:F3} s");
    Console.WriteLine($"Average: {timer.Elapsed.TotalMilliseconds / count:F3} ms");
    Console.WriteLine($"FPS: {count / timer.Elapsed.TotalSeconds:F2}");
    Console.WriteLine($"Output count: {outputs.Length}");

    for (int i = 0; i < outputs.Length; i++)
    {
        RknnTensorOutput t = outputs[i];
        ReadOnlySpan<float> values = t.Values;
        double sum = 0;
        float min = float.PositiveInfinity;
        float max = float.NegativeInfinity;
        foreach (float value in values)
        {
            min = MathF.Min(min, value);
            max = MathF.Max(max, value);
            sum += value;
        }
        Console.WriteLine();
        Console.WriteLine($"Output {i}");
        Console.WriteLine($" shape: [{string.Join(", ", t.Info.Shape)}]");
        Console.WriteLine(" dtype: Float32 (requested from RKNN runtime)");
        Console.WriteLine($" min: {min:G8}");
        Console.WriteLine($" max: {max:G8}");
        Console.WriteLine($" mean: {sum / values.Length:G8}");
        Console.WriteLine(" first values: " + string.Join(", ",
            t.Values.Take(20).Select(v => v.ToString("G7", CultureInfo.InvariantCulture))));
    }

    Console.WriteLine("\nDone.");
}
catch (DllNotFoundException ex)
{
    Console.Error.WriteLine("librknnrt.so not found. Install a matching RKNN runtime and run sudo ldconfig.");
    Console.Error.WriteLine(ex.Message);
    Environment.ExitCode = 1;
}
catch (Exception ex)
{
    Console.Error.WriteLine(ex);
    Environment.ExitCode = 1;
}

static void PrintInfo(string kind, RknnTensorInfo t)
{
    Console.WriteLine($"{kind}[{t.Index}] {t.Name}: " +
                      $"[{string.Join(",", t.Shape)}], {t.Type}, {t.Format}, " +
                      $"elements={t.ElementCount}, zp={t.ZeroPoint}, scale={t.Scale:G7}");
}
