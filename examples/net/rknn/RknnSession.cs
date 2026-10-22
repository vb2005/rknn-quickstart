using System.Runtime.InteropServices;

namespace RknnPInvokeNet9;

public sealed record RknnTensorInfo(
    int Index, string Name, int[] Shape, string Type,
    string Format, int ElementCount, int ZeroPoint, float Scale);

// Always returned as FP32, because outputs_get uses want_float = 1.
public sealed record RknnTensorOutput(RknnTensorInfo Info, float[] Values);

/// <summary>
/// One RKNN model/context. Instances are not thread-safe: serialize Infer calls
/// or create separate model contexts when parallel inference is required.
/// </summary>
public sealed unsafe class RknnSession : IDisposable
{
    private ulong _context;
    private bool _disposed;

    public RknnTensorInfo InputInfo { get; }
    public IReadOnlyList<RknnTensorInfo> OutputInfos { get; }
    public string ApiVersion { get; }
    public string DriverVersion { get; }

    public RknnSession(string modelPath, int coreMask = RknnNative.NpuCore0_1_2)
    {
        if (!OperatingSystem.IsLinux() || RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
            throw new PlatformNotSupportedException("Requires Linux ARM64 (AArch64).");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("RKNN model not found", modelPath);

        VerifyAbi();

        try
        {
            // modelSize = 0 tells rknn_init that modelPath is a UTF-8 file path.
            RknnNative.Check(RknnNative.Init(out _context, Path.GetFullPath(modelPath), 0, 0, 0), "rknn_init");
            RknnNative.Check(RknnNative.SetCoreMask(_context, coreMask), "rknn_set_core_mask");

            RknnInputOutputNum counts = default;
            RknnNative.Check(RknnNative.Query(_context, RknnNative.QueryInOutNum,
                &counts, (uint)sizeof(RknnInputOutputNum)), "rknn_query(IN_OUT_NUM)");

            if (counts.InputCount != 1)
                throw new NotSupportedException($"Demo supports one input, model requires {counts.InputCount}.");
            if (counts.OutputCount is 0 or > 64)
                throw new InvalidOperationException($"Unexpected output count: {counts.OutputCount}.");

            InputInfo = QueryTensor(RknnNative.QueryInputAttr, 0);
            var infos = new List<RknnTensorInfo>((int)counts.OutputCount);
            for (uint i = 0; i < counts.OutputCount; i++)
                infos.Add(QueryTensor(RknnNative.QueryOutputAttr, i));
            OutputInfos = infos;

            RknnSdkVersion sdk = default;
            if (RknnNative.Query(_context, RknnNative.QuerySdkVersion,
                    &sdk, (uint)sizeof(RknnSdkVersion)) == 0)
            {
                ApiVersion = sdk.GetApiVersion();
                DriverVersion = sdk.GetDriverVersion();
            }
            else
            {
                ApiVersion = "unknown";
                DriverVersion = "unknown";
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public RknnTensorOutput[] InferRgb24(byte[] rgb, int width, int height)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(rgb);

        // The MVP's input is [1, H, W, 3], RGB, NHWC, uint8.
        int expected = checked(width * height * 3);
        if (rgb.Length != expected)
            throw new ArgumentException($"Expected {expected} RGB bytes, got {rgb.Length}.", nameof(rgb));
        if (InputInfo.ElementCount != expected)
            throw new InvalidOperationException(
                $"Model input has {InputInfo.ElementCount} elements; supplied RGB image has {expected}. " +
                "Check model dimensions and/or preprocessing.");

        fixed (byte* data = rgb)
        {
            RknnInput input = new()
            {
                Index = 0,
                Buffer = data,
                Size = checked((uint)rgb.Length),
                PassThrough = 0, // RKNN may convert NHWC/uint8 to the model's own input layout.
                Type = RknnTensorType.UInt8,
                Format = RknnTensorFormat.Nhwc
            };

            RknnNative.Check(RknnNative.InputsSet(_context, 1, &input), "rknn_inputs_set");
            RknnNative.Check(RknnNative.Run(_context, 0), "rknn_run");

            int n = OutputInfos.Count;
            RknnOutput* outputs = stackalloc RknnOutput[n];
            for (int i = 0; i < n; i++)
            {
                outputs[i] = new RknnOutput
                {
                    WantFloat = 1, // dequantize INT8/UINT8 outputs to float32
                    IsPrealloc = 0, // runtime allocates, we release below
                    Index = (uint)i
                };
            }

            RknnNative.Check(RknnNative.OutputsGet(_context, (uint)n, outputs, 0), "rknn_outputs_get");
            try
            {
                var result = new RknnTensorOutput[n];
                for (int i = 0; i < n; i++)
                {
                    RknnOutput o = outputs[i];
                    if (o.Buffer == null || o.Size % sizeof(float) != 0)
                        throw new InvalidOperationException($"Invalid float output buffer {i}.");

                    int elementCount = checked((int)(o.Size / sizeof(float)));
                    var values = new float[elementCount];
                    new ReadOnlySpan<float>(o.Buffer, elementCount).CopyTo(values);
                    result[i] = new RknnTensorOutput(OutputInfos[i], values);
                }
                return result;
            }
            finally
            {
                RknnNative.Check(RknnNative.OutputsRelease(_context, (uint)n, outputs),
                    "rknn_outputs_release");
            }
        }
    }

    private RknnTensorInfo QueryTensor(int queryCode, uint index)
    {
        RknnTensorAttr attr = default;
        attr.Index = index;
        RknnNative.Check(RknnNative.Query(_context, queryCode, &attr,
            (uint)sizeof(RknnTensorAttr)), $"rknn_query(tensor {index})");

        return new RknnTensorInfo(
            (int)index, attr.GetName(), attr.GetDimensions(),
            attr.Type.ToString(), attr.Format.ToString(),
            checked((int)attr.NElements), attr.ZeroPoint, attr.Scale);
    }

    private static void VerifyAbi()
    {
        // Checked against rknn_api.h on AArch64 / LP64.
        if (sizeof(RknnInput) != 32 || sizeof(RknnOutput) != 24 ||
            sizeof(RknnTensorAttr) != 376)
            throw new InvalidOperationException(
                $"Unexpected ABI sizes: input={sizeof(RknnInput)} " +
                $"output={sizeof(RknnOutput)} attr={sizeof(RknnTensorAttr)}");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_context != 0)
        {
            RknnNative.Destroy(_context);
            _context = 0;
        }
    }
}
