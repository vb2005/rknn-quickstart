using System.Runtime.InteropServices;
using System.Text;

namespace RknnPInvokeNet9;

// Based on rknpu2/runtime/Linux/librknn_api/include/rknn_api.h.
// ABI: Linux AArch64, 64-bit pointers; rknn_context is uint64_t.
internal static unsafe class RknnNative
{
    private const string Library = "librknnrt.so";

    internal const int QueryInOutNum = 0;
    internal const int QueryInputAttr = 1;
    internal const int QueryOutputAttr = 2;
    internal const int QuerySdkVersion = 5;

    internal const int NpuCore0_1_2 = 7; // 1 | 2 | 4

    [DllImport(Library, EntryPoint = "rknn_init", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Init(
        out ulong context,
        [MarshalAs(UnmanagedType.LPUTF8Str)] string modelPath,
        uint modelSize, uint flags, nint extend);

    [DllImport(Library, EntryPoint = "rknn_destroy", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Destroy(ulong context);

    [DllImport(Library, EntryPoint = "rknn_set_core_mask", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int SetCoreMask(ulong context, int coreMask);

    [DllImport(Library, EntryPoint = "rknn_query", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Query(ulong context, int command, void* result, uint resultSize);

    [DllImport(Library, EntryPoint = "rknn_inputs_set", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int InputsSet(ulong context, uint count, RknnInput* inputs);

    [DllImport(Library, EntryPoint = "rknn_run", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int Run(ulong context, nint extend);

    [DllImport(Library, EntryPoint = "rknn_outputs_get", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int OutputsGet(ulong context, uint count, RknnOutput* outputs, nint extend);

    [DllImport(Library, EntryPoint = "rknn_outputs_release", CallingConvention = CallingConvention.Cdecl)]
    internal static extern int OutputsRelease(ulong context, uint count, RknnOutput* outputs);

    internal static void Check(int code, string operation)
    {
        if (code != 0)
            throw new InvalidOperationException($"{operation} failed (RKNN code {code}).");
    }
}

internal enum RknnTensorFormat : int
{
    Nchw = 0,
    Nhwc = 1,
    Nc1Hwc2 = 2,
    Undefined = 3
}

internal enum RknnTensorType : int
{
    Float32 = 0,
    Float16 = 1,
    Int8 = 2,
    UInt8 = 3,
    Int16 = 4,
    UInt16 = 5,
    Int32 = 6,
    UInt32 = 7,
    Int64 = 8,
    Bool = 9,
    Int4 = 10,
    BFloat16 = 11
}

[StructLayout(LayoutKind.Sequential)]
internal struct RknnInputOutputNum
{
    public uint InputCount;
    public uint OutputCount;
}

// Crucial: no automatic C# bool marshaling (native uint8_t is exactly 1 byte).
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct RknnInput
{
    public uint Index;
    public void* Buffer;
    public uint Size;
    public byte PassThrough;
    public RknnTensorType Type;
    public RknnTensorFormat Format;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct RknnOutput
{
    public byte WantFloat;
    public byte IsPrealloc;
    public uint Index;
    public void* Buffer;
    public uint Size;
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct RknnTensorAttr
{
    public uint Index;
    public uint NDimensions;
    public fixed uint Dimensions[16];
    public fixed byte Name[256];
    public uint NElements;
    public uint Size;
    public RknnTensorFormat Format;
    public RknnTensorType Type;
    public int QuantizationType;
    public sbyte FractionalLength;
    public int ZeroPoint;
    public float Scale;
    public uint WidthStride;
    public uint SizeWithStride;
    public byte PassThrough;
    public uint HeightStride;

    public int[] GetDimensions()
    {
        int n = checked((int)NDimensions);
        if (n > 16) throw new InvalidOperationException($"Invalid n_dims={n}.");
        var result = new int[n];
        fixed (uint* p = Dimensions)
        {
            for (int i = 0; i < n; i++)
                result[i] = checked((int)p[i]);
        }
        return result;
    }

    public string GetName()
    {
        fixed (byte* ptr = Name)
            return RknnUtf8.Decode(ptr, 256);
    }
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct RknnSdkVersion
{
    public fixed byte ApiVersion[256];
    public fixed byte DriverVersion[256];

    public string GetApiVersion()
    {
        fixed (byte* p = ApiVersion)
            return RknnUtf8.Decode(p, 256);
    }

    public string GetDriverVersion()
    {
        fixed (byte* p = DriverVersion)
            return RknnUtf8.Decode(p, 256);
    }
}

internal static unsafe class RknnUtf8
{
    internal static string Decode(byte* ptr, int limit)
    {
        int size = 0;
        while (size < limit && ptr[size] != 0) size++;
        return Encoding.UTF8.GetString(new ReadOnlySpan<byte>(ptr, size));
    }
}
