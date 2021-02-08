package aspen;

import aspen.AppModel;

public class Aspen
{
    // ------------------------------------------------------------------------
    // public methods
    // ------------------------------------------------------------------------
    public static AppModel LoadAppModel(String filename)
    {
        // this is a new object; we can let Java own it
        AppModel m = new AppModel(nativeLoadAppModel(filename), true);
        return m;
    }

    public static MachModel LoadMachineModel(String filename)
    {
        // this is a new object; we can let Java own it
        MachModel m = new MachModel(nativeLoadMachineModel(filename), true);
        return m;
    }

    // ------------------------------------------------------------------------
    // implementation details
    // ------------------------------------------------------------------------
    private static native long nativeLoadAppModel(String filename);
    private static native long nativeLoadMachineModel(String filename);

    // initialization before use
    static
    {
        Aspen.Initialize();
    }

    // before use, must load native library
    public static void Initialize()
    {
        System.loadLibrary("aspenjni");
    }

}
