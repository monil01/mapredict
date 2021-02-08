package aspen;

public class MachModel
{
    // ------------------------------------------------------------------------
    // public methods
    // ------------------------------------------------------------------------
    public MachComponent GetMachine()
    {
        // this is an object within the machine model; don't let java free it
        return new MachComponent(nativeGetMachine(this.native_ptr), false);
    }

    // ------------------------------------------------------------------------
    // implementation details
    // ------------------------------------------------------------------------
    private long native_ptr = 0;
    private boolean can_delete = false;
    private native void nativeFinalize(long obj);
    private native long nativeGetMachine(long obj);

    // initialization before use; load native library
    static
    {
        Aspen.Initialize();
    }

    // finalize must know whether or not to free native object
    protected void finalize()
    {
        if (this.can_delete)
            nativeFinalize(this.native_ptr);
    }

    // construct a java object from native object, set ownership
    protected MachModel(long np, boolean cd)
    {
        this.native_ptr = np;
        this.can_delete = cd;
    }
}
