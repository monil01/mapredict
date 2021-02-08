package aspen;

public class Expression
{
    // ------------------------------------------------------------------------
    // public methods
    // ------------------------------------------------------------------------
    public String GetText()
    {
        return nativeGetText(this.native_ptr);
    }

    // ------------------------------------------------------------------------
    // implementation details
    // ------------------------------------------------------------------------
    private long native_ptr = 0;
    private boolean can_delete = false;
    private native void nativeFinalize(long obj);
    private native String nativeGetText(long obj);

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
    protected Expression(long np, boolean cd)
    {
        this.native_ptr = np;
        this.can_delete = cd;
    }
}
