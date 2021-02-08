package aspen;

import aspen.Expression;

public class AppModel
{
    // ------------------------------------------------------------------------
    // public methods
    // ------------------------------------------------------------------------
    public String GetName()
    {
        return nativeGetName(this.native_ptr);
    }
    public Expression GetResourceRequirementExpression(String res)
    {
        // this is a new expression; we can let Java own it
        return new Expression(
            nativeGetResourceRequirementExpression(this.native_ptr, res),
            true);
    }

    // ------------------------------------------------------------------------
    // implementation details
    // ------------------------------------------------------------------------
    private long native_ptr = 0;
    private boolean can_delete = false;
    private native void nativeFinalize(long obj);
    private native String nativeGetName(long obj);
    private native long nativeGetResourceRequirementExpression(long obj, String res);

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
    protected AppModel(long np, boolean cd)
    {
        this.native_ptr = np;
        this.can_delete = cd;
    }
}
