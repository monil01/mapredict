package org.jllvm;

import org.jllvm.LLVMModule.Hashable_LLVMOpaqueModule;
import org.jllvm.LLVMType.Hashable_LLVMOpaqueType;
import org.jllvm.LLVMValue.Hashable_LLVMOpaqueValue;
import org.jllvm.bindings.Core;
import org.jllvm.bindings.SWIGTYPE_p_LLVMOpaqueContext;

import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;

/**
 * <a href=
 * "http://llvm.org/docs/ProgrammersManual.html#achieving-isolation-with-llvmcontext"
 * >LLVM documentation explains LLVM contexts</a> and their usage rules.
 * 
 * <p>
 * As for LLVM contexts, each of jllvm's {@link LLVMContext} objects should be
 * accessed by a single thread at a time, but different {@link LLVMContext}
 * objects can be accessed by different threads.
 * </p>
 * 
 * <p>
 * We avoid the global context in jllvm. When the API exists for using the
 * global context, it's easy to use it accidentally when you intend to specify
 * a different context, and then it's difficult to find these accidental uses,
 * which typically produce racy LLVM failures. Again, avoiding the global
 * context is important if you might run LLVM concurrently in multiple
 * threads. One case where running LLVM concurrently in multiple threads is
 * very likely but perhaps unexpected in Java is when you define a
 * {@code finalize} method to clean up LLVM memory. For example, let's say you
 * have a test suite that runs multiple test cases serially all in a single
 * thread, and let's say each test case runs LLVM once, but let's say you
 * leave cleanup from each test case to a {@code finalize} method, which the
 * JVM runs in a separate thread. If each test case uses a separate context,
 * the potential overlap between the {@code finalize} for one test case and
 * the start of the next test case is safe, and the {@code finalize} method
 * can easily perform cleanup merely by calling {@link #dispose} on its
 * context.
 * </p>
 * 
 * <p>
 * All jllvm objects of type {@link LLVMContext} are stored in a cache
 * ({@link #context_cache}). Unlike many other jllvm types whose objects are
 * cached, {@link LLVMContext} has several fields (described below) besides
 * the LLVM object address, and it would probably be impossible to examine a
 * new LLVM object returned by LLVM to assess whether a previously cached
 * jllvm object and its fields are appropriate for it. In caches for other
 * kinds of jllvm objects, that would be a problem, as explained in
 * {@link LLVMValue}'s documentation. Fortunately, the LLVM object underlying
 * an {@link LLVMContext} object is never owned by another LLVM object and
 * thus never disposed of without an explicit call from {@link #dispose}, which
 * purges the old jllvm object from the cache so that it is never necessary to
 * perform this assessment. In caches for other kinds of jllvm objects,
 * disposal is not so easily controlled in jllvm.
 * </p>
 * 
 * <p>
 * The aforementioned fields of {@link LLVMContext} are actually the
 * aforementioned caches for other jllvm objects. Specifically, they store the
 * {@link LLVMModule}, {@link LLVMType}, and {@link LLVMValue} objects whose
 * underlying LLVM objects are associated with this {@link LLVMContext}
 * object's underlying LLVM object. When {@link #dispose} is called, the
 * underlying LLVM objects are freed (except that an {@link LLVMModule}'s LLVM
 * object might have become owned by an {@link LLVMExecutionEngine}'s LLVM
 * object and so might have already been disposed along with it instead), and
 * so those caches become stale, but those caches should never manage to be
 * used after {@link #dispose} anyway. When the {@link LLVMContext} object is
 * garbage-collected, those caches can then be garbage-collected. However, an
 * {@link LLVMContext} object is not garbage-collected until {@link #dispose}
 * is called for it because it remains in the cache of {@link LLVMContext}
 * objects ({@link #context_cache}) until then. Thus, in all cases, we don't
 * have to worry that a cache will be garbage-collected before the LLVM
 * objects underlying the jllvm objects it stores are disposed.
 * </p>
 */
public class LLVMContext {
	/**
	 * Same as {@link SWIGTYPE_p_LLVMOpaqueContext} except instances compare
	 * equal when and only when the underlying LLVM object addresses are equal.
	 * {@link #hashCode} is adjusted accordingly.
	 */
	private static class Hashable_LLVMOpaqueContext
		extends SWIGTYPE_p_LLVMOpaqueContext
	{
		/**
		 * Construct using the LLVM object address contained in an existing
		 * {@link SWIGTYPE_p_LLVMOpaqueContext}.
		 * 
		 * @param o
		 *          contains the LLVM object address to store
		 */
		public Hashable_LLVMOpaqueContext(SWIGTYPE_p_LLVMOpaqueContext o) {
			super(getCPtr(o), false);
		}

		@Override
		public int hashCode() {
			long cPtr = getCPtr(this);
			// This is the hashCode documented for Long.hashCode at
			// http://docs.oracle.com/javase/7/docs/api/java/lang/Long.html.
			// We reproduce the formula here instead of calling Long.hashCode so
			// we don't have to waste time constructing a Long object.
			return (int) (cPtr ^ (cPtr >>> 32));
		}
	
		@Override
		public boolean equals(java.lang.Object obj) {
			if (!(obj instanceof Hashable_LLVMOpaqueContext)) {
				return false;
			}
			return getCPtr(this) == getCPtr((Hashable_LLVMOpaqueContext) obj);
		}
	}
	
	/**
	 * Because different {@link LLVMContext} objects might be accessed by
	 * different threads, such as someone's {@code finalize} method calling
	 * {@link #dispose}, this cache must support concurrency.
	 */
	private static ConcurrentHashMap<Hashable_LLVMOpaqueContext,LLVMContext> context_cache = null;

	final private Hashable_LLVMOpaqueContext instance;
	final private HashMap<Hashable_LLVMOpaqueModule,LLVMModule> module_cache = new HashMap<>();
	final private HashMap<Hashable_LLVMOpaqueType,LLVMType> type_cache = new HashMap<>();
	final private HashMap<Hashable_LLVMOpaqueValue,LLVMValue> value_cache = new HashMap<>();
	
	public LLVMContext() {
		instance = new Hashable_LLVMOpaqueContext(Core.LLVMContextCreate());
		if(context_cache == null)
			context_cache = new ConcurrentHashMap<>();
		context_cache.put(instance,this);
	}
	
	private LLVMContext(Hashable_LLVMOpaqueContext c) {
		instance = c;
		context_cache.put(instance,this);
	}
	
	/**
	 * Get a cached {@link LLVMContext} object.
	 * 
	 * @param c
	 *          contains the address of the LLVM object for which a cached
	 *          {@link LLVMContext} object should be sought
	 * @return the cached {@link LLVMContext} object, or a new
	 *         {@link LLVMContext} object for {@code c} if none has previously
	 *         been cached for {@code c} (probably always the global context,
	 *         which LLVM creates without a call to {@link LLVMContext}'s
	 *         constructor)
	 */
	public static LLVMContext getContext(SWIGTYPE_p_LLVMOpaqueContext c) {
		if(context_cache == null)
			context_cache = new ConcurrentHashMap<Hashable_LLVMOpaqueContext,LLVMContext>();
		Hashable_LLVMOpaqueContext hc = new Hashable_LLVMOpaqueContext(c);
		LLVMContext result = context_cache.get(hc);
		if(result == null) {
			result = new LLVMContext(hc);
		}
		assert(result != null);
		return result;
	}

	public SWIGTYPE_p_LLVMOpaqueContext getInstance() {
		return instance;
	}
	
	public long getMetadataKindID(String name) {
		return Core.LLVMGetMDKindIDInContext(instance,name,name.length());
	}
	
	// Avoid using global context: see LLVMContext documentation.
	//public static LLVMContext getGlobalContext() {
	//	return getContext(Core.LLVMGetGlobalContext());
	//}
	
	/** Intended to be used only by {@link LLVMModule}. */
	public HashMap<Hashable_LLVMOpaqueModule,LLVMModule> getModuleCache() {
		return module_cache;
	}
	
	/** Intended to be used only by {@link LLVMType}. */
	public HashMap<Hashable_LLVMOpaqueType,LLVMType> getTypeCache() {
		return type_cache;
	}
	
	/** Intended to be used only by {@link LLVMValue}. */
	public HashMap<Hashable_LLVMOpaqueValue,LLVMValue> getValueCache() {
		return value_cache;
	}
	
	/**
	 * This automatically disposes of any associated {@link LLVMModule}s that
	 * haven't already been disposed of, but be sure to dispose of any
	 * {@link LLVMExecutionEngine}s for those {@link LLVMModule}s first or there
	 * may be LLVM failures when you do dispose of those
	 * {@link LLVMExecutionEngine}s.
	 */
	public void dispose() {
		// Remove the jllvm object from the cache before disposing the LLVM object
		// so that, if LLVM running in another thread reuses the LLVM object's
		// address for a new LLVM context object, this jllvm object is
		// guaranteed not to be fetched from the cache for it. This jllvm object's
		// fields would rarely be appropriate for it.
		context_cache.remove(instance);
		Core.LLVMContextDispose(instance);
	}
}
