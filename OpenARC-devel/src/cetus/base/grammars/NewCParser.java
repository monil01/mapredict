// $ANTLR 2.7.7 (2010-12-23): "NewCParser.g" -> "NewCParser.java"$

package cetus.base.grammars;

import antlr.TokenBuffer;
import antlr.TokenStreamException;
import antlr.TokenStreamIOException;
import antlr.ANTLRException;
import antlr.LLkParser;
import antlr.Token;
import antlr.TokenStream;
import antlr.RecognitionException;
import antlr.NoViableAltException;
import antlr.MismatchedTokenException;
import antlr.SemanticException;
import antlr.ParserSharedInputState;
import antlr.collections.impl.BitSet;

import java.io.*;
import antlr.CommonAST;
import antlr.DumpASTVisitor;
import java.util.*;
import cetus.hir.*;
import openacc.hir.CUDASpecifier;
import openacc.hir.OpenCLSpecifier;
import static cetus.hir.AttributeSpecifier.Attribute;
@SuppressWarnings({"unchecked", "cast"})

public class NewCParser extends antlr.LLkParser       implements NEWCTokenTypes
 {

Expression baseEnum = null,curEnum = null;
NewCLexer curLexer=null;
boolean isFuncDef = false;
boolean isExtern = false;
PreprocessorInfoChannel preprocessorInfoChannel = null;
SymbolTable symtab = null;
CompoundStatement curr_cstmt = null;
boolean hastypedef = false;
HashMap typetable = null;
List currproc = new ArrayList();
Declaration prev_decl = null;
boolean old_style_func = false;
HashMap func_decl_list = new HashMap();

public void getPreprocessorInfoChannel(PreprocessorInfoChannel preprocChannel)
{
  preprocessorInfoChannel = preprocChannel;
}

public void setLexer(NewCLexer lexer)
{
  curLexer=lexer;
  curLexer.setParser(this);
}

public NewCLexer getLexer()
{
  return curLexer;
}

/*
 * Retreive all buffered pragmas and comments up to this
 * token number
 */
public List getPragma(int a)
{
  return
      preprocessorInfoChannel.extractLinesPrecedingTokenNumber(new Integer(a));
}

/*
 * Add pragmas and line directives as PreAnnotation
 */
public void putPragma(Token sline, SymbolTable sym)
{
  List v  = null;
  // Get token number and get all buffered information
  v = getPragma(((CToken)sline).getTokenNumber());

  // Go though the list of pragmas and comments
  Pragma p = null;
  PreAnnotation anote = null;
  int vsize = v.size();
  for (int i = 0; i < vsize; i++) {
    p = (Pragma)v.get(i);
    anote = new PreAnnotation(p.str);
    anote.setPrintMethod(PreAnnotation.print_raw_method);
    // Add PreAnnotation as Statement in a block
    if (sym instanceof CompoundStatement)
      ((CompoundStatement)sym).addStatement(new DeclarationStatement(anote));
    // Add PreAnnotation in other SymbolTables
    else
      sym.addDeclaration(anote);
  }
}

// Suppport C++-style single-line comments?
public static boolean CPPComments = true;
public Stack symtabstack = new Stack();
public Stack typestack = new Stack();

public void enterSymtab(SymbolTable curr_symtab)
{
  symtabstack.push(symtab);
  typetable = new HashMap();
  typestack.push(typetable);
  symtab = curr_symtab;
}

public void exitSymtab()
{
  Object o = symtabstack.pop();
  if (o != null) {
    typestack.pop();
    typetable = (HashMap)(typestack.peek());
    symtab = (SymbolTable)o;
  }
}

public boolean isTypedefName(String name)
{
  //System.err.println("Typename "+name);
  int n = typestack.size()-1;
  Object d = null;
  while(n>=0) {
    d = ((HashMap)(typestack.get(n))).get(name);
    if (d != null )
      return true;
    n--;
  }
  if (name.equals("__builtin_va_list"))
    return true;
  // [NVL support added by Joel E. Denny]
  if (name.equals("__builtin_nvl_heap"))
    return true;

  //System.err.println("Typename "+name+" not found");
  return false;
}

int traceDepth = 0;

public void reportError(RecognitionException ex)
{
  try {
    System.err.println("Cetus Parsing Error: " + "Exception Type: "
        + ex.getClass().getName());
    System.err.println("Source: " + getLexer().lineObject.getSource()
        + " Line:" + ex.getLine() + " Column: " + ex.getColumn()
        + " token name:" + tokenNames[LA(1)]);
    ex.printStackTrace(System.err);
    Tools.exit(1);
  } catch (TokenStreamException e) {
    System.err.println("Cetus Parsing Error: "+ex);
    ex.printStackTrace(System.err);
    Tools.exit(1);
  }
}

public void reportError(String s)
{
  System.err.println("Cetus Parsing Error from String: " + s);
}

public void reportWarning(String s)
{
  System.err.println("Cetus Parsing Warning from String: " + s);
}

public void match(int t) throws MismatchedTokenException
{
  boolean debugging = false;
  if ( debugging ) {
    for (int x=0; x<traceDepth; x++)
      System.out.print(" ");
    try {
      System.out.println("Match(" + tokenNames[t] + ") with LA(1)="
          + tokenNames[LA(1)] + ((inputState.guessing>0)?
          " [inputState.guessing " + inputState.guessing + "]":""));
    } catch (TokenStreamException e) {
      System.out.println("Match("+tokenNames[t]+") "
          + ((inputState.guessing>0)?
          " [inputState.guessing "+ inputState.guessing + "]":""));
    }
  }
  try {
    if ( LA(1)!=t ) {
      if ( debugging ) {
        for (int x=0; x<traceDepth; x++)
          System.out.print(" ");
        System.out.println("token mismatch: "+tokenNames[LA(1)]
            + "!=" + tokenNames[t]);
      }
      throw new MismatchedTokenException
          (tokenNames, LT(1), t, false, getFilename());
    } else {
      // mark token as consumed -- fetch next token deferred until LA/LT
      consume();
    }
  } catch (TokenStreamException e) {
  }
}

public void traceIn(String rname)
{
  traceDepth += 1;
  for (int x=0; x<traceDepth; x++)
    System.out.print(" ");
  try {
    System.out.println("> "+rname+"; LA(1)==("+ tokenNames[LT(1).getType()]
        + ") " + LT(1).getText() + " [inputState.guessing "
        + inputState.guessing + "]");
  } catch (TokenStreamException e) {
  }
}

public void traceOut(String rname)
{
  for (int x=0; x<traceDepth; x++)
    System.out.print(" ");
  try {
    System.out.println("< "+rname+"; LA(1)==("+ tokenNames[LT(1).getType()]
        + ") " + LT(1).getText() + " [inputState.guessing "
        + inputState.guessing + "]");
  } catch (TokenStreamException e) {
  }
  traceDepth -= 1;
}

/* Normalizes switch statement by removing unncessary compound statement */
private void unwrapSwitch(SwitchStatement swstmt) {
    List<CompoundStatement> cstmts = (new DFIterator<CompoundStatement>(
            swstmt, CompoundStatement.class)).getList();
    cstmts.remove(0);
    for (int i = cstmts.size()-1; i >= 0; i--) {
        CompoundStatement cstmt = cstmts.get(i);
        if (cstmt.getParent() instanceof CompoundStatement &&
            !IRTools.containsClass(cstmt, VariableDeclaration.class)) {
            CompoundStatement parent = (CompoundStatement)cstmt.getParent();
            List<Traversable> children = cstmt.getChildren();
            while (!children.isEmpty()) {
                Statement child = (Statement)children.get(0);
                child.detach();
                parent.addStatementBefore(cstmt, child);
            }
            cstmt.detach();
        }
    }
}


protected NewCParser(TokenBuffer tokenBuf, int k) {
  super(tokenBuf,k);
  tokenNames = _tokenNames;
}

public NewCParser(TokenBuffer tokenBuf) {
  this(tokenBuf,2);
}

protected NewCParser(TokenStream lexer, int k) {
  super(lexer,k);
  tokenNames = _tokenNames;
}

public NewCParser(TokenStream lexer) {
  this(lexer,2);
}

public NewCParser(ParserSharedInputState state) {
  super(state,2);
  tokenNames = _tokenNames;
}

	public final TranslationUnit  translationUnit(
		TranslationUnit init_tunit
	) throws RecognitionException, TokenStreamException {
		TranslationUnit tunit;
		
		
		/* build a new Translation Unit */
		if (init_tunit == null)
		tunit = new TranslationUnit(getLexer().originalSource);
		else
		tunit = init_tunit;
		enterSymtab(tunit);
		
		
		try {      // for error handling
			{
			switch ( LA(1)) {
			case LITERAL_typedef:
			case SEMI:
			case LITERAL_asm:
			case LITERAL_volatile:
			case LITERAL_struct:
			case LITERAL_union:
			case LITERAL_enum:
			case LITERAL_auto:
			case LITERAL_register:
			case LITERAL___thread:
			case LITERAL_extern:
			case LITERAL_static:
			case LITERAL_inline:
			case LITERAL_const:
			case LITERAL_restrict:
			case LITERAL___nvl__:
			case LITERAL___nvl_wp__:
			case LITERAL_void:
			case LITERAL_char:
			case LITERAL_short:
			case LITERAL_int:
			case LITERAL_long:
			case LITERAL_float:
			case LITERAL_double:
			case LITERAL_signed:
			case LITERAL_unsigned:
			case LITERAL__Bool:
			case LITERAL__Complex:
			case LITERAL__Imaginary:
			case 36:
			case 37:
			case 38:
			case 39:
			case 40:
			case 41:
			case 42:
			case 43:
			case LITERAL_size_t:
			case 45:
			case 46:
			case 47:
			case 48:
			case 49:
			case LITERAL___global__:
			case LITERAL___shared__:
			case LITERAL___host__:
			case LITERAL___device__:
			case LITERAL___constant__:
			case LITERAL___noinline__:
			case LITERAL___kernel:
			case LITERAL___global:
			case LITERAL___local:
			case LITERAL___constant:
			case LITERAL_typeof:
			case LPAREN:
			case LITERAL___complex:
			case ID:
			case LITERAL___declspec:
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			case STAR:
			{
				externalList(tunit);
				break;
			}
			case EOF:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				exitSymtab();
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
		return tunit;
	}
	
	public final void externalList(
		TranslationUnit tunit
	) throws RecognitionException, TokenStreamException {
		
		
		try {      // for error handling
			{
			int _cnt5=0;
			_loop5:
			do {
				if ((_tokenSet_1.member(LA(1)))) {
					externalDef(tunit);
				}
				else {
					if ( _cnt5>=1 ) { break _loop5; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt5++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
	}
	
	public final void externalDef(
		TranslationUnit tunit
	) throws RecognitionException, TokenStreamException {
		
		Token  esemi = null;
		Declaration decl = null;
		
		try {      // for error handling
			switch ( LA(1)) {
			case LITERAL_asm:
			{
				asm_expr();
				break;
			}
			case SEMI:
			{
				esemi = LT(1);
				match(SEMI);
				if ( inputState.guessing==0 ) {
					putPragma(esemi,symtab);
				}
				break;
			}
			default:
				boolean synPredMatched8 = false;
				if (((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2))))) {
					int _m8 = mark();
					synPredMatched8 = true;
					inputState.guessing++;
					try {
						{
						if ((LA(1)==LITERAL_typedef) && (true)) {
							match(LITERAL_typedef);
						}
						else if ((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2)))) {
							declaration();
						}
						else {
							throw new NoViableAltException(LT(1), getFilename());
						}
						
						}
					}
					catch (RecognitionException pe) {
						synPredMatched8 = false;
					}
					rewind(_m8);
inputState.guessing--;
				}
				if ( synPredMatched8 ) {
					decl=declaration();
					if ( inputState.guessing==0 ) {
						
						if (decl != null) {
						//PrintTools.printStatus("Adding Declaration: ",3);
						//PrintTools.printlnStatus(decl,3);
						tunit.addDeclaration(decl);
						}
						
					}
				}
				else {
					boolean synPredMatched10 = false;
					if (((_tokenSet_4.member(LA(1))) && (_tokenSet_5.member(LA(2))))) {
						int _m10 = mark();
						synPredMatched10 = true;
						inputState.guessing++;
						try {
							{
							functionPrefix();
							}
						}
						catch (RecognitionException pe) {
							synPredMatched10 = false;
						}
						rewind(_m10);
inputState.guessing--;
					}
					if ( synPredMatched10 ) {
						decl=functionDef();
						if ( inputState.guessing==0 ) {
							
							//PrintTools.printStatus("Adding Declaration: ",3);
							//PrintTools.printlnStatus(decl,3);
							tunit.addDeclaration(decl);
							
						}
					}
					else if ((_tokenSet_6.member(LA(1))) && (_tokenSet_7.member(LA(2)))) {
						decl=typelessDeclaration();
						if ( inputState.guessing==0 ) {
							
							//PrintTools.printStatus("Adding Declaration: ",3);
							//PrintTools.printlnStatus(decl,3);
							tunit.addDeclaration(decl);
							
						}
					}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				}}
			}
			catch (RecognitionException ex) {
				if (inputState.guessing==0) {
					reportError(ex);
					recover(ex,_tokenSet_8);
				} else {
				  throw ex;
				}
			}
		}
		
	public final Declaration  declaration() throws RecognitionException, TokenStreamException {
		Declaration bdecl;
		
		Token  dsemi = null;
		bdecl=null; List dspec=null; List idlist=null;
		
		try {      // for error handling
			dspec=declSpecifiers();
			{
			switch ( LA(1)) {
			case LPAREN:
			case ID:
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			case STAR:
			{
				idlist=initDeclList();
				break;
			}
			case SEMI:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				
				if (idlist != null) {
				if (old_style_func) {
				Declarator d  = null;
				Declaration newdecl = null;
				int idlist_size = idlist.size();
				for (int i = 0; i < idlist_size; i++) {
				d = (Declarator)idlist.get(i);
				newdecl = new VariableDeclaration(dspec,d);
				func_decl_list.put(d.getID().toString(),newdecl);
				}
				bdecl = null;
				} else
				bdecl = new VariableDeclaration(dspec,idlist);
				prev_decl = null;
				} else {
				// Looks like a forward declaration
				if (prev_decl != null) {
				bdecl = prev_decl;
				prev_decl = null;
				}
				}
				hastypedef = false;
				
			}
			{
			int _cnt28=0;
			_loop28:
			do {
				if ((LA(1)==SEMI) && (_tokenSet_9.member(LA(2)))) {
					dsemi = LT(1);
					match(SEMI);
				}
				else {
					if ( _cnt28>=1 ) { break _loop28; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt28++;
			} while (true);
			}
			if ( inputState.guessing==0 ) {
				
				int sline = 0;
				sline = dsemi.getLine();
				putPragma(dsemi,symtab);
				hastypedef = false;
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_9);
			} else {
			  throw ex;
			}
		}
		return bdecl;
	}
	
	public final void functionPrefix() throws RecognitionException, TokenStreamException {
		
		Token  flcurly = null;
		Declarator decl = null;
		
		try {      // for error handling
			{
			boolean synPredMatched14 = false;
			if (((_tokenSet_10.member(LA(1))) && (_tokenSet_11.member(LA(2))))) {
				int _m14 = mark();
				synPredMatched14 = true;
				inputState.guessing++;
				try {
					{
					functionDeclSpecifiers();
					}
				}
				catch (RecognitionException pe) {
					synPredMatched14 = false;
				}
				rewind(_m14);
inputState.guessing--;
			}
			if ( synPredMatched14 ) {
				functionDeclSpecifiers();
			}
			else if ((_tokenSet_6.member(LA(1))) && (_tokenSet_12.member(LA(2)))) {
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			decl=declarator();
			{
			_loop16:
			do {
				if ((_tokenSet_2.member(LA(1)))) {
					declaration();
				}
				else {
					break _loop16;
				}
				
			} while (true);
			}
			{
			switch ( LA(1)) {
			case VARARGS:
			{
				match(VARARGS);
				break;
			}
			case SEMI:
			case LCURLY:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			_loop19:
			do {
				if ((LA(1)==SEMI)) {
					match(SEMI);
				}
				else {
					break _loop19;
				}
				
			} while (true);
			}
			flcurly = LT(1);
			match(LCURLY);
			if ( inputState.guessing==0 ) {
				putPragma(flcurly,symtab);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
	}
	
	public final Procedure  functionDef() throws RecognitionException, TokenStreamException {
		Procedure curFunc;
		
		
		CompoundStatement stmt=null;
		Declaration decl=null;
		Declarator bdecl=null;
		List dspec=null;
		curFunc = null;
		String declName=null;
		int dcount = 0;
		SymbolTable prev_symtab =null;
		SymbolTable temp_symtab = new CompoundStatement();
		
		
		try {      // for error handling
			{
			boolean synPredMatched209 = false;
			if (((_tokenSet_10.member(LA(1))) && (_tokenSet_11.member(LA(2))))) {
				int _m209 = mark();
				synPredMatched209 = true;
				inputState.guessing++;
				try {
					{
					functionDeclSpecifiers();
					}
				}
				catch (RecognitionException pe) {
					synPredMatched209 = false;
				}
				rewind(_m209);
inputState.guessing--;
			}
			if ( synPredMatched209 ) {
				if ( inputState.guessing==0 ) {
					isFuncDef = true;
				}
				dspec=functionDeclSpecifiers();
			}
			else if ((_tokenSet_6.member(LA(1))) && (_tokenSet_12.member(LA(2)))) {
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			if ( inputState.guessing==0 ) {
				if (dspec == null) dspec = new ArrayList();
			}
			bdecl=declarator();
			if ( inputState.guessing==0 ) {
				enterSymtab(temp_symtab); old_style_func = true; func_decl_list.clear();
			}
			{
			_loop211:
			do {
				if ((_tokenSet_2.member(LA(1)))) {
					declaration();
					if ( inputState.guessing==0 ) {
						dcount++;
					}
				}
				else {
					break _loop211;
				}
				
			} while (true);
			}
			{
			switch ( LA(1)) {
			case VARARGS:
			{
				match(VARARGS);
				break;
			}
			case SEMI:
			case LCURLY:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			_loop214:
			do {
				if ((LA(1)==SEMI)) {
					match(SEMI);
				}
				else {
					break _loop214;
				}
				
			} while (true);
			}
			if ( inputState.guessing==0 ) {
				
				old_style_func = false;
				exitSymtab();
				isFuncDef = false;
				if (dcount > 0) {
				HashMap hm = null;
				NameID name = null;
				Declaration tdecl = null;
				/**
				*  This implementation is not so good since it relies on
				* the fact that function parameter starts from the second
				*  children and getChildren returns a reference to the
				* actual internal list
				*/
				List<Traversable> bdecl_children = bdecl.getChildren();
				int bdecl_size = bdecl_children.size();
				for (int i = 1; i < bdecl_size; i++) {
				VariableDeclaration vdec = (VariableDeclaration)bdecl_children.get(i);
				List decl_ids = vdec.getDeclaredIDs();
				int decl_ids_size = decl_ids.size();
				for (int j = 0; j < decl_ids_size; j++) {
				// declarator name
				name = (NameID)decl_ids.get(j);
				// find matching Declaration
				tdecl = (Declaration)(func_decl_list.get(name.toString()));
				if (tdecl == null) {
				PrintTools.printlnStatus("cannot find symbol " + name
				+ "in old style function declaration, now assuming an int",1);
				tdecl = new VariableDeclaration(
				Specifier.INT, new VariableDeclarator(name.clone()));
				}
				bdecl_children.set(i, tdecl);
				tdecl.setParent(bdecl);
				}
				}
				Iterator diter = temp_symtab.getDeclarations().iterator();
				Object tobject = null;
				while (diter.hasNext()) {
				tobject = diter.next();
				if (tobject instanceof PreAnnotation)
				symtab.addDeclaration((Declaration)tobject);
				}
				}
				
				
			}
			stmt=compoundStatement();
			if ( inputState.guessing==0 ) {
				
				// support for K&R style declaration: "dcount" is counting the number of
				// declaration in old style.
				// [Extended by Joel E. Denny to collect simple attributes]
				List<Attribute> attrs = new ArrayList<>();
				List dspec_new = new ArrayList();
				for (Object obj : dspec) {
				if (obj instanceof Attribute)
				attrs.add((Attribute)obj);
				else
				dspec_new.add(obj);
				}
				curFunc = new Procedure(dspec_new, bdecl, stmt, dcount>0,
				attrs.isEmpty() ? null : new AttributeSpecifier(attrs));
				PrintTools.printStatus("Creating Procedure: ",1);
				PrintTools.printlnStatus(bdecl,1);
				// already handled in constructor
				currproc.clear();
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_8);
			} else {
			  throw ex;
			}
		}
		return curFunc;
	}
	
	public final Declaration  typelessDeclaration() throws RecognitionException, TokenStreamException {
		Declaration decl;
		
		Token  tdsemi = null;
		decl=null; List idlist=null;
		
		try {      // for error handling
			idlist=initDeclList();
			tdsemi = LT(1);
			match(SEMI);
			if ( inputState.guessing==0 ) {
				putPragma(tdsemi,symtab);
			}
			if ( inputState.guessing==0 ) {
				decl = new VariableDeclaration(new ArrayList(),idlist);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_8);
			} else {
			  throw ex;
			}
		}
		return decl;
	}
	
	public final void asm_expr() throws RecognitionException, TokenStreamException {
		
		Token  asmlcurly = null;
		Expression expr1 = null;
		
		try {      // for error handling
			match(LITERAL_asm);
			{
			switch ( LA(1)) {
			case LITERAL_volatile:
			{
				match(LITERAL_volatile);
				break;
			}
			case LCURLY:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			asmlcurly = LT(1);
			match(LCURLY);
			if ( inputState.guessing==0 ) {
				putPragma(asmlcurly,symtab);
			}
			expr1=expr();
			match(RCURLY);
			{
			int _cnt24=0;
			_loop24:
			do {
				if ((LA(1)==SEMI) && (_tokenSet_8.member(LA(2)))) {
					match(SEMI);
				}
				else {
					if ( _cnt24>=1 ) { break _loop24; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt24++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_8);
			} else {
			  throw ex;
			}
		}
	}
	
	public final List  functionDeclSpecifiers() throws RecognitionException, TokenStreamException {
		List dspec;
		
		
		dspec = new ArrayList();
		Specifier type=null;
		Specifier tqual=null;
		Specifier tspec=null;
		List<Attribute> attrs;
		
		
		try {      // for error handling
			{
			int _cnt219=0;
			_loop219:
			do {
				switch ( LA(1)) {
				case LITERAL_extern:
				case LITERAL_static:
				case LITERAL_inline:
				{
					type=functionStorageClassSpecifier();
					if ( inputState.guessing==0 ) {
						dspec.add(type);
					}
					break;
				}
				case LITERAL___declspec:
				{
					vcDeclSpec();
					break;
				}
				default:
					if ((_tokenSet_13.member(LA(1))) && (_tokenSet_4.member(LA(2)))) {
						tqual=typeQualifier();
						if ( inputState.guessing==0 ) {
							dspec.add(tqual);
						}
					}
					else {
						boolean synPredMatched218 = false;
						if (((_tokenSet_14.member(LA(1))) && (_tokenSet_11.member(LA(2))))) {
							int _m218 = mark();
							synPredMatched218 = true;
							inputState.guessing++;
							try {
								{
								if ((LA(1)==LITERAL_struct) && (true)) {
									match(LITERAL_struct);
								}
								else if ((LA(1)==LITERAL_union) && (true)) {
									match(LITERAL_union);
								}
								else if ((LA(1)==LITERAL_enum) && (true)) {
									match(LITERAL_enum);
								}
								else if ((_tokenSet_14.member(LA(1))) && (true)) {
									tspec=typeSpecifier();
								}
								else {
									throw new NoViableAltException(LT(1), getFilename());
								}
								
								}
							}
							catch (RecognitionException pe) {
								synPredMatched218 = false;
							}
							rewind(_m218);
inputState.guessing--;
						}
						if ( synPredMatched218 ) {
							tspec=typeSpecifier();
							if ( inputState.guessing==0 ) {
								dspec.add(tspec);
							}
						}
						else if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG)) && (LA(2)==LPAREN)) {
							attrs=attributeDecl();
							if ( inputState.guessing==0 ) {
								dspec.addAll(attrs);
							}
						}
					else {
						if ( _cnt219>=1 ) { break _loop219; } else {throw new NoViableAltException(LT(1), getFilename());}
					}
					}}
					_cnt219++;
				} while (true);
				}
			}
			catch (RecognitionException ex) {
				if (inputState.guessing==0) {
					reportError(ex);
					recover(ex,_tokenSet_6);
				} else {
				  throw ex;
				}
			}
			return dspec;
		}
		
	public final Declarator  declarator() throws RecognitionException, TokenStreamException {
		Declarator decl;
		
		Token  id = null;
		
		Expression expr1=null;
		String declName = null;
		decl = null;
		Declarator tdecl = null;
		IDExpression idex = null;
		List plist = null;
		List bp = null;
		Specifier aspec = null;
		boolean isArraySpec = false;
		boolean isNested = false;
		List llist = new ArrayList();
		List tlist = null;
		
		
		try {      // for error handling
			{
			switch ( LA(1)) {
			case STAR:
			{
				bp=pointerGroup();
				break;
			}
			case LPAREN:
			case ID:
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				/* if(bp == null) bp = new LinkedList(); */
			}
			{
			switch ( LA(1)) {
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			{
				attributeDecl();
				break;
			}
			case LPAREN:
			case ID:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			switch ( LA(1)) {
			case ID:
			{
				id = LT(1);
				match(ID);
				if ( inputState.guessing==0 ) {
					
					putPragma(id,symtab);
					declName = id.getText();
					idex = new NameID(declName);
					if(hastypedef) {
					typetable.put(declName,"typedef");
					//System.err.println("Add typedef declname: " + declName); 
					}
					
				}
				break;
			}
			case LPAREN:
			{
				match(LPAREN);
				tdecl=declarator();
				match(RPAREN);
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG)) && (LA(2)==LPAREN)) {
				attributeDecl();
			}
			else if ((_tokenSet_15.member(LA(1))) && (_tokenSet_16.member(LA(2)))) {
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			{
			_loop189:
			do {
				switch ( LA(1)) {
				case LPAREN:
				{
					plist=declaratorParamaterList();
					break;
				}
				case LBRACKET:
				{
					match(LBRACKET);
					{
					switch ( LA(1)) {
					case LPAREN:
					case ID:
					case Number:
					case StringLiteral:
					case LITERAL___asm:
					case STAR:
					case BAND:
					case PLUS:
					case MINUS:
					case LITERAL___builtin_nvl_get_root:
					case LITERAL___builtin_nvl_alloc_nv:
					case INC:
					case DEC:
					case LITERAL_sizeof:
					case LITERAL___alignof__:
					case LITERAL___builtin_va_arg:
					case LITERAL___builtin_offsetof:
					case BNOT:
					case LNOT:
					case LITERAL___real:
					case LITERAL___imag:
					case CharLiteral:
					{
						expr1=expr();
						break;
					}
					case RBRACKET:
					{
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					match(RBRACKET);
					if ( inputState.guessing==0 ) {
						
						isArraySpec = true;
						llist.add(expr1);
						
					}
					break;
				}
				default:
				{
					break _loop189;
				}
				}
			} while (true);
			}
			if ( inputState.guessing==0 ) {
				
				/* Possible combinations []+, () */
				if (plist != null) {
				/* () */
				;
				} else {
				/* []+ */
				if (isArraySpec) {
				aspec = new ArraySpecifier(llist);
				tlist = new ArrayList();
				tlist.add(aspec);
				}
				}
				if (bp == null)
				bp = new ArrayList();
				if (tdecl != null) { // assume tlist == null
				//assert tlist == null : "Assertion (tlist == null) failed 2";
				if (tlist == null)
				tlist = new ArrayList();
				decl = new NestedDeclarator(bp,tdecl,plist,tlist);
				} else {
				if (plist != null) // assume tlist == null
				decl = new ProcedureDeclarator(bp,idex,plist);
				else {
				if (tlist != null)
				decl = new VariableDeclarator(bp,idex,tlist);
				else
				decl = new VariableDeclarator(bp,idex);
				}
				}
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_17);
			} else {
			  throw ex;
			}
		}
		return decl;
	}
	
	public final List  initDeclList() throws RecognitionException, TokenStreamException {
		List dlist;
		
		
		Declarator decl=null;
		dlist = new ArrayList();
		
		
		try {      // for error handling
			decl=initDecl();
			if ( inputState.guessing==0 ) {
				dlist.add(decl);
			}
			{
			_loop153:
			do {
				if ((LA(1)==COMMA) && (_tokenSet_6.member(LA(2)))) {
					match(COMMA);
					decl=initDecl();
					if ( inputState.guessing==0 ) {
						dlist.add(decl);
					}
				}
				else {
					break _loop153;
				}
				
			} while (true);
			}
			{
			switch ( LA(1)) {
			case COMMA:
			{
				match(COMMA);
				break;
			}
			case SEMI:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_18);
			} else {
			  throw ex;
			}
		}
		return dlist;
	}
	
	public final Expression  expr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		Token  c = null;
		
		ret_expr = null;
		Expression expr1=null,expr2=null;
		List elist = new ArrayList();
		
		
		try {      // for error handling
			ret_expr=assignExpr();
			if ( inputState.guessing==0 ) {
				elist.add(ret_expr);
			}
			{
			_loop268:
			do {
				if ((LA(1)==COMMA) && (_tokenSet_19.member(LA(2)))) {
					c = LT(1);
					match(COMMA);
					expr1=assignExpr();
					if ( inputState.guessing==0 ) {
						elist.add(expr1);
					}
				}
				else {
					break _loop268;
				}
				
			} while (true);
			}
			if ( inputState.guessing==0 ) {
				
				if (elist.size() > 1) {
				ret_expr = new CommaExpression(elist);
				}
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_20);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final List  declSpecifiers() throws RecognitionException, TokenStreamException {
		List decls;
		
		decls = new ArrayList(); Specifier spec = null; Specifier temp=null;
		
		try {      // for error handling
			{
			{
			_loop37:
			do {
				switch ( LA(1)) {
				case LITERAL___attribute:
				case LITERAL___asm:
				case LITERAL___OSX_AVAILABLE_STARTING:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
				{
					attributeDecl();
					break;
				}
				case LITERAL___declspec:
				{
					vcDeclSpec();
					break;
				}
				default:
					if ((_tokenSet_21.member(LA(1))) && (_tokenSet_2.member(LA(2)))) {
						spec=storageClassSpecifier();
						if ( inputState.guessing==0 ) {
							decls.add(spec);
						}
					}
					else if ((_tokenSet_13.member(LA(1))) && (_tokenSet_2.member(LA(2)))) {
						spec=typeQualifier();
						if ( inputState.guessing==0 ) {
							decls.add(spec);
						}
					}
				else {
					break _loop37;
				}
				}
			} while (true);
			}
			temp=typeSpecifier();
			if ( inputState.guessing==0 ) {
				decls.add(temp);
			}
			{
			_loop43:
			do {
				if ((_tokenSet_21.member(LA(1))) && (_tokenSet_22.member(LA(2)))) {
					spec=storageClassSpecifier();
					if ( inputState.guessing==0 ) {
						decls.add(spec);
					}
				}
				else if ((_tokenSet_13.member(LA(1))) && (_tokenSet_22.member(LA(2)))) {
					spec=typeQualifier();
					if ( inputState.guessing==0 ) {
						decls.add(spec);
					}
				}
				else {
					boolean synPredMatched42 = false;
					if (((_tokenSet_23.member(LA(1))) && (_tokenSet_24.member(LA(2))))) {
						int _m42 = mark();
						synPredMatched42 = true;
						inputState.guessing++;
						try {
							{
							if ((LA(1)==LITERAL_struct) && (true)) {
								match(LITERAL_struct);
							}
							else if ((LA(1)==LITERAL_union) && (true)) {
								match(LITERAL_union);
							}
							else if ((LA(1)==LITERAL_enum) && (true)) {
								match(LITERAL_enum);
							}
							else if ((_tokenSet_23.member(LA(1))) && (true)) {
								typeSpecifier_noTypeDefName();
							}
							else {
								throw new NoViableAltException(LT(1), getFilename());
							}
							
							}
						}
						catch (RecognitionException pe) {
							synPredMatched42 = false;
						}
						rewind(_m42);
inputState.guessing--;
					}
					if ( synPredMatched42 ) {
						temp=typeSpecifier_noTypeDefName();
						if ( inputState.guessing==0 ) {
							decls.add(temp);
						}
					}
					else if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG)) && (LA(2)==LPAREN)) {
						attributeDecl();
					}
					else if ((LA(1)==LITERAL___declspec)) {
						vcDeclSpec();
					}
					else {
						break _loop43;
					}
					}
				} while (true);
				}
				}
			}
			catch (RecognitionException ex) {
				if (inputState.guessing==0) {
					reportError(ex);
					recover(ex,_tokenSet_25);
				} else {
				  throw ex;
				}
			}
			return decls;
		}
		
	public final List  declSpecifiers_old() throws RecognitionException, TokenStreamException {
		List decls;
		
		decls = new ArrayList(); Specifier spec = null; Specifier temp=null;
		
		try {      // for error handling
			{
			int _cnt33=0;
			_loop33:
			do {
				switch ( LA(1)) {
				case LITERAL___attribute:
				case LITERAL___asm:
				case LITERAL___OSX_AVAILABLE_STARTING:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
				{
					attributeDecl();
					break;
				}
				case LITERAL___declspec:
				{
					vcDeclSpec();
					break;
				}
				default:
					if ((_tokenSet_21.member(LA(1))) && (_tokenSet_26.member(LA(2)))) {
						spec=storageClassSpecifier();
						if ( inputState.guessing==0 ) {
							decls.add(spec);
						}
					}
					else if ((_tokenSet_13.member(LA(1))) && (_tokenSet_26.member(LA(2)))) {
						spec=typeQualifier();
						if ( inputState.guessing==0 ) {
							decls.add(spec);
						}
					}
					else {
						boolean synPredMatched32 = false;
						if (((_tokenSet_14.member(LA(1))) && (_tokenSet_27.member(LA(2))))) {
							int _m32 = mark();
							synPredMatched32 = true;
							inputState.guessing++;
							try {
								{
								if ((LA(1)==LITERAL_struct) && (true)) {
									match(LITERAL_struct);
								}
								else if ((LA(1)==LITERAL_union) && (true)) {
									match(LITERAL_union);
								}
								else if ((LA(1)==LITERAL_enum) && (true)) {
									match(LITERAL_enum);
								}
								else if ((_tokenSet_14.member(LA(1))) && (true)) {
									typeSpecifier();
								}
								else {
									throw new NoViableAltException(LT(1), getFilename());
								}
								
								}
							}
							catch (RecognitionException pe) {
								synPredMatched32 = false;
							}
							rewind(_m32);
inputState.guessing--;
						}
						if ( synPredMatched32 ) {
							temp=typeSpecifier();
							if ( inputState.guessing==0 ) {
								decls.add(temp);
							}
						}
					else {
						if ( _cnt33>=1 ) { break _loop33; } else {throw new NoViableAltException(LT(1), getFilename());}
					}
					}}
					_cnt33++;
				} while (true);
				}
			}
			catch (RecognitionException ex) {
				if (inputState.guessing==0) {
					reportError(ex);
					recover(ex,_tokenSet_0);
				} else {
				  throw ex;
				}
			}
			return decls;
		}
		
/*********************************
 * Specifiers                    *
 *********************************/
	public final Specifier  storageClassSpecifier() throws RecognitionException, TokenStreamException {
		Specifier cspec;
		
		Token  scstypedef = null;
		cspec= null;
		
		try {      // for error handling
			switch ( LA(1)) {
			case LITERAL_auto:
			{
				match(LITERAL_auto);
				if ( inputState.guessing==0 ) {
					cspec = Specifier.AUTO;
				}
				break;
			}
			case LITERAL_register:
			{
				match(LITERAL_register);
				if ( inputState.guessing==0 ) {
					cspec = Specifier.REGISTER;
				}
				break;
			}
			case LITERAL_typedef:
			{
				scstypedef = LT(1);
				match(LITERAL_typedef);
				if ( inputState.guessing==0 ) {
					cspec = Specifier.TYPEDEF; hastypedef = true; putPragma(scstypedef,symtab);
				}
				break;
			}
			case LITERAL_extern:
			case LITERAL_static:
			case LITERAL_inline:
			{
				cspec=functionStorageClassSpecifier();
				break;
			}
			case LITERAL___thread:
			{
				match(LITERAL___thread);
				if ( inputState.guessing==0 ) {
					cspec = Specifier.THREAD;
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_28);
			} else {
			  throw ex;
			}
		}
		return cspec;
	}
	
	public final Specifier  typeQualifier() throws RecognitionException, TokenStreamException {
		Specifier tqual;
		
		tqual=null;
		
		try {      // for error handling
			switch ( LA(1)) {
			case LITERAL_const:
			{
				match(LITERAL_const);
				if ( inputState.guessing==0 ) {
					tqual = Specifier.CONST;
				}
				break;
			}
			case LITERAL_volatile:
			{
				match(LITERAL_volatile);
				if ( inputState.guessing==0 ) {
					tqual = Specifier.VOLATILE;
				}
				break;
			}
			case LITERAL_restrict:
			{
				match(LITERAL_restrict);
				if ( inputState.guessing==0 ) {
					tqual = Specifier.RESTRICT;
				}
				break;
			}
			case LITERAL___nvl__:
			{
				match(LITERAL___nvl__);
				if ( inputState.guessing==0 ) {
					tqual = Specifier.NVL;
				}
				break;
			}
			case LITERAL___nvl_wp__:
			{
				match(LITERAL___nvl_wp__);
				if ( inputState.guessing==0 ) {
					tqual = Specifier.NVL_WP;
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_29);
			} else {
			  throw ex;
			}
		}
		return tqual;
	}
	
/***************************************************
 * Should a basic type be an int value or a class ? *
 ****************************************************/
	public final Specifier  typeSpecifier() throws RecognitionException, TokenStreamException {
		Specifier types;
		
		
		types = null;
		String tname = null;
		Expression expr1 = null;
		List tyname = null;
		List attList = null;
		boolean typedefold = false;
		
		
		try {      // for error handling
			if ( inputState.guessing==0 ) {
				typedefold = hastypedef; hastypedef = false;
			}
			{
			switch ( LA(1)) {
			case LITERAL_void:
			{
				match(LITERAL_void);
				if ( inputState.guessing==0 ) {
					types = Specifier.VOID;
				}
				break;
			}
			case LITERAL_char:
			{
				match(LITERAL_char);
				if ( inputState.guessing==0 ) {
					types = Specifier.CHAR;
				}
				break;
			}
			case LITERAL_short:
			{
				match(LITERAL_short);
				if ( inputState.guessing==0 ) {
					types = Specifier.SHORT;
				}
				break;
			}
			case LITERAL_int:
			{
				match(LITERAL_int);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT;
				}
				break;
			}
			case LITERAL_long:
			{
				match(LITERAL_long);
				if ( inputState.guessing==0 ) {
					types = Specifier.LONG;
				}
				break;
			}
			case LITERAL_float:
			{
				match(LITERAL_float);
				if ( inputState.guessing==0 ) {
					types = Specifier.FLOAT;
				}
				break;
			}
			case LITERAL_double:
			{
				match(LITERAL_double);
				if ( inputState.guessing==0 ) {
					types = Specifier.DOUBLE;
				}
				break;
			}
			case LITERAL_signed:
			{
				match(LITERAL_signed);
				if ( inputState.guessing==0 ) {
					types = Specifier.SIGNED;
				}
				break;
			}
			case LITERAL_unsigned:
			{
				match(LITERAL_unsigned);
				if ( inputState.guessing==0 ) {
					types = Specifier.UNSIGNED;
				}
				break;
			}
			case LITERAL__Bool:
			{
				match(LITERAL__Bool);
				if ( inputState.guessing==0 ) {
					types = Specifier.CBOOL;
				}
				break;
			}
			case LITERAL__Complex:
			{
				match(LITERAL__Complex);
				if ( inputState.guessing==0 ) {
					types = Specifier.CCOMPLEX;
				}
				break;
			}
			case LITERAL__Imaginary:
			{
				match(LITERAL__Imaginary);
				if ( inputState.guessing==0 ) {
					types = Specifier.CIMAGINARY;
				}
				break;
			}
			case LITERAL___thread:
			{
				match(LITERAL___thread);
				if ( inputState.guessing==0 ) {
					types = Specifier.THREAD;
				}
				break;
			}
			case LITERAL___nvl__:
			{
				match(LITERAL___nvl__);
				if ( inputState.guessing==0 ) {
					types = Specifier.NVL;
				}
				break;
			}
			case LITERAL___nvl_wp__:
			{
				match(LITERAL___nvl_wp__);
				if ( inputState.guessing==0 ) {
					types = Specifier.NVL_WP;
				}
				break;
			}
			case 36:
			{
				match(36);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT8_T;
				}
				break;
			}
			case 37:
			{
				match(37);
				if ( inputState.guessing==0 ) {
					types = Specifier.UINT8_T;
				}
				break;
			}
			case 38:
			{
				match(38);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT16_T;
				}
				break;
			}
			case 39:
			{
				match(39);
				if ( inputState.guessing==0 ) {
					types = Specifier.UINT16_T;
				}
				break;
			}
			case 40:
			{
				match(40);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT32_T;
				}
				break;
			}
			case 41:
			{
				match(41);
				if ( inputState.guessing==0 ) {
					types = Specifier.UINT32_T;
				}
				break;
			}
			case 42:
			{
				match(42);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT64_T;
				}
				break;
			}
			case 43:
			{
				match(43);
				if ( inputState.guessing==0 ) {
					types = Specifier.UINT64_T;
				}
				break;
			}
			case LITERAL_size_t:
			{
				match(LITERAL_size_t);
				if ( inputState.guessing==0 ) {
					types = Specifier.SIZE_T;
				}
				break;
			}
			case 45:
			{
				match(45);
				if ( inputState.guessing==0 ) {
					types = Specifier.FLOAT128;
				}
				break;
			}
			case 46:
			{
				match(46);
				if ( inputState.guessing==0 ) {
					types = Specifier.__FLOAT128;
				}
				break;
			}
			case 47:
			{
				match(47);
				if ( inputState.guessing==0 ) {
					types = Specifier.__FLOAT80;
				}
				break;
			}
			case 48:
			{
				match(48);
				if ( inputState.guessing==0 ) {
					types = Specifier.__IBM128;
				}
				break;
			}
			case 49:
			{
				match(49);
				if ( inputState.guessing==0 ) {
					types = Specifier.FLOAT16;
				}
				match(LITERAL__Nullable);
				if ( inputState.guessing==0 ) {
					types = Specifier._NULLABLE;
				}
				break;
			}
			case LITERAL___global__:
			{
				match(LITERAL___global__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_GLOBAL;
				}
				break;
			}
			case LITERAL___shared__:
			{
				match(LITERAL___shared__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_SHARED;
				}
				break;
			}
			case LITERAL___host__:
			{
				match(LITERAL___host__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_HOST;
				}
				break;
			}
			case LITERAL___device__:
			{
				match(LITERAL___device__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_DEVICE;
				}
				break;
			}
			case LITERAL___constant__:
			{
				match(LITERAL___constant__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_CONSTANT;
				}
				break;
			}
			case LITERAL___noinline__:
			{
				match(LITERAL___noinline__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_NOINLINE;
				}
				break;
			}
			case LITERAL___kernel:
			{
				match(LITERAL___kernel);
				if ( inputState.guessing==0 ) {
					types = OpenCLSpecifier.OPENCL_KERNEL;
				}
				break;
			}
			case LITERAL___global:
			{
				match(LITERAL___global);
				if ( inputState.guessing==0 ) {
					types = OpenCLSpecifier.OPENCL_GLOBAL;
				}
				break;
			}
			case LITERAL___local:
			{
				match(LITERAL___local);
				if ( inputState.guessing==0 ) {
					types = OpenCLSpecifier.OPENCL_LOCAL;
				}
				break;
			}
			case LITERAL___constant:
			{
				match(LITERAL___constant);
				if ( inputState.guessing==0 ) {
					types = OpenCLSpecifier.OPENCL_CONSTANT;
				}
				break;
			}
			case LITERAL_struct:
			case LITERAL_union:
			{
				types=structOrUnionSpecifier();
				{
				_loop50:
				do {
					if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG)) && (LA(2)==LPAREN)) {
						attributeDecl();
					}
					else {
						break _loop50;
					}
					
				} while (true);
				}
				break;
			}
			case LITERAL_enum:
			{
				types=enumSpecifier();
				break;
			}
			case ID:
			{
				types=typedefName();
				break;
			}
			case LITERAL_typeof:
			{
				match(LITERAL_typeof);
				match(LPAREN);
				{
				boolean synPredMatched53 = false;
				if (((_tokenSet_30.member(LA(1))) && (_tokenSet_31.member(LA(2))))) {
					int _m53 = mark();
					synPredMatched53 = true;
					inputState.guessing++;
					try {
						{
						typeName2();
						}
					}
					catch (RecognitionException pe) {
						synPredMatched53 = false;
					}
					rewind(_m53);
inputState.guessing--;
				}
				if ( synPredMatched53 ) {
					tyname=typeName2();
					if ( inputState.guessing==0 ) {
						types = new TypeofSpecifier(tyname);
					}
					{
					_loop55:
					do {
						if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG))) {
							attList=attributeDecl();
						}
						else {
							break _loop55;
						}
						
					} while (true);
					}
				}
				else if ((_tokenSet_19.member(LA(1))) && (_tokenSet_32.member(LA(2)))) {
					expr1=expr();
					if ( inputState.guessing==0 ) {
						types = new TypeofSpecifier(expr1);
					}
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				match(RPAREN);
				break;
			}
			case LITERAL___complex:
			{
				match(LITERAL___complex);
				if ( inputState.guessing==0 ) {
					types = Specifier.DOUBLE;
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				hastypedef = typedefold;
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_29);
			} else {
			  throw ex;
			}
		}
		return types;
	}
	
	public final List<Attribute>  attributeDecl() throws RecognitionException, TokenStreamException {
		List<Attribute> list;
		
		
		list = new ArrayList<>();
		List<Attribute> subList;
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case LITERAL___attribute:
			{
				match(LITERAL___attribute);
				match(LPAREN);
				match(LPAREN);
				subList=attributeList();
				match(RPAREN);
				match(RPAREN);
				if ( inputState.guessing==0 ) {
					list.addAll(subList);
				}
				{
				_loop126:
				do {
					if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG)) && (LA(2)==LPAREN)) {
						subList=attributeDecl();
						if ( inputState.guessing==0 ) {
							list.addAll(subList);
						}
					}
					else {
						break _loop126;
					}
					
				} while (true);
				}
				break;
			}
			case LITERAL___asm:
			{
				match(LITERAL___asm);
				match(LPAREN);
				stringConst();
				match(RPAREN);
				break;
			}
			case LITERAL___OSX_AVAILABLE_STARTING:
			{
				match(LITERAL___OSX_AVAILABLE_STARTING);
				match(LPAREN);
				{
				int _cnt131=0;
				_loop131:
				do {
					if ((_tokenSet_33.member(LA(1)))) {
						{
						_loop129:
						do {
							if ((LA(1)==COMMA)) {
								match(COMMA);
							}
							else {
								break _loop129;
							}
							
						} while (true);
						}
						{
						switch ( LA(1)) {
						case Number:
						{
							match(Number);
							break;
						}
						case StringLiteral:
						{
							match(StringLiteral);
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
					}
					else {
						if ( _cnt131>=1 ) { break _loop131; } else {throw new NoViableAltException(LT(1), getFilename());}
					}
					
					_cnt131++;
				} while (true);
				}
				match(RPAREN);
				break;
			}
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			{
				match(LITERAL___OSX_AVAILABLE_BUT_DEPRECATED);
				match(LPAREN);
				{
				int _cnt136=0;
				_loop136:
				do {
					if ((_tokenSet_33.member(LA(1)))) {
						{
						_loop134:
						do {
							if ((LA(1)==COMMA)) {
								match(COMMA);
							}
							else {
								break _loop134;
							}
							
						} while (true);
						}
						{
						switch ( LA(1)) {
						case Number:
						{
							match(Number);
							break;
						}
						case StringLiteral:
						{
							match(StringLiteral);
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
					}
					else {
						if ( _cnt136>=1 ) { break _loop136; } else {throw new NoViableAltException(LT(1), getFilename());}
					}
					
					_cnt136++;
				} while (true);
				}
				match(RPAREN);
				break;
			}
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			{
				match(LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG);
				match(LPAREN);
				{
				int _cnt141=0;
				_loop141:
				do {
					if ((_tokenSet_33.member(LA(1)))) {
						{
						_loop139:
						do {
							if ((LA(1)==COMMA)) {
								match(COMMA);
							}
							else {
								break _loop139;
							}
							
						} while (true);
						}
						{
						switch ( LA(1)) {
						case Number:
						{
							match(Number);
							break;
						}
						case StringLiteral:
						{
							match(StringLiteral);
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
					}
					else {
						if ( _cnt141>=1 ) { break _loop141; } else {throw new NoViableAltException(LT(1), getFilename());}
					}
					
					_cnt141++;
				} while (true);
				}
				match(RPAREN);
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_34);
			} else {
			  throw ex;
			}
		}
		return list;
	}
	
	public final void vcDeclSpec() throws RecognitionException, TokenStreamException {
		
		
		try {      // for error handling
			match(LITERAL___declspec);
			match(LPAREN);
			{
			_loop118:
			do {
				if ((LA(1)==ID)) {
					extendedDeclModifier();
				}
				else {
					break _loop118;
				}
				
			} while (true);
			}
			match(RPAREN);
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_28);
			} else {
			  throw ex;
			}
		}
	}
	
	public final Specifier  typeSpecifier_noTypeDefName() throws RecognitionException, TokenStreamException {
		Specifier types;
		
		
		types = null;
		String tname = null;
		Expression expr1 = null;
		List tyname = null;
		List attList = null;
		boolean typedefold = false;
		
		
		try {      // for error handling
			if ( inputState.guessing==0 ) {
				typedefold = hastypedef; hastypedef = false;
			}
			{
			switch ( LA(1)) {
			case LITERAL_void:
			{
				match(LITERAL_void);
				if ( inputState.guessing==0 ) {
					types = Specifier.VOID;
				}
				break;
			}
			case LITERAL_char:
			{
				match(LITERAL_char);
				if ( inputState.guessing==0 ) {
					types = Specifier.CHAR;
				}
				break;
			}
			case LITERAL_short:
			{
				match(LITERAL_short);
				if ( inputState.guessing==0 ) {
					types = Specifier.SHORT;
				}
				break;
			}
			case LITERAL_int:
			{
				match(LITERAL_int);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT;
				}
				break;
			}
			case LITERAL_long:
			{
				match(LITERAL_long);
				if ( inputState.guessing==0 ) {
					types = Specifier.LONG;
				}
				break;
			}
			case LITERAL_float:
			{
				match(LITERAL_float);
				if ( inputState.guessing==0 ) {
					types = Specifier.FLOAT;
				}
				break;
			}
			case LITERAL_double:
			{
				match(LITERAL_double);
				if ( inputState.guessing==0 ) {
					types = Specifier.DOUBLE;
				}
				break;
			}
			case LITERAL_signed:
			{
				match(LITERAL_signed);
				if ( inputState.guessing==0 ) {
					types = Specifier.SIGNED;
				}
				break;
			}
			case LITERAL_unsigned:
			{
				match(LITERAL_unsigned);
				if ( inputState.guessing==0 ) {
					types = Specifier.UNSIGNED;
				}
				break;
			}
			case LITERAL__Bool:
			{
				match(LITERAL__Bool);
				if ( inputState.guessing==0 ) {
					types = Specifier.CBOOL;
				}
				break;
			}
			case LITERAL__Complex:
			{
				match(LITERAL__Complex);
				if ( inputState.guessing==0 ) {
					types = Specifier.CCOMPLEX;
				}
				break;
			}
			case LITERAL__Imaginary:
			{
				match(LITERAL__Imaginary);
				if ( inputState.guessing==0 ) {
					types = Specifier.CIMAGINARY;
				}
				break;
			}
			case LITERAL___thread:
			{
				match(LITERAL___thread);
				if ( inputState.guessing==0 ) {
					types = Specifier.THREAD;
				}
				break;
			}
			case LITERAL___nvl__:
			{
				match(LITERAL___nvl__);
				if ( inputState.guessing==0 ) {
					types = Specifier.NVL;
				}
				break;
			}
			case LITERAL___nvl_wp__:
			{
				match(LITERAL___nvl_wp__);
				if ( inputState.guessing==0 ) {
					types = Specifier.NVL_WP;
				}
				break;
			}
			case 36:
			{
				match(36);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT8_T;
				}
				break;
			}
			case 37:
			{
				match(37);
				if ( inputState.guessing==0 ) {
					types = Specifier.UINT8_T;
				}
				break;
			}
			case 38:
			{
				match(38);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT16_T;
				}
				break;
			}
			case 39:
			{
				match(39);
				if ( inputState.guessing==0 ) {
					types = Specifier.UINT16_T;
				}
				break;
			}
			case 40:
			{
				match(40);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT32_T;
				}
				break;
			}
			case 41:
			{
				match(41);
				if ( inputState.guessing==0 ) {
					types = Specifier.UINT32_T;
				}
				break;
			}
			case 42:
			{
				match(42);
				if ( inputState.guessing==0 ) {
					types = Specifier.INT64_T;
				}
				break;
			}
			case 43:
			{
				match(43);
				if ( inputState.guessing==0 ) {
					types = Specifier.UINT64_T;
				}
				break;
			}
			case LITERAL_size_t:
			{
				match(LITERAL_size_t);
				if ( inputState.guessing==0 ) {
					types = Specifier.SIZE_T;
				}
				break;
			}
			case 45:
			{
				match(45);
				if ( inputState.guessing==0 ) {
					types = Specifier.FLOAT128;
				}
				break;
			}
			case 46:
			{
				match(46);
				if ( inputState.guessing==0 ) {
					types = Specifier.__FLOAT128;
				}
				break;
			}
			case 47:
			{
				match(47);
				if ( inputState.guessing==0 ) {
					types = Specifier.__FLOAT80;
				}
				break;
			}
			case 48:
			{
				match(48);
				if ( inputState.guessing==0 ) {
					types = Specifier.__IBM128;
				}
				break;
			}
			case 49:
			{
				match(49);
				if ( inputState.guessing==0 ) {
					types = Specifier.FLOAT16;
				}
				break;
			}
			case LITERAL___global__:
			{
				match(LITERAL___global__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_GLOBAL;
				}
				break;
			}
			case LITERAL___shared__:
			{
				match(LITERAL___shared__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_SHARED;
				}
				break;
			}
			case LITERAL___host__:
			{
				match(LITERAL___host__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_HOST;
				}
				break;
			}
			case LITERAL___device__:
			{
				match(LITERAL___device__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_DEVICE;
				}
				break;
			}
			case LITERAL___constant__:
			{
				match(LITERAL___constant__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_CONSTANT;
				}
				break;
			}
			case LITERAL___noinline__:
			{
				match(LITERAL___noinline__);
				if ( inputState.guessing==0 ) {
					types = CUDASpecifier.CUDA_NOINLINE;
				}
				break;
			}
			case LITERAL___kernel:
			{
				match(LITERAL___kernel);
				if ( inputState.guessing==0 ) {
					types = OpenCLSpecifier.OPENCL_KERNEL;
				}
				break;
			}
			case LITERAL___global:
			{
				match(LITERAL___global);
				if ( inputState.guessing==0 ) {
					types = OpenCLSpecifier.OPENCL_GLOBAL;
				}
				break;
			}
			case LITERAL___local:
			{
				match(LITERAL___local);
				if ( inputState.guessing==0 ) {
					types = OpenCLSpecifier.OPENCL_LOCAL;
				}
				break;
			}
			case LITERAL___constant:
			{
				match(LITERAL___constant);
				if ( inputState.guessing==0 ) {
					types = OpenCLSpecifier.OPENCL_CONSTANT;
				}
				break;
			}
			case LITERAL_struct:
			case LITERAL_union:
			{
				types=structOrUnionSpecifier();
				{
				_loop59:
				do {
					if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG)) && (LA(2)==LPAREN)) {
						attributeDecl();
					}
					else {
						break _loop59;
					}
					
				} while (true);
				}
				break;
			}
			case LITERAL_enum:
			{
				types=enumSpecifier();
				break;
			}
			case LITERAL_typeof:
			{
				match(LITERAL_typeof);
				match(LPAREN);
				{
				boolean synPredMatched62 = false;
				if (((_tokenSet_30.member(LA(1))) && (_tokenSet_31.member(LA(2))))) {
					int _m62 = mark();
					synPredMatched62 = true;
					inputState.guessing++;
					try {
						{
						typeName2();
						}
					}
					catch (RecognitionException pe) {
						synPredMatched62 = false;
					}
					rewind(_m62);
inputState.guessing--;
				}
				if ( synPredMatched62 ) {
					tyname=typeName2();
					if ( inputState.guessing==0 ) {
						types = new TypeofSpecifier(tyname);
					}
					{
					_loop64:
					do {
						if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG))) {
							attList=attributeDecl();
						}
						else {
							break _loop64;
						}
						
					} while (true);
					}
				}
				else if ((_tokenSet_19.member(LA(1))) && (_tokenSet_32.member(LA(2)))) {
					expr1=expr();
					if ( inputState.guessing==0 ) {
						types = new TypeofSpecifier(expr1);
					}
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				match(RPAREN);
				break;
			}
			case LITERAL___complex:
			{
				match(LITERAL___complex);
				if ( inputState.guessing==0 ) {
					types = Specifier.DOUBLE;
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				hastypedef = typedefold;
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_29);
			} else {
			  throw ex;
			}
		}
		return types;
	}
	
	public final Specifier  functionStorageClassSpecifier() throws RecognitionException, TokenStreamException {
		Specifier type;
		
		type= null;
		
		try {      // for error handling
			switch ( LA(1)) {
			case LITERAL_extern:
			{
				match(LITERAL_extern);
				if ( inputState.guessing==0 ) {
					type= Specifier.EXTERN;
				}
				break;
			}
			case LITERAL_static:
			{
				match(LITERAL_static);
				if ( inputState.guessing==0 ) {
					type= Specifier.STATIC;
				}
				break;
			}
			case LITERAL_inline:
			{
				match(LITERAL_inline);
				if ( inputState.guessing==0 ) {
					type= Specifier.INLINE;
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_28);
			} else {
			  throw ex;
			}
		}
		return type;
	}
	
	public final Specifier  structOrUnionSpecifier() throws RecognitionException, TokenStreamException {
		Specifier spec;
		
		Token  i = null;
		Token  l = null;
		Token  l1 = null;
		Token  sou3 = null;
		
		ClassDeclaration decls = null;
		String name=null;
		int type=0;
		spec = null;
		int linenum = 0;
		
		
		try {      // for error handling
			type=structOrUnion();
			{
			switch ( LA(1)) {
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			{
				attributeDecl();
				break;
			}
			case LCURLY:
			case ID:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			boolean synPredMatched71 = false;
			if (((LA(1)==ID) && (LA(2)==LCURLY))) {
				int _m71 = mark();
				synPredMatched71 = true;
				inputState.guessing++;
				try {
					{
					match(ID);
					match(LCURLY);
					}
				}
				catch (RecognitionException pe) {
					synPredMatched71 = false;
				}
				rewind(_m71);
inputState.guessing--;
			}
			if ( synPredMatched71 ) {
				i = LT(1);
				match(ID);
				l = LT(1);
				match(LCURLY);
				if ( inputState.guessing==0 ) {
					
					name = i.getText();linenum = i.getLine(); putPragma(i,symtab);
					String sname = null;
					if (type == 1) {
					decls = new ClassDeclaration(ClassDeclaration.STRUCT, new NameID(name));
					spec = new UserSpecifier(new NameID("struct "+name));
					} else {
					decls = new ClassDeclaration(ClassDeclaration.UNION, new NameID(name));
					spec = new UserSpecifier(new NameID("union "+name));
					}
					
				}
				{
				switch ( LA(1)) {
				case LITERAL_volatile:
				case LITERAL_struct:
				case LITERAL_union:
				case LITERAL_enum:
				case LITERAL___thread:
				case LITERAL_const:
				case LITERAL_restrict:
				case LITERAL___nvl__:
				case LITERAL___nvl_wp__:
				case LITERAL_void:
				case LITERAL_char:
				case LITERAL_short:
				case LITERAL_int:
				case LITERAL_long:
				case LITERAL_float:
				case LITERAL_double:
				case LITERAL_signed:
				case LITERAL_unsigned:
				case LITERAL__Bool:
				case LITERAL__Complex:
				case LITERAL__Imaginary:
				case 36:
				case 37:
				case 38:
				case 39:
				case 40:
				case 41:
				case 42:
				case 43:
				case LITERAL_size_t:
				case 45:
				case 46:
				case 47:
				case 48:
				case 49:
				case LITERAL___global__:
				case LITERAL___shared__:
				case LITERAL___host__:
				case LITERAL___device__:
				case LITERAL___constant__:
				case LITERAL___noinline__:
				case LITERAL___kernel:
				case LITERAL___global:
				case LITERAL___local:
				case LITERAL___constant:
				case LITERAL_typeof:
				case LITERAL___complex:
				case ID:
				{
					structDeclarationList(decls);
					break;
				}
				case RCURLY:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				if ( inputState.guessing==0 ) {
					
					if (symtab instanceof ClassDeclaration) {
					int si = symtabstack.size()-1;
					for (;si>=0;si--) {
					if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
					((SymbolTable)symtabstack.get(si)).addDeclaration(decls);
					break;
					}
					}
					} else
					symtab.addDeclaration(decls);
					
				}
				match(RCURLY);
			}
			else if ((LA(1)==LCURLY)) {
				l1 = LT(1);
				match(LCURLY);
				if ( inputState.guessing==0 ) {
					
					name = "named_"+getLexer().originalSource +"_"+ ((CToken)l1).getTokenNumber();
					name = name.replaceAll("[.]","_");
					name = name.replaceAll("-","_");
					linenum = l1.getLine(); putPragma(l1,symtab);
					if (type == 1) {
					decls = new ClassDeclaration(ClassDeclaration.STRUCT, new NameID(name));
					spec = new UserSpecifier(new NameID("struct "+name));
					} else {
					decls = new ClassDeclaration(ClassDeclaration.UNION, new NameID(name));
					spec = new UserSpecifier(new NameID("union "+name));
					}
					
				}
				{
				switch ( LA(1)) {
				case LITERAL_volatile:
				case LITERAL_struct:
				case LITERAL_union:
				case LITERAL_enum:
				case LITERAL___thread:
				case LITERAL_const:
				case LITERAL_restrict:
				case LITERAL___nvl__:
				case LITERAL___nvl_wp__:
				case LITERAL_void:
				case LITERAL_char:
				case LITERAL_short:
				case LITERAL_int:
				case LITERAL_long:
				case LITERAL_float:
				case LITERAL_double:
				case LITERAL_signed:
				case LITERAL_unsigned:
				case LITERAL__Bool:
				case LITERAL__Complex:
				case LITERAL__Imaginary:
				case 36:
				case 37:
				case 38:
				case 39:
				case 40:
				case 41:
				case 42:
				case 43:
				case LITERAL_size_t:
				case 45:
				case 46:
				case 47:
				case 48:
				case 49:
				case LITERAL___global__:
				case LITERAL___shared__:
				case LITERAL___host__:
				case LITERAL___device__:
				case LITERAL___constant__:
				case LITERAL___noinline__:
				case LITERAL___kernel:
				case LITERAL___global:
				case LITERAL___local:
				case LITERAL___constant:
				case LITERAL_typeof:
				case LITERAL___complex:
				case ID:
				{
					structDeclarationList(decls);
					break;
				}
				case RCURLY:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				if ( inputState.guessing==0 ) {
					
					if (symtab instanceof ClassDeclaration) {
					int si = symtabstack.size()-1;
					for (;si>=0;si--) {
					if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
					((SymbolTable)symtabstack.get(si)).addDeclaration(decls);
					break;
					}
					}
					} else
					symtab.addDeclaration(decls);
					
				}
				match(RCURLY);
			}
			else if ((LA(1)==ID) && (_tokenSet_29.member(LA(2)))) {
				sou3 = LT(1);
				match(ID);
				if ( inputState.guessing==0 ) {
					
					name = sou3.getText();linenum = sou3.getLine(); putPragma(sou3,symtab);
					if(type == 1) {
					spec = new UserSpecifier(new NameID("struct "+name));
					decls = new ClassDeclaration(ClassDeclaration.STRUCT,new NameID(name),true);
					} else {
					spec = new UserSpecifier(new NameID("union "+name));
					decls = new ClassDeclaration(ClassDeclaration.UNION,new NameID(name),true);
					}
					prev_decl = decls;
					
				}
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_29);
			} else {
			  throw ex;
			}
		}
		return spec;
	}
	
	public final Specifier  enumSpecifier() throws RecognitionException, TokenStreamException {
		Specifier spec;
		
		Token  i = null;
		Token  el1 = null;
		Token  espec2 = null;
		
		cetus.hir.Enumeration espec = null;
		String enumN = null;
		List elist=null;
		spec = null;
		
		
		try {      // for error handling
			match(LITERAL_enum);
			{
			switch ( LA(1)) {
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			{
				attributeDecl();
				break;
			}
			case LCURLY:
			case ID:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			boolean synPredMatched108 = false;
			if (((LA(1)==ID) && (LA(2)==LCURLY))) {
				int _m108 = mark();
				synPredMatched108 = true;
				inputState.guessing++;
				try {
					{
					match(ID);
					match(LCURLY);
					}
				}
				catch (RecognitionException pe) {
					synPredMatched108 = false;
				}
				rewind(_m108);
inputState.guessing--;
			}
			if ( synPredMatched108 ) {
				i = LT(1);
				match(ID);
				if ( inputState.guessing==0 ) {
					putPragma(i,symtab);
				}
				match(LCURLY);
				elist=enumList();
				match(RCURLY);
				if ( inputState.guessing==0 ) {
					enumN =i.getText();
				}
			}
			else if ((LA(1)==LCURLY)) {
				el1 = LT(1);
				match(LCURLY);
				if ( inputState.guessing==0 ) {
					putPragma(el1,symtab);
				}
				elist=enumList();
				match(RCURLY);
				if ( inputState.guessing==0 ) {
					
					enumN = getLexer().originalSource +"_"+ ((CToken)el1).getTokenNumber();
					enumN =enumN.replaceAll("[.]","_");
					enumN =enumN.replaceAll("-","_");
					
				}
			}
			else if ((LA(1)==ID) && (_tokenSet_29.member(LA(2)))) {
				espec2 = LT(1);
				match(ID);
				if ( inputState.guessing==0 ) {
					enumN =espec2.getText();putPragma(espec2,symtab);
				}
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			if ( inputState.guessing==0 ) {
				
				if (elist != null) {
				espec = new cetus.hir.Enumeration(new NameID(enumN),elist);
				if (symtab instanceof ClassDeclaration) {
				int si = symtabstack.size()-1;
				for (;si>=0;si--) {
				if (!(symtabstack.get(si) instanceof ClassDeclaration)) {
				((SymbolTable)symtabstack.get(si)).addDeclaration(espec);
				break;
				}
				}
				} else
				symtab.addDeclaration(espec);
				}
				spec = new UserSpecifier(new NameID("enum "+enumN));
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_29);
			} else {
			  throw ex;
			}
		}
		return spec;
	}
	
	public final Specifier  typedefName() throws RecognitionException, TokenStreamException {
		Specifier name;
		
		Token  i = null;
		name = null;
		
		try {      // for error handling
			if (!(isTypedefName ( LT(1).getText() )))
			  throw new SemanticException("isTypedefName ( LT(1).getText() )");
			i = LT(1);
			match(ID);
			if ( inputState.guessing==0 ) {
				name = new UserSpecifier(new NameID(i.getText()));
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_29);
			} else {
			  throw ex;
			}
		}
		return name;
	}
	
	public final List  typeName2() throws RecognitionException, TokenStreamException {
		List tname;
		
		
		tname=null;
		Declarator decl = null;
		
		
		try {      // for error handling
			tname=specifierQualifierList();
			{
			switch ( LA(1)) {
			case LPAREN:
			case STAR:
			case LBRACKET:
			{
				decl=nonemptyAbstractDeclarator2(tname, true);
				break;
			}
			case RPAREN:
			case COMMA:
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_35);
			} else {
			  throw ex;
			}
		}
		return tname;
	}
	
	public final int  structOrUnion() throws RecognitionException, TokenStreamException {
		int type;
		
		type=0;
		
		try {      // for error handling
			switch ( LA(1)) {
			case LITERAL_struct:
			{
				match(LITERAL_struct);
				if ( inputState.guessing==0 ) {
					type = 1;
				}
				break;
			}
			case LITERAL_union:
			{
				match(LITERAL_union);
				if ( inputState.guessing==0 ) {
					type = 2;
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_36);
			} else {
			  throw ex;
			}
		}
		return type;
	}
	
	public final void structDeclarationList(
		ClassDeclaration cdecl
	) throws RecognitionException, TokenStreamException {
		
		Declaration sdecl= null;/*SymbolTable prev_symtab = symtab;*/
		
		try {      // for error handling
			if ( inputState.guessing==0 ) {
				enterSymtab(cdecl);
			}
			{
			int _cnt76=0;
			_loop76:
			do {
				if ((_tokenSet_30.member(LA(1)))) {
					sdecl=structDeclaration();
					if ( inputState.guessing==0 ) {
						if(sdecl != null ) cdecl.addDeclaration(sdecl);
					}
				}
				else {
					if ( _cnt76>=1 ) { break _loop76; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt76++;
			} while (true);
			}
			if ( inputState.guessing==0 ) {
				exitSymtab(); /*symtab = prev_symtab;*/
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_37);
			} else {
			  throw ex;
			}
		}
	}
	
	public final Declaration  structDeclaration() throws RecognitionException, TokenStreamException {
		Declaration sdecl;
		
		
		List bsqlist=null;
		List bsdlist=null;
		sdecl=null;
		
		
		try {      // for error handling
			bsqlist=specifierQualifierList();
			bsdlist=structDeclaratorList();
			{
			switch ( LA(1)) {
			case COMMA:
			{
				match(COMMA);
				break;
			}
			case SEMI:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			int _cnt80=0;
			_loop80:
			do {
				if ((LA(1)==SEMI)) {
					match(SEMI);
				}
				else {
					if ( _cnt80>=1 ) { break _loop80; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt80++;
			} while (true);
			}
			if ( inputState.guessing==0 ) {
				sdecl = new VariableDeclaration(bsqlist,bsdlist); hastypedef = false;
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_38);
			} else {
			  throw ex;
			}
		}
		return sdecl;
	}
	
	public final List  specifierQualifierList() throws RecognitionException, TokenStreamException {
		List sqlist;
		
		
		sqlist=new ArrayList();
		Specifier tspec=null;
		Specifier tqual=null;
		
		
		try {      // for error handling
			{
			{
			_loop89:
			do {
				if ((_tokenSet_13.member(LA(1))) && (_tokenSet_30.member(LA(2)))) {
					tqual=typeQualifier();
					if ( inputState.guessing==0 ) {
						sqlist.add(tqual);
					}
				}
				else {
					break _loop89;
				}
				
			} while (true);
			}
			tspec=typeSpecifier();
			if ( inputState.guessing==0 ) {
				sqlist.add(tspec);
			}
			{
			_loop95:
			do {
				boolean synPredMatched94 = false;
				if (((_tokenSet_23.member(LA(1))) && (_tokenSet_39.member(LA(2))))) {
					int _m94 = mark();
					synPredMatched94 = true;
					inputState.guessing++;
					try {
						{
						if ((LA(1)==LITERAL_struct) && (true)) {
							match(LITERAL_struct);
						}
						else if ((LA(1)==LITERAL_union) && (true)) {
							match(LITERAL_union);
						}
						else if ((LA(1)==LITERAL_enum) && (true)) {
							match(LITERAL_enum);
						}
						else if ((_tokenSet_23.member(LA(1))) && (true)) {
							typeSpecifier_noTypeDefName();
						}
						else {
							throw new NoViableAltException(LT(1), getFilename());
						}
						
						}
					}
					catch (RecognitionException pe) {
						synPredMatched94 = false;
					}
					rewind(_m94);
inputState.guessing--;
				}
				if ( synPredMatched94 ) {
					tspec=typeSpecifier_noTypeDefName();
					if ( inputState.guessing==0 ) {
						sqlist.add(tspec);
					}
				}
				else if ((_tokenSet_13.member(LA(1))) && (_tokenSet_40.member(LA(2)))) {
					tqual=typeQualifier();
					if ( inputState.guessing==0 ) {
						sqlist.add(tqual);
					}
				}
				else {
					break _loop95;
				}
				
			} while (true);
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_41);
			} else {
			  throw ex;
			}
		}
		return sqlist;
	}
	
	public final List  structDeclaratorList() throws RecognitionException, TokenStreamException {
		List sdlist;
		
		
		sdlist = new ArrayList();
		Declarator sdecl=null;
		
		
		try {      // for error handling
			sdecl=structDeclarator();
			if ( inputState.guessing==0 ) {
				
				// why am I getting a null value here ?
				if (sdecl != null)
				sdlist.add(sdecl);
				
			}
			{
			_loop98:
			do {
				if ((LA(1)==COMMA) && (_tokenSet_42.member(LA(2)))) {
					match(COMMA);
					sdecl=structDeclarator();
					if ( inputState.guessing==0 ) {
						
						// [DEBUG by Seyong Lee] may be null if used in a function parameter declaration
						if( sdecl != null )
						sdlist.add(sdecl);
						
					}
				}
				else {
					break _loop98;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_43);
			} else {
			  throw ex;
			}
		}
		return sdlist;
	}
	
	public final List  specifierQualifierList_old() throws RecognitionException, TokenStreamException {
		List sqlist;
		
		
		sqlist=new ArrayList();
		Specifier tspec=null;
		Specifier tqual=null;
		
		
		try {      // for error handling
			{
			int _cnt85=0;
			_loop85:
			do {
				boolean synPredMatched84 = false;
				if (((_tokenSet_14.member(LA(1))) && (_tokenSet_44.member(LA(2))))) {
					int _m84 = mark();
					synPredMatched84 = true;
					inputState.guessing++;
					try {
						{
						if ((LA(1)==LITERAL_struct) && (true)) {
							match(LITERAL_struct);
						}
						else if ((LA(1)==LITERAL_union) && (true)) {
							match(LITERAL_union);
						}
						else if ((LA(1)==LITERAL_enum) && (true)) {
							match(LITERAL_enum);
						}
						else if ((_tokenSet_14.member(LA(1))) && (true)) {
							typeSpecifier();
						}
						else {
							throw new NoViableAltException(LT(1), getFilename());
						}
						
						}
					}
					catch (RecognitionException pe) {
						synPredMatched84 = false;
					}
					rewind(_m84);
inputState.guessing--;
				}
				if ( synPredMatched84 ) {
					tspec=typeSpecifier();
					if ( inputState.guessing==0 ) {
						sqlist.add(tspec);
					}
				}
				else if ((_tokenSet_13.member(LA(1))) && (_tokenSet_45.member(LA(2)))) {
					tqual=typeQualifier();
					if ( inputState.guessing==0 ) {
						sqlist.add(tqual);
					}
				}
				else {
					if ( _cnt85>=1 ) { break _loop85; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt85++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
		return sqlist;
	}
	
	public final Declarator  structDeclarator() throws RecognitionException, TokenStreamException {
		Declarator sdecl;
		
		
		// [Modified by Joel E. Denny to handle unnamed bit-fields.]
		sdecl=new VariableDeclarator(new NameID(""));
		Expression expr1=null;
		
		
		try {      // for error handling
			{
			if ((_tokenSet_6.member(LA(1))) && (_tokenSet_46.member(LA(2)))) {
				sdecl=declarator();
			}
			else if ((_tokenSet_47.member(LA(1))) && (_tokenSet_48.member(LA(2)))) {
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			{
			switch ( LA(1)) {
			case COLON:
			{
				match(COLON);
				expr1=assignExpr();
				break;
			}
			case SEMI:
			case COMMA:
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				
				if (sdecl != null && expr1 != null) {
				// [Modified by Joel E. Denny not to throw away expressions that Cetus
				// cannot simplify to an IntegerLiteral but which are constant expressions.
				// For example, "sizeof(int)".]
				expr1 = Symbolic.simplify(expr1);
				//if (expr1 instanceof IntegerLiteral)
				sdecl.addTrailingSpecifier(new BitfieldSpecifier(expr1));
				//else
				//  ; // need to throw parse error
				}
				
			}
			{
			_loop103:
			do {
				if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG))) {
					attributeDecl();
				}
				else {
					break _loop103;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_43);
			} else {
			  throw ex;
			}
		}
		return sdecl;
	}
	
	public final Expression  assignExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		ret_expr = null; Expression expr1=null; AssignmentOperator code=null;
		
		try {      // for error handling
			ret_expr=conditionalExpr();
			{
			switch ( LA(1)) {
			case ASSIGN:
			case DIV_ASSIGN:
			case PLUS_ASSIGN:
			case MINUS_ASSIGN:
			case STAR_ASSIGN:
			case MOD_ASSIGN:
			case RSHIFT_ASSIGN:
			case LSHIFT_ASSIGN:
			case BAND_ASSIGN:
			case BOR_ASSIGN:
			case BXOR_ASSIGN:
			{
				code=assignOperator();
				expr1=assignExpr();
				if ( inputState.guessing==0 ) {
					ret_expr = new AssignmentExpression(ret_expr,code,expr1);
				}
				break;
			}
			case SEMI:
			case RCURLY:
			case RPAREN:
			case COMMA:
			case COLON:
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			case RBRACKET:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_49);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final List  enumList() throws RecognitionException, TokenStreamException {
		List elist;
		
		
		Declarator enum1=null;
		elist = new ArrayList();
		
		
		try {      // for error handling
			enum1=enumerator();
			if ( inputState.guessing==0 ) {
				elist.add(enum1);
			}
			{
			_loop111:
			do {
				if ((LA(1)==COMMA) && (LA(2)==ID)) {
					match(COMMA);
					enum1=enumerator();
					if ( inputState.guessing==0 ) {
						elist.add(enum1);
					}
				}
				else {
					break _loop111;
				}
				
			} while (true);
			}
			{
			switch ( LA(1)) {
			case COMMA:
			{
				match(COMMA);
				break;
			}
			case RCURLY:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_37);
			} else {
			  throw ex;
			}
		}
		return elist;
	}
	
	public final Declarator  enumerator() throws RecognitionException, TokenStreamException {
		Declarator decl;
		
		Token  i = null;
		decl=null;Expression expr2=null; String val = null;
		
		try {      // for error handling
			i = LT(1);
			match(ID);
			if ( inputState.guessing==0 ) {
				
				val = i.getText();
				decl = new VariableDeclarator(new NameID(val));
				
			}
			{
			switch ( LA(1)) {
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			{
				attributeDecl();
				break;
			}
			case RCURLY:
			case COMMA:
			case ASSIGN:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			switch ( LA(1)) {
			case ASSIGN:
			{
				match(ASSIGN);
				expr2=constExpr();
				if ( inputState.guessing==0 ) {
					decl.setInitializer(new Initializer(expr2));
				}
				break;
			}
			case RCURLY:
			case COMMA:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_50);
			} else {
			  throw ex;
			}
		}
		return decl;
	}
	
	public final Expression  constExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		ret_expr = null;
		
		try {      // for error handling
			ret_expr=conditionalExpr();
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_51);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final void extendedDeclModifier() throws RecognitionException, TokenStreamException {
		
		
		try {      // for error handling
			match(ID);
			{
			switch ( LA(1)) {
			case LPAREN:
			{
				match(LPAREN);
				{
				switch ( LA(1)) {
				case Number:
				{
					match(Number);
					break;
				}
				case StringLiteral:
				{
					{
					int _cnt123=0;
					_loop123:
					do {
						if ((LA(1)==StringLiteral)) {
							match(StringLiteral);
						}
						else {
							if ( _cnt123>=1 ) { break _loop123; } else {throw new NoViableAltException(LT(1), getFilename());}
						}
						
						_cnt123++;
					} while (true);
					}
					break;
				}
				case ID:
				{
					match(ID);
					match(ASSIGN);
					match(ID);
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				match(RPAREN);
				break;
			}
			case RPAREN:
			case ID:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_52);
			} else {
			  throw ex;
			}
		}
	}
	
	public final List<Attribute>  attributeList() throws RecognitionException, TokenStreamException {
		List<Attribute> list;
		
		list = new ArrayList<>(); Attribute attr;
		
		try {      // for error handling
			attr=attribute();
			if ( inputState.guessing==0 ) {
				if (attr != null) list.add(attr);
			}
			{
			_loop144:
			do {
				if ((LA(1)==COMMA)) {
					match(COMMA);
					attr=attribute();
					if ( inputState.guessing==0 ) {
						if (attr != null) list.add(attr);
					}
				}
				else {
					break _loop144;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_53);
			} else {
			  throw ex;
			}
		}
		return list;
	}
	
	protected final String  stringConst() throws RecognitionException, TokenStreamException {
		String name;
		
		Token  sl = null;
		name = "";
		
		try {      // for error handling
			{
			int _cnt421=0;
			_loop421:
			do {
				if ((LA(1)==StringLiteral)) {
					sl = LT(1);
					match(StringLiteral);
					if ( inputState.guessing==0 ) {
						name += sl.getText();
					}
				}
				else {
					if ( _cnt421>=1 ) { break _loop421; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt421++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_54);
			} else {
			  throw ex;
			}
		}
		return name;
	}
	
	public final Attribute  attribute() throws RecognitionException, TokenStreamException {
		Attribute spec;
		
		Token  id = null;
		spec = null;
		
		try {      // for error handling
			{
			switch ( LA(1)) {
			case LITERAL_typedef:
			case LITERAL_volatile:
			case LITERAL_auto:
			case LITERAL_register:
			case LITERAL___thread:
			case LITERAL_extern:
			case LITERAL_static:
			case LITERAL_inline:
			case LITERAL_const:
			case LITERAL_restrict:
			case LITERAL___nvl__:
			case LITERAL___nvl_wp__:
			case ID:
			{
				{
				switch ( LA(1)) {
				case ID:
				{
					id = LT(1);
					match(ID);
					if ( inputState.guessing==0 ) {
						spec = new Attribute(id.getText());
					}
					break;
				}
				case LITERAL_typedef:
				case LITERAL_auto:
				case LITERAL_register:
				case LITERAL___thread:
				case LITERAL_extern:
				case LITERAL_static:
				case LITERAL_inline:
				{
					storageClassSpecifier();
					break;
				}
				case LITERAL_volatile:
				case LITERAL_const:
				case LITERAL_restrict:
				case LITERAL___nvl__:
				case LITERAL___nvl_wp__:
				{
					typeQualifier();
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				{
				switch ( LA(1)) {
				case LPAREN:
				{
					match(LPAREN);
					{
					if ((LA(1)==ID) && (_tokenSet_55.member(LA(2)))) {
						match(ID);
					}
					else if ((_tokenSet_55.member(LA(1))) && (_tokenSet_32.member(LA(2)))) {
					}
					else {
						throw new NoViableAltException(LT(1), getFilename());
					}
					
					}
					{
					switch ( LA(1)) {
					case LPAREN:
					case ID:
					case Number:
					case StringLiteral:
					case LITERAL___asm:
					case STAR:
					case BAND:
					case PLUS:
					case MINUS:
					case LITERAL___builtin_nvl_get_root:
					case LITERAL___builtin_nvl_alloc_nv:
					case INC:
					case DEC:
					case LITERAL_sizeof:
					case LITERAL___alignof__:
					case LITERAL___builtin_va_arg:
					case LITERAL___builtin_offsetof:
					case BNOT:
					case LNOT:
					case LITERAL___real:
					case LITERAL___imag:
					case CharLiteral:
					{
						expr();
						break;
					}
					case RPAREN:
					{
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					match(RPAREN);
					break;
				}
				case RPAREN:
				case COMMA:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				break;
			}
			case RPAREN:
			case COMMA:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_56);
			} else {
			  throw ex;
			}
		}
		return spec;
	}
	
	public final Declarator  initDecl() throws RecognitionException, TokenStreamException {
		Declarator decl;
		
		
		decl = null;
		//Initializer binit=null;
		Object binit = null;
		Expression expr1=null;
		
		
		try {      // for error handling
			decl=declarator();
			{
			_loop157:
			do {
				if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG))) {
					attributeDecl();
				}
				else {
					break _loop157;
				}
				
			} while (true);
			}
			{
			switch ( LA(1)) {
			case ASSIGN:
			{
				match(ASSIGN);
				binit=initializer();
				break;
			}
			case COLON:
			{
				match(COLON);
				expr1=expr();
				break;
			}
			case SEMI:
			case COMMA:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				
				if (binit instanceof Expression)
				binit = new Initializer((Expression)binit);
				if (binit != null) {
				decl.setInitializer((Initializer)binit);
				/*
				System.out.println("Initializer " + decl.getClass());
				decl.print(System.out);
				System.out.print(" ");
				((Initializer)binit).print(System.out);
				System.out.println("");
				*/
				}
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_43);
			} else {
			  throw ex;
			}
		}
		return decl;
	}
	
	public final Object  initializer() throws RecognitionException, TokenStreamException {
		Object binit;
		
		
		binit = null;
		Expression expr1 = null;
		List ilist = null;
		Initializer init = null;
		
		
		try {      // for error handling
			if ((_tokenSet_57.member(LA(1))) && (_tokenSet_58.member(LA(2)))) {
				{
				if ((_tokenSet_19.member(LA(1))) && (_tokenSet_59.member(LA(2)))) {
					binit=assignExpr();
					if ( inputState.guessing==0 ) {
						((Expression)binit).setParens(false);
					}
				}
				else if ((_tokenSet_60.member(LA(1))) && (_tokenSet_61.member(LA(2)))) {
					binit=initializerElementLabel();
				}
				else if ((LA(1)==LCURLY)) {
					ilist=lcurlyInitializer();
					if ( inputState.guessing==0 ) {
						binit = new Initializer(ilist);
					}
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
			}
			else if ((LA(1)==LCURLY) && (_tokenSet_62.member(LA(2)))) {
				ilist=lcurlyInitializer();
				if ( inputState.guessing==0 ) {
					binit = new Initializer(ilist);
				}
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_63);
			} else {
			  throw ex;
			}
		}
		return binit;
	}
	
	public final List  pointerGroup() throws RecognitionException, TokenStreamException {
		List bp;
		
		
		bp = new ArrayList();
		Specifier temp = null;
		boolean b_const = false;
		boolean b_volatile = false;
		boolean b_restrict = false;
		boolean b_nvl = false;
		boolean b_nvl_wp = false;
		boolean b_nullable = false;
		
		
		try {      // for error handling
			{
			int _cnt163=0;
			_loop163:
			do {
				if ((LA(1)==STAR)) {
					match(STAR);
					{
					_loop162:
					do {
						if ((_tokenSet_13.member(LA(1)))) {
							temp=typeQualifier();
							if ( inputState.guessing==0 ) {
								
								if (temp == Specifier.CONST)
								b_const = true;
								else if (temp == Specifier.VOLATILE)
								b_volatile = true;
								else if (temp == Specifier.RESTRICT)
								b_restrict = true;
								else if (temp == Specifier.NVL)
								b_nvl = true;
								else if (temp == Specifier.NVL_WP)
								b_nvl_wp = true;
								else if (temp == Specifier._NULLABLE)
								b_nullable = true;
								
							}
						}
						else {
							break _loop162;
						}
						
					} while (true);
					}
					if ( inputState.guessing==0 ) {
						
						// 5 quals
						if (b_const && b_restrict && b_volatile && b_nvl && b_nvl_wp)
						bp.add(PointerSpecifier.CONST_RESTRICT_VOLATILE_NVL_NVL_WP);
						
						// 4 quals
						else if (b_const && b_restrict && b_volatile && b_nvl)
						bp.add(PointerSpecifier.CONST_RESTRICT_VOLATILE_NVL);
						else if (b_const && b_restrict && b_volatile && b_nvl_wp)
						bp.add(PointerSpecifier.CONST_RESTRICT_VOLATILE_NVL_WP);
						else if (b_const && b_restrict && b_nvl && b_nvl_wp)
						bp.add(PointerSpecifier.CONST_RESTRICT_NVL_NVL_WP);
						else if (b_const && b_volatile && b_nvl && b_nvl_wp)
						bp.add(PointerSpecifier.CONST_VOLATILE_NVL_NVL_WP);
						else if (b_restrict && b_volatile && b_nvl && b_nvl_wp)
						bp.add(PointerSpecifier.RESTRICT_VOLATILE_NVL_NVL_WP);
						
						// 3 quals
						else if (b_const && b_restrict && b_volatile)
						bp.add(PointerSpecifier.CONST_RESTRICT_VOLATILE);
						else if (b_const && b_restrict && b_nvl)
						bp.add(PointerSpecifier.CONST_RESTRICT_NVL);
						else if (b_const && b_volatile && b_nvl)
						bp.add(PointerSpecifier.CONST_VOLATILE_NVL);
						else if (b_restrict && b_volatile && b_nvl)
						bp.add(PointerSpecifier.RESTRICT_VOLATILE_NVL);
						else if (b_const && b_restrict && b_nvl_wp)
						bp.add(PointerSpecifier.CONST_RESTRICT_NVL_WP);
						else if (b_const && b_volatile && b_nvl_wp)
						bp.add(PointerSpecifier.CONST_VOLATILE_NVL_WP);
						else if (b_restrict && b_volatile && b_nvl_wp)
						bp.add(PointerSpecifier.RESTRICT_VOLATILE_NVL_WP);
						else if (b_const && b_nvl && b_nvl_wp)
						bp.add(PointerSpecifier.CONST_NVL_NVL_WP);
						else if (b_restrict && b_nvl && b_nvl_wp)
						bp.add(PointerSpecifier.RESTRICT_NVL_NVL_WP);
						else if (b_volatile && b_nvl && b_nvl_wp)
						bp.add(PointerSpecifier.VOLATILE_NVL_NVL_WP);
						
						// 2 quals
						else if (b_const && b_restrict)
						bp.add(PointerSpecifier.CONST_RESTRICT);
						else if (b_const && b_volatile)
						bp.add(PointerSpecifier.CONST_VOLATILE);
						else if (b_restrict && b_volatile)
						bp.add(PointerSpecifier.RESTRICT_VOLATILE);
						else if (b_const && b_nvl)
						bp.add(PointerSpecifier.CONST_NVL);
						else if (b_restrict && b_nvl)
						bp.add(PointerSpecifier.RESTRICT_NVL);
						else if (b_volatile && b_nvl)
						bp.add(PointerSpecifier.VOLATILE_NVL);
						else if (b_const && b_nvl_wp)
						bp.add(PointerSpecifier.CONST_NVL_WP);
						else if (b_restrict && b_nvl_wp)
						bp.add(PointerSpecifier.RESTRICT_NVL_WP);
						else if (b_volatile && b_nvl_wp)
						bp.add(PointerSpecifier.VOLATILE_NVL_WP);
						else if (b_nvl && b_nvl_wp)
						bp.add(PointerSpecifier.NVL_NVL_WP);
						
						// 1 qual
						else if (b_const)
						bp.add(PointerSpecifier.CONST);
						else if (b_restrict)
						bp.add(PointerSpecifier.RESTRICT);
						else if (b_volatile)
						bp.add(PointerSpecifier.VOLATILE);
						else if (b_nvl)
						bp.add(PointerSpecifier.NVL);
						else if (b_nvl_wp)
						bp.add(PointerSpecifier.NVL_WP);
						else if (b_nullable)
						bp.add(PointerSpecifier._NULLABLE);
						
						// 0 quals
						else
						bp.add(PointerSpecifier.UNQUALIFIED);
						
						b_const = false;
						b_volatile = false;
						b_restrict = false;
						b_nvl = false;
						b_nvl_wp = false;
						
					}
				}
				else {
					if ( _cnt163>=1 ) { break _loop163; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt163++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_64);
			} else {
			  throw ex;
			}
		}
		return bp;
	}
	
	public final List  idList() throws RecognitionException, TokenStreamException {
		List ilist;
		
		Token  idl1 = null;
		Token  idl2 = null;
		
		int i = 1;
		String name;
		Specifier temp = null;
		ilist = new ArrayList();
		
		
		try {      // for error handling
			idl1 = LT(1);
			match(ID);
			if ( inputState.guessing==0 ) {
				
				name = idl1.getText();
				ilist.add(
				new VariableDeclaration(new VariableDeclarator(new NameID(name))));
				
			}
			{
			_loop166:
			do {
				if ((LA(1)==COMMA) && (LA(2)==ID)) {
					match(COMMA);
					idl2 = LT(1);
					match(ID);
					if ( inputState.guessing==0 ) {
						
						name = idl2.getText();
						ilist.add(
						new VariableDeclaration(new VariableDeclarator(new NameID(name))));
						
					}
				}
				else {
					break _loop166;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_56);
			} else {
			  throw ex;
			}
		}
		return ilist;
	}
	
	public final Initializer  initializerElementLabel() throws RecognitionException, TokenStreamException {
		Initializer ret;
		
		Token  id = null;
		
		Expression expr1 = null, expr2=null, expr3=null;
		ret=null;
		List list = null;
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case LBRACKET:
			{
				match(LBRACKET);
				{
				boolean synPredMatched172 = false;
				if (((_tokenSet_19.member(LA(1))) && (_tokenSet_65.member(LA(2))))) {
					int _m172 = mark();
					synPredMatched172 = true;
					inputState.guessing++;
					try {
						{
						expr1=constExpr();
						match(VARARGS);
						}
					}
					catch (RecognitionException pe) {
						synPredMatched172 = false;
					}
					rewind(_m172);
inputState.guessing--;
				}
				if ( synPredMatched172 ) {
					expr1=rangeExpr();
				}
				else if ((_tokenSet_19.member(LA(1))) && (_tokenSet_66.member(LA(2)))) {
					expr2=constExpr();
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				match(RBRACKET);
				match(ASSIGN);
				{
				switch ( LA(1)) {
				case LPAREN:
				case ID:
				case Number:
				case StringLiteral:
				case LITERAL___asm:
				case STAR:
				case BAND:
				case PLUS:
				case MINUS:
				case LITERAL___builtin_nvl_get_root:
				case LITERAL___builtin_nvl_alloc_nv:
				case INC:
				case DEC:
				case LITERAL_sizeof:
				case LITERAL___alignof__:
				case LITERAL___builtin_va_arg:
				case LITERAL___builtin_offsetof:
				case BNOT:
				case LNOT:
				case LITERAL___real:
				case LITERAL___imag:
				case CharLiteral:
				{
					expr3=assignExpr();
					break;
				}
				case LCURLY:
				{
					list=lcurlyInitializer();
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				if ( inputState.guessing==0 ) {
					
					Expression sub = (expr1 != null)? expr1: expr2;
					if (sub != null) {
					Expression designator = new ArrayAccess(new NameID(""), sub);
					if (expr3 != null) {
					ret = new Initializer(designator, expr3);
					expr3.setParens(false);
					} else if (list != null) {
					ret = new Initializer(designator, list);
					}
					}
					
				}
				break;
			}
			case ID:
			{
				match(ID);
				match(COLON);
				break;
			}
			case DOT:
			{
				match(DOT);
				id = LT(1);
				match(ID);
				match(ASSIGN);
				{
				switch ( LA(1)) {
				case LPAREN:
				case ID:
				case Number:
				case StringLiteral:
				case LITERAL___asm:
				case STAR:
				case BAND:
				case PLUS:
				case MINUS:
				case LITERAL___builtin_nvl_get_root:
				case LITERAL___builtin_nvl_alloc_nv:
				case INC:
				case DEC:
				case LITERAL_sizeof:
				case LITERAL___alignof__:
				case LITERAL___builtin_va_arg:
				case LITERAL___builtin_offsetof:
				case BNOT:
				case LNOT:
				case LITERAL___real:
				case LITERAL___imag:
				case CharLiteral:
				{
					expr3=assignExpr();
					break;
				}
				case LCURLY:
				{
					list=lcurlyInitializer();
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				if ( inputState.guessing==0 ) {
					
					Expression designator = new AccessExpression(new NameID(""),
					AccessOperator.MEMBER_ACCESS, new NameID(id.getText()));
					if (expr3 != null) {
					ret = new Initializer(designator, expr3);
					expr3.setParens(false);
					} else if (list != null) {
					ret = new Initializer(designator, list);
					}
					
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_63);
			} else {
			  throw ex;
			}
		}
		return ret;
	}
	
	public final List  lcurlyInitializer() throws RecognitionException, TokenStreamException {
		List ilist;
		
		ilist = new ArrayList();
		
		try {      // for error handling
			match(LCURLY);
			{
			switch ( LA(1)) {
			case LCURLY:
			case LPAREN:
			case ID:
			case Number:
			case StringLiteral:
			case LITERAL___asm:
			case STAR:
			case LBRACKET:
			case DOT:
			case BAND:
			case PLUS:
			case MINUS:
			case LITERAL___builtin_nvl_get_root:
			case LITERAL___builtin_nvl_alloc_nv:
			case INC:
			case DEC:
			case LITERAL_sizeof:
			case LITERAL___alignof__:
			case LITERAL___builtin_va_arg:
			case LITERAL___builtin_offsetof:
			case BNOT:
			case LNOT:
			case LITERAL___real:
			case LITERAL___imag:
			case CharLiteral:
			{
				ilist=initializerList();
				{
				switch ( LA(1)) {
				case COMMA:
				{
					match(COMMA);
					break;
				}
				case RCURLY:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				break;
			}
			case RCURLY:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			match(RCURLY);
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_67);
			} else {
			  throw ex;
			}
		}
		return ilist;
	}
	
	public final Expression  rangeExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		ret_expr = null;
		
		try {      // for error handling
			constExpr();
			match(VARARGS);
			constExpr();
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_68);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final List  initializerList() throws RecognitionException, TokenStreamException {
		List ilist;
		
		Object init = null; ilist = new ArrayList();
		
		try {      // for error handling
			{
			init=initializer();
			if ( inputState.guessing==0 ) {
				ilist.add(init);
			}
			}
			{
			_loop181:
			do {
				if ((LA(1)==COMMA) && (_tokenSet_57.member(LA(2)))) {
					match(COMMA);
					init=initializer();
					if ( inputState.guessing==0 ) {
						ilist.add(init);
					}
				}
				else {
					break _loop181;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_50);
			} else {
			  throw ex;
			}
		}
		return ilist;
	}
	
	public final List  declaratorParamaterList() throws RecognitionException, TokenStreamException {
		List plist;
		
		plist = new ArrayList();
		
		try {      // for error handling
			match(LPAREN);
			{
			boolean synPredMatched193 = false;
			if (((_tokenSet_2.member(LA(1))) && (_tokenSet_69.member(LA(2))))) {
				int _m193 = mark();
				synPredMatched193 = true;
				inputState.guessing++;
				try {
					{
					declSpecifiers();
					}
				}
				catch (RecognitionException pe) {
					synPredMatched193 = false;
				}
				rewind(_m193);
inputState.guessing--;
			}
			if ( synPredMatched193 ) {
				plist=parameterTypeList();
			}
			else if ((_tokenSet_70.member(LA(1))) && (_tokenSet_15.member(LA(2)))) {
				{
				switch ( LA(1)) {
				case ID:
				{
					plist=idList();
					break;
				}
				case RPAREN:
				case COMMA:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			{
			switch ( LA(1)) {
			case COMMA:
			{
				match(COMMA);
				break;
			}
			case RPAREN:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			match(RPAREN);
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_15);
			} else {
			  throw ex;
			}
		}
		return plist;
	}
	
	public final List  parameterTypeList() throws RecognitionException, TokenStreamException {
		List ptlist;
		
		ptlist = new ArrayList(); Declaration pdecl = null;
		
		try {      // for error handling
			pdecl=parameterDeclaration();
			if ( inputState.guessing==0 ) {
				ptlist.add(pdecl);
			}
			{
			_loop199:
			do {
				if ((LA(1)==SEMI||LA(1)==COMMA) && (_tokenSet_2.member(LA(2)))) {
					{
					switch ( LA(1)) {
					case COMMA:
					{
						match(COMMA);
						break;
					}
					case SEMI:
					{
						match(SEMI);
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					pdecl=parameterDeclaration();
					if ( inputState.guessing==0 ) {
						ptlist.add(pdecl);
					}
				}
				else {
					break _loop199;
				}
				
			} while (true);
			}
			{
			if ((LA(1)==SEMI||LA(1)==COMMA) && (LA(2)==VARARGS)) {
				{
				switch ( LA(1)) {
				case COMMA:
				{
					match(COMMA);
					break;
				}
				case SEMI:
				{
					match(SEMI);
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				match(VARARGS);
				if ( inputState.guessing==0 ) {
					
					ptlist.add(
					new VariableDeclaration(new VariableDeclarator(new NameID("..."))));
					
				}
			}
			else if ((LA(1)==RPAREN||LA(1)==COMMA) && (_tokenSet_71.member(LA(2)))) {
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_56);
			} else {
			  throw ex;
			}
		}
		return ptlist;
	}
	
	public final Declaration  parameterDeclaration() throws RecognitionException, TokenStreamException {
		Declaration pdecl;
		
		
		pdecl =null;
		List dspec = null;
		Declarator decl = null;
		boolean prevhastypedef = hastypedef;
		hastypedef = false;
		
		
		try {      // for error handling
			dspec=declSpecifiers();
			{
			boolean synPredMatched205 = false;
			if (((_tokenSet_6.member(LA(1))) && (_tokenSet_72.member(LA(2))))) {
				int _m205 = mark();
				synPredMatched205 = true;
				inputState.guessing++;
				try {
					{
					declarator();
					}
				}
				catch (RecognitionException pe) {
					synPredMatched205 = false;
				}
				rewind(_m205);
inputState.guessing--;
			}
			if ( synPredMatched205 ) {
				decl=declarator();
			}
			else if ((_tokenSet_73.member(LA(1))) && (_tokenSet_74.member(LA(2)))) {
				decl=nonemptyAbstractDeclarator();
			}
			else if ((_tokenSet_75.member(LA(1)))) {
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			if ( inputState.guessing==0 ) {
				
				if (decl != null) {
				pdecl = new VariableDeclaration(dspec,decl);
				if (isFuncDef) {
				currproc.add(pdecl);
				}
				} else
				pdecl = new VariableDeclaration(
				dspec,new VariableDeclarator(new NameID("")));
				hastypedef = prevhastypedef;
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_75);
			} else {
			  throw ex;
			}
		}
		return pdecl;
	}
	
	public final Declarator  nonemptyAbstractDeclarator() throws RecognitionException, TokenStreamException {
		Declarator adecl;
		
		
		Expression expr1=null;
		List plist=null;
		List bp = null;
		Declarator tdecl = null;
		Specifier aspec = null;
		boolean isArraySpec = false;
		boolean isNested = false;
		List llist = new ArrayList();
		List tlist = null;
		boolean empty = true;
		adecl = null;
		
		
		try {      // for error handling
			{
			switch ( LA(1)) {
			case STAR:
			{
				bp=pointerGroup();
				{
				_loop348:
				do {
					switch ( LA(1)) {
					case LPAREN:
					{
						{
						match(LPAREN);
						{
						switch ( LA(1)) {
						case LITERAL_typedef:
						case LITERAL_volatile:
						case LITERAL_struct:
						case LITERAL_union:
						case LITERAL_enum:
						case LITERAL_auto:
						case LITERAL_register:
						case LITERAL___thread:
						case LITERAL_extern:
						case LITERAL_static:
						case LITERAL_inline:
						case LITERAL_const:
						case LITERAL_restrict:
						case LITERAL___nvl__:
						case LITERAL___nvl_wp__:
						case LITERAL_void:
						case LITERAL_char:
						case LITERAL_short:
						case LITERAL_int:
						case LITERAL_long:
						case LITERAL_float:
						case LITERAL_double:
						case LITERAL_signed:
						case LITERAL_unsigned:
						case LITERAL__Bool:
						case LITERAL__Complex:
						case LITERAL__Imaginary:
						case 36:
						case 37:
						case 38:
						case 39:
						case 40:
						case 41:
						case 42:
						case 43:
						case LITERAL_size_t:
						case 45:
						case 46:
						case 47:
						case 48:
						case 49:
						case LITERAL___global__:
						case LITERAL___shared__:
						case LITERAL___host__:
						case LITERAL___device__:
						case LITERAL___constant__:
						case LITERAL___noinline__:
						case LITERAL___kernel:
						case LITERAL___global:
						case LITERAL___local:
						case LITERAL___constant:
						case LITERAL_typeof:
						case LPAREN:
						case LITERAL___complex:
						case ID:
						case LITERAL___declspec:
						case LITERAL___attribute:
						case LITERAL___asm:
						case LITERAL___OSX_AVAILABLE_STARTING:
						case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
						case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
						case STAR:
						case LBRACKET:
						{
							{
							switch ( LA(1)) {
							case LPAREN:
							case STAR:
							case LBRACKET:
							{
								{
								tdecl=nonemptyAbstractDeclarator();
								}
								break;
							}
							case LITERAL_typedef:
							case LITERAL_volatile:
							case LITERAL_struct:
							case LITERAL_union:
							case LITERAL_enum:
							case LITERAL_auto:
							case LITERAL_register:
							case LITERAL___thread:
							case LITERAL_extern:
							case LITERAL_static:
							case LITERAL_inline:
							case LITERAL_const:
							case LITERAL_restrict:
							case LITERAL___nvl__:
							case LITERAL___nvl_wp__:
							case LITERAL_void:
							case LITERAL_char:
							case LITERAL_short:
							case LITERAL_int:
							case LITERAL_long:
							case LITERAL_float:
							case LITERAL_double:
							case LITERAL_signed:
							case LITERAL_unsigned:
							case LITERAL__Bool:
							case LITERAL__Complex:
							case LITERAL__Imaginary:
							case 36:
							case 37:
							case 38:
							case 39:
							case 40:
							case 41:
							case 42:
							case 43:
							case LITERAL_size_t:
							case 45:
							case 46:
							case 47:
							case 48:
							case 49:
							case LITERAL___global__:
							case LITERAL___shared__:
							case LITERAL___host__:
							case LITERAL___device__:
							case LITERAL___constant__:
							case LITERAL___noinline__:
							case LITERAL___kernel:
							case LITERAL___global:
							case LITERAL___local:
							case LITERAL___constant:
							case LITERAL_typeof:
							case LITERAL___complex:
							case ID:
							case LITERAL___declspec:
							case LITERAL___attribute:
							case LITERAL___asm:
							case LITERAL___OSX_AVAILABLE_STARTING:
							case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
							case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
							{
								plist=parameterTypeList();
								break;
							}
							default:
							{
								throw new NoViableAltException(LT(1), getFilename());
							}
							}
							}
							if ( inputState.guessing==0 ) {
								empty = false;
							}
							break;
						}
						case RPAREN:
						case COMMA:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						{
						switch ( LA(1)) {
						case COMMA:
						{
							match(COMMA);
							break;
						}
						case RPAREN:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						match(RPAREN);
						}
						if ( inputState.guessing==0 ) {
							
							if(empty)
							plist = new ArrayList();
							empty = true;
							
						}
						break;
					}
					case LBRACKET:
					{
						{
						match(LBRACKET);
						{
						switch ( LA(1)) {
						case LPAREN:
						case ID:
						case Number:
						case StringLiteral:
						case LITERAL___asm:
						case STAR:
						case BAND:
						case PLUS:
						case MINUS:
						case LITERAL___builtin_nvl_get_root:
						case LITERAL___builtin_nvl_alloc_nv:
						case INC:
						case DEC:
						case LITERAL_sizeof:
						case LITERAL___alignof__:
						case LITERAL___builtin_va_arg:
						case LITERAL___builtin_offsetof:
						case BNOT:
						case LNOT:
						case LITERAL___real:
						case LITERAL___imag:
						case CharLiteral:
						{
							expr1=expr();
							break;
						}
						case RBRACKET:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						match(RBRACKET);
						}
						if ( inputState.guessing==0 ) {
							
							isArraySpec = true;
							llist.add(expr1);
							
						}
						break;
					}
					default:
					{
						break _loop348;
					}
					}
				} while (true);
				}
				break;
			}
			case LPAREN:
			case LBRACKET:
			{
				{
				int _cnt357=0;
				_loop357:
				do {
					switch ( LA(1)) {
					case LPAREN:
					{
						{
						match(LPAREN);
						{
						switch ( LA(1)) {
						case LITERAL_typedef:
						case LITERAL_volatile:
						case LITERAL_struct:
						case LITERAL_union:
						case LITERAL_enum:
						case LITERAL_auto:
						case LITERAL_register:
						case LITERAL___thread:
						case LITERAL_extern:
						case LITERAL_static:
						case LITERAL_inline:
						case LITERAL_const:
						case LITERAL_restrict:
						case LITERAL___nvl__:
						case LITERAL___nvl_wp__:
						case LITERAL_void:
						case LITERAL_char:
						case LITERAL_short:
						case LITERAL_int:
						case LITERAL_long:
						case LITERAL_float:
						case LITERAL_double:
						case LITERAL_signed:
						case LITERAL_unsigned:
						case LITERAL__Bool:
						case LITERAL__Complex:
						case LITERAL__Imaginary:
						case 36:
						case 37:
						case 38:
						case 39:
						case 40:
						case 41:
						case 42:
						case 43:
						case LITERAL_size_t:
						case 45:
						case 46:
						case 47:
						case 48:
						case 49:
						case LITERAL___global__:
						case LITERAL___shared__:
						case LITERAL___host__:
						case LITERAL___device__:
						case LITERAL___constant__:
						case LITERAL___noinline__:
						case LITERAL___kernel:
						case LITERAL___global:
						case LITERAL___local:
						case LITERAL___constant:
						case LITERAL_typeof:
						case LPAREN:
						case LITERAL___complex:
						case ID:
						case LITERAL___declspec:
						case LITERAL___attribute:
						case LITERAL___asm:
						case LITERAL___OSX_AVAILABLE_STARTING:
						case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
						case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
						case STAR:
						case LBRACKET:
						{
							{
							switch ( LA(1)) {
							case LPAREN:
							case STAR:
							case LBRACKET:
							{
								{
								tdecl=nonemptyAbstractDeclarator();
								}
								break;
							}
							case LITERAL_typedef:
							case LITERAL_volatile:
							case LITERAL_struct:
							case LITERAL_union:
							case LITERAL_enum:
							case LITERAL_auto:
							case LITERAL_register:
							case LITERAL___thread:
							case LITERAL_extern:
							case LITERAL_static:
							case LITERAL_inline:
							case LITERAL_const:
							case LITERAL_restrict:
							case LITERAL___nvl__:
							case LITERAL___nvl_wp__:
							case LITERAL_void:
							case LITERAL_char:
							case LITERAL_short:
							case LITERAL_int:
							case LITERAL_long:
							case LITERAL_float:
							case LITERAL_double:
							case LITERAL_signed:
							case LITERAL_unsigned:
							case LITERAL__Bool:
							case LITERAL__Complex:
							case LITERAL__Imaginary:
							case 36:
							case 37:
							case 38:
							case 39:
							case 40:
							case 41:
							case 42:
							case 43:
							case LITERAL_size_t:
							case 45:
							case 46:
							case 47:
							case 48:
							case 49:
							case LITERAL___global__:
							case LITERAL___shared__:
							case LITERAL___host__:
							case LITERAL___device__:
							case LITERAL___constant__:
							case LITERAL___noinline__:
							case LITERAL___kernel:
							case LITERAL___global:
							case LITERAL___local:
							case LITERAL___constant:
							case LITERAL_typeof:
							case LITERAL___complex:
							case ID:
							case LITERAL___declspec:
							case LITERAL___attribute:
							case LITERAL___asm:
							case LITERAL___OSX_AVAILABLE_STARTING:
							case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
							case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
							{
								plist=parameterTypeList();
								break;
							}
							default:
							{
								throw new NoViableAltException(LT(1), getFilename());
							}
							}
							}
							if ( inputState.guessing==0 ) {
								empty = false;
							}
							break;
						}
						case RPAREN:
						case COMMA:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						{
						switch ( LA(1)) {
						case COMMA:
						{
							match(COMMA);
							break;
						}
						case RPAREN:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						match(RPAREN);
						}
						if ( inputState.guessing==0 ) {
							
							if (empty)
							plist = new ArrayList();
							empty = true;
							
						}
						break;
					}
					case LBRACKET:
					{
						{
						match(LBRACKET);
						{
						switch ( LA(1)) {
						case LPAREN:
						case ID:
						case Number:
						case StringLiteral:
						case LITERAL___asm:
						case STAR:
						case BAND:
						case PLUS:
						case MINUS:
						case LITERAL___builtin_nvl_get_root:
						case LITERAL___builtin_nvl_alloc_nv:
						case INC:
						case DEC:
						case LITERAL_sizeof:
						case LITERAL___alignof__:
						case LITERAL___builtin_va_arg:
						case LITERAL___builtin_offsetof:
						case BNOT:
						case LNOT:
						case LITERAL___real:
						case LITERAL___imag:
						case CharLiteral:
						{
							expr1=expr();
							break;
						}
						case RBRACKET:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						match(RBRACKET);
						}
						if ( inputState.guessing==0 ) {
							
							isArraySpec = true;
							llist.add(expr1);
							
						}
						break;
					}
					default:
					{
						if ( _cnt357>=1 ) { break _loop357; } else {throw new NoViableAltException(LT(1), getFilename());}
					}
					}
					_cnt357++;
				} while (true);
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				
				if (isArraySpec) {
				/* []+ */
				aspec = new ArraySpecifier(llist);
				tlist = new ArrayList();
				tlist.add(aspec);
				}
				NameID idex = null;
				// nested declarator (tlist == null ?)
				if (bp == null)
				bp = new ArrayList();
				// assume tlist == null
				if (tdecl != null) {
				//assert tlist == null : "Assertion (tlist == null) failed 2";
				if (tlist == null)
				tlist = new ArrayList();
				adecl = new NestedDeclarator(bp,tdecl,plist,tlist);
				} else {
				idex = new NameID("");
				if (plist != null) // assume tlist == null
				adecl = new ProcedureDeclarator(bp,idex,plist);
				else {
				if (tlist != null)
				adecl = new VariableDeclarator(bp,idex,tlist);
				else
				adecl = new VariableDeclarator(bp,idex);
				}
				}
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_76);
			} else {
			  throw ex;
			}
		}
		return adecl;
	}
	
	public final CompoundStatement  compoundStatement() throws RecognitionException, TokenStreamException {
		CompoundStatement stmt;
		
		Token  lcur = null;
		Token  rcur = null;
		
		stmt = null;
		int linenum = 0;
		SymbolTable prev_symtab = null;
		CompoundStatement prev_cstmt = null;
		
		
		try {      // for error handling
			lcur = LT(1);
			match(LCURLY);
			if ( inputState.guessing==0 ) {
				
				linenum = lcur.getLine();
				prev_symtab = symtab;
				prev_cstmt = curr_cstmt;
				stmt = new CompoundStatement();
				enterSymtab(stmt);
				stmt.setLineNumber(linenum);
				putPragma(lcur,prev_symtab);
				curr_cstmt = stmt;
				
			}
			{
			_loop240:
			do {
				boolean synPredMatched237 = false;
				if (((_tokenSet_77.member(LA(1))) && (_tokenSet_3.member(LA(2))))) {
					int _m237 = mark();
					synPredMatched237 = true;
					inputState.guessing++;
					try {
						{
						if ((LA(1)==LITERAL_typedef) && (true)) {
							match(LITERAL_typedef);
						}
						else if ((LA(1)==LITERAL___label__)) {
							match(LITERAL___label__);
						}
						else if ((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2)))) {
							declaration();
						}
						else {
							throw new NoViableAltException(LT(1), getFilename());
						}
						
						}
					}
					catch (RecognitionException pe) {
						synPredMatched237 = false;
					}
					rewind(_m237);
inputState.guessing--;
				}
				if ( synPredMatched237 ) {
					declarationList();
				}
				else {
					boolean synPredMatched239 = false;
					if (((_tokenSet_78.member(LA(1))) && (_tokenSet_79.member(LA(2))))) {
						int _m239 = mark();
						synPredMatched239 = true;
						inputState.guessing++;
						try {
							{
							nestedFunctionDef();
							}
						}
						catch (RecognitionException pe) {
							synPredMatched239 = false;
						}
						rewind(_m239);
inputState.guessing--;
					}
					if ( synPredMatched239 ) {
						nestedFunctionDef();
					}
					else {
						break _loop240;
					}
					}
				} while (true);
				}
				{
				switch ( LA(1)) {
				case LITERAL_typedef:
				case SEMI:
				case LCURLY:
				case LITERAL_volatile:
				case LITERAL_struct:
				case LITERAL_union:
				case LITERAL_enum:
				case LITERAL_auto:
				case LITERAL_register:
				case LITERAL___thread:
				case LITERAL_extern:
				case LITERAL_static:
				case LITERAL_inline:
				case LITERAL_const:
				case LITERAL_restrict:
				case LITERAL___nvl__:
				case LITERAL___nvl_wp__:
				case LITERAL_void:
				case LITERAL_char:
				case LITERAL_short:
				case LITERAL_int:
				case LITERAL_long:
				case LITERAL_float:
				case LITERAL_double:
				case LITERAL_signed:
				case LITERAL_unsigned:
				case LITERAL__Bool:
				case LITERAL__Complex:
				case LITERAL__Imaginary:
				case 36:
				case 37:
				case 38:
				case 39:
				case 40:
				case 41:
				case 42:
				case 43:
				case LITERAL_size_t:
				case 45:
				case 46:
				case 47:
				case 48:
				case 49:
				case LITERAL___global__:
				case LITERAL___shared__:
				case LITERAL___host__:
				case LITERAL___device__:
				case LITERAL___constant__:
				case LITERAL___noinline__:
				case LITERAL___kernel:
				case LITERAL___global:
				case LITERAL___local:
				case LITERAL___constant:
				case LITERAL_typeof:
				case LPAREN:
				case LITERAL___complex:
				case ID:
				case LITERAL___declspec:
				case Number:
				case StringLiteral:
				case LITERAL___attribute:
				case LITERAL___asm:
				case LITERAL___OSX_AVAILABLE_STARTING:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
				case STAR:
				case LITERAL_while:
				case LITERAL_do:
				case LITERAL_for:
				case LITERAL_goto:
				case LITERAL_continue:
				case LITERAL_break:
				case LITERAL_return:
				case LITERAL_case:
				case LITERAL_default:
				case LITERAL_if:
				case LITERAL_switch:
				case BAND:
				case PLUS:
				case MINUS:
				case LITERAL___builtin_nvl_get_root:
				case LITERAL___builtin_nvl_alloc_nv:
				case INC:
				case DEC:
				case LITERAL_sizeof:
				case LITERAL___alignof__:
				case LITERAL___builtin_va_arg:
				case LITERAL___builtin_offsetof:
				case BNOT:
				case LNOT:
				case LITERAL___real:
				case LITERAL___imag:
				case CharLiteral:
				{
					statementList();
					break;
				}
				case RCURLY:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				rcur = LT(1);
				match(RCURLY);
				if ( inputState.guessing==0 ) {
					
					linenum = rcur.getLine();
					putPragma(rcur,symtab);
					curr_cstmt = prev_cstmt;
					exitSymtab();
					
				}
			}
			catch (RecognitionException ex) {
				if (inputState.guessing==0) {
					
					System.err.println("Cetus does not support C99 features yet");
					System.err.println("Please check if Declarations and Statements are interleaved");
					reportError(ex);
					
				} else {
					throw ex;
				}
			}
			return stmt;
		}
		
	public final void declarationList() throws RecognitionException, TokenStreamException {
		
		Declaration decl=null;List tlist = new ArrayList();
		
		try {      // for error handling
			{
			int _cnt224=0;
			_loop224:
			do {
				if ((LA(1)==LITERAL___label__) && (LA(2)==ID)) {
					localLabelDeclaration();
				}
				else {
					boolean synPredMatched223 = false;
					if (((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2))))) {
						int _m223 = mark();
						synPredMatched223 = true;
						inputState.guessing++;
						try {
							{
							declarationPredictor();
							}
						}
						catch (RecognitionException pe) {
							synPredMatched223 = false;
						}
						rewind(_m223);
inputState.guessing--;
					}
					if ( synPredMatched223 ) {
						decl=declaration();
						if ( inputState.guessing==0 ) {
							if(decl != null ) curr_cstmt.addDeclaration(decl, LT(-1).getLine());
						}
					}
					else {
						if ( _cnt224>=1 ) { break _loop224; } else {throw new NoViableAltException(LT(1), getFilename());}
					}
					}
					_cnt224++;
				} while (true);
				}
			}
			catch (RecognitionException ex) {
				if (inputState.guessing==0) {
					reportError(ex);
					recover(ex,_tokenSet_80);
				} else {
				  throw ex;
				}
			}
		}
		
	public final void localLabelDeclaration() throws RecognitionException, TokenStreamException {
		
		
		try {      // for error handling
			{
			match(LITERAL___label__);
			match(ID);
			{
			_loop230:
			do {
				if ((LA(1)==COMMA) && (LA(2)==ID)) {
					match(COMMA);
					match(ID);
				}
				else {
					break _loop230;
				}
				
			} while (true);
			}
			{
			switch ( LA(1)) {
			case COMMA:
			{
				match(COMMA);
				break;
			}
			case SEMI:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			int _cnt233=0;
			_loop233:
			do {
				if ((LA(1)==SEMI) && (_tokenSet_80.member(LA(2)))) {
					match(SEMI);
				}
				else {
					if ( _cnt233>=1 ) { break _loop233; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt233++;
			} while (true);
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_80);
			} else {
			  throw ex;
			}
		}
	}
	
	public final void declarationPredictor() throws RecognitionException, TokenStreamException {
		
		Declaration decl=null;
		
		try {      // for error handling
			{
			if ((LA(1)==LITERAL_typedef) && (LA(2)==EOF)) {
				match(LITERAL_typedef);
			}
			else if ((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2)))) {
				decl=declaration();
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
	}
	
	public final void nestedFunctionDef() throws RecognitionException, TokenStreamException {
		
		Declarator decl=null;
		
		try {      // for error handling
			{
			switch ( LA(1)) {
			case LITERAL_auto:
			{
				match(LITERAL_auto);
				break;
			}
			case LITERAL_volatile:
			case LITERAL_struct:
			case LITERAL_union:
			case LITERAL_enum:
			case LITERAL___thread:
			case LITERAL_extern:
			case LITERAL_static:
			case LITERAL_inline:
			case LITERAL_const:
			case LITERAL_restrict:
			case LITERAL___nvl__:
			case LITERAL___nvl_wp__:
			case LITERAL_void:
			case LITERAL_char:
			case LITERAL_short:
			case LITERAL_int:
			case LITERAL_long:
			case LITERAL_float:
			case LITERAL_double:
			case LITERAL_signed:
			case LITERAL_unsigned:
			case LITERAL__Bool:
			case LITERAL__Complex:
			case LITERAL__Imaginary:
			case 36:
			case 37:
			case 38:
			case 39:
			case 40:
			case 41:
			case 42:
			case 43:
			case LITERAL_size_t:
			case 45:
			case 46:
			case 47:
			case 48:
			case 49:
			case LITERAL___global__:
			case LITERAL___shared__:
			case LITERAL___host__:
			case LITERAL___device__:
			case LITERAL___constant__:
			case LITERAL___noinline__:
			case LITERAL___kernel:
			case LITERAL___global:
			case LITERAL___local:
			case LITERAL___constant:
			case LITERAL_typeof:
			case LPAREN:
			case LITERAL___complex:
			case ID:
			case LITERAL___declspec:
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			case STAR:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			{
			boolean synPredMatched246 = false;
			if (((_tokenSet_10.member(LA(1))) && (_tokenSet_11.member(LA(2))))) {
				int _m246 = mark();
				synPredMatched246 = true;
				inputState.guessing++;
				try {
					{
					functionDeclSpecifiers();
					}
				}
				catch (RecognitionException pe) {
					synPredMatched246 = false;
				}
				rewind(_m246);
inputState.guessing--;
			}
			if ( synPredMatched246 ) {
				functionDeclSpecifiers();
			}
			else if ((_tokenSet_6.member(LA(1))) && (_tokenSet_81.member(LA(2)))) {
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			decl=declarator();
			{
			_loop248:
			do {
				if ((_tokenSet_2.member(LA(1)))) {
					declaration();
				}
				else {
					break _loop248;
				}
				
			} while (true);
			}
			compoundStatement();
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_80);
			} else {
			  throw ex;
			}
		}
	}
	
	public final void statementList() throws RecognitionException, TokenStreamException {
		
		Statement statb = null; Declaration decl = null;
		
		try {      // for error handling
			{
			int _cnt251=0;
			_loop251:
			do {
				if (((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2))))&&(isTypedefName (LT(1).getText()))) {
					decl=declaration();
					if ( inputState.guessing==0 ) {
						
						DeclarationStatement declStat = new DeclarationStatement(decl);
						declStat.setLineNumber(LT(-1).getLine());
						curr_cstmt.addStatement(declStat);
						
					}
				}
				else if ((_tokenSet_82.member(LA(1))) && (_tokenSet_83.member(LA(2)))) {
					statb=statement();
					if ( inputState.guessing==0 ) {
						curr_cstmt.addStatement(statb);
					}
				}
				else if ((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2)))) {
					decl=declaration();
					if ( inputState.guessing==0 ) {
						
						DeclarationStatement declStat = new DeclarationStatement(decl);
						declStat.setLineNumber(LT(-1).getLine());
						curr_cstmt.addStatement(declStat);
						
					}
				}
				else {
					if ( _cnt251>=1 ) { break _loop251; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				
				_cnt251++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_37);
			} else {
			  throw ex;
			}
		}
	}
	
	public final Statement  statement() throws RecognitionException, TokenStreamException {
		Statement statb;
		
		Token  tsemi = null;
		Token  exprsemi = null;
		Token  twhile = null;
		Token  tdo = null;
		Token  tfor = null;
		Token  tgoto = null;
		Token  gotoTarget = null;
		Token  tcontinue = null;
		Token  tbreak = null;
		Token  treturn = null;
		Token  lid = null;
		Token  tcase = null;
		Token  tdefault = null;
		Token  tif = null;
		Token  tswitch = null;
		
		Expression stmtb_expr;
		statb = null;
		Expression expr1=null, expr2=null, expr3=null;
		Statement stmt1=null,stmt2=null;
		Declaration decl = null;
		int a=0;
		int sline = 0;
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case SEMI:
			{
				tsemi = LT(1);
				match(SEMI);
				if ( inputState.guessing==0 ) {
					
					sline = tsemi.getLine();
					statb = new NullStatement();
					putPragma(tsemi,symtab);
					
				}
				break;
			}
			case LCURLY:
			{
				statb=compoundStatement();
				break;
			}
			case LITERAL_while:
			{
				twhile = LT(1);
				match(LITERAL_while);
				match(LPAREN);
				if ( inputState.guessing==0 ) {
					
					sline = twhile.getLine();
					putPragma(twhile,symtab);
					
				}
				expr1=expr();
				match(RPAREN);
				stmt1=statement();
				if ( inputState.guessing==0 ) {
					
					statb = new WhileLoop(expr1, stmt1);
					statb.setLineNumber(sline);
					
				}
				break;
			}
			case LITERAL_do:
			{
				tdo = LT(1);
				match(LITERAL_do);
				if ( inputState.guessing==0 ) {
					
					sline = tdo.getLine();
					putPragma(tdo,symtab);
					
				}
				stmt1=statement();
				match(LITERAL_while);
				match(LPAREN);
				expr1=expr();
				match(RPAREN);
				match(SEMI);
				if ( inputState.guessing==0 ) {
					
					statb = new DoLoop(stmt1, expr1);
					statb.setLineNumber(sline);
					
				}
				break;
			}
			case LITERAL_for:
			{
				tfor = LT(1);
				match(LITERAL_for);
				if ( inputState.guessing==0 ) {
					
					sline = tfor.getLine();
					putPragma(tfor,symtab);
					
				}
				match(LPAREN);
				{
				if (((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2))))&&(isTypedefName (LT(1).getText()))) {
					decl=declaration();
				}
				else if ((_tokenSet_84.member(LA(1))) && (_tokenSet_85.member(LA(2)))) {
					{
					switch ( LA(1)) {
					case LPAREN:
					case ID:
					case Number:
					case StringLiteral:
					case LITERAL___asm:
					case STAR:
					case BAND:
					case PLUS:
					case MINUS:
					case LITERAL___builtin_nvl_get_root:
					case LITERAL___builtin_nvl_alloc_nv:
					case INC:
					case DEC:
					case LITERAL_sizeof:
					case LITERAL___alignof__:
					case LITERAL___builtin_va_arg:
					case LITERAL___builtin_offsetof:
					case BNOT:
					case LNOT:
					case LITERAL___real:
					case LITERAL___imag:
					case CharLiteral:
					{
						expr1=expr();
						break;
					}
					case SEMI:
					{
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					match(SEMI);
				}
				else if ((_tokenSet_2.member(LA(1))) && (_tokenSet_3.member(LA(2)))) {
					decl=declaration();
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				{
				switch ( LA(1)) {
				case LPAREN:
				case ID:
				case Number:
				case StringLiteral:
				case LITERAL___asm:
				case STAR:
				case BAND:
				case PLUS:
				case MINUS:
				case LITERAL___builtin_nvl_get_root:
				case LITERAL___builtin_nvl_alloc_nv:
				case INC:
				case DEC:
				case LITERAL_sizeof:
				case LITERAL___alignof__:
				case LITERAL___builtin_va_arg:
				case LITERAL___builtin_offsetof:
				case BNOT:
				case LNOT:
				case LITERAL___real:
				case LITERAL___imag:
				case CharLiteral:
				{
					expr2=expr();
					break;
				}
				case SEMI:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				match(SEMI);
				{
				switch ( LA(1)) {
				case LPAREN:
				case ID:
				case Number:
				case StringLiteral:
				case LITERAL___asm:
				case STAR:
				case BAND:
				case PLUS:
				case MINUS:
				case LITERAL___builtin_nvl_get_root:
				case LITERAL___builtin_nvl_alloc_nv:
				case INC:
				case DEC:
				case LITERAL_sizeof:
				case LITERAL___alignof__:
				case LITERAL___builtin_va_arg:
				case LITERAL___builtin_offsetof:
				case BNOT:
				case LNOT:
				case LITERAL___real:
				case LITERAL___imag:
				case CharLiteral:
				{
					expr3=expr();
					break;
				}
				case RPAREN:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				match(RPAREN);
				stmt1=statement();
				if ( inputState.guessing==0 ) {
					
					if (expr1 != null) {
					statb = new ForLoop(new ExpressionStatement(expr1), expr2, expr3, stmt1);
					} else {
					statb = new ForLoop(new NullStatement(), expr2, expr3, stmt1);
					if (decl != null) {
					// Constructs a legal ANSI C scope to handle C99's block scope
					CompoundStatement cstmt = new CompoundStatement();
					cstmt.addDeclaration(decl);
					cstmt.addStatement(statb);
					statb = cstmt;
					}
					}
					statb.setLineNumber(sline);
					
				}
				break;
			}
			case LITERAL_goto:
			{
				tgoto = LT(1);
				match(LITERAL_goto);
				if ( inputState.guessing==0 ) {
					
					sline = tgoto.getLine();
					putPragma(tgoto,symtab);
					
				}
				gotoTarget = LT(1);
				match(ID);
				match(SEMI);
				if ( inputState.guessing==0 ) {
					
					statb = new GotoStatement(new NameID(gotoTarget.getText()));
					statb.setLineNumber(sline);
					
				}
				break;
			}
			case LITERAL_continue:
			{
				tcontinue = LT(1);
				match(LITERAL_continue);
				match(SEMI);
				if ( inputState.guessing==0 ) {
					
					sline = tcontinue.getLine();
					statb = new ContinueStatement();
					statb.setLineNumber(sline);
					putPragma(tcontinue,symtab);
					
				}
				break;
			}
			case LITERAL_break:
			{
				tbreak = LT(1);
				match(LITERAL_break);
				match(SEMI);
				if ( inputState.guessing==0 ) {
					
					sline = tbreak.getLine();
					statb = new BreakStatement();
					statb.setLineNumber(sline);
					putPragma(tbreak,symtab);
					
				}
				break;
			}
			case LITERAL_return:
			{
				treturn = LT(1);
				match(LITERAL_return);
				if ( inputState.guessing==0 ) {
					
					sline = treturn.getLine();
					
				}
				{
				switch ( LA(1)) {
				case LPAREN:
				case ID:
				case Number:
				case StringLiteral:
				case LITERAL___asm:
				case STAR:
				case BAND:
				case PLUS:
				case MINUS:
				case LITERAL___builtin_nvl_get_root:
				case LITERAL___builtin_nvl_alloc_nv:
				case INC:
				case DEC:
				case LITERAL_sizeof:
				case LITERAL___alignof__:
				case LITERAL___builtin_va_arg:
				case LITERAL___builtin_offsetof:
				case BNOT:
				case LNOT:
				case LITERAL___real:
				case LITERAL___imag:
				case CharLiteral:
				{
					expr1=expr();
					break;
				}
				case SEMI:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				match(SEMI);
				if ( inputState.guessing==0 ) {
					
					if (expr1 != null)
					statb=new ReturnStatement(expr1);
					else
					statb=new ReturnStatement();
					statb.setLineNumber(sline);
					putPragma(treturn,symtab);
					
				}
				break;
			}
			case LITERAL_case:
			{
				tcase = LT(1);
				match(LITERAL_case);
				if ( inputState.guessing==0 ) {
					
					sline = tcase.getLine();
					
				}
				{
				boolean synPredMatched262 = false;
				if (((_tokenSet_19.member(LA(1))) && (_tokenSet_65.member(LA(2))))) {
					int _m262 = mark();
					synPredMatched262 = true;
					inputState.guessing++;
					try {
						{
						constExpr();
						match(VARARGS);
						}
					}
					catch (RecognitionException pe) {
						synPredMatched262 = false;
					}
					rewind(_m262);
inputState.guessing--;
				}
				if ( synPredMatched262 ) {
					expr1=rangeExpr();
				}
				else if ((_tokenSet_19.member(LA(1))) && (_tokenSet_86.member(LA(2)))) {
					expr1=constExpr();
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				if ( inputState.guessing==0 ) {
					
					statb = new Case(expr1);
					statb.setLineNumber(sline);
					putPragma(tcase,symtab);
					
				}
				match(COLON);
				{
				if ((_tokenSet_82.member(LA(1))) && (_tokenSet_87.member(LA(2)))) {
					stmt1=statement();
					if ( inputState.guessing==0 ) {
						
						CompoundStatement cstmt = new CompoundStatement();
						cstmt.addStatement(statb);
						statb = cstmt;
						cstmt.addStatement(stmt1);
						
					}
				}
				else if ((_tokenSet_88.member(LA(1))) && (_tokenSet_89.member(LA(2)))) {
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				break;
			}
			case LITERAL_default:
			{
				tdefault = LT(1);
				match(LITERAL_default);
				if ( inputState.guessing==0 ) {
					
					sline = tdefault.getLine();
					statb = new Default();
					statb.setLineNumber(sline);
					putPragma(tdefault,symtab);
					
				}
				match(COLON);
				{
				if ((_tokenSet_82.member(LA(1))) && (_tokenSet_87.member(LA(2)))) {
					stmt1=statement();
					if ( inputState.guessing==0 ) {
						
						CompoundStatement cstmt = new CompoundStatement();
						cstmt.addStatement(statb);
						statb = cstmt;
						cstmt.addStatement(stmt1);
						
					}
				}
				else if ((_tokenSet_88.member(LA(1))) && (_tokenSet_89.member(LA(2)))) {
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				break;
			}
			case LITERAL_if:
			{
				tif = LT(1);
				match(LITERAL_if);
				if ( inputState.guessing==0 ) {
					
					sline = tif.getLine();
					putPragma(tif,symtab);
					
				}
				match(LPAREN);
				expr1=expr();
				match(RPAREN);
				stmt1=statement();
				{
				if ((LA(1)==LITERAL_else) && (_tokenSet_82.member(LA(2)))) {
					match(LITERAL_else);
					stmt2=statement();
				}
				else if ((_tokenSet_88.member(LA(1))) && (_tokenSet_89.member(LA(2)))) {
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				if ( inputState.guessing==0 ) {
					
					if (stmt2 != null)
					statb = new IfStatement(expr1,stmt1,stmt2);
					else
					statb = new IfStatement(expr1,stmt1);
					statb.setLineNumber(sline);
					
				}
				break;
			}
			case LITERAL_switch:
			{
				tswitch = LT(1);
				match(LITERAL_switch);
				match(LPAREN);
				if ( inputState.guessing==0 ) {
					
					sline = tswitch.getLine();
					
				}
				expr1=expr();
				match(RPAREN);
				if ( inputState.guessing==0 ) {
					
					statb = new SwitchStatement(expr1);
					statb.setLineNumber(sline);
					putPragma(tswitch,symtab);
					
				}
				stmt1=statement();
				if ( inputState.guessing==0 ) {
					
					((SwitchStatement)statb).setBody((CompoundStatement)stmt1);
					unwrapSwitch((SwitchStatement)statb);
					
				}
				break;
			}
			default:
				if ((_tokenSet_19.member(LA(1))) && (_tokenSet_85.member(LA(2)))) {
					stmtb_expr=expr();
					exprsemi = LT(1);
					match(SEMI);
					if ( inputState.guessing==0 ) {
						
						sline = exprsemi.getLine();
						putPragma(exprsemi,symtab);
						/* I really shouldn't do this test */
						statb = new ExpressionStatement(stmtb_expr);
						statb.setLineNumber(sline);
						
					}
				}
				else if ((LA(1)==ID) && (LA(2)==COLON)) {
					lid = LT(1);
					match(ID);
					match(COLON);
					if ( inputState.guessing==0 ) {
						
						sline = lid.getLine();
						Object o = null;
						Declaration target = null;
						statb = new Label(new NameID(lid.getText()));
						statb.setLineNumber(sline);
						putPragma(lid,symtab);
						
					}
					{
					if (((LA(1) >= LITERAL___attribute && LA(1) <= LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG)) && (LA(2)==LPAREN)) {
						attributeDecl();
					}
					else if ((_tokenSet_88.member(LA(1))) && (_tokenSet_89.member(LA(2)))) {
					}
					else {
						throw new NoViableAltException(LT(1), getFilename());
					}
					
					}
					{
					if ((_tokenSet_82.member(LA(1))) && (_tokenSet_87.member(LA(2)))) {
						stmt1=statement();
						if ( inputState.guessing==0 ) {
							
							CompoundStatement cstmt = new CompoundStatement();
							cstmt.addStatement(statb);
							statb = cstmt;
							cstmt.addStatement(stmt1);
							
						}
					}
					else if ((_tokenSet_88.member(LA(1))) && (_tokenSet_89.member(LA(2)))) {
					}
					else {
						throw new NoViableAltException(LT(1), getFilename());
					}
					
					}
				}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_88);
			} else {
			  throw ex;
			}
		}
		return statb;
	}
	
	public final Expression  conditionalExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		ret_expr=null; Expression expr1=null,expr2=null,expr3=null;
		
		try {      // for error handling
			expr1=logicalOrExpr();
			if ( inputState.guessing==0 ) {
				ret_expr = expr1;
			}
			{
			switch ( LA(1)) {
			case QUESTION:
			{
				match(QUESTION);
				{
				switch ( LA(1)) {
				case LPAREN:
				case ID:
				case Number:
				case StringLiteral:
				case LITERAL___asm:
				case STAR:
				case BAND:
				case PLUS:
				case MINUS:
				case LITERAL___builtin_nvl_get_root:
				case LITERAL___builtin_nvl_alloc_nv:
				case INC:
				case DEC:
				case LITERAL_sizeof:
				case LITERAL___alignof__:
				case LITERAL___builtin_va_arg:
				case LITERAL___builtin_offsetof:
				case BNOT:
				case LNOT:
				case LITERAL___real:
				case LITERAL___imag:
				case CharLiteral:
				{
					expr2=expr();
					break;
				}
				case COLON:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				match(COLON);
				expr3=conditionalExpr();
				if ( inputState.guessing==0 ) {
					ret_expr = new ConditionalExpression(expr1,expr2,expr3);
				}
				break;
			}
			case SEMI:
			case VARARGS:
			case RCURLY:
			case RPAREN:
			case COMMA:
			case COLON:
			case ASSIGN:
			case LITERAL___attribute:
			case LITERAL___asm:
			case LITERAL___OSX_AVAILABLE_STARTING:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
			case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
			case RBRACKET:
			case DIV_ASSIGN:
			case PLUS_ASSIGN:
			case MINUS_ASSIGN:
			case STAR_ASSIGN:
			case MOD_ASSIGN:
			case RSHIFT_ASSIGN:
			case LSHIFT_ASSIGN:
			case BAND_ASSIGN:
			case BOR_ASSIGN:
			case BXOR_ASSIGN:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_90);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final AssignmentOperator  assignOperator() throws RecognitionException, TokenStreamException {
		AssignmentOperator code;
		
		code = null;
		
		try {      // for error handling
			switch ( LA(1)) {
			case ASSIGN:
			{
				match(ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.NORMAL;
				}
				break;
			}
			case DIV_ASSIGN:
			{
				match(DIV_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.DIVIDE;
				}
				break;
			}
			case PLUS_ASSIGN:
			{
				match(PLUS_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.ADD;
				}
				break;
			}
			case MINUS_ASSIGN:
			{
				match(MINUS_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.SUBTRACT;
				}
				break;
			}
			case STAR_ASSIGN:
			{
				match(STAR_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.MULTIPLY;
				}
				break;
			}
			case MOD_ASSIGN:
			{
				match(MOD_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.MODULUS;
				}
				break;
			}
			case RSHIFT_ASSIGN:
			{
				match(RSHIFT_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.SHIFT_RIGHT;
				}
				break;
			}
			case LSHIFT_ASSIGN:
			{
				match(LSHIFT_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.SHIFT_LEFT;
				}
				break;
			}
			case BAND_ASSIGN:
			{
				match(BAND_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.BITWISE_AND;
				}
				break;
			}
			case BOR_ASSIGN:
			{
				match(BOR_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.BITWISE_INCLUSIVE_OR;
				}
				break;
			}
			case BXOR_ASSIGN:
			{
				match(BXOR_ASSIGN);
				if ( inputState.guessing==0 ) {
					code = AssignmentOperator.BITWISE_EXCLUSIVE_OR;
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_19);
			} else {
			  throw ex;
			}
		}
		return code;
	}
	
	public final Expression  logicalOrExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=logicalAndExpr();
			{
			_loop275:
			do {
				if ((LA(1)==LOR)) {
					match(LOR);
					expr1=logicalAndExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new BinaryExpression(ret_expr,BinaryOperator.LOGICAL_OR,expr1);
					}
				}
				else {
					break _loop275;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_91);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  logicalAndExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=inclusiveOrExpr();
			{
			_loop278:
			do {
				if ((LA(1)==LAND)) {
					match(LAND);
					expr1=inclusiveOrExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new BinaryExpression(ret_expr,BinaryOperator.LOGICAL_AND,expr1);
					}
				}
				else {
					break _loop278;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_92);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  inclusiveOrExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=exclusiveOrExpr();
			{
			_loop281:
			do {
				if ((LA(1)==BOR)) {
					match(BOR);
					expr1=exclusiveOrExpr();
					if ( inputState.guessing==0 ) {
						
						ret_expr = new BinaryExpression
						(ret_expr,BinaryOperator.BITWISE_INCLUSIVE_OR,expr1);
						
					}
				}
				else {
					break _loop281;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_93);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  exclusiveOrExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=bitAndExpr();
			{
			_loop284:
			do {
				if ((LA(1)==BXOR)) {
					match(BXOR);
					expr1=bitAndExpr();
					if ( inputState.guessing==0 ) {
						
						ret_expr = new BinaryExpression
						(ret_expr,BinaryOperator.BITWISE_EXCLUSIVE_OR,expr1);
						
					}
				}
				else {
					break _loop284;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_94);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  bitAndExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=equalityExpr();
			{
			_loop287:
			do {
				if ((LA(1)==BAND)) {
					match(BAND);
					expr1=equalityExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new BinaryExpression(ret_expr,BinaryOperator.BITWISE_AND,expr1);
					}
				}
				else {
					break _loop287;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_95);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  equalityExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=relationalExpr();
			{
			_loop291:
			do {
				if ((LA(1)==EQUAL||LA(1)==NOT_EQUAL)) {
					{
					switch ( LA(1)) {
					case EQUAL:
					{
						match(EQUAL);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.COMPARE_EQ;
						}
						break;
					}
					case NOT_EQUAL:
					{
						match(NOT_EQUAL);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.COMPARE_NE;
						}
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					expr1=relationalExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new BinaryExpression(ret_expr,code,expr1);
					}
				}
				else {
					break _loop291;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_96);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  relationalExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=shiftExpr();
			{
			_loop295:
			do {
				if (((LA(1) >= LT && LA(1) <= GTE))) {
					{
					switch ( LA(1)) {
					case LT:
					{
						match(LT);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.COMPARE_LT;
						}
						break;
					}
					case LTE:
					{
						match(LTE);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.COMPARE_LE;
						}
						break;
					}
					case GT:
					{
						match(GT);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.COMPARE_GT;
						}
						break;
					}
					case GTE:
					{
						match(GTE);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.COMPARE_GE;
						}
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					expr1=shiftExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new BinaryExpression(ret_expr,code,expr1);
					}
				}
				else {
					break _loop295;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_97);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  shiftExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=additiveExpr();
			{
			_loop299:
			do {
				if ((LA(1)==LSHIFT||LA(1)==RSHIFT)) {
					{
					switch ( LA(1)) {
					case LSHIFT:
					{
						match(LSHIFT);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.SHIFT_LEFT;
						}
						break;
					}
					case RSHIFT:
					{
						match(RSHIFT);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.SHIFT_RIGHT;
						}
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					expr1=additiveExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new BinaryExpression(ret_expr,code,expr1);
					}
				}
				else {
					break _loop299;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_98);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  additiveExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=multExpr();
			{
			_loop303:
			do {
				if ((LA(1)==PLUS||LA(1)==MINUS)) {
					{
					switch ( LA(1)) {
					case PLUS:
					{
						match(PLUS);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.ADD;
						}
						break;
					}
					case MINUS:
					{
						match(MINUS);
						if ( inputState.guessing==0 ) {
							code=BinaryOperator.SUBTRACT;
						}
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					expr1=multExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new BinaryExpression(ret_expr,code,expr1);
					}
				}
				else {
					break _loop303;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_99);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  multExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1, expr2; ret_expr=null;
		BinaryOperator code = null;
		
		
		try {      // for error handling
			ret_expr=castExpr();
			{
			_loop307:
			do {
				if ((_tokenSet_100.member(LA(1)))) {
					{
					switch ( LA(1)) {
					case STAR:
					{
						match(STAR);
						if ( inputState.guessing==0 ) {
							code = BinaryOperator.MULTIPLY;
						}
						break;
					}
					case DIV:
					{
						match(DIV);
						if ( inputState.guessing==0 ) {
							code=BinaryOperator.DIVIDE;
						}
						break;
					}
					case MOD:
					{
						match(MOD);
						if ( inputState.guessing==0 ) {
							code=BinaryOperator.MODULUS;
						}
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
					expr1=castExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new BinaryExpression(ret_expr,code,expr1);
					}
				}
				else {
					break _loop307;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_101);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  castExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		ret_expr = null;
		Expression expr1 = null;
		List tname = null;
		List init = null;
		
		
		try {      // for error handling
			boolean synPredMatched336 = false;
			if (((LA(1)==LPAREN) && (_tokenSet_30.member(LA(2))))) {
				int _m336 = mark();
				synPredMatched336 = true;
				inputState.guessing++;
				try {
					{
					match(LPAREN);
					typeName2();
					match(RPAREN);
					}
				}
				catch (RecognitionException pe) {
					synPredMatched336 = false;
				}
				rewind(_m336);
inputState.guessing--;
			}
			if ( synPredMatched336 ) {
				match(LPAREN);
				tname=typeName2();
				match(RPAREN);
				{
				switch ( LA(1)) {
				case LPAREN:
				case ID:
				case Number:
				case StringLiteral:
				case LITERAL___asm:
				case STAR:
				case BAND:
				case PLUS:
				case MINUS:
				case LITERAL___builtin_nvl_get_root:
				case LITERAL___builtin_nvl_alloc_nv:
				case INC:
				case DEC:
				case LITERAL_sizeof:
				case LITERAL___alignof__:
				case LITERAL___builtin_va_arg:
				case LITERAL___builtin_offsetof:
				case BNOT:
				case LNOT:
				case LITERAL___real:
				case LITERAL___imag:
				case CharLiteral:
				{
					expr1=castExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new Typecast(tname,expr1);
					}
					break;
				}
				case LCURLY:
				{
					init=lcurlyInitializer();
					if ( inputState.guessing==0 ) {
						ret_expr = new CompoundLiteral(tname, init);
					}
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
			}
			else if ((_tokenSet_19.member(LA(1))) && (_tokenSet_102.member(LA(2)))) {
				ret_expr=unaryExpr();
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_67);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final List  typeName() throws RecognitionException, TokenStreamException {
		List tname;
		
		
		tname=null;
		Declarator decl = null;
		
		
		try {      // for error handling
			tname=specifierQualifierList();
			{
			switch ( LA(1)) {
			case LPAREN:
			case STAR:
			case LBRACKET:
			{
				decl=nonemptyAbstractDeclarator();
				if ( inputState.guessing==0 ) {
					tname.add(decl);
				}
				break;
			}
			case EOF:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
		return tname;
	}
	
	public final Declarator  nonemptyAbstractDeclarator2(
		List tname, boolean isRoot
	) throws RecognitionException, TokenStreamException {
		Declarator adecl;
		
		
		Expression expr1=null;
		List plist=null;
		List bp = null;
		Declarator tdecl = null;
		Specifier aspec = null;
		boolean isArraySpec = false;
		boolean isNested = false;
		List llist = new ArrayList();
		List tlist = null;
		boolean empty = true;
		boolean parenExist = false;
		adecl = null;
		
		
		try {      // for error handling
			{
			switch ( LA(1)) {
			case STAR:
			{
				bp=pointerGroup();
				{
				_loop368:
				do {
					switch ( LA(1)) {
					case LPAREN:
					{
						{
						match(LPAREN);
						{
						switch ( LA(1)) {
						case LITERAL_typedef:
						case LITERAL_volatile:
						case LITERAL_struct:
						case LITERAL_union:
						case LITERAL_enum:
						case LITERAL_auto:
						case LITERAL_register:
						case LITERAL___thread:
						case LITERAL_extern:
						case LITERAL_static:
						case LITERAL_inline:
						case LITERAL_const:
						case LITERAL_restrict:
						case LITERAL___nvl__:
						case LITERAL___nvl_wp__:
						case LITERAL_void:
						case LITERAL_char:
						case LITERAL_short:
						case LITERAL_int:
						case LITERAL_long:
						case LITERAL_float:
						case LITERAL_double:
						case LITERAL_signed:
						case LITERAL_unsigned:
						case LITERAL__Bool:
						case LITERAL__Complex:
						case LITERAL__Imaginary:
						case 36:
						case 37:
						case 38:
						case 39:
						case 40:
						case 41:
						case 42:
						case 43:
						case LITERAL_size_t:
						case 45:
						case 46:
						case 47:
						case 48:
						case 49:
						case LITERAL___global__:
						case LITERAL___shared__:
						case LITERAL___host__:
						case LITERAL___device__:
						case LITERAL___constant__:
						case LITERAL___noinline__:
						case LITERAL___kernel:
						case LITERAL___global:
						case LITERAL___local:
						case LITERAL___constant:
						case LITERAL_typeof:
						case LPAREN:
						case LITERAL___complex:
						case ID:
						case LITERAL___declspec:
						case LITERAL___attribute:
						case LITERAL___asm:
						case LITERAL___OSX_AVAILABLE_STARTING:
						case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
						case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
						case STAR:
						case LBRACKET:
						{
							{
							switch ( LA(1)) {
							case LPAREN:
							case STAR:
							case LBRACKET:
							{
								{
								tdecl=nonemptyAbstractDeclarator2(tname, false);
								}
								break;
							}
							case LITERAL_typedef:
							case LITERAL_volatile:
							case LITERAL_struct:
							case LITERAL_union:
							case LITERAL_enum:
							case LITERAL_auto:
							case LITERAL_register:
							case LITERAL___thread:
							case LITERAL_extern:
							case LITERAL_static:
							case LITERAL_inline:
							case LITERAL_const:
							case LITERAL_restrict:
							case LITERAL___nvl__:
							case LITERAL___nvl_wp__:
							case LITERAL_void:
							case LITERAL_char:
							case LITERAL_short:
							case LITERAL_int:
							case LITERAL_long:
							case LITERAL_float:
							case LITERAL_double:
							case LITERAL_signed:
							case LITERAL_unsigned:
							case LITERAL__Bool:
							case LITERAL__Complex:
							case LITERAL__Imaginary:
							case 36:
							case 37:
							case 38:
							case 39:
							case 40:
							case 41:
							case 42:
							case 43:
							case LITERAL_size_t:
							case 45:
							case 46:
							case 47:
							case 48:
							case 49:
							case LITERAL___global__:
							case LITERAL___shared__:
							case LITERAL___host__:
							case LITERAL___device__:
							case LITERAL___constant__:
							case LITERAL___noinline__:
							case LITERAL___kernel:
							case LITERAL___global:
							case LITERAL___local:
							case LITERAL___constant:
							case LITERAL_typeof:
							case LITERAL___complex:
							case ID:
							case LITERAL___declspec:
							case LITERAL___attribute:
							case LITERAL___asm:
							case LITERAL___OSX_AVAILABLE_STARTING:
							case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
							case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
							{
								plist=parameterTypeList();
								break;
							}
							default:
							{
								throw new NoViableAltException(LT(1), getFilename());
							}
							}
							}
							if ( inputState.guessing==0 ) {
								empty = false;
							}
							break;
						}
						case RPAREN:
						case COMMA:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						{
						switch ( LA(1)) {
						case COMMA:
						{
							match(COMMA);
							break;
						}
						case RPAREN:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						match(RPAREN);
						}
						if ( inputState.guessing==0 ) {
							
							if(empty)
							plist = new ArrayList();
							empty = true;
							parenExist = true;
							
						}
						break;
					}
					case LBRACKET:
					{
						{
						match(LBRACKET);
						{
						switch ( LA(1)) {
						case LPAREN:
						case ID:
						case Number:
						case StringLiteral:
						case LITERAL___asm:
						case STAR:
						case BAND:
						case PLUS:
						case MINUS:
						case LITERAL___builtin_nvl_get_root:
						case LITERAL___builtin_nvl_alloc_nv:
						case INC:
						case DEC:
						case LITERAL_sizeof:
						case LITERAL___alignof__:
						case LITERAL___builtin_va_arg:
						case LITERAL___builtin_offsetof:
						case BNOT:
						case LNOT:
						case LITERAL___real:
						case LITERAL___imag:
						case CharLiteral:
						{
							expr1=expr();
							break;
						}
						case RBRACKET:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						match(RBRACKET);
						}
						if ( inputState.guessing==0 ) {
							
							isArraySpec = true;
							llist.add(expr1);
							
						}
						break;
					}
					default:
					{
						break _loop368;
					}
					}
				} while (true);
				}
				break;
			}
			case LPAREN:
			case LBRACKET:
			{
				{
				int _cnt377=0;
				_loop377:
				do {
					switch ( LA(1)) {
					case LPAREN:
					{
						{
						match(LPAREN);
						{
						switch ( LA(1)) {
						case LITERAL_typedef:
						case LITERAL_volatile:
						case LITERAL_struct:
						case LITERAL_union:
						case LITERAL_enum:
						case LITERAL_auto:
						case LITERAL_register:
						case LITERAL___thread:
						case LITERAL_extern:
						case LITERAL_static:
						case LITERAL_inline:
						case LITERAL_const:
						case LITERAL_restrict:
						case LITERAL___nvl__:
						case LITERAL___nvl_wp__:
						case LITERAL_void:
						case LITERAL_char:
						case LITERAL_short:
						case LITERAL_int:
						case LITERAL_long:
						case LITERAL_float:
						case LITERAL_double:
						case LITERAL_signed:
						case LITERAL_unsigned:
						case LITERAL__Bool:
						case LITERAL__Complex:
						case LITERAL__Imaginary:
						case 36:
						case 37:
						case 38:
						case 39:
						case 40:
						case 41:
						case 42:
						case 43:
						case LITERAL_size_t:
						case 45:
						case 46:
						case 47:
						case 48:
						case 49:
						case LITERAL___global__:
						case LITERAL___shared__:
						case LITERAL___host__:
						case LITERAL___device__:
						case LITERAL___constant__:
						case LITERAL___noinline__:
						case LITERAL___kernel:
						case LITERAL___global:
						case LITERAL___local:
						case LITERAL___constant:
						case LITERAL_typeof:
						case LPAREN:
						case LITERAL___complex:
						case ID:
						case LITERAL___declspec:
						case LITERAL___attribute:
						case LITERAL___asm:
						case LITERAL___OSX_AVAILABLE_STARTING:
						case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
						case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
						case STAR:
						case LBRACKET:
						{
							{
							switch ( LA(1)) {
							case LPAREN:
							case STAR:
							case LBRACKET:
							{
								{
								tdecl=nonemptyAbstractDeclarator2(tname, false);
								}
								break;
							}
							case LITERAL_typedef:
							case LITERAL_volatile:
							case LITERAL_struct:
							case LITERAL_union:
							case LITERAL_enum:
							case LITERAL_auto:
							case LITERAL_register:
							case LITERAL___thread:
							case LITERAL_extern:
							case LITERAL_static:
							case LITERAL_inline:
							case LITERAL_const:
							case LITERAL_restrict:
							case LITERAL___nvl__:
							case LITERAL___nvl_wp__:
							case LITERAL_void:
							case LITERAL_char:
							case LITERAL_short:
							case LITERAL_int:
							case LITERAL_long:
							case LITERAL_float:
							case LITERAL_double:
							case LITERAL_signed:
							case LITERAL_unsigned:
							case LITERAL__Bool:
							case LITERAL__Complex:
							case LITERAL__Imaginary:
							case 36:
							case 37:
							case 38:
							case 39:
							case 40:
							case 41:
							case 42:
							case 43:
							case LITERAL_size_t:
							case 45:
							case 46:
							case 47:
							case 48:
							case 49:
							case LITERAL___global__:
							case LITERAL___shared__:
							case LITERAL___host__:
							case LITERAL___device__:
							case LITERAL___constant__:
							case LITERAL___noinline__:
							case LITERAL___kernel:
							case LITERAL___global:
							case LITERAL___local:
							case LITERAL___constant:
							case LITERAL_typeof:
							case LITERAL___complex:
							case ID:
							case LITERAL___declspec:
							case LITERAL___attribute:
							case LITERAL___asm:
							case LITERAL___OSX_AVAILABLE_STARTING:
							case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
							case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
							{
								plist=parameterTypeList();
								break;
							}
							default:
							{
								throw new NoViableAltException(LT(1), getFilename());
							}
							}
							}
							if ( inputState.guessing==0 ) {
								empty = false;
							}
							break;
						}
						case RPAREN:
						case COMMA:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						{
						switch ( LA(1)) {
						case COMMA:
						{
							match(COMMA);
							break;
						}
						case RPAREN:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						match(RPAREN);
						}
						if ( inputState.guessing==0 ) {
							
							if (empty)
							plist = new ArrayList();
							empty = true;
							parenExist = true;
							
						}
						break;
					}
					case LBRACKET:
					{
						{
						match(LBRACKET);
						{
						switch ( LA(1)) {
						case LPAREN:
						case ID:
						case Number:
						case StringLiteral:
						case LITERAL___asm:
						case STAR:
						case BAND:
						case PLUS:
						case MINUS:
						case LITERAL___builtin_nvl_get_root:
						case LITERAL___builtin_nvl_alloc_nv:
						case INC:
						case DEC:
						case LITERAL_sizeof:
						case LITERAL___alignof__:
						case LITERAL___builtin_va_arg:
						case LITERAL___builtin_offsetof:
						case BNOT:
						case LNOT:
						case LITERAL___real:
						case LITERAL___imag:
						case CharLiteral:
						{
							expr1=expr();
							break;
						}
						case RBRACKET:
						{
							break;
						}
						default:
						{
							throw new NoViableAltException(LT(1), getFilename());
						}
						}
						}
						match(RBRACKET);
						}
						if ( inputState.guessing==0 ) {
							
							isArraySpec = true;
							llist.add(expr1);
							
						}
						break;
					}
					default:
					{
						if ( _cnt377>=1 ) { break _loop377; } else {throw new NoViableAltException(LT(1), getFilename());}
					}
					}
					_cnt377++;
				} while (true);
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			if ( inputState.guessing==0 ) {
				
				if (isArraySpec) {
				/* []+ */
				aspec = new ArraySpecifier(llist);
				tlist = new ArrayList();
				tlist.add(aspec);
				}
				NameID idex = null;
				// nested declarator (tlist == null ?)
				if (bp == null)
				bp = new ArrayList();
				// assume tlist == null
				if (tdecl != null) {
				//assert tlist == null : "Assertion (tlist == null) failed 2";
				if (tlist == null)
				tlist = new ArrayList();
				adecl = new NestedDeclarator(bp,tdecl,plist,tlist);
				} else if( !isRoot || (plist != null) || (tlist != null) ) {
					//Create empty declarator only for expressions inside a parenthesis.
				idex = new NameID("");
				if (plist != null) // assume tlist == null
				adecl = new ProcedureDeclarator(bp,idex,plist);
				else {
				if (tlist != null)
				adecl = new VariableDeclarator(bp,idex,tlist);
				else
				adecl = new VariableDeclarator(bp,idex);
				}
				}
				if( isRoot ) {
					if (adecl != null) {
						tname.add(adecl);
					} else if( !bp.isEmpty() ) {
						tname.addAll(bp);
					}
				}
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_35);
			} else {
			  throw ex;
			}
		}
		return adecl;
	}
	
	public final Expression  postfixExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		ret_expr=null;
		Expression expr1=null;
		Expression expr2=null;
		List tname = null;
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case LPAREN:
			case ID:
			case Number:
			case StringLiteral:
			case CharLiteral:
			{
				expr1=primaryExpr();
				if ( inputState.guessing==0 ) {
					ret_expr = expr1;
				}
				{
				switch ( LA(1)) {
				case LPAREN:
				case LBRACKET:
				case DOT:
				case PTR:
				case INC:
				case DEC:
				{
					ret_expr=postfixSuffix(expr1);
					break;
				}
				case SEMI:
				case VARARGS:
				case RCURLY:
				case RPAREN:
				case COMMA:
				case COLON:
				case ASSIGN:
				case LITERAL___attribute:
				case LITERAL___asm:
				case LITERAL___OSX_AVAILABLE_STARTING:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
				case STAR:
				case RBRACKET:
				case DIV_ASSIGN:
				case PLUS_ASSIGN:
				case MINUS_ASSIGN:
				case STAR_ASSIGN:
				case MOD_ASSIGN:
				case RSHIFT_ASSIGN:
				case LSHIFT_ASSIGN:
				case BAND_ASSIGN:
				case BOR_ASSIGN:
				case BXOR_ASSIGN:
				case LOR:
				case LAND:
				case BOR:
				case BXOR:
				case BAND:
				case EQUAL:
				case NOT_EQUAL:
				case LT:
				case LTE:
				case GT:
				case GTE:
				case LSHIFT:
				case RSHIFT:
				case PLUS:
				case MINUS:
				case DIV:
				case MOD:
				case QUESTION:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				break;
			}
			case LITERAL___builtin_nvl_get_root:
			{
				match(LITERAL___builtin_nvl_get_root);
				{
				{
				match(LPAREN);
				{
				expr1=assignExpr();
				}
				match(COMMA);
				{
				tname=typeName2();
				}
				match(RPAREN);
				}
				if ( inputState.guessing==0 ) {
					ret_expr = new NVLGetRootExpression(expr1, tname);
				}
				}
				{
				switch ( LA(1)) {
				case LPAREN:
				case LBRACKET:
				case DOT:
				case PTR:
				case INC:
				case DEC:
				{
					ret_expr=postfixSuffix(ret_expr);
					break;
				}
				case SEMI:
				case VARARGS:
				case RCURLY:
				case RPAREN:
				case COMMA:
				case COLON:
				case ASSIGN:
				case LITERAL___attribute:
				case LITERAL___asm:
				case LITERAL___OSX_AVAILABLE_STARTING:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
				case STAR:
				case RBRACKET:
				case DIV_ASSIGN:
				case PLUS_ASSIGN:
				case MINUS_ASSIGN:
				case STAR_ASSIGN:
				case MOD_ASSIGN:
				case RSHIFT_ASSIGN:
				case LSHIFT_ASSIGN:
				case BAND_ASSIGN:
				case BOR_ASSIGN:
				case BXOR_ASSIGN:
				case LOR:
				case LAND:
				case BOR:
				case BXOR:
				case BAND:
				case EQUAL:
				case NOT_EQUAL:
				case LT:
				case LTE:
				case GT:
				case GTE:
				case LSHIFT:
				case RSHIFT:
				case PLUS:
				case MINUS:
				case DIV:
				case MOD:
				case QUESTION:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				break;
			}
			case LITERAL___builtin_nvl_alloc_nv:
			{
				match(LITERAL___builtin_nvl_alloc_nv);
				{
				{
				match(LPAREN);
				{
				expr1=assignExpr();
				}
				match(COMMA);
				{
				expr2=assignExpr();
				}
				match(COMMA);
				{
				tname=typeName2();
				}
				match(RPAREN);
				}
				if ( inputState.guessing==0 ) {
					ret_expr = new NVLAllocNVExpression(expr1, expr2, tname);
				}
				}
				{
				switch ( LA(1)) {
				case LPAREN:
				case LBRACKET:
				case DOT:
				case PTR:
				case INC:
				case DEC:
				{
					ret_expr=postfixSuffix(ret_expr);
					break;
				}
				case SEMI:
				case VARARGS:
				case RCURLY:
				case RPAREN:
				case COMMA:
				case COLON:
				case ASSIGN:
				case LITERAL___attribute:
				case LITERAL___asm:
				case LITERAL___OSX_AVAILABLE_STARTING:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED:
				case LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG:
				case STAR:
				case RBRACKET:
				case DIV_ASSIGN:
				case PLUS_ASSIGN:
				case MINUS_ASSIGN:
				case STAR_ASSIGN:
				case MOD_ASSIGN:
				case RSHIFT_ASSIGN:
				case LSHIFT_ASSIGN:
				case BAND_ASSIGN:
				case BOR_ASSIGN:
				case BXOR_ASSIGN:
				case LOR:
				case LAND:
				case BOR:
				case BXOR:
				case BAND:
				case EQUAL:
				case NOT_EQUAL:
				case LT:
				case LTE:
				case GT:
				case GTE:
				case LSHIFT:
				case RSHIFT:
				case PLUS:
				case MINUS:
				case DIV:
				case MOD:
				case QUESTION:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_67);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final Expression  primaryExpr() throws RecognitionException, TokenStreamException {
		Expression p;
		
		Token  prim_id = null;
		Token  prim_num = null;
		
		Expression expr1=null;
		CompoundStatement cstmt = null;
		p=null;
		String name = null;
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case ID:
			{
				prim_id = LT(1);
				match(ID);
				if ( inputState.guessing==0 ) {
					
					name = prim_id.getText();
					p=SymbolTools.getOrphanID(name);
					
				}
				break;
			}
			case Number:
			{
				prim_num = LT(1);
				match(Number);
				if ( inputState.guessing==0 ) {
					
					// [Modified by Joel E. Denny to pass base and suffix to IntegerLiteral.]
					name = prim_num.getText();
					boolean handled = false;
					name = name.toUpperCase();
					String suffix = name.replaceAll("[X0-9A-E\\-+.]","");
					name = name.replaceAll("L","");
					name = name.replaceAll("U","");
					IntegerLiteral.Base integerBase = IntegerLiteral.Base.DECIMAL;
					if (name.startsWith("0X")) {
					integerBase = IntegerLiteral.Base.HEX;
					suffix = suffix.replaceAll("F", "");
					}
					else {
					if (name.startsWith("0") && !name.equals("0"))
					integerBase = IntegerLiteral.Base.OCTAL;
					name = name.replaceAll("F","");
					name = name.replaceAll("I","");
					// 1.0IF can be generated from _Complex_I
					}
					try {
					Integer i2 = Integer.decode(name);
					p=new IntegerLiteral(integerBase, i2.intValue(), suffix);
					handled = true;
					} catch(NumberFormatException e) {
					;
					}
					if (handled == false) {
					try {
					Long in = Long.decode(name);
					p=new IntegerLiteral(integerBase, in.longValue(), suffix);
					handled = true;
					} catch(NumberFormatException e) {
					;
					}
					}
					if (handled == false) {
					try {
					double d = Double.parseDouble(name);
					if (suffix.matches("F|L|IF"))
					p = new FloatLiteral(d, suffix);
					else
					p = new FloatLiteral(d);
					handled = true;
					} catch(NumberFormatException e) {
					p=new NameID(name);
					PrintTools.printlnStatus("Strange number "+name,0);
					}
					}
					
				}
				break;
			}
			case CharLiteral:
			{
				name=charConst();
				if ( inputState.guessing==0 ) {
					
					// [Modified by Joel E. Denny to handle wide characters.]
					if(name.charAt(0) == 'L' && name.charAt(2) != '\\'
					|| name.charAt(0) == '\'' && name.charAt(1) != '\\')
					p = new CharLiteral(name);
					// escape sequence is not handled at this point
					else {
					p = new EscapeLiteral(name);
					}
					
				}
				break;
			}
			case StringLiteral:
			{
				name=stringConst();
				if ( inputState.guessing==0 ) {
					
					p=new StringLiteral(name);
					((StringLiteral)p).stripQuotes();
					
				}
				break;
			}
			default:
				boolean synPredMatched413 = false;
				if (((LA(1)==LPAREN) && (LA(2)==LCURLY))) {
					int _m413 = mark();
					synPredMatched413 = true;
					inputState.guessing++;
					try {
						{
						match(LPAREN);
						match(LCURLY);
						}
					}
					catch (RecognitionException pe) {
						synPredMatched413 = false;
					}
					rewind(_m413);
inputState.guessing--;
				}
				if ( synPredMatched413 ) {
					match(LPAREN);
					cstmt=compoundStatement();
					match(RPAREN);
					if ( inputState.guessing==0 ) {
						
						PrintTools.printlnStatus("[DEBUG] Warning: CompoundStatement Expression !",1);
						p = new StatementExpression(cstmt);
						
					}
				}
				else if ((LA(1)==LPAREN) && (_tokenSet_19.member(LA(2)))) {
					match(LPAREN);
					expr1=expr();
					match(RPAREN);
					if ( inputState.guessing==0 ) {
						
						p=expr1;
						
					}
				}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_54);
			} else {
			  throw ex;
			}
		}
		return p;
	}
	
	public final Expression  postfixSuffix(
		Expression expr1
	) throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		Token  ptr_id = null;
		Token  dot_id = null;
		
		Expression expr2=null;
		SymbolTable saveSymtab = null;
		String s;
		ret_expr = expr1;
		List args = null;
		
		
		try {      // for error handling
			{
			int _cnt327=0;
			_loop327:
			do {
				switch ( LA(1)) {
				case PTR:
				{
					match(PTR);
					ptr_id = LT(1);
					match(ID);
					if ( inputState.guessing==0 ) {
						
						ret_expr = new AccessExpression(
						ret_expr, AccessOperator.POINTER_ACCESS, SymbolTools.getOrphanID(ptr_id.getText()));
						
					}
					break;
				}
				case DOT:
				{
					match(DOT);
					dot_id = LT(1);
					match(ID);
					if ( inputState.guessing==0 ) {
						
						ret_expr = new AccessExpression(
						ret_expr, AccessOperator.MEMBER_ACCESS, SymbolTools.getOrphanID(dot_id.getText()));
						
					}
					break;
				}
				case LPAREN:
				{
					args=functionCall();
					if ( inputState.guessing==0 ) {
						
						if (args == null)
						ret_expr = new FunctionCall(ret_expr);
						else
						ret_expr = new FunctionCall(ret_expr,args);
						
					}
					break;
				}
				case LBRACKET:
				{
					match(LBRACKET);
					expr2=expr();
					match(RBRACKET);
					if ( inputState.guessing==0 ) {
						
						if (ret_expr instanceof ArrayAccess) {
						ArrayAccess aacc = (ArrayAccess)ret_expr;
						int dim = aacc.getNumIndices();
						int n = 0;
						List alist = new ArrayList();
						for (n = 0;n < dim; n++) {
						alist.add(aacc.getIndex(n).clone());
						}
						alist.add(expr2);
						aacc.setIndices(alist);
						} else
						ret_expr = new ArrayAccess(ret_expr,expr2);
						
					}
					break;
				}
				case INC:
				{
					match(INC);
					if ( inputState.guessing==0 ) {
						ret_expr = new UnaryExpression(UnaryOperator.POST_INCREMENT,ret_expr);
					}
					break;
				}
				case DEC:
				{
					match(DEC);
					if ( inputState.guessing==0 ) {
						ret_expr = new UnaryExpression(UnaryOperator.POST_DECREMENT,ret_expr);
					}
					break;
				}
				default:
				{
					if ( _cnt327>=1 ) { break _loop327; } else {throw new NoViableAltException(LT(1), getFilename());}
				}
				}
				_cnt327++;
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_67);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final List  functionCall() throws RecognitionException, TokenStreamException {
		List args;
		
		args=null;
		
		try {      // for error handling
			match(LPAREN);
			{
			switch ( LA(1)) {
			case LPAREN:
			case ID:
			case Number:
			case StringLiteral:
			case LITERAL___asm:
			case STAR:
			case BAND:
			case PLUS:
			case MINUS:
			case LITERAL___builtin_nvl_get_root:
			case LITERAL___builtin_nvl_alloc_nv:
			case INC:
			case DEC:
			case LITERAL_sizeof:
			case LITERAL___alignof__:
			case LITERAL___builtin_va_arg:
			case LITERAL___builtin_offsetof:
			case BNOT:
			case LNOT:
			case LITERAL___real:
			case LITERAL___imag:
			case CharLiteral:
			{
				args=argExprList();
				break;
			}
			case RPAREN:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			match(RPAREN);
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_54);
			} else {
			  throw ex;
			}
		}
		return args;
	}
	
	public final List  argExprList() throws RecognitionException, TokenStreamException {
		List eList;
		
		
		Expression expr1 = null;
		eList=new ArrayList();
		Declaration pdecl = null;
		
		
		try {      // for error handling
			expr1=assignExpr();
			if ( inputState.guessing==0 ) {
				eList.add(expr1);
			}
			{
			_loop417:
			do {
				if ((LA(1)==COMMA)) {
					match(COMMA);
					{
					if ((_tokenSet_19.member(LA(1))) && (_tokenSet_32.member(LA(2)))) {
						expr1=assignExpr();
						if ( inputState.guessing==0 ) {
							eList.add(expr1);
						}
					}
					else if ((_tokenSet_2.member(LA(1))) && (_tokenSet_103.member(LA(2)))) {
						pdecl=parameterDeclaration();
						if ( inputState.guessing==0 ) {
							eList.add(pdecl);
						}
					}
					else {
						throw new NoViableAltException(LT(1), getFilename());
					}
					
					}
				}
				else {
					break _loop417;
				}
				
			} while (true);
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_53);
			} else {
			  throw ex;
			}
		}
		return eList;
	}
	
	public final Expression  unaryExpr() throws RecognitionException, TokenStreamException {
		Expression ret_expr;
		
		
		Expression expr1=null;
		UnaryOperator code;
		ret_expr = null;
		List tname = null;
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case LPAREN:
			case ID:
			case Number:
			case StringLiteral:
			case LITERAL___builtin_nvl_get_root:
			case LITERAL___builtin_nvl_alloc_nv:
			case CharLiteral:
			{
				ret_expr=postfixExpr();
				break;
			}
			case INC:
			{
				match(INC);
				expr1=castExpr();
				if ( inputState.guessing==0 ) {
					ret_expr = new UnaryExpression(UnaryOperator.PRE_INCREMENT, expr1);
				}
				break;
			}
			case DEC:
			{
				match(DEC);
				expr1=castExpr();
				if ( inputState.guessing==0 ) {
					ret_expr = new UnaryExpression(UnaryOperator.PRE_DECREMENT, expr1);
				}
				break;
			}
			case STAR:
			case BAND:
			case PLUS:
			case MINUS:
			case BNOT:
			case LNOT:
			case LITERAL___real:
			case LITERAL___imag:
			{
				code=unaryOperator();
				expr1=castExpr();
				if ( inputState.guessing==0 ) {
					ret_expr = new UnaryExpression(code, expr1);
				}
				break;
			}
			case LITERAL_sizeof:
			{
				match(LITERAL_sizeof);
				{
				boolean synPredMatched381 = false;
				if (((LA(1)==LPAREN) && (_tokenSet_30.member(LA(2))))) {
					int _m381 = mark();
					synPredMatched381 = true;
					inputState.guessing++;
					try {
						{
						match(LPAREN);
						typeName2();
						}
					}
					catch (RecognitionException pe) {
						synPredMatched381 = false;
					}
					rewind(_m381);
inputState.guessing--;
				}
				if ( synPredMatched381 ) {
					match(LPAREN);
					tname=typeName2();
					match(RPAREN);
					if ( inputState.guessing==0 ) {
						ret_expr = new SizeofExpression(tname);
					}
				}
				else if ((_tokenSet_19.member(LA(1))) && (_tokenSet_102.member(LA(2)))) {
					expr1=unaryExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new SizeofExpression(expr1);
					}
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				break;
			}
			case LITERAL___alignof__:
			{
				match(LITERAL___alignof__);
				{
				boolean synPredMatched384 = false;
				if (((LA(1)==LPAREN) && (_tokenSet_30.member(LA(2))))) {
					int _m384 = mark();
					synPredMatched384 = true;
					inputState.guessing++;
					try {
						{
						match(LPAREN);
						typeName2();
						}
					}
					catch (RecognitionException pe) {
						synPredMatched384 = false;
					}
					rewind(_m384);
inputState.guessing--;
				}
				if ( synPredMatched384 ) {
					match(LPAREN);
					tname=typeName2();
					match(RPAREN);
					if ( inputState.guessing==0 ) {
						ret_expr = new AlignofExpression(tname);
					}
				}
				else if ((_tokenSet_19.member(LA(1))) && (_tokenSet_102.member(LA(2)))) {
					expr1=unaryExpr();
					if ( inputState.guessing==0 ) {
						ret_expr = new AlignofExpression(expr1);
					}
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
				break;
			}
			case LITERAL___builtin_va_arg:
			{
				match(LITERAL___builtin_va_arg);
				{
				{
				match(LPAREN);
				{
				expr1=unaryExpr();
				}
				match(COMMA);
				{
				tname=typeName2();
				}
				match(RPAREN);
				}
				if ( inputState.guessing==0 ) {
					ret_expr = new VaArgExpression(expr1, tname);
				}
				}
				break;
			}
			case LITERAL___builtin_offsetof:
			{
				match(LITERAL___builtin_offsetof);
				{
				{
				match(LPAREN);
				{
				tname=typeName2();
				}
				match(COMMA);
				{
				expr1=unaryExpr();
				}
				match(RPAREN);
				}
				if ( inputState.guessing==0 ) {
					ret_expr = new OffsetofExpression(tname, expr1);
				}
				}
				break;
			}
			case LITERAL___asm:
			{
				{
				ret_expr=gnuAsmExpr();
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_67);
			} else {
			  throw ex;
			}
		}
		return ret_expr;
	}
	
	public final UnaryOperator  unaryOperator() throws RecognitionException, TokenStreamException {
		UnaryOperator code;
		
		code = null;
		
		try {      // for error handling
			switch ( LA(1)) {
			case BAND:
			{
				match(BAND);
				if ( inputState.guessing==0 ) {
					code = UnaryOperator.ADDRESS_OF;
				}
				break;
			}
			case STAR:
			{
				match(STAR);
				if ( inputState.guessing==0 ) {
					code = UnaryOperator.DEREFERENCE;
				}
				break;
			}
			case PLUS:
			{
				match(PLUS);
				if ( inputState.guessing==0 ) {
					code = UnaryOperator.PLUS;
				}
				break;
			}
			case MINUS:
			{
				match(MINUS);
				if ( inputState.guessing==0 ) {
					code = UnaryOperator.MINUS;
				}
				break;
			}
			case BNOT:
			{
				match(BNOT);
				if ( inputState.guessing==0 ) {
					code = UnaryOperator.BITWISE_COMPLEMENT;
				}
				break;
			}
			case LNOT:
			{
				match(LNOT);
				if ( inputState.guessing==0 ) {
					code = UnaryOperator.LOGICAL_NEGATION;
				}
				break;
			}
			case LITERAL___real:
			{
				match(LITERAL___real);
				if ( inputState.guessing==0 ) {
					code = null;
				}
				break;
			}
			case LITERAL___imag:
			{
				match(LITERAL___imag);
				if ( inputState.guessing==0 ) {
					code = null;
				}
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_19);
			} else {
			  throw ex;
			}
		}
		return code;
	}
	
	public final Expression  gnuAsmExpr() throws RecognitionException, TokenStreamException {
		Expression ret;
		
		
		ret = null;
		String str = "";
		List<Traversable> expr_list = new ArrayList<Traversable>();
		int count = 0;
		
		
		try {      // for error handling
			if ( inputState.guessing==0 ) {
				count = mark();
			}
			match(LITERAL___asm);
			{
			switch ( LA(1)) {
			case LITERAL_volatile:
			{
				match(LITERAL_volatile);
				break;
			}
			case LPAREN:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			match(LPAREN);
			stringConst();
			{
			if ((LA(1)==COLON) && (_tokenSet_104.member(LA(2)))) {
				match(COLON);
				{
				switch ( LA(1)) {
				case StringLiteral:
				{
					strOptExprPair(expr_list);
					{
					_loop400:
					do {
						if ((LA(1)==COMMA)) {
							match(COMMA);
							strOptExprPair(expr_list);
						}
						else {
							break _loop400;
						}
						
					} while (true);
					}
					break;
				}
				case RPAREN:
				case COLON:
				{
					break;
				}
				default:
				{
					throw new NoViableAltException(LT(1), getFilename());
				}
				}
				}
				{
				if ((LA(1)==COLON) && (_tokenSet_104.member(LA(2)))) {
					match(COLON);
					{
					switch ( LA(1)) {
					case StringLiteral:
					{
						strOptExprPair(expr_list);
						{
						_loop404:
						do {
							if ((LA(1)==COMMA)) {
								match(COMMA);
								strOptExprPair(expr_list);
							}
							else {
								break _loop404;
							}
							
						} while (true);
						}
						break;
					}
					case RPAREN:
					case COLON:
					{
						break;
					}
					default:
					{
						throw new NoViableAltException(LT(1), getFilename());
					}
					}
					}
				}
				else if ((LA(1)==RPAREN||LA(1)==COLON) && (_tokenSet_105.member(LA(2)))) {
				}
				else {
					throw new NoViableAltException(LT(1), getFilename());
				}
				
				}
			}
			else if ((LA(1)==RPAREN||LA(1)==COLON) && (_tokenSet_105.member(LA(2)))) {
			}
			else {
				throw new NoViableAltException(LT(1), getFilename());
			}
			
			}
			{
			switch ( LA(1)) {
			case COLON:
			{
				match(COLON);
				stringConst();
				{
				_loop407:
				do {
					if ((LA(1)==COMMA)) {
						match(COMMA);
						stringConst();
					}
					else {
						break _loop407;
					}
					
				} while (true);
				}
				break;
			}
			case RPAREN:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
			match(RPAREN);
			if ( inputState.guessing==0 ) {
				
				// Recover the original stream and stores it in "SomeExpression" augmented with
				// list of evaluated expressions.
				for (int i=count-mark()+1; i <= 0; i++)
				str += " " + LT(i).getText();
				ret = new SomeExpression(str, expr_list);
				
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_67);
			} else {
			  throw ex;
			}
		}
		return ret;
	}
	
	public final void strOptExprPair(
		List<Traversable> expr_list
	) throws RecognitionException, TokenStreamException {
		
		Expression e = null;
		
		try {      // for error handling
			stringConst();
			{
			switch ( LA(1)) {
			case LPAREN:
			{
				match(LPAREN);
				{
				e=expr();
				}
				match(RPAREN);
				if ( inputState.guessing==0 ) {
					expr_list.add(e);
				}
				break;
			}
			case RPAREN:
			case COMMA:
			case COLON:
			{
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_106);
			} else {
			  throw ex;
			}
		}
	}
	
	protected final String  charConst() throws RecognitionException, TokenStreamException {
		String name;
		
		Token  cl = null;
		name = null;
		
		try {      // for error handling
			cl = LT(1);
			match(CharLiteral);
			if ( inputState.guessing==0 ) {
				name=cl.getText();
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_54);
			} else {
			  throw ex;
			}
		}
		return name;
	}
	
	protected final void intConst() throws RecognitionException, TokenStreamException {
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case IntOctalConst:
			{
				match(IntOctalConst);
				break;
			}
			case LongOctalConst:
			{
				match(LongOctalConst);
				break;
			}
			case UnsignedOctalConst:
			{
				match(UnsignedOctalConst);
				break;
			}
			case IntIntConst:
			{
				match(IntIntConst);
				break;
			}
			case LongIntConst:
			{
				match(LongIntConst);
				break;
			}
			case UnsignedIntConst:
			{
				match(UnsignedIntConst);
				break;
			}
			case IntHexConst:
			{
				match(IntHexConst);
				break;
			}
			case LongHexConst:
			{
				match(LongHexConst);
				break;
			}
			case UnsignedHexConst:
			{
				match(UnsignedHexConst);
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
	}
	
	protected final void floatConst() throws RecognitionException, TokenStreamException {
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case FloatDoubleConst:
			{
				match(FloatDoubleConst);
				break;
			}
			case DoubleDoubleConst:
			{
				match(DoubleDoubleConst);
				break;
			}
			case LongDoubleConst:
			{
				match(LongDoubleConst);
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
	}
	
	public final void dummy() throws RecognitionException, TokenStreamException {
		
		
		try {      // for error handling
			switch ( LA(1)) {
			case NTypedefName:
			{
				match(NTypedefName);
				break;
			}
			case NInitDecl:
			{
				match(NInitDecl);
				break;
			}
			case NDeclarator:
			{
				match(NDeclarator);
				break;
			}
			case NStructDeclarator:
			{
				match(NStructDeclarator);
				break;
			}
			case NDeclaration:
			{
				match(NDeclaration);
				break;
			}
			case NCast:
			{
				match(NCast);
				break;
			}
			case NPointerGroup:
			{
				match(NPointerGroup);
				break;
			}
			case NExpressionGroup:
			{
				match(NExpressionGroup);
				break;
			}
			case NFunctionCallArgs:
			{
				match(NFunctionCallArgs);
				break;
			}
			case NNonemptyAbstractDeclarator:
			{
				match(NNonemptyAbstractDeclarator);
				break;
			}
			case NInitializer:
			{
				match(NInitializer);
				break;
			}
			case NStatementExpr:
			{
				match(NStatementExpr);
				break;
			}
			case NEmptyExpression:
			{
				match(NEmptyExpression);
				break;
			}
			case NParameterTypeList:
			{
				match(NParameterTypeList);
				break;
			}
			case NFunctionDef:
			{
				match(NFunctionDef);
				break;
			}
			case NCompoundStatement:
			{
				match(NCompoundStatement);
				break;
			}
			case NParameterDeclaration:
			{
				match(NParameterDeclaration);
				break;
			}
			case NCommaExpr:
			{
				match(NCommaExpr);
				break;
			}
			case NUnaryExpr:
			{
				match(NUnaryExpr);
				break;
			}
			case NLabel:
			{
				match(NLabel);
				break;
			}
			case NPostfixExpr:
			{
				match(NPostfixExpr);
				break;
			}
			case NRangeExpr:
			{
				match(NRangeExpr);
				break;
			}
			case NStringSeq:
			{
				match(NStringSeq);
				break;
			}
			case NInitializerElementLabel:
			{
				match(NInitializerElementLabel);
				break;
			}
			case NLcurlyInitializer:
			{
				match(NLcurlyInitializer);
				break;
			}
			case NAsmAttribute:
			{
				match(NAsmAttribute);
				break;
			}
			case NGnuAsmExpr:
			{
				match(NGnuAsmExpr);
				break;
			}
			case NTypeMissing:
			{
				match(NTypeMissing);
				break;
			}
			default:
			{
				throw new NoViableAltException(LT(1), getFilename());
			}
			}
		}
		catch (RecognitionException ex) {
			if (inputState.guessing==0) {
				reportError(ex);
				recover(ex,_tokenSet_0);
			} else {
			  throw ex;
			}
		}
	}
	
	
	public static final String[] _tokenNames = {
		"<0>",
		"EOF",
		"<2>",
		"NULL_TREE_LOOKAHEAD",
		"\"typedef\"",
		"SEMI",
		"VARARGS",
		"LCURLY",
		"\"asm\"",
		"\"volatile\"",
		"RCURLY",
		"\"struct\"",
		"\"union\"",
		"\"enum\"",
		"\"auto\"",
		"\"register\"",
		"\"__thread\"",
		"\"extern\"",
		"\"static\"",
		"\"inline\"",
		"\"const\"",
		"\"restrict\"",
		"\"__nvl__\"",
		"\"__nvl_wp__\"",
		"\"void\"",
		"\"char\"",
		"\"short\"",
		"\"int\"",
		"\"long\"",
		"\"float\"",
		"\"double\"",
		"\"signed\"",
		"\"unsigned\"",
		"\"_Bool\"",
		"\"_Complex\"",
		"\"_Imaginary\"",
		"\"int8_t\"",
		"\"uint8_t\"",
		"\"int16_t\"",
		"\"uint16_t\"",
		"\"int32_t\"",
		"\"uint32_t\"",
		"\"int64_t\"",
		"\"uint64_t\"",
		"\"size_t\"",
		"\"_Float128\"",
		"\"__float128\"",
		"\"__float80\"",
		"\"__ibm128\"",
		"\"_Float16\"",
		"\"_Nullable\"",
		"\"__global__\"",
		"\"__shared__\"",
		"\"__host__\"",
		"\"__device__\"",
		"\"__constant__\"",
		"\"__noinline__\"",
		"\"__kernel\"",
		"\"__global\"",
		"\"__local\"",
		"\"__constant\"",
		"\"typeof\"",
		"LPAREN",
		"RPAREN",
		"\"__complex\"",
		"ID",
		"COMMA",
		"COLON",
		"ASSIGN",
		"\"__declspec\"",
		"Number",
		"StringLiteral",
		"\"__attribute\"",
		"\"__asm\"",
		"\"__OSX_AVAILABLE_STARTING\"",
		"\"__OSX_AVAILABLE_BUT_DEPRECATED\"",
		"\"__OSX_AVAILABLE_BUT_DEPRECATED_MSG\"",
		"STAR",
		"LBRACKET",
		"RBRACKET",
		"DOT",
		"\"__label__\"",
		"\"while\"",
		"\"do\"",
		"\"for\"",
		"\"goto\"",
		"\"continue\"",
		"\"break\"",
		"\"return\"",
		"\"case\"",
		"\"default\"",
		"\"if\"",
		"\"else\"",
		"\"switch\"",
		"DIV_ASSIGN",
		"PLUS_ASSIGN",
		"MINUS_ASSIGN",
		"STAR_ASSIGN",
		"MOD_ASSIGN",
		"RSHIFT_ASSIGN",
		"LSHIFT_ASSIGN",
		"BAND_ASSIGN",
		"BOR_ASSIGN",
		"BXOR_ASSIGN",
		"LOR",
		"LAND",
		"BOR",
		"BXOR",
		"BAND",
		"EQUAL",
		"NOT_EQUAL",
		"LT",
		"LTE",
		"GT",
		"GTE",
		"LSHIFT",
		"RSHIFT",
		"PLUS",
		"MINUS",
		"DIV",
		"MOD",
		"\"__builtin_nvl_get_root\"",
		"\"__builtin_nvl_alloc_nv\"",
		"PTR",
		"INC",
		"DEC",
		"QUESTION",
		"\"sizeof\"",
		"\"__alignof__\"",
		"\"__builtin_va_arg\"",
		"\"__builtin_offsetof\"",
		"BNOT",
		"LNOT",
		"\"__real\"",
		"\"__imag\"",
		"CharLiteral",
		"IntOctalConst",
		"LongOctalConst",
		"UnsignedOctalConst",
		"IntIntConst",
		"LongIntConst",
		"UnsignedIntConst",
		"IntHexConst",
		"LongHexConst",
		"UnsignedHexConst",
		"FloatDoubleConst",
		"DoubleDoubleConst",
		"LongDoubleConst",
		"NTypedefName",
		"NInitDecl",
		"NDeclarator",
		"NStructDeclarator",
		"NDeclaration",
		"NCast",
		"NPointerGroup",
		"NExpressionGroup",
		"NFunctionCallArgs",
		"NNonemptyAbstractDeclarator",
		"NInitializer",
		"NStatementExpr",
		"NEmptyExpression",
		"NParameterTypeList",
		"NFunctionDef",
		"NCompoundStatement",
		"NParameterDeclaration",
		"NCommaExpr",
		"NUnaryExpr",
		"NLabel",
		"NPostfixExpr",
		"NRangeExpr",
		"NStringSeq",
		"NInitializerElementLabel",
		"NLcurlyInitializer",
		"NAsmAttribute",
		"NGnuAsmExpr",
		"NTypeMissing",
		"\"__extension__\"",
		"Vocabulary",
		"Whitespace",
		"Comment",
		"CPPComment",
		"a line directive",
		"Space",
		"LineDirective",
		"BadStringLiteral",
		"Escape",
		"IntSuffix",
		"NumberSuffix",
		"Digit",
		"HexDigit",
		"HexFloatTail",
		"Exponent",
		"IDMEAT",
		"WideCharLiteral",
		"WideStringLiteral"
	};
	
	private static final long[] mk_tokenSet_0() {
		long[] data = { 2L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_0 = new BitSet(mk_tokenSet_0());
	private static final long[] mk_tokenSet_1() {
		long[] data = { 9222246136947931952L, 16163L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_1 = new BitSet(mk_tokenSet_1());
	private static final long[] mk_tokenSet_2() {
		long[] data = { 4610560118520543760L, 7971L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_2 = new BitSet(mk_tokenSet_2());
	private static final long[] mk_tokenSet_3() {
		long[] data = { 9223372036854774448L, 16163L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_3 = new BitSet(mk_tokenSet_3());
	private static final long[] mk_tokenSet_4() {
		long[] data = { 9222246136947882496L, 16163L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_4 = new BitSet(mk_tokenSet_4());
	private static final long[] mk_tokenSet_5() {
		long[] data = { 9223372036854774512L, 32547L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_5 = new BitSet(mk_tokenSet_5());
	private static final long[] mk_tokenSet_6() {
		long[] data = { 4611686018427387904L, 16130L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_6 = new BitSet(mk_tokenSet_6());
	private static final long[] mk_tokenSet_7() {
		long[] data = { 4611686018443117088L, 32542L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_7 = new BitSet(mk_tokenSet_7());
	private static final long[] mk_tokenSet_8() {
		long[] data = { 9222246136947931954L, 16163L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_8 = new BitSet(mk_tokenSet_8());
	private static final long[] mk_tokenSet_9() {
		long[] data = { 9222246136947933170L, -5305222768051208221L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_9 = new BitSet(mk_tokenSet_9());
	private static final long[] mk_tokenSet_10() {
		long[] data = { 4610560118520494592L, 7971L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_10 = new BitSet(mk_tokenSet_10());
	private static final long[] mk_tokenSet_11() {
		long[] data = { 9223372036854725248L, 16163L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_11 = new BitSet(mk_tokenSet_11());
	private static final long[] mk_tokenSet_12() {
		long[] data = { 9222246136947931888L, 32547L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_12 = new BitSet(mk_tokenSet_12());
	private static final long[] mk_tokenSet_13() {
		long[] data = { 15729152L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_13 = new BitSet(mk_tokenSet_13());
	private static final long[] mk_tokenSet_14() {
		long[] data = { 4610560118516430848L, 3L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_14 = new BitSet(mk_tokenSet_14());
	private static final long[] mk_tokenSet_15() {
		long[] data = { -1125899906843920L, 24383L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_15 = new BitSet(mk_tokenSet_15());
	private static final long[] mk_tokenSet_16() {
		long[] data = { -14L, -268435457L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_16 = new BitSet(mk_tokenSet_16());
	private static final long[] mk_tokenSet_17() {
		long[] data = { -4612811918334231824L, 7999L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_17 = new BitSet(mk_tokenSet_17());
	private static final long[] mk_tokenSet_18() {
		long[] data = { 32L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_18 = new BitSet(mk_tokenSet_18());
	private static final long[] mk_tokenSet_19() {
		long[] data = { 4611686018427387904L, -5305222768856390974L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_19 = new BitSet(mk_tokenSet_19());
	private static final long[] mk_tokenSet_20() {
		long[] data = { -9223372036854774752L, 32780L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_20 = new BitSet(mk_tokenSet_20());
	private static final long[] mk_tokenSet_21() {
		long[] data = { 1032208L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_21 = new BitSet(mk_tokenSet_21());
	private static final long[] mk_tokenSet_22() {
		long[] data = { -1125899906844112L, 32551L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_22 = new BitSet(mk_tokenSet_22());
	private static final long[] mk_tokenSet_23() {
		long[] data = { 4610560118516430848L, 1L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_23 = new BitSet(mk_tokenSet_23());
	private static final long[] mk_tokenSet_24() {
		long[] data = { -1125899906843984L, 32551L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_24 = new BitSet(mk_tokenSet_24());
	private static final long[] mk_tokenSet_25() {
		long[] data = { -4611686018427387872L, 32518L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_25 = new BitSet(mk_tokenSet_25());
	private static final long[] mk_tokenSet_26() {
		long[] data = { 4610560118520543762L, 7971L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_26 = new BitSet(mk_tokenSet_26());
	private static final long[] mk_tokenSet_27() {
		long[] data = { 9223372036854774418L, 7971L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_27 = new BitSet(mk_tokenSet_27());
	private static final long[] mk_tokenSet_28() {
		long[] data = { -1125899906844110L, 32551L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_28 = new BitSet(mk_tokenSet_28());
	private static final long[] mk_tokenSet_29() {
		long[] data = { -1125899906844110L, 32559L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_29 = new BitSet(mk_tokenSet_29());
	private static final long[] mk_tokenSet_30() {
		long[] data = { 4610560118519577088L, 3L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_30 = new BitSet(mk_tokenSet_30());
	private static final long[] mk_tokenSet_31() {
		long[] data = { -968064L, 32515L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_31 = new BitSet(mk_tokenSet_31());
	private static final long[] mk_tokenSet_32() {
		long[] data = { -1125899907810688L, -1073650985L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_32 = new BitSet(mk_tokenSet_32());
	private static final long[] mk_tokenSet_33() {
		long[] data = { 0L, 196L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_33 = new BitSet(mk_tokenSet_33());
	private static final long[] mk_tokenSet_34() {
		long[] data = { -1125899906842894L, -5305222767782887425L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_34 = new BitSet(mk_tokenSet_34());
	private static final long[] mk_tokenSet_35() {
		long[] data = { -9223372036854775808L, 7940L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_35 = new BitSet(mk_tokenSet_35());
	private static final long[] mk_tokenSet_36() {
		long[] data = { 128L, 7938L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_36 = new BitSet(mk_tokenSet_36());
	private static final long[] mk_tokenSet_37() {
		long[] data = { 1024L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_37 = new BitSet(mk_tokenSet_37());
	private static final long[] mk_tokenSet_38() {
		long[] data = { 4610560118519578112L, 3L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_38 = new BitSet(mk_tokenSet_38());
	private static final long[] mk_tokenSet_39() {
		long[] data = { -1125899907810654L, 32527L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_39 = new BitSet(mk_tokenSet_39());
	private static final long[] mk_tokenSet_40() {
		long[] data = { -1125899907810782L, 32527L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_40 = new BitSet(mk_tokenSet_40());
	private static final long[] mk_tokenSet_41() {
		long[] data = { -4611686018427387870L, 32526L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_41 = new BitSet(mk_tokenSet_41());
	private static final long[] mk_tokenSet_42() {
		long[] data = { 4611686018427387936L, 16142L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_42 = new BitSet(mk_tokenSet_42());
	private static final long[] mk_tokenSet_43() {
		long[] data = { 32L, 4L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_43 = new BitSet(mk_tokenSet_43());
	private static final long[] mk_tokenSet_44() {
		long[] data = { 9223372036853807746L, 7939L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_44 = new BitSet(mk_tokenSet_44());
	private static final long[] mk_tokenSet_45() {
		long[] data = { 4610560118519577090L, 3L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_45 = new BitSet(mk_tokenSet_45());
	private static final long[] mk_tokenSet_46() {
		long[] data = { 4611686018443117088L, 32526L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_46 = new BitSet(mk_tokenSet_46());
	private static final long[] mk_tokenSet_47() {
		long[] data = { 32L, 7948L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_47 = new BitSet(mk_tokenSet_47());
	private static final long[] mk_tokenSet_48() {
		long[] data = { 9222246136946966048L, -5305222768856383537L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_48 = new BitSet(mk_tokenSet_48());
	private static final long[] mk_tokenSet_49() {
		long[] data = { -9223372036854774752L, 40716L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_49 = new BitSet(mk_tokenSet_49());
	private static final long[] mk_tokenSet_50() {
		long[] data = { 1024L, 4L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_50 = new BitSet(mk_tokenSet_50());
	private static final long[] mk_tokenSet_51() {
		long[] data = { 1088L, 32780L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_51 = new BitSet(mk_tokenSet_51());
	private static final long[] mk_tokenSet_52() {
		long[] data = { -9223372036854775808L, 2L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_52 = new BitSet(mk_tokenSet_52());
	private static final long[] mk_tokenSet_53() {
		long[] data = { -9223372036854775808L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_53 = new BitSet(mk_tokenSet_53());
	private static final long[] mk_tokenSet_54() {
		long[] data = { -4611686018427386784L, 8791026471553597212L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_54 = new BitSet(mk_tokenSet_54());
	private static final long[] mk_tokenSet_55() {
		long[] data = { -4611686018427387904L, -5305222768856390974L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_55 = new BitSet(mk_tokenSet_55());
	private static final long[] mk_tokenSet_56() {
		long[] data = { -9223372036854775808L, 4L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_56 = new BitSet(mk_tokenSet_56());
	private static final long[] mk_tokenSet_57() {
		long[] data = { 4611686018427388032L, -5305222768856309054L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_57 = new BitSet(mk_tokenSet_57());
	private static final long[] mk_tokenSet_58() {
		long[] data = { 9222246136946966176L, -1073650977L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_58 = new BitSet(mk_tokenSet_58());
	private static final long[] mk_tokenSet_59() {
		long[] data = { 9222246136946966176L, -1073650985L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_59 = new BitSet(mk_tokenSet_59());
	private static final long[] mk_tokenSet_60() {
		long[] data = { 0L, 81922L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_60 = new BitSet(mk_tokenSet_60());
	private static final long[] mk_tokenSet_61() {
		long[] data = { 4611686018427387904L, -5305222768856390966L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_61 = new BitSet(mk_tokenSet_61());
	private static final long[] mk_tokenSet_62() {
		long[] data = { 4611686018427389056L, -5305222768856309054L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_62 = new BitSet(mk_tokenSet_62());
	private static final long[] mk_tokenSet_63() {
		long[] data = { 1056L, 4L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_63 = new BitSet(mk_tokenSet_63());
	private static final long[] mk_tokenSet_64() {
		long[] data = { -4611686018427387870L, 24326L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_64 = new BitSet(mk_tokenSet_64());
	private static final long[] mk_tokenSet_65() {
		long[] data = { 9222246136946965184L, -1099511536957L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_65 = new BitSet(mk_tokenSet_65());
	private static final long[] mk_tokenSet_66() {
		long[] data = { 9222246136946965120L, -1099511504189L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_66 = new BitSet(mk_tokenSet_66());
	private static final long[] mk_tokenSet_67() {
		long[] data = { -9223372036854774688L, 4755801205429550876L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_67 = new BitSet(mk_tokenSet_67());
	private static final long[] mk_tokenSet_68() {
		long[] data = { 0L, 32776L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_68 = new BitSet(mk_tokenSet_68());
	private static final long[] mk_tokenSet_69() {
		long[] data = { -1360L, 32551L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_69 = new BitSet(mk_tokenSet_69());
	private static final long[] mk_tokenSet_70() {
		long[] data = { -9223372036854775808L, 6L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_70 = new BitSet(mk_tokenSet_70());
	private static final long[] mk_tokenSet_71() {
		long[] data = { -1125899906843918L, 24383L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_71 = new BitSet(mk_tokenSet_71());
	private static final long[] mk_tokenSet_72() {
		long[] data = { -4611686018411658720L, 32518L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_72 = new BitSet(mk_tokenSet_72());
	private static final long[] mk_tokenSet_73() {
		long[] data = { 4611686018427387904L, 24576L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_73 = new BitSet(mk_tokenSet_73());
	private static final long[] mk_tokenSet_74() {
		long[] data = { -1125899906844112L, -5305222768856334361L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_74 = new BitSet(mk_tokenSet_74());
	private static final long[] mk_tokenSet_75() {
		long[] data = { -9223372036854775776L, 4L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_75 = new BitSet(mk_tokenSet_75());
	private static final long[] mk_tokenSet_76() {
		long[] data = { -9223372036854775774L, 4L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_76 = new BitSet(mk_tokenSet_76());
	private static final long[] mk_tokenSet_77() {
		long[] data = { 4610560118520543760L, 139043L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_77 = new BitSet(mk_tokenSet_77());
	private static final long[] mk_tokenSet_78() {
		long[] data = { 9222246136947898880L, 16163L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_78 = new BitSet(mk_tokenSet_78());
	private static final long[] mk_tokenSet_79() {
		long[] data = { 9223372036854774416L, 32547L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_79 = new BitSet(mk_tokenSet_79());
	private static final long[] mk_tokenSet_80() {
		long[] data = { 9222246136947932848L, -5305222768051208221L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_80 = new BitSet(mk_tokenSet_80());
	private static final long[] mk_tokenSet_81() {
		long[] data = { 9222246136947931792L, 32547L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_81 = new BitSet(mk_tokenSet_81());
	private static final long[] mk_tokenSet_82() {
		long[] data = { 4611686018427388064L, -5305222768051346750L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_82 = new BitSet(mk_tokenSet_82());
	private static final long[] mk_tokenSet_83() {
		long[] data = { 9222246136947932848L, -268468225L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_83 = new BitSet(mk_tokenSet_83());
	private static final long[] mk_tokenSet_84() {
		long[] data = { 4611686018427387936L, -5305222768856390974L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_84 = new BitSet(mk_tokenSet_84());
	private static final long[] mk_tokenSet_85() {
		long[] data = { 9222246136946965152L, -1073650985L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_85 = new BitSet(mk_tokenSet_85());
	private static final long[] mk_tokenSet_86() {
		long[] data = { 9222246136946965120L, -1099511536949L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_86 = new BitSet(mk_tokenSet_86());
	private static final long[] mk_tokenSet_87() {
		long[] data = { 9222246136947932848L, -32769L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_87 = new BitSet(mk_tokenSet_87());
	private static final long[] mk_tokenSet_88() {
		long[] data = { 9222246136947932848L, -5305222767782903837L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_88 = new BitSet(mk_tokenSet_88());
	private static final long[] mk_tokenSet_89() {
		long[] data = { -78L, -32769L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_89 = new BitSet(mk_tokenSet_89());
	private static final long[] mk_tokenSet_90() {
		long[] data = { -9223372036854774688L, 1098437926684L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_90 = new BitSet(mk_tokenSet_90());
	private static final long[] mk_tokenSet_91() {
		long[] data = { -9223372036854774688L, 4611687116865314588L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_91 = new BitSet(mk_tokenSet_91());
	private static final long[] mk_tokenSet_92() {
		long[] data = { -9223372036854774688L, 4611688216376942364L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_92 = new BitSet(mk_tokenSet_92());
	private static final long[] mk_tokenSet_93() {
		long[] data = { -9223372036854774688L, 4611690415400197916L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_93 = new BitSet(mk_tokenSet_93());
	private static final long[] mk_tokenSet_94() {
		long[] data = { -9223372036854774688L, 4611694813446709020L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_94 = new BitSet(mk_tokenSet_94());
	private static final long[] mk_tokenSet_95() {
		long[] data = { -9223372036854774688L, 4611703609539731228L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_95 = new BitSet(mk_tokenSet_95());
	private static final long[] mk_tokenSet_96() {
		long[] data = { -9223372036854774688L, 4611721201725775644L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_96 = new BitSet(mk_tokenSet_96());
	private static final long[] mk_tokenSet_97() {
		long[] data = { -9223372036854774688L, 4611826754842042140L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_97 = new BitSet(mk_tokenSet_97());
	private static final long[] mk_tokenSet_98() {
		long[] data = { -9223372036854774688L, 4613937817167372060L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_98 = new BitSet(mk_tokenSet_98());
	private static final long[] mk_tokenSet_99() {
		long[] data = { -9223372036854774688L, 4620693216608427804L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_99 = new BitSet(mk_tokenSet_99());
	private static final long[] mk_tokenSet_100() {
		long[] data = { 0L, 108086391056900096L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_100 = new BitSet(mk_tokenSet_100());
	private static final long[] mk_tokenSet_101() {
		long[] data = { -9223372036854774688L, 4647714814372650780L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_101 = new BitSet(mk_tokenSet_101());
	private static final long[] mk_tokenSet_102() {
		long[] data = { -4611686018427386144L, -1073610786L, 255L, 0L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_102 = new BitSet(mk_tokenSet_102());
	private static final long[] mk_tokenSet_103() {
		long[] data = { -1392L, 32551L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_103 = new BitSet(mk_tokenSet_103());
	private static final long[] mk_tokenSet_104() {
		long[] data = { -9223372036854775808L, 136L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_104 = new BitSet(mk_tokenSet_104());
	private static final long[] mk_tokenSet_105() {
		long[] data = { -9223372036854774688L, 4755801205429551004L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_105 = new BitSet(mk_tokenSet_105());
	private static final long[] mk_tokenSet_106() {
		long[] data = { -9223372036854775808L, 12L, 0L, 0L};
		return data;
	}
	public static final BitSet _tokenSet_106 = new BitSet(mk_tokenSet_106());
	
	}
