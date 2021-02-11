// $ANTLR 2.7.7 (2010-12-23): "NewCParser.g" -> "NewCParser.java"$

package cetus.base.grammars;

public interface NEWCTokenTypes {
	int EOF = 1;
	int NULL_TREE_LOOKAHEAD = 3;
	int LITERAL_typedef = 4;
	int SEMI = 5;
	int VARARGS = 6;
	int LCURLY = 7;
	int LITERAL_asm = 8;
	int LITERAL_volatile = 9;
	int RCURLY = 10;
	int LITERAL_struct = 11;
	int LITERAL_union = 12;
	int LITERAL_enum = 13;
	int LITERAL_auto = 14;
	int LITERAL_register = 15;
	int LITERAL___thread = 16;
	int LITERAL_extern = 17;
	int LITERAL_static = 18;
	int LITERAL_inline = 19;
	int LITERAL_const = 20;
	int LITERAL_restrict = 21;
	int LITERAL___nvl__ = 22;
	int LITERAL___nvl_wp__ = 23;
	int LITERAL_void = 24;
	int LITERAL_char = 25;
	int LITERAL_short = 26;
	int LITERAL_int = 27;
	int LITERAL_long = 28;
	int LITERAL_float = 29;
	int LITERAL_double = 30;
	int LITERAL_signed = 31;
	int LITERAL_unsigned = 32;
	int LITERAL__Bool = 33;
	int LITERAL__Complex = 34;
	int LITERAL__Imaginary = 35;
	// "int8_t" = 36
	// "uint8_t" = 37
	// "int16_t" = 38
	// "uint16_t" = 39
	// "int32_t" = 40
	// "uint32_t" = 41
	// "int64_t" = 42
	// "uint64_t" = 43
	int LITERAL_size_t = 44;
	// "_Float128" = 45
	// "__float128" = 46
	// "__float80" = 47
	// "__ibm128" = 48
	// "_Float16" = 49
	int LITERAL__Nullable = 50;
	int LITERAL___global__ = 51;
	int LITERAL___shared__ = 52;
	int LITERAL___host__ = 53;
	int LITERAL___device__ = 54;
	int LITERAL___constant__ = 55;
	int LITERAL___noinline__ = 56;
	int LITERAL___kernel = 57;
	int LITERAL___global = 58;
	int LITERAL___local = 59;
	int LITERAL___constant = 60;
	int LITERAL_typeof = 61;
	int LPAREN = 62;
	int RPAREN = 63;
	int LITERAL___complex = 64;
	int ID = 65;
	int COMMA = 66;
	int COLON = 67;
	int ASSIGN = 68;
	int LITERAL___declspec = 69;
	int Number = 70;
	int StringLiteral = 71;
	int LITERAL___attribute = 72;
	int LITERAL___asm = 73;
	int LITERAL___OSX_AVAILABLE_STARTING = 74;
	int LITERAL___OSX_AVAILABLE_BUT_DEPRECATED = 75;
	int LITERAL___OSX_AVAILABLE_BUT_DEPRECATED_MSG = 76;
	int STAR = 77;
	int LBRACKET = 78;
	int RBRACKET = 79;
	int DOT = 80;
	int LITERAL___label__ = 81;
	int LITERAL_while = 82;
	int LITERAL_do = 83;
	int LITERAL_for = 84;
	int LITERAL_goto = 85;
	int LITERAL_continue = 86;
	int LITERAL_break = 87;
	int LITERAL_return = 88;
	int LITERAL_case = 89;
	int LITERAL_default = 90;
	int LITERAL_if = 91;
	int LITERAL_else = 92;
	int LITERAL_switch = 93;
	int DIV_ASSIGN = 94;
	int PLUS_ASSIGN = 95;
	int MINUS_ASSIGN = 96;
	int STAR_ASSIGN = 97;
	int MOD_ASSIGN = 98;
	int RSHIFT_ASSIGN = 99;
	int LSHIFT_ASSIGN = 100;
	int BAND_ASSIGN = 101;
	int BOR_ASSIGN = 102;
	int BXOR_ASSIGN = 103;
	int LOR = 104;
	int LAND = 105;
	int BOR = 106;
	int BXOR = 107;
	int BAND = 108;
	int EQUAL = 109;
	int NOT_EQUAL = 110;
	int LT = 111;
	int LTE = 112;
	int GT = 113;
	int GTE = 114;
	int LSHIFT = 115;
	int RSHIFT = 116;
	int PLUS = 117;
	int MINUS = 118;
	int DIV = 119;
	int MOD = 120;
	int LITERAL___builtin_nvl_get_root = 121;
	int LITERAL___builtin_nvl_alloc_nv = 122;
	int PTR = 123;
	int INC = 124;
	int DEC = 125;
	int QUESTION = 126;
	int LITERAL_sizeof = 127;
	int LITERAL___alignof__ = 128;
	int LITERAL___builtin_va_arg = 129;
	int LITERAL___builtin_offsetof = 130;
	int BNOT = 131;
	int LNOT = 132;
	int LITERAL___real = 133;
	int LITERAL___imag = 134;
	int CharLiteral = 135;
	int IntOctalConst = 136;
	int LongOctalConst = 137;
	int UnsignedOctalConst = 138;
	int IntIntConst = 139;
	int LongIntConst = 140;
	int UnsignedIntConst = 141;
	int IntHexConst = 142;
	int LongHexConst = 143;
	int UnsignedHexConst = 144;
	int FloatDoubleConst = 145;
	int DoubleDoubleConst = 146;
	int LongDoubleConst = 147;
	int NTypedefName = 148;
	int NInitDecl = 149;
	int NDeclarator = 150;
	int NStructDeclarator = 151;
	int NDeclaration = 152;
	int NCast = 153;
	int NPointerGroup = 154;
	int NExpressionGroup = 155;
	int NFunctionCallArgs = 156;
	int NNonemptyAbstractDeclarator = 157;
	int NInitializer = 158;
	int NStatementExpr = 159;
	int NEmptyExpression = 160;
	int NParameterTypeList = 161;
	int NFunctionDef = 162;
	int NCompoundStatement = 163;
	int NParameterDeclaration = 164;
	int NCommaExpr = 165;
	int NUnaryExpr = 166;
	int NLabel = 167;
	int NPostfixExpr = 168;
	int NRangeExpr = 169;
	int NStringSeq = 170;
	int NInitializerElementLabel = 171;
	int NLcurlyInitializer = 172;
	int NAsmAttribute = 173;
	int NGnuAsmExpr = 174;
	int NTypeMissing = 175;
	int LITERAL___extension__ = 176;
	int Vocabulary = 177;
	int Whitespace = 178;
	int Comment = 179;
	int CPPComment = 180;
	int PREPROC_DIRECTIVE = 181;
	int Space = 182;
	int LineDirective = 183;
	int BadStringLiteral = 184;
	int Escape = 185;
	int IntSuffix = 186;
	int NumberSuffix = 187;
	int Digit = 188;
	int HexDigit = 189;
	int HexFloatTail = 190;
	int Exponent = 191;
	int IDMEAT = 192;
	int WideCharLiteral = 193;
	int WideStringLiteral = 194;
}