/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0.4"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 9 "parser/AspenParser.y" /* yacc.c:339  */

    #include "common/AST.h"
    #include "parser/AspenParseNode.h"
    ASTAppModel *globalapp = NULL;
    ASTMachModel *globalmach = NULL;

    extern int yylex();
    extern int yylineno;
    extern char *yytext;
    void yyerror(const char *);

#line 78 "parser/AspenParser.cpp" /* yacc.c:339  */

# ifndef YY_NULLPTR
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULLPTR nullptr
#  else
#   define YY_NULLPTR 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "AspenParser.hpp".  */
#ifndef YY_YY_PARSER_ASPENPARSER_HPP_INCLUDED
# define YY_YY_PARSER_ASPENPARSER_HPP_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif
/* "%code requires" blocks.  */
#line 23 "parser/AspenParser.y" /* yacc.c:355  */

#include "parser/AspenParseNode.h"

typedef struct YYLTYPE {
  int first_line;
  int first_column;
  int last_line;
  int last_column;
  int first_filepos;
  int last_filepos;
  string filename;
} YYLTYPE;

# define YYLTYPE_IS_DECLARED 1 /* alert the parser that we have our own definition */

# define YYLLOC_DEFAULT(Current, Rhs, N)                               \
    do                                                                 \
      if (N)                                                           \
        {                                                              \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;       \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;     \
          (Current).first_filepos= YYRHSLOC (Rhs, 1).first_filepos;    \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;        \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;      \
          (Current).last_filepos = YYRHSLOC (Rhs, N).last_filepos;     \
          (Current).filename     = YYRHSLOC (Rhs, 1).filename;         \
        }                                                              \
      else                                                             \
        { /* empty RHS */                                              \
          (Current).first_line   = (Current).last_line   =             \
            YYRHSLOC (Rhs, 0).last_line;                               \
          (Current).first_column = (Current).last_column =             \
            YYRHSLOC (Rhs, 0).last_column;                             \
          (Current).first_filepos = (Current).last_filepos =           \
            YYRHSLOC (Rhs, 0).last_filepos;                            \
          (Current).filename  = "";                                    \
        }                                                              \
    while (0)


#line 149 "parser/AspenParser.cpp" /* yacc.c:355  */

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    TKW_PARAM = 258,
    TKW_ENUM = 259,
    TKW_IN = 260,
    TKW_WITH = 261,
    TKW_OF = 262,
    TKW_SIZE = 263,
    TKW_MODEL = 264,
    TKW_KERNEL = 265,
    TKW_DATA = 266,
    TKW_SAMPLE = 267,
    TKW_IMPORT = 268,
    TKW_AS = 269,
    TKW_TO = 270,
    TKW_FROM = 271,
    TKW_CALL = 272,
    TKW_ITERATE = 273,
    TKW_MAP = 274,
    TKW_PAR = 275,
    TKW_SEQ = 276,
    TKW_EXECUTE = 277,
    TKW_IF = 278,
    TKW_ELSE = 279,
    TKW_PROBABILITY = 280,
    TKW_SWITCH = 281,
    TKW_RESOURCE = 282,
    TKW_CONFLICT = 283,
    TKW_POWER = 284,
    TKW_STATIC = 285,
    TKW_DYNAMIC = 286,
    TKW_PROPERTY = 287,
    TIDENT = 288,
    TSTRING = 289,
    TINT = 290,
    TREAL = 291,
    TKW_10POWER = 292,
    TDOTDOT = 293,
    TEQUAL = 294,
    TLPAREN = 295,
    TRPAREN = 296,
    TLBRACE = 297,
    TRBRACE = 298,
    TLBRACKET = 299,
    TRBRACKET = 300,
    TCOMMA = 301,
    TDOT = 302,
    TPLUS = 303,
    TMINUS = 304,
    TMUL = 305,
    TDIV = 306,
    TEXP = 307,
    TCOMPEQ = 308,
    TCOMPNE = 309,
    TCOMPLT = 310,
    TCOMPGT = 311,
    TCOMPLE = 312,
    TCOMPGE = 313,
    TAND = 314,
    TOR = 315,
    NEG = 316
  };
#endif
/* Tokens.  */
#define TKW_PARAM 258
#define TKW_ENUM 259
#define TKW_IN 260
#define TKW_WITH 261
#define TKW_OF 262
#define TKW_SIZE 263
#define TKW_MODEL 264
#define TKW_KERNEL 265
#define TKW_DATA 266
#define TKW_SAMPLE 267
#define TKW_IMPORT 268
#define TKW_AS 269
#define TKW_TO 270
#define TKW_FROM 271
#define TKW_CALL 272
#define TKW_ITERATE 273
#define TKW_MAP 274
#define TKW_PAR 275
#define TKW_SEQ 276
#define TKW_EXECUTE 277
#define TKW_IF 278
#define TKW_ELSE 279
#define TKW_PROBABILITY 280
#define TKW_SWITCH 281
#define TKW_RESOURCE 282
#define TKW_CONFLICT 283
#define TKW_POWER 284
#define TKW_STATIC 285
#define TKW_DYNAMIC 286
#define TKW_PROPERTY 287
#define TIDENT 288
#define TSTRING 289
#define TINT 290
#define TREAL 291
#define TKW_10POWER 292
#define TDOTDOT 293
#define TEQUAL 294
#define TLPAREN 295
#define TRPAREN 296
#define TLBRACE 297
#define TRBRACE 298
#define TLBRACKET 299
#define TRBRACKET 300
#define TCOMMA 301
#define TDOT 302
#define TPLUS 303
#define TMINUS 304
#define TMUL 305
#define TDIV 306
#define TEXP 307
#define TCOMPEQ 308
#define TCOMPNE 309
#define TCOMPLT 310
#define TCOMPGT 311
#define TCOMPLE 312
#define TCOMPGE 313
#define TAND 314
#define TOR 315
#define NEG 316

/* Value type.  */

/* Location type.  */
#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE YYLTYPE;
struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


extern YYSTYPE yylval;
extern YYLTYPE yylloc;
int yyparse (void);

#endif /* !YY_YY_PARSER_ASPENPARSER_HPP_INCLUDED  */

/* Copy the second part of user declarations.  */

#line 303 "parser/AspenParser.cpp" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE
# if (defined __GNUC__                                               \
      && (2 < __GNUC__ || (__GNUC__ == 2 && 96 <= __GNUC_MINOR__)))  \
     || defined __SUNPRO_C && 0x5110 <= __SUNPRO_C
#  define YY_ATTRIBUTE(Spec) __attribute__(Spec)
# else
#  define YY_ATTRIBUTE(Spec) /* empty */
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# define YY_ATTRIBUTE_PURE   YY_ATTRIBUTE ((__pure__))
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# define YY_ATTRIBUTE_UNUSED YY_ATTRIBUTE ((__unused__))
#endif

#if !defined _Noreturn \
     && (!defined __STDC_VERSION__ || __STDC_VERSION__ < 201112)
# if defined _MSC_VER && 1200 <= _MSC_VER
#  define _Noreturn __declspec (noreturn)
# else
#  define _Noreturn YY_ATTRIBUTE ((__noreturn__))
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
             && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  23
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   566

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  62
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  54
/* YYNRULES -- Number of rules.  */
#define YYNRULES  145
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  342

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   316

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   162,   162,   166,   173,   182,   190,   198,   203,   211,
     219,   220,   224,   225,   226,   227,   228,   232,   233,   238,
     239,   243,   250,   255,   265,   270,   275,   280,   288,   289,
     298,   307,   308,   312,   323,   332,   340,   349,   350,   353,
     354,   358,   359,   362,   383,   384,   388,   389,   393,   394,
     398,   399,   407,   408,   409,   413,   414,   415,   419,   424,
     429,   438,   443,   454,   458,   462,   466,   471,   476,   486,
     493,   499,   513,   518,   523,   528,   536,   541,   549,   556,
     560,   567,   571,   579,   583,   591,   592,   593,   594,   595,
     596,   597,   598,   602,   607,   613,   621,   627,   636,   641,
     649,   654,   662,   663,   667,   671,   675,   680,   685,   695,
     705,   709,   713,   720,   721,   722,   726,   727,   731,   732,
     736,   737,   741,   742,   746,   747,   751,   754,   755,   756,
     757,   758,   759,   760,   761,   762,   763,   764,   765,   766,
     770,   778,   779,   780,   784,   790
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "TKW_PARAM", "TKW_ENUM", "TKW_IN",
  "TKW_WITH", "TKW_OF", "TKW_SIZE", "TKW_MODEL", "TKW_KERNEL", "TKW_DATA",
  "TKW_SAMPLE", "TKW_IMPORT", "TKW_AS", "TKW_TO", "TKW_FROM", "TKW_CALL",
  "TKW_ITERATE", "TKW_MAP", "TKW_PAR", "TKW_SEQ", "TKW_EXECUTE", "TKW_IF",
  "TKW_ELSE", "TKW_PROBABILITY", "TKW_SWITCH", "TKW_RESOURCE",
  "TKW_CONFLICT", "TKW_POWER", "TKW_STATIC", "TKW_DYNAMIC", "TKW_PROPERTY",
  "TIDENT", "TSTRING", "TINT", "TREAL", "TKW_10POWER", "TDOTDOT", "TEQUAL",
  "TLPAREN", "TRPAREN", "TLBRACE", "TRBRACE", "TLBRACKET", "TRBRACKET",
  "TCOMMA", "TDOT", "TPLUS", "TMINUS", "TMUL", "TDIV", "TEXP", "TCOMPEQ",
  "TCOMPNE", "TCOMPLT", "TCOMPGT", "TCOMPLE", "TCOMPGE", "TAND", "TOR",
  "NEG", "$accept", "begin", "mach", "machcontent", "component",
  "componentstatements", "componentstmt", "subcomponent", "property",
  "conflict", "resource", "traitdefinitions", "power", "app", "kernels",
  "kernel", "ident", "value", "optionalident", "constant",
  "globalstatements", "localstatements", "kernelstatements",
  "execstatements", "globalstmt", "localstmt", "kernelstmt", "execstmt",
  "paramstmt", "arraylist", "valuelist", "identlist", "samplestmt",
  "datastmt", "importstmt", "kernelcall", "comparison", "ifthencontrol",
  "caseitems", "probabilitycontrol", "switchcontrol", "optionalstring",
  "controlstmt", "tofrom", "optionaltraitlist", "traitlist", "trait",
  "argdecllist", "argdecl", "quantity", "expr", "functioncall",
  "arguments", "namedarguments", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316
};
# endif

#define YYPACT_NINF -224

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-224)))

#define YYTABLE_NINF -38

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      87,    59,    83,    93,    99,   108,   123,    48,  -224,   110,
    -224,  -224,  -224,  -224,  -224,  -224,    24,   113,   152,   158,
      15,   160,   183,  -224,  -224,  -224,    59,   100,   184,   200,
     272,   212,   100,  -224,   224,   513,  -224,   133,  -224,  -224,
    -224,   100,   100,  -224,  -224,  -224,    16,  -224,   187,  -224,
     155,   114,  -224,   206,  -224,   378,   254,   239,   249,   263,
     264,   262,   -14,   422,  -224,  -224,  -224,  -224,  -224,  -224,
     100,   100,   455,   226,   134,   100,   100,   100,   100,   100,
      -2,   261,  -224,   274,    59,    -4,  -224,  -224,  -224,   275,
     269,   265,   270,   270,   270,  -224,   280,  -224,  -224,   408,
     168,   479,  -224,   438,   443,   112,   112,   226,   226,   226,
     286,   100,   288,  -224,    38,  -224,  -224,   289,   283,   307,
     316,  -224,  -224,  -224,  -224,  -224,   100,   309,   100,   100,
     169,   312,   119,   421,   100,   322,   326,  -224,   408,   100,
     408,   186,  -224,   334,    59,    59,   213,  -224,    59,    59,
      59,    59,   346,   346,   -18,   341,   361,   342,   421,   170,
    -224,  -224,  -224,  -224,  -224,  -224,  -224,  -224,  -224,  -224,
    -224,   408,   355,   270,   487,  -224,    -3,   234,  -224,  -224,
     365,   119,   392,   369,  -224,   270,   270,  -224,   376,   389,
     393,   346,   100,   270,   388,   185,  -224,  -224,  -224,   100,
     430,  -224,  -224,  -224,   243,  -224,   403,   421,  -224,   212,
     100,   346,   346,   449,   449,   415,   409,   -16,   465,    -9,
     420,   423,  -224,   408,   440,  -224,    22,    -3,   421,   218,
    -224,   228,   435,   437,   230,   245,   270,   -23,  -224,   415,
     441,   100,   100,   100,   100,   100,   100,   100,   100,   442,
    -224,   456,   449,   457,    55,   417,  -224,  -224,   244,   279,
    -224,  -224,   449,   449,  -224,  -224,   135,  -224,  -224,    50,
     449,  -224,  -224,   408,   408,   408,   408,   408,   408,   449,
     449,   298,   270,   447,  -224,   464,  -224,  -224,   313,   325,
     492,   469,   493,   511,  -224,   340,   352,   367,  -224,    45,
     501,   104,  -224,  -224,   270,  -224,  -224,   500,  -224,   510,
     504,  -224,   506,  -224,   270,   516,  -224,    90,   512,   505,
    -224,    18,  -224,   449,  -224,   509,   511,   100,   500,   449,
    -224,   379,   270,  -224,   460,  -224,   394,   514,  -224,  -224,
    -224,  -224
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,     0,     2,     4,
       6,     3,     5,    53,    54,    52,    37,     0,     0,     0,
       0,     0,     0,     1,     8,     7,     0,     0,     0,     0,
       0,     0,     0,    80,     0,     0,    38,    37,    39,    40,
      43,     0,     0,   129,   127,   128,    63,   139,     0,    76,
       0,     0,    44,     0,    79,     0,    81,     0,     0,     0,
       0,     0,     0,     0,    10,    16,    12,    15,    13,    14,
     141,     0,     0,   138,     0,     0,     0,     0,     0,     0,
       0,     0,    69,     0,     0,     0,    31,    45,   126,     0,
       0,     0,     0,     0,    20,    17,     0,     9,    11,   142,
       0,     0,   132,    65,     0,   133,   134,   135,   136,   137,
       0,   141,     0,    77,     0,    30,    32,     0,    82,     0,
       0,    28,    29,    19,    18,   140,     0,   130,     0,   141,
       0,     0,     0,     0,     0,     0,     0,    21,   143,     0,
      64,     0,    66,     0,     0,     0,     0,   122,     0,     0,
      41,    41,   102,   102,   102,     0,     0,     0,     0,     0,
      46,    48,    56,   105,    57,    55,   104,   110,   111,   112,
      60,   144,     0,     0,     0,    67,     0,     0,   125,   124,
       0,     0,     0,    83,    42,     0,     0,   103,     0,     0,
       0,   102,     0,     0,     0,     0,    47,    33,    49,     0,
      23,   131,    72,    73,     0,    68,     0,     0,   123,     0,
     141,   102,   102,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    34,   145,     0,    70,     0,     0,     0,     0,
      78,     0,     0,     0,     0,     0,     0,     0,    50,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      98,     0,     0,     0,     0,    22,    74,    75,     0,     0,
      35,    84,     0,     0,   106,   107,   113,    58,    51,     0,
       0,    85,    86,    87,    88,    89,    90,    91,    92,     0,
       0,     0,     0,     0,    24,     0,    71,    36,     0,     0,
       0,     0,     0,   116,    59,     0,     0,     0,    96,     0,
       0,     0,   108,   109,     0,   114,   115,     0,    61,    93,
       0,    97,     0,   100,     0,     0,    26,   113,   120,   117,
     118,     0,    99,     0,    25,     0,   116,     0,     0,     0,
      95,     0,     0,    62,     0,   119,     0,     0,    27,   121,
      94,   101
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -224,  -224,  -224,  -224,   544,  -224,   491,  -224,  -224,  -224,
    -224,  -224,  -224,  -224,  -224,   470,    10,  -224,   405,  -224,
    -224,   351,  -151,   320,    19,  -153,  -150,  -223,  -131,  -224,
     333,  -224,  -224,  -121,  -120,  -224,   -57,   240,   281,  -224,
    -224,  -130,  -224,   247,   236,  -224,   237,  -224,   385,   -20,
     -24,   -30,  -107,  -224
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     7,     8,     9,    10,    63,    64,    65,    66,    67,
      68,   255,    69,    11,    85,    86,    43,    44,   185,    45,
      51,   158,   159,   237,    12,   160,   161,   238,    13,   177,
     204,    50,   163,    14,    15,   166,   217,   167,   219,   168,
     169,   188,   170,   293,   308,   319,   320,   146,   147,   220,
      99,    47,   100,   118
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      33,    54,   162,    46,   130,   196,    84,   195,    55,   198,
     236,    17,   164,   165,   268,   249,   187,    72,    73,    95,
     267,    74,   141,   189,   190,   240,    32,   162,    25,    31,
      32,   110,   202,   203,   250,    32,    36,   164,   165,   115,
     111,   155,    96,   241,   242,   198,   268,   101,    23,    52,
     104,   105,   106,   107,   108,   109,   229,   256,   257,    32,
     329,   216,   234,   235,    75,    76,    77,    78,    79,   312,
      87,    26,   121,   122,   123,   196,   162,   259,   132,   198,
     133,   232,   233,   236,   198,   198,   164,   165,   313,    32,
       1,     2,    16,   294,   114,   283,     3,   162,     4,    32,
       5,   281,   138,   231,   140,   291,   292,   164,   165,   198,
     171,   288,   289,     1,     2,   174,    18,     1,     2,   295,
       6,     4,   144,     5,    84,     4,    19,     5,   296,   297,
     145,   198,    20,    37,   191,    38,    39,    40,   198,   198,
      41,    21,   290,     6,   315,   198,   198,   198,    32,    42,
     291,   292,    27,   200,   178,   179,    22,    28,   182,   183,
     184,   184,    77,    78,    79,   211,   212,   103,   218,    38,
      39,    40,   331,    70,    41,   223,    34,    71,   336,   230,
      26,   198,   148,    42,   271,   272,   198,   149,   150,   151,
     152,   153,   154,   155,    29,   156,   157,   148,    82,   251,
      30,    83,   149,   150,   151,   152,   153,   154,   155,   125,
     156,   157,   142,   197,   126,   126,   266,   218,   218,   273,
     274,   275,   276,   277,   278,    35,    80,   175,   222,    48,
     148,    81,   126,    49,   284,   149,   150,   151,   152,   153,
     154,   155,   148,   156,   157,    53,    70,   149,   150,   151,
     152,   153,   154,   155,   180,   156,   157,   148,    56,   181,
      89,   260,   149,   150,   151,   152,   153,   154,   155,   261,
     156,   157,    90,   264,   126,     1,     2,   205,    79,   251,
     206,   316,    91,     4,   317,     5,   225,   286,   265,   226,
     226,   148,    92,    93,   324,    94,   149,   150,   151,   152,
     153,   154,   155,   334,   156,   157,   112,   113,   117,   119,
     148,   120,   338,   124,    32,   149,   150,   151,   152,   153,
     154,   155,   287,   156,   157,   148,   129,   131,   134,   135,
     149,   150,   151,   152,   153,   154,   155,   148,   156,   157,
     136,   298,   149,   150,   151,   152,   153,   154,   155,   137,
     156,   157,   148,   139,   143,   172,   302,   149,   150,   151,
     152,   153,   154,   155,   148,   156,   157,   173,   303,   149,
     150,   151,   152,   153,   154,   155,   176,   156,   157,   148,
     187,   192,   194,   309,   149,   150,   151,   152,   153,   154,
     155,   148,   156,   157,   199,   310,   149,   150,   151,   152,
     153,   154,   155,   193,   156,   157,   148,   207,   209,   210,
     311,   149,   150,   151,   152,   153,   154,   155,   213,   156,
     157,   221,   337,    88,     1,     2,    75,    76,    77,    78,
      79,   214,     4,   148,     5,   215,   224,   340,   149,   150,
     151,   152,   153,   154,   155,   227,   156,   157,   236,    57,
      58,   239,    59,    60,    61,    62,    75,    76,    77,    78,
      79,   148,   252,   285,   253,    97,   149,   150,   151,   152,
     153,   154,   155,   254,   156,   157,   -37,   262,    70,   263,
     300,   128,    71,   270,   279,    26,   -37,   -37,   -37,   -37,
     -37,    75,    76,    77,    78,    79,   102,   301,   280,   282,
     304,   339,   305,    75,    76,    77,    78,    79,    75,    76,
      77,    78,    79,    75,    76,    77,    78,    79,   243,   244,
     245,   246,   247,   248,   127,   307,   306,    75,    76,    77,
      78,    79,   201,   318,   321,    75,    76,    77,    78,    79,
      57,    58,   314,    59,    60,    61,    62,   322,   323,   325,
     332,   328,   327,    24,    98,   116,   186,   341,   228,   269,
     258,   330,   333,   299,   326,   335,   208
};

static const yytype_uint16 yycheck[] =
{
      20,    31,   133,    27,   111,   158,    10,   158,    32,   159,
      33,     1,   133,   133,   237,    24,    34,    41,    42,    33,
      43,     5,   129,   153,   154,    41,    44,   158,     9,    14,
      44,    33,    35,    36,    43,    44,    26,   158,   158,    43,
      42,    23,    62,    59,    60,   195,   269,    71,     0,    30,
      74,    75,    76,    77,    78,    79,   207,    35,    36,    44,
      42,   191,   213,   214,    48,    49,    50,    51,    52,    24,
      51,    47,    92,    93,    94,   228,   207,   228,    40,   229,
      42,   211,   212,    33,   234,   235,   207,   207,    43,    44,
       3,     4,    33,    43,    84,    40,     9,   228,    11,    44,
      13,   252,   126,   210,   128,    15,    16,   228,   228,   259,
     134,   262,   263,     3,     4,   139,    33,     3,     4,   270,
      33,    11,     3,    13,    10,    11,    33,    13,   279,   280,
      11,   281,    33,    33,   154,    35,    36,    37,   288,   289,
      40,    33,     7,    33,    40,   295,   296,   297,    44,    49,
      15,    16,    39,   173,   144,   145,    33,    44,   148,   149,
     150,   151,    50,    51,    52,   185,   186,    33,   192,    35,
      36,    37,   323,    40,    40,   199,    16,    44,   329,   209,
      47,   331,    12,    49,   241,   242,   336,    17,    18,    19,
      20,    21,    22,    23,    42,    25,    26,    12,    43,   219,
      42,    46,    17,    18,    19,    20,    21,    22,    23,    41,
      25,    26,    43,    43,    46,    46,   236,   241,   242,   243,
     244,   245,   246,   247,   248,    42,    39,    41,    43,    45,
      12,    44,    46,    33,   254,    17,    18,    19,    20,    21,
      22,    23,    12,    25,    26,    33,    40,    17,    18,    19,
      20,    21,    22,    23,    41,    25,    26,    12,    34,    46,
       6,    43,    17,    18,    19,    20,    21,    22,    23,    41,
      25,    26,    33,    43,    46,     3,     4,    43,    52,   299,
      46,   301,    33,    11,   304,    13,    43,    43,    43,    46,
      46,    12,    29,    29,   314,    33,    17,    18,    19,    20,
      21,    22,    23,   327,    25,    26,    45,    33,    33,    40,
      12,    46,   332,    33,    44,    17,    18,    19,    20,    21,
      22,    23,    43,    25,    26,    12,    40,    39,    39,    46,
      17,    18,    19,    20,    21,    22,    23,    12,    25,    26,
      33,    43,    17,    18,    19,    20,    21,    22,    23,    33,
      25,    26,    12,    44,    42,    33,    43,    17,    18,    19,
      20,    21,    22,    23,    12,    25,    26,    41,    43,    17,
      18,    19,    20,    21,    22,    23,    42,    25,    26,    12,
      34,    40,    40,    43,    17,    18,    19,    20,    21,    22,
      23,    12,    25,    26,    39,    43,    17,    18,    19,    20,
      21,    22,    23,    42,    25,    26,    12,    42,    16,    40,
      43,    17,    18,    19,    20,    21,    22,    23,    42,    25,
      26,    33,    43,    45,     3,     4,    48,    49,    50,    51,
      52,    42,    11,    12,    13,    42,     6,    43,    17,    18,
      19,    20,    21,    22,    23,    42,    25,    26,    33,    27,
      28,    42,    30,    31,    32,    33,    48,    49,    50,    51,
      52,    12,    42,    46,    41,    43,    17,    18,    19,    20,
      21,    22,    23,    33,    25,    26,    38,    42,    40,    42,
      33,    38,    44,    42,    42,    47,    48,    49,    50,    51,
      52,    48,    49,    50,    51,    52,    41,    33,    42,    42,
       8,    41,    33,    48,    49,    50,    51,    52,    48,    49,
      50,    51,    52,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    45,    14,    33,    48,    49,    50,
      51,    52,    45,    33,    24,    48,    49,    50,    51,    52,
      27,    28,    41,    30,    31,    32,    33,    43,    42,    33,
      41,    46,    40,     9,    63,    85,   151,    43,   207,   239,
     227,   321,   326,   282,   317,   328,   181
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     9,    11,    13,    33,    63,    64,    65,
      66,    75,    86,    90,    95,    96,    33,    78,    33,    33,
      33,    33,    33,     0,    66,    86,    47,    39,    44,    42,
      42,    14,    44,   111,    16,    42,    78,    33,    35,    36,
      37,    40,    49,    78,    79,    81,   112,   113,    45,    33,
      93,    82,    86,    33,   113,   112,    34,    27,    28,    30,
      31,    32,    33,    67,    68,    69,    70,    71,    72,    74,
      40,    44,   112,   112,     5,    48,    49,    50,    51,    52,
      39,    44,    43,    46,    10,    76,    77,    86,    45,     6,
      33,    33,    29,    29,    33,    33,   111,    43,    68,   112,
     114,   112,    41,    33,   112,   112,   112,   112,   112,   112,
      33,    42,    45,    33,    78,    43,    77,    33,   115,    40,
      46,   111,   111,   111,    33,    41,    46,    45,    38,    40,
     114,    39,    40,    42,    39,    46,    33,    33,   112,    44,
     112,   114,    43,    42,     3,    11,   109,   110,    12,    17,
      18,    19,    20,    21,    22,    23,    25,    26,    83,    84,
      87,    88,    90,    94,    95,    96,    97,    99,   101,   102,
     104,   112,    33,    41,   112,    41,    42,    91,    78,    78,
      41,    46,    78,    78,    78,    80,    80,    34,   103,   103,
     103,   111,    40,    42,    40,    84,    87,    43,    88,    39,
     111,    45,    35,    36,    92,    43,    46,    42,   110,    16,
      40,   111,   111,    42,    42,    42,   103,    98,   112,   100,
     111,    33,    43,   112,     6,    43,    46,    42,    83,    84,
     113,   114,   103,   103,    84,    84,    33,    85,    89,    42,
      41,    59,    60,    53,    54,    55,    56,    57,    58,    24,
      43,   111,    42,    41,    33,    73,    35,    36,    92,    84,
      43,    41,    42,    42,    43,    43,   111,    43,    89,    85,
      42,    98,    98,   112,   112,   112,   112,   112,   112,    42,
      42,    84,    42,    40,   111,    46,    43,    43,    84,    84,
       7,    15,    16,   105,    43,    84,    84,    84,    43,   100,
      33,    33,    43,    43,     8,    33,    33,    14,   106,    43,
      43,    43,    24,    43,    41,    40,   111,   111,    33,   107,
     108,    24,    43,    42,   111,    33,   105,    40,    46,    42,
      99,    84,    41,   106,   112,   108,    84,    43,   111,    41,
      43,    43
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    62,    63,    63,    64,    65,    65,    65,    65,    66,
      67,    67,    68,    68,    68,    68,    68,    69,    69,    70,
      70,    71,    72,    72,    73,    73,    73,    73,    74,    74,
      75,    76,    76,    77,    77,    77,    77,    78,    78,    79,
      79,    80,    80,    81,    82,    82,    83,    83,    84,    84,
      85,    85,    86,    86,    86,    87,    87,    87,    88,    88,
      88,    89,    89,    90,    90,    90,    90,    90,    90,    90,
      91,    91,    92,    92,    92,    92,    93,    93,    94,    95,
      95,    96,    96,    97,    97,    98,    98,    98,    98,    98,
      98,    98,    98,    99,    99,    99,   100,   100,   101,   101,
     102,   102,   103,   103,   104,   104,   104,   104,   104,   104,
     104,   104,   104,   105,   105,   105,   106,   106,   107,   107,
     108,   108,   109,   109,   110,   110,   111,   112,   112,   112,
     112,   112,   112,   112,   112,   112,   112,   112,   112,   112,
     113,   114,   114,   114,   115,   115
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     2,     2,     5,
       1,     2,     1,     1,     1,     1,     1,     2,     3,     3,
       2,     4,     8,     6,     2,     5,     4,     7,     3,     3,
       6,     1,     2,     5,     6,     8,     9,     1,     3,     1,
       1,     0,     1,     1,     1,     2,     1,     2,     1,     2,
       1,     2,     1,     1,     1,     1,     1,     1,     5,     6,
       1,     4,     7,     4,     8,     6,     8,     9,    10,     5,
       3,     5,     1,     1,     3,     3,     1,     3,     4,     4,
       3,     4,     6,     2,     5,     3,     3,     3,     3,     3,
       3,     3,     3,     7,    11,     9,     4,     5,     4,     8,
       7,    11,     0,     1,     1,     1,     5,     5,     7,     7,
       1,     1,     1,     0,     2,     2,     0,     2,     1,     3,
       1,     4,     1,     3,     2,     2,     3,     1,     1,     1,
       4,     7,     3,     3,     3,     3,     3,     3,     2,     1,
       4,     0,     1,     3,     3,     5
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static unsigned
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  unsigned res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
 }

#  define YY_LOCATION_PRINT(File, Loc)          \
  yy_location_print_ (File, &(Loc))

# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value, Location); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  YYUSE (yylocationp);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                       , &(yylsp[(yyi + 1) - (yynrhs)])                       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.
       'yyls': related to locations.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[3];

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yylsp = yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  yylsp[0] = yylloc;
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yyls1, yysize * sizeof (*yylsp),
                    &yystacksize);

        yyls = yyls1;
        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location.  */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
#line 163 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    globalmach = (yyvsp[0].mach);
}
#line 1763 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 3:
#line 167 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    globalapp = (yyvsp[0].app);
}
#line 1771 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 4:
#line 174 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.mach) = new ASTMachModel((yyvsp[0].machcontent).first, (yyvsp[0].machcontent).second);
    (yyvsp[0].machcontent).first.release();
    (yyvsp[0].machcontent).second.release();
}
#line 1781 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 5:
#line 183 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    // allocate/clear both vectors, because we don't know
    // if the next one might be a component
    (yyval.machcontent).first.clear();
    (yyval.machcontent).second.clear();
    (yyval.machcontent).first.push_back((yyvsp[0].stmt));
}
#line 1793 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 6:
#line 191 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    // allocate/clear both vectors, because we don't know
    // if the next one might be a globalstmt
    (yyval.machcontent).first.clear();
    (yyval.machcontent).second.clear();
    (yyval.machcontent).second.push_back((yyvsp[0].component));
}
#line 1805 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 7:
#line 199 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.machcontent) = (yyvsp[-1].machcontent);
    (yyval.machcontent).first.push_back((yyvsp[0].stmt));
}
#line 1814 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 8:
#line 204 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.machcontent) = (yyvsp[-1].machcontent);
    (yyval.machcontent).second.push_back((yyvsp[0].component));
}
#line 1823 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 9:
#line 212 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.component) = new ASTMachComponent((yyvsp[-4].str), (yyvsp[-3].str), (yyvsp[-1].astnodelist));
    (yyvsp[-1].astnodelist).release();
}
#line 1832 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 10:
#line 219 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.astnodelist).clear(); (yyval.astnodelist).push_back((yyvsp[0].astnode)); }
#line 1838 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 11:
#line 220 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.astnodelist)=(yyvsp[-1].astnodelist); (yyval.astnodelist).push_back((yyvsp[0].astnode)); }
#line 1844 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 12:
#line 224 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.astnode) = (yyvsp[0].property); }
#line 1850 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 13:
#line 225 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.astnode) = (yyvsp[0].resource); }
#line 1856 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 14:
#line 226 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.astnode) = (yyvsp[0].power); }
#line 1862 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 15:
#line 227 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.astnode) = (yyvsp[0].conflict); }
#line 1868 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 16:
#line 228 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.astnode) = (yyvsp[0].subcomponent); }
#line 1874 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 17:
#line 232 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.subcomponent) = new ASTSubComponent((yyvsp[-1].str), (yyvsp[0].str), NULL); }
#line 1880 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 18:
#line 233 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.subcomponent) = new ASTSubComponent((yyvsp[-2].str), (yyvsp[0].str), (yyvsp[-1].expr)); }
#line 1886 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 19:
#line 238 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.property) = new ASTMachProperty((yyvsp[-1].str), (yyvsp[0].expr)); }
#line 1892 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 20:
#line 239 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.property) = new ASTMachProperty((yyvsp[0].str), NULL); }
#line 1898 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 21:
#line 244 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.conflict) = new ASTResourceConflict((yyvsp[-2].str), (yyvsp[0].str));
}
#line 1906 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 22:
#line 251 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.resource) = new ASTMachResource((yyvsp[-6].str), (yyvsp[-4].str), (yyvsp[-2].expr), (yyvsp[0].traitdeflist));
    (yyvsp[0].traitdeflist).release();
}
#line 1915 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 23:
#line 256 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    ParseVector<ASTTraitDefinition*> empty;
    empty.clear();
    (yyval.resource) = new ASTMachResource((yyvsp[-4].str), (yyvsp[-2].str), (yyvsp[0].expr), empty);
    empty.release();
}
#line 1926 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 24:
#line 266 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.traitdeflist).clear();
    (yyval.traitdeflist).push_back(new ASTTraitDefinition((yyvsp[-1].str), (yyvsp[0].expr)));
}
#line 1935 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 25:
#line 271 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.traitdeflist).clear();
    (yyval.traitdeflist).push_back(new ASTTraitDefinition((yyvsp[-4].str), (yyvsp[-2].str), (yyvsp[0].expr)));
}
#line 1944 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 26:
#line 276 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.traitdeflist) = (yyvsp[-3].traitdeflist);
    (yyval.traitdeflist).push_back(new ASTTraitDefinition((yyvsp[-1].str), (yyvsp[0].expr)));
}
#line 1953 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 27:
#line 281 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.traitdeflist) = (yyvsp[-6].traitdeflist);
    (yyval.traitdeflist).push_back(new ASTTraitDefinition((yyvsp[-4].str), (yyvsp[-2].str), (yyvsp[0].expr)));
}
#line 1962 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 28:
#line 288 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.power) = new ASTMachPower((yyvsp[0].expr), NULL); }
#line 1968 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 29:
#line 289 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.power) = new ASTMachPower(NULL, (yyvsp[0].expr)); }
#line 1974 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 30:
#line 299 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.app) = new ASTAppModel((yyvsp[-4].str), (yyvsp[-2].stmtlist), (yyvsp[-1].kernellist));
    (yyvsp[-2].stmtlist).release();
    (yyvsp[-1].kernellist).release();
}
#line 1984 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 31:
#line 307 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.kernellist).clear(); (yyval.kernellist).push_back((yyvsp[0].kernel)); }
#line 1990 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 32:
#line 308 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.kernellist) = (yyvsp[-1].kernellist); (yyval.kernellist).push_back((yyvsp[0].kernel)); }
#line 1996 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 33:
#line 313 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    ParseVector<ASTKernelArgDecl*> empty1;
    empty1.clear();
    ParseVector<ASTStatement*> empty2;
    empty2.clear();
    (yyval.kernel) = new ASTKernel((yyvsp[-3].ident), (yyvsp[-1].controlstmtlist), empty1, empty2);
    (yyvsp[-1].controlstmtlist).release();
    empty1.release();
    empty2.release();
}
#line 2011 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 34:
#line 324 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    ParseVector<ASTKernelArgDecl*> empty;
    empty.clear();
    (yyval.kernel) = new ASTKernel((yyvsp[-4].ident), (yyvsp[-1].controlstmtlist), empty, (yyvsp[-2].stmtlist));
    (yyvsp[-1].controlstmtlist).release();
    (yyvsp[-2].stmtlist).release();
    empty.release();
}
#line 2024 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 35:
#line 333 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    ParseVector<ASTStatement*> empty;
    empty.clear();
    (yyval.kernel) = new ASTKernel((yyvsp[-6].ident), (yyvsp[-1].controlstmtlist), (yyvsp[-4].argdecllist), empty);
    (yyvsp[-4].argdecllist).release();
    (yyvsp[-1].controlstmtlist).release();
}
#line 2036 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 36:
#line 341 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.kernel) = new ASTKernel((yyvsp[-7].ident), (yyvsp[-1].controlstmtlist), (yyvsp[-5].argdecllist), (yyvsp[-2].stmtlist));
    (yyvsp[-5].argdecllist).release();
    (yyvsp[-1].controlstmtlist).release();
    (yyvsp[-2].stmtlist).release();
}
#line 2047 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 37:
#line 349 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.ident) = new Identifier((yyvsp[0].str)); }
#line 2053 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 38:
#line 350 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.ident) = (yyvsp[0].ident); (yyval.ident)->Prefix((yyvsp[-2].str)); }
#line 2059 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 39:
#line 353 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Integer(atol((yyvsp[0].str).c_str())); }
#line 2065 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 40:
#line 354 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Real(atof((yyvsp[0].str).c_str())); }
#line 2071 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 41:
#line 358 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.ident) = NULL; }
#line 2077 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 42:
#line 359 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.ident) = (yyvsp[0].ident); }
#line 2083 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 43:
#line 363 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    double v = 1;
    if ((yyvsp[0].str) == "nano")  v = 1e-9;
    if ((yyvsp[0].str) == "micro") v = 1e-6;
    if ((yyvsp[0].str) == "milli") v = 1e-3;
    if ((yyvsp[0].str) == "kilo")  v = 1e3;
    if ((yyvsp[0].str) == "mega")  v = 1e6;
    if ((yyvsp[0].str) == "giga")  v = 1e9;
    if ((yyvsp[0].str) == "tera")  v = 1e12;
    if ((yyvsp[0].str) == "peta")  v = 1e15;
    if ((yyvsp[0].str) == "exa")   v = 1e18;
    (yyval.expr) = new Real(v);
}
#line 2101 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 44:
#line 383 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmtlist).clear(); (yyval.stmtlist).push_back((yyvsp[0].stmt)); }
#line 2107 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 45:
#line 384 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmtlist)=(yyvsp[-1].stmtlist); (yyval.stmtlist).push_back((yyvsp[0].stmt)); }
#line 2113 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 46:
#line 388 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmtlist).clear(); (yyval.stmtlist).push_back((yyvsp[0].stmt)); }
#line 2119 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 47:
#line 389 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmtlist)=(yyvsp[-1].stmtlist); (yyval.stmtlist).push_back((yyvsp[0].stmt)); }
#line 2125 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 48:
#line 393 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.controlstmtlist).clear(); (yyval.controlstmtlist).push_back((yyvsp[0].controlstmt)); }
#line 2131 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 49:
#line 394 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.controlstmtlist)=(yyvsp[-1].controlstmtlist); (yyval.controlstmtlist).push_back((yyvsp[0].controlstmt)); }
#line 2137 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 50:
#line 398 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.execstmtlist).clear(); (yyval.execstmtlist).push_back((yyvsp[0].execstmt)); }
#line 2143 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 51:
#line 399 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.execstmtlist)=(yyvsp[-1].execstmtlist); (yyval.execstmtlist).push_back((yyvsp[0].execstmt)); }
#line 2149 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 52:
#line 407 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2155 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 53:
#line 408 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2161 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 54:
#line 409 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2167 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 55:
#line 413 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2173 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 56:
#line 414 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2179 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 57:
#line 415 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.stmt) = (yyvsp[0].stmt); }
#line 2185 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 58:
#line 420 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTExecutionBlock((yyvsp[-3].str), NULL, (yyvsp[-1].execstmtlist));
    (yyvsp[-1].execstmtlist).release();
}
#line 2194 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 59:
#line 425 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTExecutionBlock((yyvsp[-3].str), (yyvsp[-4].expr), (yyvsp[-1].execstmtlist));
    (yyvsp[-1].execstmtlist).release();
}
#line 2203 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 60:
#line 429 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.controlstmt) =  (yyvsp[0].controlstmt); }
#line 2209 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 61:
#line 439 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.execstmt) = new ASTRequiresStatement((yyvsp[-3].str), (yyvsp[-2].expr), NULL, (yyvsp[-1].str), (yyvsp[0].traitlist));
    (yyvsp[0].traitlist).release();
}
#line 2218 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 62:
#line 444 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.execstmt) = new ASTRequiresStatement((yyvsp[-6].str), (yyvsp[-5].expr), (yyvsp[-2].expr), (yyvsp[-1].str), (yyvsp[0].traitlist));
    (yyvsp[0].traitlist).release();
}
#line 2227 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 63:
#line 455 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTAssignStatement((yyvsp[-2].ident),(yyvsp[0].expr));
}
#line 2235 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 64:
#line 459 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTAssignRangeStatement((yyvsp[-6].ident), (yyvsp[-4].expr), (yyvsp[-2].expr), (yyvsp[0].expr));
}
#line 2243 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 65:
#line 463 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTAssignEnumStatement((yyvsp[-4].ident), (yyvsp[-2].expr), (yyvsp[0].str));
}
#line 2251 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 66:
#line 467 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTAssignStatement((yyvsp[-6].ident), new Array((yyvsp[-1].exprlist)));
    (yyvsp[-1].exprlist).release();
}
#line 2260 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 67:
#line 472 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTAssignStatement((yyvsp[-7].ident), new FunctionCall((yyvsp[-3].str),(yyvsp[-1].exprlist)));
    (yyvsp[-1].exprlist).release();
}
#line 2269 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 68:
#line 477 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTAssignStatement((yyvsp[-8].ident), new Table((yyvsp[-1].valuelist)));
    (yyvsp[-1].valuelist).release();
}
#line 2278 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 69:
#line 487 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTEnumDeclaration((yyvsp[-3].str), (yyvsp[-1].stringlist));
}
#line 2286 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 70:
#line 494 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.valuelist).clear();
    (yyval.valuelist).append((yyvsp[-1].valuelist));
    (yyvsp[-1].valuelist).release();
}
#line 2296 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 71:
#line 500 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.valuelist) = (yyvsp[-4].valuelist);

    // add two sentinels so we know where to break
    (yyval.valuelist).push_back(1e37);
    (yyval.valuelist).push_back(-1e37);

    (yyval.valuelist).append((yyvsp[-1].valuelist));
    (yyvsp[-1].valuelist).release();
}
#line 2311 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 72:
#line 514 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.valuelist).clear();
    (yyval.valuelist).push_back(atol((yyvsp[0].str).c_str()));
}
#line 2320 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 73:
#line 519 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.valuelist).clear();
    (yyval.valuelist).push_back(atof((yyvsp[0].str).c_str()));
}
#line 2329 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 74:
#line 524 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.valuelist) = (yyvsp[-2].valuelist);
    (yyval.valuelist).push_back(atol((yyvsp[0].str).c_str()));
}
#line 2338 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 75:
#line 529 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.valuelist) = (yyvsp[-2].valuelist);
    (yyval.valuelist).push_back(atof((yyvsp[0].str).c_str()));
}
#line 2347 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 76:
#line 537 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stringlist).clear();
    (yyval.stringlist).push_back((yyvsp[0].str));
}
#line 2356 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 77:
#line 542 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stringlist) = (yyvsp[-2].stringlist);
    (yyval.stringlist).push_back((yyvsp[0].str));
}
#line 2365 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 78:
#line 550 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTSampleStatement((yyvsp[-2].ident),(yyvsp[0].call));
}
#line 2373 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 79:
#line 557 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTDataStatement((yyvsp[-2].str),NULL,(yyvsp[0].call));
}
#line 2381 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 80:
#line 561 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTDataStatement((yyvsp[-1].str),(yyvsp[0].expr),NULL);
}
#line 2389 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 81:
#line 568 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTImportStatement((yyvsp[-2].str),(yyvsp[0].str));
}
#line 2397 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 82:
#line 572 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.stmt) = new ASTImportStatement((yyvsp[-4].str),(yyvsp[-2].str),(yyvsp[0].assignlist));
    (yyvsp[0].assignlist).release();
}
#line 2406 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 83:
#line 580 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlKernelCallStatement((yyvsp[0].ident));
}
#line 2414 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 84:
#line 584 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlKernelCallStatement((yyvsp[-3].ident), (yyvsp[-1].exprlist));
    (yyvsp[-1].exprlist).release();
}
#line 2423 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 85:
#line 591 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Comparison("and",(yyvsp[-2].expr),(yyvsp[0].expr)); }
#line 2429 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 86:
#line 592 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Comparison("or",(yyvsp[-2].expr),(yyvsp[0].expr)); }
#line 2435 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 87:
#line 593 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Comparison("==",(yyvsp[-2].expr),(yyvsp[0].expr)); }
#line 2441 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 88:
#line 594 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Comparison("!=",(yyvsp[-2].expr),(yyvsp[0].expr)); }
#line 2447 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 89:
#line 595 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Comparison("<", (yyvsp[-2].expr),(yyvsp[0].expr)); }
#line 2453 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 90:
#line 596 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Comparison(">", (yyvsp[-2].expr),(yyvsp[0].expr)); }
#line 2459 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 91:
#line 597 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Comparison("<=",(yyvsp[-2].expr),(yyvsp[0].expr)); }
#line 2465 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 92:
#line 598 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new Comparison(">=",(yyvsp[-2].expr),(yyvsp[0].expr)); }
#line 2471 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 93:
#line 603 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlIfThenStatement((yyvsp[-4].expr), new ASTControlSequentialStatement("", (yyvsp[-1].controlstmtlist)), NULL);
    (yyvsp[-1].controlstmtlist).release();
}
#line 2480 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 94:
#line 608 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlIfThenStatement((yyvsp[-8].expr), new ASTControlSequentialStatement("", (yyvsp[-5].controlstmtlist)), new ASTControlSequentialStatement("", (yyvsp[-1].controlstmtlist)));
    (yyvsp[-5].controlstmtlist).release();
    (yyvsp[-1].controlstmtlist).release();
}
#line 2490 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 95:
#line 614 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlIfThenStatement((yyvsp[-6].expr), new ASTControlSequentialStatement("", (yyvsp[-3].controlstmtlist)), (yyvsp[0].controlstmt));
    (yyvsp[-3].controlstmtlist).release();
}
#line 2499 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 96:
#line 622 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.caselist).clear();
    (yyval.caselist).push_back(new ASTCaseItem((yyvsp[-3].expr), new ASTControlSequentialStatement("", (yyvsp[-1].controlstmtlist))));
    (yyvsp[-1].controlstmtlist).release();
}
#line 2509 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 97:
#line 628 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.caselist) = (yyvsp[-4].caselist);
    (yyval.caselist).push_back(new ASTCaseItem((yyvsp[-3].expr), new ASTControlSequentialStatement("", (yyvsp[-1].controlstmtlist))));
    (yyvsp[-1].controlstmtlist).release();
}
#line 2519 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 98:
#line 637 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlProbabilityStatement((yyvsp[-1].caselist), NULL);
    (yyvsp[-1].caselist).release();
}
#line 2528 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 99:
#line 642 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlProbabilityStatement((yyvsp[-5].caselist), new ASTControlSequentialStatement("", (yyvsp[-2].controlstmtlist)));
    (yyvsp[-5].caselist).release();
}
#line 2537 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 100:
#line 650 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlSwitchStatement(new Identifier((yyvsp[-4].str)), (yyvsp[-1].caselist), NULL);
    (yyvsp[-1].caselist).release();
}
#line 2546 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 101:
#line 655 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlSwitchStatement(new Identifier((yyvsp[-8].str)), (yyvsp[-5].caselist), new ASTControlSequentialStatement("", (yyvsp[-2].controlstmtlist)));
    (yyvsp[-5].caselist).release();
}
#line 2555 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 102:
#line 662 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.str) = ""; }
#line 2561 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 103:
#line 663 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.str) = (yyvsp[0].str); }
#line 2567 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 104:
#line 668 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = (yyvsp[0].controlstmt);
}
#line 2575 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 105:
#line 672 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) =  (yyvsp[0].controlstmt);
}
#line 2583 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 106:
#line 676 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlParallelStatement((yyvsp[-3].str), (yyvsp[-1].controlstmtlist));
    (yyvsp[-1].controlstmtlist).release();
}
#line 2592 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 107:
#line 681 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = new ASTControlSequentialStatement((yyvsp[-3].str), (yyvsp[-1].controlstmtlist));
    (yyvsp[-1].controlstmtlist).release();
}
#line 2601 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 108:
#line 686 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    if ((yyvsp[-1].controlstmtlist).size() == 0)
        ; // can't happen in current grammar; if we change grammar to allow it, then error here
    else if ((yyvsp[-1].controlstmtlist).size() == 1)
        (yyval.controlstmt) = new ASTControlIterateStatement((yyvsp[-3].str), (yyvsp[-5].ident), (yyvsp[-4].expr), (yyvsp[-1].controlstmtlist)[0]);
    else // size > 1
        (yyval.controlstmt) = new ASTControlIterateStatement((yyvsp[-3].str), (yyvsp[-5].ident), (yyvsp[-4].expr), new ASTControlSequentialStatement("", (yyvsp[-1].controlstmtlist)));
    (yyvsp[-1].controlstmtlist).release();
}
#line 2615 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 109:
#line 696 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    if ((yyvsp[-1].controlstmtlist).size() == 0)
        ; // can't happen in current grammar; if we change grammar to allow it, then error here
    else if ((yyvsp[-1].controlstmtlist).size() == 1)
        (yyval.controlstmt) = new ASTControlMapStatement((yyvsp[-3].str), (yyvsp[-5].ident), (yyvsp[-4].expr), (yyvsp[-1].controlstmtlist)[0]);
    else // size > 1
        (yyval.controlstmt) = new ASTControlMapStatement((yyvsp[-3].str), (yyvsp[-5].ident), (yyvsp[-4].expr), new ASTControlSequentialStatement("", (yyvsp[-1].controlstmtlist)));
    (yyvsp[-1].controlstmtlist).release();
}
#line 2629 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 110:
#line 706 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = (yyvsp[0].controlstmt);
}
#line 2637 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 111:
#line 710 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = (yyvsp[0].controlstmt);
}
#line 2645 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 112:
#line 714 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.controlstmt) = (yyvsp[0].controlstmt);
}
#line 2653 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 113:
#line 720 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.str) = ""; }
#line 2659 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 114:
#line 721 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.str) = (yyvsp[0].str); }
#line 2665 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 115:
#line 722 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.str) = (yyvsp[0].str); }
#line 2671 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 116:
#line 726 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.traitlist).clear(); }
#line 2677 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 117:
#line 727 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.traitlist) = (yyvsp[0].traitlist); }
#line 2683 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 118:
#line 731 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.traitlist).clear(); (yyval.traitlist).push_back((yyvsp[0].trait)); }
#line 2689 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 119:
#line 732 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.traitlist)=(yyvsp[-2].traitlist); (yyval.traitlist).push_back((yyvsp[0].trait)); }
#line 2695 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 120:
#line 736 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.trait) = new ASTTrait((yyvsp[0].str), NULL); }
#line 2701 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 121:
#line 737 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.trait) = new ASTTrait((yyvsp[-3].str), (yyvsp[-1].expr)); }
#line 2707 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 122:
#line 741 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.argdecllist).clear(); (yyval.argdecllist).push_back((yyvsp[0].argdecl)); }
#line 2713 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 123:
#line 742 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.argdecllist)=(yyvsp[-2].argdecllist); (yyval.argdecllist).push_back((yyvsp[0].argdecl)); }
#line 2719 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 124:
#line 746 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.argdecl) = new ASTKernelArgDecl("data", (yyvsp[0].ident)); }
#line 2725 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 125:
#line 747 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.argdecl) = new ASTKernelArgDecl("param", (yyvsp[0].ident)); }
#line 2731 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 126:
#line 751 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[-1].expr); }
#line 2737 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 127:
#line 754 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[0].expr); }
#line 2743 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 128:
#line 755 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[0].expr); }
#line 2749 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 129:
#line 756 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[0].ident); }
#line 2755 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 130:
#line 757 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new ArrayDereference(new Identifier((yyvsp[-3].str)), (yyvsp[-1].expr)); }
#line 2761 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 131:
#line 758 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new TableDereference(new Identifier((yyvsp[-6].str)), (yyvsp[-4].expr), (yyvsp[-1].expr)); }
#line 2767 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 132:
#line 759 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[-1].expr); }
#line 2773 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 133:
#line 760 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExpr("+", (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 2779 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 134:
#line 761 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExpr("-", (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 2785 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 135:
#line 762 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExpr("*", (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 2791 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 136:
#line 763 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExpr("/", (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 2797 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 137:
#line 764 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new BinaryExpr("^", (yyvsp[-2].expr), (yyvsp[0].expr)); }
#line 2803 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 138:
#line 765 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = new UnaryExpr("-", (yyvsp[0].expr)); }
#line 2809 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 139:
#line 766 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.expr) = (yyvsp[0].call); }
#line 2815 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 140:
#line 771 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.call) = new FunctionCall((yyvsp[-3].str), (yyvsp[-1].exprlist));
    (yyvsp[-1].exprlist).release();
}
#line 2824 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 141:
#line 778 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.exprlist).clear(); }
#line 2830 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 142:
#line 779 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.exprlist).clear(); (yyval.exprlist).push_back((yyvsp[0].expr)); }
#line 2836 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 143:
#line 780 "parser/AspenParser.y" /* yacc.c:1646  */
    { (yyval.exprlist)=(yyvsp[-2].exprlist); (yyval.exprlist).push_back((yyvsp[0].expr)); }
#line 2842 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 144:
#line 785 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    ///\todo: decide if namedarguments needs ident or TIDENT here.
    (yyval.assignlist).clear();
    (yyval.assignlist).push_back(new ASTAssignStatement(new Identifier((yyvsp[-2].str)),(yyvsp[0].expr)));
}
#line 2852 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;

  case 145:
#line 791 "parser/AspenParser.y" /* yacc.c:1646  */
    {
    (yyval.assignlist)=(yyvsp[-4].assignlist);
    (yyval.assignlist).push_back(new ASTAssignStatement(new Identifier((yyvsp[-2].str)),(yyvsp[0].expr)));
}
#line 2861 "parser/AspenParser.cpp" /* yacc.c:1646  */
    break;


#line 2865 "parser/AspenParser.cpp" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }

  yyerror_range[1] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  yyerror_range[1] = yylsp[1-yylen];
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, yyerror_range, 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 797 "parser/AspenParser.y" /* yacc.c:1906  */


void yyerror(const char *s)
{
    cerr << "ERROR: " << s << " text=\""<<yytext<<"\" line="<<yylineno<< endl;
    cerr << "(Detailed location:"
        << " file='" << yylloc.filename << "'"
        << ", filepos=" << yylloc.first_filepos << " to " << yylloc.last_filepos
        << ", lines=" << yylloc.first_line << " to " << yylloc.last_line
        << ", columns=" << yylloc.first_column << " to " << yylloc.last_column
        << ")" << endl;
}
