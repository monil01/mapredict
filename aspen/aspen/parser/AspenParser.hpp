/* A Bison parser, made by GNU Bison 3.0.4.  */

/* Bison interface for Yacc-like parsers in C

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
#line 23 "parser/AspenParser.y" /* yacc.c:1909  */

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


#line 85 "parser/AspenParser.hpp" /* yacc.c:1909  */

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
