// neater primitive data types
// based on https://github.com/Luke-A-C-Roberts/mystdlib/blob/main/headers/primatives.hpp  

#pragma once

/*********DataTypes*********/
/*   Primative Datatypes   */
//-----------Ints----------//

typedef signed long long i128;
typedef signed long i64;
typedef signed int i32;
typedef signed short i16;
typedef signed char i8;
typedef const signed long long ci128;
typedef const signed long ci64;
typedef const signed int ci32;
typedef const signed short ci16;
typedef const signed char ci8;

//----------UInts----------//

typedef unsigned long long u128;
typedef unsigned long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef const unsigned long long cu128;
typedef const unsigned long cu64;
typedef const unsigned int cu32;
typedef const unsigned short cu16;
typedef const unsigned char cu8;

// typedef unsigned long int size;
// typedef const unsigned long int csize;

//----------Float----------//

typedef long double f128;
typedef double f64;
typedef float f32;
typedef const long double cf128;
typedef const double cf64;
typedef const float cf32;

/*        Characters       */
//---------Strings---------//

typedef int wchar;
typedef char* str;
typedef int* wstr;

typedef const char cchar;
typedef const int cwchar;
typedef const char* cstr;
typedef const int* cwstr;

/*          Other          */
typedef const bool cbool;
typedef void* ptr;
typedef const void* cptr;

/*********DataSizes*********/

