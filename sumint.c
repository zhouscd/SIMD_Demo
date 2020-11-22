#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "zintrin.h"
#include "ccpuid.h"


// Compiler name
#define MACTOSTR(x)	#x
#define MACROVALUESTR(x)	MACTOSTR(x)
#if defined(__ICL)	// Intel C++
#  if defined(__VERSION__)
#    define COMPILER_NAME	"Intel C++ " __VERSION__
#  elif defined(__INTEL_COMPILER_BUILD_DATE)
#    define COMPILER_NAME	"Intel C++ (" MACROVALUESTR(__INTEL_COMPILER_BUILD_DATE) ")"
#  else
#    define COMPILER_NAME	"Intel C++"
#  endif	// #  if defined(__VERSION__)
#elif defined(_MSC_VER)	// Microsoft VC++
#  if defined(_MSC_FULL_VER)
#    define COMPILER_NAME	"Microsoft VC++ (" MACROVALUESTR(_MSC_FULL_VER) ")"
#  elif defined(_MSC_VER)
#    define COMPILER_NAME	"Microsoft VC++ (" MACROVALUESTR(_MSC_VER) ")"
#  else
#    define COMPILER_NAME	"Microsoft VC++"
#  endif	// #  if defined(_MSC_FULL_VER)
#elif defined(__GNUC__)	// GCC
#  if defined(__CYGWIN__)
#    define COMPILER_NAME	"GCC(Cygmin) " __VERSION__
#  elif defined(__MINGW32__)
#    define COMPILER_NAME	"GCC(MinGW) " __VERSION__
#  else
#    define COMPILER_NAME	"GCC " __VERSION__
#  endif	// #  if defined(_MSC_FULL_VER)
#else
#  define COMPILER_NAME	"Unknown Compiler"
#endif	// #if defined(__ICL)	// Intel C++


//////////////////////////////////////////////////
// sumint: 32位整数数组求和的函数
//////////////////////////////////////////////////

// 32位整数数组求和_基本版.
//
// result: 返回数组求和结果.
// pbuf: 数组的首地址.
// cntbuf: 数组长度.
int32_t sumint_base(const int32_t* pbuf, size_t cntbuf)
{
	int32_t s = 0;	// 求和变量.
	size_t i;
	for(i=0; i<cntbuf; ++i)
	{
		s += pbuf[i];
	}
	return s;
}

#ifdef INTRIN_MMX
// 32位整数数组求和_MMX版.
int32_t sumint_mmx(const int32_t* pbuf, size_t cntbuf)
{
	int32_t s = 0;	// 求和变量.
	size_t i;
	size_t nBlockWidth = 2;	// 块宽. MMX寄存器能一次处理2个int32_t.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m64 midSum = _mm_setzero_si64();	// 求和变量。[MMX] PXOR, 赋初值0.
	__m64 midLoad;	// 加载.
	const __m64* p = (const __m64*)pbuf;	// MMX批量处理时所用的指针.
	const int32_t* q;	// 单个数据处理时所用指针.

	// MMX批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		midLoad = *p;	// [MMX] MOVQ. 加载.
		midSum = _mm_add_pi32(midSum, midLoad);	// [MMX] PADDD. 32位整数紧缩环绕加法.
		p ++;
	}
	// 合并.
	q = (const int32_t*)&midSum;
	s = q[0] + q[1];

	// 处理剩下的.
	q = (const int32_t*)p;
	for(i=0; i<cntRem; ++i)
	{
		s += q[i];
	}

	// 清理MMX状态.
	_mm_empty();	// [MMX] EMMS.

	return s;
}

// 32位整数数组求和_MMX四路循环展开版.
int32_t sumint_mmx_4loop(const int32_t* pbuf, size_t cntbuf)
{
	int32_t s = 0;	// 返回值.
	size_t i;
	size_t nBlockWidth = 2*4;	// 块宽. MMX寄存器能一次处理2个int32_t，然后循环展开4次.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m64 midSum = _mm_setzero_si64();	// 求和变量。[MMX] PXOR, 赋初值0.
	__m64 midSum1 = _mm_setzero_si64();
	__m64 midSum2 = _mm_setzero_si64();
	__m64 midSum3 = _mm_setzero_si64();
	__m64 midLoad;	// 加载.
	__m64 midLoad1;
	__m64 midLoad2;
	__m64 midLoad3;
	const __m64* p = (const __m64*)pbuf;	// MMX批量处理时所用的指针.
	const int32_t* q;	// 单个数据处理时所用指针.

	// SSE批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		midLoad = *p;	// [MMX] MOVQ. 加载.
		midLoad1 = *(p+1);
		midLoad2 = *(p+2);
		midLoad3 = *(p+3);
		midSum = _mm_add_pi32(midSum, midLoad);	// [MMX] PADDD. 32位整数紧缩环绕加法.
		midSum1 = _mm_add_pi32(midSum1, midLoad1);
		midSum2 = _mm_add_pi32(midSum2, midLoad2);
		midSum3 = _mm_add_pi32(midSum3, midLoad3);
		p += 4;	// 四路循环展开.
	}
	// 合并.
	midSum = _mm_add_pi32(midSum, midSum1);	// 两两合并(0~1).
	midSum2 = _mm_add_pi32(midSum2, midSum3);	// 两两合并(2~3).
	midSum = _mm_add_pi32(midSum, midSum2);	// 两两合并(0~3).
	q = (const int32_t*)&midSum;
	s = q[0] + q[1];

	// 处理剩下的.
	q = (const int32_t*)p;
	for(i=0; i<cntRem; ++i)
	{
		s += q[i];
	}

	// 清理MMX状态.
	_mm_empty();	// [MMX] EMMS.

	return s;
}
#endif	// #ifdef INTRIN_MMX


#ifdef INTRIN_SSE2
// 32位整数数组求和_SSE版.
int32_t sumint_sse(const int32_t* pbuf, size_t cntbuf)
{
	int32_t s = 0;	// 求和变量.
	size_t i;
	size_t nBlockWidth = 4;	// 块宽. SSE寄存器能一次处理4个int32_t.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m128i xidSum = _mm_setzero_si128();	// 求和变量。[SSE2] PXOR. 赋初值0.
	__m128i xidLoad;	// 加载.
	const __m128i* p = (const __m128i*)pbuf;	// SSE批量处理时所用的指针.
	const int32_t* q;	// 单个数据处理时所用指针.

	// SSE批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		xidLoad = _mm_load_si128(p);	// [SSE2] MOVDQA. 加载.
		xidSum = _mm_add_epi32(xidSum, xidLoad);	// [SSE2] PADDD. 32位整数紧缩环绕加法.
		p ++;
	}
	// 合并.
	q = (const int32_t*)&xidSum;
	s = q[0] + q[1] + q[2] + q[3];

	// 处理剩下的.
	q = (const int32_t*)p;
	for(i=0; i<cntRem; ++i)
	{
		s += q[i];
	}

	return s;
}

// 32位整数数组求和_SSE四路循环展开版.
int32_t sumint_sse_4loop(const int32_t* pbuf, size_t cntbuf)
{
	int32_t s = 0;	// 返回值.
	size_t i;
	size_t nBlockWidth = 4*4;	// 块宽. SSE寄存器能一次处理4个int32_t，然后循环展开4次.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m128i xidSum = _mm_setzero_si128();	// 求和变量。[SSE2] PXOR. 赋初值0.
	__m128i xidSum1 = _mm_setzero_si128();
	__m128i xidSum2 = _mm_setzero_si128();
	__m128i xidSum3 = _mm_setzero_si128();
	__m128i xidLoad;	// 加载.
	__m128i xidLoad1;
	__m128i xidLoad2;
	__m128i xidLoad3;
	const __m128i* p = (const __m128i*)pbuf;	// SSE批量处理时所用的指针.
	const int32_t* q;	// 单个数据处理时所用指针.

	// SSE批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		xidLoad = _mm_load_si128(p);	// [SSE2] MOVDQA. 加载.
		xidLoad1 = _mm_load_si128(p+1);
		xidLoad2 = _mm_load_si128(p+2);
		xidLoad3 = _mm_load_si128(p+3);
		xidSum = _mm_add_epi32(xidSum, xidLoad);	// [SSE2] PADDD. 32位整数紧缩环绕加法.
		xidSum1 = _mm_add_epi32(xidSum1, xidLoad1);
		xidSum2 = _mm_add_epi32(xidSum2, xidLoad2);
		xidSum3 = _mm_add_epi32(xidSum3, xidLoad3);
		p += 4;	// 四路循环展开.
	}
	// 合并.
	xidSum = _mm_add_epi32(xidSum, xidSum1);	// 两两合并(0~1).
	xidSum2 = _mm_add_epi32(xidSum2, xidSum3);	// 两两合并(2~3).
	xidSum = _mm_add_epi32(xidSum, xidSum2);	// 两两合并(0~3).
	q = (const int32_t*)&xidSum;
	s = q[0] + q[1] + q[2] + q[3];

	// 处理剩下的.
	q = (const int32_t*)p;
	for(i=0; i<cntRem; ++i)
	{
		s += q[i];
	}

	return s;
}
#endif	// #ifdef INTRIN_SSE2





//////////////////////////////////////////////////
// main
//////////////////////////////////////////////////

// 变量对齐.
#ifndef ATTR_ALIGN
#  if defined(__GNUC__)	// GCC
#    define ATTR_ALIGN(n)	__attribute__((aligned(n)))
#  else	// 否则使用VC格式.
#    define ATTR_ALIGN(n)	__declspec(align(n))
#  endif
#endif	// #ifndef ATTR_ALIGN


#define BUFSIZE	409600
ATTR_ALIGN(32) int32_t buf[BUFSIZE];

// 测试时的函数类型
typedef int32_t (*TESTPROC)(const int32_t* pbuf, size_t cntbuf);

// 进行测试
void runTest(const char* szname, TESTPROC proc)
{
	const int testloop = 4000;	// 重复运算几次延长时间，避免计时精度问题.
	int j;
	clock_t	tm0, dt;	// 存储时间.
	double mps;	// M/s.
	volatile int32_t n=0;	// 避免内循环被优化.

	tm0 = clock();

	for(j=1; j<=testloop; ++j)	// 重复运算几次延长时间，避免计时开销带来的影响.
	{
		n = proc(buf, BUFSIZE);	// 避免内循环被编译优化消掉.
	}
	dt = clock() - tm0;

	// show
	mps = (double)testloop*BUFSIZE*CLOCKS_PER_SEC/(1024.0*1024.0*dt);
	printf("%s:\t%.0f M/s\t%ld ms //%ld\n", szname, mps,dt, n);
}

int main(int argc, char* argv[])
{
	char szBuf[64];
	int i;

	printf("simdsumint v1.00 (%dbit)\n", INTRIN_WORDSIZE);
	printf("Compiler: %s\n", COMPILER_NAME);
	cpu_getbrand(szBuf);
	printf("CPU:\t%s\n", szBuf);
	printf("\n");

	// init buf
	srand( (unsigned)time( NULL ) );
	for (i = 0; i < BUFSIZE; i++) buf[i] = (int32_t)(rand() & 0x7fff);	// 使用&0x7fff是为了使数值在一定范围内，便于观察结果是否正确.

    printf("%ld\n", buf[2]);
	// test
	runTest("sumint_base", sumint_base);	// 32位整数数组求和_基本版.
#ifdef INTRIN_MMX
	if (simd_mmx(NULL))
	{
		runTest("sumint_mmx", sumint_mmx);	// 32位整数数组求和_MMX版.
		runTest("sumint_mmx_4loop", sumint_mmx_4loop);	// 32位整数数组求和_MMX四路循环展开版.
	}
#endif	// #ifdef INTRIN_MMX
#ifdef INTRIN_SSE2
	if (simd_sse_level(NULL) >= SIMD_SSE_2)
	{
		runTest("sumint_sse", sumint_sse);	// 32位整数数组求和_SSE版.
		runTest("sumint_sse_4loop", sumint_sse_4loop);	// 32位整数数组求和_SSE四路循环展开版.
	}
#endif	// #ifdef INTRIN_SSE2

	return 0;
}