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
// sumfloat: 单精度浮点数组求和的函数
//////////////////////////////////////////////////

// 单精度浮点数组求和_基本版.
//
// result: 返回数组求和结果.
// pbuf: 数组的首地址.
// cntbuf: 数组长度.
float sumfloat_base(const float* pbuf, size_t cntbuf)
{
	float s = 0;	// 求和变量.
	size_t i;
	for(i=0; i<cntbuf; ++i)
	{
		s += pbuf[i];
	}
	return s;
}

#ifdef INTRIN_SSE
// 单精度浮点数组求和_SSE版.
float sumfloat_sse(const float* pbuf, size_t cntbuf)
{
	float s = 0;	// 求和变量.
	size_t i;
	size_t nBlockWidth = 4;	// 块宽. SSE寄存器能一次处理4个float.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m128 xfsSum = _mm_setzero_ps();	// 求和变量。[SSE] 赋初值0
	__m128 xfsLoad;	// 加载.
	const float* p = pbuf;	// SSE批量处理时所用的指针.
	const float* q;	// 将SSE变量上的多个数值合并时所用指针.

	// SSE批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		xfsLoad = _mm_load_ps(p);	// [SSE] 加载
		xfsSum = _mm_add_ps(xfsSum, xfsLoad);	// [SSE] 单精浮点紧缩加法
		p += nBlockWidth;
	}
	// 合并.
	q = (const float*)&xfsSum;
	s = q[0] + q[1] + q[2] + q[3];

	// 处理剩下的.
	for(i=0; i<cntRem; ++i)
	{
		s += p[i];
	}

	return s;
}

// 单精度浮点数组求和_SSE四路循环展开版.
float sumfloat_sse_4loop(const float* pbuf, size_t cntbuf)
{
	float s = 0;	// 返回值.
	size_t i;
	size_t nBlockWidth = 4*4;	// 块宽. SSE寄存器能一次处理4个float，然后循环展开4次.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m128 xfsSum = _mm_setzero_ps();	// 求和变量。[SSE] 赋初值0
	__m128 xfsSum1 = _mm_setzero_ps();
	__m128 xfsSum2 = _mm_setzero_ps();
	__m128 xfsSum3 = _mm_setzero_ps();
	__m128 xfsLoad;	// 加载.
	__m128 xfsLoad1;
	__m128 xfsLoad2;
	__m128 xfsLoad3;
	const float* p = pbuf;	// SSE批量处理时所用的指针.
	const float* q;	// 将SSE变量上的多个数值合并时所用指针.

	// SSE批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		xfsLoad = _mm_load_ps(p);	// [SSE] 加载.
		xfsLoad1 = _mm_load_ps(p+4);
		xfsLoad2 = _mm_load_ps(p+8);
		xfsLoad3 = _mm_load_ps(p+12);
		xfsSum = _mm_add_ps(xfsSum, xfsLoad);	// [SSE] 单精浮点紧缩加法
		xfsSum1 = _mm_add_ps(xfsSum1, xfsLoad1);
		xfsSum2 = _mm_add_ps(xfsSum2, xfsLoad2);
		xfsSum3 = _mm_add_ps(xfsSum3, xfsLoad3);
		p += nBlockWidth;
	}
	// 合并.
	xfsSum = _mm_add_ps(xfsSum, xfsSum1);	// 两两合并(0~1).
	xfsSum2 = _mm_add_ps(xfsSum2, xfsSum3);	// 两两合并(2~3).
	xfsSum = _mm_add_ps(xfsSum, xfsSum2);	// 两两合并(0~3).
	q = (const float*)&xfsSum;
	s = q[0] + q[1] + q[2] + q[3];

	// 处理剩下的.
	for(i=0; i<cntRem; ++i)
	{
		s += p[i];
	}

	return s;
}
#endif	// #ifdef INTRIN_SSE


#ifdef INTRIN_AVX
// 单精度浮点数组求和_AVX版.
float sumfloat_avx(const float* pbuf, size_t cntbuf)
{
	float s = 0;	// 求和变量.
	size_t i;
	size_t nBlockWidth = 8;	// 块宽. AVX寄存器能一次处理8个float.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m256 yfsSum = _mm256_setzero_ps();	// 求和变量。[AVX] 赋初值0
	__m256 yfsLoad;	// 加载.
	const float* p = pbuf;	// AVX批量处理时所用的指针.
	const float* q;	// 将AVX变量上的多个数值合并时所用指针.

	// AVX批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		yfsLoad = _mm256_load_ps(p);	// [AVX] 加载
		yfsSum = _mm256_add_ps(yfsSum, yfsLoad);	// [AVX] 单精浮点紧缩加法
		p += nBlockWidth;
	}
	// 合并.
	q = (const float*)&yfsSum;
	s = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

	// 处理剩下的.
	for(i=0; i<cntRem; ++i)
	{
		s += p[i];
	}

	return s;
}

// 单精度浮点数组求和_AVX四路循环展开版.
float sumfloat_avx_4loop(const float* pbuf, size_t cntbuf)
{
	float s = 0;	// 求和变量.
	size_t i;
	size_t nBlockWidth = 8*4;	// 块宽. AVX寄存器能一次处理8个float，然后循环展开4次.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m256 yfsSum = _mm256_setzero_ps();	// 求和变量。[AVX] 赋初值0
	__m256 yfsSum1 = _mm256_setzero_ps();
	__m256 yfsSum2 = _mm256_setzero_ps();
	__m256 yfsSum3 = _mm256_setzero_ps();
	__m256 yfsLoad;	// 加载.
	__m256 yfsLoad1;
	__m256 yfsLoad2;
	__m256 yfsLoad3;
	const float* p = pbuf;	// AVX批量处理时所用的指针.
	const float* q;	// 将AVX变量上的多个数值合并时所用指针.

	// AVX批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		yfsLoad = _mm256_load_ps(p);	// [AVX] 加载.
		yfsLoad1 = _mm256_load_ps(p+8);
		yfsLoad2 = _mm256_load_ps(p+16);
		yfsLoad3 = _mm256_load_ps(p+24);
		yfsSum = _mm256_add_ps(yfsSum, yfsLoad);	// [AVX] 单精浮点紧缩加法
		yfsSum1 = _mm256_add_ps(yfsSum1, yfsLoad1);
		yfsSum2 = _mm256_add_ps(yfsSum2, yfsLoad2);
		yfsSum3 = _mm256_add_ps(yfsSum3, yfsLoad3);
		p += nBlockWidth;
	}
	// 合并.
	yfsSum = _mm256_add_ps(yfsSum, yfsSum1);	// 两两合并(0~1).
	yfsSum2 = _mm256_add_ps(yfsSum2, yfsSum3);	// 两两合并(2~3).
	yfsSum = _mm256_add_ps(yfsSum, yfsSum2);	// 两两合并(0~3).
	q = (const float*)&yfsSum;
	s = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];

	// 处理剩下的.
	for(i=0; i<cntRem; ++i)
	{
		s += p[i];
	}

	return s;
}

#endif	// #ifdef INTRIN_AVX



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


#define BUFSIZE	409600	// = 32KB{L1 Cache} / (2 * sizeof(float))
ATTR_ALIGN(32) float buf[BUFSIZE];

// 测试时的函数类型
typedef float (*TESTPROC)(const float* pbuf, size_t cntbuf);

// 进行测试
void runTest(const char* szname, TESTPROC proc)
{
	const int testloop = 4000;	// 重复运算几次延长时间，避免计时精度问题.
	int i;
	clock_t	tm0, dt;	// 存储时间.
	double mps;	// M/s.
	volatile float n=0;	// 避免内循环被优化.

	tm0 = clock();
	// main
	for(i=1; i<=testloop; ++i)
	{
		n = proc(buf, BUFSIZE);
	}
	dt = clock() - tm0;
	double time_s = (double)dt / CLOCKS_PER_SEC;
	mps = (double)testloop*BUFSIZE*CLOCKS_PER_SEC/(1024.0*1024.0*dt);
	//printf("%s:\t%.0f M/s\t%f s\t sum:%f\n", szname, mps, time_s, n);
	printf("%s:\t\t  io: %.0f mb/s\t  time:%f s sum:%f\n", szname, mps, time_s, n);
}

int main(int argc, char* argv[])
{
	char szBuf[64];
	int i;

	printf("simdsumfloat v1.00 (%dbit)\n", INTRIN_WORDSIZE);
	printf("Compiler: %s\n", COMPILER_NAME);
	cpu_getbrand(szBuf);
	printf("CPU:\t%s\n", szBuf);
	printf("\n");

	// init buf
	srand( (unsigned)time( NULL ) );
	for (i = 0; i < BUFSIZE; i++) buf[i] = (float)(rand() & 0x3f);	// 使用&0x3f是为了让求和后的数值不会超过float类型的有效位数，便于观察结果是否正确.

	// test
	runTest("sumfloat_base", sumfloat_base);	// 单精度浮点数组求和_基本版.
#ifdef INTRIN_SSE
	if (simd_sse_level(NULL) >= SIMD_SSE_1)
	{
		runTest("sumfloat_sse", sumfloat_sse);	// 单精度浮点数组求和_SSE版.
		runTest("sumfloat_sse_4", sumfloat_sse_4loop);	// 单精度浮点数组求和_SSE四路循环展开版.
	}
#endif	// #ifdef INTRIN_SSE
#ifdef INTRIN_AVX
	if (simd_avx_level(NULL) >= SIMD_AVX_1)
	{
		runTest("sumfloat_avx", sumfloat_avx);	// 单精度浮点数组求和_AVX版.
		runTest("sumfloat_avx_4", sumfloat_avx_4loop);	// 单精度浮点数组求和_AVX四路循环展开版.
	}
#endif	// #ifdef INTRIN_AVX

	return 0;
}
