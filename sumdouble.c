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
// sumdouble: 双精度浮点数组求和的函数
//////////////////////////////////////////////////

// 双精度浮点数组求和_基本版.
//
// result: 返回数组求和结果.
// pbuf: 数组的首地址.
// cntbuf: 数组长度.
double sumdouble_base(const double* pbuf, size_t cntbuf)
{
	double s = 0;	// 求和变量.
	size_t i;
	for(i=0; i<cntbuf; ++i)
	{
		s += pbuf[i];
	}
	return s;
}

#ifdef INTRIN_SSE2
// 双精度浮点数组求和_SSE版.
double sumdouble_sse(const double* pbuf, size_t cntbuf)
{
	double s = 0;	// 求和变量.
	size_t i;
	size_t nBlockWidth = 2;	// 块宽. SSE寄存器能一次处理2个double.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m128d xfdSum = _mm_setzero_pd();	// 求和变量。[SSE2] XORPD. 赋初值0.
	__m128d xfdLoad;	// 加载.
	const double* p = pbuf;	// SSE批量处理时所用的指针.
	const double* q;	// 将SSE变量上的多个数值合并时所用指针.

	// SSE批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		xfdLoad = _mm_load_pd(p);	// [SSE2] MOVAPD. 加载.
		xfdSum = _mm_add_pd(xfdSum, xfdLoad);	// [SSE2] ADDPD. 双精浮点紧缩加法.
		p += nBlockWidth;
	}
	// 合并.
	q = (const double*)&xfdSum;
	s = q[0] + q[1];

	// 处理剩下的.
	for(i=0; i<cntRem; ++i)
	{
		s += p[i];
	}

	return s;
}

// 双精度浮点数组求和_SSE四路循环展开版.
double sumdouble_sse_4loop(const double* pbuf, size_t cntbuf)
{
	double s = 0;	// 返回值.
	size_t i;
	size_t nBlockWidth = 2*4;	// 块宽. SSE寄存器能一次处理2个double，然后循环展开4次.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m128d xfdSum = _mm_setzero_pd();	// 求和变量。[SSE2] XORPD. 赋初值0.
	__m128d xfdSum1 = _mm_setzero_pd();
	__m128d xfdSum2 = _mm_setzero_pd();
	__m128d xfdSum3 = _mm_setzero_pd();
	__m128d xfdLoad;	// 加载.
	__m128d xfdLoad1;
	__m128d xfdLoad2;
	__m128d xfdLoad3;
	const double* p = pbuf;	// SSE批量处理时所用的指针.
	const double* q;	// 将SSE变量上的多个数值合并时所用指针.

	// SSE批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		xfdLoad = _mm_load_pd(p);	// [SSE2] MOVAPD. 加载.
		xfdLoad1 = _mm_load_pd(p+2);
		xfdLoad2 = _mm_load_pd(p+4);
		xfdLoad3 = _mm_load_pd(p+6);
		xfdSum = _mm_add_pd(xfdSum, xfdLoad);	// [SSE2] ADDPD. 双精浮点紧缩加法.
		xfdSum1 = _mm_add_pd(xfdSum1, xfdLoad1);
		xfdSum2 = _mm_add_pd(xfdSum2, xfdLoad2);
		xfdSum3 = _mm_add_pd(xfdSum3, xfdLoad3);
		p += nBlockWidth;
	}
	// 合并.
	xfdSum = _mm_add_pd(xfdSum, xfdSum1);	// 两两合并(0~1).
	xfdSum2 = _mm_add_pd(xfdSum2, xfdSum3);	// 两两合并(2~3).
	xfdSum = _mm_add_pd(xfdSum, xfdSum2);	// 两两合并(0~3).
	q = (const double*)&xfdSum;
	s = q[0] + q[1];

	// 处理剩下的.
	for(i=0; i<cntRem; ++i)
	{
		s += p[i];
	}

	return s;
}
#endif	// #ifdef INTRIN_SSE2


#ifdef INTRIN_AVX
// 双精度浮点数组求和_AVX版.
double sumdouble_avx(const double* pbuf, size_t cntbuf)
{
	double s = 0;	// 求和变量.
	size_t i;
	size_t nBlockWidth = 4;	// 块宽. AVX寄存器能一次处理4个double.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m256d yfdSum = _mm256_setzero_pd();	// 求和变量。[AVX] VXORPD. 赋初值0.
	__m256d yfdLoad;	// 加载.
	const double* p = pbuf;	// AVX批量处理时所用的指针.
	const double* q;	// 将AVX变量上的多个数值合并时所用指针.

	// AVX批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		yfdLoad = _mm256_load_pd(p);	// [AVX] VMOVAPD. 加载.
		yfdSum = _mm256_add_pd(yfdSum, yfdLoad);	// [AVX] VADDPD. 双精浮点紧缩加法.
		p += nBlockWidth;
	}
	// 合并.
	q = (const double*)&yfdSum;
	s = q[0] + q[1] + q[2] + q[3];

	// 处理剩下的.
	for(i=0; i<cntRem; ++i)
	{
		s += p[i];
	}

	return s;
}

// 双精度浮点数组求和_AVX四路循环展开版.
double sumdouble_avx_4loop(const double* pbuf, size_t cntbuf)
{
	double s = 0;	// 求和变量.
	size_t i;
	size_t nBlockWidth = 4*4;	// 块宽. AVX寄存器能一次处理8个double，然后循环展开4次.
	size_t cntBlock = cntbuf / nBlockWidth;	// 块数.
	size_t cntRem = cntbuf % nBlockWidth;	// 剩余数量.
	__m256d yfdSum = _mm256_setzero_pd();	// 求和变量。[AVX] VXORPD. 赋初值0.
	__m256d yfdSum1 = _mm256_setzero_pd();
	__m256d yfdSum2 = _mm256_setzero_pd();
	__m256d yfdSum3 = _mm256_setzero_pd();
	__m256d yfdLoad;	// 加载.
	__m256d yfdLoad1;
	__m256d yfdLoad2;
	__m256d yfdLoad3;
	const double* p = pbuf;	// AVX批量处理时所用的指针.
	const double* q;	// 将AVX变量上的多个数值合并时所用指针.

	// AVX批量处理.
	for(i=0; i<cntBlock; ++i)
	{
		yfdLoad = _mm256_load_pd(p);	// [AVX] VMOVAPD. 加载.
		yfdLoad1 = _mm256_load_pd(p+4);
		yfdLoad2 = _mm256_load_pd(p+8);
		yfdLoad3 = _mm256_load_pd(p+12);
		yfdSum = _mm256_add_pd(yfdSum, yfdLoad);	// [AVX] VADDPD. 双精浮点紧缩加法.
		yfdSum1 = _mm256_add_pd(yfdSum1, yfdLoad1);
		yfdSum2 = _mm256_add_pd(yfdSum2, yfdLoad2);
		yfdSum3 = _mm256_add_pd(yfdSum3, yfdLoad3);
		p += nBlockWidth;
	}
	// 合并.
	yfdSum = _mm256_add_pd(yfdSum, yfdSum1);	// 两两合并(0~1).
	yfdSum2 = _mm256_add_pd(yfdSum2, yfdSum3);	// 两两合并(2~3).
	yfdSum = _mm256_add_pd(yfdSum, yfdSum2);	// 两两合并(0~3).
	q = (const double*)&yfdSum;
	s = q[0] + q[1] + q[2] + q[3];

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


#define BUFSIZE	204800
ATTR_ALIGN(32) double buf[BUFSIZE];

// 测试时的函数类型
typedef double (*TESTPROC)(const double* pbuf, size_t cntbuf);

// 进行测试
void runTest(const char* szname, TESTPROC proc)
{
	const int testloop = 4000;	// 重复运算几次延长时间，避免计时精度问题.
	const clock_t TIMEOUT = CLOCKS_PER_SEC/2;	// 最短测试时间.
	int j;
	clock_t	tm0, dt;	// 存储时间.
	double mps;	// M/s
	volatile double n=0;	// 避免内循环被优化.

	tm0 = clock();

	for(j=1; j<=testloop; ++j)	// 重复运算几次延长时间，避免计时开销带来的影响.
	{
		n = proc(buf, BUFSIZE);	// 避免内循环被编译优化消掉.
	}

	dt = clock() - tm0;

	// show
	mps = (double)testloop*BUFSIZE*CLOCKS_PER_SEC/(1024.0*1024.0*dt);
	printf("%s:\t%.0f M/s\t%ld ms //%f\n", szname, mps, dt, n);
}

int main(int argc, char* argv[])
{
	char szBuf[64];
	int i;

	printf("simdsumdouble v1.00 (%dbit)\n", INTRIN_WORDSIZE);
	printf("Compiler: %s\n", COMPILER_NAME);
	cpu_getbrand(szBuf);
	printf("CPU:\t%s\n", szBuf);
	printf("\n");

	// init buf
	srand( (unsigned)time( NULL ) );
	for (i = 0; i < BUFSIZE; i++) buf[i] = (double)(rand() & 0x7fff);	// 使用&0x7fff是为了让求和后的数值在一定范围内，便于观察结果是否正确.

	// test
	runTest("sumdouble_base", sumdouble_base);	// 双精度浮点数组求和_基本版.
#ifdef INTRIN_SSE2
	if (simd_sse_level(NULL) >= SIMD_SSE_2)
	{
		runTest("sumdouble_sse", sumdouble_sse);	// 双精度浮点数组求和_SSE版.
		runTest("sumdouble_sse_4loop", sumdouble_sse_4loop);	// 双精度浮点数组求和_SSE四路循环展开版.
	}
#endif	// #ifdef INTRIN_SSE2
#ifdef INTRIN_AVX
	if (simd_avx_level(NULL) >= SIMD_AVX_1)
	{
		runTest("sumdouble_avx", sumdouble_avx);	// 双精度浮点数组求和_SSE版.
		runTest("sumdouble_avx_4loop", sumdouble_avx_4loop);	// 双精度浮点数组求和_SSE四路循环展开版.
	}
#endif	// #ifdef INTRIN_AVX

	return 0;
}