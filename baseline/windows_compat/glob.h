/* The upstream svm-predict.cpp includes POSIX glob.h but calls no glob API.
 * This empty compatibility header allows the unmodified predictor to build
 * on Windows without changing training or prediction behavior. */
#pragma once
