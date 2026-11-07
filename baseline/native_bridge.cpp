#include "MetaRecognition.h"
#include <cmath>
#include <cstring>

extern "C" __declspec(dllexport) int mr_fit_rows(double *values, int rows, int cols, int low, double *out) {
    int failed = 0;
    for (int r=0; r<rows; ++r) {
        MetaRecognition mr;
        if (low) mr.FitLow(values+r*cols, cols, cols);
        else mr.FitHigh(values+r*cols, cols, cols);
        if (!mr.is_valid()) ++failed;
        out[r*6+0] = mr.get_scale_param();
        out[r*6+1] = mr.get_shape_param();
        out[r*6+2] = mr.get_sign();
        out[r*6+3] = mr.get_translate_amount();
        out[r*6+4] = mr.get_small_score();
        out[r*6+5] = mr.is_valid();
    }
    return failed;
}

extern "C" __declspec(dllexport) void mr_score_rows(const double *values, const double *params, int rows, int cols, double *out) {
    for (int r=0; r<rows; ++r) {
        MetaRecognition mr;
        mr.set_sign((int)params[r*6+2]);
        mr.set_translate_amount((int)params[r*6+3]);
        mr.set_small_score(params[r*6+4]);
        mr.set_scale_param(params[r*6]);
        mr.set_shape_param(params[r*6+1]);
        mr.set_valid();
        for (int c=0; c<cols; ++c) out[r*cols+c] = mr.W_score(values[r*cols+c]);
    }
}
