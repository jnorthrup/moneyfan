#import "moneyfan_ane_bridge.h"

#import <Foundation/Foundation.h>
#import <dlfcn.h>

#import "ane_runtime.h"
#import "ane_mil_gen.h"

enum {
    HRM_OUT_SIDE = 16,
    HRM_SPATIAL = 16 * 16,
    HRM_IN_CHANNELS = 1,
    HRM_OUT_CHANNELS = 1,
};

static ANEKernel *g_hrm_kernel = NULL;
static int g_hrm_initialized = 0;

static void fill_f32(float *dst, int n, float value) {
    for (int i = 0; i < n; i++) {
        dst[i] = value;
    }
}

int mf_ane_hrm_available(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

    Class desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    Class inMem = NSClassFromString(@"_ANEInMemoryModel");
    Class request = NSClassFromString(@"_ANERequest");
    Class ioObj = NSClassFromString(@"_ANEIOSurfaceObject");

    return (desc && inMem && request && ioObj) ? 1 : 0;
}

int mf_ane_hrm_init_1x1_16x16(void) {
    if (g_hrm_initialized) {
        return 0;
    }

    g_hrm_initialized = 1;
    if (!mf_ane_hrm_available()) {
        // Keep initialization successful, but runtime will use CPU fallback.
        return 1;
    }

    float weight[HRM_IN_CHANNELS * HRM_OUT_CHANNELS] = { 1.0f };
    NSData *weights = mil_build_weight_blob(weight, HRM_OUT_CHANNELS, HRM_IN_CHANNELS);
    NSString *mil = mil_gen_conv(HRM_IN_CHANNELS, HRM_OUT_CHANNELS, HRM_SPATIAL);

    size_t inBytes = (size_t)HRM_SPATIAL * sizeof(float);
    size_t outBytes = (size_t)HRM_SPATIAL * sizeof(float);

    g_hrm_kernel = ane_compile(
        [mil dataUsingEncoding:NSUTF8StringEncoding],
        weights,
        1,
        &inBytes,
        1,
        &outBytes
    );

    if (g_hrm_kernel == NULL) {
        return -1;
    }

    return 0;
}

int mf_ane_hrm_eval_1x1_16x16(const float *input1, float *output16x16) {
    if (input1 == NULL || output16x16 == NULL) {
        return -1;
    }

    if (!g_hrm_initialized) {
        int rc = mf_ane_hrm_init_1x1_16x16();
        if (rc < 0) {
            return rc;
        }
    }

    float scalar = input1[0];
    float expanded[HRM_SPATIAL];
    fill_f32(expanded, HRM_SPATIAL, scalar);

    if (g_hrm_kernel != NULL) {
        ane_write_input(g_hrm_kernel, 0, expanded, sizeof(expanded));
        if (!ane_eval(g_hrm_kernel)) {
            return -2;
        }
        ane_read_output(g_hrm_kernel, 0, output16x16, (size_t)HRM_SPATIAL * sizeof(float));
        return 0;
    }

    // CPU fallback when ANE private runtime is unavailable.
    fill_f32(output16x16, HRM_SPATIAL, scalar);
    return 1;
}

void mf_ane_hrm_close_1x1_16x16(void) {
    if (g_hrm_kernel != NULL) {
        ane_free(g_hrm_kernel);
        g_hrm_kernel = NULL;
    }
    g_hrm_initialized = 0;
}

int mf_ane_sample_net_available(void) {
    return mf_ane_hrm_available();
}

int mf_ane_sample_net_init(void) {
    return mf_ane_hrm_init_1x1_16x16();
}

int mf_ane_sample_net_eval_1x1_to_16x16(const float *input1, float *output16x16) {
    return mf_ane_hrm_eval_1x1_16x16(input1, output16x16);
}

void mf_ane_sample_net_close(void) {
    mf_ane_hrm_close_1x1_16x16();
}
