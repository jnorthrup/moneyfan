#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int mf_ane_hrm_available(void);
int mf_ane_hrm_init_1x1_16x16(void);
int mf_ane_hrm_eval_1x1_16x16(const float *input1, float *output16x16);
void mf_ane_hrm_close_1x1_16x16(void);

int mf_ane_sample_net_available(void);
int mf_ane_sample_net_init(void);
int mf_ane_sample_net_eval_1x1_to_16x16(const float *input1, float *output16x16);
void mf_ane_sample_net_close(void);

#ifdef __cplusplus
}
#endif
