#ifndef BOLTZMANN_GPU
#define BOLTZMANN_GPU

void av(int blocks, ffloat *a, ffloat *b, ffloat *av_data, ffloat t);

void step_on_grid(int blocks, ffloat *a0, ffloat *a_current,    ffloat *b_current,
                  ffloat *a_next,       ffloat *b_next,
                  ffloat *a_current_hs, ffloat *b_current_hs,
                  ffloat t, ffloat t_hs, ffloat cos_omega_t, ffloat cos_omega_t_plus_dt);

void step_on_half_grid(int blocks, ffloat *a0, ffloat *a_current,    ffloat *b_current,
                       ffloat *a_next,       ffloat *b_next,
                       ffloat *a_current_hs, ffloat *b_current_hs,
                       ffloat *a_next_hs, ffloat *b_next_hs,
                       ffloat t, ffloat t_hs, ffloat cos_omega_t, ffloat cos_omega_t_plus_dt);

void step_on_grid_nr(int blocks, ffloat *a0, ffloat *a_current,    ffloat *b_current,
                  ffloat *a_next,       ffloat *b_next,
                  ffloat *a_current_hs, ffloat *b_current_hs,
                  ffloat t, ffloat t_hs, ffloat cos_omega_t, ffloat cos_omega_t_plus_dt);

void step_on_half_grid_nr(int blocks, ffloat *a0, ffloat *a_current,    ffloat *b_current,
                       ffloat *a_next,       ffloat *b_next,
                       ffloat *a_current_hs, ffloat *b_current_hs,
                       ffloat *a_next_hs, ffloat *b_next_hs,
                       ffloat t, ffloat t_hs, ffloat cos_omega_t, ffloat cos_omega_t_plus_dt);

void HandleError(cudaError_t, const char *, int);
void load_data(void);

#endif
