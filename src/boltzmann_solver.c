#include <cuda_runtime_api.h>
#include <gsl/gsl_specfunc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "constants.h"
#include "boltzmann.h"
#include "boltzmann_cli.h"
#include "boltzmann_gpu.h"

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define cuda_clean_up() \
  do { \
    cudaFree(a0); \
    for( int i = 0; i < 4; i++ ) { \
      cudaFree(a[i]); \
      cudaFree(b[i]); \
    } \
    cudaFree(av_data); \
  } while(0)

ffloat *strobe_values;

ffloat eval_norm(ffloat *, ffloat, int);
void init_strobe_array();
void print_time_evolution_of_parameters(FILE *, ffloat, ffloat *, ffloat *, int,
                                        ffloat, ffloat, ffloat, ffloat,
                                        ffloat, ffloat *, ffloat);
void print_2d_strobe(FILE *, int, ffloat *, ffloat *, ffloat *, ffloat, ffloat);
void print_2d_data(FILE *,   int, ffloat *, ffloat *, ffloat *, ffloat, ffloat);

extern int display, host_N, quiet;
extern ffloat host_E_dc, host_E_omega, host_omega, host_mu, host_alpha,
  PhiYmin, PhiYmax, host_B, t_start, frame_start;
extern char  *out_file;
extern FILE  *out, *read_from;

int device = 0;

ffloat t_max = -999; // calculation stops at this time

// grid step along
ffloat host_dPhi = 0;

// grid from 0 to PhiY_max_range is broken into host_M steps
// change it here if you need more resolution along \phi_y axis
int host_M = 3069;

// array sizes will be derived from host_M variable; following variables holds
// various sizes related to be allocated arrays
int MSIZE, PADDED_MSIZE, MP1, host_TMSIZE, SIZE_2D;

int NSIZE;

// time step
// change it here if you experience numerical instability
ffloat host_dt =   0.001; //0.0001;

// various constants that are used in the kernel are precomputed for
// optimization using following variables
ffloat host_bdt, host_nu_tilde, host_nu2, host_nu;

// macro used to access our primary data array
#define nm(pointer, n, m) (*((pointer)+(n)*PADDED_MSIZE+(m)))

// takes position one the grid along \phi_y axis and converts into \phi_y value
//#define phi_y(m) (host_dPhi*((m)-host_M-1))
#define phi_y(m) (PhiYmin+host_dPhi*((m)-1))

int main(int argc, char **argv) {
  parse_cmd(argc, argv);

  cudaSetDevice(device);

  ffloat T = host_omega>0?(2*PI/host_omega):0; // period of external a/c emf
  if( display == 9 ) {
    t_max = t_start + 101*T;
    init_strobe_array();
  } else {
    t_max = t_start + T;
  }
  if( quiet == 0 ) { printf("# t_max = %0.20f kernel=%d\n", t_max, BLTZM_KERNEL); }

  // we will allocate enough memory to accommodate range
  // from -PhiY_max_range to PhiY_max_range, but will use only
  // part of it from PhiYmin to PhiYmax
  ffloat PhiY_max_range = fabs(PhiYmin);
  if( PhiY_max_range < fabs(PhiYmax) ) {
    PhiY_max_range = fabs(PhiYmax);
  }

  //host_dPhi = PhiY_max_range/host_M;
  host_dPhi = (PhiYmax-PhiYmin)/host_M;

  NSIZE = host_N+1;
  //MSIZE = 2*host_M+3;
  MSIZE = host_M+3;
  PADDED_MSIZE = (MSIZE*sizeof(ffloat))%128==0?MSIZE:((((MSIZE*sizeof(ffloat))/128)*128+128)/sizeof(ffloat));
  printf("PADDED MEMORY FROM %d ELEMENTS PER ROW TO %d\n", MSIZE, (int)PADDED_MSIZE);

  MP1 = host_M+1; // 

  SIZE_2D = NSIZE*PADDED_MSIZE;
  const int SIZE_2Df = SIZE_2D*sizeof(ffloat);

  host_TMSIZE=host_M+1;

  host_nu = 1+host_dt/2;
  host_nu2 = host_nu * host_nu;
  host_nu_tilde = 1-host_dt/2;
  host_bdt = host_B*host_dt/(4*host_dPhi);

  load_data();

  // create a0 and populate it with f0
  ffloat *host_a0; host_a0 = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
  for( int n=0; n<host_N+1; n++ ) {
    ffloat a = gsl_sf_bessel_In(n, host_mu)*(n==0?0.5:1)/(PI*gsl_sf_bessel_In(0, host_mu))*sqrt(host_mu/(2*PI*host_alpha));
    for( int m = 0; m < host_M+3; m++ ) {
      nm(host_a0, n, m) = a*expl(-host_mu*pow(phi_y(m),2)/2);
    }
  }

  // create device_a0 and transfer data from host_a0 to device_a0
  ffloat *a0;
  HANDLE_ERROR(cudaMalloc((void **)&a0, SIZE_2Df));
  HANDLE_ERROR(cudaMemcpy(a0, host_a0, SIZE_2Df, cudaMemcpyHostToDevice));

  // create a and b 2D vectors, four of each. one for current,
  // another for next pointer on main and shifted grids
  ffloat *host_a = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
  ffloat *host_b = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));

  ffloat *a[4];
  ffloat *b[4];
  for( int i = 0; i < 4; i++ ) {
    HANDLE_ERROR(cudaMalloc((void **)&a[i], SIZE_2Df));
    HANDLE_ERROR(cudaMalloc((void **)&b[i], SIZE_2Df));

    // zero vector b[i]
    HANDLE_ERROR(cudaMemset((void *)a[i], 0, SIZE_2Df));
    HANDLE_ERROR(cudaMemset((void *)b[i], 0, SIZE_2Df));
  }

  int current = 0; int next = 1;
  int current_hs = 2; int next_hs = 3; // 'hs' - half step

  // init vectors a[0] and a[2]
  HANDLE_ERROR(cudaMemcpy(a[current], host_a0, SIZE_2Df,
                          cudaMemcpyHostToDevice));

  int blocks = (host_M+3)/TH_PER_BLOCK;

  // tiptow to the first half step
  ffloat *host_a_hs = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
  ffloat *host_b_hs = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
  ffloat cos_omega_t = 1; // cos(host_omega*t); for t = 0
  ffloat cos_omega_t_plus_dt = cos(host_omega*(host_dt));
  step_on_grid(blocks, a0, a[current], b[current], a[current_hs], b[current_hs],
               a[current], b[current], 0, 0,
               cos_omega_t, cos_omega_t_plus_dt);
  /*
  // temporary solution // FIX ME!!!
  memcpy(host_a_hs, host_a, SIZE_2D*sizeof(ffloat));
  HANDLE_ERROR(cudaMemcpy(a[current_hs], host_a_hs,
                          SIZE_2Df, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(b[current_hs], host_b_hs,
                          SIZE_2Df, cudaMemcpyHostToDevice));
  */

  // used for file names when generated data for making animation
  char *file_name_buf = (char *)calloc(128, sizeof(char));

  char buf[16384]; // output buffer for writing frame data when display==77

  int step = 0;
  ffloat frame_time = 0; int frame_number = 1;

  ffloat *host_av_data; host_av_data = (ffloat *)calloc(5, sizeof(ffloat));
  ffloat *av_data;
  HANDLE_ERROR(cudaMalloc((void **)&av_data, 6*sizeof(ffloat)));
  HANDLE_ERROR(cudaMemset((void *)av_data, 0, 6*sizeof(ffloat)));

  float t_hs = 0;

  ffloat t0 = 0;
  ffloat t = t0;
  ffloat timeout = -999;

  ffloat last_tT_reminder = 0;

  for(;;) {
    //read_from
    int ccc = 0;
    for( t = t0; t < t_max; t += host_dt ) {
      /// XXX
      //ccc++;
      //if( ccc == 51 ) { break; }

      t_hs = t + host_dt/2;
      cos_omega_t = cos(host_omega*t);
      cos_omega_t_plus_dt = cos(host_omega*(t+host_dt));
      step_on_grid(blocks, a0, a[current], b[current], a[next], b[next], a[current_hs],
                   b[current_hs], t, t_hs,
                   cos_omega_t, cos_omega_t_plus_dt);

      cudaThreadSynchronize();

      cos_omega_t = cos(host_omega*t_hs);
      cos_omega_t_plus_dt = cos(host_omega*(t_hs+host_dt));
      step_on_half_grid(blocks, a0, a[current], b[current], a[next], b[next], a[current_hs],
                        b[current_hs], a[next_hs], b[next_hs], t, t_hs,
                        cos_omega_t, cos_omega_t_plus_dt);

      /*
      if( t >= 0 ) { 
	HANDLE_ERROR(cudaMemcpy(host_a, a[current], SIZE_2Df, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(host_b, b[current], SIZE_2Df, cudaMemcpyDeviceToHost));
          sprintf(file_name_buf, "strobe.data");
          FILE *frame_file_stream = fopen(file_name_buf, "w");
          setvbuf(frame_file_stream, buf, _IOFBF, sizeof(buf));
          printf("\nWriting strobe %s\n", file_name_buf);
          print_2d_strobe(frame_file_stream, MSIZE, host_a0, host_a, host_b, host_alpha, t);
          fclose(frame_file_stream);
          frame_time = 0;

	break; } /// XXX REMOVE ME
      */

      if( host_E_omega > 0 && display == 77 && frame_time >= 0.01) {
        // we need to perform averaging of v_dr, m_x and A
        av(blocks, a[next], b[next], av_data, t);
        HANDLE_ERROR(cudaMemcpy(host_a, a[current], SIZE_2Df, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(host_b, b[current], SIZE_2Df, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(host_av_data, av_data, 6*sizeof(ffloat), cudaMemcpyDeviceToHost));
        ffloat norm = eval_norm(host_a, host_alpha, MSIZE);
        print_time_evolution_of_parameters(out, norm, host_a, host_b, MSIZE,
                                           host_mu, host_alpha, host_E_dc, host_E_omega, host_omega,
                                           host_av_data, t);
        frame_time = 0;
      }

      if( host_E_omega > 0 && display != 7 && display != 77 && display != 8 && t >= t_start ) {
        // we need to perform averaging of v_dr, m_x and A
        av(blocks, a[next], b[next], av_data, t);
      }

      if( current    == 0 ) {    current = 1;    next = 0; } else { current = 0; next = 1; }
      if( current_hs == 2 ) { current_hs = 3; next_hs = 2; } else { current_hs = 2; next_hs = 3; }

      //if( display == 9 && t >= t_start ) {
      //  ffloat tT = t/T;
      //  printf("t=%0.12f %0.12f %0.12f\n", t, , T);
      //}

      if( display == 9 && t >= t_start ) { // XXX PUT ME BACK
        ffloat tT = t/T;
        ffloat tT_reminder = tT-((int)tT);
        if( tT_reminder < last_tT_reminder ) { 
          HANDLE_ERROR(cudaMemcpy(host_a, a[current], SIZE_2Df, cudaMemcpyDeviceToHost));
          HANDLE_ERROR(cudaMemcpy(host_b, b[current], SIZE_2Df, cudaMemcpyDeviceToHost));
          sprintf(file_name_buf, "strobe%08d.data", frame_number++);
          FILE *frame_file_stream = fopen(file_name_buf, "w");
          setvbuf(frame_file_stream, buf, _IOFBF, sizeof(buf));
          printf("\nWriting strobe %s\n", file_name_buf);
          print_2d_strobe(frame_file_stream, MSIZE, host_a0, host_a, host_b, host_alpha, t);
          fclose(frame_file_stream);
          frame_time = 0;
	}
        last_tT_reminder = tT_reminder;
      }

      if( display == 7 && frame_time >= 0.01 && t > frame_start ) { // we are making movie
        HANDLE_ERROR(cudaMemcpy(host_a, a[current], SIZE_2Df, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(host_b, b[current], SIZE_2Df, cudaMemcpyDeviceToHost));
        sprintf(file_name_buf, "frame%08d.data", frame_number++);
        FILE *frame_file_stream = fopen(file_name_buf, "w");
        setvbuf(frame_file_stream, buf, _IOFBF, sizeof(buf));
        printf("\nWriting frame %s\n", file_name_buf);
        print_2d_data(frame_file_stream, MSIZE, host_a0, host_a, host_b, host_alpha, t);
        fclose(frame_file_stream);
        frame_time=0;
      }

      if( out != stdout && display != 7 ) {
        step++;
        if( step == 300 ) {
          printf("\rt=%0.9f %0.2f%%", t, t/t_max*100);
          fflush(stdout);
          step = 0;
        }
      }
      frame_time += host_dt;

      if( display == 9 && t <= t_start && frame_time >= T ) {
        frame_time == 0;
      }
    }

    HANDLE_ERROR(cudaMemcpy(host_a, a[current], SIZE_2Df, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_b, b[current], SIZE_2Df, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_av_data, av_data, 6*sizeof(ffloat), cudaMemcpyDeviceToHost));

    ffloat norm = 0;
    ffloat dphi_over_2 = host_dPhi/2.0;
    for( int m = 1; m < host_M+1; m++ ) {
      norm += (nm(host_a,0,m)+nm(host_a,0,m))*dphi_over_2;
    }
    norm *= 2*PI*sqrt(host_alpha);

    if( display == 3 ) {
      for( ffloat phi_x = -PI; phi_x < PI; phi_x += 0.01 ) {
        for( int m = 1; m < host_M; m++ ) {
          ffloat value = 0;
          ffloat value0 = 0;
          for( int n = 0; n < host_N+1; n++ ) {
            value  += nm(host_a,n,m)*cos(n*phi_x) + nm(host_b,n,m)*sin(n*phi_x);
            value0 += nm(host_a0,n,m)*cos(n*phi_x);
          }
          fprintf(out, "%0.5f %0.5f %0.20f %0.20f\n", phi_x, phi_y(m), value<0?0:value, value0<0?0:value0);
        }
      }
      fprintf(out, "# norm=%0.20f\n", norm);
      printf("# norm=%0.20f\n", norm);
      //if( out != stdout ) { fclose(out); }
      cuda_clean_up();
      return EXIT_SUCCESS;
    }

    if( display == 8 ) {
      // single shot image
      HANDLE_ERROR(cudaMemcpy(host_a, a[current], SIZE_2Df, cudaMemcpyDeviceToHost));
      HANDLE_ERROR(cudaMemcpy(host_b, b[current], SIZE_2Df, cudaMemcpyDeviceToHost));
      sprintf(file_name_buf, "frame.data");
      FILE *frame_file_stream = fopen(file_name_buf, "w");
      setvbuf(frame_file_stream, buf, _IOFBF, sizeof(buf));
      printf("\nWriting frame %s\n", file_name_buf);
      print_2d_data(frame_file_stream, MSIZE, host_a0, host_a, host_b, host_alpha, t);
      fclose(frame_file_stream);
      frame_time=0;
      return EXIT_SUCCESS;
    }

    if( display == 4 ) {
      if( quiet == 0 ) { printf("\n# norm=%0.20f\n", norm); }
      ffloat v_dr_inst = 0 ;
      ffloat v_y_inst = 0;
      ffloat m_over_m_x_inst = 0;
      for( int m = 1; m < host_M; m++ ) {
        v_dr_inst += nm(host_b,1,m)*host_dPhi;
        v_y_inst  += nm(host_a,0,m)*phi_y(m)*host_dPhi;
        m_over_m_x_inst += nm(host_a,1,m)*host_dPhi;
      }

      ffloat v_dr_multiplier = 2*gsl_sf_bessel_I0(host_mu)*PI*sqrt(host_alpha)/gsl_sf_bessel_In(1, host_mu);
      ffloat v_y_multiplier  = 4*PI*gsl_sf_bessel_I0(host_mu)/gsl_sf_bessel_In(1, host_mu);
      ffloat m_over_multiplier = PI*host_alpha*sqrt(host_alpha);
      v_dr_inst       *= v_dr_multiplier;
      v_y_inst        *= v_y_multiplier;
      m_over_m_x_inst *= m_over_multiplier;

      host_av_data[1] *= v_dr_multiplier;
      host_av_data[2] *= v_y_multiplier;
      host_av_data[3] *= m_over_multiplier;
      host_av_data[4] *= v_dr_multiplier;
      host_av_data[4] /= T;
      host_av_data[5] *= v_dr_multiplier;
      host_av_data[5] /= T;

      fprintf(out, "# display=%d E_dc=%0.20f E_omega=%0.20f omega=%0.20f mu=%0.20f alpha=%0.20f n-harmonics=%d PhiYmin=%0.20f PhiYmax=%0.20f B=%0.20f t-max=%0.20f dt=%0.20f g-grid=%d\n",
                      display,   host_E_dc,  host_E_omega,  host_omega,  host_mu,  host_alpha,  host_N,        PhiYmin,       PhiYmax,       host_B,  t_start,     host_dt,  host_M);
      fprintf(out, "#E_{dc}                \\tilde{E}_{\\omega}     \\tilde{\\omega}         mu                     v_{dr}/v_{p}         A(\\omega)              NORM     v_{y}/v_{p}    m/m_{x,k}   <v_{dr}/v_{p}>   <v_{y}/v_{p}>    <m/m_{x,k}>    Asin\n");
      fprintf(out, "%0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f\n",
              host_E_dc, host_E_omega, host_omega, host_mu, v_dr_inst, host_av_data[4], norm, v_y_inst,
              m_over_m_x_inst, host_av_data[1], host_av_data[2], host_av_data[3], host_av_data[5]);
    }

    if( read_from == NULL ) { break; }

    // scan for new parameters
    timeout = scan_for_new_parameters();
    if( timeout < -900 ) { break; } // user entered 'exit'
    t_start = t + timeout;
    t_max = t_start + T;
    t0 = t + host_dt;
    T=host_omega>0?(2*PI/host_omega):0;
    load_data(); // re-load data
    HANDLE_ERROR(cudaMemset((void *)av_data, 0, 6*sizeof(ffloat))); // clear averaging data
    if( quiet == 0 ) { printf("# t_max = %0.20f\n", t_max); }
  } // for(;;)

  if( out != NULL && out != stdout ) {
    fclose(out);
  }
  cuda_clean_up();
  return EXIT_SUCCESS;
} // end of main(...)

ffloat eval_norm(ffloat *host_a, ffloat host_alpha, int MSIZE) {
  ffloat norm = 0;
  for( int m = 1; m < 2*host_M+2; m++ ) {
    norm += nm(host_a,0,m)*host_dPhi;
  }
  norm *= 2*PI*sqrt(host_alpha);
  return norm;
} // end of eval_norm(...)

void print_time_evolution_of_parameters(FILE *out, ffloat norm, ffloat *host_a, ffloat *host_b, int MSIZE,
                                        ffloat host_mu, ffloat host_alpha, ffloat host_E_dc, ffloat host_E_omega,
                                        ffloat host_omega, ffloat *host_av_data, ffloat t)
{
  printf("\n# t=%0.20f norm=%0.20f\n", t, norm);
  ffloat v_dr_inst = 0 ;
  ffloat v_y_inst = 0;
  ffloat m_over_m_x_inst = 0;
  for( int m = 1; m < 2*host_M+2; m++ ) {
    v_dr_inst += nm(host_b,1,m)*host_dPhi;
    v_y_inst  += nm(host_a,0,m)*phi_y(m)*host_dPhi;
    m_over_m_x_inst += nm(host_a,1,m)*host_dPhi;
  }

  ffloat v_dr_multiplier = 2*gsl_sf_bessel_I0(host_mu)*PI*sqrt(host_alpha)/gsl_sf_bessel_In(1, host_mu);
  ffloat v_y_multiplier  = 4*PI*gsl_sf_bessel_I0(host_mu)/gsl_sf_bessel_In(1, host_mu);
  ffloat m_over_multiplier = PI*host_alpha*sqrt(host_alpha);
  v_dr_inst       *= v_dr_multiplier;
  v_y_inst        *= v_y_multiplier;
  m_over_m_x_inst *= m_over_multiplier;

  host_av_data[1] *= v_dr_multiplier;
  host_av_data[2] *= v_y_multiplier;
  host_av_data[3] *= m_over_multiplier;
  host_av_data[4] *= v_dr_multiplier;
  host_av_data[4] /= t;
  host_av_data[5] *= v_dr_multiplier;
  host_av_data[5] /= t;

  fprintf(out, "#E_{dc}                \\tilde{E}_{\\omega}     \\tilde{\\omega}         mu                     v_{dr}/v_{p}         A(\\omega)              NORM     v_{y}/v_{p}    m/m_{x,k}   <v_{dr}/v_{p}>   <v_{y}/v_{p}>    <m/m_{x,k}>  A_{inst}  t    Asin\n");
  fprintf(out, "%0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f\n",
          host_E_dc, host_E_omega, host_omega, host_mu, v_dr_inst, host_av_data[4], norm, v_y_inst,
          m_over_m_x_inst, host_av_data[1], host_av_data[2], host_av_data[3], cos(host_omega*t)*v_dr_inst, t, host_av_data[4]);
} // end of print_time_evolution_of_parameters(...)

void init_strobe_array() {
  printf("init_strobe_array\n");
  int size = 0;
  for( ffloat phi_x = -PI; phi_x < PI; phi_x += 0.01 ) {
    for( int m = 1; m < host_M+2; m++ ) {
      size++;
    }
  }
  strobe_values = (ffloat *)calloc(size, sizeof(ffloat));
}

// Write out distribution function to FILE *out. This used when writing data to generate animation
void print_2d_strobe(FILE *out, int MSIZE, ffloat *host_a0, ffloat *host_a, ffloat *host_b, ffloat host_alpha, ffloat t) {
  ffloat norm = 0;
  ffloat dphi_over_2 = host_dPhi/2.0;
  for( int m = 1; m < host_M+1; m++ ) {
    norm += (nm(host_a,0,m)+nm(host_a,0,m))*dphi_over_2;
  }
  norm *= 2*PI*sqrt(host_alpha);

  int i = 0;
  for( ffloat phi_x = -PI; phi_x < PI; phi_x += 0.01 ) {
    for( int m = 1; m < host_M+2; m++ ) {
      ffloat value = 0;
      for( int n = 0; n < host_N+1; n++ ) {
        value  += nm(host_a,n,m)*cos(n*phi_x) + nm(host_b,n,m)*sin(n*phi_x);
      }
      strobe_values[i] = strobe_values[i] + (value<0?0:value);
      //strobe_values[i] = value<0?0:value;
      fprintf(out, "%0.5f %0.5f %0.20f\n", phi_x, phi_y(m), strobe_values[i]);
      //fprintf(out, "%0.5f %0.5f %0.20f\n", phi_x, phi_y(m), value<0?0:value);
      i++;
    }
  }
  fprintf(out, "# norm=%0.20f\n", norm);
  fprintf(out, "# t=%0.20f\n", t);
  printf("# norm=%0.20f\n", norm);
} // end of print_2d_strobe(...)

// Write out distribution function to FILE *out. This used when writing data to generate animation
void print_2d_data(FILE *out, int MSIZE, ffloat *host_a0, ffloat *host_a, ffloat *host_b, ffloat host_alpha, ffloat t) {
  fprintf(out, "# t=%0.20f\n", t);
  ffloat norm = 0;
  for( int m = 1; m < 2*host_M+2; m++ ) {
    norm += nm(host_a,0,m)*host_dPhi;
  }
  norm *= 2*PI*sqrt(host_alpha);

  for( ffloat phi_x = -PI; phi_x < PI; phi_x += 0.01 ) {
    for( int m = 1; m < host_M+2; m++ ) {
      ffloat value = 0;
      //ffloat value0 = 0;
      for( int n = 0; n < host_N+1; n++ ) {
        value  += nm(host_a,n,m)*cos(n*phi_x) + nm(host_b,n,m)*sin(n*phi_x);
      }
      fprintf(out, "%0.5f %0.5f %0.20f\n", phi_x, phi_y(m), value<0?0:value);
    }
  }
  fprintf(out, "# norm=%0.20f\n", norm);
  printf("# norm=%0.20f\n", norm);
} // end of print_2d_data(...)
