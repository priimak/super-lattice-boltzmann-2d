#define _POSIX_C_SOURCE 1

#include <gsl/gsl_specfunc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "constants.h"
#include "boltzmann.h"
#include "boltzmann_cli.h"

void step_on_grid(ffloat *, ffloat *, ffloat *,
                  ffloat *, ffloat *, ffloat *,
                  ffloat *, ffloat,   ffloat,
                  ffloat,   ffloat);

void step_on_half_grid(ffloat *, ffloat *, ffloat *, 
                       ffloat *, ffloat *, ffloat *, 
                       ffloat *, ffloat *, ffloat *,                        
                       ffloat,   ffloat,   ffloat, ffloat);

void av(ffloat *, ffloat *, ffloat *, ffloat);

ffloat eval_norm(ffloat *, ffloat, int);
void print_time_evolution_of_parameters(FILE *, ffloat, ffloat *, ffloat *, int, 
                                        ffloat, ffloat, ffloat, ffloat, 
                                        ffloat, ffloat *, ffloat);
void print_2d_data(FILE *, int, ffloat *, ffloat *, ffloat *, ffloat);

extern int display, host_N;
extern ffloat host_omega, host_B, host_mu, host_alpha, host_E_omega, host_E_dc, PhiYmin, PhiYmax, t_start;

int N, device;
ffloat omega, B, mu, alpha, E_omega, E_dc;

extern char  *out_file;
extern FILE  *out, *read_from;

ffloat t_max = -999; // calculation stops at this time

// grid step along 
ffloat dPhi = 0;

// grid from 0 to PhiY_max_range is broken into M steps
// change it here if you need more resolution along \phi_y axis
int M = 3069;
int host_M = 3069;

// array sizes will be derived from M variable; following variables holds 
// various sizes related to be allocated arrays
int MSIZE, TMSIZE, SIZE_2D;

// time step
// change it here if you experience numerical instability
ffloat dt      = 0.001; //0.0001;
ffloat host_dt = 0.001; //0.0001;

// various constants that are used in the kernel are precomputed for 
// optimization using following variables
ffloat bdt, nu_tilde, nu2, nu;

// macro used to access our primary data array
#define nm(pointer, n, m) (*((pointer)+(n)*MSIZE+(m)))

// takes position one the grid along \phi_y axis and converts into \phi_y value
#define phi_y(m) (PhiYmin+dPhi*((m)-1))

void load_data() {
  omega   = host_omega;
  N       = host_N;
  B       = host_B;
  mu      = host_mu;
  alpha   = host_alpha;
  E_omega = host_E_omega;
  E_dc    = host_E_dc;
  dt      = host_dt;
  M       = host_M;
}

int main(int argc, char **argv) {
  parse_cmd(argc, argv);
  load_data();

  ffloat T=omega>0?(2*PI/omega):0; // period of external a/c emf
  t_max = t_start + T; 
  printf("# t_max = %0.20f\n", t_max);

  // we will allocate enough memory to accommodate range 
  // from -PhiY_max_range to PhiY_max_range, but will use only 
  // part of it from PhiYmin to PhiYmax
  ffloat PhiY_max_range = fabs(PhiYmin);
  if( PhiY_max_range < fabs(PhiYmax) ) { 
    PhiY_max_range = fabs(PhiYmax);
  }

  //dPhi = PhiY_max_range/M;
  dPhi = (PhiYmax-PhiYmin)/M;

  const int NSIZE = N+1;
  //MSIZE = 2*M+3;
  MSIZE = M+3;
  SIZE_2D = NSIZE*MSIZE;
  const int SIZE_2Df = SIZE_2D*sizeof(ffloat);

  TMSIZE=M+1;

  nu = 1+dt/2;
  nu2 = nu * nu;
  nu_tilde = 1-dt/2;
  bdt = B*dt/(4*dPhi);

  // create a0 and populate it with f0
  ffloat *a0; a0 = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
  for( int n=0; n<N+1; n++ ) {
    ffloat a = gsl_sf_bessel_In(n, mu)*(n==0?0.5:1)/(PI*gsl_sf_bessel_In(0, mu))*sqrt(mu/(2*PI*alpha));
    for( int m = 0; m < M+3; m++ ) {
      nm(a0, n, m) = a*expl(-mu*pow(phi_y(m),2)/2);
    }
  }

  // create a and b 2D vectors, four of each. one for current, 
  // another for next pointer on main and shifted grids
  ffloat *a[4], *b[4];
  for( int i = 0; i < 4; i++ ) { 
    a[i] = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
    b[i] = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
  }

  int current = 0; int next = 1;
  int current_hs = 2; int next_hs = 3; // 'hs' - half step

  // init vector a[0]
  memcpy(a[current], a0, SIZE_2Df);

  // tiptow to the first half step
  ffloat *a_hs = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
  ffloat *b_hs = (ffloat *)calloc(SIZE_2D, sizeof(ffloat));
  ffloat cos_omega_t = 1;
  ffloat cos_omega_t_plus_dt = cos(omega*(dt));
  step_on_grid(a0, a[current], b[current], a[current_hs], b[current_hs], 
               a[current], b[current], 0, 0, 
               cos_omega_t, cos_omega_t_plus_dt);

  // used for file names when generated data for making animation
  char *file_name_buf = (char *)calloc(128, sizeof(char));

  char buf[16384]; // output buffer for writing frame data when display==77

  int step = 0;
  ffloat frame_time = 0; int frame_number = 1;

  ffloat *av_data = (ffloat *)calloc(5, sizeof(ffloat));

  float t_hs = 0;

  ffloat t0 = 0;
  ffloat t = t0;
  ffloat timeout = -999;
  for(;;) {
    //read_from
    for( t = t0; t < t_max; t += dt ) {
      t_hs = t + dt/2;
      cos_omega_t = cos(omega*t);
      cos_omega_t_plus_dt = cos(omega*(t+dt));
      step_on_grid(a0, a[current], b[current], a[next], b[next], a[current_hs], 
                   b[current_hs], t, t_hs, 
                   cos_omega_t, cos_omega_t_plus_dt);

      cos_omega_t = cos(omega*t_hs);
      cos_omega_t_plus_dt = cos(omega*(t_hs+dt));
      step_on_half_grid(a0, a[current], b[current], a[next], b[next], a[current_hs], 
                        b[current_hs], a[next_hs], b[next_hs], t, t_hs, 
                        cos_omega_t, cos_omega_t_plus_dt);

      if( E_omega > 0 && display == 77 && frame_time >= 0.01) {
        // we need to perform averaging of v_dr, m_x and A
        av(a[next], b[next], av_data, t);
        ffloat norm = eval_norm(a[current], alpha, MSIZE);
        print_time_evolution_of_parameters(out, norm, a[current], b[current], MSIZE, 
                                           mu, alpha, E_dc, E_omega, omega,
                                           av_data, t);
        frame_time = 0;
      }

      if( E_omega > 0 && display != 7 && display != 77 && t >= t_start ) {
        // we need to perform averaging of v_dr, m_x and A
        av(a[next], b[next], av_data, t);
      }

      if( current    == 0 ) {    current = 1;    next = 0; } else { current = 0; next = 1; }
      if( current_hs == 2 ) { current_hs = 3; next_hs = 2; } else { current_hs = 2; next_hs = 3; }

      if( display == 7 && frame_time >= 0.01 ) { // we are making movie
        sprintf(file_name_buf, "frame%08d.data", frame_number++);
        FILE *frame_file_stream = fopen(file_name_buf, "w");
        setvbuf(frame_file_stream, buf, _IOFBF, sizeof(buf));
        printf("\nWriting frame %s\n", file_name_buf);
        print_2d_data(frame_file_stream, MSIZE, a0, a[current], b[current], alpha);
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
      frame_time += dt;
    }

    ffloat norm = eval_norm(a[current], alpha, MSIZE);

    if( display == 3 ) {
      for( ffloat phi_x = -PI; phi_x < PI; phi_x += 0.01 ) {
        for( int m = 1; m < M; m++ ) {
          ffloat value = 0;
          ffloat value0 = 0;
          for( int n = 0; n < N+1; n++ ) {
            value  += nm(a[current],n,m)*cos(n*phi_x) + nm(b[current],n,m)*sin(n*phi_x);
            value0 += nm(a0,n,m)*cos(n*phi_x);
          }
          fprintf(out, "%0.5f %0.5f %0.20f %0.20f\n", phi_x, phi_y(m), value<0?0:value, value0<0?0:value0);
        }
      }
      fprintf(out, "# norm=%0.20f\n", norm);
      printf("# norm=%0.20f\n", norm);
      return EXIT_SUCCESS;
    }

    if( display == 4 ) {
      printf("\n# norm=%0.20f\n", norm);
      ffloat v_dr_inst = 0 ;
      ffloat v_y_inst = 0;
      ffloat m_over_m_x_inst = 0;
      for( int m = 1; m < M; m++ ) {
        v_dr_inst += nm(b[current],1,m)*dPhi;
        v_y_inst  += nm(a[current],0,m)*phi_y(m)*dPhi;
        m_over_m_x_inst += nm(a[current],1,m)*dPhi;
      }

      ffloat v_dr_multiplier = 2*gsl_sf_bessel_I0(mu)*PI*sqrt(alpha)/gsl_sf_bessel_In(1, mu);
      ffloat v_y_multiplier  = 4*PI*gsl_sf_bessel_I0(mu)/gsl_sf_bessel_In(1, mu);
      ffloat m_over_multiplier = PI*alpha*sqrt(alpha);
      v_dr_inst       *= v_dr_multiplier;
      v_y_inst        *= v_y_multiplier;
      m_over_m_x_inst *= m_over_multiplier;

      av_data[1] *= v_dr_multiplier;
      av_data[2] *= v_y_multiplier;
      av_data[3] *= m_over_multiplier;
      av_data[4] *= v_dr_multiplier;
      av_data[4] /= T;
      av_data[5] *= v_dr_multiplier;
      av_data[5] /= T;

      fprintf(out, "# display=%d E_dc=%0.20f E_omega=%0.20f omega=%0.20f mu=%0.20f alpha=%0.20f n-harmonics=%d PhiYmin=%0.20f PhiYmax=%0.20f B=%0.20f t-max=%0.20f dt=%0.20f g-grid=%d\n",
                      display,   E_dc,  E_omega,  omega,  mu,  alpha,  N,        PhiYmin,       PhiYmax,       B,  t_start,     dt,  M);
      fprintf(out, "#E_{dc}                \\tilde{E}_{\\omega}     \\tilde{\\omega}         mu                     v_{dr}/v_{p}         A(\\omega)              NORM     v_{y}/v_{p}    m/m_{x,k}   <v_{dr}/v_{p}>   <v_{y}/v_{p}>    <m/m_{x,k}>    Asin\n");
      fprintf(out, "%0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f\n", 
              E_dc, E_omega, omega, mu, v_dr_inst, av_data[4], norm, v_y_inst, 
              m_over_m_x_inst, av_data[1], av_data[2], av_data[3], av_data[5]);
    }

    if( read_from == NULL ) { break; }

    // scan for new parameters
    timeout = scan_for_new_parameters();
    if( timeout < -900 ) { break; } // user entered 'exit'
    t_start = t + timeout;
    t_max = t_start + T;
    t0 = t + dt;
    T=omega>0?(2*PI/omega):0;
    memset(av_data, 0, 6*sizeof(ffloat)); // clear averaging data
    printf("# t_max = %0.20f\n", t_max);
  } // for(;;)

  if( out != NULL && out != stdout ) {
    fclose(out);
  }
  return EXIT_SUCCESS;
} // end of main(...)

ffloat eval_norm(ffloat *a, ffloat alpha, int MSIZE) {
  ffloat norm = 0;
  for( int m = 1; m < M+1; m++ ) {
    norm += nm(a,0,m)*dPhi;
  }
  norm *= 2*PI*sqrt(alpha);
  return norm;
} // end of eval_norm(...)

void print_time_evolution_of_parameters(FILE *out, ffloat norm, ffloat *a, ffloat *b, int MSIZE, 
                                        ffloat mu, ffloat alpha, ffloat E_dc, ffloat E_omega, 
                                        ffloat omega, ffloat *av_data, ffloat t) 
{
  printf("\n# t=%0.20f norm=%0.20f\n", t, norm);
  ffloat v_dr_inst = 0 ;
  ffloat v_y_inst = 0;
  ffloat m_over_m_x_inst = 0;
  for( int m = 1; m < 2*M+2; m++ ) {
    v_dr_inst += nm(b,1,m)*dPhi;
    v_y_inst  += nm(a,0,m)*phi_y(m)*dPhi;
    m_over_m_x_inst += nm(a,1,m)*dPhi;
  }

  ffloat v_dr_multiplier = 2*gsl_sf_bessel_I0(mu)*PI*sqrt(alpha)/gsl_sf_bessel_In(1, mu);
  ffloat v_y_multiplier  = 4*PI*gsl_sf_bessel_I0(mu)/gsl_sf_bessel_In(1, mu);
  ffloat m_over_multiplier = PI*alpha*sqrt(alpha);
  v_dr_inst       *= v_dr_multiplier;
  v_y_inst        *= v_y_multiplier;
  m_over_m_x_inst *= m_over_multiplier;
  
  av_data[1] *= v_dr_multiplier;
  av_data[2] *= v_y_multiplier;
  av_data[3] *= m_over_multiplier;
  av_data[4] *= v_dr_multiplier;
  av_data[4] /= t;
  av_data[5] *= v_dr_multiplier;
  av_data[5] /= t;
  
  fprintf(out, "#E_{dc}                \\tilde{E}_{\\omega}     \\tilde{\\omega}         mu                     v_{dr}/v_{p}         A(\\omega)              NORM     v_{y}/v_{p}    m/m_{x,k}   <v_{dr}/v_{p}>   <v_{y}/v_{p}>    <m/m_{x,k}>  A_{inst}  t    Asin\n");
  fprintf(out, "%0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f %0.20f\n", 
          E_dc, E_omega, omega, mu, v_dr_inst, av_data[4], norm, v_y_inst, 
          m_over_m_x_inst, av_data[1], av_data[2], av_data[3], cos(omega*t)*v_dr_inst, t, av_data[4]);
} // end of print_time_evolution_of_parameters(...)

// Write out distribution function to FILE *out. This used when writing data to generate animation
void print_2d_data(FILE *out, int MSIZE, ffloat *a0, ffloat *a, ffloat *b, ffloat alpha) {
  ffloat norm = 0;
  for( int m = 1; m < 2*M+2; m++ ) {
    norm += nm(a,0,m)*dPhi;
  }
  norm *= 2*PI*sqrt(alpha);

  for( ffloat phi_x = -PI; phi_x < PI; phi_x += 0.01 ) {
    for( int m = 1; m < 2*M+2; m++ ) {
      ffloat value = 0;
      //ffloat value0 = 0;
      for( int n = 0; n < N+1; n++ ) {
        value  += nm(a,n,m)*cos(n*phi_x) + nm(b,n,m)*sin(n*phi_x);
      }
      fprintf(out, "%0.5f %0.5f %0.20f\n", phi_x, phi_y(m), value<0?0:value);
    }
  }
  fprintf(out, "# norm=%0.20f\n", norm);
  printf("# norm=%0.20f\n", norm);
} // end of solve(...)

void step_on_grid(ffloat *a0,           ffloat *a_current,   ffloat *b_current,    
                  ffloat *a_next,       ffloat *b_next,      ffloat *a_current_hs, 
                  ffloat *b_current_hs, ffloat t,            ffloat t_hs, 
                  ffloat cos_omega_t,   ffloat cos_omega_t_plus_dt) 
{
  #pragma omp parallel for
  for( int m = 1; m <= TMSIZE; m++ ) {
    // step from (t,t+1/2) to (t+1)
    ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*phi_y(m))*dt/2;
    ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*phi_y(m))*dt/2;
    
    for( int n = 0; n < N; n++ ) {
      ffloat mu_t = n*mu_t_part;
      ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
      ffloat g = dt*nm(a0,n,m)+nm(a_current,n,m)*nu_tilde-nm(b_current,n,m)*mu_t + 
        bdt*( nm(b_current_hs,n+1,m+1) - nm(b_current_hs,n+1,m-1) - (n < 2 ? 0 : (nm(b_current_hs,n-1,m+1) - nm(b_current_hs,n-1,m-1))) );
      ffloat h = nm(b_current,n,m)*nu_tilde+nm(a_current,n,m)*mu_t + 
        bdt*( (n==1?2:1)*(n==0?0:(nm(a_current_hs,n-1,m+1)-nm(a_current_hs,n-1,m-1))) - nm(a_current_hs,n+1,m+1) + nm(a_current_hs,n+1,m-1) );
      
      ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
      nm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
      if( n > 0 ) {
        nm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
      }
    } 
  }
  #pragma end parallel for

} // end of step on grid(...)

void step_on_half_grid(ffloat *a0, ffloat *a_current,    ffloat *b_current, 
                       ffloat *a_next,       ffloat *b_next, 
                       ffloat *a_current_hs, ffloat *b_current_hs, 
                       ffloat *a_next_hs, ffloat *b_next_hs, 
                       ffloat t, ffloat t_hs, ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  #pragma omp parallel for
  for( int m = 1; m < TMSIZE; m++ ) {
    // step from (t+1/2,t+1) to (t+3/2)
    ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*phi_y(m))*dt/2;
    ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*phi_y(m))*dt/2;

    for( int n = 0; n < N; n++ ) {
      ffloat mu_t = n*mu_t_part;
      ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
      ffloat g = dt*nm(a0,n,m)+nm(a_current_hs,n,m)*nu_tilde-nm(b_current_hs,n,m)*mu_t +
        bdt*( nm(b_next,n+1,m+1) - nm(b_next,n+1,m-1) - (n < 2 ? 0 : (nm(b_next,n-1,m+1) - nm(b_next,n-1,m-1))) );
      ffloat h = nm(b_current_hs,n,m)*nu_tilde+nm(a_current_hs,n,m)*mu_t +
        bdt*( (n==1?2:1)*(n==0?0:(nm(a_next,n-1,m+1)-nm(a_next,n-1,m-1))) - nm(a_next,n+1,m+1) + nm(a_next,n+1,m-1) );
      ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
      nm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
      if( n > 0 ) {
        nm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
      }
    }
  }
  #pragma end parallel for

} // end of step_on_half_grid(...)

void av(ffloat *a, ffloat *b, ffloat *av_data, ffloat t) {
  int av_count = av_data[0] + 1; 

  ffloat v_dr_inst = 0; ffloat v_y_inst = 0; ffloat m_over_m_x_inst = 0;
  for( int m = 1; m < TMSIZE; m++ ) {
    v_dr_inst += nm(b,1,m)*dPhi;
    v_y_inst  += nm(a,0,m)*phi_y(m)*dPhi;
    m_over_m_x_inst += nm(a,1,m)*dPhi;
  }

  //v_dr_av = v_dr_av+(v_dr_inst-v_dr_av)/av_count;
  av_data[1] += (v_dr_inst-av_data[1])/av_count; // av_data[1] holds v_dr_av

  //v_y_av = v_y_av+(v_y_inst-v_y_av)/av_count;
  av_data[2] += (v_y_inst-av_data[2])/av_count; // av_data[2] holds v_y_av

  //m_over_m_x_av = m_over_m_x_av+(m_over_m_x_inst-m_over_m_x_av)/av_count;
  av_data[3] += (m_over_m_x_inst-av_data[3])/av_count; // av_data[3] holds m_over_m_x_av

  //A += cos(omega*t)*v_dr_inst*dt;
  av_data[4] += cos(omega*t)*v_dr_inst*dt; // av_data[4] holds absorption A
  av_data[5] += sin(omega*t)*v_dr_inst*dt; // av_data[4] holds sin absorption A

  av_data[0] += 1;
} // enf of av(...)
