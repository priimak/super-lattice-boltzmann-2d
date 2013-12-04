#include <stdio.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_specfunc.h>
#include "boltzmann.h"

#define PPP 64

/**
   --------------------------------------------------------------------------------
   Kernel#  | Description
   --------------------------------------------------------------------------------
   1        | One thread per point. Not using shared memory.
   2        | One thread per point using shared memory.
   310      | One thread per m-number. Not unrolled.
   311      | One thread per m-number. Not unrolled. Removed divergent flows by taking 
            | calculations at n=0 and n=1 out of the loops.
   321      | One thread per m-number. Unrolled twice. Divergent flows removed.
   341      | One thread per m-number. Unrolled four times. Divergent flows removed.
   342      | One thread per m-number. Unrolled four times. Divergent flows removed. Elements partially reused.
   4        | One thread per m-number. Loops are staggered and elemenets reused. Not unrolled.
   --------------------------------------------------------------------------------
**/

extern "C"
void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString( err ),
           file, line );
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

extern ffloat host_E_dc, host_E_omega, host_omega, host_mu, host_alpha,
  PhiYmin, PhiYmax, host_B, t_start, host_dPhi, host_dt,
  host_bdt, host_nu_tilde, host_nu2, host_nu;

extern int host_M, host_N, MSIZE, MP1, NSIZE, host_TMSIZE, PADDED_MSIZE;

__constant__ ffloat E_dc, E_omega, omega, B, dt, dPhi, nu, nu2, nu_tilde, bdt, mu, alpha, dev_PhiYmin;
__constant__ int M, N, dev_MSIZE, DPADDED_MSIZE, TMSIZE, dev_NSIZE;

#define dnm(pointer, n, m) (*((pointer)+(n)*DPADDED_MSIZE+(m)))
//#define dev_phi_y(m) (dPhi*((m)-M-1))
#define dev_phi_y(m) (dev_PhiYmin+dPhi*((m)-1))

dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
dim3 dimGrid;

// load data into symbol table
extern "C"
void load_data(void) {
  HANDLE_ERROR(cudaMemcpyToSymbol(E_dc,        &host_E_dc,     sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(E_omega,     &host_E_omega,  sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(omega,       &host_omega,    sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(B,           &host_B,        sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(dt,          &host_dt,       sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(M,           &host_M,        sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(N,           &host_N,        sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(dPhi,        &host_dPhi,     sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(mu,          &host_mu,       sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(alpha,       &host_alpha,    sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(dev_MSIZE,   &MSIZE,         sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(dev_NSIZE,   &NSIZE,         sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(TMSIZE,      &host_TMSIZE,   sizeof(int)));
  HANDLE_ERROR(cudaMemcpyToSymbol(DPADDED_MSIZE, &PADDED_MSIZE,   sizeof(int)));

  HANDLE_ERROR(cudaMemcpyToSymbol(bdt,         &host_bdt,      sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(nu_tilde,    &host_nu_tilde, sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(nu2,         &host_nu2,      sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(nu,          &host_nu,       sizeof(ffloat)));
  HANDLE_ERROR(cudaMemcpyToSymbol(dev_PhiYmin, &PhiYmin,       sizeof(ffloat)));

  dimGrid.x = (NSIZE+BLOCK_SIZE)/BLOCK_SIZE;
  dimGrid.y = (MP1+BLOCK_SIZE)/BLOCK_SIZE;
} // end of load_data()

/** BEGINING OF K4 **/ // KNOWN GOLDEN CODE
/* One thread per m-number. Reusing elements cached in registers. Not using shared memory. */
__global__ void _step_on_grid_k4(ffloat *a0,           
				 ffloat *a_current,    ffloat *b_current,
				 ffloat *a_next,       ffloat *b_next,
				 ffloat *a_current_hs, ffloat *b_current_hs,
				 ffloat t,             ffloat t_hs,
				 ffloat cos_omega_t,   ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t,t+1/2) to (t+1)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;
  ffloat b_current_hs_n_minus_1_m_plus_1  = 0;
  ffloat b_current_hs_n_minus_1_m_minus_1 = 0;
  ffloat a_current_hs_n_minus_1_m_plus_1  = 0;
  ffloat a_current_hs_n_minus_1_m_minus_1 = 0;
  ffloat b_current_hs_n_plus_1_m_plus_1;
  ffloat b_current_hs_n_plus_1_m_minus_1;
  ffloat a_current_hs_n_plus_1_m_plus_1;
  ffloat a_current_hs_n_plus_1_m_minus_1;
  for( int n = 0; n < N; n += 2 ) {
    ffloat a_center = dnm(a_current,n,m);
    ffloat b_center = dnm(b_current,n,m);
    ffloat mu_t = n*mu_t_part;
    ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
    b_current_hs_n_plus_1_m_plus_1  = dnm(b_current_hs,n+1,m+1);
    b_current_hs_n_plus_1_m_minus_1 = dnm(b_current_hs,n+1,m-1);
    a_current_hs_n_plus_1_m_plus_1  = dnm(a_current_hs,n+1,m+1);
    a_current_hs_n_plus_1_m_minus_1 = dnm(a_current_hs,n+1,m-1);

    ffloat g = dt*dnm(a0,n,m)+a_center*nu_tilde-b_center*mu_t +
      bdt*( b_current_hs_n_plus_1_m_plus_1 - b_current_hs_n_plus_1_m_minus_1 - 
	    b_current_hs_n_minus_1_m_plus_1 + b_current_hs_n_minus_1_m_minus_1 );

    ffloat h = b_center*nu_tilde+a_center*mu_t +
      bdt*( a_current_hs_n_minus_1_m_plus_1 - a_current_hs_n_minus_1_m_minus_1 - 
	    a_current_hs_n_plus_1_m_plus_1 + a_current_hs_n_plus_1_m_minus_1 );

    ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    if( n > 0 ) {
       dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
    }
    b_current_hs_n_minus_1_m_plus_1  = b_current_hs_n_plus_1_m_plus_1;
    b_current_hs_n_minus_1_m_minus_1 = b_current_hs_n_plus_1_m_minus_1;
    a_current_hs_n_minus_1_m_plus_1  = a_current_hs_n_plus_1_m_plus_1;
    a_current_hs_n_minus_1_m_minus_1 = a_current_hs_n_plus_1_m_minus_1;
  }
  b_current_hs_n_minus_1_m_plus_1  = 0;
  b_current_hs_n_minus_1_m_minus_1 = 0;
  a_current_hs_n_minus_1_m_plus_1  = 2*dnm(a_current_hs,0,m+1);
  a_current_hs_n_minus_1_m_minus_1 = 2*dnm(a_current_hs,0,m-1);
  for( int n = 1; n < N; n += 2 ) {
    ffloat a_center = dnm(a_current,n,m);
    ffloat b_center = dnm(b_current,n,m);
    ffloat mu_t = n*mu_t_part;
    ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
    b_current_hs_n_plus_1_m_plus_1  = dnm(b_current_hs,n+1,m+1);
    b_current_hs_n_plus_1_m_minus_1 = dnm(b_current_hs,n+1,m-1);
    a_current_hs_n_plus_1_m_plus_1  = dnm(a_current_hs,n+1,m+1);
    a_current_hs_n_plus_1_m_minus_1 = dnm(a_current_hs,n+1,m-1);

    ffloat g = dt*dnm(a0,n,m)+a_center*nu_tilde-b_center*mu_t +
      bdt*( b_current_hs_n_plus_1_m_plus_1 - b_current_hs_n_plus_1_m_minus_1 - 
	    b_current_hs_n_minus_1_m_plus_1 + b_current_hs_n_minus_1_m_minus_1);

    ffloat h = b_center*nu_tilde+a_center*mu_t +
      bdt*( a_current_hs_n_minus_1_m_plus_1 - a_current_hs_n_minus_1_m_minus_1 - 
	    a_current_hs_n_plus_1_m_plus_1 + a_current_hs_n_plus_1_m_minus_1 );

    ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
    b_current_hs_n_minus_1_m_plus_1  = b_current_hs_n_plus_1_m_plus_1;
    b_current_hs_n_minus_1_m_minus_1 = b_current_hs_n_plus_1_m_minus_1;
    a_current_hs_n_minus_1_m_plus_1  = a_current_hs_n_plus_1_m_plus_1;
    a_current_hs_n_minus_1_m_minus_1 = a_current_hs_n_plus_1_m_minus_1;
  }
} // end of _step_on_grid_k4(...)

__global__ void _step_on_half_grid_k4(ffloat *a0, 
				      ffloat *a_current,    ffloat *b_current,
				      ffloat *a_next,       ffloat *b_next,
				      ffloat *a_current_hs, ffloat *b_current_hs,
				      ffloat *a_next_hs,    ffloat *b_next_hs,
				      ffloat t,             ffloat t_hs,
				      ffloat cos_omega_t,   ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t+1/2,t+1) to (t+3/2)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;
  ffloat b_next_n_minus_1_m_plus_1  = 0;
  ffloat b_next_n_minus_1_m_minus_1 = 0;
  ffloat a_next_n_minus_1_m_plus_1  = 0;
  ffloat a_next_n_minus_1_m_minus_1 = 0;
  ffloat b_next_n_plus_1_m_plus_1;
  ffloat b_next_n_plus_1_m_minus_1;
  ffloat a_next_n_plus_1_m_plus_1;
  ffloat a_next_n_plus_1_m_minus_1;

  for( int n = 0; n < N; n += 2 ) {
    ffloat mu_t = n*mu_t_part;
    ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat a_center = dnm(a_current_hs,n,m);
    ffloat b_center = dnm(b_current_hs,n,m);
    b_next_n_plus_1_m_plus_1  = dnm(b_next,n+1,m+1);
    b_next_n_plus_1_m_minus_1 = dnm(b_next,n+1,m-1);
    a_next_n_plus_1_m_plus_1  = dnm(a_next,n+1,m+1);
    a_next_n_plus_1_m_minus_1 = dnm(a_next,n+1,m-1);

    ffloat g = dt*dnm(a0,n,m)+a_center*nu_tilde-b_center*mu_t +
      bdt*( b_next_n_plus_1_m_plus_1 - b_next_n_plus_1_m_minus_1 - 
	    b_next_n_minus_1_m_plus_1 + b_next_n_minus_1_m_minus_1 );

    ffloat h = b_center*nu_tilde+a_center*mu_t +
      bdt*( a_next_n_minus_1_m_plus_1-a_next_n_minus_1_m_minus_1 - 
	    a_next_n_plus_1_m_plus_1 + a_next_n_plus_1_m_minus_1 );

    ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    if( n > 0 ) {
      dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
    }
    b_next_n_minus_1_m_plus_1  = b_next_n_plus_1_m_plus_1;
    b_next_n_minus_1_m_minus_1 = b_next_n_plus_1_m_minus_1;
    a_next_n_minus_1_m_plus_1  = a_next_n_plus_1_m_plus_1;
    a_next_n_minus_1_m_minus_1 = a_next_n_plus_1_m_minus_1;
  }

  b_next_n_minus_1_m_plus_1  = 0;
  b_next_n_minus_1_m_minus_1 = 0;
  a_next_n_minus_1_m_plus_1  = 2*dnm(a_next,0,m+1);
  a_next_n_minus_1_m_minus_1 = 2*dnm(a_next,0,m-1);
  for( int n = 1; n < N; n += 2 ) {
    ffloat mu_t = n*mu_t_part;
    ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat a_center = dnm(a_current_hs,n,m);
    ffloat b_center = dnm(b_current_hs,n,m);
    b_next_n_plus_1_m_plus_1  = dnm(b_next,n+1,m+1);
    b_next_n_plus_1_m_minus_1 = dnm(b_next,n+1,m-1);
    a_next_n_plus_1_m_plus_1  = dnm(a_next,n+1,m+1);
    a_next_n_plus_1_m_minus_1 = dnm(a_next,n+1,m-1);

    ffloat g = dt*dnm(a0,n,m)+a_center*nu_tilde-b_center*mu_t +
      bdt*( b_next_n_plus_1_m_plus_1 - b_next_n_plus_1_m_minus_1 - 
	    b_next_n_minus_1_m_plus_1 + b_next_n_minus_1_m_minus_1 );

    ffloat h = b_center*nu_tilde+a_center*mu_t +
      bdt*( a_next_n_minus_1_m_plus_1-a_next_n_minus_1_m_minus_1 - 
	    a_next_n_plus_1_m_plus_1 + a_next_n_plus_1_m_minus_1 );
    ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
    b_next_n_minus_1_m_plus_1  = b_next_n_plus_1_m_plus_1;
    b_next_n_minus_1_m_minus_1 = b_next_n_plus_1_m_minus_1;
    a_next_n_minus_1_m_plus_1  = a_next_n_plus_1_m_plus_1;
    a_next_n_minus_1_m_minus_1 = a_next_n_plus_1_m_minus_1;
  }
} // end of _step_on_half_grid_k4(...)
/** END OF K4 **/ // KNOWN GOLDEN CODE

/** BEGINING OF K2 KERNELS **/
/* One thread per point using shared memory. */
__global__ void _step_on_grid_k2(ffloat *a0, 
				 ffloat *a_current,    ffloat *b_current,
				 ffloat *a_next,       ffloat *b_next,
				 ffloat *a_current_hs, ffloat *b_current_hs,
				 ffloat t,             ffloat t_hs,
				 ffloat cos_omega_t,   ffloat cos_omega_t_plus_dt)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if( m > TMSIZE || n >= N ) { return; }

  __shared__ ffloat a_c[(BLOCK_SIZE+2)*(BLOCK_SIZE+2)];
  __shared__ ffloat b_c[(BLOCK_SIZE+2)*(BLOCK_SIZE+2)];
  a_c[BLOCK_SIZE*threadIdx.x+threadIdx.y] = dnm(a_current_hs,n,m);
  b_c[BLOCK_SIZE*threadIdx.x+threadIdx.y] = dnm(b_current_hs,n,m);
  __syncthreads();

  // step from (t,t+1/2) to (t+1)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = n*mu_t_part;
  ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
  
  ffloat g = dt*dnm(a0,n,m)+dnm(a_current,n,m)*nu_tilde-dnm(b_current,n,m)*mu_t +
    bdt*( ((threadIdx.x<BLOCK_SIZE_M1 && threadIdx.y<BLOCK_SIZE_M1)?b_c[BLOCK_SIZE*(threadIdx.x+1)+threadIdx.y+1]:dnm(b_current_hs,n+1,m+1))- 
	  ((threadIdx.x<BLOCK_SIZE_M1 && threadIdx.y!=0)?b_c[BLOCK_SIZE*(threadIdx.x+1)+threadIdx.y-1]:dnm(b_current_hs,n+1,m-1)) - 
	  (n < 2 ? 0 : (
			((threadIdx.x!=0 && threadIdx.y!=BLOCK_SIZE_M1)?b_c[BLOCK_SIZE*(threadIdx.x-1)+threadIdx.y+1]:dnm(b_current_hs,n-1,m+1)) - 
			((threadIdx.x!=0 && threadIdx.y!=0)?b_c[BLOCK_SIZE*(threadIdx.x-1)+threadIdx.y-1]:dnm(b_current_hs,n-1,m-1))
			)) );

  ffloat h = dnm(b_current,n,m)*nu_tilde+dnm(a_current,n,m)*mu_t +
    bdt*( (n==1?2:1)*(n==0?0:(
			      ((threadIdx.x!=0 && threadIdx.y!=BLOCK_SIZE_M1)?a_c[BLOCK_SIZE*(threadIdx.x-1)+threadIdx.y+1]:dnm(a_current_hs,n-1,m+1)) -
			      ((threadIdx.x!=0 && threadIdx.y!=0)?a_c[BLOCK_SIZE*(threadIdx.x-1)+threadIdx.y-1]:dnm(a_current_hs,n-1,m-1))
			      )) - 
	  ((threadIdx.x<BLOCK_SIZE_M1 && threadIdx.y<BLOCK_SIZE_M1)?a_c[BLOCK_SIZE*(threadIdx.x+1)+threadIdx.y+1]:dnm(a_current_hs,n+1,m+1)) + 
	  ((threadIdx.x<BLOCK_SIZE_M1 && threadIdx.y!=0)?a_c[BLOCK_SIZE*(threadIdx.x+1)+threadIdx.y-1]:dnm(a_current_hs,n+1,m-1)));

    ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    if( n > 0 ) {
      dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
    }
} // end of _step_on_grid_k2(...)

__global__ void _step_on_half_grid_k2(ffloat *a0, ffloat *a_current,    ffloat *b_current,
                                  ffloat *a_next,       ffloat *b_next,
                                  ffloat *a_current_hs, ffloat *b_current_hs,
                                  ffloat *a_next_hs,    ffloat *b_next_hs,
                                  ffloat t, ffloat t_hs,
                                  ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if( m > TMSIZE || n >= N ) { return; }

  __shared__ ffloat a_c[(BLOCK_SIZE+2)*(BLOCK_SIZE+2)];
  __shared__ ffloat b_c[(BLOCK_SIZE+2)*(BLOCK_SIZE+2)];
  a_c[BLOCK_SIZE*threadIdx.x+threadIdx.y] = dnm(a_next,n,m);
  b_c[BLOCK_SIZE*threadIdx.x+threadIdx.y] = dnm(b_next,n,m);
  __syncthreads();

  // step from (t+1/2,t+1) to (t+3/2)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;
  ffloat mu_t = n*mu_t_part;
  ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
  ffloat g = dt*dnm(a0,n,m)+dnm(a_current_hs,n,m)*nu_tilde-dnm(b_current_hs,n,m)*mu_t +
    
    bdt*( ((threadIdx.x<BLOCK_SIZE_M1 && threadIdx.y<BLOCK_SIZE_M1)?b_c[BLOCK_SIZE*(threadIdx.x+1)+threadIdx.y+1]:dnm(b_next,n+1,m+1))- 
	  ((threadIdx.x<BLOCK_SIZE_M1 && threadIdx.y!=0)?b_c[BLOCK_SIZE*(threadIdx.x+1)+threadIdx.y-1]:dnm(b_next,n+1,m-1)) - 
	  (n < 2 ? 0 : (
			((threadIdx.x!=0 && threadIdx.y!=BLOCK_SIZE_M1)?b_c[BLOCK_SIZE*(threadIdx.x-1)+threadIdx.y+1]:dnm(b_next,n-1,m+1)) - 
			((threadIdx.x!=0 && threadIdx.y!=0)?b_c[BLOCK_SIZE*(threadIdx.x-1)+threadIdx.y-1]:dnm(b_next,n-1,m-1))
			)) );
  
  
  ffloat h = dnm(b_current_hs,n,m)*nu_tilde+dnm(a_current_hs,n,m)*mu_t +
    bdt*( (n==1?2:1)*(n==0?0:(
			      ((threadIdx.x!=0 && threadIdx.y!=BLOCK_SIZE_M1)?a_c[BLOCK_SIZE*(threadIdx.x-1)+threadIdx.y+1]:dnm(a_next,n-1,m+1)) -
			      ((threadIdx.x!=0 && threadIdx.y!=0)?a_c[BLOCK_SIZE*(threadIdx.x-1)+threadIdx.y-1]:dnm(a_next,n-1,m-1))
			      )) - 
	  ((threadIdx.x<BLOCK_SIZE_M1 && threadIdx.y<BLOCK_SIZE_M1)?a_c[BLOCK_SIZE*(threadIdx.x+1)+threadIdx.y+1]:dnm(a_next,n+1,m+1)) + 
	  ((threadIdx.x<BLOCK_SIZE_M1 && threadIdx.y!=0)?a_c[BLOCK_SIZE*(threadIdx.x+1)+threadIdx.y-1]:dnm(a_next,n+1,m-1)));
  
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
  if( n > 0 ) {
    dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
  }
} // end of _step_on_half_grid_k2(...)
/** END OF K2 KERNELS **/

/** BEGINING OF K1 KERNELS **/
/* One thread per point. Not using shared memory. */
__global__ void _step_on_grid_k1(ffloat *a0, ffloat *a_current,    ffloat *b_current,
				 ffloat *a_next,       ffloat *b_next,
				 ffloat *a_current_hs, ffloat *b_current_hs,
				 ffloat t, ffloat t_hs,
				 ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if( m > TMSIZE || n >= N ) { return; }

  // step from (t,t+1/2) to (t+1)
  ffloat mu_t_part	  = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t		  =  n*mu_t_part;
  ffloat mu_t_plus_1	  = n*mu_t_plus_1_part;

  ffloat g = dt*dnm(a0,n,m)+dnm(a_current,n,m)*nu_tilde-dnm(b_current,n,m)*mu_t +
    bdt*( dnm(b_current_hs,n+1,m+1) - dnm(b_current_hs,n+1,m-1) - 
	  (n < 2 ? 0 : (dnm(b_current_hs,n-1,m+1) - dnm(b_current_hs,n-1,m-1))) );

  ffloat h = dnm(b_current,n,m)*nu_tilde+dnm(a_current,n,m)*mu_t +
    bdt*( (n==1?2:1)*(n==0?0:(dnm(a_current_hs,n-1,m+1)-dnm(a_current_hs,n-1,m-1))) - 
	  dnm(a_current_hs,n+1,m+1) + dnm(a_current_hs,n+1,m-1) );
  
    ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    if( n > 0 ) {
      dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
    }
} // end of _step_on_grid_k1(...)

__global__ void _step_on_half_grid_k1(ffloat *a0, ffloat *a_current,    ffloat *b_current,
				      ffloat *a_next,       ffloat *b_next,
				      ffloat *a_current_hs, ffloat *b_current_hs,
				      ffloat *a_next_hs,    ffloat *b_next_hs,
				      ffloat t, ffloat t_hs,
				      ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if( m > TMSIZE || n >= N ) { return; }

  // step from (t+1/2,t+1) to (t+3/2)
  ffloat mu_t_part	  = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;
  ffloat mu_t		  = n*mu_t_part;
  ffloat mu_t_plus_1	  = n*mu_t_plus_1_part;

  ffloat g = dt*dnm(a0,n,m)+dnm(a_current_hs,n,m)*nu_tilde-dnm(b_current_hs,n,m)*mu_t +
    bdt*( dnm(b_next,n+1,m+1) - dnm(b_next,n+1,m-1) - 
	  (n < 2 ? 0 : (dnm(b_next,n-1,m+1) - dnm(b_next,n-1,m-1))) );

  ffloat h = dnm(b_current_hs,n,m)*nu_tilde+dnm(a_current_hs,n,m)*mu_t +
    bdt*( (n==1?2:1)*(n==0?0:(dnm(a_next,n-1,m+1)-dnm(a_next,n-1,m-1))) - 
	  dnm(a_next,n+1,m+1) + dnm(a_next,n+1,m-1) );

  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
  if( n > 0 ) {
    dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
  }
} // end of _step_on_half_grid_k1(...)
/** END OF K1 KERNELS **/

/** BEGINING OF 310 KERNELS **/
/* One thread per m-number. Not unrolled. Kernel 310 */
__global__ void _step_on_grid_k3_unroll_1_type_0
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t,t+1/2) to (t+1)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  for( int n = 0; n < N; n++ ) {
    ffloat mu_t = n*mu_t_part;
    ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat g = dt*dnm(a0,n,m)+dnm(a_current,n,m)*nu_tilde-dnm(b_current,n,m)*mu_t +
      bdt*( dnm(b_current_hs,n+1,m+1) - dnm(b_current_hs,n+1,m-1) - (n < 2 ? 0 : (dnm(b_current_hs,n-1,m+1) - dnm(b_current_hs,n-1,m-1))) );
    ffloat h = dnm(b_current,n,m)*nu_tilde+dnm(a_current,n,m)*mu_t +
      bdt*( (n==1?2:1)*(n==0?0:(dnm(a_current_hs,n-1,m+1)-dnm(a_current_hs,n-1,m-1))) - dnm(a_current_hs,n+1,m+1) + dnm(a_current_hs,n+1,m-1) );

    ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    if( n > 0 ) {
      dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
    }
  }
} // end of _step_on_grid_k3_unroll_1_type_0(...)

__global__ void _step_on_half_grid_k3_unroll_1_type_0
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat *a_next_hs,    ffloat *b_next_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t+1/2,t+1) to (t+3/2)
  ffloat mu_t_part        = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;
  for( int n = 0; n < N; n++ ) {
    ffloat mu_t = n*mu_t_part;
    ffloat mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat g = dt*dnm(a0,n,m)+dnm(a_current_hs,n,m)*nu_tilde-dnm(b_current_hs,n,m)*mu_t +
      bdt*( dnm(b_next,n+1,m+1) - dnm(b_next,n+1,m-1) - (n < 2 ? 0 : (dnm(b_next,n-1,m+1) - dnm(b_next,n-1,m-1))) );
    ffloat h = dnm(b_current_hs,n,m)*nu_tilde+dnm(a_current_hs,n,m)*mu_t +
      bdt*( (n==1?2:1)*(n==0?0:(dnm(a_next,n-1,m+1)-dnm(a_next,n-1,m-1))) - dnm(a_next,n+1,m+1) + dnm(a_next,n+1,m-1) );
    ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    if( n > 0 ) {
      dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
    }
  }
} // end of _step_on_half_grid_k3_unroll_1_type_0(...)
/** END OF 310 KERNELS **/

/** BEGINING OF 311 KERNELS **/
/**
 * One thread per m-number. Not unrolled. Removed divergent flows by taking calculations 
 * at n=0 and n=1 out of the loops. Kernel 311
 */
__global__ void _step_on_grid_k3_unroll_1_type_1
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t,t+1/2) to (t+1)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = 0;
  ffloat mu_t_plus_1 = 0;
  ffloat g = dt*dnm(a0,0,m)+dnm(a_current,0,m)*nu_tilde-dnm(b_current,0,m)*mu_t +
    bdt*( dnm(b_current_hs,1,m+1) - dnm(b_current_hs,1,m-1) );
  ffloat h = dnm(b_current,0,m)*nu_tilde+dnm(a_current,0,m)*mu_t +
    bdt*( - dnm(a_current_hs,1,m+1) + dnm(a_current_hs,1,m-1) );
  
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next,0,m) = (g*nu - h*mu_t_plus_1)/xi;

  mu_t = mu_t_part;
  mu_t_plus_1 = mu_t_plus_1_part;
  g = dt*dnm(a0,1,m)+dnm(a_current,1,m)*nu_tilde-dnm(b_current,1,m)*mu_t +
    bdt*( dnm(b_current_hs,2,m+1) - dnm(b_current_hs,2,m-1) );
  h = dnm(b_current,1,m)*nu_tilde+dnm(a_current,1,m)*mu_t +
    bdt*( 2*((dnm(a_current_hs,0,m+1)-dnm(a_current_hs,0,m-1))) - dnm(a_current_hs,2,m+1) + dnm(a_current_hs,2,m-1) );
  
  xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next,1,m) = (g*nu - h*mu_t_plus_1)/xi;
  dnm(b_next,1,m) = (g*mu_t_plus_1 + h*nu)/xi;

  for( int n = 2; n < N; n++ ) {
    mu_t = n*mu_t_part;
    mu_t_plus_1 = n*mu_t_plus_1_part;
    g = dt*dnm(a0,n,m)+dnm(a_current,n,m)*nu_tilde-dnm(b_current,n,m)*mu_t +
      bdt*( dnm(b_current_hs,n+1,m+1) - dnm(b_current_hs,n+1,m-1) - ((dnm(b_current_hs,n-1,m+1) - dnm(b_current_hs,n-1,m-1))) );
    h = dnm(b_current,n,m)*nu_tilde+dnm(a_current,n,m)*mu_t +
      bdt*( ((dnm(a_current_hs,n-1,m+1)-dnm(a_current_hs,n-1,m-1))) - dnm(a_current_hs,n+1,m+1) + dnm(a_current_hs,n+1,m-1) );

    xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
  }
} // end of _step_on_grid_k3_unroll_1_type_1(...)

__global__ void _step_on_half_grid_k3_unroll_1_type_1
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat *a_next_hs,    ffloat *b_next_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t+1/2,t+1) to (t+3/2)
  ffloat mu_t_part        = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = 0;
  ffloat mu_t_plus_1 = 0;
  ffloat g = dt*dnm(a0,0,m)+dnm(a_current_hs,0,m)*nu_tilde-dnm(b_current_hs,0,m)*mu_t +
    bdt*( dnm(b_next,1,m+1) - dnm(b_next,1,m-1) );
  ffloat h = dnm(b_current_hs,0,m)*nu_tilde+dnm(a_current_hs,0,m)*mu_t +
    bdt*( - dnm(a_next,1,m+1) + dnm(a_next,1,m-1) );
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,0,m) = (g*nu - h*mu_t_plus_1)/xi;

  mu_t = mu_t_part;
  mu_t_plus_1 = mu_t_plus_1_part;
  g = dt*dnm(a0,1,m)+dnm(a_current_hs,1,m)*nu_tilde-dnm(b_current_hs,1,m)*mu_t +
    bdt*( dnm(b_next,2,m+1) - dnm(b_next,2,m-1) );
  h = dnm(b_current_hs,1,m)*nu_tilde+dnm(a_current_hs,1,m)*mu_t +
    bdt*( 2*((dnm(a_next,0,m+1)-dnm(a_next,0,m-1))) - dnm(a_next,2,m+1) + dnm(a_next,2,m-1) );
  xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,1,m) = (g*nu - h*mu_t_plus_1)/xi;
  dnm(b_next_hs,1,m) = (g*mu_t_plus_1 + h*nu)/xi;

  for( int n = 2; n < N; n++ ) {
    mu_t = n*mu_t_part;
    mu_t_plus_1 = n*mu_t_plus_1_part;
    g = dt*dnm(a0,n,m)+dnm(a_current_hs,n,m)*nu_tilde-dnm(b_current_hs,n,m)*mu_t +
      bdt*( dnm(b_next,n+1,m+1) - dnm(b_next,n+1,m-1) - ((dnm(b_next,n-1,m+1) - dnm(b_next,n-1,m-1))) );
    h = dnm(b_current_hs,n,m)*nu_tilde+dnm(a_current_hs,n,m)*mu_t +
      bdt*( ((dnm(a_next,n-1,m+1)-dnm(a_next,n-1,m-1))) - dnm(a_next,n+1,m+1) + dnm(a_next,n+1,m-1) );
    xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;
  }
} // end of _step_on_half_grid_k3_unroll_1_type_1(...)
/** END OF 311 KERNELS **/

/** BEGINING OF 321 KERNELS **/
/**
 * One thread per m-number. Unrolled twice. Divergent flows.
 */
__global__ void _step_on_grid_k3_unroll_2_type_1
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t,t+1/2) to (t+1)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = 0;
  ffloat mu_t_plus_1 = 0;
  ffloat g = dt*dnm(a0,0,m)+dnm(a_current,0,m)*nu_tilde-dnm(b_current,0,m)*mu_t +
    bdt*( dnm(b_current_hs,1,m+1) - dnm(b_current_hs,1,m-1) );
  ffloat h = dnm(b_current,0,m)*nu_tilde+dnm(a_current,0,m)*mu_t +
    bdt*( - dnm(a_current_hs,1,m+1) + dnm(a_current_hs,1,m-1) );
  
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next,0,m) = (g*nu - h*mu_t_plus_1)/xi;

  mu_t = mu_t_part;
  mu_t_plus_1 = mu_t_plus_1_part;
  g = dt*dnm(a0,1,m)+dnm(a_current,1,m)*nu_tilde-dnm(b_current,1,m)*mu_t +
    bdt*( dnm(b_current_hs,2,m+1) - dnm(b_current_hs,2,m-1) );
  h = dnm(b_current,1,m)*nu_tilde+dnm(a_current,1,m)*mu_t +
    bdt*( 2*((dnm(a_current_hs,0,m+1)-dnm(a_current_hs,0,m-1))) - dnm(a_current_hs,2,m+1) + dnm(a_current_hs,2,m-1) );
  
  xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next,1,m) = (g*nu - h*mu_t_plus_1)/xi;
  dnm(b_next,1,m) = (g*mu_t_plus_1 + h*nu)/xi;

  for( int n = 2; n < (N-2); n += 2 ) {
    mu_t = n*mu_t_part;
    mu_t_plus_1 = n*mu_t_plus_1_part;
    g = dt*dnm(a0,n,m)+dnm(a_current,n,m)*nu_tilde-dnm(b_current,n,m)*mu_t +
      bdt*( dnm(b_current_hs,n+1,m+1) - dnm(b_current_hs,n+1,m-1) - ((dnm(b_current_hs,n-1,m+1) - dnm(b_current_hs,n-1,m-1))) );
    h = dnm(b_current,n,m)*nu_tilde+dnm(a_current,n,m)*mu_t +
      bdt*( ((dnm(a_current_hs,n-1,m+1)-dnm(a_current_hs,n-1,m-1))) - dnm(a_current_hs,n+1,m+1) + dnm(a_current_hs,n+1,m-1) );

    xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;

    //int (n+1) = n + 1;
    //if( (n+1) >= N ) { break; }

    ffloat mu_t_2 = (n+1)*mu_t_part;
    ffloat mu_t_plus_1_2 = (n+1)*mu_t_plus_1_part;
    ffloat g2 = dt*dnm(a0,(n+1),m)+dnm(a_current,(n+1),m)*nu_tilde-dnm(b_current,(n+1),m)*mu_t_2 +
      bdt*( dnm(b_current_hs,(n+1)+1,m+1) - dnm(b_current_hs,(n+1)+1,m-1) - ((dnm(b_current_hs,(n+1)-1,m+1) - dnm(b_current_hs,(n+1)-1,m-1))) );
    ffloat h2 = dnm(b_current,(n+1),m)*nu_tilde+dnm(a_current,(n+1),m)*mu_t_2 +
      bdt*( ((dnm(a_current_hs,(n+1)-1,m+1)-dnm(a_current_hs,(n+1)-1,m-1))) - dnm(a_current_hs,(n+1)+1,m+1) + dnm(a_current_hs,(n+1)+1,m-1) );

    ffloat xi2 = nu2 + mu_t_plus_1_2*mu_t_plus_1_2;
    dnm(a_next,(n+1),m) = (g2*nu - h2*mu_t_plus_1_2)/xi2;
    dnm(b_next,(n+1),m) = (g2*mu_t_plus_1_2 + h2*nu)/xi2;
  }
} // end of _step_on_grid_k3_unroll_2_type_1(...)

__global__ void _step_on_half_grid_k3_unroll_2_type_1
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat *a_next_hs,    ffloat *b_next_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t+1/2,t+1) to (t+3/2)
  ffloat mu_t_part        = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = 0;
  ffloat mu_t_plus_1 = 0;
  ffloat g = dt*dnm(a0,0,m)+dnm(a_current_hs,0,m)*nu_tilde-dnm(b_current_hs,0,m)*mu_t +
    bdt*( dnm(b_next,1,m+1) - dnm(b_next,1,m-1) );
  ffloat h = dnm(b_current_hs,0,m)*nu_tilde+dnm(a_current_hs,0,m)*mu_t +
    bdt*( - dnm(a_next,1,m+1) + dnm(a_next,1,m-1) );
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,0,m) = (g*nu - h*mu_t_plus_1)/xi;

  mu_t = mu_t_part;
  mu_t_plus_1 = mu_t_plus_1_part;
  g = dt*dnm(a0,1,m)+dnm(a_current_hs,1,m)*nu_tilde-dnm(b_current_hs,1,m)*mu_t +
    bdt*( dnm(b_next,2,m+1) - dnm(b_next,2,m-1) );
  h = dnm(b_current_hs,1,m)*nu_tilde+dnm(a_current_hs,1,m)*mu_t +
    bdt*( 2*((dnm(a_next,0,m+1)-dnm(a_next,0,m-1))) - dnm(a_next,2,m+1) + dnm(a_next,2,m-1) );
  xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,1,m) = (g*nu - h*mu_t_plus_1)/xi;
  dnm(b_next_hs,1,m) = (g*mu_t_plus_1 + h*nu)/xi;

  for( int n = 2; n < (N-2); n += 2 ) {
    mu_t = n*mu_t_part;
    mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat mu_t_2 = mu_t + mu_t_part;
    ffloat mu_t_plus_1_2 = mu_t_plus_1 + mu_t_plus_1_part;

    g = dt*dnm(a0,n,m)+dnm(a_current_hs,n,m)*nu_tilde-dnm(b_current_hs,n,m)*mu_t +
      bdt*( dnm(b_next,n+1,m+1) - dnm(b_next,n+1,m-1) - ((dnm(b_next,n-1,m+1) - dnm(b_next,n-1,m-1))) );

    ffloat g2 = dt*dnm(a0,(n+1),m)+dnm(a_current_hs,(n+1),m)*nu_tilde-dnm(b_current_hs,(n+1),m)*mu_t_2 +
      bdt*( dnm(b_next,(n+1)+1,m+1) - dnm(b_next,(n+1)+1,m-1) - ((dnm(b_next,(n+1)-1,m+1) - dnm(b_next,(n+1)-1,m-1))) );

    h = dnm(b_current_hs,n,m)*nu_tilde+dnm(a_current_hs,n,m)*mu_t +
      bdt*( ((dnm(a_next,n-1,m+1)-dnm(a_next,n-1,m-1))) - dnm(a_next,n+1,m+1) + dnm(a_next,n+1,m-1) );
    ffloat h2 = dnm(b_current_hs,(n+1),m)*nu_tilde+dnm(a_current_hs,(n+1),m)*mu_t_2 +
      bdt*( ((dnm(a_next,(n+1)-1,m+1)-dnm(a_next,(n+1)-1,m-1))) - dnm(a_next,(n+1)+1,m+1) + dnm(a_next,(n+1)+1,m-1) );

    xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;

    //int (n+1) = n + 1;
    //if( (n+1) >= N ) { break; }
    ffloat xi2 = nu2 + mu_t_plus_1_2*mu_t_plus_1_2;
    dnm(a_next_hs,(n+1),m) = (g2*nu - h2*mu_t_plus_1_2)/xi2;
    dnm(b_next_hs,(n+1),m) = (g2*mu_t_plus_1_2 + h2*nu)/xi2;
  }
} // end of _step_on_half_grid_k3_unroll_2_type_1(...)
/** END OF 321 KERNELS **/

/** BEGINING OF 341 KERNELS **/
/**
 * One thread per m-number. Unrolled 4 times. Removed divergent flows by taking calculations 
 * at n=0 and n=1 out of the loops. Kernel 341
 */
__global__ void _step_on_grid_k3_unroll_4_type_1
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t,t+1/2) to (t+1)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = 0;
  ffloat mu_t_plus_1 = 0;
  ffloat g = dt*dnm(a0,0,m)+dnm(a_current,0,m)*nu_tilde-dnm(b_current,0,m)*mu_t +
    bdt*( dnm(b_current_hs,1,m+1) - dnm(b_current_hs,1,m-1) );
  ffloat h = dnm(b_current,0,m)*nu_tilde+dnm(a_current,0,m)*mu_t +
    bdt*( - dnm(a_current_hs,1,m+1) + dnm(a_current_hs,1,m-1) );
  
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next,0,m) = (g*nu - h*mu_t_plus_1)/xi;

  mu_t = mu_t_part;
  mu_t_plus_1 = mu_t_plus_1_part;
  g = dt*dnm(a0,1,m)+dnm(a_current,1,m)*nu_tilde-dnm(b_current,1,m)*mu_t +
    bdt*( dnm(b_current_hs,2,m+1) - dnm(b_current_hs,2,m-1) );
  h = dnm(b_current,1,m)*nu_tilde+dnm(a_current,1,m)*mu_t +
    bdt*( 2*((dnm(a_current_hs,0,m+1)-dnm(a_current_hs,0,m-1))) - dnm(a_current_hs,2,m+1) + dnm(a_current_hs,2,m-1) );
  
  xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next,1,m) = (g*nu - h*mu_t_plus_1)/xi;
  dnm(b_next,1,m) = (g*mu_t_plus_1 + h*nu)/xi;

  for( int n = 2; n < (N-4); n += 4 ) {
    mu_t = n*mu_t_part;
    ffloat mu_t_2 = mu_t + mu_t_part;
    ffloat mu_t_3 = mu_t_2 + mu_t_part;
    ffloat mu_t_4 = mu_t_3 + mu_t_part;

    mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat mu_t_plus_1_2 = mu_t_plus_1 + mu_t_plus_1_part;
    ffloat mu_t_plus_1_3 = mu_t_plus_1_2 + mu_t_plus_1_part;
    ffloat mu_t_plus_1_4 = mu_t_plus_1_3 + mu_t_plus_1_part;

    g = dt*dnm(a0,n,m)+dnm(a_current,n,m)*nu_tilde-dnm(b_current,n,m)*mu_t +
      bdt*( dnm(b_current_hs,n+1,m+1) - dnm(b_current_hs,n+1,m-1) - ((dnm(b_current_hs,n-1,m+1) - dnm(b_current_hs,n-1,m-1))) );

    ffloat g2 = dt*dnm(a0,(n+1),m)+dnm(a_current,(n+1),m)*nu_tilde-dnm(b_current,(n+1),m)*mu_t_2 +
      bdt*( dnm(b_current_hs,(n+1)+1,m+1) - dnm(b_current_hs,(n+1)+1,m-1) - ((dnm(b_current_hs,(n+1)-1,m+1) - dnm(b_current_hs,(n+1)-1,m-1))) );

    ffloat g3 = dt*dnm(a0,(n+2),m)+dnm(a_current,(n+2),m)*nu_tilde-dnm(b_current,(n+2),m)*mu_t_3 +
      bdt*( dnm(b_current_hs,(n+2)+1,m+1) - dnm(b_current_hs,(n+2)+1,m-1) - ((dnm(b_current_hs,(n+2)-1,m+1) - dnm(b_current_hs,(n+2)-1,m-1))) );

    ffloat g4 = dt*dnm(a0,(n+3),m)+dnm(a_current,(n+3),m)*nu_tilde-dnm(b_current,(n+3),m)*mu_t_4 +
      bdt*( dnm(b_current_hs,(n+3)+1,m+1) - dnm(b_current_hs,(n+3)+1,m-1) - ((dnm(b_current_hs,(n+3)-1,m+1) - dnm(b_current_hs,(n+3)-1,m-1))) );

    h = dnm(b_current,n,m)*nu_tilde+dnm(a_current,n,m)*mu_t +
      bdt*( ((dnm(a_current_hs,n-1,m+1)-dnm(a_current_hs,n-1,m-1))) - dnm(a_current_hs,n+1,m+1) + dnm(a_current_hs,n+1,m-1) );

    ffloat h2 = dnm(b_current,(n+1),m)*nu_tilde+dnm(a_current,(n+1),m)*mu_t_2 +
      bdt*( ((dnm(a_current_hs,(n+1)-1,m+1)-dnm(a_current_hs,(n+1)-1,m-1))) - dnm(a_current_hs,(n+1)+1,m+1) + dnm(a_current_hs,(n+1)+1,m-1) );

    ffloat h3 = dnm(b_current,(n+2),m)*nu_tilde+dnm(a_current,(n+2),m)*mu_t_3 +
      bdt*( ((dnm(a_current_hs,(n+2)-1,m+1)-dnm(a_current_hs,(n+2)-1,m-1))) - dnm(a_current_hs,(n+2)+1,m+1) + dnm(a_current_hs,(n+2)+1,m-1) );

    ffloat h4 = dnm(b_current,(n+3),m)*nu_tilde+dnm(a_current,(n+3),m)*mu_t_4 +
      bdt*( ((dnm(a_current_hs,(n+3)-1,m+1)-dnm(a_current_hs,(n+3)-1,m-1))) - dnm(a_current_hs,(n+3)+1,m+1) + dnm(a_current_hs,(n+3)+1,m-1) );

    xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;

    //int (n+1) = n + 1;
    //if( (n+1) >= N ) { break; }

    ffloat xi2 = nu2 + mu_t_plus_1_2*mu_t_plus_1_2;
    dnm(a_next,(n+1),m) = (g2*nu - h2*mu_t_plus_1_2)/xi2;
    dnm(b_next,(n+1),m) = (g2*mu_t_plus_1_2 + h2*nu)/xi2;

    ffloat xi3 = nu2 + mu_t_plus_1_3*mu_t_plus_1_3;
    dnm(a_next,(n+2),m) = (g3*nu - h3*mu_t_plus_1_3)/xi3;
    dnm(b_next,(n+2),m) = (g3*mu_t_plus_1_3 + h3*nu)/xi3;

    ffloat xi4 = nu2 + mu_t_plus_1_3*mu_t_plus_1_4;
    dnm(a_next,(n+3),m) = (g4*nu - h4*mu_t_plus_1_4)/xi4;
    dnm(b_next,(n+3),m) = (g4*mu_t_plus_1_4 + h4*nu)/xi4;
  }
} // end of _step_on_grid_k3_unroll_4_type_1(...)

__global__ void _step_on_half_grid_k3_unroll_4_type_1
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat *a_next_hs,    ffloat *b_next_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t+1/2,t+1) to (t+3/2)
  ffloat mu_t_part        = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = 0;
  ffloat mu_t_plus_1 = 0;
  ffloat g = dt*dnm(a0,0,m)+dnm(a_current_hs,0,m)*nu_tilde-dnm(b_current_hs,0,m)*mu_t +
    bdt*( dnm(b_next,1,m+1) - dnm(b_next,1,m-1) );
  ffloat h = dnm(b_current_hs,0,m)*nu_tilde+dnm(a_current_hs,0,m)*mu_t +
    bdt*( - dnm(a_next,1,m+1) + dnm(a_next,1,m-1) );
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,0,m) = (g*nu - h*mu_t_plus_1)/xi;

  mu_t = mu_t_part;
  mu_t_plus_1 = mu_t_plus_1_part;
  g = dt*dnm(a0,1,m)+dnm(a_current_hs,1,m)*nu_tilde-dnm(b_current_hs,1,m)*mu_t +
    bdt*( dnm(b_next,2,m+1) - dnm(b_next,2,m-1) );
  h = dnm(b_current_hs,1,m)*nu_tilde+dnm(a_current_hs,1,m)*mu_t +
    bdt*( 2*((dnm(a_next,0,m+1)-dnm(a_next,0,m-1))) - dnm(a_next,2,m+1) + dnm(a_next,2,m-1) );
  xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,1,m) = (g*nu - h*mu_t_plus_1)/xi;
  dnm(b_next_hs,1,m) = (g*mu_t_plus_1 + h*nu)/xi;

  for( int n = 2; n < (N-4); n += 4 ) {
    mu_t = n*mu_t_part;
    ffloat mu_t_2 = mu_t   + mu_t_part;
    ffloat mu_t_3 = mu_t_2 + mu_t_part;
    ffloat mu_t_4 = mu_t_3 + mu_t_part;

    mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat mu_t_plus_1_2 = mu_t_plus_1 + mu_t_plus_1_part;
    ffloat mu_t_plus_1_3 = mu_t_plus_1_2 + mu_t_plus_1_part;
    ffloat mu_t_plus_1_4 = mu_t_plus_1_3 + mu_t_plus_1_part;

    g = dt*dnm(a0,n,m)+dnm(a_current_hs,n,m)*nu_tilde-dnm(b_current_hs,n,m)*mu_t +
      bdt*( dnm(b_next,n+1,m+1) - dnm(b_next,n+1,m-1) - ((dnm(b_next,n-1,m+1) - dnm(b_next,n-1,m-1))) );

    ffloat g2 = dt*dnm(a0,(n+1),m)+dnm(a_current_hs,(n+1),m)*nu_tilde-dnm(b_current_hs,(n+1),m)*mu_t_2 +
      bdt*( dnm(b_next,(n+1)+1,m+1) - dnm(b_next,(n+1)+1,m-1) - ((dnm(b_next,(n+1)-1,m+1) - dnm(b_next,(n+1)-1,m-1))) );

    ffloat g3 = dt*dnm(a0,(n+2),m)+dnm(a_current_hs,(n+2),m)*nu_tilde-dnm(b_current_hs,(n+2),m)*mu_t_3 +
      bdt*( dnm(b_next,(n+2)+1,m+1) - dnm(b_next,(n+2)+1,m-1) - ((dnm(b_next,(n+2)-1,m+1) - dnm(b_next,(n+2)-1,m-1))) );

    ffloat g4 = dt*dnm(a0,(n+3),m)+dnm(a_current_hs,(n+3),m)*nu_tilde-dnm(b_current_hs,(n+3),m)*mu_t_4 +
      bdt*( dnm(b_next,(n+3)+1,m+1) - dnm(b_next,(n+3)+1,m-1) - ((dnm(b_next,(n+3)-1,m+1) - dnm(b_next,(n+3)-1,m-1))) );

    h = dnm(b_current_hs,n,m)*nu_tilde+dnm(a_current_hs,n,m)*mu_t +
      bdt*( ((dnm(a_next,n-1,m+1)-dnm(a_next,n-1,m-1))) - dnm(a_next,n+1,m+1) + dnm(a_next,n+1,m-1) );

    ffloat h2 = dnm(b_current_hs,(n+1),m)*nu_tilde+dnm(a_current_hs,(n+1),m)*mu_t_2 +
      bdt*( ((dnm(a_next,(n+1)-1,m+1)-dnm(a_next,(n+1)-1,m-1))) - dnm(a_next,(n+1)+1,m+1) + dnm(a_next,(n+1)+1,m-1) );

    ffloat h3 = dnm(b_current_hs,(n+2),m)*nu_tilde+dnm(a_current_hs,(n+2),m)*mu_t_3 +
      bdt*( ((dnm(a_next,(n+2)-1,m+1)-dnm(a_next,(n+2)-1,m-1))) - dnm(a_next,(n+2)+1,m+1) + dnm(a_next,(n+2)+1,m-1) );

    ffloat h4 = dnm(b_current_hs,(n+3),m)*nu_tilde+dnm(a_current_hs,(n+3),m)*mu_t_3 +
      bdt*( ((dnm(a_next,(n+3)-1,m+1)-dnm(a_next,(n+3)-1,m-1))) - dnm(a_next,(n+3)+1,m+1) + dnm(a_next,(n+3)+1,m-1) );

    xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;

    //int (n+1) = n + 1;
    //if( (n+1) >= N ) { break; }
    ffloat xi2 = nu2 + mu_t_plus_1_2*mu_t_plus_1_2;
    dnm(a_next_hs,(n+1),m) = (g2*nu - h2*mu_t_plus_1_2)/xi2;
    dnm(b_next_hs,(n+1),m) = (g2*mu_t_plus_1_2 + h2*nu)/xi2;

    ffloat xi3 = nu2 + mu_t_plus_1_3*mu_t_plus_1_3;
    dnm(a_next_hs,(n+2),m) = (g3*nu - h3*mu_t_plus_1_3)/xi3;
    dnm(b_next_hs,(n+2),m) = (g3*mu_t_plus_1_3 + h3*nu)/xi3;

    ffloat xi4 = nu2 + mu_t_plus_1_4*mu_t_plus_1_4;
    dnm(a_next_hs,(n+3),m) = (g4*nu - h4*mu_t_plus_1_4)/xi4;
    dnm(b_next_hs,(n+3),m) = (g4*mu_t_plus_1_4 + h4*nu)/xi4;
  }
} // end of _step_on_half_grid_k3_unroll_4_type_1(...)
/** END OF 341 KERNELS **/

/** BEGINING OF 342 KERNELS **/
/**
 * One thread per m-number. Unrolled 4 times. Removed divergent flows by taking calculations 
 * at n=0 and n=1 out of the loops. Elements partially reused. Kernel 342.
 */
__global__ void _step_on_grid_k3_unroll_4_type_2
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t,t+1/2) to (t+1)
  ffloat mu_t_part = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = 0;
  ffloat mu_t_plus_1 = 0;
  ffloat g = dt*dnm(a0,0,m)+dnm(a_current,0,m)*nu_tilde-dnm(b_current,0,m)*mu_t +
    bdt*( dnm(b_current_hs,1,m+1) - dnm(b_current_hs,1,m-1) );
  ffloat h = dnm(b_current,0,m)*nu_tilde+dnm(a_current,0,m)*mu_t +
    bdt*( - dnm(a_current_hs,1,m+1) + dnm(a_current_hs,1,m-1) );
  
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next,0,m) = (g*nu - h*mu_t_plus_1)/xi;

  mu_t = mu_t_part;
  mu_t_plus_1 = mu_t_plus_1_part;
  g = dt*dnm(a0,1,m)+dnm(a_current,1,m)*nu_tilde-dnm(b_current,1,m)*mu_t +
    bdt*( dnm(b_current_hs,2,m+1) - dnm(b_current_hs,2,m-1) );
  h = dnm(b_current,1,m)*nu_tilde+dnm(a_current,1,m)*mu_t +
    bdt*( 2*((dnm(a_current_hs,0,m+1)-dnm(a_current_hs,0,m-1))) - dnm(a_current_hs,2,m+1) + dnm(a_current_hs,2,m-1) );
  
  xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next,1,m) = (g*nu - h*mu_t_plus_1)/xi;
  dnm(b_next,1,m) = (g*mu_t_plus_1 + h*nu)/xi;

  for( int n = 2; n < (N-4); n += 4 ) {
    mu_t = n*mu_t_part;
    ffloat mu_t_2 = mu_t + mu_t_part;
    ffloat mu_t_3 = mu_t_2 + mu_t_part;
    ffloat mu_t_4 = mu_t_3 + mu_t_part;

    mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat mu_t_plus_1_2 = mu_t_plus_1 + mu_t_plus_1_part;
    ffloat mu_t_plus_1_3 = mu_t_plus_1_2 + mu_t_plus_1_part;
    ffloat mu_t_plus_1_4 = mu_t_plus_1_3 + mu_t_plus_1_part;

    ffloat b_current_hs_n_plus_1_m_plus_1  = dnm(b_current_hs,n+1,m+1);
    ffloat b_current_hs_n_plus_1_m_minus_1 = dnm(b_current_hs,n+1,m-1);
    g = dt*dnm(a0,n,m)+dnm(a_current,n,m)*nu_tilde-dnm(b_current,n,m)*mu_t +
      bdt*( b_current_hs_n_plus_1_m_plus_1 - b_current_hs_n_plus_1_m_minus_1 - ((dnm(b_current_hs,n-1,m+1) - dnm(b_current_hs,n-1,m-1))) );

    ffloat g2 = dt*dnm(a0,(n+1),m)+dnm(a_current,(n+1),m)*nu_tilde-dnm(b_current,(n+1),m)*mu_t_2 +
      bdt*( dnm(b_current_hs,n+2,m+1) - dnm(b_current_hs,n+2,m-1) - ((dnm(b_current_hs,n,m+1) - dnm(b_current_hs,n,m-1))) );

    ffloat g3 = dt*dnm(a0,(n+2),m)+dnm(a_current,(n+2),m)*nu_tilde-dnm(b_current,(n+2),m)*mu_t_3 +
      bdt*( dnm(b_current_hs,(n+3),m+1) - dnm(b_current_hs,(n+3),m-1) - (( b_current_hs_n_plus_1_m_plus_1 - b_current_hs_n_plus_1_m_minus_1) ) );

    ffloat g4 = dt*dnm(a0,(n+3),m)+dnm(a_current,(n+3),m)*nu_tilde-dnm(b_current,(n+3),m)*mu_t_4 +
      bdt*( dnm(b_current_hs,(n+4),m+1) - dnm(b_current_hs,(n+4),m-1) - ((dnm(b_current_hs,(n+2),m+1) - dnm(b_current_hs,(n+2),m-1))) );

    ffloat a_current_hs_n_plus_1_m_plus_1 = dnm(a_current_hs,n+1,m+1);
    h = dnm(b_current,n,m)*nu_tilde+dnm(a_current,n,m)*mu_t +
      bdt*( ((dnm(a_current_hs,n-1,m+1)-dnm(a_current_hs,n-1,m-1))) - a_current_hs_n_plus_1_m_plus_1 + dnm(a_current_hs,n+1,m-1) );

    ffloat h2 = dnm(b_current,(n+1),m)*nu_tilde+dnm(a_current,(n+1),m)*mu_t_2 +
      bdt*( ((dnm(a_current_hs,n,m+1)-dnm(a_current_hs,n,m-1))) - dnm(a_current_hs,(n+2),m+1) + dnm(a_current_hs,(n+2),m-1) );

    ffloat h3 = dnm(b_current,(n+2),m)*nu_tilde+dnm(a_current,(n+2),m)*mu_t_3 +
      bdt*( (( a_current_hs_n_plus_1_m_plus_1-dnm(a_current_hs,(n+1),m-1))) - dnm(a_current_hs,(n+3),m+1) + dnm(a_current_hs,(n+3),m-1) );

    ffloat h4 = dnm(b_current,(n+3),m)*nu_tilde+dnm(a_current,(n+3),m)*mu_t_4 +
      bdt*( ((dnm(a_current_hs,(n+2),m+1)-dnm(a_current_hs,(n+2),m-1))) - dnm(a_current_hs,(n+4),m+1) + dnm(a_current_hs,(n+4),m-1) );

    xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next,n,m) = (g*mu_t_plus_1 + h*nu)/xi;

    //int (n+1) = n + 1;
    //if( (n+1) >= N ) { break; }

    ffloat xi2 = nu2 + mu_t_plus_1_2*mu_t_plus_1_2;
    dnm(a_next,(n+1),m) = (g2*nu - h2*mu_t_plus_1_2)/xi2;
    dnm(b_next,(n+1),m) = (g2*mu_t_plus_1_2 + h2*nu)/xi2;

    ffloat xi3 = nu2 + mu_t_plus_1_3*mu_t_plus_1_3;
    dnm(a_next,(n+2),m) = (g3*nu - h3*mu_t_plus_1_3)/xi3;
    dnm(b_next,(n+2),m) = (g3*mu_t_plus_1_3 + h3*nu)/xi3;

    ffloat xi4 = nu2 + mu_t_plus_1_3*mu_t_plus_1_4;
    dnm(a_next,(n+3),m) = (g4*nu - h4*mu_t_plus_1_4)/xi4;
    dnm(b_next,(n+3),m) = (g4*mu_t_plus_1_4 + h4*nu)/xi4;
  }
} // end of _step_on_grid_k3_unroll_4_type_2(...)

__global__ void _step_on_half_grid_k3_unroll_4_type_2
    (ffloat *a0, ffloat *a_current,    ffloat *b_current,
     ffloat *a_next,       ffloat *b_next,
     ffloat *a_current_hs, ffloat *b_current_hs,
     ffloat *a_next_hs,    ffloat *b_next_hs,
     ffloat t, ffloat t_hs,
     ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
  const int m = threadIdx.x+blockDim.x*blockIdx.x;
  if( m==0 || m > TMSIZE ) { return; }

  // step from (t+1/2,t+1) to (t+3/2)
  ffloat mu_t_part        = (E_dc + E_omega*cos_omega_t+B*dev_phi_y(m))*dt/2;
  ffloat mu_t_plus_1_part = (E_dc + E_omega*cos_omega_t_plus_dt+B*dev_phi_y(m))*dt/2;

  ffloat mu_t = 0;
  ffloat mu_t_plus_1 = 0;
  ffloat g = dt*dnm(a0,0,m)+dnm(a_current_hs,0,m)*nu_tilde-dnm(b_current_hs,0,m)*mu_t +
    bdt*( dnm(b_next,1,m+1) - dnm(b_next,1,m-1) );
  ffloat h = dnm(b_current_hs,0,m)*nu_tilde+dnm(a_current_hs,0,m)*mu_t +
    bdt*( - dnm(a_next,1,m+1) + dnm(a_next,1,m-1) );
  ffloat xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,0,m) = (g*nu - h*mu_t_plus_1)/xi;

  mu_t = mu_t_part;
  mu_t_plus_1 = mu_t_plus_1_part;
  g = dt*dnm(a0,1,m)+dnm(a_current_hs,1,m)*nu_tilde-dnm(b_current_hs,1,m)*mu_t +
    bdt*( dnm(b_next,2,m+1) - dnm(b_next,2,m-1) );
  h = dnm(b_current_hs,1,m)*nu_tilde+dnm(a_current_hs,1,m)*mu_t +
    bdt*( 2*((dnm(a_next,0,m+1)-dnm(a_next,0,m-1))) - dnm(a_next,2,m+1) + dnm(a_next,2,m-1) );
  xi = nu2 + mu_t_plus_1*mu_t_plus_1;
  dnm(a_next_hs,1,m) = (g*nu - h*mu_t_plus_1)/xi;
  dnm(b_next_hs,1,m) = (g*mu_t_plus_1 + h*nu)/xi;

  for( int n = 2; n < (N-4); n += 4 ) {
    mu_t = n*mu_t_part;
    ffloat mu_t_2 = mu_t   + mu_t_part;
    ffloat mu_t_3 = mu_t_2 + mu_t_part;
    ffloat mu_t_4 = mu_t_3 + mu_t_part;

    mu_t_plus_1 = n*mu_t_plus_1_part;
    ffloat mu_t_plus_1_2 = mu_t_plus_1 + mu_t_plus_1_part;
    ffloat mu_t_plus_1_3 = mu_t_plus_1_2 + mu_t_plus_1_part;
    ffloat mu_t_plus_1_4 = mu_t_plus_1_3 + mu_t_plus_1_part;

    g = dt*dnm(a0,n,m)+dnm(a_current_hs,n,m)*nu_tilde-dnm(b_current_hs,n,m)*mu_t +
      bdt*( dnm(b_next,n+1,m+1) - dnm(b_next,n+1,m-1) - ((dnm(b_next,n-1,m+1) - dnm(b_next,n-1,m-1))) );

    ffloat g2 = dt*dnm(a0,(n+1),m)+dnm(a_current_hs,(n+1),m)*nu_tilde-dnm(b_current_hs,(n+1),m)*mu_t_2 +
      bdt*( dnm(b_next,(n+2),m+1) - dnm(b_next,(n+2),m-1) - ((dnm(b_next,n,m+1) - dnm(b_next,n,m-1))) );

    ffloat g3 = dt*dnm(a0,(n+2),m)+dnm(a_current_hs,(n+2),m)*nu_tilde-dnm(b_current_hs,(n+2),m)*mu_t_3 +
      bdt*( dnm(b_next,(n+3),m+1) - dnm(b_next,(n+3),m-1) - ((dnm(b_next,(n+1),m+1) - dnm(b_next,(n+1),m-1))) );

    ffloat g4 = dt*dnm(a0,(n+3),m)+dnm(a_current_hs,(n+3),m)*nu_tilde-dnm(b_current_hs,(n+3),m)*mu_t_4 +
      bdt*( dnm(b_next,(n+4),m+1) - dnm(b_next,(n+4),m-1) - ((dnm(b_next,(n+2),m+1) - dnm(b_next,(n+2),m-1))) );

    h = dnm(b_current_hs,n,m)*nu_tilde+dnm(a_current_hs,n,m)*mu_t +
      bdt*( ((dnm(a_next,n-1,m+1)-dnm(a_next,n-1,m-1))) - dnm(a_next,n+1,m+1) + dnm(a_next,n+1,m-1) );

    ffloat h2 = dnm(b_current_hs,(n+1),m)*nu_tilde+dnm(a_current_hs,(n+1),m)*mu_t_2 +
      bdt*( ((dnm(a_next,n,m+1)-dnm(a_next,n,m-1))) - dnm(a_next,(n+2),m+1) + dnm(a_next,(n+2),m-1) );

    ffloat h3 = dnm(b_current_hs,(n+2),m)*nu_tilde+dnm(a_current_hs,(n+2),m)*mu_t_3 +
      bdt*( ((dnm(a_next,(n+1),m+1)-dnm(a_next,(n+1),m-1))) - dnm(a_next,(n+3),m+1) + dnm(a_next,(n+3),m-1) );

    ffloat h4 = dnm(b_current_hs,(n+3),m)*nu_tilde+dnm(a_current_hs,(n+3),m)*mu_t_3 +
      bdt*( ((dnm(a_next,(n+2),m+1)-dnm(a_next,(n+2),m-1))) - dnm(a_next,(n+4),m+1) + dnm(a_next,(n+4),m-1) );

    xi = nu2 + mu_t_plus_1*mu_t_plus_1;
    dnm(a_next_hs,n,m) = (g*nu - h*mu_t_plus_1)/xi;
    dnm(b_next_hs,n,m) = (g*mu_t_plus_1 + h*nu)/xi;

    //int (n+1) = n + 1;
    //if( (n+1) >= N ) { break; }
    ffloat xi2 = nu2 + mu_t_plus_1_2*mu_t_plus_1_2;
    dnm(a_next_hs,(n+1),m) = (g2*nu - h2*mu_t_plus_1_2)/xi2;
    dnm(b_next_hs,(n+1),m) = (g2*mu_t_plus_1_2 + h2*nu)/xi2;

    ffloat xi3 = nu2 + mu_t_plus_1_3*mu_t_plus_1_3;
    dnm(a_next_hs,(n+2),m) = (g3*nu - h3*mu_t_plus_1_3)/xi3;
    dnm(b_next_hs,(n+2),m) = (g3*mu_t_plus_1_3 + h3*nu)/xi3;

    ffloat xi4 = nu2 + mu_t_plus_1_4*mu_t_plus_1_4;
    dnm(a_next_hs,(n+3),m) = (g4*nu - h4*mu_t_plus_1_4)/xi4;
    dnm(b_next_hs,(n+3),m) = (g4*mu_t_plus_1_4 + h4*nu)/xi4;
  }
} // end of _step_on_half_grid_k3_unroll_4_type_2(...)
/** END OF 342 KERNELS **/

__global__ void av_gpu_parallel(ffloat *a, ffloat *b, ffloat *av_data, ffloat t) {
  //threadIdx.x;
  //blockIdx.x;
  //blockDim.x; // number of threads per block

  __shared__ ffloat v_dr_acc[PPP];
  __shared__ ffloat v_y_acc[PPP];
  __shared__ ffloat m_over_m_x_inst_acc[PPP];

  int thid = threadIdx.x;
  v_dr_acc[thid]            = 0;
  v_y_acc[thid]             = 0;
  m_over_m_x_inst_acc[thid] = 0;
  for( int i = thid+1; i < TMSIZE; i += PPP ) {
    v_dr_acc[thid]            += dnm(b,1,i)*dPhi;
    v_y_acc[thid]             += dnm(a,0,i)*dev_phi_y(i)*dPhi;
    m_over_m_x_inst_acc[thid] += dnm(a,1,i)*dPhi;
  }

  __syncthreads();

  //for(int delta = PPP/2; delta > 0; delta /= 2 ) {
  //int delta = PPP/2;
  //  for( int i = thid; i < delta; i++ ) {
  //    v_dr_acc[i]            += v_dr_acc[i+delta];
  //    v_y_acc[i]             += v_y_acc[i+delta];
  //    m_over_m_x_inst_acc[i] += m_over_m_x_inst_acc[i+delta];
  //  }
  //  __syncthreads();
    //}
    //__syncthreads();

  if( thid == 0 ) {
    int av_count = av_data[0] + 1;
    ffloat v_dr_inst = 0; ffloat v_y_inst = 0; ffloat m_over_m_x_inst = 0;
    for( int m = 0; m < PPP; m++ ) {
      v_dr_inst += v_dr_acc[m];
      v_y_inst  += v_y_acc[m];
      m_over_m_x_inst += m_over_m_x_inst_acc[thid];
    }
    //ffloat v_dr_inst = v_dr_acc[0]; ffloat v_y_inst = v_y_acc[0]; ffloat m_over_m_x_inst = m_over_m_x_inst_acc[0];
    //v_dr_av = v_dr_av+(v_dr_inst-v_dr_av)/av_count;
    av_data[1] += (v_dr_inst-av_data[1])/av_count; // av_data[1] holds v_dr_av

    //v_y_av = v_y_av+(v_y_inst-v_y_av)/av_count;
    av_data[2] += (v_y_inst-av_data[2])/av_count; // av_data[2] holds v_y_av

    //m_over_m_x_av = m_over_m_x_av+(m_over_m_x_inst-m_over_m_x_av)/av_count;
    av_data[3] += (m_over_m_x_inst-av_data[3])/av_count; // av_data[3] holds m_over_m_x_av

    //A += cos(omega*t)*v_dr_inst*dt;
    av_data[4] += cos(omega*t)*v_dr_inst*dt; // av_data[4] holds absorption A
    av_data[5] += sin(omega*t)*v_dr_inst*dt; // av_data[5] holds sin absorption A

    av_data[0] += 1;
  }
} // end of av_gpu_parallel(...)

__global__ void av_gpu(ffloat *a, ffloat *b, ffloat *av_data, ffloat t) {
  int av_count = av_data[0] + 1;

  ffloat v_dr_inst = 0; ffloat v_y_inst = 0; ffloat m_over_m_x_inst = 0;
  for( int m = 1; m < TMSIZE; m++ ) {
    v_dr_inst += dnm(b,1,m)*dPhi;
    v_y_inst  += dnm(a,0,m)*dev_phi_y(m)*dPhi;
    m_over_m_x_inst += dnm(a,1,m)*dPhi;
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
} // end of av_gpu(...)

extern "C"
void step_on_grid(int blocks, ffloat *a0, ffloat *a_current,    ffloat *b_current,
                  ffloat *a_next,       ffloat *b_next,
                  ffloat *a_current_hs, ffloat *b_current_hs,
                  ffloat t, ffloat t_hs, ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{

#if BLTZM_KERNEL == 1
  _step_on_grid_k1<<<dimGrid, dimBlock>>>(a0, a_current, b_current, a_next, b_next,
                                         a_current_hs, b_current_hs,
                                         t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 2
  _step_on_grid_k2<<<dimGrid, dimBlock>>>(a0, a_current, b_current, a_next, b_next,
					  a_current_hs, b_current_hs,
					  t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 310
  _step_on_grid_k3_unroll_1_type_0<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
							    a_current_hs, b_current_hs,
							    t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 311
  _step_on_grid_k3_unroll_1_type_1<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
							    a_current_hs, b_current_hs,
							    t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 321
  _step_on_grid_k3_unroll_2_type_1<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
							    a_current_hs, b_current_hs,
							    t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 341
  _step_on_grid_k3_unroll_4_type_1<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
							    a_current_hs, b_current_hs,
							    t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 342
  _step_on_grid_k3_unroll_4_type_2<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
							    a_current_hs, b_current_hs,
							    t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 4
  _step_on_grid_k4<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
					    a_current_hs, b_current_hs,
					    t, t_hs, cos_omega_t, cos_omega_t_plus_dt);
#endif
}

extern "C"
void step_on_half_grid(int blocks, ffloat *a0, ffloat *a_current,    ffloat *b_current,
                       ffloat *a_next,       ffloat *b_next,
                       ffloat *a_current_hs, ffloat *b_current_hs,
                       ffloat *a_next_hs, ffloat *b_next_hs,
                       ffloat t, ffloat t_hs, ffloat cos_omega_t, ffloat cos_omega_t_plus_dt)
{
#if BLTZM_KERNEL   == 1
  _step_on_half_grid_k1<<<dimGrid, dimBlock>>>(a0, a_current, b_current, a_next, b_next,
					       a_current_hs, b_current_hs, a_next_hs, b_next_hs,
					       t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 2
  _step_on_half_grid_k2<<<dimGrid, dimBlock>>>(a0, a_current, b_current, a_next, b_next,
					       a_current_hs, b_current_hs, a_next_hs, b_next_hs,
					       t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 310
  _step_on_half_grid_k3_unroll_1_type_0<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
								 a_current_hs, b_current_hs, a_next_hs, b_next_hs,
								 t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 311
  _step_on_half_grid_k3_unroll_1_type_1<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
								 a_current_hs, b_current_hs, a_next_hs, b_next_hs,
								 t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 321
  _step_on_half_grid_k3_unroll_2_type_1<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
								 a_current_hs, b_current_hs, a_next_hs, b_next_hs,
								 t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 341
  _step_on_half_grid_k3_unroll_4_type_1<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
								 a_current_hs, b_current_hs, a_next_hs, b_next_hs,
								 t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 342
  _step_on_half_grid_k3_unroll_4_type_2<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
								 a_current_hs, b_current_hs, a_next_hs, b_next_hs,
								 t, t_hs, cos_omega_t, cos_omega_t_plus_dt);

#elif BLTZM_KERNEL == 4
  _step_on_half_grid_k4<<<blocks,TH_PER_BLOCK>>>(a0, a_current, b_current, a_next, b_next,
						 a_current_hs, b_current_hs, a_next_hs, b_next_hs,
						 t, t_hs, cos_omega_t, cos_omega_t_plus_dt);
#endif
}

extern "C"
void av(int blocks, ffloat *a, ffloat *b, ffloat *av_data, ffloat t) {
  av_gpu_parallel<<<1,PPP>>>(a, b, av_data, t);
  /** av_gpu<<<1,1>>>(a, b, av_data, t); **/ // sequential reduction averaging. very slow.
}
