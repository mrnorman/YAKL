
#include <iostream>
#include "YAKL.h"

using yakl::Array;
using yakl::styleC;
using yakl::memHost;
using yakl::memDevice;
using yakl::COLON;
using yakl::timer_start;
using yakl::timer_stop;


template <class real>
void miniWeather_tend_x(int nx, int nz, char const *label, bool use_pfor) {
  using yakl::c::SimpleBounds;
  using yakl::c::parallel_for;
  int constexpr NUM_VARS = 4;
  int constexpr hs = 2;
  int constexpr ID_DENS = 0;
  int constexpr ID_UMOM = 1;
  int constexpr ID_WMOM = 2;
  int constexpr ID_RHOT = 3;
  real constexpr hv_beta = 0.01;
  real constexpr sten_size = 4;
  real constexpr C0 = 28;
  real dx = 1;
  real dz = 1;
  real dt = 2;
  yakl::Array<real,3,yakl::memHost,yakl::styleC> host_state("state",NUM_VARS,nz+2*hs,nx+2*hs);
  yakl::Array<real,3,yakl::memHost,yakl::styleC> host_flux ("flux" ,NUM_VARS,nz,nx+1);
  yakl::Array<real,3,yakl::memHost,yakl::styleC> host_tend ("tend" ,NUM_VARS,nz,nx);
  yakl::Array<real,1,yakl::memHost,yakl::styleC> host_hy_dens_cell      ("hy_dens_cell      ",nz+2*hs);
  yakl::Array<real,1,yakl::memHost,yakl::styleC> host_hy_dens_theta_cell("hy_dens_theta_cell",nz+2*hs);
  srand (17);
  for (int l=0; l < NUM_VARS; l++) {
    for (int k=0; k < nz+2*hs; k++) {
      for (int i=0; i < nx+2*hs; i++) {
        host_state(ID_DENS,k,i) = 1 + ((double) rand() / (double) RAND_MAX);
        host_state(ID_UMOM,k,i) = ((double) rand() / (double) RAND_MAX) * 10 - 5;
        host_state(ID_WMOM,k,i) = ((double) rand() / (double) RAND_MAX) * 10 - 5;
        host_state(ID_RHOT,k,i) = 300 + ((double) rand() / (double) RAND_MAX) * 10 - 5;
      }
    }
  }
  host_hy_dens_cell       = 1;
  host_hy_dens_theta_cell = 300;

  //Compute the hyperviscosity coeficient
  real hv_coef = -hv_beta * dx / (16*dt);

  if (use_pfor) {

    auto state              = host_state             .createDeviceCopy();
    auto flux               = host_flux              .createDeviceObject();
    auto tend               = host_tend              .createDeviceObject();
    auto hy_dens_cell       = host_hy_dens_cell      .createDeviceCopy();
    auto hy_dens_theta_cell = host_hy_dens_theta_cell.createDeviceCopy();

    yakl::timer_start(label);

    //Compute fluxes in the x-direction for each cell
    // for (k=0; k<nz; k++) {
    //   for (i=0; i<nx+1; i++) {
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<2>(nz,nx+1) , YAKL_LAMBDA (int k, int i ) {
      yakl::SArray<real,1,4> stencil;
      yakl::SArray<real,1,NUM_VARS> d3_vals;
      yakl::SArray<real,1,NUM_VARS> vals;
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (int ll=0; ll<NUM_VARS; ll++) {
        for (int s=0; s < sten_size; s++) { stencil(s) = state(ll,hs+k,i+s); }
        //Fourth-order-accurate interpolation of the state
        vals(ll) = -stencil(0)/12 + (7*stencil(1))/12 + (7*stencil(2))/12 - stencil(3)/12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals(ll) = -stencil(0) + 3*stencil(1) - 3*stencil(2) + stencil(3);
      }
      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      real r = vals(ID_DENS) + hy_dens_cell(hs+k);
      real u = vals(ID_UMOM) / r;
      real w = vals(ID_WMOM) / r;
      real t = ( vals(ID_RHOT) + hy_dens_theta_cell(hs+k) ) / r;
      real p = C0*std::pow((r*t),static_cast<real>(1.4));
      //Compute the flux vector
      flux(ID_DENS,k,i) = r*u     - hv_coef*d3_vals(ID_DENS);
      flux(ID_UMOM,k,i) = r*u*u+p - hv_coef*d3_vals(ID_UMOM);
      flux(ID_WMOM,k,i) = r*u*w   - hv_coef*d3_vals(ID_WMOM);
      flux(ID_RHOT,k,i) = r*u*t   - hv_coef*d3_vals(ID_RHOT);
    });
    //Use the fluxes to compute tendencies for each cell
    // for (ll=0; ll<NUM_VARS; ll++) {
    //   for (k=0; k<nz; k++) {
    //     for (i=0; i<nx; i++) {
    parallel_for( YAKL_AUTO_LABEL() , SimpleBounds<3>(NUM_VARS,nz,nx) , YAKL_LAMBDA ( int ll, int k, int i ) {
      tend(ll,k,i) = -( flux(ll,k,i+1) - flux(ll,k,i) ) / dx;
    });

    yakl::timer_stop(label);

    std::cout << yakl::intrinsics::sum(yakl::intrinsics::abs(tend)) << std::endl;

  } else {

#ifdef YAKL_SEPARATE_MEMORY_SPACE
    auto &state              = host_state             ;
    auto &flux               = host_flux              ;
    auto &tend               = host_tend              ;
    auto &hy_dens_cell       = host_hy_dens_cell      ;
    auto &hy_dens_theta_cell = host_hy_dens_theta_cell;
#else
    auto state              = host_state             .createDeviceCopy();
    auto flux               = host_flux              .createDeviceObject();
    auto tend               = host_tend              .createDeviceObject();
    auto hy_dens_cell       = host_hy_dens_cell      .createDeviceCopy();
    auto hy_dens_theta_cell = host_hy_dens_theta_cell.createDeviceCopy();
#endif

    yakl::timer_start(label);

    //Compute fluxes in the x-direction for each cell
    for (int k=0; k<nz; k++) {
      for (int i=0; i<nx+1; i++) {
        yakl::SArray<real,1,4> stencil;
        yakl::SArray<real,1,NUM_VARS> d3_vals;
        yakl::SArray<real,1,NUM_VARS> vals;
        //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        for (int ll=0; ll<NUM_VARS; ll++) {
          for (int s=0; s < sten_size; s++) { stencil(s) = state(ll,hs+k,i+s); }
          //Fourth-order-accurate interpolation of the state
          vals(ll) = -stencil(0)/12 + (7*stencil(1))/12 + (7*stencil(2))/12 - stencil(3)/12;
          //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
          d3_vals(ll) = -stencil(0) + 3*stencil(1) - 3*stencil(2) + stencil(3);
        }
        //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        real r = vals(ID_DENS) + hy_dens_cell(hs+k);
        real u = vals(ID_UMOM) / r;
        real w = vals(ID_WMOM) / r;
        real t = ( vals(ID_RHOT) + hy_dens_theta_cell(hs+k) ) / r;
        real p = C0*std::pow((r*t),static_cast<real>(1.4));
        //Compute the flux vector
        flux(ID_DENS,k,i) = r*u     - hv_coef*d3_vals(ID_DENS);
        flux(ID_UMOM,k,i) = r*u*u+p - hv_coef*d3_vals(ID_UMOM);
        flux(ID_WMOM,k,i) = r*u*w   - hv_coef*d3_vals(ID_WMOM);
        flux(ID_RHOT,k,i) = r*u*t   - hv_coef*d3_vals(ID_RHOT);
      }
    }
    //Use the fluxes to compute tendencies for each cell
    for (int ll=0; ll<NUM_VARS; ll++) {
      for (int k=0; k<nz; k++) {
        for (int i=0; i<nx; i++) {
          tend(ll,k,i) = -( flux(ll,k,i+1) - flux(ll,k,i) ) / dx;
        }
      }
    }

    yakl::timer_stop(label);

    std::cout << yakl::intrinsics::sum(yakl::intrinsics::abs(tend)) << std::endl;

  }

}


int main() {
  yakl::init();
  {
    int nx = 2048;
    int nz = 1024;
    int niter = 10;

    for (int i=0; i<niter; i++) { miniWeather_tend_x<double>(nx,nz,"miniWeather_tend_x_double_pfor",true ); }
    for (int i=0; i<niter; i++) { miniWeather_tend_x<float >(nx,nz,"miniWeather_tend_x_float_pfor" ,true ); }
    for (int i=0; i<niter; i++) { miniWeather_tend_x<double>(nx,nz,"miniWeather_tend_x_double_for" ,false); }
    for (int i=0; i<niter; i++) { miniWeather_tend_x<float >(nx,nz,"miniWeather_tend_x_float_for"  ,false); }

  }
  yakl::finalize();
  
  return 0;
}

