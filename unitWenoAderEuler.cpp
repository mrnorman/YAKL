
#include "const.h"
#include "SArray.h"
#include "Array.h"
#include "TransformMatrices.h"
#include "WenoLimiter.h"
#include "YAKL.h"

// Easier to read indexing of the state vector
int const numVars = 3;

int const ID_DENS = 0;
int const ID_UMOM = 1;
int const ID_RHOT = 2;

// Some physical constants
real const pi    = 3.1415926535897932384626433832795028842;
real const grav  = 9.8;
real const cp    = 1004.;
real const cv    = 717.;
real const rd    = 287.;
real const p0    = 1.0e5;
real const c0    = 27.5629410929725921310572974482;
real const gamm  = 1.40027894002789400278940027894;


inline _HOSTDEV double mypow ( double const x , double const p ) { return pow (x,p); }
inline _HOSTDEV float  mypow ( float  const x , float  const p ) { return powf(x,p); }
inline _HOSTDEV double mysqrt( double const x ) { return sqrt (x); }
inline _HOSTDEV float  mysqrt( float  const x ) { return sqrtf(x); }
inline _HOSTDEV double myfabs( double const x ) { return fabs (x); }
inline _HOSTDEV float  myfabs( float  const x ) { return fabsf(x); }


// Enforce solid wall BCs on the state (they're not actually used in the shock tube sim)
// Just extend the existing state constant interp into boundaries
class boundariesClass {
public:
  _YAKL void operator() (ulong ignore , Array<real> &state) {
    for (int j=0; j<numVars; j++) {
      for (int i=0; i<hs; i++) {
        state(j,      i) = state(j,hs);
        state(j,nx+hs+i) = state(j,nx+hs-1);
      }
    }
  }
};


void computeTimeStep(Array<real> const &state, real &dt) {
  real umax = 0.;
  for (int i=0; i<nx; i++) {
    real r = state(ID_DENS,hs+i);
    real u = state(ID_UMOM,hs+i) / r;
    real t = state(ID_RHOT,hs+i) / r;
    real p = c0 * mypow( r*t , gamm );
    real tmp = myfabs(u) + mysqrt(gamm*p/r);
    if (tmp > umax) umax = tmp;
  }
  dt = dx*cfl/umax;
}


class reconAderClass {
public:
  _YAKL void operator() (ulong i, Array<real> const &state, SArray<real,ord,ord,ord> &wenoRecon, SArray<real,ord,tord> const &c2g_lower,
                         SArray<real,tord,tord> const &deriv, real const &dt, Array<real> &fluxLimits, Array<real> &stateLimits ) {
    SArray<real,ord> stencil;
    SArray<real,ord> coefs;
    SArray<real,numVars,tord,tord> state_dts;
    SArray<real,tord,tord> ruu_dts;
    SArray<real,tord,tord> rtgamm_dts;
    SArray<real,numVars,tord,tord> flux_dts;
    WenoLimiter<real> weno;

    // Reconstruct with WENO, compute time dervatives, and then time average, then store fluxes
    // for (int i=0; i<nx; i++) {

    // Reconstruct tord GLL points for each of the state variables using WENO Limiting
    for (int j=0; j<numVars; j++) {
      // Store a stencil of values
      for (int ii=0; ii<ord; ii++) {
        stencil(ii) = state(j,i+ii);
      }

      // Compute the WENO-limited coefficients
      weno.compute_weno_coefs( wenoRecon , stencil , coefs );

      // Compute GLL points from polynomial coefficients
      for (int ii=0; ii<tord; ii++) {
        state_dts(j,0,ii) = 0._fp;
        for (int s=0; s<ord; s++) {
          state_dts(j,0,ii) += c2g_lower(s,ii) * coefs(s);
        }
      }
    }

    // Zero out the necessary Differential Transform arrays
    for (int kt=0; kt<tord; kt++) {
      for (int ii=0; ii<tord; ii++) {
        ruu_dts   (kt,ii) = 0._fp;
        flux_dts  (ID_RHOT,kt,ii) = 0._fp;
        rtgamm_dts(kt,ii) = 0._fp;
      }
    }

    // Compute the fluxes at each of the GLL points (zeroth-order DTs)
    for (int ii=0; ii<tord; ii++) {
      real r = state_dts(ID_DENS,0,ii);
      real u = state_dts(ID_UMOM,0,ii) / r;
      real t = state_dts(ID_RHOT,0,ii) / r;
      flux_dts(ID_DENS,0,ii) = r*u;
      ruu_dts   (0,ii) = r*u*u;
      rtgamm_dts(0,ii) = mypow( r*t , gamm );
      flux_dts(ID_UMOM,0,ii) = ruu_dts(0,ii) + c0*rtgamm_dts(0,ii);
      flux_dts(ID_RHOT,0,ii) = r*u*t;
    }

    // Compute time derivatives at each of the GLL points
    for (int kt=0; kt<tord-1; kt++) {
      // Compute the next time level of state DTs
      for (int j=0; j<numVars; j++) {
        for (int ii=0; ii<tord; ii++) {
          real d_dx = 0.;
          for (int s=0; s<tord; s++) {
            d_dx += deriv(s,ii) * flux_dts(j,kt,s);
          }
          state_dts(j,kt+1,ii) = -d_dx / (kt+1._fp);
        }
      }

      // Compute the flux DTs at the next time level
      for (int ii=0; ii<tord; ii++) {
        real tot1 = 0._fp;
        real tot2 = 0._fp;
        real tot3 = 0._fp;
        for (int rt=0; rt<=kt+1; rt++) {
          tot1 += state_dts(ID_UMOM,rt,ii) * state_dts(ID_UMOM,kt+1-rt,ii) - state_dts(ID_DENS,rt,ii) * ruu_dts (        kt+1-rt,ii);
          tot2 += state_dts(ID_UMOM,rt,ii) * state_dts(ID_RHOT,kt+1-rt,ii) - state_dts(ID_DENS,rt,ii) * flux_dts(ID_RHOT,kt+1-rt,ii);
          if (rt <= kt) {
            tot3 += (kt+1-rt) * ( gamm * rtgamm_dts(rt,ii) * state_dts(ID_RHOT,kt+1-rt,ii) - state_dts(ID_RHOT,rt,ii) * rtgamm_dts(kt+1-rt,ii) );
          }
        }
        ruu_dts   (kt+1,ii) = tot1 / state_dts(ID_DENS,0,ii);
        rtgamm_dts(kt+1,ii) = ( gamm * rtgamm_dts(0,ii) *state_dts(ID_RHOT,kt+1,ii) + tot3 / (kt+1._fp) ) / state_dts(ID_RHOT,0,ii);
        flux_dts(ID_DENS,kt+1,ii) = state_dts(ID_UMOM,kt+1,ii);
        flux_dts(ID_UMOM,kt+1,ii) = ruu_dts(kt+1,ii) + c0*rtgamm_dts(kt+1,ii)/2._fp;
        flux_dts(ID_RHOT,kt+1,ii) = tot2 / state_dts(ID_DENS,0,ii);
      }
    }

    //Compute the time average using the derivatives
    real dtmult = dt;
    for (int kt=1; kt<tord; kt++) {
      for (int j=0; j<numVars; j++) {
        for (int ii=0; ii<tord; ii++) {
          state_dts(j,0,ii) += state_dts(j,kt,ii) * dtmult / (kt+1._fp);
          flux_dts (j,0,ii) += flux_dts (j,kt,ii) * dtmult / (kt+1._fp);
        }
      }
      dtmult *= dt;
    }

    // Store the limits of the flux reconstructed from this cell
    for (int j=0; j<numVars; j++) {
      fluxLimits (j,i  ,1) = flux_dts (j,0,0     );
      stateLimits(j,i  ,1) = state_dts(j,0,0     );
      fluxLimits (j,i+1,0) = flux_dts (j,0,tord-1);
      stateLimits(j,i+1,0) = state_dts(j,0,tord-1);
    }
  }
};


class enforceFluxBoundariesClass {
public:
  _YAKL void operator() (ulong i, Array<real> &stateLimits , Array<real> &fluxLimits ) {
    for (int j=0; j<numVars; j++) {
      fluxLimits (j,0 ,0) = fluxLimits (j,0 ,1);
      stateLimits(j,0 ,0) = stateLimits(j,0 ,1);
      fluxLimits (j,nx,1) = fluxLimits (j,nx,0);
      stateLimits(j,nx,1) = stateLimits(j,nx,0);
    }
  }
};


class computeFluxClass {
public:
  _YAKL void operator() ( ulong i , Array<real> &stateLimits , Array<real> &fluxLimits , Array<real> &flux ) {
    real tol = 1e-7_fp;
    SArray<real,numVars> w;
    // Compute Numerical Fluxes
    // for (int i=0; i<nx+1; i++) {
    // Compute interface fluid state values
    real r = 0.5_fp * ( stateLimits(ID_DENS,i,0) + stateLimits(ID_DENS,i,1) );
    real u = 0.5_fp * ( stateLimits(ID_UMOM,i,0) + stateLimits(ID_UMOM,i,1) ) / r;
    real t = 0.5_fp * ( stateLimits(ID_RHOT,i,0) + stateLimits(ID_RHOT,i,1) ) / r;
    real p = c0 * mypow( r*t , gamm );
    real c = mysqrt(gamm * p / r);

    if (u > tol) {
      w(0) =  fluxLimits(0,i,0) - fluxLimits(2,i,0)/t;
    } else {
      w(0) =  fluxLimits(0,i,1) - fluxLimits(2,i,1)/t;
    }
    if (u-c > tol) {
      w(1) =  fluxLimits(0,i,0)*u/(2*c) - fluxLimits(1,i,0)/(2*c) + fluxLimits(2,i,0)/(2*t);
    } else {
      w(1) =  fluxLimits(0,i,1)*u/(2*c) - fluxLimits(1,i,1)/(2*c) + fluxLimits(2,i,1)/(2*t);
    }
    if (u+c > tol) {
      w(2) = -fluxLimits(0,i,0)*u/(2*c) + fluxLimits(1,i,0)/(2*c) + fluxLimits(2,i,0)/(2*t);
    } else {
      w(2) = -fluxLimits(0,i,1)*u/(2*c) + fluxLimits(1,i,1)/(2*c) + fluxLimits(2,i,1)/(2*t);
    }

    flux(0,i) = w(0)   + w(1)       + w(2)      ;
    flux(1,i) = w(0)*u + w(1)*(u-c) + w(2)*(u+c);
    flux(2,i) =          w(1)*t     + w(2)*t    ;
  }
};


class computeTendenciesClass {
public:
  _YAKL void operator() ( ulong i , Array<real> &flux , Array<real> &tend ) {
    // Compute tendencies
    for (int j=0; j<numVars; j++) {
    //for (int i=0; i<nx; i++) {
      tend(j,i) = - ( flux(j,i+1) - flux(j,i) ) / dx;
    }
  }
};


class applyTendenciesClass {
public:
  _YAKL void operator() ( ulong i , Array<real> &state , Array<real> &tend , real dt ) {
    for (int j=0; j<numVars; j++) {
    //for (int i=0; i<nx; i++) {
      state(j,hs+i) = state(j,hs+i) + dt * tend(j,i);
    }
  }
};


void tendendies(Array<real> const &state, SArray<real,ord,ord,ord> &wenoRecon, SArray<real,ord,tord> const &c2g_lower,
                SArray<real,tord,tord> const &deriv, real const dt, Array<real> &flux, Array<real> &tend) {
  Array<real> fluxLimits;
  Array<real> stateLimits;

  stateLimits.setup(numVars,nx+1,2);
  fluxLimits .setup(numVars,nx+1,2);

  yakl::Launcher<yakl::targetCPUSerial> launcher;

  reconAderClass reconAder;
  launcher.parallelFor( nx , reconAder , state , wenoRecon , c2g_lower , deriv , dt , fluxLimits , stateLimits );

  enforceFluxBoundariesClass enforceFluxBoundaries;
  launcher.parallelFor( 1 , enforceFluxBoundaries , stateLimits , fluxLimits );

  computeFluxClass computeFlux;
  launcher.parallelFor( nx+1 , computeFlux , stateLimits , fluxLimits , flux );

  computeTendenciesClass computeTendencies;
  launcher.parallelFor( nx , computeTendencies , flux , tend );
}


int main() {
  Array<real> state;
  Array<real> flux;
  Array<real> tend;
  SArray<real,ord,ord,ord> c2g_lower_tmp;
  SArray<real,ord,tord> c2g_lower;
  SArray<real,ord,ord,ord> wenoRecon;
  SArray<real,tord,tord> g2c, c2g, c2d, deriv;
  TransformMatrices<real> transform;
  int n, num_steps;
  real etime;
  real dt;

  state.setup(numVars,nx+2*hs);
  flux .setup(numVars,nx+1);
  tend .setup(numVars,nx);

  transform.weno_sten_to_coefs( wenoRecon );

  transform.gll_to_coefs  ( g2c);
  transform.coefs_to_deriv( c2d);
  transform.coefs_to_gll  ( c2g);
  deriv = ( c2g * c2d * g2c ) / dx;

  transform.coefs_to_gll_lower( c2g_lower_tmp );
  for (int j=0; j<ord; j++) {
    for (int i=0; i<tord; i++) {
      c2g_lower(j,i) = c2g_lower_tmp(tord-1,j,i);
    }
  }

  state = 0.;
  for (int i=hs; i<nx+hs; i++) {
    if (i < nx/2) {
      state(ID_DENS,i) = 1._fp;
      state(ID_RHOT,i) = mypow(1.0_fp/c0,1._fp/gamm); // p = 1
    } else {
      state(ID_DENS,i) = 0.125_fp;
      state(ID_RHOT,i) = mypow(0.1_fp/c0,1._fp/gamm); // p = 0.1
    }
  }

  computeTimeStep(state, dt);

  yakl::Launcher<yakl::targetCPUSerial> launcher;

  etime = 0.;
  while (etime < sim_time) {
  // for (int i=0; i<20; i++) {
    // computeTimeStep(state, dt);
    if (etime + dt > sim_time) { dt = sim_time - etime; }

    boundariesClass boundaries;
    launcher.parallelFor( 1 , boundaries , state );
    tendendies(state, wenoRecon, c2g_lower, deriv, dt, flux, tend);

    applyTendenciesClass applyTendencies;
    launcher.parallelFor( nx , applyTendencies , state , tend , dt );

    etime += dt;
    // std::cout << etime << "\n";
  }

  launcher.synchronize();

  for (int i=hs; i<nx+hs; i++) {
    std::cout << (i-hs+0.5) / nx << " " << state(0,i) << " " << state(1,i)/state(0,i) << " " << c0*mypow(state(2,i),gamm) << "\n";
  }

}
