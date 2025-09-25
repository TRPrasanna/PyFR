<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<% gamma = c['gamma'] %>
<% eps = 1.0e-3 %>
<% tiny = 1.0e-14 %>

<%pyfr:macro name='rsolve_1d' params='ul, ur, nf'>
    // HLLC+ (Chen et al., SISC 2020):
    // - Pressure fix: add A_p = pf * (0, nx, ny, S*)^T, here nx=1, ny=0
    // - Shear fix: add to transverse momenta: tcoef * (S_K/(S_K - S*)) * g * ΔV
    //   where tcoef = φL φR / (φR - φL), φK = ρ_K (S_K - U_K)

    // Left/right fluxes, velocities, pressures
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;

    // Star states
    fpdtype_t usl[${nvars}], usr[${nvars}];

    ${pyfr.expand('inviscid_flux_1d', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux_1d', 'ur', 'fr', 'pr', 'vr')};

    // Roe-averaged enthalpy and normal velocity (local frame: normal = index 0)
    fpdtype_t srl = sqrt(ul[0]);
    fpdtype_t srr = sqrt(ur[0]);
    fpdtype_t H = (srl*(pr + ur[${ndims + 1}]) + srr*(pl + ul[${ndims + 1}]))
                / (srl*ur[0] + srr*ul[0]);

    fpdtype_t u = (srl*vl[0] + srr*vr[0]) / (srl + srr);
    fpdtype_t a = sqrt((${gamma} - 1.0)*(H - 0.5*u*u));

    // HLLC wave speeds and contact speed
    fpdtype_t sl = u - a;
    fpdtype_t sr = u + a;
    fpdtype_t sstar = (pr - pl + ul[0]*vl[0]*(sl - vl[0]) - ur[0]*vr[0]*(sr - vr[0]))
                    / (ul[0]*(sl - vl[0]) - ur[0]*(sr - vr[0]) + ${tiny});

    // Common factors for star states
    fpdtype_t ul_com = (sl - vl[0]) / (sl - sstar + ${tiny});
    fpdtype_t ur_com = (sr - vr[0]) / (sr - sstar + ${tiny});

    // Star state mass
    usl[0] = ul_com*ul[0];
    usr[0] = ur_com*ur[0];

    // Star state momentum
    usl[1] = ul_com*ul[0]*sstar;   // normal momentum
    usr[1] = ur_com*ur[0]*sstar;
% for i in range(2, ndims + 1):
    usl[${i}] = ul_com*ul[${i}];   // transverse momentum preserved across contact
    usr[${i}] = ur_com*ur[${i}];
% endfor

    // Star state energy (standard HLLC form)
    usl[${nvars - 1}] = ul_com*(ul[${nvars - 1}] + (sstar - vl[0])*(ul[0]*sstar + pl/(sl - vl[0] + ${tiny})));
    usr[${nvars - 1}] = ur_com*(ur[${nvars - 1}] + (sstar - vr[0])*(ur[0]*sstar + pr/(sr - vr[0] + ${tiny})));

    // ==== HLLC+ ingredients ====

    // Local sound speeds and speeds for Mach and sensors
    fpdtype_t aL = sqrt(${gamma}*pl/(ul[0] + ${tiny}));
    fpdtype_t aR = sqrt(${gamma}*pr/(ur[0] + ${tiny}));
    fpdtype_t UL = vl[0], UR = vr[0];

    // |u| for Mach function (use full speed magnitude in the local frame)
    fpdtype_t v2L = 0.0, v2R = 0.0;
% for i in range(ndims):
    v2L += vl[${i}]*vl[${i}];
    v2R += vr[${i}]*vr[${i}];
% endfor
    fpdtype_t ML = sqrt(v2L)/(aL + ${tiny});
    fpdtype_t MR = sqrt(v2R)/(aR + ${tiny});
    fpdtype_t Mloc = ML > MR ? ML : MR;
    Mloc = Mloc < 1.0 ? Mloc : 1.0;   // clamp to [0,1]

    // f(M) from the paper (Eq. 3.13)
    fpdtype_t mm = Mloc*Mloc;
    fpdtype_t fM = Mloc*sqrt(4.0 + (1.0 - mm)*(1.0 - mm)) / (1.0 + mm);

    // Simple shock sensor for f*: set to 1 near shocks (Eq. 3.16 style, interface-local)
    int shock_flag = ((UL - aL > 0.0 && UR - aR < 0.0) || (UL + aL > 0.0 && UR + aR < 0.0));
    fpdtype_t fstar = shock_flag ? 1.0 : fM;

    // Pressure-based function g = 1 - h*M, with h ≈ min(pL/pR, pR/pL) at this interface
    fpdtype_t ratio = pl/pr;
    ratio = ratio < 0 ? -ratio : ratio;
    fpdtype_t invratio = pr/(pl + ${tiny});
    invratio = invratio < 0 ? -invratio : invratio;
    fpdtype_t h = ratio < invratio ? ratio : invratio;
    if (h > 1.0) h = 1.0;
    fpdtype_t g = 1.0 - h*Mloc;
    if (g < 0.0) g = 0.0;
    if (g > 1.0) g = 1.0;

    // φL, φR and coefficients for pd and shear terms
    fpdtype_t phiL = ul[0]*(sl - UL);
    fpdtype_t phiR = ur[0]*(sr - UR);
    fpdtype_t denom_phi = phiR - phiL;
    fpdtype_t tcoef = (fabs(denom_phi) > ${tiny}) ? (phiL*phiR/denom_phi) : 0.0;

    // ΔU and ΔV_i (local frame)
    fpdtype_t dU = UR - UL;
    fpdtype_t dV[${ndims}];
% for i in range(ndims):
    dV[${i}] = vr[${i}] - vl[${i}];
% endfor

    // pd and pf = (f* - 1)*pd
    fpdtype_t pd = tcoef*dU;
    fpdtype_t pf = (fstar - 1.0)*pd;

    // Shear viscosity factors S_K/(S_K - S*) for K=L,R
    fpdtype_t facL = sl/(sl - sstar + ${tiny});
    fpdtype_t facR = sr/(sr - sstar + ${tiny});

    // ==== Compose fluxes with HLLC+ corrections ====
    fpdtype_t fsl_i, fsr_i;

% for i in range(nvars):
    // Base HLLC star fluxes
    fsl_i = fl[${i}] + sl*(usl[${i}] - ul[${i}]);
    fsr_i = fr[${i}] + sr*(usr[${i}] - ur[${i}]);

    // Pressure fix adds to normal momentum (i==1) and energy (i==nvars-1)
% if i == 1:
    fsl_i += pf;         // nx=1 in local frame
    fsr_i += pf;
% elif i == (nvars - 1):
    fsl_i += pf*sstar;   // energy component uses S*
    fsr_i += pf*sstar;
% endif

    // Shear viscosity goes into transverse momentum components only (indices 2..ndims)
% if 2 <= i <= ndims:
    // index in velocity array for this transverse component is (i-1)
    fsl_i += tcoef * facL * g * dV[${i - 1}];
    fsr_i += tcoef * facR * g * dV[${i - 1}];
% endif

    // Upwind selection with corrected star fluxes
    nf[${i}] = (0.0 <= sl) ? fl[${i}]
             : (sl <= 0.0 && 0.0 <= sstar) ? fsl_i
             : (sstar <= 0.0 && 0.0 <= sr) ? fsr_i
             : fr[${i}];
% endfor
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve1d'/>
