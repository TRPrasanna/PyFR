<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<% eps = 0.001 %>

<%pyfr:macro name='rsolve_1d' params='ul, ur, nf'>
    // Low-Mach Roe (LM-Roe):
    // Scale the normal-velocity jump in the acoustic parts by
    //   beta = min(1, |va[0]| / a)
    // where va is the Roe-averaged velocity and a the Roe-averaged sound speed.

    // Left/right fluxes, velocities, pressures
    fpdtype_t fl[${nvars}], fr[${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;

    // Roe-averaged velocity and component jumps
    fpdtype_t va[${ndims}], dv[${ndims}];

    ${pyfr.expand('inviscid_flux_1d', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux_1d', 'ur', 'fr', 'pr', 'vr')};

    // Roe-averaged density square roots
    fpdtype_t srl = sqrt(ul[0]);
    fpdtype_t srr = sqrt(ur[0]);

    // Roe-averaged density product and enthalpy
    fpdtype_t roa = srl*srr;
    fpdtype_t ha  = (srl*(pr + ur[${nvars - 1}]) + srr*(pl + ul[${nvars - 1}])) /
                    (srl*ur[0] + srr*ul[0]);

    // Roe-averaged velocity
    fpdtype_t inv_rsum = 1.0 / (srl + srr);
% for i in range(ndims):
    va[${i}] = (vl[${i}]*srl + vr[${i}]*srr) * inv_rsum;
% endfor

    // Roe-averaged sound speed
    fpdtype_t qq = ${pyfr.dot('va[{i}]', i=ndims)};
    fpdtype_t a  = sqrt(${c['gamma'] - 1}*(ha - 0.5*qq));
    fpdtype_t a_safe = (a > 1e-14) ? a : 1e-14;

    // Eigenvalues (absolute)
    fpdtype_t l1 = fabs(va[0] - a);
    fpdtype_t l2 = fabs(va[0]);
    fpdtype_t l3 = fabs(va[0] + a);

    // Entropy fix on acoustic eigenvalues
    l1 = (l1 < ${eps}) ? ${0.5 / eps}*(l1*l1 + ${eps**2}) : l1;
    l3 = (l3 < ${eps}) ? ${0.5 / eps}*(l3*l3 + ${eps**2}) : l3;

    // State jumps
% for i in range(ndims):
    dv[${i}] = vr[${i}] - vl[${i}];
% endfor
    fpdtype_t dro = ur[0] - ul[0];
    fpdtype_t dp  = pr - pl;

    // Low-Mach scaling on the normal-velocity jump (acoustic only)
    fpdtype_t beta = fabs(va[0]) / a_safe;   // local normal Mach number
    beta = (beta < 1.0) ? beta : 1.0;
    fpdtype_t dv0_eff = beta * dv[0];

    fpdtype_t inv_a2 = 1.0 / (2.0*a_safe*a_safe);

    // Wave strengths for mass equation
    fpdtype_t v1 = (dp - roa*a_safe*dv0_eff)*inv_a2;
    fpdtype_t v2 = dro - dp*2.0*inv_a2;          // contact wave unchanged
    fpdtype_t v3 = (dp + roa*a_safe*dv0_eff)*inv_a2;
    nf[0] = 0.5*(fl[0] + fr[0]) - (l1*v1 + l2*v2 + l3*v3);

    // Momentum components
% for i in range(ndims):
% if i == 0:
    // Normal momentum uses dv0_eff inside acoustic parts
    v1 = (dp - roa*a_safe*dv0_eff)*inv_a2*(va[${i}] - a_safe);
    v2 = (dro - dp*2.0*inv_a2)*va[${i}];
    v3 = (dp + roa*a_safe*dv0_eff)*inv_a2*(va[${i}] + a_safe);
% else:
    // Tangential momentum: shear jump is dv[i], not scaled
    v1 = (dp - roa*a_safe*dv0_eff)*inv_a2*va[${i}];
    v2 = (dro - dp*2.0*inv_a2)*va[${i}] + roa*dv[${i}];
    v3 = (dp + roa*a_safe*dv0_eff)*inv_a2*va[${i}];
% endif
    nf[${i + 1}] = 0.5*(fl[${i + 1}] + fr[${i + 1}]) - (l1*v1 + l2*v2 + l3*v3);
% endfor

    // Energy component
    v1 = (dp - roa*a_safe*dv0_eff)*inv_a2*(ha - a_safe*va[0]);
    // Contact-wave piece v2: keep the original dv[0] here
    v2 = roa*(${pyfr.dot('va[{i}]', 'dv[{i}]', i=ndims)} - va[0]*dv[0]) +
         (dro - dp*2.0*inv_a2)*qq*0.5;
    v3 = (dp + roa*a_safe*dv0_eff)*inv_a2*(ha + a_safe*va[0]);
    nf[${nvars - 1}] = 0.5*(fl[${nvars - 1}] + fr[${nvars - 1}]) - (l1*v1 + l2*v2 + l3*v3);
</%pyfr:macro>

<%include file='pyfr.solvers.euler.kernels.rsolvers.rsolve1d'/>
