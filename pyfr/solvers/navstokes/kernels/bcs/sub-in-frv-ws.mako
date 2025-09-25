<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>

<% tau = c['ldg-tau'] %>

<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ${c['rho']};
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = (${c['rho']}) * (${c[v]});
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl, nl, magnl'>
    // Viscous states
    fpdtype_t ur[${nvars}], gradur[${ndims}][${nvars}];
    ${pyfr.expand('bc_ldg_state', 'ul', 'nl', 'ur')};
    ${pyfr.expand('bc_ldg_grad_state', 'ul', 'nl', 'gradul', 'gradur')};

    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
    ${pyfr.expand('artificial_viscosity_add', 'gradur', 'fvr', 'artviscl')};

    // Inviscid state
    fpdtype_t ucomm[${nvars}];
    fpdtype_t ficommtensor[${ndims}][${nvars}], ficomm[${nvars}], fvcomm, fl[${ndims}][${nvars}];
    fpdtype_t vcomm[${ndims}], vl[${ndims}];
    fpdtype_t pcomm, pl;

    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl')};

    ucomm[0] = ${c['rho']};
    //ucomm[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ucomm[${i + 1}] = ucomm[0] * (${c[v]});
% endfor
    //ucomm[${nvars - 1}] = ${1.0/(c['gamma'] - 1.0)}*${c['p']} + 0.5*(1.0/ucomm[0])*${pyfr.dot('ucomm[{i}]', i=(1, ndims + 1))}; // possibly ill-posed
    ucomm[${nvars - 1}] = ${1.0/(c['gamma'] - 1.0)}*pl + 0.5*(1.0/ucomm[0])*${pyfr.dot('ucomm[{i}]', i=(1, ndims + 1))};

    ${pyfr.expand('inviscid_flux', 'ucomm', 'ficommtensor', 'pcomm', 'vcomm')};

% for i in range(nvars):
    ficomm[${i}] = ${' + '.join(f'nl[{j}]*ficommtensor[{j}][{i}]' for j in range(ndims))};
    fvcomm = ${' + '.join(f'nl[{j}]*fvr[{j}][{i}]' for j in range(ndims))};
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ur[${i}]);
% endif
    ul[${i}] = magnl*(ficomm[${i}] + fvcomm);
% endfor
</%pyfr:macro>
