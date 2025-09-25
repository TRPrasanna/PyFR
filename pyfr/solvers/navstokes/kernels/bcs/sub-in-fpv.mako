<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur' externs='ploc, t'>
    ur[0] = ul[0];
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = ul[0] * (${c[v]});
% endfor
    ur[${nvars - 1}] = (${c['p']}) / ((${c['gamma']}) - 1.0) + 0.5 * (1.0/ur[0]) * ${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_state' func='bc_rsolve_state'/>
<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>
