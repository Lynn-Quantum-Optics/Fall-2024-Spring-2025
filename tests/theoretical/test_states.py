import numpy as np
import finding_states.states_and_witnesses as states
import finding_states.operations as op

### Test States ###
TEST1 = np.cos(np.pi/8)*states.HH + np.sin(np.pi/8)*states.VV
TEST1_RHO = TEST1 @ op.adjoint(TEST1)