from dana import *
from learningOptions import *
from learningHardCodedOptions import *


def init_weights(L, gain=1):
    # Wmin, Wmax = 0.25, 0.75
    W = L._weights
    N = np.random.normal(Wmean, 0.005, W.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    L._weights = gain*W*(Wmin + (Wmax - Wmin)*N)

SNc_dop   = zeros((SNc_neurons,1), """  D2_IPSC = - alpha_DA_arD2 * DAtoD2c;
                             Ir = np.maximum(Irew, Ie_rew);
                             I = Ir + D2_IPSC;
                             n = noise(I,SNc_N);
                             It = I + n;
                             u = positiveClip(It - SNc_h);
                             dV/dt = (-V + u)/SNc_tau; Irew; Ie_rew; SNc_h; V_lag; DAtoD2c""")
                             # u/dt = (-u + (It - SNc_h))/SNc_tau;
                             # V = positiveClip(u); Irew; Ie_rew; SNc_h; V_lag; DAtoD2c""")
                             # n = correlatedNoise(I,n,SNc_N,alpha_Rew_DA,SNc_N_tau);

Cortex_cog   = zeros((n,1), """Is = I + Iext; 
                             n = noise(Is,Cortex_N);
                             It = Is + n;
                             u = positiveClip(It - Cortex_h);
                             dV/dt = (-V + u)/Cortex_tau; I; Iext""")
Cortex_mot   = zeros((1,n), """Is = I + Iext; 
                             n = noise(Is,Cortex_N);
                             It = Is + n;
                             u = positiveClip(It - Cortex_h);
                             dV/dt = (-V + u)/Cortex_tau; I; Iext""")
Cortex_ass   = zeros((n,n), """Is = I + Iext; 
                             n = noise(Is,Cortex_N);
                             It = Is + n;
                             u = positiveClip(It - Cortex_h);
                             dV/dt = (-V + u)/Cortex_tau; I; Iext""")

Striatum_cog = zeros((n,1), """Is = I*ctxStr(DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

Striatum_mot = zeros((1,n), """Is = I*ctxStr(DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

Striatum_ass = zeros((n,n), """Is = I*ctxStr(DA);
                             n = striatalNoise(Is,n);
                             It = Is + n;
                             Vh = strThreshold(DA);
                             Vc = strSlope(DA);
                             u = sigmoid(It - Striatum_h,Vmin,Vmax,Vh,Vc);
                             dV/dt = (-V + u)/Striatum_tau; I; DA""")

STN_cog      = zeros((n,1), """Is = I;
                             n = noise(Is,STN_N);
                             It = Is + n;
                             u = positiveClip(It - STN_h);
                             dV/dt = (-V + u)/STN_tau; I""")
STN_mot      = zeros((1,n), """Is = I;
                             n = noise(Is,STN_N);
                             It = Is + n;
                             u = positiveClip(It - STN_h);
                             dV/dt = (-V + u)/STN_tau; I""")
GPi_cog      = zeros((n,1), """Is = I;
                             n = noise(Is,GPi_N);
                             It = Is + n;
                             u = positiveClip(It - GPi_h);
                             dV/dt = (-V + u)/GPi_tau; I""")
GPi_mot      = zeros((1,n), """Is = I;
                             n = noise(Is,GPi_N);
                             It = Is + n;
                             u = positiveClip(It - GPi_h);
                             dV/dt = (-V + u)/GPi_tau; I""")
Thalamus_cog = zeros((n,1), """Is = I;
                             n = noise(Is,Thalamus_N);
                             It = Is + n;
                             u = positiveClip(It - Thalamus_h);
                             dV/dt = (-V + u)/Thalamus_tau; I""")
Thalamus_mot = zeros((1,n), """Is = I;
                             n = noise(Is,Thalamus_N);
                             It = Is + n;
                             u = positiveClip(It - Thalamus_h);
                             dV/dt = (-V + u)/Thalamus_tau; I""")



# Connectivity
# -----------------------------------------------------------------------------
W = DenseConnection( Cortex_cog('V'),   Striatum_cog('I'), 1.0)
if cogInitialWeights == None:
    init_weights(W)
else:
    W._weights = cogInitialWeights
W_cortex_cog_to_striatum_cog = W

if doPrint:
    print "Cognitive weights: ", np.diag(W._weights)

W = DenseConnection( Cortex_mot('V'),   Striatum_mot('I'), 1.0)
if motInitialWeights != None:
    W._weights = motInitialWeights
else:
    init_weights(W)
W_cortex_mot_to_striatum_mot = W

W = DenseConnection( Cortex_ass('V'),   Striatum_ass('I'), 1.0)
init_weights(W)
W = DenseConnection( Cortex_cog('V'),   Striatum_ass('I'), np.ones((1,2*n-1)))
init_weights(W,0.2)
W = DenseConnection( Cortex_mot('V'),   Striatum_ass('I'), np.ones((2*n-1,1)))
init_weights(W,0.2)
DenseConnection( Cortex_cog('V'),   STN_cog('I'),       1.0 )
DenseConnection( Cortex_mot('V'),   STN_mot('I'),       1.0 )
if n_availableOptions == 3:
    DenseConnection( Striatum_cog('V'), GPi_cog('I'),      -2.4 )
    DenseConnection( Striatum_mot('V'), GPi_mot('I'),      -2.4 )
else:
    DenseConnection( Striatum_cog('V'), GPi_cog('I'),      -2.4 )
    DenseConnection( Striatum_mot('V'), GPi_mot('I'),      -2.4 )
DenseConnection( Striatum_ass('V'), GPi_cog('I'),      -2.0*np.ones((1,2*n-1)))
DenseConnection( Striatum_ass('V'), GPi_mot('I'),      -2.0*np.ones((2*n-1,1)))
if n_availableOptions == 3:
    DenseConnection( STN_cog('V'),      GPi_cog('I'),       1.0*np.ones((2*n-1,1)))
    DenseConnection( STN_mot('V'),      GPi_mot('I'),       1.0*np.ones((1,2*n-1)))
else:
    DenseConnection( STN_cog('V'),      GPi_cog('I'),       1.0*np.ones((2*n-1,1)))
    DenseConnection( STN_mot('V'),      GPi_mot('I'),       1.0*np.ones((1,2*n-1)))
DenseConnection( GPi_cog('V'),      Thalamus_cog('I'), -0.5 )
DenseConnection( GPi_mot('V'),      Thalamus_mot('I'), -0.5 )
DenseConnection( Thalamus_cog('V'), Cortex_cog('I'),    1.0 )
DenseConnection( Thalamus_mot('V'), Cortex_mot('I'),    1.0 )
DenseConnection( Cortex_cog('V'),   Thalamus_cog('I'),  0.4 )
DenseConnection( Cortex_mot('V'),   Thalamus_mot('I'),  0.4 )
DenseConnection( SNc_dop('V'),      Striatum_cog('DA'), 1.0/DA_neurons )
DenseConnection( SNc_dop('V'),      Striatum_mot('DA'), 1.0/DA_neurons )
DenseConnection( SNc_dop('V'),      Striatum_ass('DA'), 1.0/DA_neurons )
