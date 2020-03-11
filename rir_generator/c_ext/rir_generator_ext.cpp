#define _USE_MATH_DEFINES
#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "math.h"
#include "numpy/arrayobject.h"

#define ROUND(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

static PyObject* generateRIR(PyObject *self, PyObject *args);

static PyMethodDef Methods[] = {
    {"generate_rir_ext", generateRIR, METH_VARARGS, "Generate room impulse responses."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rir_generator_module = {
    PyModuleDef_HEAD_INIT,
    "rir_generator_ext",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC
PyInit_c_ext(void) {
    import_array();
    PyObject* m = PyModule_Create(&rir_generator_module);
    if (!m) {
        return NULL;
    }
    return m;
}

double sinc(double x)
{
    if (x == 0)
        return(1.);
    else
        return(sin(x)/x);
}

double sim_microphone(double x, double y, double z, double* angle, char mtype)
{
    if (mtype=='b' || mtype=='c' || mtype=='s' || mtype=='h')
    {
        double gain, vartheta, varphi, rho;

        // Polar Pattern         rho
        // ---------------------------
        // Bidirectional         0
        // Hypercardioid         0.25
        // Cardioid              0.5
        // Subcardioid           0.75
        // Omnidirectional       1

        switch(mtype)
        {
        case 'b':
            rho = 0;
            break;
        case 'h':
            rho = 0.25;
            break;
        case 'c':
            rho = 0.5;
            break;
        case 's':
            rho = 0.75;
            break;
        };

        vartheta = acos(z/sqrt(pow(x,2)+pow(y,2)+pow(z,2)));
        varphi = atan2(y,x);

        gain = sin(M_PI/2-angle[1]) * sin(vartheta) * cos(angle[0]-varphi) + cos(M_PI/2-angle[1]) * cos(vartheta);
        gain = rho + (1-rho) * gain;

        return gain;
    }
    else
    {
        return 1;
    }
}

static PyObject* generateRIR(PyObject *self, PyObject *args)
{
    // Load parameters
    double          c;
    double          fs;
    PyArrayObject*  rrArray;
    PyArrayObject*  ssArray;
    PyArrayObject*  LLArray;
    PyArrayObject*  beta_input;
    double*         beta = new double[6];
    int             nSamples;
    char*           microphone_type;
    int             nOrder;
    int             nDimension;
    PyArrayObject*  angleArray;
    int             isHighPassFilter;
    double          reverberation_time = 0;

    if (!PyArg_ParseTuple(args, "ddOOOOisiiOp",
                          &c, &fs, &rrArray, &ssArray, &LLArray, &beta_input, &nSamples,
                          &microphone_type, &nOrder, &nDimension, &angleArray, &isHighPassFilter)) {
        return NULL;
    }

    auto rr = (double*) rrArray->data;
    auto ss = (double*) ssArray->data;
    auto LL = (double*) LLArray->data;
    auto angle = (double*) angleArray->data;
    int  nMicrophones = (int) PyArray_DIM(rrArray, 0);

    // Reflection coefficients or reverberation time?
    if (!PyArray_DIMS(beta_input))
    {
        double V = LL[0]*LL[1]*LL[2];
        double S = 2*(LL[0]*LL[2]+LL[1]*LL[2]+LL[0]*LL[1]);
        reverberation_time = *((double*) beta_input->data);
        if (reverberation_time != 0) {
            double alfa = 24*V*log(10.0)/(c*S*reverberation_time);
            if (alfa > 1) {
                PyErr_SetString(
                    PyExc_ValueError,
                    "Error: The reflection coefficients cannot be calculated using the current "
                    "room parameters, i.e. room size and reverberation time.\n           Please "
                    "specify the reflection coefficients or change the room parameters.");
                return NULL;
            }
            for (int i=0;i<6;i++)
                beta[i] = sqrt(1-alfa);
        }
        else
        {
            for (int i=0;i<6;i++)
                beta[i] = 0;
        }
    }
    else
    {
        for (int i=0;i<6;i++)
            beta[i] = ((double*) (beta_input->data))[i];
    }

    // Room Dimension
    if (nDimension == 2)
    {
        beta[4] = 0;
        beta[5] = 0;
    }

    // Number of samples (optional)
    if (nSamples == -1)
    {
        if (PyArray_DIMS(beta_input))
        {
            double V = LL[0]*LL[1]*LL[2];
            double alpha = ((1-pow(beta[0],2))+(1-pow(beta[1],2)))*LL[1]*LL[2] +
                ((1-pow(beta[2],2))+(1-pow(beta[3],2)))*LL[0]*LL[2] +
                ((1-pow(beta[4],2))+(1-pow(beta[5],2)))*LL[0]*LL[1];
            reverberation_time = 24*log(10.0)*V/(c*alpha);
            if (reverberation_time < 0.128)
                reverberation_time = 0.128;
        }
        nSamples = (int) (reverberation_time * fs);
    }

    // Create output vector
    npy_intp dims[] = {nMicrophones, nSamples};
    auto h = (PyArrayObject*) PyArray_ZEROS(2, dims, 12, 1);
    double* imp = (double*) h->data;

    // Temporary variables and constants (high-pass filter)
    const double W = 2*M_PI*100/fs; // The cut-off frequency equals 100 Hz
    const double R1 = exp(-W);
    const double B1 = 2*R1*cos(W);
    const double B2 = -R1 * R1;
    const double A1 = -(1+R1);
    double       X0;
    double*      Y = new double[3];

    // Temporary variables and constants (image-method)
    const double Fc = 1; // The cut-off frequency equals fs/2 - Fc is the normalized cut-off frequency.
    const int    Tw = 2 * ROUND(0.004*fs); // The width of the low-pass FIR equals 8 ms
    const double cTs = c/fs;
    double*      LPI = new double[Tw];
    double*      r = new double[3];
    double*      s = new double[3];
    double*      L = new double[3];
    double       Rm[3];
    double       Rp_plus_Rm[3];
    double       refl[3];
    double       fdist,dist;
    double       gain;
    int          startPosition;
    int          n1, n2, n3;
    int          q, j, k;
    int          mx, my, mz;
    int          n;

    s[0] = ss[0]/cTs; s[1] = ss[1]/cTs; s[2] = ss[2]/cTs;
    L[0] = LL[0]/cTs; L[1] = LL[1]/cTs; L[2] = LL[2]/cTs;

    for (int idxMicrophone = 0; idxMicrophone < nMicrophones ; idxMicrophone++)
    {
        // [x_1 x_2 ... x_N y_1 y_2 ... y_N z_1 z_2 ... z_N]
        r[0] = rr[idxMicrophone + 0*nMicrophones] / cTs;
        r[1] = rr[idxMicrophone + 1*nMicrophones] / cTs;
        r[2] = rr[idxMicrophone + 2*nMicrophones] / cTs;

        n1 = (int) ceil(nSamples/(2*L[0]));
        n2 = (int) ceil(nSamples/(2*L[1]));
        n3 = (int) ceil(nSamples/(2*L[2]));

        // Generate room impulse response
        for (mx = -n1 ; mx <= n1 ; mx++)
        {
            Rm[0] = 2*mx*L[0];

            for (my = -n2 ; my <= n2 ; my++)
            {
                Rm[1] = 2*my*L[1];

                for (mz = -n3 ; mz <= n3 ; mz++)
                {
                    Rm[2] = 2*mz*L[2];

                    for (q = 0 ; q <= 1 ; q++)
                    {
                        Rp_plus_Rm[0] = (1-2*q)*s[0] - r[0] + Rm[0];
                        refl[0] = pow(beta[0], abs(mx-q)) * pow(beta[1], abs(mx));

                        for (j = 0 ; j <= 1 ; j++)
                        {
                            Rp_plus_Rm[1] = (1-2*j)*s[1] - r[1] + Rm[1];
                            refl[1] = pow(beta[2], abs(my-j)) * pow(beta[3], abs(my));

                            for (k = 0 ; k <= 1 ; k++)
                            {
                                Rp_plus_Rm[2] = (1-2*k)*s[2] - r[2] + Rm[2];
                                refl[2] = pow(beta[4],abs(mz-k)) * pow(beta[5], abs(mz));

                                dist = sqrt(pow(Rp_plus_Rm[0], 2) + pow(Rp_plus_Rm[1], 2) + pow(Rp_plus_Rm[2], 2));

                                if (abs(2*mx-q)+abs(2*my-j)+abs(2*mz-k) <= nOrder || nOrder == -1)
                                {
                                    fdist = floor(dist);
                                    if (fdist < nSamples)
                                    {
                                        gain = sim_microphone(Rp_plus_Rm[0], Rp_plus_Rm[1], Rp_plus_Rm[2], angle, microphone_type[0])
                                            * refl[0]*refl[1]*refl[2]/(4*M_PI*dist*cTs);

                                        for (n = 0 ; n < Tw ; n++)
                                            LPI[n] =  0.5 * (1 - cos(2*M_PI*((n+1-(dist-fdist))/Tw))) * Fc * sinc(M_PI*Fc*(n+1-(dist-fdist)-(Tw/2)));

                                        startPosition = (int) fdist-(Tw/2)+1;
                                        for (n = 0 ; n < Tw; n++)
                                            if (startPosition+n >= 0 && startPosition+n < nSamples)
                                                imp[idxMicrophone + nMicrophones*(startPosition+n)] += gain * LPI[n];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 'Original' high-pass filter as proposed by Allen and Berkley.
        if (isHighPassFilter == 1)
        {
            for (int idx = 0 ; idx < 3 ; idx++) {Y[idx] = 0;}
            for (int idx = 0 ; idx < nSamples ; idx++)
            {
                X0 = imp[idxMicrophone+nMicrophones*idx];
                Y[2] = Y[1];
                Y[1] = Y[0];
                Y[0] = B1*Y[1] + B2*Y[2] + X0;
                imp[idxMicrophone+nMicrophones*idx] = Y[0] + A1*Y[1] + R1*Y[2];
            }
        }
    }

    PyObject* beta_hat;
    if (reverberation_time != 0) {
        beta_hat = PyFloat_FromDouble(beta[0]);
    } else {
        Py_INCREF(Py_None);
        beta_hat = Py_None;
    }

    delete[] beta;
    delete[] Y;
    delete[] LPI;
    delete[] r;
    delete[] s;
    delete[] L;

    auto ret = PyTuple_Pack(2, h, beta_hat);
    return ret;
}
