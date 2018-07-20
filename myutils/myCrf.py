import numpy as np
import pydensecrf.densecrf as dcrf

def dense_crf(img, output_probs):
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs+1e-20)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=30, compat=3)
    d.addPairwiseBilateral(sxy=40, srgb=20, rgbim=img, compat=10)

    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
