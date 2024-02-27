import numpy as np
import qec.codes
import scipy.sparse

def test_hq_codes():
    SAVE = False
    
    qcode = qec.codes.HQ12Code()
    qcode.estimate_min_distance(reduce_logical_basis=True)
    print(qcode)
    
    if SAVE:
        scipy.sparse.save_npz("HQ_544_80_12_hx.npz", qcode.hx)
        scipy.sparse.save_npz("HQ_544_80_12_lx.npz", qcode.lx)
        scipy.sparse.save_npz("HQ_544_80_12_hz.npz", qcode.hz)
        scipy.sparse.save_npz("HQ_544_80_12_lz.npz", qcode.lz)


    qcode = qec.codes.HQ16Code()
    qcode.estimate_min_distance(reduce_logical_basis=True)
    print(qcode)
    
    if SAVE:
        scipy.sparse.save_npz("HQ_714_100_16_hx.npz", qcode.hx)
        scipy.sparse.save_npz("HQ_714_100_16_lx.npz", qcode.lx)
        scipy.sparse.save_npz("HQ_714_100_16_hz.npz", qcode.hz)
        scipy.sparse.save_npz("HQ_714_100_16_lz.npz", qcode.lz)

    qcode = qec.codes.HQ20Code()
    qcode.estimate_min_distance(reduce_logical_basis=True)
    print(qcode)

    if SAVE:
        scipy.sparse.save_npz("HQ_1020_136_20_hx.npz", qcode.hx)
        scipy.sparse.save_npz("HQ_1020_136_20_lx.npz", qcode.lx)
        scipy.sparse.save_npz("HQ_1020_136_20_hz.npz", qcode.hz)
        scipy.sparse.save_npz("HQ_1020_136_20_lz.npz", qcode.lz)
        
    qcode = qec.codes.HQ24Code()
    qcode.estimate_min_distance(reduce_logical_basis=True)
    print(qcode)

    if SAVE:
        scipy.sparse.save_npz("HQ_1428_184_24_hx.npz", qcode.hx)
        scipy.sparse.save_npz("HQ_1428_184_24_lx.npz", qcode.lx)
        scipy.sparse.save_npz("HQ_1428_184_24_hz.npz", qcode.hz)
        scipy.sparse.save_npz("HQ_1428_184_24_lz.npz", qcode.lz)

if __name__ == "__main__":
    test_hq_codes()