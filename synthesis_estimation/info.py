COLUMNS_TO_DELETE=[164,510,530,542,544,
                   548,549,550,552,558,
                   561,563,564,565,566,
                   568,570,571,572,573,
                   575,577,578,611,612,
                   617,618,619,621,622,
                   623,625,626,627,629,
                   630,631,632,633,696]

import numpy as np
import warnings

def compute_summary_statistics(data):
    summary_stats = {
        "mean": np.mean(data, axis=0),
        "median": np.median(data, axis=0),
        "std": np.std(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0)
    }
    return summary_stats

def getParamMap():

    vals = []
    #print(len(vals))

    vals += ["RF_IN_AMOUNT"]
    vals += ["RF_OUT_AMOUNT"]
    vals += ["RF_SIZE"]
    vals += [f"LSU_IN1_{x}" for x in range(8)]
    vals += [f"LSU_IN2_{x}" for x in range(8)]
    vals += [f"LSU_OUT_{x}" for x in range(8)]
    vals += [f"ALU_OUT_{x}" for x in range(8)]
    vals += ["ALU2_ENABLE"]
    vals += [f"ALU2_IN1_{x}" for x in range(8)]
    vals += [f"ALU2_IN2_{x}" for x in range(8)]
    vals += [f"ALU2_OUT_{x}" for x in range(8)]
    vals += ["MULDIV_TYPE_1"]
    vals += ["MULDIV_TYPE_2"]
    vals += [f"MULDIV_IN1_{x}" for x in range(8)]
    vals += [f"MULDIV_IN2_{x}" for x in range(8)]
    vals += [f"MULDIV_OUT_{x}" for x in range(8)]
    
    new_list = vals
    print(len(new_list))
    return new_list

def main():
    param_map = getParamMap()
    warnings.simplefilter("ignore")
    train_data = np.load("data/train_data.npy", mmap_mode='r')
    train_summary = compute_summary_statistics(train_data)
    if (False):
        it = 0
        for x in param_map:
            if it > -1:
                print("%3d | %-20s" % (it, x))
            it += 1
        
        assert(False)
    for i in range(len(train_summary["mean"])): #[0,1,2,3,4,5, 437, 582]:
        if train_summary["mean"][i] > -1:
            try:
                print("%d | %20s | %.3f" % (i, str(param_map[i]), train_summary["mean"][i]))
            except:
                print(train_summary["max"][i])
    
    #print(null_columns)


if __name__ == "__main__":
    main()