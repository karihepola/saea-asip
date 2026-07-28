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
    vals = [
    "MAX_ALU_OPS",
    "MAX_LOAD_OPS",
    "MAX_STORE_OPS",
    "MAX_MUL_OPS",
    "MAX_DIV_OPS",
    "MAX_JUMP_OPS",
    "MAX_LIMMS",
    "MAX_BOOLEAN_READ",
    "MAX_BOOLEAN_WRITE",
    "MAX_GUARDS",
]

    # Append values for MAX
    vals += [f"MAX_REG_ACC_{x}" for x in range(128)]
    vals += [f"MAX_BYPASS_{x}" for x in range(32)]
    vals += [f"MAX_RF_READ_{x}" for x in range(1, 9)]
    vals += [f"MAX_RF_WRITE_{x}" for x in range(1, 5)]
    vals += [f"MAX_MOVE_{x}" for x in range(255)]

    # Append values for MED
    vals += [
        "MED_CYCLE",
        "MED_ALU_OPS",
        "MED_LOAD_OPS",
        "MED_STORE_OPS",
        "MED_MUL_OPS",
        "MED_JUMP_OPS",
        "MED_LIMMS",
        "MED_BOOLEAN_READ",
        "MED_BOOLEAN_WRITE",
        "MED_GUARDS",
    ]
    vals += [f"MED_REG_ACC_{x}" for x in range(64)]
    vals += [f"MED_BYPASS_{x}" for x in range(12)]
    vals += [f"MED_RF_READ_{x}" for x in range(1, 3)]
    vals += [f"MED_RF_WRITE_{x}" for x in range(1, 2)]
    vals += [f"MED_MOVE_{x}" for x in range(56)]

    # Append values for MIN
    vals += [
        "MIN_CYCLE",
        "MIN_ALU_OPS",
        "MIN_LOAD_OPS",
        "MIN_STORE_OPS",
        "MIN_MUL_OPS",
        "MIN_JUMP_OPS",
        "MIN_LIMMS",
        "MIN_BOOLEAN_READ",
        "MIN_BOOLEAN_WRITE",
        "MIN_GUARDS",
    ]
    vals += [f"MIN_REG_ACC_{x}" for x in range(16)]
    vals += [f"MIN_BYPASS_{x}" for x in range(5)]
    vals += [f"MIN_RF_READ_{x}" for x in range(1, 2)]
    vals += [f"MIN_RF_WRITE_{x}" for x in range(1, 2)]
    vals += [f"MIN_MOVE_{x}" for x in range(21)]

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
    return new_list

def main():
    param_map = getParamMap()
    warnings.simplefilter("ignore")
    train_data = np.load("data/data_normalized.npy", mmap_mode='r')[:1000]
    train_summary = compute_summary_statistics(train_data)
        
    for i in range(len(train_summary["mean"])):
        if train_summary["mean"][i] > -1:
            try:
                print("%d | %20s | %.3f" % (i, str(param_map[i]), train_summary["mean"][i]))
            except:
                print(train_summary["max"][i])
    

if __name__ == "__main__":
    main()