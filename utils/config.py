USE_ENSEMBLE_MODELS : bool = True
USE_XGBOOST  : bool = False
UWO_LAMBDA : float = 0
USE_MIN_PROGRAM_MODEL : bool = False

USE_HETERO_MODELS: bool = False

UWO_USE_MIN : bool = False

UWO_ROBUST_MEAN : bool = False

USE_SIMULATOR : bool = False
USE_HOT_START : bool = False

NUMBER_OF_ENSEMBLES = 10

USE_WARM : bool = False

SNR_UWO_LAMBDA : float = 0
SNR_UWO_ENABLE : bool = False

USE_PNN : bool = False


####
USE_SYNTHESIS_ENSEMBLE_MODELS : bool = False
###

#### FOR GENERAL PROGRAM ESTIMATOR ####

# If True, only use 6 program params, else all of them
GENERAL_PROGRAM_MINIMAL_DATA    : bool = False
GENERAL_PROGRAM_LEARNING_RATE   : float = 0.0003
GENERAL_PROGRAM_WEIGHT_DECAY    : float = 0.00004

MIN_GENERAL_PROGRAM_NUM_INPUTS   : int = 130
MIN_GENERAL_PROGRAM_MODEL_NAME   : str = "min_general_program_model.pth"
MIN_GENERAL_PROGRAM_NUM_FEATURES : int = 200
GENERAL_PROGRAM_NUM_INPUTS       : int = 722
GENERAL_PROGRAM_NUM_FEATURES     : int = 512
GENERAL_PROGRAM_DEPTH            : int = 2
GENERAL_PROGRAM_NORMALIZATION    : str = "layernorm"
GENERAL_PROGRAM_ACTIVATION       : str = "relu"

GENERAL_PROGRAM_MODEL_NAME      : str = "min_general_program_model.pth" if GENERAL_PROGRAM_MINIMAL_DATA else "general_program_model.pth"
GENERAL_PROGRAM_DATA_NAME       : str = "min-data.npy" if GENERAL_PROGRAM_MINIMAL_DATA else "data.npy"
#######################################

####### FOR SYNTHESIS ESTIMATOR #######

SYNTHESIS_LEARNING_RATE         : float = 0.0003
SYNTHESIS_WEIGHT_DECAY          : float = 0.00004

SYNTHESIS_NUM_INPUTS            : int = 86
SYNTHESIS_NUM_FEATURES          : int = 128
SYNTHESIS_DEPTH                 : int = 2
SYNTHESIS_NORMALIZATION         : str = "layernorm"
SYNTHESIS_ACTIVATION            : str = "relu"

SYNTHESIS_MODEL_NAME            : str = "synthesis_model.pth"
SYNTHESIS_DATA_NAME             : str = "data.npy"
#######################################

###### FOR TOY PROGRAM ESTIMATOR ######

PROGRAM_LEARNING_RATE           : float = 0.0003
PROGRAM_WEIGHT_DECAY            : float = 0.00004

PROGRAM_NUM_INPUTS              : int = 86
PROGRAM_NUM_FEATURES            : int = 128
PROGRAM_DEPTH                   : int = 2
PROGRAM_NORMALIZATION           : str = "layernorm"
PROGRAM_ACTIVATION              : str = "relu"

PROGRAM_MODEL_NAME              : str = "program_model.pth"
PROGRAM_DATA_NAME               : str = "toy-nettle-aes.npy"
#######################################

####### FOR SYNTHESIS ESTIMATOR #######

AREA_LEARNING_RATE         : float = 0.0003
AREA_WEIGHT_DECAY          : float = 0.00004

AREA_NUM_INPUTS            : int = 86
AREA_NUM_FEATURES          : int = 128
AREA_DEPTH                 : int = 2
AREA_NORMALIZATION         : str = "layernorm"
AREA_ACTIVATION            : str = "relu"

AREA_MODEL_NAME            : str = "area_model.pth"
AREA_DATA_NAME             : str = "data.npy"