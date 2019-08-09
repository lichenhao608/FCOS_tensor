import os
from yacs.config import CfgNode as CN

cfg = CN()
cfg.MODEL = CN()

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
cfg.INPUT = CN()
cfg.INPUT.MIN_SIZE_TRAIN = (800,)
# -1 means disabled and it will use MIN_SIZE_TRAIN
cfg.INPUT.MIN_SIZE_RANGE_TRAIN = (-1, -1)
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.MIN_SZIE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1333
# value used for image normalization
cfg.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
cfg.INPUT.PIXEL_STD = [1., 1., 1.]

# ---------------------------------------------------------------------------- #
# Data Set
# ---------------------------------------------------------------------------- #
cfg.DATASETS = CN()
cfg.DATASETS.TRAIN = ()
cfg.DATASETS.TEST = ()

# ---------------------------------------------------------------------------- #
# Dataloader
# ---------------------------------------------------------------------------- #
cfg.DATALOADER = CN()
cfg.DATALOADER.NUM_WORKERS = 1
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
cfg.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
cfg.DATALOADER.ASPECT_RATIO_GROUPING = True

# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
cfg.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
cfg.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
cfg.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
cfg.MODEL.GROUP_NORM.EPSILON = 1e-5

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
cfg.MODEL.RPN = CN()
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
cfg.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
cfg.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
cfg.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
cfg.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
cfg.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
cfg.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
cfg.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
cfg.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
cfg.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
cfg.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
cfg.MODEL.ROI_HEADS = CN()
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
cfg.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
cfg.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


cfg.MODEL.ROI_BOX_HEAD = CN()
cfg.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
cfg.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# GN
cfg.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
cfg.MODEL.ROI_BOX_HEAD.DILATION = 1
cfg.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
cfg.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4


cfg.MODEL.ROI_MASK_HEAD = CN()
cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
cfg.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
cfg.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
cfg.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
cfg.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
cfg.MODEL.ROI_MASK_HEAD.USE_GN = False

cfg.MODEL.ROI_KEYPOINT_HEAD = CN()
cfg.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
cfg.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
cfg.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
cfg.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
cfg.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
cfg.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
cfg.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
cfg.MODEL.RESNETS.RES5_DILATION = 1

cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

cfg.MODEL.RESNETS.FIXED_BLOCKS = 1

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
cfg.MODEL.FCOS = CN()
cfg.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
cfg.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
cfg.MODEL.FCOS.PRIOR_PROB = 0.01
cfg.MODEL.FCOS.INFERENCE_TH = 0.05
cfg.MODEL.FCOS.NMS_TH = 0.6
cfg.MODEL.FCOS.PRE_NMS_TOP_N = 1000

# Focal loss parameter: alpha
cfg.MODEL.FCOS.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
cfg.MODEL.FCOS.LOSS_GAMMA = 2.0

# the number of convolutions used in the cls and bbox tower
cfg.MODEL.FCOS.NUM_CONVS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
cfg.SOLVER = CN()
cfg.SOLVER.MAX_ITER = 40000

cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.BIAS_LR_FACTOR = 2

cfg.SOLVER.MOMENTUM = 0.9

cfg.SOLVER.WEIGHT_DECAY = 0.0005
cfg.SOLVER.WEIGHT_DECAY_BIAS = 0

cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = (30000,)

cfg.SOLVER.WARMUP_FACTOR = 1.0 / 3
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.WARMUP_METHOD = "linear"

cfg.SOLVER.CHECKPOINT_PERIOD = 2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
cfg.SOLVER.IMS_PER_BATCH = 16

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
cfg.TEST = CN()
cfg.TEST.EXPECTED_RESULTS = []
cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
cfg.TEST.IMS_PER_BATCH = 8
# Number of detections per image
cfg.TEST.DETECTIONS_PER_IMG = 100


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
cfg.OUTPUT_DIR = "."

cfg.PATHS_CATALOG = os.path.join(
    os.path.dirname(__file__), "paths_catalog.py")
