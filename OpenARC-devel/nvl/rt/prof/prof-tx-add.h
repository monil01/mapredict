// In the future, this data should be replaced based on NVM device. See
// related todo in nvlrt_tx_tryShadowUpdate in ../nvlrt-pmemobj.c. See
// ../../README for instructions on generating this data.

#define SHADOW_UPDATE_COST_MODE_DEFAULT NVLRT_COST_INFINITE

static const size_t shadowUpdateMsyncCutoff = 0;

static const double shadowUpdateMsyncUndoA = 0;
static const double shadowUpdateMsyncUndoB = 0;
static const double shadowUpdateMsyncUndoC = 0;

static const double shadowUpdateMsyncSbwfA = 0;
static const double shadowUpdateMsyncSbwfB = 0;
static const double shadowUpdateMsyncSbwfC = 0;

static const double shadowUpdateMsyncWriteA = 0;
static const double shadowUpdateMsyncWriteB = 0;
static const double shadowUpdateMsyncWriteC = 0;

static const size_t shadowUpdateClflushCutoff = 0;

static const double shadowUpdateClflushUndoA = 0;
static const double shadowUpdateClflushUndoB = 0;
static const double shadowUpdateClflushUndoC = 0;

static const double shadowUpdateClflushSbwfA = 0;
static const double shadowUpdateClflushSbwfB = 0;
static const double shadowUpdateClflushSbwfC = 0;

static const double shadowUpdateClflushWriteA = 0;
static const double shadowUpdateClflushWriteB = 0;
static const double shadowUpdateClflushWriteC = 0;
