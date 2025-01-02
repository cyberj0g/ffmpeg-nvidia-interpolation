/*
 * GPU-accelerated frame interpolation filter using Nvidia Optical Flow FRUC.
 * Requires Turing (GeForce RTX 20 series) GPU or later.
 * Based on Ffmpeg's FPS filter.
 */

#include <float.h>
#include <stdint.h>

#include "NvOFFRUC.h"
#include <dlfcn.h>

#include "libavutil/common.h"
#include "libavutil/avassert.h"
#include "libavutil/eval.h"
#include "libavutil/mathematics.h"
#include "libavutil/opt.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/cuda_check.h"
#include "cuda/load_helper.h"
#include "avfilter.h"
#include "ccfifo.h"
#include "filters.h"
#include "internal.h"
#include "video.h"

#define CHECK_CU(x) FF_CUDA_CHECK_DL(ctx, cu, x)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

enum EOFAction
{
    EOF_ACTION_ROUND,
    EOF_ACTION_PASS,
    EOF_ACTION_NB
};

static const char *const var_names[] = {
    "source_fps",
    "ntsc",
    "pal",
    "film",
    "ntsc_film",
    NULL};

enum var_name
{
    VAR_SOURCE_FPS,
    VAR_FPS_NTSC,
    VAR_FPS_PAL,
    VAR_FPS_FILM,
    VAR_FPS_NTSC_FILM,
    VARS_NB
};

static const double ntsc_fps = 30000.0 / 1001.0;
static const double pal_fps = 25.0;
static const double film_fps = 24.0;
static const double ntsc_film_fps = 24000.0 / 1001.0;

#define NVFRUC_SURFACE_COUNT 3

typedef struct NVIContext
{
    const AVClass *class;

    enum AVPixelFormat in_fmt;
    enum AVPixelFormat out_fmt;

    double start_time; ///< pts, in seconds, of the expected first frame

    char *framerate; ///< expression that defines the target framerate
    int rounding;    ///< AVRounding method for timestamps
    int eof_action;  ///< action performed for last frame in FIFO

    void *frucLibHandle;
    NvOFFRUCHandle fruc;

    AVCUDADeviceContext *hwctx;

    /* Set during outlink configuration */
    int64_t in_pts_off;  ///< input frame pts offset for start_time handling
    int64_t out_pts_off; ///< output frame pts offset for start_time handling

    /* Runtime state */
    int status;         ///< buffered input status
    int64_t status_pts; ///< buffered input status timestamp

    AVFrame *frames[2]; ///< buffered frames
    int frames_count;   ///< number of buffered frames

    CUdeviceptr nvi_surfaces[NVFRUC_SURFACE_COUNT]; /// < NVOFFRUC input and output surfaces, static for the filter lifetime
    int no_interpolate;                             /// < flag to skip interpolating current frame and output source frame
    int cur_input_idx;                              /// < we pass one frame at a time, registered input sufaces need to be rolled in a circular buffer fashion
    NvOFFRUC_PROCESS_IN_PARAMS nvi_in;              /// < interpolation input params
    int width;
    int height;

    CCFifo cc_fifo; ///< closed captions

    int64_t next_pts; ///< pts of the next frame to output

    /* statistics */
    int cur_frame_out; ///< number of times current frame has been output
    int frames_in;     ///< number of frames on input
    int frames_out;    ///< number of frames on output
    int dup;           ///< number of frames duplicated
    int drop;          ///< number of framed dropped
} NVIContext;

#define OFFSET(x) offsetof(NVIContext, x)
#define V AV_OPT_FLAG_VIDEO_PARAM
#define F AV_OPT_FLAG_FILTERING_PARAM
static const AVOption nvinterpolate_options[] = {
    {"fps", "A string describing desired output framerate", OFFSET(framerate), AV_OPT_TYPE_STRING, {.str = "25"}, 0, 0, V | F},
    {"start_time", "Assume the first PTS should be this value.", OFFSET(start_time), AV_OPT_TYPE_DOUBLE, {.dbl = DBL_MAX}, -DBL_MAX, DBL_MAX, V | F},
    {"round", "set rounding method for timestamps", OFFSET(rounding), AV_OPT_TYPE_INT, {.i64 = AV_ROUND_NEAR_INF}, 0, 5, V | F, "round"},
    {"zero", "round towards 0", 0, AV_OPT_TYPE_CONST, {.i64 = AV_ROUND_ZERO}, 0, 0, V | F, "round"},
    {"inf", "round away from 0", 0, AV_OPT_TYPE_CONST, {.i64 = AV_ROUND_INF}, 0, 0, V | F, "round"},
    {"down", "round towards -infty", 0, AV_OPT_TYPE_CONST, {.i64 = AV_ROUND_DOWN}, 0, 0, V | F, "round"},
    {"up", "round towards +infty", 0, AV_OPT_TYPE_CONST, {.i64 = AV_ROUND_UP}, 0, 0, V | F, "round"},
    {"near", "round to nearest", 0, AV_OPT_TYPE_CONST, {.i64 = AV_ROUND_NEAR_INF}, 0, 0, V | F, "round"},
    {"eof_action", "action performed for last frame", OFFSET(eof_action), AV_OPT_TYPE_INT, {.i64 = EOF_ACTION_ROUND}, 0, EOF_ACTION_NB - 1, V | F, "eof_action"},
    {"round", "round similar to other frames", 0, AV_OPT_TYPE_CONST, {.i64 = EOF_ACTION_ROUND}, 0, 0, V | F, "eof_action"},
    {"pass", "pass through last frame", 0, AV_OPT_TYPE_CONST, {.i64 = EOF_ACTION_PASS}, 0, 0, V | F, "eof_action"},
    {NULL}};

AVFILTER_DEFINE_CLASS(nvinterpolate);

PtrToFuncNvOFFRUCCreate NvOFFRUCCreate = NULL;
PtrToFuncNvOFFRUCRegisterResource NvOFFRUCRegisterResource = NULL;
PtrToFuncNvOFFRUCUnregisterResource NvOFFRUCUnregisterResource = NULL;
PtrToFuncNvOFFRUCProcess NvOFFRUCProcess = NULL;
PtrToFuncNvOFFRUCDestroy NvOFFRUCDestroy = NULL;

static av_cold int init_fruc(AVFilterContext *ctx, NVIContext *s)
{
    AVBufferRef *out_ref = NULL;
    NvOFFRUC_CREATE_PARAM createParams = {0};
    NvOFFRUCHandle hFRUC;
    NvOFFRUC_STATUS status = NvOFFRUC_SUCCESS;
    NvOFFRUC_REGISTER_RESOURCE_PARAM regOutParam = {0};
    int ret;
    CUcontext dummy;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    for (int i = 0; i < NVFRUC_SURFACE_COUNT; i++)
    {
        // allocate linear CUDA memory manually, NVOFFRUC seem to have a bug in handling pitched allocations
        if (s->nvi_surfaces[i])
            cu->cuMemFree(s->nvi_surfaces[i]);
        ret = CHECK_CU(cu->cuMemAlloc(&s->nvi_surfaces[i], s->width * s->height * 3 / 2));
        if (ret < 0)
            return AVERROR(ENOMEM);
    }

    createParams.uiHeight = s->height;
    createParams.uiWidth = s->width;
    createParams.eResourceType = CudaResource;
    createParams.eSurfaceFormat = NV12Surface;
    createParams.eCUDAResourceType = CudaResourceCuDevicePtr;

    // Initialize FRUC pipeline which internally initializes Optical flow engine
    status = NvOFFRUCCreate(
        &createParams,
        &hFRUC);

    if (status != NvOFFRUC_SUCCESS)
    {
        av_log(ctx, AV_LOG_ERROR, "NvOFFRUCCreate failed with status %d\n", status);
        goto fail;
    }

    s->fruc = hFRUC;

    memcpy(
        regOutParam.pArrResource,
        s->nvi_surfaces,
        NVFRUC_SURFACE_COUNT);
    regOutParam.uiCount = NVFRUC_SURFACE_COUNT;
    status = NvOFFRUCRegisterResource(
        hFRUC,
        &regOutParam);

    if (status != NvOFFRUC_SUCCESS)
    {
        av_log(ctx, AV_LOG_ERROR, "NvOFFRUCRegisterResource failed with status %d\n", status);
        goto fail;
    }
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    av_log(ctx, AV_LOG_VERBOSE, "NVFRUC initialized for %dx%d\n", s->width, s->height);
    return 0;
fail:
    av_log(ctx, AV_LOG_ERROR, "error initializing NVFRUC\n");
    av_buffer_unref(&out_ref);
    return ret;
}

static av_cold int init(AVFilterContext *ctx)
{
    const char *error_message;
    void *hDLL;
    NVIContext *s = ctx->priv;
    s->status_pts = AV_NOPTS_VALUE;
    s->next_pts = AV_NOPTS_VALUE;

    s->cur_input_idx = 1;
    for (int i = 0; i < 3; i++)
        s->nvi_surfaces[i] = 0;

    hDLL = dlopen("libNvOFFRUC.so", RTLD_LAZY);

    if (hDLL != NULL)
    {
        NvOFFRUCCreate = (PtrToFuncNvOFFRUCCreate)dlsym(hDLL, CreateProcName);
        NvOFFRUCRegisterResource = (PtrToFuncNvOFFRUCRegisterResource)dlsym(hDLL, RegisterResourceProcName);
        NvOFFRUCUnregisterResource = (PtrToFuncNvOFFRUCUnregisterResource)dlsym(hDLL, UnregisterResourceProcName);
        NvOFFRUCProcess = (PtrToFuncNvOFFRUCProcess)dlsym(hDLL, ProcessProcName);
        NvOFFRUCDestroy = (PtrToFuncNvOFFRUCDestroy)dlsym(hDLL, DestroyProcName);

        if (
            !NvOFFRUCCreate || !NvOFFRUCRegisterResource || !NvOFFRUCUnregisterResource || !NvOFFRUCProcess || !NvOFFRUCDestroy)
        {
            av_log(ctx, AV_LOG_ERROR, "error looking up libNvOFFFruc symbols\n");
            dlclose(hDLL);
            return AVERROR(ENOMEM);
        }
    }
    else
    {
        av_log(ctx, AV_LOG_ERROR, "error loading libNvOFFFruc library: ");
        error_message = dlerror();
        if (error_message)
        {
            av_log(ctx, AV_LOG_ERROR, "%s\n", error_message);
        }
        else
        {
            av_log(ctx, AV_LOG_ERROR, "unknown error\n");
        }
        return AVERROR(ENOMEM);
    }

    s->frucLibHandle = hDLL;

    av_log(ctx, AV_LOG_VERBOSE, "init\n");
    return 0;
}

/* Remove the first frame from the buffer, returning it */
static AVFrame *shift_frame(AVFilterContext *ctx, NVIContext *s)
{
    AVFrame *frame;

    /* Must only be called when there are frames in the buffer */
    av_assert1(s->frames_count > 0);

    frame = s->frames[0];
    s->frames[0] = s->frames[1];
    s->frames[1] = NULL;
    s->frames_count--;

    /* Update statistics counters */
    s->frames_out += s->cur_frame_out;
    if (s->cur_frame_out > 1)
    {
        av_log(ctx, AV_LOG_DEBUG, "Duplicated frame with pts %" PRId64 " %d times\n",
               frame->pts, s->cur_frame_out - 1);
        s->dup += s->cur_frame_out - 1;
    }
    else if (s->cur_frame_out == 0)
    {
        av_log(ctx, AV_LOG_DEBUG, "Dropping frame with pts %" PRId64 "\n",
               frame->pts);
        s->drop++;
    }
    s->cur_frame_out = 0;

    return frame;
}

static av_cold int destroy_fruc(AVFilterContext *ctx)
{
    NvOFFRUC_STATUS status = NvOFFRUC_SUCCESS;
    NVIContext *s = ctx->priv;
    NvOFFRUC_UNREGISTER_RESOURCE_PARAM stUnregisterResourceParam = {0};
    memcpy(
        stUnregisterResourceParam.pArrResource,
        s->nvi_surfaces,
        NVFRUC_SURFACE_COUNT);
    stUnregisterResourceParam.uiCount = NVFRUC_SURFACE_COUNT;

    status = NvOFFRUCUnregisterResource(
        s->fruc,
        &stUnregisterResourceParam);
    if (status != NvOFFRUC_SUCCESS)
    {
        av_log(ctx, AV_LOG_ERROR, "NvOFFRUCUnregisterResource failed with error code: %d\n", status);
        return -1;
    }

    // Destroy FRUC instance
    status = NvOFFRUCDestroy(s->fruc);
    if (status != NvOFFRUC_SUCCESS)
    {
        av_log(ctx, AV_LOG_ERROR, "NvOFFRUCDestroy failed with error code: %d\n", status);
        return -1;
    }
    av_log(ctx, AV_LOG_VERBOSE, "NVFRUC destroyed: %d\n", status);
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    NVIContext *s = ctx->priv;

    ff_ccfifo_uninit(&s->cc_fifo);

    if (s->fruc)
        destroy_fruc(ctx);

    if (s->frucLibHandle)
        dlclose(s->frucLibHandle);

    if (s->hwctx)
        for (int i = 0; i < NVFRUC_SURFACE_COUNT; i++)
        {
            if (s->nvi_surfaces[i])
                s->hwctx->internal->cuda_dl->cuMemFree(s->nvi_surfaces[i]);
        }

    av_log(ctx, AV_LOG_VERBOSE, "uninitialized, %d frames in, %d frames out; %d frames dropped, "
                                "%d frames duplicated.\n",
           s->frames_in, s->frames_out, s->drop, s->dup);
}

static int NV12FrameToCudaLinear(AVFilterContext *ctx, CudaFunctions *cu, CUdeviceptr dst, int width, int height, AVFrame *frame)
{
    int ret;
    CUDA_MEMCPY2D y = {
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcDevice = (CUdeviceptr)frame->data[0],
        .srcPitch = frame->linesize[0],
        .dstDevice = dst,
        .dstPitch = width,
        .WidthInBytes = width,
        .Height = height,
        .srcY = 0};
    ret = CHECK_CU(cu->cuMemcpy2D(&y));
    if (ret < 0)
        return AVERROR(ENOMEM);

    CUDA_MEMCPY2D uv = {
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcDevice = (CUdeviceptr)frame->data[1],
        .srcPitch = frame->linesize[1],
        .dstDevice = dst + width * height,
        .dstPitch = width,
        .WidthInBytes = width,
        .Height = height / 2,
        .srcY = 0};
    ret = CHECK_CU(cu->cuMemcpy2D(&uv));
    if (ret < 0)
        return AVERROR(ENOMEM);

    return 0;
}

static int CudaLinearToNV12Frame(AVFilterContext *ctx, CudaFunctions *cu, AVFrame *dstFrame, CUdeviceptr src, int width, int height)
{
    int ret;
    CUDA_MEMCPY2D y = {
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcDevice = src,
        .srcPitch = width,
        .dstDevice = (CUdeviceptr)dstFrame->data[0],
        .dstPitch = dstFrame->linesize[0],
        .WidthInBytes = width,
        .Height = height,
        .srcY = 0};
    ret = CHECK_CU(cu->cuMemcpy2D(&y));
    if (ret < 0)
        return AVERROR(ENOMEM);

    CUDA_MEMCPY2D uv = {
        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE,
        .srcDevice = src + width * height,
        .srcPitch = width,
        .dstDevice = (CUdeviceptr)dstFrame->data[1],
        .dstPitch = dstFrame->linesize[1],
        .WidthInBytes = width,
        .Height = height / 2,
        .srcY = 0};
    ret = CHECK_CU(cu->cuMemcpy2D(&uv));
    if (ret < 0)
        return AVERROR(ENOMEM);

    return 0;
}

static int config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = ctx->inputs[0];
    NVIContext *s = ctx->priv;
    AVHWFramesContext *in_frames_ctx;
    AVCUDADeviceContext *device_hwctx;

    double var_values[VARS_NB], res;
    int ret;

    if (inlink->format != AV_PIX_FMT_CUDA)
    {
        av_log(ctx, AV_LOG_ERROR, "Only CUDA format is supported.\n");
        return AVERROR(EINVAL);
    }

    var_values[VAR_SOURCE_FPS] = av_q2d(inlink->frame_rate);
    var_values[VAR_FPS_NTSC] = ntsc_fps;
    var_values[VAR_FPS_PAL] = pal_fps;
    var_values[VAR_FPS_FILM] = film_fps;
    var_values[VAR_FPS_NTSC_FILM] = ntsc_film_fps;
    ret = av_expr_parse_and_eval(&res, s->framerate,
                                 var_names, var_values,
                                 NULL, NULL, NULL, NULL, NULL, 0, ctx);
    if (ret < 0)
        return ret;

    outlink->frame_rate = av_d2q(res, INT_MAX);
    outlink->time_base = av_inv_q(outlink->frame_rate);

    /* Calculate the input and output pts offsets for start_time */
    if (s->start_time != DBL_MAX && s->start_time != AV_NOPTS_VALUE)
    {
        double first_pts = s->start_time * AV_TIME_BASE;
        if (first_pts < INT64_MIN || first_pts > INT64_MAX)
        {
            av_log(ctx, AV_LOG_ERROR, "Start time %f cannot be represented in internal time base\n",
                   s->start_time);
            return AVERROR(EINVAL);
        }
        s->in_pts_off = av_rescale_q_rnd(first_pts, AV_TIME_BASE_Q, inlink->time_base,
                                         s->rounding | AV_ROUND_PASS_MINMAX);
        s->out_pts_off = av_rescale_q_rnd(first_pts, AV_TIME_BASE_Q, outlink->time_base,
                                          s->rounding | AV_ROUND_PASS_MINMAX);
        s->next_pts = s->out_pts_off;
        av_log(ctx, AV_LOG_VERBOSE, "Set first pts to (in:%" PRId64 " out:%" PRId64 ") from start time %f\n",
               s->in_pts_off, s->out_pts_off, s->start_time);
    }

    ret = ff_ccfifo_init(&s->cc_fifo, outlink->frame_rate, ctx);
    if (ret < 0)
    {
        av_log(ctx, AV_LOG_ERROR, "Failure to setup CC FIFO queue\n");
        return ret;
    }

    // CUDA context should be already initialized here
    if (!ctx->inputs[0]->hw_frames_ctx)
    {
        av_log(ctx, AV_LOG_ERROR, "no HW context available, make sure GPU acceleration is properly configured\n");
        return -1;
    }

    in_frames_ctx = (AVHWFramesContext *)ctx->inputs[0]->hw_frames_ctx->data;
    device_hwctx = in_frames_ctx->device_ctx->hwctx;
    s->in_fmt = in_frames_ctx->sw_format;
    s->out_fmt = s->in_fmt;
    s->hwctx = device_hwctx;
    s->width = ctx->inputs[0]->w;
    s->height = ctx->inputs[0]->h;

    ret = init_fruc(ctx, s);

    av_log(ctx, AV_LOG_VERBOSE, "nvinterpolate=%d/%d\n", outlink->frame_rate.num, outlink->frame_rate.den);

    if (ret < 0)
        return ret;
    return 0;
}

/* Read a frame from the input and save it in the buffer */
static int read_frame(AVFilterContext *ctx, NVIContext *s, AVFilterLink *inlink, AVFilterLink *outlink)
{
    AVFrame *frame;
    int ret;
    int64_t in_pts;

    CUcontext dummy;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;

    /* Must only be called when we have buffer room available */
    av_assert1(s->frames_count < 2);

    ret = ff_inlink_consume_frame(inlink, &frame);
    /* Caller must have run ff_inlink_check_available_frame first */
    av_assert1(ret);
    if (ret < 0)
        return ret;

    /* Convert frame pts to output timebase.
     * The dance with offsets is required to match the rounding behaviour of the
     * previous version of the fps filter when using the start_time option. */
    in_pts = frame->pts;
    frame->pts = s->out_pts_off + av_rescale_q_rnd(in_pts - s->in_pts_off,
                                                   inlink->time_base, outlink->time_base,
                                                   s->rounding | AV_ROUND_PASS_MINMAX);

    av_log(ctx, AV_LOG_DEBUG, "Read frame with in pts %" PRId64 ", out pts %" PRId64 "\n",
           in_pts, frame->pts);

    ff_ccfifo_extract(&s->cc_fifo, frame);
    s->frames[s->frames_count++] = frame;
    s->frames_in++;

    av_log(ctx, AV_LOG_DEBUG, "Prepare NVI input params for pts %ld in surface idx %d\n",
           frame->pts, s->cur_input_idx);

    // copy frame to input surface
    CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));

    ret = NV12FrameToCudaLinear(ctx, cu, s->nvi_surfaces[s->cur_input_idx], s->width, s->height, frame);

    if (ret < 0)
        return AVERROR(ENOMEM);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    // prepare interpolation input params
    s->nvi_in.stFrameDataInput.pFrame = &s->nvi_surfaces[s->cur_input_idx];
    s->nvi_in.stFrameDataInput.nTimeStamp = frame->pts;
    s->nvi_in.uSyncWait.MutexAcquireKey.uiKeyForRenderTextureAcquire = 0;
    s->nvi_in.uSyncWait.MutexAcquireKey.uiKeyForInterpTextureAcquire = 0;
    s->nvi_in.bSkipWarp = 0;

    // switch next input surface index
    s->cur_input_idx = 1 + (s->cur_input_idx % 2);

    return 1;
}

/* Write a frame to the output */
static int write_frame(AVFilterContext *ctx, NVIContext *s, AVFilterLink *outlink, int *again)
{
    AVFrame *frame;
    NvOFFRUC_PROCESS_OUT_PARAMS stOutParams;
    NvOFFRUC_STATUS status;
    CUcontext dummy;
    int ret;
    bool bHasFrameRepetitionOccured;
    CudaFunctions *cu;
    status = NvOFFRUC_SUCCESS;

    av_assert1(s->frames_count == 2 || (s->status && s->frames_count == 1));

    /* We haven't yet determined the pts of the first frame */
    if (s->next_pts == AV_NOPTS_VALUE)
    {
        if (s->frames[0]->pts != AV_NOPTS_VALUE)
        {
            s->next_pts = s->frames[0]->pts;
            av_log(ctx, AV_LOG_VERBOSE, "Set first pts to %" PRId64 "\n", s->next_pts);
        }
        else
        {
            av_log(ctx, AV_LOG_WARNING, "Discarding initial frame(s) with no "
                                        "timestamp.\n");
            frame = shift_frame(ctx, s);
            av_frame_free(&frame);
            *again = 1;
            return 0;
        }
    }

    /* There are two conditions where we want to drop a frame:
     * - If we have two buffered frames and the second frame is acceptable
     *   as the next output frame, then drop the first buffered frame.
     * - If we have status (EOF) set, drop frames when we hit the
     *   status timestamp. */
    if ((s->frames_count == 2 && s->frames[1]->pts <= s->next_pts) ||
        (s->status && s->status_pts <= s->next_pts))
    {

        // NVI: set flag to not interpolate on next step and just output
        // last frame
        s->no_interpolate = 1;
        av_log(ctx, AV_LOG_DEBUG, "Set no interpolate\n");

        frame = shift_frame(ctx, s);
        av_frame_free(&frame);
        *again = 1;
        return 0;

        /* Output a copy of the first buffered frame */
    }
    else
    {
        // NVI: call the interpolation here with proper timestamps
        // create output HW frame
        if (s->no_interpolate)
        {
            s->no_interpolate = 0;
            frame = av_frame_clone(s->frames[0]);
            av_log(ctx, AV_LOG_DEBUG, "Output source frame at pts %ld \n", s->frames[0]->pts);
        }
        else
        {
            stOutParams.stFrameDataOutput.pFrame = &s->nvi_surfaces[0];
            stOutParams.stFrameDataOutput.nTimeStamp = s->next_pts;
            stOutParams.stFrameDataOutput.bHasFrameRepetitionOccurred = &bHasFrameRepetitionOccured;
            stOutParams.uSyncSignal.MutexReleaseKey.uiKeyForRenderTextureRelease = 0;
            stOutParams.uSyncSignal.MutexReleaseKey.uiKeyForInterpolateRelease = 0;

            cu = s->hwctx->internal->cuda_dl;
            ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
            if (ret < 0)
                return AVERROR(ENOMEM);

            status = NvOFFRUCProcess(
                s->fruc,
                &s->nvi_in,
                &stOutParams);
            if (status != NvOFFRUC_SUCCESS)
            {
                av_log(ctx, AV_LOG_ERROR, "NvOFFRUCCreate failed with status %d\n", status);
                return AVERROR(ENOMEM);
            }
            // create output frame
            frame = av_frame_alloc();
            ret = av_hwframe_get_buffer(s->frames[0]->hw_frames_ctx, frame, 0);
            if (ret < 0)
                return AVERROR(ENOMEM);

            ret = CudaLinearToNV12Frame(ctx, cu, frame, s->nvi_surfaces[0], s->width, s->height);

            if (ret < 0)
                return AVERROR(ENOMEM);

            av_log(ctx, AV_LOG_DEBUG, "NV interpolate at pts %ld repetition %d\n", s->next_pts, bHasFrameRepetitionOccured);

            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        }

        if (!frame)
            return AVERROR(ENOMEM);
        // Make sure Closed Captions will not be duplicated
        ff_ccfifo_inject(&s->cc_fifo, frame);
        frame->pts = s->next_pts++;
        frame->duration = 1;

        av_log(ctx, AV_LOG_DEBUG, "Writing frame with pts %" PRId64 " to pts %" PRId64 "\n",
               s->frames[0]->pts, frame->pts);
        s->cur_frame_out++;
        *again = 1;
        return ff_filter_frame(outlink, frame);
    }
}

/* Convert status_pts to outlink timebase */
static void update_eof_pts(AVFilterContext *ctx, NVIContext *s, AVFilterLink *inlink, AVFilterLink *outlink, int64_t status_pts)
{
    int eof_rounding = (s->eof_action == EOF_ACTION_PASS) ? AV_ROUND_UP : s->rounding;
    s->status_pts = av_rescale_q_rnd(status_pts, inlink->time_base, outlink->time_base,
                                     eof_rounding | AV_ROUND_PASS_MINMAX);

    av_log(ctx, AV_LOG_DEBUG, "EOF is at pts %" PRId64 "\n", s->status_pts);
}

static int activate(AVFilterContext *ctx)
{
    NVIContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];

    int ret;
    int again = 0;
    int64_t status_pts;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    /* No buffered status: normal operation */
    if (!s->status)
    {

        /* Read available input frames if we have room */
        while (s->frames_count < 2 && ff_inlink_check_available_frame(inlink))
        {
            ret = read_frame(ctx, s, inlink, outlink);
            if (ret < 0)
                return ret;
        }

        /* We do not yet have enough frames to produce output */
        if (s->frames_count < 2)
        {
            /* Check if we've hit EOF (or otherwise that an error status is set) */
            ret = ff_inlink_acknowledge_status(inlink, &s->status, &status_pts);
            if (ret > 0)
                update_eof_pts(ctx, s, inlink, outlink, status_pts);

            if (!ret)
            {
                /* If someone wants us to output, we'd better ask for more input */
                FF_FILTER_FORWARD_WANTED(outlink, inlink);
                return 0;
            }
        }
    }

    /* Buffered frames are available, so generate an output frame */
    if (s->frames_count > 0)
    {
        ret = write_frame(ctx, s, outlink, &again);
        /* Couldn't generate a frame, so schedule us to perform another step */
        if (again && ff_inoutlink_check_flow(inlink, outlink))
            ff_filter_set_ready(ctx, 100);
        return ret;
    }

    /* No frames left, so forward the status */
    if (s->status && s->frames_count == 0)
    {
        ff_outlink_set_status(outlink, s->status, s->next_pts);
        return 0;
    }

    return FFERROR_NOT_READY;
}

static const AVFilterPad avfilter_vf_fps_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
    },
};

const AVFilter ff_vf_nvinterpolate = {
    .name = "nvinterpolate",
    .description = NULL_IF_CONFIG_SMALL("Interpolate frames up to desired FPS with Nvidia FRUC"),
    .init = init,
    .uninit = uninit,
    .priv_size = sizeof(NVIContext),
    .priv_class = &nvinterpolate_class,
    .activate = activate,
    .flags = AVFILTER_FLAG_METADATA_ONLY,
    FILTER_INPUTS(ff_video_default_filterpad),
    FILTER_OUTPUTS(avfilter_vf_fps_outputs),
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};

#pragma GCC diagnostic pop
