/**
 * GStreamer / NNStreamer tensor_decoder subplugin, "Pose estimation"
 * Copyright (C) 2019 Samsung Electronics Co. Ltd.
 * Copyright (C) 2019 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation;
 * version 2.1 of the License.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file        tensordec-blazepose.c
 * @date        09 November 2022
 * @brief       NNStreamer tensor-decoder subplugin, "blazepose estimation",
 *              which converts tensors to video stream w/ pose on
 *              transparent background.
 *              This code is NYI/WIP and not compilable.
 *
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Torsten Schulz <torsten.schulz@gmail.com>
 * @bug         No known bugs except for NYI items
 *
 * option1: Video Output Dimension (WIDTH:HEIGHT)
 * option2: Input Dimension (WIDTH:HEIGHT)
 * option3: Location of label file (optional)
 * 	The file describes the keypoints and their body connections.
 * 	A line per keypoint description is expected with the following syntax:
 * 	<label name> <keypoint id> <keypoint id>
 *
 * 	For instance, the posenet model label description of model
 * 	https://www.tensorflow.org/lite/examples/pose_estimation/overview
 * 	would be the following:
 * 	nose 1 2 3 4
 * 	leftEye 0 2 3
 * 	rightEye 0 1 4
 * 	leftEar 0 1
 * 	rightEar 0 2
 * 	leftShoulder 6 7 11
 * 	rightShoulder 5 8 12
 * 	leftElbow 5 9
 * 	rightElbow 6 10
 * 	leftWrist 7
 * 	rightWrist 8
 * 	leftHip 5 12 13
 * 	rightHip 6 11 14
 * 	leftKnee 11 15
 * 	rightKnee 12 16
 * 	leftAnkle 13
 * 	rightAnkle 14
 *
 * option4: Mode (optional)
 *      Available: heatmap-only (default)
 *                 heatmap-offset
 *
 * 	Expected input dims:
 * 		Note: Width, Height are related to heatmap resolution.
 * 		- heatmap-only:
 *   			Tensors mapping: Heatmap
 *   			Tensor[0]: #labels x width x height (float32, label probability)
 *                    		(e.g., 14 x 33 x 33 )
 * 		- heatmap-offset:
 * 			Compatible with posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
 *   			Tensors mapping: Heatmap, Offset
 *   			Tensor[0]: #labels : width : height (float32, label sigmoid probability)
 *	                    	(e.g., 17 x 9 x 9 )
 *   			Tensor[1]: #labels x 2: width : height (float32, Offset position within heatmap grid)
 *	                    	(e.g., 34 x 9 x 9 )
 *
 * Pipeline:
 * 	v4l2src
 * 	   |
 * 	videoconvert
 * 	   |
 * 	videoscale -- tee ------------------------------------------------- compositor -- videoconvert -- ximagesink 
 * 	                |                                                       |
 *		   videoscale							| 
 * 	                |                                                       |
 *		   tensor_converter -- tensor_transform -- tensor_filter -- tensor_decoder
 *
 * 	- Used model is posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite
 * 	- Resize image into 257:257 at the second videoscale.
 * 	- Transform RGB value into float32 in range [0,1] at tensor_transform.
 *
 * 	gst-launch-1.0 v4l2src ! videoconvert ! videoscale ! \
 * 	   video/x-raw,format=RGB,width=640,height=480,framerate=30/1 ! \tee name=t \
 * 	   t. ! queue ! videoscale ! video/x-raw,width=257,height=257,format=RGB ! \
 * 	   tensor_converter ! tensor_transform mode=arithmetic option=typecast:float32,add:-127.5,div:127.5 ! \
 * 	   tensor_filter framework=tensorflow-lite model=posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite ! \
 * 	   tensor_decoder mode=pose_estimation option1=640:480 option2=257:257 option3=pose_label.txt option4=heatmap-offset ! \
 * 	   compositor name=mix sink_0::zorder=1 sink_1::zorder=0 ! videoconvert ! ximagesink \
 * 	   t. ! queue ! mix.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <glib.h>
#include <gst/gst.h>
#include <nnstreamer_plugin_api_decoder.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_log.h>
#include <nnstreamer_util.h>
#include "tensordecutil.h"

void init_blazepose (void) __attribute__ ((constructor));
void finish_blazepose (void) __attribute__ ((destructor));

/* font.c */
extern uint8_t rasters[][13];

#define PIXEL_VALUE               (0xFFFFFFFF)

#define POSE_MD_MAX_LABEL_SZ 33
#define POSE_MD_MAX_CONNECTIONS_SZ 16

/**
 * @brief Macro for calculating sigmoid
 */
#define _sigmoid(x) \
    (1.f / (1.f + expf (-x)))

/**
 * @brief Data structure for key body point description.
 */
static struct pose_metadata_s
{
  gchar label[POSE_MD_MAX_LABEL_SZ]; /**< Key body name */
  gint connections[POSE_MD_MAX_CONNECTIONS_SZ];/**< Connections list */
  gint num_connections; /** Total number of connections */
} pose_metadata_default[] = {
  {
    "top", {
  1}, 1}, {
    "neck", {
  0, 2, 5, 8, 11}, 5}, {
    "r_shoulder", {
  1, 3}, 2}, {
    "r_elbow", {
  2, 4}, 2}, {
    "r_wrist", {
  3}, 1}, {
    "l_shoulder", {
  1, 6}, 2}, {
    "l_elbow", {
  5, 7}, 2}, {
    "l_wrist", {
  6}, 1}, {
    "r_hip", {
  1, 9}, 2}, {
    "r_knee", {
  8, 10}, 2}, {
    "r_ankle", {
  9}, 1}, {
    "l_hip", {
  1, 12}, 2}, {
    "l_knee", {
  11, 13}, 2}, {
    "l_ankle", {
  12}, 1}
};

typedef struct pose_metadata_s pose_metadata_t;

#define POSE_SIZE_DEFAULT   (sizeof(pose_metadata_default) / sizeof(pose_metadata_t))

/**
 * @todo Fill in the value at build time or hardcode this. It's const value
 * @brief The bitmap of characters
 * [Character (ASCII)][Height][Width]
 */
static singleLineSprite_t singleLineSprite;

/**
 * @brief Data structure for boundig box info.
 */
typedef struct
{
  /* From option1 */
  guint width; /**< Output Video Width */
  guint height; /**< Output Video Height */

  /* From option2 */
  guint i_width; /**< Input Video Width */
  guint i_height; /**< Input Video Height */

  /* From option3 */
  pose_metadata_t *metadata; /**< Pose metadata from file, if any*/
  guint total_labels; /**< Total number of key body point */
} pose_data;

/**
 * @brief Load key body metadata from file
 *
 * The file describes the different key body point reported by the model,
 * with one line dedicated per key body point.
 *
 * The first word is the key body string, followed by its connections with other key body point.
 * Connections are represented through key body integer id
 * Token separator is space, .i.e. ' '
 *
 * File example of fallback configuration:
 *
 * top 1
 * neck 0 2 5 8 11
 * r_shoulder 1 3
 * r_elbow 2 4
 * r_wrist 3
 * l_shoulder 1 6
 * l_elbow 5 7
 * l_wrist 6 1
 * r_hip 1 9
 * r_knee 8 10
 * r_ankle 9
 * l_hip 1 12
 * l_knee 11 13
 * l_ankle 12
 *
 * @param[in] file_path The filename path to load
 * @param[in] pd The pose data object
 * @return Return TRUE on file loading success, otherwise FALSE
 */
static gboolean
pose_load_metadata_from_file (pose_data * pd, const gchar * file_path)
{
  size_t len;
  GError *err = NULL;
  gchar *contents = NULL;
  gchar **lines;
  guint i;

  if (!g_file_test (file_path, G_FILE_TEST_EXISTS)) {
    GST_WARNING ("Labels file %s does not exist !", file_path);
    return FALSE;
  }

  if (!g_file_get_contents (file_path, &contents, &len, &err)) {
    ml_loge ("Unable to read file %s with error %s.", file_path, err->message);
    g_clear_error (&err);
    return FALSE;
  }

  if (contents[len - 1] == '\n')
    contents[len - 1] = '\0';

  lines = g_strsplit (contents, "\n", -1);
  pd->total_labels = g_strv_length (lines);
  pd->metadata = g_new0 (pose_metadata_t, pd->total_labels);

  for (i = 0; i < pd->total_labels; i++) {
    guint j;
    guint len;
    gchar **tokens;

    g_strstrip (lines[i]);
    tokens = g_strsplit (lines[i], " ", -1);
    len = g_strv_length (tokens);
    if (len > POSE_MD_MAX_CONNECTIONS_SZ) {
      GST_WARNING ("Too many connections (%d) declared, clamping (%d)\n",
          len, POSE_MD_MAX_CONNECTIONS_SZ);
      len = POSE_MD_MAX_CONNECTIONS_SZ;
    }
    g_strlcpy (pd->metadata[i].label, tokens[0], POSE_MD_MAX_LABEL_SZ);
    pd->metadata[i].num_connections = len - 1;
    for (j = 1; j < len; j++)
      pd->metadata[i].connections[j - 1] =
          (gint) g_ascii_strtoll (tokens[j], NULL, 10);

    g_strfreev (tokens);
  }

  g_strfreev (lines);
  g_free (contents);

  return TRUE;
}

/** @brief Return pose metadata by id */
static inline pose_metadata_t *
pose_get_metadata_by_id (pose_data * data, guint id)
{
  pose_metadata_t *md = data->metadata;

  if (id > data->total_labels)
    return NULL;

  return &md[id];
}

/** @brief tensordec-plugin's TensorDecDef callback */
static int
blazepose_init (void **pdata)
{
  pose_data *data;

  data = *pdata = g_new0 (pose_data, 1);
  if (data == NULL) {
    GST_ERROR ("Failed to allocate memory for decoder subplugin.");
    return FALSE;
  }

  data->width = 0;
  data->height = 0;
  data->i_width = 0;
  data->i_height = 0;

  data->metadata = pose_metadata_default;
  data->total_labels = POSE_SIZE_DEFAULT;

  initSingleLineSprite (singleLineSprite, rasters, PIXEL_VALUE);

  return TRUE;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static void
blazepose_exit (void **pdata)
{
  pose_data *data = *pdata;

  if (data->metadata != pose_metadata_default)
    g_free (data->metadata);

  g_free (*pdata);
  *pdata = NULL;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static int
blazepose_setOption (void **pdata, int opNum, const char *param)
{
  pose_data *data = *pdata;

  if (opNum == 0) {
    /* option1 = output video size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    data->width = 0;
    data->height = 0;
    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR
          ("mode-option-1 of pose estimation is video output dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE;              /* Ignore this param */
    }
    if (rank > 2) {
      GST_WARNING
          ("mode-option-1 of pose estimation is video output dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    data->width = dim[0];
    data->height = dim[1];
    return TRUE;
  } else if (opNum == 1) {
    /* option1 = input model size (width:height) */
    tensor_dim dim;
    int rank = gst_tensor_parse_dimension (param, dim);

    data->i_width = 0;
    data->i_height = 0;
    if (param == NULL || *param == '\0')
      return TRUE;

    if (rank < 2) {
      GST_ERROR
          ("mode-option-2 of pose estimation is input video dimension (WIDTH:HEIGHT). The given parameter, \"%s\", is not acceptable.",
          param);
      return TRUE;
    }
    if (rank > 2) {
      GST_WARNING
          ("mode-option-2 of pose esitmiation is input video dimension (WIDTH:HEIGHT). The third and later elements of the given parameter, \"%s\", are ignored.",
          param);
    }
    data->i_width = dim[0];
    data->i_height = dim[1];
    return TRUE;
  } else if (opNum == 2) {
    return pose_load_metadata_from_file (data, (const gchar *) param);
  }

  GST_INFO ("Property mode-option-%d is ignored", opNum + 1);
  return TRUE;
}

/**
 * @brief check the num_tensors is valid
*/
static int
_check_tensors (const GstTensorsConfig * config)
{
  unsigned int i;
  g_return_val_if_fail (config != NULL, FALSE);

  for (i = 1; i < config->info.num_tensors; ++i) {
    g_return_val_if_fail (config->info.info[i - 1].type ==
        config->info.info[i].type, FALSE);
  }
  return TRUE;
}

/**
 * @brief tensordec-plugin's TensorDecDef callback
 *
 * [Pose Estimation]
 * Just one tensor with [ 14 (#Joint), WIDTH, HEIGHT, 1]
 * One WIDTH:HEIGHT for the each joint.
 * Have to find max value after Gaussian Blur
 *
 */
static GstCaps *
blazepose_getOutCaps (void **pdata, const GstTensorsConfig * config)
{
  pose_data *data = *pdata;
  GstCaps *caps;
  int i;
  char *str;

  const uint32_t *dim;

  if (!_check_tensors (config))
    return NULL;

  /* Check if the first tensor is compatible */
  dim = config->info.info[0].dimension;
  g_return_val_if_fail (dim[0] == 195, NULL);
  for (i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
    g_return_val_if_fail (dim[i] == 1, NULL);

  str = g_strdup_printf ("video/x-raw, format = RGBA, " /* Use alpha channel to make the background transparent */
      "width = %u, height = %u", data->width, data->height);
  caps = gst_caps_from_string (str);
  setFramerateFromConfig (caps, config);
  g_free (str);

  return caps;
}

/** @brief tensordec-plugin's TensorDecDef callback */
static size_t
blazepose_getTransformSize (void **pdata, const GstTensorsConfig * config,
                            GstCaps * caps, size_t size, GstCaps * othercaps, GstPadDirection direction)
{
  UNUSED (pdata);
  UNUSED (config);
  UNUSED (caps);
  UNUSED (size);
  UNUSED (othercaps);
  UNUSED (direction);
  return 0;
}

/** @brief Represents a pose */
typedef struct
{
  int valid;
  int x;
  int y;
  int z;
  gfloat visibility;
  gfloat probability;
} blazepose;

/**
 * @brief Check if a value is within lower and upper bounds
 * @param value the value to check
 * @param lower_b the lower bound (inclusive)
 * @param upper_b the uppoer bound (exlcusive)
 * @return TRUE if the value is within the bounds, otherwise FALSE
 */
static gboolean
is_value_within(int value, int lower_b, int upper_b)
{
    if (value < lower_b) {
        return FALSE;
    } else if (value >= upper_b) {
        return FALSE;
    } else {
        return TRUE;
    }
}

/**
 * @brief Fill in pixel with PIXEL_VALUE at x,y position. Make thicker (x+1, y+1)
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] coordinate of pixel
 */
static void
setpixel (uint32_t * frame, pose_data * data, int x, int y)
{
    if (is_value_within(x, 0, data->width) && is_value_within(y, 0, data->height)) {
        uint32_t *pos = &frame[y * data->width + x];
        *pos = PIXEL_VALUE;

        if (x + 1 < (int) data->width) {
            pos = &frame[y * data->width + x + 1];
            *pos = PIXEL_VALUE;
        }
        if (y + 1 < (int) data->height) {
            pos = &frame[(y + 1) * data->width + x];
            *pos = PIXEL_VALUE;
        }
    }
}

/**
 * @brief Draw line with dot at the end of line
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] coordinate of two end point of line
 */
static void
draw_line_with_dot (uint32_t * frame, pose_data * data, int x1, int y1, int x2,
    int y2)
{
  int i, dx, sx, dy, sy, err;
  uint32_t *pos;
  int xx[40] =
      { -4, 0, 4, 0, -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1,
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3
  };
  int yy[40] =
      { 0, -4, 0, 4, -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -3, -2,
    -1, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, -1, 0, 1
  };

  int xs, ys, xe, ye;

  if (x1 > x2) {
    xs = x2;
    ys = y2;
    xe = x1;
    ye = y1;
  } else {
    xs = x1;
    ys = y1;
    xe = x2;
    ye = y2;
  }


  for (i = 0; i < 40; i++) {
    if ((ys + yy[i] >= 0) && (ys + yy[i] < (int) data->height) &&
        (xs + xx[i] >= 0) && (xs + xx[i] < (int) data->width)) {
      pos = &frame[(ys + yy[i]) * data->width + xs + xx[i]];
      *pos = PIXEL_VALUE;
    }
    if ((ye + yy[i] >= 0) && (ye + yy[i] < (int) data->height) &&
        (xe + xx[i] >= 0) && (xe + xx[i] < (int) data->width)) {
      pos = &frame[(ye + yy[i]) * data->width + xe + xx[i]];
      *pos = PIXEL_VALUE;
    }
  }


  dx = abs (xe - xs);
  sx = xs < xe ? 1 : -1;
  dy = abs (ye - ys);
  sy = ys < ye ? 1 : -1;
  err = (dx > dy ? dx : -dy) / 2;

  while (setpixel (frame, data, xs, ys), xs != xe || ys != ye) {
    int e2 = err;
    if (e2 > -dx) {
      err -= dy;
      xs += sx;
    }
    if (e2 < dy) {
      err += dx;
      ys += sy;
    }
  }
}

/**
 * @brief Draw lable with the given results (pose) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] results The final results to be drawn.
 */
static void
draw_label (uint32_t * frame, pose_data * data, blazepose * xydata)
{
  int x1, y1, x2, y2;
  uint32_t *pos1, *pos2;

  guint i, j, label_len;
  guint pose_size = data->total_labels;
  char *label;
  for (i = 0; i < pose_size; i++) {
    if (xydata[i].valid) {
      pose_metadata_t *md = pose_get_metadata_by_id (data, i);
      x1 = xydata[i].x;
      y1 = xydata[i].y;
      if (md == NULL)
        continue;
      label = md->label;
      label_len = strlen (label);
      y1 = MAX (0, (y1 - 14));
      pos1 = &frame[y1 * data->width + x1];
      for (j = 0; j < label_len; j++) {
        unsigned int char_index = label[j];
        if ((x1 + 8) > (int) data->width)
          break;
        pos2 = pos1;
        for (y2 = 0; y2 < 13; y2++) {
          for (x2 = 0; x2 < 8; x2++) {
            *(pos2 + x2) = singleLineSprite[char_index][y2][x2];
          }
          pos2 += data->width;
        }
        x1 += 9;
        pos1 += 9;
      }
    }
  }
}

/**
 * @brief Draw with the given results (pose) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] bdata The bouding-box internal data.
 * @param[in] results The final results to be drawn.
 */
static void
draw (GstMapInfo * out_info, pose_data * data, GArray * results)
{
  guint i;
  gint j;
  uint32_t *frame = (uint32_t *) out_info->data;        /* Let's draw per pixel (4bytes) */
  guint pose_size = data->total_labels;

  blazepose **XYdata = g_new0 (blazepose *, pose_size);
  if (!XYdata) {
    ml_loge ("The memory allocation is failed.");
    return;
  }

  for (i = 0; i < pose_size; i++) {
    XYdata[i] = &g_array_index (results, blazepose, i);
    if (XYdata[i]->probability < 0.5) {
      XYdata[i]->valid = FALSE;
    }
  }

  for (i = 0; i < pose_size; i++) {
    pose_metadata_t *smd;
    if (XYdata[i]->valid == FALSE)
      continue;
    smd = pose_get_metadata_by_id (data, i);
    if (smd == NULL)
      continue;
    for (j = 0; j < smd->num_connections; j++) {
      guint k = smd->connections[j];
      /* Have we already drawn the connection ? */
      if ((k > data->total_labels) || (k < i))
        continue;
      /* Is the body point valid ? */
      if (XYdata[k]->valid == FALSE)
        continue;
      draw_line_with_dot (frame, data,
          XYdata[i]->x, XYdata[i]->y, XYdata[k]->x, XYdata[k]->y);
    }
  }

  draw_label (frame, data, *XYdata);

  g_free (XYdata);
}

/** @brief tensordec-plugin's TensorDecDef callback */
static GstFlowReturn
blazepose_decode (void **pdata, const GstTensorsConfig * ,
                  const GstTensorMemory * input, GstBuffer * outbuf)
{
  pose_data *data = *pdata;
  const size_t size = (size_t) data->width * data->height * 4;   /* RGBA */
  GstMapInfo out_info;
  GstMemory *out_mem;
  GArray *results = NULL;
  float *output_ld_3d = NULL, *output_poseflag = NULL;
  guint pose_size, index;

  g_assert (outbuf); /** GST Internal Bug */
  /* Ensure we have outbuf properly allocated */
  if (gst_buffer_get_size (outbuf) == 0) {
    out_mem = gst_allocator_alloc (NULL, size, NULL);
  } else {
    if (gst_buffer_get_size (outbuf) < size) {
      gst_buffer_set_size (outbuf, size);
    }
    out_mem = gst_buffer_get_all_memory (outbuf);
  }
  if (!gst_memory_map (out_mem, &out_info, GST_MAP_WRITE)) {
    gst_memory_unref (out_mem);
    ml_loge ("Cannot map output memory / tensordec-pose.\n");
    return GST_FLOW_ERROR;
  }
  /** reset the buffer with alpha 0 / black */
  memset (out_info.data, 0, size);

  pose_size = data->total_labels;

  results = g_array_sized_new (FALSE, TRUE, sizeof (blazepose), pose_size);

  // output tensors
  // [0] 195:1:1:1 - 39*5 (x, y, z, visibility, presence) x,y 0-255 | z | visbility,presence [min_float, max_float]
  // [1] 1:1:1:1
  // [2a] 1:256:256:1 (person segmentation propability 0.0 - 1) for FULL
  // [2b] 1:128:128:1 (person segmentation propability 0.0 - 1) for LITE and HEAVY
  // [3] 39:64:64:1 output_heatmap
  // [4] 117:1:1:1 - 39*3 (world x, world y, world z) (meters from hip -1.5 - 1.5)
  g_assert_cmpint(input[0].size,>=,sizeof(float)*pose_size*5);
  output_ld_3d = input[0].data;

  g_assert_cmpint(input[1].size,>=,sizeof(float)*1);
  output_poseflag = input[1].data;

  if (output_poseflag[0] <= 0.5) {
    goto skip;
  }
  for (index = 0; index < pose_size; index++) {
    blazepose p;
    p.x = (output_ld_3d[0] * (float)data->width) / (float)data->i_width;
    p.y = (output_ld_3d[1] * (float)data->height) / (float)data->i_width;
    p.z = (output_ld_3d[2] * (float)data->width) / (float)data->i_width;
    p.visibility = _sigmoid(output_ld_3d[3]);
    p.probability = _sigmoid(output_ld_3d[4]);
    p.valid = (p.probability > 0.95) &&
              (p.x >= 0) && ((guint) p.x < data->width) &&
              (p.y >= 0) && ((guint) p.y < data->height);

    g_array_append_val (results, p);
    output_ld_3d += 5 * sizeof(float);
  }

  draw (&out_info, data, results);

  skip:
  
  g_array_free (results, TRUE);
  gst_memory_unmap (out_mem, &out_info);
  if (gst_buffer_get_size (outbuf) == 0)
    gst_buffer_append_memory (outbuf, out_mem);
  else
    gst_memory_unref (out_mem);

  return GST_FLOW_OK;
}

static gchar decoder_subplugin_blazepose_estimation[] = "blazepose_estimation";
/** @brief Pose Estimation tensordec-plugin TensorDecDef instance */
static GstTensorDecoderDef blazeposeEstimation = {
  .modename = decoder_subplugin_blazepose_estimation,
  .init = blazepose_init,
  .exit = blazepose_exit,
  .setOption = blazepose_setOption,
  .getOutCaps = blazepose_getOutCaps,
  .getTransformSize = blazepose_getTransformSize,
  .decode = blazepose_decode
};

/** @brief Initialize this object for tensordec-plugin */
void
init_blazepose (void)
{
  nnstreamer_decoder_probe (&blazeposeEstimation);
}

/** @brief Destruct this object for tensordec-plugin */
void
finish_blazepose (void)
{
  nnstreamer_decoder_exit (blazeposeEstimation.modename);
}
