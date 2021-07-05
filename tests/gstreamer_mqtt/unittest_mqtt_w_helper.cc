/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * @file        unittest_mqtt_w_helper.cc
 * @date        28 May 2021
 * @brief       Unit test for GStreamer MQTT elements using GstMqttTestHelper
 * @see         https://github.com/nnstreamer/nnstreamer
 * @author      Wook Song <wook16.song@samsung.com>
 * @bug         No known bugs
 */

#include <glib.h>
#include <gst/base/gstbasesrc.h>
#include <gst/check/gstharness.h>
#include <gst/gst.h>
#include <gtest/gtest.h>

#include <MQTTAsync.h>
#include <unittest_util.h>

#include <future>

#include "GstMqttTestHelper.hh"
#include "mqttcommon.h"

std::unique_ptr<GstMqttTestHelper> GstMqttTestHelper::mInstance;
std::once_flag GstMqttTestHelper::mOnceFlag;

/**
 * @brief A mock function for MQTTAsync_create() in paho-mqtt-c
 */
int MQTTAsync_create (MQTTAsync *handle, const char *serverURI,
    const char *clientId, int persistence_type, void *persistence_context)
{
  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_connect() in paho-mqtt-c
 */
int MQTTAsync_connect (MQTTAsync handle,
    const MQTTAsync_connectOptions *options)
{
  MQTTAsync_successData data;
  void *ctx = GstMqttTestHelper::getInstance ().getContext ();
  auto ret = std::async (std::launch::async, options->onSuccess, ctx, &data);

  GstMqttTestHelper::getInstance ().setIsConnected (true);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_setCallbacks() in paho-mqtt-c
 */
int MQTTAsync_setCallbacks(MQTTAsync handle, void * context,
    MQTTAsync_connectionLost * cl, MQTTAsync_messageArrived * ma,
    MQTTAsync_deliveryComplete * dc)
{
  GstMqttTestHelper::getInstance ().init (context);
  GstMqttTestHelper::getInstance ().setCallbacks (cl, ma, dc);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_send() in paho-mqtt-c
 */
int MQTTAsync_send (MQTTAsync handle, const char *destinationName,
    int payloadlen, const void * payload, int qos, int retained,
    MQTTAsync_responseOptions * response)
{
  void *ctx = GstMqttTestHelper::getInstance ().getContext ();
  std::future<void> ret;
  MQTTAsync_successData data;

  if (GstMqttTestHelper::getInstance ().getFailSend ()) {
    MQTTAsync_failureData failure_data;

    failure_data.code = -1;
    failure_data.message = "";
    ret = std::async (std::launch::async, response->onFailure, ctx,
        &failure_data);
    return MQTTASYNC_FAILURE;
  }

  ret = std::async (std::launch::async, response->onSuccess, ctx,
      &data);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for int MQTTAsync_isConnected() in paho-mqtt-c
 */
int MQTTAsync_isConnected (MQTTAsync handle)
{
  return GstMqttTestHelper::getInstance ().getIsConnected ();
}

/**
 * @brief A mock function for MQTTAsync_disconnect() in paho-mqtt-c
 */
int MQTTAsync_disconnect (MQTTAsync handle,
    const MQTTAsync_disconnectOptions *options)
{
  void *ctx;
  std::future<void> ret;
  MQTTAsync_successData data;

  if (!options)
    return MQTTASYNC_SUCCESS;

  ctx = options->context;
  GstMqttTestHelper::getInstance ().setIsConnected (false);
  if (GstMqttTestHelper::getInstance ().getFailDisconnect ()) {
    MQTTAsync_failureData fdata;

    fdata.code = -1;
    fdata.message = "";
    ret = std::async (std::launch::async, options->onFailure, ctx,
      &fdata);

    return MQTTASYNC_FAILURE;
  }

  ret = std::async (std::launch::async, options->onSuccess, ctx,
      &data);

  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_destroy() in paho-mqtt-c
 */
void MQTTAsync_destroy (MQTTAsync *handle)
{
  return;
}

/**
 * @brief A mock function for MQTTAsync_subscribe() in paho-mqtt-c
 */
int MQTTAsync_subscribe (MQTTAsync handle, const char * topic, int qos,
    MQTTAsync_responseOptions * response)
{
  MQTTAsync_successData data;
  std::future<void> ret;
  void *ctx = response->context;

  if (GstMqttTestHelper::getInstance ().getFailSubscribe ()) {
    MQTTAsync_failureData fdata;

    fdata.code = -1;
    fdata.message = "";
    ret = std::async (std::launch::async, response->onFailure, ctx, &fdata);
    return MQTTASYNC_FAILURE;
  }

  ret = std::async (std::launch::async, response->onSuccess, ctx, &data);
  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A mock function for MQTTAsync_unsubscribe() in paho-mqtt-c
 */
int MQTTAsync_unsubscribe (MQTTAsync handle, const char * topic,
    MQTTAsync_responseOptions * response)
{
  void *ctx = response->context;
  MQTTAsync_successData data;
  std::future<void> ret;

  if (GstMqttTestHelper::getInstance ().getFailUnsubscribe ()) {
    MQTTAsync_failureData fdata;

    fdata.code = -1;
    fdata.message = "";
    ret = std::async (std::launch::async, response->onFailure, ctx, &fdata);
    return MQTTASYNC_FAILURE;
  }

  ret = std::async (std::launch::async, response->onSuccess, ctx, &data);
  return MQTTASYNC_SUCCESS;
}

/**
 * @brief A helper function to fill the timestamp information into the header
 */
void _set_ts_gst_mqtt_message_hdr (GstElement *elm, GstMQTTMessageHdr *hdr,
    const GstClockTimeDiff diff_sent, const GstClockTime duration)
{
  GstClockTime base_time;
  GstClockTime cur_time;
  GstClockTimeDiff diff;
  GstClock *clock;

  hdr->base_time_epoch = GST_CLOCK_TIME_NONE;
  clock = gst_test_clock_new ();
  base_time = gst_element_get_base_time (elm) + diff_sent;
  cur_time = gst_clock_get_time (clock);
  gst_object_unref (clock);

  diff = GST_CLOCK_DIFF (base_time, cur_time);
  hdr->base_time_epoch = g_get_real_time () * GST_US_TO_NS_MULTIPLIER - diff;
  hdr->sent_time_epoch = hdr->base_time_epoch + diff_sent;
  hdr->duration = duration;
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (push a GstBuffer)
 */
TEST (testMqttSinkWithHelper, sinkPush0)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;

  g_object_set (h->element, "debug", true, NULL);
  gst_harness_add_src_parse (h, "videotestsrc is-live=1 ! queue", TRUE);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  ret = gst_harness_push_from_src (h);

  EXPECT_EQ (ret, GST_FLOW_OK);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (Push multiple GstBuffers with num-buffers)
 */
TEST (testMqttSinkWithHelper, sinkPush1)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  const gint num_buffers = 10;
  gint i;

  g_object_set (h->element, "num-buffers", num_buffers, NULL);
  g_object_set (h->element, "debug", true, NULL);

  gst_harness_add_src_parse (h, "videotestsrc is-live=1 ! queue", TRUE);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  for (i = 0; i < num_buffers; ++i) {
    ret = gst_harness_push_from_src (h);
    EXPECT_EQ (ret, GST_FLOW_OK);
  }

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (MQTTAsync_send failure case)
 */
TEST (testMqttSinkWithHelper, sinkPush0_n)
{
  const static gsize data_size = 1024;
  GstHarness *h = gst_harness_new ("mqttsink");
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "debug", true, NULL);

  in_buf = gst_harness_create_buffer (h, data_size);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().setFailSend (true);
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_ERROR);
  GstMqttTestHelper::getInstance ().setFailSend (false);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (MQTTAsync_disconnect failure case)
 */
TEST (testMqttSinkWithHelper, sinkPush1_n)
{
  const static gsize data_size = 1024;
  GstHarness *h = gst_harness_new ("mqttsink");
  GstBuffer *in_buf;
  GstFlowReturn ret;

  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "debug", true, NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();

  in_buf = gst_harness_create_buffer (h, data_size);
  GstMqttTestHelper::getInstance ().setFailDisconnect (true);
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_OK);
  GstMqttTestHelper::getInstance ().setFailDisconnect (false);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (Push an empty buffer)
 */
TEST (testMqttSinkWithHelper, sinkPush2_n)
{
  GstHarness *h;
  GstBuffer *in_buf;
  GstFlowReturn ret;

  h = gst_harness_new ("mqttsink");
  ASSERT_TRUE (h != NULL);

  g_object_set (h->element, "debug", true, NULL);
  GstMqttTestHelper::getInstance ().initFailFlags ();

  in_buf = gst_buffer_new ();
  ret = gst_harness_push (h, in_buf);

  EXPECT_EQ (ret, GST_FLOW_ERROR);

  gst_harness_teardown (h);
}

/**
 * @brief Test for mqttsink with GstMqttTestHelper (Push GstBuffers more then num-buffers)
 */
TEST (testMqttSinkWithHelper, sinkPush3_n)
{
  GstHarness *h = gst_harness_new ("mqttsink");
  GstFlowReturn ret;
  const gint num_buffers = 10;
  gint i;

  g_object_set (h->element, "num-buffers", num_buffers, NULL);
  g_object_set (h->element, "debug", true, NULL);

  gst_harness_add_src_parse (h, "videotestsrc is-live=1 ! queue", TRUE);
  GstMqttTestHelper::getInstance ().initFailFlags ();
  for (i = 0; i < num_buffers; ++i) {
    ret = gst_harness_push_from_src (h);

    EXPECT_EQ (ret, GST_FLOW_OK);
  }

  ret = gst_harness_push_from_src (h);
  EXPECT_NE (ret, GST_FLOW_OK);

  gst_harness_teardown (h);
}

/**
 * @brief A helper function for the generation of a dummy MQTT message
 */
static void _gen_dummy_mqtt_msg (MQTTAsync_message *msg, GstMQTTMessageHdr *hdr,
    const gsize len_buf)
{
  gboolean mapped;
  GstBuffer *buf;
  GstMemory *mem;
  GstMapInfo map;

  buf = gst_buffer_new_allocate (NULL, len_buf, NULL);
  ASSERT_FALSE (buf == NULL);

  mem = gst_buffer_get_all_memory (buf);
  ASSERT_FALSE (mem == NULL);

  mapped = gst_memory_map (mem, &map, GST_MAP_READ);
  ASSERT_EQ (mapped, TRUE);

  memcpy (msg->payload, hdr, GST_MQTT_LEN_MSG_HDR);
  memcpy (&((guint8 *) msg->payload)[GST_MQTT_LEN_MSG_HDR], map.data,
      len_buf);

  gst_memory_unmap (mem, &map);
  gst_buffer_unref (buf);
}

/**
 * @brief Test mqttsrc using a proper pipeline description #1
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch0)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline = g_strdup_printf (
      "mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d ! "
      "capsfilter caps=%s ! videoconvert ! videoscale ! fakevideosink",
      topic_name, 1, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;

  pipeline = gst_parse_launch (str_pipeline, &err);
  ASSERT_FALSE (pipeline == NULL);
  ASSERT_TRUE (err == NULL);

  GstMqttTestHelper::getInstance ().initFailFlags ();

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof(*msg));
  ASSERT_FALSE (msg == NULL);

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CPAS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CPAS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = (MQTTAsync_message *) g_try_malloc0 (msg->payloadlen);
  ASSERT_FALSE (msg->payload == NULL);

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (cur_state, GST_STATE_PLAYING);

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (pipeline);

  g_free (topic_name);
  g_free (str_pipeline);
  g_free (caps_str);
}

/**
 * @brief Test mqttsrc using a proper pipeline description #2 (dynamically re-negotiating GstCaps)
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch1)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline = g_strdup_printf (
      "mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d ! "
      "capsfilter caps=%s ! videoconvert ! videoscale ! fakevideosink",
      topic_name, 2, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;

  pipeline = gst_parse_launch (str_pipeline, &err);
  ASSERT_FALSE (pipeline == NULL);
  ASSERT_TRUE (err == NULL);

  GstMqttTestHelper::getInstance ().initFailFlags ();

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof(*msg));
  ASSERT_FALSE (msg == NULL);

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CPAS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CPAS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = g_try_malloc0 (msg->payloadlen);
  ASSERT_FALSE (msg->payload == NULL);

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (cur_state, GST_STATE_PLAYING);

  g_free (caps_str);

  /** Changing caps while the pipeline is in the GST_STATE_PLAYING state */
  caps_str = g_strdup ("video/x-raw,width=320,height=160,format=YUY2");
  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CPAS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CPAS_STR - 1));
  memcpy (msg->payload, &hdr, GST_MQTT_LEN_MSG_HDR);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (pipeline);

  g_free (topic_name);
  g_free (str_pipeline);
  g_free (caps_str);
}

/**
 * @brief Fail test case for mqttsrc #0 (MQTTAsync_subscribe failure case)
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch0_n)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline = g_strdup_printf (
      "mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d ! "
      "capsfilter caps=%s ! videoconvert ! videoscale ! fakevideosink",
      topic_name, 1, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;

  pipeline = gst_parse_launch (str_pipeline, &err);
  ASSERT_FALSE (pipeline == NULL);
  ASSERT_TRUE (err == NULL);

  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().setFailSubscribe (TRUE);

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof(*msg));
  ASSERT_FALSE (msg == NULL);

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CPAS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CPAS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = (MQTTAsync_message *) g_try_malloc0 (msg->payloadlen);
  ASSERT_FALSE (msg->payload == NULL);

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_FAILURE);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  GstMqttTestHelper::getInstance ().setFailSubscribe (FALSE);
  gst_object_unref (pipeline);

  g_free (topic_name);
  g_free (str_pipeline);
  g_free (caps_str);
}

/**
 * @brief Fail test case for mqttsrc #1 (MQTTAsync_disconnect failure case)
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch1_n)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline = g_strdup_printf (
      "mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d ! "
      "capsfilter caps=%s ! videoconvert ! videoscale ! fakevideosink",
      topic_name, 1, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;

  pipeline = gst_parse_launch (str_pipeline, &err);
  ASSERT_FALSE (pipeline == NULL);
  ASSERT_TRUE (err == NULL);

  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().setFailDisconnect (TRUE);

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof(*msg));
  ASSERT_FALSE (msg == NULL);

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CPAS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CPAS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = (MQTTAsync_message *) g_try_malloc0 (msg->payloadlen);
  ASSERT_FALSE (msg->payload == NULL);

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  EXPECT_EQ (cur_state, GST_STATE_PLAYING);

  GstMqttTestHelper::getInstance ().setFailDisconnect (FALSE);

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (pipeline);

  g_free (topic_name);
  g_free (str_pipeline);
  g_free (caps_str);
}

/**
 * @brief Fail test case for mqttsrc #2 (MQTTAsync_unsubscribe failure case)
 */
TEST (testMqttSrcWithHelper, srcNormalLaunch2)
{
  const gsize len_buf = 1024;
  gchar *caps_str = g_strdup ("video/x-raw,width=640,height=320,format=RGB");
  gchar *topic_name = g_strdup ("test_topic");
  gchar *str_pipeline = g_strdup_printf (
      "mqttsrc sub-topic=%s debug=true is-live=true num-buffers=%d ! "
      "capsfilter caps=%s ! videoconvert ! videoscale ! fakevideosink",
      topic_name, 1, caps_str);
  GError *err = NULL;
  GstElement *pipeline;
  GstStateChangeReturn ret;
  GstState cur_state;
  GstMQTTMessageHdr hdr;
  MQTTAsync_message *msg;
  std::future<int> ma_ret;

  pipeline = gst_parse_launch (str_pipeline, &err);
  ASSERT_FALSE (pipeline == NULL);
  ASSERT_TRUE (err == NULL);

  GstMqttTestHelper::getInstance ().initFailFlags ();
  GstMqttTestHelper::getInstance ().setFailUnsubscribe (TRUE);

  msg = (MQTTAsync_message *) g_try_malloc0 (sizeof(*msg));
  ASSERT_FALSE (msg == NULL);

  _set_ts_gst_mqtt_message_hdr (pipeline, &hdr, GST_SECOND, 500 * GST_MSECOND);
  ret = gst_element_set_state (pipeline, GST_STATE_PAUSED);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_NO_PREROLL);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  memset (hdr.gst_caps_str, '\0', GST_MQTT_MAX_LEN_GST_CPAS_STR);
  memcpy (hdr.gst_caps_str, caps_str,
      MIN (strlen (caps_str), GST_MQTT_MAX_LEN_GST_CPAS_STR - 1));
  hdr.num_mems = 1;
  hdr.size_mems[0] = len_buf;

  msg->payloadlen = GST_MQTT_LEN_MSG_HDR + len_buf;
  msg->payload = (MQTTAsync_message *) g_try_malloc0 (msg->payloadlen);
  ASSERT_FALSE (msg->payload == NULL);

  ret = gst_element_set_state (pipeline, GST_STATE_PLAYING);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  _gen_dummy_mqtt_msg (msg, &hdr, len_buf);

  ma_ret = std::async (std::launch::async,
      GstMqttTestHelper::getInstance ().getCbMessageArrived (),
      GstMqttTestHelper::getInstance ().getContext (), topic_name, 0, msg);
  EXPECT_TRUE (ma_ret.get ());

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_FAILURE);
  EXPECT_EQ (cur_state, GST_STATE_PAUSED);

  GstMqttTestHelper::getInstance ().setFailUnsubscribe (FALSE);

  ret = gst_element_set_state (pipeline, GST_STATE_NULL);
  EXPECT_NE (ret, GST_STATE_CHANGE_FAILURE);

  ret = gst_element_get_state (pipeline, &cur_state, NULL, GST_CLOCK_TIME_NONE);
  EXPECT_EQ (ret, GST_STATE_CHANGE_SUCCESS);
  gst_object_unref (pipeline);

  g_free (topic_name);
  g_free (str_pipeline);
  g_free (caps_str);
}

/**
 * @brief Main GTest
 */
int
main (int argc, char **argv)
{
  int result = -1;

  try {
    testing::InitGoogleTest (&argc, argv);
  } catch (...) {
    g_warning ("catch 'testing::internal::<unnamed>::ClassUniqueToAlwaysTrue'");
  }

  gst_init (&argc, &argv);

  try {
    result = RUN_ALL_TESTS ();
  } catch (...) {
    g_warning ("catch `testing::internal::GoogleTestFailureException`");
  }

  return result;
}
