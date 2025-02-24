#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/mezhuev_m_sobel_edge_detection_seq/include/seq.hpp"

TEST(mezhuev_m_sobel_edge_detection_seq, test_basic_case) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      in[y * kWidth + x] = static_cast<uint8_t>(x * 50);
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  bool has_edges = false;
  for (size_t i = 0; i < kImageSize; ++i) {
    if (out[i] > 0) {
      has_edges = true;
      break;
    }
  }
  ASSERT_TRUE(has_edges);
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_uniform_image) {
  constexpr int kSize = 5;
  constexpr int kImageSize = kSize * kSize;

  std::vector<uint8_t> in(kImageSize, 128);
  std::vector<uint8_t> out(kImageSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  ASSERT_TRUE(std::all_of(out.begin(), out.end(), [](uint8_t val) { return val == 0; }));
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_empty_input) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_FALSE(sobel_task->PreProcessingImpl());
  ASSERT_FALSE(sobel_task->RunImpl());
  ASSERT_FALSE(sobel_task->ValidationImpl());
  ASSERT_FALSE(sobel_task->PostProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_sharp_contrast) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      in[y * kWidth + x] = (x < kWidth / 2) ? 0 : 255;
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  bool has_edges = false;
  for (size_t i = 0; i < kImageSize; ++i) {
    if (out[i] > 0) {
      has_edges = true;
      break;
    }
  }
  ASSERT_TRUE(has_edges);
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_small_image) {
  constexpr int kWidth = 3;
  constexpr int kHeight = 3;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 255);
  std::vector<uint8_t> out(kImageSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  ASSERT_TRUE(std::all_of(out.begin(), out.end(), [](uint8_t val) { return val == 0; }));
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_noisy_image) {
  constexpr int kWidth = 10;
  constexpr int kHeight = 10;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t i = 0; i < kImageSize; ++i) {
    in[i] = static_cast<uint8_t>(rand() % 256);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  bool has_edges = std::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; });
  ASSERT_TRUE(has_edges);
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_1x1_image) {
  constexpr int kImageSize = 1;

  std::vector<uint8_t> in(kImageSize, 128);
  std::vector<uint8_t> out(kImageSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);

  ASSERT_FALSE(sobel_task->PreProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_2x2_image) {
  constexpr int kImageSize = 4;

  std::vector<uint8_t> in(kImageSize, 255);
  std::vector<uint8_t> out(kImageSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);

  ASSERT_FALSE(sobel_task->PreProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_seq, test_horizontal_gradient) {
  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      in[y * kWidth + x] = static_cast<uint8_t>(x * 50);
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_seq::SobelEdgeDetectionSeq>(task_data);
  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  bool has_edges = std::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; });
  ASSERT_TRUE(has_edges);
}