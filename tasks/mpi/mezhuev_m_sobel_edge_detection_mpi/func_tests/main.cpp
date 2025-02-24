#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi/mezhuev_m_sobel_edge_detection_mpi/include/mpi.hpp"

TEST(mezhuev_m_sobel_edge_detection_mpi, test_basic_case) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

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

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);

  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    bool has_edges = false;
    for (size_t i = 0; i < kImageSize; ++i) {
      if (out[i] > 0) {
        has_edges = true;
        break;
      }
    }
    ASSERT_TRUE(has_edges);
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_small_image_3x3) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr int kSize = 3;
  constexpr int kImageSize = kSize * kSize;

  std::vector<uint8_t> in(kImageSize, 255);
  std::vector<uint8_t> out(kImageSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);

  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    bool all_zero = true;
    for (size_t i = 0; i < kImageSize; ++i) {
      if (out[i] > 0) {
        all_zero = false;
        break;
      }
    }
    ASSERT_TRUE(all_zero);
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_empty_input) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);

  ASSERT_FALSE(sobel_task->PreProcessingImpl());
  ASSERT_FALSE(sobel_task->RunImpl());
  ASSERT_FALSE(sobel_task->ValidationImpl());
  ASSERT_FALSE(sobel_task->PostProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_single_bright_pixel) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  in[(kHeight / 2) * kWidth + (kWidth / 2)] = 255;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);

  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    bool has_edges = std::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; });
    ASSERT_TRUE(has_edges);
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_large_image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr int kWidth = 1024;
  constexpr int kHeight = 1024;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 128);
  std::vector<uint8_t> out(kImageSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);

  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_vertical_gradient) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  for (size_t y = 0; y < kHeight; ++y) {
    for (size_t x = 0; x < kWidth; ++x) {
      in[y * kWidth + x] = static_cast<uint8_t>(y * 50);
    }
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);

  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    bool has_edges = std::any_of(out.begin(), out.end(), [](uint8_t val) { return val > 0; });
    ASSERT_TRUE(has_edges);
  }
}

TEST(mezhuev_m_sobel_edge_detection_mpi, test_black_image) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  constexpr int kWidth = 5;
  constexpr int kHeight = 5;
  constexpr int kImageSize = kWidth * kHeight;

  std::vector<uint8_t> in(kImageSize, 0);
  std::vector<uint8_t> out(kImageSize, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(in.data());
  task_data->inputs_count.emplace_back(kImageSize);
  task_data->outputs.emplace_back(out.data());
  task_data->outputs_count.emplace_back(kImageSize);

  auto sobel_task = std::make_shared<mezhuev_m_sobel_edge_detection_mpi::SobelEdgeDetection>(world, task_data);

  ASSERT_TRUE(sobel_task->PreProcessingImpl());
  ASSERT_TRUE(sobel_task->RunImpl());
  ASSERT_TRUE(sobel_task->ValidationImpl());
  ASSERT_TRUE(sobel_task->PostProcessingImpl());

  if (world.rank() == 0) {
    bool all_zero = std::all_of(out.begin(), out.end(), [](uint8_t val) { return val == 0; });
    ASSERT_TRUE(all_zero);
  }
}
