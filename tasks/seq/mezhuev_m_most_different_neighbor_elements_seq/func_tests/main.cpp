#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "seq/mezhuev_m_most_different_neighbor_elements_seq/include/seq.hpp"

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationEmptyTaskData) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationMissingInputs) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(3);
  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationSmallInputSize) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationCorrectInputs) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 2, 3};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  EXPECT_TRUE(task.ValidationImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PreProcessingEmptyInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs = {};
  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PreProcessingValidInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 2, 3};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  EXPECT_TRUE(task.PreProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PreProcessingInvalidSizeInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, RunImplCorrectInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 3, 2, 7};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  taskData->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());

  std::vector<int> expected_result = {2, 7};
  EXPECT_EQ(task.getResult(), expected_result);
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, RunImplEqualNeighbors) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {5, 5, 5, 5};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  taskData->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());

  std::vector<int> expected_result = {5, 5};
  EXPECT_EQ(task.getResult(), expected_result);
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PostProcessingValidOutput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {10, 20, 30};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  std::vector<int> output_data(2);
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));
  taskData->outputs_count.push_back(2);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);

  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  EXPECT_TRUE(task.PostProcessingImpl());

  std::vector<int> expected_output = {10, 20};
  EXPECT_EQ(output_data, expected_output);
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, PostProcessingEmptyOutput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();

  std::vector<int> input_data = {10, 20};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  taskData->outputs.clear();
  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);

  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  EXPECT_FALSE(task.PostProcessingImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, RunImplInsufficientInput) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {10};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  ASSERT_FALSE(task.RunImpl());
}

TEST(mezhuev_m_most_different_neighbor_elements_seq, ValidationWithNoOutputSpace) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<int> input_data = {1, 2, 3};
  taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->inputs_count.push_back(static_cast<uint32_t>(input_data.size()));
  taskData->outputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskData->outputs_count.push_back(1);

  mezhuev_m_most_different_neighbor_elements_seq::MostDifferentNeighborElements task(taskData);
  EXPECT_FALSE(task.ValidationImpl());
}
