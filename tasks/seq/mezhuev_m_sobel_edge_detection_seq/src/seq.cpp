#include "seq/mezhuev_m_sobel_edge_detection_seq/include/seq.hpp"

#include <cmath>
#include <iostream>

namespace mezhuev_m_sobel_edge_detection_seq {

bool SobelEdgeDetectionSeq::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->inputs_count.empty()) {
    return false;
  }

  size_t total_pixels = task_data->inputs_count[0];
  size_t width = static_cast<size_t>(std::sqrt(total_pixels));
  size_t height = width;

  if (width < 3 || height < 3) {
    return false;
  }

  gradient_x.resize(width * height);
  gradient_y.resize(width * height);

  return true;
}

bool SobelEdgeDetectionSeq::ValidationImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
    return false;
  }

  if (!task_data->inputs[0] || !task_data->outputs[0]) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SobelEdgeDetectionSeq::RunImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  uint8_t* input_image = task_data->inputs[0];
  uint8_t* output_image = task_data->outputs[0];

  if (!input_image || !output_image) {
    return false;
  }

  size_t total_pixels = task_data->inputs_count[0];
  size_t width = static_cast<size_t>(std::sqrt(total_pixels));
  size_t height = width;

  if (width < 3 || height < 3) {
    return false;
  }

  static constexpr int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  static constexpr int sobel_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  for (size_t y = 1; y < height - 1; ++y) {
    for (size_t x = 1; x < width - 1; ++x) {
      int32_t gx = 0, gy = 0;

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          uint8_t pixel_value = input_image[(y + ky) * width + (x + kx)];
          gx += sobel_x[ky + 1][kx + 1] * pixel_value;
          gy += sobel_y[ky + 1][kx + 1] * pixel_value;
        }
      }

      int magnitude = static_cast<int>(std::sqrt(static_cast<double>(gx * gx + gy * gy)));
      output_image[y * width + x] = static_cast<uint8_t>(std::min(std::max(magnitude, 0), 255));
    }
  }

  return true;
}

bool SobelEdgeDetectionSeq::PostProcessingImpl() {
  if (!task_data || task_data->outputs.empty() || !task_data->outputs[0]) {
    return false;
  }

  size_t output_size = task_data->outputs_count[0];

  if (output_size == 0) {
    return false;
  }

  for (size_t i = 0; i < output_size; ++i) {
    if (task_data->outputs[0][i] > 0) {
      return true;
    }
  }

  return true;
}

}  // namespace mezhuev_m_sobel_edge_detection_seq