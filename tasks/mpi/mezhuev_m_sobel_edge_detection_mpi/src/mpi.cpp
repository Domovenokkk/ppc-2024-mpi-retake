#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "mpi/mezhuev_m_sobel_edge_detection_mpi/include/mpi.hpp"

namespace mezhuev_m_sobel_edge_detection_mpi {

bool SobelEdgeDetection::PreProcessingImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs_count.empty() || task_data->inputs_count[0] == 0 || task_data->outputs_count.empty() ||
      task_data->outputs_count[0] == 0) {
    return false;
  }

  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  size_t data_size = task_data->inputs_count[0];
  size_t width = static_cast<size_t>(std::sqrt(data_size));
  size_t height = width;

  if (width < 3 || height < 3) {
    return false;
  }

  gradient_x.resize(data_size);
  gradient_y.resize(data_size);
  return true;
}

bool SobelEdgeDetection::ValidationImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs.size() != 1 || task_data->outputs.size() != 1) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool SobelEdgeDetection::RunImpl() {
  if (!task_data || task_data->inputs.empty() || task_data->outputs.empty() || task_data->inputs_count.empty() ||
      task_data->outputs_count.empty()) {
    return false;
  }

  uint8_t* input_image = task_data->inputs[0];
  uint8_t* output_image = task_data->outputs[0];

  if (input_image == nullptr || output_image == nullptr) {
    return false;
  }

  size_t data_size = task_data->inputs_count[0];
  size_t width = static_cast<size_t>(std::sqrt(data_size));
  size_t height = width;

  if (width < 3 || height < 3) {
    return false;
  }

  boost::mpi::communicator world;
  int rank = world.rank();
  int size = world.size();

  if (height <= static_cast<size_t>(size)) {
    if (rank == 0) {
      for (size_t y = 1; y < height - 1; ++y) {
        for (size_t x = 1; x < width - 1; ++x) {
          int gx = 0, gy = 0;
          static constexpr int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
          static constexpr int sobel_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

          for (int ky = -1; ky <= 1; ++ky) {
            for (int kx = -1; kx <= 1; ++kx) {
              uint8_t pixel_value = input_image[(y + ky) * width + (x + kx)];
              gx += sobel_x[ky + 1][kx + 1] * pixel_value;
              gy += sobel_y[ky + 1][kx + 1] * pixel_value;
            }
          }

          int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
          output_image[y * width + x] = static_cast<uint8_t>(std::min(magnitude, 255));
        }
      }
    }
    return true;
  }

  static constexpr int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  static constexpr int sobel_y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  size_t rows_per_process = height / size;
  size_t extra_rows = height % size;
  size_t start_row = rank * rows_per_process + std::min(rank, static_cast<int>(extra_rows));
  size_t end_row = (rank + 1) * rows_per_process + std::min(rank + 1, static_cast<int>(extra_rows));

  if (rank > 0) {
    world.send(rank - 1, 0, input_image + start_row * width, width);
    world.recv(rank - 1, 0, input_image + (start_row - 1) * width, width);
  }
  if (rank < size - 1) {
    world.recv(rank + 1, 0, input_image + end_row * width, width);
    world.send(rank + 1, 0, input_image + (end_row - 1) * width, width);
  }

  for (size_t y = std::max(start_row, size_t(1)); y < std::min(end_row, height - 1); ++y) {
    for (size_t x = 1; x < width - 1; ++x) {
      int gx = 0, gy = 0;

      for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
          uint8_t pixel_value = input_image[(y + ky) * width + (x + kx)];
          gx += sobel_x[ky + 1][kx + 1] * pixel_value;
          gy += sobel_y[ky + 1][kx + 1] * pixel_value;
        }
      }

      int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
      output_image[y * width + x] = static_cast<uint8_t>(std::min(magnitude, 255));
    }
  }

  if (rank == 0) {
    for (int i = 1; i < size; ++i) {
      size_t worker_start_row = i * rows_per_process + std::min(i, static_cast<int>(extra_rows));
      size_t worker_end_row = (i + 1) * rows_per_process + std::min(i + 1, static_cast<int>(extra_rows));
      worker_end_row = std::min(worker_end_row, height);

      size_t rows_to_receive = worker_end_row - worker_start_row;
      if (rows_to_receive > 0) {
        world.recv(i, 0, output_image + worker_start_row * width, rows_to_receive * width);
      }
    }
  } else {
    size_t rows_to_send = end_row - start_row;
    if (rows_to_send > 0) {
      world.send(0, 0, output_image + start_row * width, rows_to_send * width);
    }
  }

  return true;
}


bool SobelEdgeDetection::PostProcessingImpl() {
  if (!task_data || task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
    return false;
  }

  size_t output_size = task_data->outputs_count[0];

  for (size_t i = 0; i < output_size; ++i) {
    if (task_data->outputs[0][i] > 0) {
      return true;
    }
  }

  return true;
}

}  // namespace mezhuev_m_sobel_edge_detection_mpi
