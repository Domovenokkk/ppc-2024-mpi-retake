#include "mpi/mezhuev_m_lattice_torus/include/mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <ranges>
#include <vector>

namespace mezhuev_m_lattice_torus_mpi {

bool GridTorusTopologyParallel::PreProcessingImpl() { return true; }

bool GridTorusTopologyParallel::ValidationImpl() {
  bool is_valid = true;

  if (world_.rank() == 0) {
    if (!task_data || task_data->inputs.empty() || task_data->inputs_count.empty() || task_data->outputs.empty() ||
        task_data->outputs_count.empty()) {
      is_valid = false;
    } else {
      auto validate_buffers = [&](const auto& buffers, const auto& counts, size_t& total_size) {
        for (size_t i = 0; i < counts.size(); ++i) {
          if (buffers[i] == nullptr || counts[i] == 0) {
            is_valid = false;
            return;
          }
          if (reinterpret_cast<uintptr_t>(buffers[i]) % alignof(uint8_t) != 0) {
            is_valid = false;
            return;
          }
          total_size += counts[i];
        }
      };

      size_t total_input_size = 0;
      size_t total_output_size = 0;
      validate_buffers(task_data->inputs, task_data->inputs_count, total_input_size);
      validate_buffers(task_data->outputs, task_data->outputs_count, total_output_size);

      if (total_input_size != total_output_size) {
        is_valid = false;
      }

      int size = world_.size();
      int grid_dim = 0;
      if (size == 1) {
        grid_dim = 1;
      } else if (size <= 4) {
        grid_dim = 2;
      } else {
        grid_dim = static_cast<int>(std::sqrt(size));
        if (grid_dim * grid_dim != size) {
          is_valid = false;
        }
      }
    }
  }

  bool global_valid = false;
  boost::mpi::all_reduce(world_, is_valid, global_valid, std::logical_and<>());
  return global_valid;
}

bool GridTorusTopologyParallel::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();
  int grid_dim = 0;
  if (size == 1) {
    grid_dim = 1;
  } else if (size <= 4) {
    grid_dim = 2;
  } else {
    grid_dim = static_cast<int>(std::sqrt(size));
  }

  if (size == 1) {
    std::copy(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0], task_data->outputs[0]);
    return true;
  }

  world_.barrier();

  auto compute_neighbors = [grid_dim, size](int rank) -> std::vector<int> {
    int x = rank % grid_dim;
    int y = rank / grid_dim;

    int left = (((x - 1 + grid_dim) % grid_dim) + (y * grid_dim));
    int right = (((x + 1) % grid_dim) + (y * grid_dim));
    int up = (x + (((y - 1 + grid_dim) % grid_dim) * grid_dim));
    int down = (x + (((y + 1) % grid_dim) * grid_dim));

    std::vector<int> neighbors = {left, right, up, down};

    std::erase_if(neighbors, [size, rank](int r) { return r >= size || r == rank; });
    std::ranges::sort(neighbors);
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    return neighbors;
  };

  auto neighbors = compute_neighbors(rank);

  std::vector<uint8_t> send_buffer(task_data->inputs_count[0]);
  std::copy(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0], send_buffer.begin());
  std::vector<uint8_t> recv_buffer(task_data->inputs_count[0]);

  for (int neighbor : neighbors) {
    if (rank < neighbor) {
      world_.send(neighbor, 0, send_buffer);
      world_.recv(neighbor, 0, recv_buffer);
    } else {
      world_.recv(neighbor, 0, recv_buffer);
      world_.send(neighbor, 0, send_buffer);
    }
  }

  std::copy(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0], task_data->outputs[0]);

  world_.barrier();
  return true;
}

bool GridTorusTopologyParallel::PostProcessingImpl() {
  if (!task_data) {
    return false;
  }
  for (size_t i = 0; i < task_data->inputs.size(); ++i) {
    if (std::memcmp(task_data->inputs[i], task_data->outputs[i], task_data->inputs_count[i]) != 0) {
      return false;
    }
  }
  return true;
}

}  // namespace mezhuev_m_lattice_torus_mpi
