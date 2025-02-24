#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace mezhuev_m_sobel_edge_detection_mpi {

class SobelEdgeDetection : public ppc::core::Task {
 public:
  SobelEdgeDetection(boost::mpi::communicator& world, std::shared_ptr<ppc::core::TaskData> task_data)
      : Task(std::move(task_data)), world_(world) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  const std::vector<int>& get_gradient_x() const { return gradient_x; }
  const std::vector<int>& get_gradient_y() const { return gradient_y; }

 private:
  boost::mpi::communicator& world_;
  std::vector<int> gradient_x;
  std::vector<int> gradient_y;
};

}  // namespace mezhuev_m_sobel_edge_detection_mpi