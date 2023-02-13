#include <iostream>
#include <bvh/triangle.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/linear_bvh_builder.hpp>
#include "third_party/happly/happly.h"
#include "reinsertion_optimizer.hpp"
#include "low_precision_optimizer.hpp"

int main() {
    happly::PLYData ply_data("model.ply");
    std::vector<std::array<double, 3>> v_pos = ply_data.getVertexPositions();
    std::vector<std::vector<size_t>> f_idx = ply_data.getFaceIndices<size_t>();

    std::vector<bvh::Triangle<float>> triangles;
    for (const auto &face : f_idx) {
        triangles.emplace_back(bvh::Vector3<float>(v_pos[face[0]][0], v_pos[face[0]][1], v_pos[face[0]][2]),
                               bvh::Vector3<float>(v_pos[face[1]][0], v_pos[face[1]][1], v_pos[face[1]][2]),
                               bvh::Vector3<float>(v_pos[face[2]][0], v_pos[face[2]][1], v_pos[face[2]][2]));
    }

    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());

    bvh::Bvh<float> bvh;
    bvh::LinearBvhBuilder<bvh::Bvh<float>, uint32_t> builder(bvh);
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

    ReinsertionOptimizer reinsert_optimizer(bvh, 0.4, 1);
    std::cout << reinsert_optimizer.sah_cost() << std::endl;
    std::cout << reinsert_optimizer.sah_cost_recursive(bvh.nodes[0]) << std::endl;
    reinsert_optimizer.optimize();

    LowPrecisionOptimizer lp_optimizer(bvh, 0.15, 0.2, 1, 7, 8);
    std::cout << lp_optimizer.sah_cost() << std::endl;
    std::cout << lp_optimizer.sah_cost_recursive(0, bvh.nodes[0].bounding_box_proxy()) << std::endl;
     lp_optimizer.optimize();
}
