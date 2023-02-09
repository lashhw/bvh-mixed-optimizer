#include <iostream>
#include <bvh/triangle.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include "third_party/happly/happly.h"

double half_area(const float *bounds) {
    double d_x = double(bounds[1]) - double(bounds[0]);
    double d_y = double(bounds[3]) - double(bounds[2]);
    double d_z = double(bounds[5]) - double(bounds[4]);
    return (d_x + d_y) * d_z + d_x * d_y;
}

double sah_cost_recursive(const bvh::Bvh<float> &bvh, const bvh::Bvh<float>::Node &node, double c_t, double c_i) {
    if (node.is_leaf())
        return c_i * node.primitive_count;
    else {
        const bvh::Bvh<float>::Node &left_node = bvh.nodes[node.first_child_or_primitive];
        const bvh::Bvh<float>::Node &right_node = bvh.nodes[node.first_child_or_primitive+1];
        return c_t +
               half_area(left_node.bounds) / half_area(node.bounds) * sah_cost_recursive(bvh, left_node, c_t, c_i) +
               half_area(right_node.bounds) / half_area(node.bounds) * sah_cost_recursive(bvh, right_node, c_t, c_i);
    }
}

double sah_cost(const bvh::Bvh<float> &bvh, double c_t, double c_i) {
    double cost = 0;
    double root_half_area = half_area(bvh.nodes[0].bounds);

    for (int i = 0; i < bvh.node_count; i++) {
        bvh::Bvh<float>::Node &node = bvh.nodes[i];
        if (node.is_leaf())
            cost += node.primitive_count * c_i * half_area(node.bounds) / root_half_area;
        else
            cost += c_t * half_area(node.bounds) / root_half_area;
    }

    return cost;
}

int main() {
    happly::PLYData ply_data("kitchen.ply");
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
    std::cout << "global bounding box: ("
              << global_bbox.min[0] << ", " << global_bbox.min[1] << ", " << global_bbox.min[2] << "), ("
              << global_bbox.max[0] << ", " << global_bbox.max[1] << ", " << global_bbox.max[2] << ")" << std::endl;

    bvh::Bvh<float> bvh;
    bvh::SweepSahBuilder<bvh::Bvh<float>> builder(bvh);
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

    std::cout << sah_cost(bvh, 0.3, 1) << std::endl;
    std::cout << sah_cost_recursive(bvh, bvh.nodes[0], 0.3, 1) << std::endl;
}
