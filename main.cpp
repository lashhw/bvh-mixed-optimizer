#include <iostream>
#include <bvh/triangle.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/linear_bvh_builder.hpp>
#include "third_party/happly/happly.h"
#include "utility.hpp"
#include "low_precision_optimizer.hpp"

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

    std::queue<int> queue;
    queue.push(0);
    while (!queue.empty()) {
        int curr = queue.front();
        queue.pop();
        bvh::Bvh<float>::Node &node = bvh.nodes[curr];
        if (node.is_leaf()) {
            cost += node.primitive_count * c_i * half_area(node.bounds) / root_half_area;
        } else {
            cost += c_t * half_area(node.bounds) / root_half_area;
            queue.push(node.first_child_or_primitive);
            queue.push(node.first_child_or_primitive + 1);
        }
    }

    return cost;
}

int find_target_idx(const bvh::Bvh<float> &bvh, int insert_idx) {
    int best_idx = -1;
    double best_cost = std::numeric_limits<double>::max();
    bvh::BoundingBox<float> insert_bbox = bvh.nodes[insert_idx].bounding_box_proxy();
    std::stack<std::pair<int, double>> stack;
    stack.emplace(bvh.nodes[0].first_child_or_primitive, 0);
    stack.emplace(bvh.nodes[0].first_child_or_primitive + 1, 0);
    while (!stack.empty()) {
        std::pair<int, double> curr = stack.top();
        stack.pop();
        double before_half_area = half_area(bvh.nodes[curr.first].bounds);
        bvh::BoundingBox<float> after_bbox = bvh.nodes[curr.first].bounding_box_proxy();
        after_bbox.extend(insert_bbox);
        double after_half_area = half_area(after_bbox);
        if (curr.second + after_half_area < best_cost) {
            best_idx = curr.first;
            best_cost = curr.second + after_half_area;
        }
        if (!bvh.nodes[curr.first].is_leaf()) {
            int left_idx = bvh.nodes[curr.first].first_child_or_primitive;
            int right_idx = left_idx + 1;
            stack.emplace(left_idx, curr.second + after_half_area - before_half_area);
            stack.emplace(right_idx, curr.second + after_half_area - before_half_area);
        }
    }
    return best_idx;
}

void optimize(bvh::Bvh<float> &bvh) {
    // calculate parent index
    int parent[bvh.node_count];
    parent[0] = 0;
    std::queue<int> queue;
    queue.push(0);
    while (!queue.empty()) {
        int curr = queue.front();
        queue.pop();
        if (!bvh.nodes[curr].is_leaf()) {
            int left = bvh.nodes[curr].first_child_or_primitive;
            int right = left + 1;
            parent[left] = curr;
            parent[right] = curr;
            queue.push(left);
            queue.push(right);
        }
    }
    auto update_parent = [&](int idx) {
        if (!bvh.nodes[idx].is_leaf()) {
            parent[bvh.nodes[idx].first_child_or_primitive] = idx;
            parent[bvh.nodes[idx].first_child_or_primitive + 1] = idx;
        }
    };

    for (int iter = 0; iter < 100; iter++) {
        // sort node by surface area
        std::vector<std::pair<double, int>> area_idx_pair;
        queue.push(0);
        while (!queue.empty()) {
            int curr = queue.front();
            queue.pop();
            area_idx_pair.emplace_back(half_area(bvh.nodes[curr].bounds), curr);
            if (!bvh.nodes[curr].is_leaf()) {
                int left = bvh.nodes[curr].first_child_or_primitive;
                int right = left + 1;
                queue.push(left);
                queue.push(right);
            }
        }
        std::sort(area_idx_pair.begin(), area_idx_pair.end(), std::greater<>());

        for (auto [_, victim_idx] : area_idx_pair) {
            // skip root, root's children, and leaf node
            if (victim_idx == 0 ||
                victim_idx == bvh.nodes[0].first_child_or_primitive ||
                victim_idx == bvh.nodes[0].first_child_or_primitive + 1 ||
                bvh.nodes[victim_idx].is_leaf())
                continue;

            // remove victim's left child
            int victim_left_idx = bvh.nodes[victim_idx].first_child_or_primitive;
            int victim_right_idx = victim_left_idx + 1;
            bvh.nodes[victim_idx] = bvh.nodes[victim_right_idx];
            update_parent(victim_idx);
            for (int curr = parent[victim_idx]; curr != 0; curr = parent[curr])
                update_bbox(bvh, curr);

            // place left child
            int target_idx = find_target_idx(bvh, victim_left_idx);
            bvh.nodes[victim_right_idx] = bvh.nodes[target_idx];
            update_parent(victim_right_idx);
            bvh.nodes[target_idx].bounding_box_proxy() = bvh::BoundingBox<float>::empty();
            bvh.nodes[target_idx].bounding_box_proxy().extend(bvh.nodes[victim_left_idx].bounding_box_proxy());
            bvh.nodes[target_idx].bounding_box_proxy().extend(bvh.nodes[victim_right_idx].bounding_box_proxy());
            bvh.nodes[target_idx].primitive_count = 0;
            bvh.nodes[target_idx].first_child_or_primitive = victim_left_idx;
            update_parent(target_idx);
            for (int curr = parent[target_idx]; curr != 0; curr = parent[curr])
                update_bbox(bvh, curr);

            // special case
            if (target_idx == victim_idx)
                victim_idx = victim_right_idx;

            // remove victim's right child
            int victim_parent_idx = parent[victim_idx];
            int victim_sibling_idx;
            if (bvh.nodes[victim_parent_idx].first_child_or_primitive == victim_idx)
                victim_sibling_idx = victim_idx + 1;
            else if (bvh.nodes[victim_parent_idx].first_child_or_primitive + 1 == victim_idx)
                victim_sibling_idx = victim_idx - 1;
            else
                assert(false);
            bvh.nodes[victim_parent_idx] = bvh.nodes[victim_sibling_idx];
            update_parent(victim_parent_idx);
            for (int curr = parent[victim_parent_idx]; curr != 0; curr = parent[curr])
                update_bbox(bvh, curr);

            // determine where to place right child
            target_idx = find_target_idx(bvh, victim_idx);
            bvh.nodes[victim_sibling_idx] = bvh.nodes[target_idx];
            update_parent(victim_sibling_idx);
            bvh.nodes[target_idx].bounding_box_proxy() = bvh::BoundingBox<float>::empty();
            bvh.nodes[target_idx].bounding_box_proxy().extend(bvh.nodes[victim_sibling_idx].bounding_box_proxy());
            bvh.nodes[target_idx].bounding_box_proxy().extend(bvh.nodes[victim_idx].bounding_box_proxy());
            bvh.nodes[target_idx].primitive_count = 0;
            bvh.nodes[target_idx].first_child_or_primitive = std::min(victim_idx, victim_sibling_idx);
            update_parent(target_idx);
            for (int curr = parent[target_idx]; curr != 0; curr = parent[curr])
                update_bbox(bvh, curr);

            std::cout << sah_cost(bvh, 0.3, 1) << std::endl;
        }
    }
}

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

    std::cout << sah_cost(bvh, 0.3, 1) << std::endl;
    std::cout << sah_cost_recursive(bvh, bvh.nodes[0], 0.3, 1) << std::endl;

    LowPrecisionOptimizer optimizer(bvh, 0.3, 0.4, 1, 7, 8);
    std::cout << optimizer.sah_cost() << std::endl;
    std::cout << optimizer.sah_cost_recursive(0, bvh.nodes[0].bounding_box_proxy()) << std::endl;
    optimizer.optimize();

    // optimize(bvh);
}
