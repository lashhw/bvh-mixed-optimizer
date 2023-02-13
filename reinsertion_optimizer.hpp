#ifndef BVH_MIXED_OPTIMIZER_REINSERTION_OPTIMIZER_HPP
#define BVH_MIXED_OPTIMIZER_REINSERTION_OPTIMIZER_HPP

#include "utility.hpp"

struct ReinsertionOptimizer {
    bvh::Bvh<float> &bvh;
    double c_t;
    double c_i;

    ReinsertionOptimizer(bvh::Bvh<float> &bvh,
                         double c_t,
                         double c_i) : bvh(bvh),
                                       c_t(c_t),
                                       c_i(c_i) { }

    double sah_cost_recursive(const bvh::Bvh<float>::Node &node) {
        if (node.is_leaf())
            return c_i * node.primitive_count;
        else {
            const bvh::Bvh<float>::Node &left_node = bvh.nodes[node.first_child_or_primitive];
            const bvh::Bvh<float>::Node &right_node = bvh.nodes[node.first_child_or_primitive + 1];
            return c_t +
                   half_area(left_node.bounds) / half_area(node.bounds) * sah_cost_recursive(left_node) +
                   half_area(right_node.bounds) / half_area(node.bounds) * sah_cost_recursive(right_node);
        }
    }

    double sah_cost() {
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

    int find_target_idx(int insert_idx) {
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

    void optimize() {
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
                int target_idx = find_target_idx(victim_left_idx);
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
                target_idx = find_target_idx(victim_idx);
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

                std::cout << sah_cost() << std::endl;
            }
        }
    }

};

#endif //BVH_MIXED_OPTIMIZER_REINSERTION_OPTIMIZER_HPP
