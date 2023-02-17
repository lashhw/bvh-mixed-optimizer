#ifndef BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP
#define BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP

#include <mpfr.h>
#include "utility.hpp"

struct LowPrecisionOptimizer {
    bvh::Bvh<float> &bvh;
    double c_t_l;
    double c_t_h;
    double c_i;
    mpfr_t tmp;
    mpfr_exp_t exp_min;
    mpfr_exp_t exp_max;
    std::vector<double> actual_half_area;
    std::vector<int> parent;

    LowPrecisionOptimizer(bvh::Bvh<float> &bvh,
                          double c_t_l,
                          double c_t_h,
                          double c_i,
                          mpfr_prec_t mantissa_width,
                          mpfr_exp_t exponent_width) : bvh(bvh),
                                                       c_t_l(c_t_l),
                                                       c_t_h(c_t_h),
                                                       c_i(c_i) {
        mpfr_init2(tmp, mantissa_width + 1);
        exp_min = -(1 << (exponent_width - 1)) + 2;
        exp_max = (1 << (exponent_width - 1));
        actual_half_area.resize(bvh.node_count);
        parent.resize(bvh.node_count);

        // calculate parent index
        parent[0] = -1;
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
    }

    ~LowPrecisionOptimizer() {
        mpfr_clear(tmp);
    }

    void check_exponent_and_set_inf(mpfr_t &num) {
        if (mpfr_number_p(num)) {
            if (mpfr_get_exp(num) < exp_min) mpfr_set_zero(num, mpfr_signbit(num) ? -1 : 1);
            else if (mpfr_get_exp(num) > exp_max) mpfr_set_inf(num, mpfr_signbit(num) ? -1 : 1);
        }
    }

    bvh::BoundingBox<float> to_lp_bbox(const bvh::BoundingBox<float> &bbox) {
        bvh::BoundingBox<float> bbox_low;
        for (int i = 0; i < 3; i++) {
            mpfr_set_flt(tmp, bbox.min[i], MPFR_RNDD);
            check_exponent_and_set_inf(tmp);
            bbox_low.min[i] = mpfr_get_flt(tmp, MPFR_RNDD);
            mpfr_set_flt(tmp, bbox.max[i], MPFR_RNDU);
            check_exponent_and_set_inf(tmp);
            bbox_low.max[i] = mpfr_get_flt(tmp, MPFR_RNDU);
        }
        return bbox_low;
    }

    double sah_cost_recursive(int idx, const bvh::BoundingBox<float> &actual_bbox) {
        const bvh::Bvh<float>::Node &node = bvh.nodes[idx];
        if (node.is_leaf())
            return c_i * node.primitive_count;
        else {
            int left_idx = node.first_child_or_primitive;
            double cost = 0;
            for (int i = 0; i <= 1; i++) {  // 0 for left child, 1 for right child
                const bvh::Bvh<float>::Node &child_node = bvh.nodes[left_idx + i];
                if (child_node.low_precision) {
                    bvh::BoundingBox<float> child_bbox = to_lp_bbox(child_node.bounding_box_proxy());
                    child_bbox.shrink(actual_bbox);
                    cost += c_t_l;
                    cost += half_area(child_bbox) / half_area(actual_bbox) *
                            sah_cost_recursive(left_idx + i, child_bbox);
                } else {
                    cost += c_t_h;
                    cost += half_area(child_node.bounds) / half_area(actual_bbox) *
                            sah_cost_recursive(left_idx + i, child_node.bounding_box_proxy());
                }
            }
            return cost;
        }
    }

    double sah_cost() {
        double cost = 0;
        fill_actual_half_area();

        std::queue<int> queue;
        queue.push(0);
        while (!queue.empty()) {
            int curr = queue.front();
            queue.pop();

            bvh::Bvh<float>::Node &node = bvh.nodes[curr];
            if (node.is_leaf())
                cost += node.primitive_count * c_i * actual_half_area[curr];
            else {
                for (int i = 0; i <= 1; i++) {  // 0 for left child, 1 for right child
                    queue.push(node.first_child_or_primitive + i);
                    bvh::Bvh<float>::Node &child_node = bvh.nodes[node.first_child_or_primitive + i];
                    if (child_node.low_precision)
                        cost += c_t_l * actual_half_area[curr];
                    else
                        cost += c_t_h * actual_half_area[curr];
                }
            }
        }

        return cost / half_area(bvh.nodes[0].bounds);
    }

    void fill_actual_half_area() {
        std::queue<std::pair<int, bvh::BoundingBox<float>>> queue;
        queue.emplace(0, bvh::BoundingBox<float>::full());
        while (!queue.empty()) {
            auto [idx, parent_bbox] = queue.front();
            queue.pop();
            bvh::BoundingBox<float> tmp_bbox;
            if (bvh.nodes[idx].low_precision) {
                tmp_bbox = parent_bbox;
                tmp_bbox.shrink(to_lp_bbox(bvh.nodes[idx].bounding_box_proxy()));
                actual_half_area[idx] = half_area(tmp_bbox);
            } else {
                tmp_bbox = bvh.nodes[idx].bounding_box_proxy();
                actual_half_area[idx] = half_area(tmp_bbox);
            }
            if (!bvh.nodes[idx].is_leaf()) {
                queue.emplace(bvh.nodes[idx].first_child_or_primitive, tmp_bbox);
                queue.emplace(bvh.nodes[idx].first_child_or_primitive + 1, tmp_bbox);
            }
        }
    }

    std::pair<int, bool> find_target_idx(int insert_idx) {
        double best_cost = std::numeric_limits<double>::infinity();
        int best_idx;
        bool best_is_lp;

        bvh::Bvh<float>::Node &insert_node = bvh.nodes[insert_idx];
        bvh::BoundingBox<float> insert_bbox = insert_node.bounding_box_proxy();

        std::stack<std::tuple<int, double, double>> stack;
        stack.emplace(bvh.nodes[0].first_child_or_primitive, std::numeric_limits<float>::infinity(), 0);
        stack.emplace(bvh.nodes[0].first_child_or_primitive + 1, std::numeric_limits<float>::infinity(), 0);
        while (!stack.empty()) {
            auto [curr_idx, c_lp, c_hp] = stack.top();
            stack.pop();

            bvh::Bvh<float>::Node &curr_node = bvh.nodes[curr_idx];
            bvh::Bvh<float>::Node &sibling_node = bvh.nodes[get_sibling_idx(curr_idx)];
            bvh::Bvh<float>::Node &parent_node = bvh.nodes[parent[curr_idx]];

            bvh::BoundingBox<float> parent_bbox = parent_node.bounding_box_proxy();
            double parent_lp_half_area = half_area(to_lp_bbox(parent_bbox));
            double parent_hp_half_area = half_area(parent_bbox);
            double c_parent_old = 0;
            if (parent_node.low_precision) {
                c_parent_old += (curr_node.low_precision ? c_t_l : c_t_h) * parent_lp_half_area;
                c_parent_old += (sibling_node.low_precision ? c_t_l : c_t_h) * parent_lp_half_area;
            } else {
                c_parent_old += (curr_node.low_precision ? c_t_l : c_t_h) * parent_hp_half_area;
                c_parent_old += (sibling_node.low_precision ? c_t_l : c_t_h) * parent_hp_half_area;
            }

            parent_bbox.extend(insert_bbox);
            parent_lp_half_area = half_area(to_lp_bbox(parent_bbox));
            parent_hp_half_area = half_area(parent_bbox);
            double c_lp_parent_is_lp = c_lp + c_t_l * parent_lp_half_area - c_parent_old;
            c_lp_parent_is_lp += (sibling_node.low_precision ? c_t_l : c_t_h) * parent_lp_half_area;
            double c_lp_parent_is_hp = c_hp + c_t_l * parent_hp_half_area - c_parent_old;
            c_lp_parent_is_hp += (sibling_node.low_precision ? c_t_l : c_t_h) * parent_hp_half_area;
            double c_hp_parent_is_lp = c_lp + c_t_h * parent_lp_half_area - c_parent_old;
            c_hp_parent_is_lp += (sibling_node.low_precision ? c_t_l : c_t_h) * parent_lp_half_area;
            double c_hp_parent_is_hp = c_hp + c_t_h * parent_hp_half_area - c_parent_old;
            c_hp_parent_is_hp += (sibling_node.low_precision ? c_t_l : c_t_h) * parent_hp_half_area;

            c_lp = std::min(c_lp_parent_is_lp, c_lp_parent_is_hp);
            c_hp = std::min(c_hp_parent_is_lp, c_hp_parent_is_hp);
            if (!curr_node.is_leaf()) {
                stack.emplace(curr_node.first_child_or_primitive, c_lp, c_hp);
                stack.emplace(curr_node.first_child_or_primitive + 1, c_lp, c_hp);
            }

            bvh::BoundingBox<float> direct_bbox = insert_node.bounding_box_proxy();
            direct_bbox.extend(curr_node.bounding_box_proxy());
            double direct_lp_half_area = half_area(to_lp_bbox(direct_bbox));
            double direct_hp_half_area = half_area(direct_bbox);
            double c_direct_lp = (insert_node.low_precision ? c_t_l : c_t_h) * direct_lp_half_area;
            c_direct_lp += (curr_node.low_precision ? c_t_l : c_t_h) * direct_lp_half_area;
            double c_direct_hp = (insert_node.low_precision ? c_t_l : c_t_h) * direct_hp_half_area;
            c_direct_hp += (curr_node.low_precision ? c_t_l : c_t_h) * direct_hp_half_area;

            double c_total_lp = c_lp + c_direct_lp;
            double c_total_hp = c_hp + c_direct_hp;

            if (c_total_lp < c_total_hp) {
                if (c_total_lp < best_cost) {
                    best_cost = c_total_lp;
                    best_idx = curr_idx;
                    best_is_lp = true;
                }
            } else {
                if (c_total_hp < best_cost) {
                    best_cost = c_total_hp;
                    best_idx = curr_idx;
                    best_is_lp = false;
                }
            }
        }

        return std::make_pair(best_idx, best_is_lp);
    }

    void update_parent (int idx) {
        if (!bvh.nodes[idx].is_leaf()) {
            parent[bvh.nodes[idx].first_child_or_primitive] = idx;
            parent[bvh.nodes[idx].first_child_or_primitive + 1] = idx;
        }
    }

    int get_sibling_idx(int curr_idx) {
        bvh::Bvh<float>::Node &parent_node = bvh.nodes[parent[curr_idx]];
        if (parent_node.first_child_or_primitive == curr_idx)
            return curr_idx + 1;
        else if (parent_node.first_child_or_primitive + 1 == curr_idx)
            return curr_idx - 1;
        else
            assert(false);
    }

    void propagate_bbox(int modified_idx) {
        bvh::Bvh<float>::Node &modified_node = bvh.nodes[modified_idx];
        std::stack<int> stack_1;
        for (int curr_idx = modified_idx; curr_idx != 0; curr_idx = parent[curr_idx])
            stack_1.push(curr_idx);

        double c_lp = std::numeric_limits<double>::infinity();
        double c_hp = 0;
        std::stack<std::tuple<int, bool, bool>> stack_2;
        while (!stack_1.empty()) {
            int curr_idx = stack_1.top();
            int sibling_idx = get_sibling_idx(curr_idx);
            bvh::Bvh<float>::Node &sibling_node = bvh.nodes[sibling_idx];
            bvh::Bvh<float>::Node &parent_node = bvh.nodes[parent[curr_idx]];
            stack_1.pop();

            double parent_lp_half_area = half_area(to_lp_bbox(parent_node.bounding_box_proxy()));
            double parent_hp_half_area = half_area(parent_node.bounding_box_proxy());
            double c_lp_parent_is_lp = c_lp + c_t_l * parent_lp_half_area;
            double c_lp_parent_is_hp = c_hp + c_t_l * parent_hp_half_area;
            double c_hp_parent_is_lp = c_lp + c_t_h * parent_lp_half_area;
            double c_hp_parent_is_hp = c_hp + c_t_h * parent_hp_half_area;
            if (sibling_node.low_precision) {
                c_lp_parent_is_lp += c_t_l * parent_lp_half_area;
                c_lp_parent_is_hp += c_t_l * parent_hp_half_area;
                c_hp_parent_is_lp += c_t_l * parent_lp_half_area;
                c_hp_parent_is_hp += c_t_l * parent_hp_half_area;
            } else {
                c_lp_parent_is_lp += c_t_h * parent_lp_half_area;
                c_lp_parent_is_hp += c_t_h * parent_hp_half_area;
                c_hp_parent_is_lp += c_t_h * parent_lp_half_area;
                c_hp_parent_is_hp += c_t_h * parent_hp_half_area;
            }

            bool lp_parent_is_lp;
            if (c_lp_parent_is_lp < c_lp_parent_is_hp) {
                lp_parent_is_lp = true;
                c_lp = c_lp_parent_is_lp;
            } else {
                lp_parent_is_lp = false;
                c_lp = c_lp_parent_is_hp;
            }

            bool hp_parent_is_lp;
            if (c_hp_parent_is_lp < c_hp_parent_is_hp) {
                hp_parent_is_lp = true;
                c_hp = c_hp_parent_is_lp;
            } else {
                hp_parent_is_lp = false;
                c_hp = c_hp_parent_is_hp;
            }

            stack_2.emplace(curr_idx, lp_parent_is_lp, hp_parent_is_lp);
        }

        bool lp = modified_node.low_precision;
        while (!stack_2.empty()) {
            auto [idx, lp_parent_is_lp, hp_parent_is_lp] = stack_2.top();
            stack_2.pop();
            bvh.nodes[idx].low_precision = lp;
            lp = lp ? lp_parent_is_lp : hp_parent_is_lp;
        }
    }

    void optimize() {
        for (int iter = 0; iter < 100; iter++) {
            // sort node by surface area
            fill_actual_half_area();
            std::vector<int> victim_indices(bvh.node_count);
            std::iota(victim_indices.begin(), victim_indices.end(), 0);
            std::sort(victim_indices.begin(), victim_indices.end(), [&](int x, int y) {
                return actual_half_area[x] > actual_half_area[y];
            });

            for (auto victim_idx : victim_indices) {
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
                propagate_bbox(victim_idx);

                // place left child
                auto [target_idx, is_lp] = find_target_idx(victim_left_idx);
                bvh.nodes[victim_right_idx] = bvh.nodes[target_idx];
                update_parent(victim_right_idx);
                bvh.nodes[target_idx].bounding_box_proxy() = bvh::BoundingBox<float>::empty();
                bvh.nodes[target_idx].bounding_box_proxy().extend(bvh.nodes[victim_left_idx].bounding_box_proxy());
                bvh.nodes[target_idx].bounding_box_proxy().extend(bvh.nodes[victim_right_idx].bounding_box_proxy());
                bvh.nodes[target_idx].primitive_count = 0;
                bvh.nodes[target_idx].first_child_or_primitive = victim_left_idx;
                bvh.nodes[target_idx].low_precision = is_lp;
                update_parent(target_idx);
                for (int curr = parent[target_idx]; curr != 0; curr = parent[curr])
                    update_bbox(bvh, curr);
                propagate_bbox(target_idx);

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
                propagate_bbox(victim_parent_idx);

                // determine where to place right child
                std::tie(target_idx, is_lp) = find_target_idx(victim_idx);
                bvh.nodes[victim_sibling_idx] = bvh.nodes[target_idx];
                update_parent(victim_sibling_idx);
                bvh.nodes[target_idx].bounding_box_proxy() = bvh::BoundingBox<float>::empty();
                bvh.nodes[target_idx].bounding_box_proxy().extend(bvh.nodes[victim_sibling_idx].bounding_box_proxy());
                bvh.nodes[target_idx].bounding_box_proxy().extend(bvh.nodes[victim_idx].bounding_box_proxy());
                bvh.nodes[target_idx].primitive_count = 0;
                bvh.nodes[target_idx].first_child_or_primitive = std::min(victim_idx, victim_sibling_idx);
                bvh.nodes[target_idx].low_precision = is_lp;
                update_parent(target_idx);
                for (int curr = parent[target_idx]; curr != 0; curr = parent[curr])
                    update_bbox(bvh, curr);
                propagate_bbox(target_idx);

                std::cout << sah_cost() << std::endl;
            }
        }
    }
};


#endif //BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP
