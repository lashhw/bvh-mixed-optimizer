#ifndef BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP
#define BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP

#include <mpfr.h>

struct LowPrecisionOptimizer {
    bvh::Bvh<float> &bvh;
    double c_t_l;
    double c_t_h;
    double c_i;
    double lp_discount;
    mpfr_t tmp;
    mpfr_exp_t exp_min;
    mpfr_exp_t exp_max;
    std::vector<double> actual_cost;
    std::vector<int> parent;

    LowPrecisionOptimizer(bvh::Bvh<float> &bvh,
                          double c_t_l,
                          double c_t_h,
                          double c_i,
                          mpfr_prec_t mantissa_width,
                          mpfr_exp_t exponent_width) : bvh(bvh),
                                                       c_t_l(c_t_l),
                                                       c_t_h(c_t_h),
                                                       c_i(c_i),
                                                       lp_discount(c_t_l / c_t_h) {
        mpfr_init2(tmp, mantissa_width + 1);
        exp_min = -(1 << (exponent_width - 1)) + 2;
        exp_max = (1 << (exponent_width - 1));
        actual_cost.resize(bvh.node_count);
        parent.resize(bvh.node_count);
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

    bvh::BoundingBox<float> to_bbox_low(const bvh::BoundingBox<float> &bbox) {
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
                    bvh::BoundingBox<float> child_bbox = to_bbox_low(child_node.bounding_box_proxy());
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
        double root_half_area = half_area(bvh.nodes[0].bounds);

        std::queue<std::pair<int, bvh::BoundingBox<float>>> queue;
        queue.emplace(0, bvh::BoundingBox<float>::full());
        while (!queue.empty()) {
            auto [curr, actual_parent_bbox] = queue.front();
            queue.pop();

            bvh::Bvh<float>::Node &node = bvh.nodes[curr];
            bvh::BoundingBox<float> actual_bbox;
            if (node.low_precision) {
                actual_bbox = actual_parent_bbox;
                actual_bbox.shrink(to_bbox_low(node.bounding_box_proxy()));
            } else
                actual_bbox = node.bounding_box_proxy();
            double actual_half_area = half_area(actual_bbox);

            if (node.is_leaf())
                cost += node.primitive_count * c_i * actual_half_area / root_half_area;
            else {
                int left_idx = node.first_child_or_primitive;
                for (int i = 0; i <= 1; i++) {  // 0 for left child, 1 for right child
                    if (bvh.nodes[left_idx + i].low_precision)
                        cost += c_t_l * actual_half_area / root_half_area;
                    else
                        cost += c_t_h * actual_half_area / root_half_area;
                    queue.emplace(left_idx + i, actual_bbox);
                }
            }
        }

        return cost;
    }

    void fill_actual_cost() {
        std::queue<std::pair<int, bvh::BoundingBox<float>>> queue;
        queue.emplace(0, bvh::BoundingBox<float>::full());
        while (!queue.empty()) {
            auto [idx, parent_bbox] = queue.front();
            queue.pop();
            bvh::BoundingBox<float> tmp_bbox;
            if (bvh.nodes[idx].low_precision) {
                tmp_bbox = parent_bbox;
                tmp_bbox.shrink(to_bbox_low(bvh.nodes[idx].bounding_box_proxy()));
                actual_cost[idx] = lp_discount * half_area(tmp_bbox);
            } else {
                tmp_bbox = bvh.nodes[idx].bounding_box_proxy();
                actual_cost[idx] = half_area(tmp_bbox);
            }
            if (!bvh.nodes[idx].is_leaf()) {
                queue.emplace(bvh.nodes[idx].first_child_or_primitive, tmp_bbox);
                queue.emplace(bvh.nodes[idx].first_child_or_primitive + 1, tmp_bbox);
            }
        }
    }

    std::pair<int, bool> find_target_idx(int insert_idx) {
        // calculate actual cost
        fill_actual_cost();

        bvh::BoundingBox<float> insert_bbox = bvh.nodes[insert_idx].bounding_box_proxy();
        int best_idx = -1;
        bool best_is_lp;
        double best_cost = std::numeric_limits<double>::max();

        std::stack<std::tuple<int, int, double, double, bvh::BoundingBox<float>>> stack;
        bvh::BoundingBox<float> root_bbox = bvh.nodes[0].bounding_box_proxy();
        stack.emplace(bvh.nodes[0].first_child_or_primitive, 0, 0, 0, root_bbox);
        stack.emplace(bvh.nodes[0].first_child_or_primitive + 1, 0, 0, 0, root_bbox);
        while (!stack.empty()) {
            auto [idx, parent_idx, parent_lp_induced_cost, parent_hp_induced_cost, parent_lp_bbox] = stack.top();
            stack.pop();
            // calculate lp_cost
            //   - parent is lp
            bvh::BoundingBox<float> lp_lp_bbox = bvh.nodes[idx].bounding_box_proxy();
            lp_lp_bbox.extend(insert_bbox);
            lp_lp_bbox = to_bbox_low(lp_lp_bbox);
            lp_lp_bbox.shrink(parent_lp_bbox);
            double lp_lp_cost = parent_lp_induced_cost + lp_discount * half_area(lp_lp_bbox);
            //   - parent is hp
            bvh::BoundingBox<float> lp_hp_bbox = lp_lp_bbox;
            lp_hp_bbox.shrink(bvh.nodes[parent_idx].bounding_box_proxy());
            double lp_hp_cost = parent_hp_induced_cost + lp_discount * half_area(lp_hp_bbox);
            //   - compare
            double lp_cost;
            bvh::BoundingBox<float> lp_bbox;
            if (lp_lp_cost < lp_hp_cost) {
                lp_cost = lp_lp_cost;
                lp_bbox = lp_lp_bbox;
            } else {
                lp_cost = lp_hp_cost;
                lp_bbox = lp_hp_bbox;
            }
            // calculate hp_cost
            bvh::BoundingBox<float> hp_bbox = bvh.nodes[idx].bounding_box_proxy();
            hp_bbox.extend(insert_bbox);
            double hp_cost;
            if (parent_lp_induced_cost < parent_hp_induced_cost)
                hp_cost = parent_lp_induced_cost + half_area(hp_bbox);
            else
                hp_cost = parent_hp_induced_cost + half_area(hp_bbox);
            // choose between lp_cost and hp_cost
            if (lp_cost < hp_cost) {
                if (lp_cost < best_cost) {
                    best_idx = idx;
                    best_is_lp = true;
                    best_cost = lp_cost;
                }
            } else {
                if (hp_cost < best_cost) {
                    best_idx = idx;
                    best_is_lp = false;
                    best_cost = hp_cost;
                }
            }
            double lp_induced_cost = lp_cost - actual_cost[idx];
            double hp_induced_cost = hp_cost - actual_cost[idx];
            if (!bvh.nodes[idx].is_leaf()) {
                int left_idx = bvh.nodes[idx].first_child_or_primitive;
                int right_idx = left_idx + 1;
                stack.emplace(left_idx, idx, lp_induced_cost, hp_induced_cost, lp_bbox);
                stack.emplace(right_idx, idx, lp_induced_cost, hp_induced_cost, lp_bbox);
            }
        }
        return std::make_pair(best_idx, best_is_lp);
    }

    void propagate_bbox(int modified_idx) {
        std::stack<int> stack;
        for (int curr = parent[modified_idx]; curr != 0; curr = parent[curr]) {
            update_bbox(bvh, curr);
            stack.push(curr);
        }
        bvh::BoundingBox<float> parent_bbox = bvh.nodes[0].bounding_box_proxy();
        while (!stack.empty()) {
            int curr = stack.top();
            stack.pop();
            bvh::BoundingBox<float> lp_bbox = to_bbox_low(bvh.nodes[curr].bounding_box_proxy());
            lp_bbox.shrink(parent_bbox);
            double lp_cost = lp_discount * half_area(lp_bbox);
            double hp_cost = half_area(bvh.nodes[curr].bounds);
            if (lp_cost < hp_cost) {
                bvh.nodes[curr].low_precision = true;
                parent_bbox = lp_bbox;
            } else {
                bvh.nodes[curr].low_precision = false;
                parent_bbox = bvh.nodes[curr].bounding_box_proxy();
            }
        }
    }

    void optimize() {
        // calculate parent index
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
            // sort node by cost
            fill_actual_cost();
            std::vector<int> victim_indices(bvh.node_count);
            std::iota(victim_indices.begin(), victim_indices.end(), 0);
            std::sort(victim_indices.begin(), victim_indices.end(), [&](int x, int y) {
                return actual_cost[x] > actual_cost[y];
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
                update_parent(target_idx);
                propagate_bbox(target_idx);

                std::cout << sah_cost() << std::endl;
            }
        }
    }
};


#endif //BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP
