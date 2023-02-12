#ifndef BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP
#define BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP

#include <mpfr.h>

struct LowPrecisionOptimizer {
    bvh::Bvh<float> &bvh;
    double c_t_l;
    double c_t_h;
    double c_i;
    mpfr_t tmp;
    mpfr_exp_t exp_min;
    mpfr_exp_t exp_max;
    std::vector<double> actual_half_area_arr;
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
        actual_half_area_arr.resize(bvh.node_count);
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
                actual_bbox.shrink(to_lp_bbox(node.bounding_box_proxy()));
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

    void fill_actual_half_area_arr() {
        std::queue<std::pair<int, bvh::BoundingBox<float>>> queue;
        queue.emplace(0, bvh::BoundingBox<float>::full());
        while (!queue.empty()) {
            auto [idx, parent_bbox] = queue.front();
            queue.pop();
            bvh::BoundingBox<float> tmp_bbox;
            if (bvh.nodes[idx].low_precision) {
                tmp_bbox = parent_bbox;
                tmp_bbox.shrink(to_lp_bbox(bvh.nodes[idx].bounding_box_proxy()));
                actual_half_area_arr[idx] = half_area(tmp_bbox);
            } else {
                tmp_bbox = bvh.nodes[idx].bounding_box_proxy();
                actual_half_area_arr[idx] = half_area(tmp_bbox);
            }
            if (!bvh.nodes[idx].is_leaf()) {
                queue.emplace(bvh.nodes[idx].first_child_or_primitive, tmp_bbox);
                queue.emplace(bvh.nodes[idx].first_child_or_primitive + 1, tmp_bbox);
            }
        }
    }

    std::pair<int, bool> find_target_idx(int insert_idx) {
        bvh::BoundingBox<float> insert_bbox = bvh.nodes[insert_idx].bounding_box_proxy();
        int best_idx = -1;
        bool best_is_lp;
        double best_cost = std::numeric_limits<double>::max();

        std::stack<int> stack;
        stack.push(bvh.nodes[0].first_child_or_primitive);
        stack.push(bvh.nodes[0].first_child_or_primitive + 1);
        while (!stack.empty()) {
            int curr_idx = stack.top();
            stack.pop();

            int parent_idx = parent[curr_idx];
            int sibling_idx;
            if (bvh.nodes[parent_idx].first_child_or_primitive == curr_idx)
                sibling_idx = curr_idx + 1;
            else if (bvh.nodes[parent_idx].first_child_or_primitive + 1 == curr_idx)
                sibling_idx = curr_idx - 1;
            else
                assert(false);
            bvh::Bvh<float>::Node &sibling_node = bvh.nodes[sibling_idx];

            bvh::Bvh<float>::Node &curr_node = bvh.nodes[curr_idx];
            bvh::BoundingBox<float> direct_bbox = curr_node.bounding_box_proxy();
            direct_bbox.extend(sibling_node.bounding_box_proxy());
            double direct_lp_half_area = half_area(to_lp_bbox(direct_bbox));
            double direct_hp_half_area = half_area(direct_bbox);
            bvh::Bvh<float>::Node &curr_left_node = bvh.nodes[curr_node.first_child_or_primitive];
            bvh::Bvh<float>::Node &curr_right_node = bvh.nodes[curr_node.first_child_or_primitive + 1];

            double c_direct_lp = 0;
            c_direct_lp += (curr_left_node.low_precision ? c_t_l : c_t_h) * direct_lp_half_area;
            c_direct_lp += (curr_right_node.low_precision ? c_t_l : c_t_h) * direct_lp_half_area;

            double c_direct_hp = 0;
            c_direct_hp += (curr_left_node.low_precision ? c_t_l : c_t_h) * direct_hp_half_area;
            c_direct_hp += (curr_right_node.low_precision ? c_t_l : c_t_h) * direct_hp_half_area;

            double c_induced = 0;
            for (int induced_idx = parent[curr_idx]; induced_idx != -1; induced_idx = parent[induced_idx]) {
                bvh::Bvh<float>::Node &induced_node = bvh.nodes[induced_idx];
                bvh::Bvh<float>::Node &induced_left_node = bvh.nodes[induced_node.first_child_or_primitive];
                bvh::Bvh<float>::Node &induced_right_node = bvh.nodes[induced_node.first_child_or_primitive + 1];
                double induced_lp_half_area = half_area(to_lp_bbox(induced_node.bounding_box_proxy()));
                double induced_hp_half_area = half_area(induced_node.bounds);
                if (induced_node.low_precision) {
                    c_induced -= (induced_left_node.low_precision ? c_t_l : c_t_h) * induced_lp_half_area;
                    c_induced -= (induced_right_node.low_precision ? c_t_l : c_t_h) * induced_lp_half_area;
                } else {
                    c_induced -= (induced_left_node.low_precision ? c_t_l : c_t_h) * induced_hp_half_area;
                    c_induced -= (induced_right_node.low_precision ? c_t_l : c_t_h) * induced_hp_half_area;
                }
            }

            if (!curr_node.is_leaf()) {
                stack.push(curr_node.first_child_or_primitive);
                stack.push(curr_node.first_child_or_primitive + 1);
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

    void propagate_bbox(int modified_idx) {
        bvh::Bvh<float>::Node &modified_node = bvh.nodes[modified_idx];
        double c_lp = half_area(to_lp_bbox(modified_node.bounding_box_proxy()));
        double c_hp = half_area(modified_node.bounds);
        if (modified_node.is_leaf()) {
            c_lp *= c_i * modified_node.primitive_count;
            c_hp *= c_i * modified_node.primitive_count;
        } else {
            bvh::Bvh<float>::Node &modified_left_node = bvh.nodes[modified_node.first_child_or_primitive];
            bvh::Bvh<float>::Node &modified_right_node = bvh.nodes[modified_node.first_child_or_primitive + 1];
            double c_t = 0;
            c_t += modified_left_node.low_precision ? c_t_l : c_t_h;
            c_t += modified_right_node.low_precision ? c_t_l : c_t_h;
            c_lp *= c_t;
            c_hp *= c_t;
        }

        std::stack<std::tuple<int, bool, bool>> stack;
        for (int curr = modified_idx; curr != 0; curr = parent[curr]) {
            update_bbox(bvh, parent[curr]);
            bvh::Bvh<float>::Node &parent_node = bvh.nodes[parent[curr]];
            double c_sibling;
            if (parent_node.first_child_or_primitive == curr)
                c_sibling = bvh.nodes[curr + 1].low_precision ? c_t_l : c_t_h;
            else if (parent_node.first_child_or_primitive + 1 == curr)
                c_sibling = bvh.nodes[curr - 1].low_precision ? c_t_l : c_t_h;
            else
                assert(false);
            double parent_lp_half_area = half_area(to_lp_bbox(parent_node.bounding_box_proxy()));
            double c_parent_lp_curr_lp = (c_sibling + c_t_l) * parent_lp_half_area + c_lp;
            double c_parent_lp_curr_hp = (c_sibling + c_t_h) * parent_lp_half_area + c_hp;
            double parent_hp_half_area = half_area(parent_node.bounds);
            double c_parent_hp_curr_lp = (c_sibling + c_t_l) * parent_hp_half_area + c_lp;
            double c_parent_hp_curr_hp = (c_sibling + c_t_h) * parent_hp_half_area + c_hp;

            bool parent_lp_curr_lp;
            if (c_parent_lp_curr_lp < c_parent_lp_curr_hp) {
                c_lp = c_parent_lp_curr_lp;
                parent_lp_curr_lp = true;
            } else {
                c_lp = c_parent_lp_curr_hp;
                parent_lp_curr_lp = false;
            }

            bool parent_hp_curr_lp;
            if (c_parent_hp_curr_lp < c_parent_hp_curr_hp) {
                c_hp = c_parent_hp_curr_lp;
                parent_hp_curr_lp = true;
            } else {
                c_hp = c_parent_hp_curr_hp;
                parent_hp_curr_lp = false;
            }

            stack.emplace(curr, parent_lp_curr_lp, parent_hp_curr_lp);
        }

        bool parent_is_lp = false;
        while (!stack.empty()) {
            auto [curr, parent_lp_curr_lp, parent_hp_curr_lp] = stack.top();
            stack.pop();
            if (parent_is_lp) {
                bvh.nodes[curr].low_precision = parent_lp_curr_lp;
                parent_is_lp = parent_lp_curr_lp;
            } else {
                bvh.nodes[curr].low_precision = parent_hp_curr_lp;
                parent_is_lp = parent_hp_curr_lp;
            }
        }
    }

    void optimize() {
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

        for (int iter = 0; iter < 100; iter++) {
            // sort node by surface area
            fill_actual_half_area_arr();
            std::vector<int> victim_indices(bvh.node_count);
            std::iota(victim_indices.begin(), victim_indices.end(), 0);
            std::sort(victim_indices.begin(), victim_indices.end(), [&](int x, int y) {
                return actual_half_area_arr[x] > actual_half_area_arr[y];
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
                bvh.nodes[target_idx].low_precision = is_lp;
                update_parent(target_idx);
                propagate_bbox(target_idx);

                std::cout << sah_cost() << std::endl;
            }
        }
    }
};


#endif //BVH_MIXED_OPTIMIZER_LOW_PRECISION_OPTIMIZER_HPP
