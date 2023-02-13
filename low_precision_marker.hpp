#ifndef BVH_MIXED_OPTIMIZER_LOW_PRECISION_MARKER_HPP
#define BVH_MIXED_OPTIMIZER_LOW_PRECISION_MARKER_HPP

#include <mpfr.h>

struct LowPrecisionMarker {
    bvh::Bvh<float> &bvh;
    double c_t_l;
    double c_t_h;
    double c_i;
    mpfr_t tmp;
    mpfr_exp_t exp_min;
    mpfr_exp_t exp_max;

    LowPrecisionMarker(bvh::Bvh<float> &bvh,
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
    }

    ~LowPrecisionMarker() {
        mpfr_clear(tmp);
    }

    void check_exponent_and_set_inf(mpfr_t &num) const {
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

    void mark() {
        std::vector<double> c_high(bvh.node_count);
        std::vector<double> c_low(bvh.node_count);
        auto set_leaf_c = [&](size_t node_idx) {
            c_high[node_idx] = c_i * bvh.nodes[node_idx].primitive_count;
            c_low[node_idx] = c_i * bvh.nodes[node_idx].primitive_count;
        };

        std::vector<std::pair<bool, bool>> m_high(bvh.node_count);
        std::vector<std::pair<bool, bool>> m_low(bvh.node_count);

        std::stack<std::pair<size_t, bool>> stk;
        stk.emplace(0, true);
        while (!stk.empty()) {
            auto [curr_idx, first] = stk.top();
            stk.pop();

            size_t left_child_idx = bvh.nodes[curr_idx].first_child_or_primitive;
            size_t right_child_idx = left_child_idx + 1;

            if (first) {
                if (!bvh.nodes[curr_idx].is_leaf()) {
                    stk.emplace(curr_idx, false);
                    stk.emplace(left_child_idx, true);
                    stk.emplace(right_child_idx, true);
                } else set_leaf_c(curr_idx);
            } else {
                bvh::BoundingBox<float> curr_bbox_high = bvh.nodes[curr_idx].bounding_box_proxy();
                bvh::BoundingBox<float> curr_bbox_low = to_bbox_low(curr_bbox_high);
                bvh::BoundingBox<float> left_bbox_high = bvh.nodes[left_child_idx].bounding_box_proxy();
                bvh::BoundingBox<float> left_bbox_low = to_bbox_low(left_bbox_high);
                bvh::BoundingBox<float> right_bbox_high = bvh.nodes[right_child_idx].bounding_box_proxy();
                bvh::BoundingBox<float> right_bbox_low = to_bbox_low(right_bbox_high);
                double p_left_high_curr_low = half_area(left_bbox_high) / half_area(curr_bbox_low);
                double p_right_high_curr_low = half_area(right_bbox_high) / half_area(curr_bbox_low);
                double p_left_low_curr_low = half_area(left_bbox_low) / half_area(curr_bbox_low);
                double p_right_low_curr_low = half_area(right_bbox_low) / half_area(curr_bbox_low);
                double p_left_high_curr_high = half_area(left_bbox_high) / half_area(curr_bbox_high);
                double p_right_high_curr_high = half_area(right_bbox_high) / half_area(curr_bbox_high);
                // This case should be handled correctly, but it is quite difficult :(
                // left_bbox_low.shrink(curr_bbox_high);
                // right_bbox_low.shrink(curr_bbox_high);
                double p_left_low_curr_high = half_area(left_bbox_low) / half_area(curr_bbox_high);
                double p_right_low_curr_high = half_area(right_bbox_low) / half_area(curr_bbox_high);

                double c_left_high_curr_high = c_t_h + p_left_high_curr_high * c_high[left_child_idx];
                double c_left_low_curr_high = c_t_l + p_left_low_curr_high * c_low[left_child_idx];
                double c_right_high_curr_high = c_t_h + p_right_high_curr_high * c_high[right_child_idx];
                double c_right_low_curr_high = c_t_l + p_right_low_curr_high * c_low[right_child_idx];
                double c_left_high_curr_low = c_t_h + p_left_high_curr_low * c_high[left_child_idx];
                double c_left_low_curr_low = c_t_l + p_left_low_curr_low * c_low[left_child_idx];
                double c_right_high_curr_low = c_t_h + p_right_high_curr_low * c_high[right_child_idx];
                double c_right_low_curr_low = c_t_l + p_right_low_curr_low * c_low[right_child_idx];

                if (c_left_high_curr_high < c_left_low_curr_high) {
                    m_high[curr_idx].first = true;
                    c_high[curr_idx] = c_left_high_curr_high;
                } else {
                    m_high[curr_idx].first = false;
                    c_high[curr_idx] = c_left_low_curr_high;
                }

                if (c_right_high_curr_high < c_right_low_curr_high) {
                    m_high[curr_idx].second = true;
                    c_high[curr_idx] += c_right_high_curr_high;
                } else {
                    m_high[curr_idx].second = false;
                    c_high[curr_idx] += c_right_low_curr_high;
                }

                if (c_left_high_curr_low < c_left_low_curr_low) {
                    m_low[curr_idx].first = true;
                    c_low[curr_idx] = c_left_high_curr_low;
                } else {
                    m_low[curr_idx].first = false;
                    c_low[curr_idx] = c_left_low_curr_low;
                }

                if (c_right_high_curr_low < c_right_low_curr_low) {
                    m_low[curr_idx].second = true;
                    c_low[curr_idx] += c_right_high_curr_low;
                } else {
                    m_low[curr_idx].second = false;
                    c_low[curr_idx] += c_right_low_curr_low;
                }
            }
        }

        if (c_high[0] < c_low[0]) stk.emplace(0, true);
        else stk.emplace(0, false);
        while (!stk.empty()) {
            auto [curr_idx, high] = stk.top();
            stk.pop();

            bvh.nodes[curr_idx].low_precision = !high;

            if (bvh.nodes[curr_idx].is_leaf()) continue;

            size_t left_child_idx = bvh.nodes[curr_idx].first_child_or_primitive;
            size_t right_child_idx = left_child_idx + 1;

            if (high) {
                stk.emplace(left_child_idx, m_high[curr_idx].first);
                stk.emplace(right_child_idx, m_high[curr_idx].second);
            } else {
                stk.emplace(left_child_idx, m_low[curr_idx].first);
                stk.emplace(right_child_idx, m_low[curr_idx].second);
            }
        }
    }
};

#endif //BVH_MIXED_OPTIMIZER_LOW_PRECISION_MARKER_HPP
