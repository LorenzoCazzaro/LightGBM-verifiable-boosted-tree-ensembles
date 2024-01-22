#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

#include <iostream>

using namespace std;

namespace vl {

typedef uint32_t feature_t;
typedef float float_t;
typedef int label_t;

typedef std::vector<float_t> instance_t;

namespace constants {

constexpr feature_t invalid_feature = feature_t(-1);
constexpr float_t invalid_threshold = float_t(-1);
constexpr float_t invalid_score = float_t(-1);

/* Note: throughout the code, we assume valid labels are 0 and 1. */
constexpr label_t invalid_label = label_t(-1);

constexpr float_t inf = std::numeric_limits<float_t>::max();

}  // namespace constants

struct hyper_rectangle_t {
    hyper_rectangle_t() : score(0.0), norm(constants::inf), empty(false) {}
    float_t score;
    float_t norm;  // ||dist(x,H)||_p
    bool empty;
    std::vector<std::pair<float_t, float_t>> H;

    void set_empty() {
        for (uint32_t i = 0; i < H.size(); i++) {
            if (H[i].first >= H[i].second) {
                empty = true;
                return;
            }
        }
    }
};

struct min_perturbation {
    double norm;
    std::vector<float_t> delta;
};

double norm(instance_t const& x, float p) {
    if (p == 0) {
        double ret = 0.0;
        for (auto x_i : x) ret += x_i != 0;
        return ret;
    } else if (p != constants::inf) {
        double ret = 0.0;
        for (auto x_i : x) ret += pow(abs(x_i), p);
        return pow(ret, 1.0 / p);
    } else {
        double ret = 0.0;
        for (auto x_i : x) ret = std::max<double>(abs(x_i), ret);
        return ret;
    }
}

}  // namespace vl