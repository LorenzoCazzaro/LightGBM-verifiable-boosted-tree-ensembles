#pragma once

namespace vl {

struct identity {
    float_t operator()(const float_t score) const { return score; }
};

typedef identity inverse_link_function_type;

}  // namespace vl