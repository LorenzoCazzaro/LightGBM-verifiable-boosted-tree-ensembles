#pragma once

#include "tree.hpp"

#include <algorithm>
#include <map>
#include <list>

namespace vl {

struct tree_ensemble {
    tree_ensemble() {}

    void parse(std::ifstream& in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        is >> line;
        assert(line == "classifier-forest");
        uint32_t num_trees;
        is >> num_trees;
        std::cout << "num_trees = " << num_trees << std::endl;
        m_trees.resize(num_trees);
        for (uint32_t i = 0; i != num_trees; ++i) {
            // std::cout << "parsing " << i << "-th tree" << std::endl;
            m_trees[i].parse(in);
        }
    }

    float_t raw_prediction_score(instance_t const& x) const {
        float_t rps = 0.0;
        for (auto const& t : m_trees) rps += t.raw_prediction_score(x);
        return rps;
    }

    template <typename InverseLinkFunction>
    std::pair<label_t, float_t> predict(instance_t const& x, InverseLinkFunction const& i,
                                        const float_t tau) const {
        const float_t rps = raw_prediction_score(x);
        if (i(rps) >= tau) return {1, rps};
        return {0, rps};
    }

    template <typename InverseLinkFunction>
    bool stable(instance_t const& x, const label_t y, const float_t rps,  //
                InverseLinkFunction const& i, const float_t tau,          //
                const float p, const float k                              //
    ) {
        assert(y == 0 or y == 1);
        const float_t max_gain = solve_opt_problem(x, y, p, k);
        if ((y == 0 and i(rps + max_gain) < tau) or  //
            (y == 1 and i(rps - max_gain) >= tau)) {
            return true;
        }
        return false;
    }

    template <typename InverseLinkFunction>
    bool robust(instance_t const& x, const label_t y,             //
                InverseLinkFunction const& i, const float_t tau,  //
                const float p, const float k                      //
    ) {
        assert(y == 0 or y == 1);
        auto [predicted_label, rps] = predict(x, i, tau);
        if (predicted_label == y) {
            const float_t max_gain = solve_opt_problem(x, y, p, k);
            if ((y == 0 and i(rps + max_gain) < tau) or  //
                (y == 1 and i(rps - max_gain) >= tau)) {
                return true;
            }
        }
        return false;
    }

    void annotate() {
        for (auto& t : m_trees) t.annotate();
    }

    uint32_t num_trees() const { return m_trees.size(); }

    uint32_t num_features() const {
        assert(!m_trees.empty());
        return m_trees.front().num_features();
    }

    void print(std::ostream& out) const {
        out << "Print tree ensemble, contains " << m_trees.size() << " trees" << endl;
        for (auto const& t : m_trees) t.print(out);
    }

    float_t solve_opt_problem(instance_t const& x, const label_t y,  //
                              const float p, const float k)          //
    {
        if (p == constants::inf) {
            float_t max_gain = 0.0;
            for (auto& t : m_trees) max_gain += t.reachable(x, y, p, k);
            return max_gain;
        } else if (p == 0) {
            for (auto& t : m_trees) t.compute_norms(x, p);
            const uint32_t m = num_trees();
            std::vector<std::vector<float_t>> M(m + 1, std::vector<float_t>(uint64_t(k) + 1, 0.0));
            for (uint64_t i = 1; i <= m; ++i) {
                auto const& t = m_trees[i - 1];
                const float_t rps = t.raw_prediction_score(x);
                for (uint64_t q = 0; q <= uint64_t(k); ++q) {
                    if (t.min_norm() > q) {  // all norms > q
                        M[i][q] = M[i - 1][q];
                    } else {
                        float_t max_q = -constants::inf;
                        for (uint64_t j = 0; j != t.num_leaves(); ++j) {
                            auto hr = t.hyper_rectangle(j);
                            if (!hr.empty) {
                                if (uint64_t(hr.norm) <= q) {
                                    float_t gain = 0;
                                    if (y == 0) gain = hr.score - rps;
                                    if (y == 1) gain = rps - hr.score;
                                    if (M[i - 1][q - uint64_t(hr.norm)] + gain > max_q) {
                                        max_q = M[i - 1][q - uint64_t(hr.norm)] + gain;
                                    }
                                } else {
                                    break;  // since hyper-rectangles are sorted by norm
                                }
                            }
                        }
                        M[i][q] = std::max(M[i - 1][q], max_q);
                    }
                }
            }

            // for (auto const& v : M) {
            //     for (auto x : v) std::cout << x << " ";
            //     std::cout << std::endl;
            // }

            return M[m - 1][k];
        } else {
            throw std::runtime_error("Unsupported norm yet");
        }
    }

private:
    std::vector<tree> m_trees;
};

}  // namespace vl
