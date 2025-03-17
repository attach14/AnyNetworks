#include "Hasse.h"
#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"
#include "HasseNode.h"
#include "Spectra/MatOp/SparseSymMatProd.h"
#include <armadillo>

#include <atomic>
#include <cassert>
#include <functional>
#include <queue>
#include <set>
#include <stdexcept>

#include <algorithm>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

Hasse::Hasse() {
    root = new HasseNode;
    for (int i = 0; i < HasseMx; i++) {
        f[i] = 0;
    }
}

Hasse::Hasse(int object_type) {
    root = new HasseNode;
    this->object_type = object_type;
    for (int i = 0; i < HasseMx; i++) {
        f[i] = 0;
    }
}

Hasse::~Hasse() {
    delete root;
}

std::vector<int>& Hasse::MakePretty(const std::vector<int>& word) {
    std::set<int> f;
    for (auto c : word) {
        f.insert(c);
    }
    std::vector<int> res;
    for (auto c : f) {
        res.push_back(c);
    }
}

bool Hasse::FirstInSecond(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    std::unordered_set<int> set(vec2.begin(), vec2.end());
    for (const int& element : vec1) {
        if (set.find(element) == set.end()) {
            return false;
        }
    }
    return true;
}

void Hasse::UpdateWeight(std::vector<int> node, double weight) {
    std::vector<int> word = MakePretty(data);
    HasseNode* res = NodeFinder(word);
    if (res == nullptr) {
        return;
    }
    res->weight = weight;
}

bool Hasse::CompareWords(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    return vec1 == vec2;
}

//tp == 0 for graphs
//tp == 1 for hypergraphs
//tp == 2 for simplicial complexes
//tp == 3 for combinatorial complexes
HasseNode* Hasse::NodeFinder(const std::vector<int>& word) {
    if (word.empty()) {
        return nullptr;
    }
    if (this->object_type < 2) {
        if ((word.size() > 2) && (this->object_type == 0)) {
            return nullptr;
        }
        if (word.size() == 1) {
            for (const auto c:root->cofaces) {
                if (c->word[0] == word[0]) {
                    return c;
                }
            }
            return nullptr;
        }
        for (const auto c:root->cofaces) {
            if (c->word[0] == word[0]) {
                for (const auto j:c->cofaces) {
                    if (CompareWords(j->word, word)) {
                        return j;
                    }
                }
                return nullptr;
            }
            return nullptr;
        }
    }
    if (this->object_type == 2) {
        HasseNode* now = root;
        bool ok;
        while (true) {
            if (now->word.size() == word.size()) {
                return now;
            }
            ok = false;
            for (const auto c : now->cofaces) {
                if (FirstInSecond(c->word, word)) {
                    now = c;
                    ok = true;
                    break;
                }
            }
            if(!ok) {
                return nullptr;
            }
        }
        return nullptr;
    }
    if (this->object_type == 3) {
        //in progress
    }
}

std::vector<std::vector<int>> findWords(const std::vector<int>& vec, int len) {
    std::vector<std::vector<int>> res;
    std::vector<int> cur;

    std::function<void(int)> recurs = [&](int start) {
        if (cur.size() == len) {
            res.push_back(cur);
            return;
        }

        for (int i = start; i < vec.size(); i++) {
            cur.push_back(vec[i]);
            recurs(i + 1);
            cur.pop_back();
        }
    };
    recurs(0);
    return res;
}

HasseNode* Hasse::AddSingleNode(int curNode, double weight) {
    HasseNode* check = NodeFinder({curNode});
    if (check != nullptr) {
        return check;
    }
    HasseNode* res = new HasseNode;
    res->weight = weight;
    res->word = {curNode};
    res->faces = {root};
    root->cofaces.push_back(res);
    f[0]++;
    return res;
}

void Hasse::InsertNode(const std::vector<int>& data, double weight = 1) {
    if (data.empty()) {
        return;
    }
    std::vector<int> word = MakePretty(data);
    if (NodeFinder(word) != nullptr) {
        return;
    }
    if (this->object_type < 2) {
        if ((word.size() > 2) && (this->object_type == 0)) {
            return;
        }
        if (word.size() == 1) {
            AddSingleNode(word[0], weight);
            return;
        }
        HasseNode* res = new HasseNode;
        res->word = word;
        res->weight = weight;
        f[res->word.size() - 1]++;
        for (int i = 0; i < word.size(); i++) {
            HasseNode* cur = AddSingleNode(word[i], 1);
            cur->cofaces.push_back(res);
            res->faces.push_back(cur);
        }
    }

    if (this->object_type == 2) {
        if (NodeFinder(word) != nullptr) {
            return;
        }
        if (word.size() == 1) {
            AddSingleNode(word[0], weight);
            return;
        }
        std::vector<HasseNode*> prevDim;
        std::vector<HasseNode*> curDim;
        std::vector<std::vector<int>> allWords;
        for (int i = 0; i < word.size(); i++) {
            prevDim.push_back(AddSingleNode(word[i], 1));
        }
        for (int y = 2; y <= word.size(); y++) {
            allWords = findWords(word, y);
            for (const auto& curWord : allWords) {
                HasseNode* check = NodeFinder(curWord);
                if (check != nullptr) {
                    curDim.push_back(check);
                    continue;
                }
                HasseNode* res = new HasseNode;
                res->word = curWord;
                f[res->word.size() - 1]++;
                res->weight = 1;
                if (y == word.size()) {
                    res->weight = weight;
                }
                for (const auto prev : prevDim) {
                    if (FirstInSecond(prev->word, curWord)) {
                        prev->cofaces.push_back(res);
                        res->faces.push_back(prev);
                    }
                }
                curDim.push_back(res);
            }
            prevDim = curDim;
            curDim.clear();
        }
    }
    if (this->object_type == 3) {
        //in progress
    }
}

void Hasse::SoftDeleteAllCofaces(HasseNode* v) {
    f[v->word.size() - 1]--;
    for (auto c:v->cofaces) {
        SoftDeleteAllCofaces(c);
    }
}

void Hasse::RemoveNode(const std::vector<int>& data) {
    std::vector<int> word = MakePretty(data);
    HasseNode* res = NodeFinder(word);
    if (res == nullptr) {
        return;
    }
    delete res;
}

void Hasse::RecursiveMaxElements(HasseNode* v, std::vector<HasseNode*>& ans, std::unordered_set<HasseNode*>& checked) {
    if (checked.count(v)) {
        return;
    }
    if (v->cofaces.empty()) {
        ans.push_back(v);
        return;
    }
    for (const auto c : v->cofaces) {
        RecursiveMaxElements(c, ans, checked);
    }
    checked.insert(v);
}

std::vector<std::vector<int>> Hasse::GetMaxElements() {
    std::vector<HasseNode*> ans;
    std::unordered_set<HasseNode*> checked;
    RecursiveMaxElements(root, ans, checked);
    std::vector<std::vector<int>> res;
    for (const auto c : ans) {
        if (c == root) {
            continue;
        }
        res.push_back(c->word);
    }
    return res;
}

void Hasse::RecursiveAllElements(HasseNode* v, std::vector<HasseNode*>& ans, std::unordered_set<HasseNode*>& checked) {
    if (checked.count(v)) {
        return;
    }
    ans.push_back(v);
    for (const auto c : v->cofaces) {
        RecursiveAllElements(c, ans, checked);
    }
}

std::vector<std::vector<int>> Hasse::GetAllElements() {
    std::vector<HasseNode*> ans;
    std::unordered_set<HasseNode*> checked;
    RecursiveAllElements(root, ans, checked);
    std::vector<std::vector<int>> res;
    for (const auto c : ans) {
        if (c == root) {
            continue;
        }
        res.push_back(c->word);
    }
    return res;
}

void Hasse::RecursiveFixedDimElements(HasseNode* v, std::vector<HasseNode*>& ans, std::unordered_set<HasseNode*>& checked, int dim) {
    if (checked.count(v)) {
        return;
    }
    if (v->word.size() == dim + 1) {
        ans.push_back(v);
        return;
    }
    if (v->word.size() > dim + 1) {
        return;
    }
    for (const auto c : v->cofaces) {
        RecursiveFixedDimElements(c, ans, checked, dim);
    }
    checked.insert(v);
}

std::vector<std::vector<int>> Hasse::GetFixedDimElements(int dim) {
    std::vector<HasseNode*> ans;
    std::unordered_set<HasseNode*> checked;
    RecursiveFixedDimElements(root, ans, checked, dim);
    std::vector<std::vector<int>> res;
    for (const auto c : ans) {
        if (c == root) {
            continue;
        }
        res.push_back(c->word);
    }
    return res;
}

int Hasse::Size() {
    int ans = 0;
    for (int i = 0; i < HasseMx; i++) {
        ans += f[i];
    }
    return ans;
}

int Hasse::Dimension() {
    int ans = -1;
    for (int i = 0; i < HasseMx; i++) {
        if (f[i] > 0) {
            ans = i;
        }
    }
    return ans;
}

int Hasse::EulerCharacteristic() {
    int ans = 0;
    for (int i = 0; i < HasseMx; i++) {
        if (f[i] == 0)
            break;
        if (i % 2) {
            ans -= f[i];
        }
        else {
            ans += f[i];
        }
    }
    return ans;
}

std::vector<std::pair<int, int>> Hasse::FVector() {
    std::vector<std::pair<int, int>> ans;
    for (int i = 0; i < HasseMx; i++) {
        if (f[i] > 0) {
            ans.push_back(std::make_pair(i, f[i]));
        }
    }
    return ans;
}
 
void Hasse::Rec(int n, int cur, int pos, int sum, std::vector<int>& res, int k) {
    int x = sum;
    x += (1 << cur);
    if (pos + 1 == k) {
        res.push_back(x);
    } else {
        for (int i = cur + 1; i < n; i++) {
            Rec(n, i, pos + 1, x, res, k);
        }
    }
    if (n - cur - 1 + pos < k) {
        return;
    }
    for (int i = cur + 1; i < n; i++) {
        Rec(n, i, pos, sum, res, k);
    }
}

std::vector<int> Hasse::Combinations(int n, int k) {
    std::vector<int> res;
    std::vector<int> ans;
    Rec(n, 0, 0, 0, res, k);
    std::set<int> f;
    for (auto c : res) {
        if (f.count(c)) {
            continue;
        }
        f.insert(c);
        ans.push_back(c);
    }
    return ans;
}


IntMatrix Hasse::BoundaryMatrix(int k = 1, int p = 1) {
    int orient = 1;
    IntMatrix bd(1, 1);
    if (k <= p) {
        throw std::runtime_error("k must be greater than p\n");
        return bd;
    }
    std::vector<HasseNode*> k_simp, prev_simp;
    std::unordered_set<HasseNode*> checked;
    RecursiveFixedDimElements(root, k_simp, checked, k);
    checked.clear();
    RecursiveFixedDimElements(root, prev_simp, checked, p);
    int k_len = k_simp.size();
    int prev_len = prev_simp.size();
    IntMatrix res(prev_len, k_len);
    res.setZero();
    std::unordered_map<HasseNode*, int> translation;
    for (size_t i = 0; i < k_simp.size(); i++) {
        auto up = k_simp[i];
        translation[up] = i;
    }

    for (size_t i = 0; i < prev_len; i++) {
        std::vector<HasseNode*> k_cofaces;
        checked.clear();
        RecursiveFixedDimElements(prev_simp[i], k_cofaces, checked, k);
        for (auto coface : k_cofaces) {
            auto pos = translation[coface];
            res.insert(i, pos) = Orientation(prev_simp[i]->word, k_simp[pos]->word);
        }
    }
    res.makeCompressed();
    return res;
}

int Hasse::Orientation(const std::vector<int>& vec1, const std::vector<int>& vec2) {
    int ans = 0;
    for (int i = 0, j = 0; i < vec2.size(); i++) {
        while (vec2[j] != vec1[i]) {
            j += 1;
        }
        ans += j - i;
    }
    if (ans % 2) {
        return -1;
    }
    return 1;
}

DoubleMatrix Hasse::LaplacianMatrix(int k, int p, int q, bool weighted) {
    if (weighted) {
        return LaplacianMatrixWeight(k, p, q);
    }
    IntMatrix B1 = BoundaryMatrix(k, p);
    IntMatrix B2 = BoundaryMatrix(q, k);
    DoubleMatrix res = B1.transpose() * B1 + B2 * B2.transpose();
    return res;
}

DiagMatrix Hasse::GetDimensionWeights(int k) {
    std::vector<HasseNode*> all;
    std::unordered_set<HasseNode*> checked;
    RecursiveFixedDimElements(root, all, checked, k);
    DiagMatrix res(all.size(), all.size());
    res.setZero();
    for (int i = 0; i < all.size(); i++) {
        res.diagonal()[i] = all[i]->weight;
    }
    return res;
}

DoubleMatrix Hasse::LaplacianMatrixWeight(int k, int p = 1, int q = 1) {
    DoubleMatrix B1 = BoundaryMatrix(k, p);
    DoubleMatrix B2 = BoundaryMatrix(q, k);
    DiagMatrix Wkp_inv = GetDimensionWeights(p).inverse();
    DiagMatrix Wk = GetDimensionWeights(k);
    DiagMatrix Wk_inv = Wk.inverse();
    DiagMatrix Wkq = GetDimensionWeights(q);
    DoubleMatrix res = B1.transpose() * Wkp_inv * B1 * Wk + Wk_inv * B2 * Wkq * B2.transpose();
    return res;
}

std::pair<vec, mat> Hasse::LaplacianSpectre(int k, int p = 1, int q = 1, bool weighted = false) {
    vec val;
    mat res;
    mat lap(1, 1, fill::ones);
    if (weighted) {
        lap = laplacianMatrixWeight(k, p, q);
    }
    else {
        lap = laplacianMatrix(k, p, q);
    }
    arma::eig_sym(val, res, lap);
    return { val, res };
}

int Hasse::BettiNumber(int pos) {
    int dim = 0;
    while (f[dim] != 0) {
        dim++;
    }
    dim --;
    vec res(dim + 1, fill::zeros);
    vector<mat> bounds;
    mat zer(1, 1, fill::zeros);
    bounds.push_back(zer);
    mat B;
    for (int k = 1; k < dim + 1; k++) {
        B = arma::abs(boundaryMatrix(k, k - 1));
        B = reduce(B);
        bounds.push_back(B);
    }
    for (int k = 0; k < dim + 1; k++) {
        int ker = 0, im = 0;
        if (k == 0) {
            ker = f[0];
        } else {
            B = bounds[k];
            for (int i = 0; i < B.n_cols; i++) {
                if (B.col(i).max() == 0) {
                    ker++;
                }
            }
        }
        if (k == dim) {
            im = 0;
        } else {
            B = bounds[k + 1];
            for (int i = 0; i < B.n_rows; i++) {
                if (B.row(i).max() == 0) {
                    continue;
                }
                im++;
            }
        }
        if (k == pos) {
            return ker - im;
        }
        res(k) = ker - im;
    }
    return res(pos);
}

mat Hasse::Reduce(mat A, int x = 0) {
    if (x >= A.n_rows || x >= A.n_cols) {
        return A;
    }
    for (int i = x; i < A.n_rows; i++) {
        for (int j = x; j < A.n_cols; j++) {
            if (A(i, j) == 1) {
                A.swap_rows(x, i);
                A.swap_cols(x, j);
                rowvec row_x = A.row(x);
                for (int y = x + 1; y < A.n_rows; y++) {
                    if (A(y, x) == 1) {
                        rowvec row_y = A.row(y);
                        for (int k = 0; k < row_y.size(); k++) {
                            row_y(k) += row_x(k);
                            if (row_y(k) == 2) {
                                row_y(k) = 0;
                            }
                        }
                        A.row(y) = row_y;
                    }
                }
                colvec col_x = A.col(x);
                for (int y = x + 1; y < A.n_cols; y++) {
                    if (A(x, y) == 1) {
                        colvec col_y = A.col(y);
                        for (int k = 0; k < col_y.size(); k++) {
                            col_y(k) += col_x(k);
                            if (col_y(k) == 2) {
                                col_y(k) = 0;
                            }
                        }
                        A.col(y) = col_y;
                    }
                }
                return Reduce(A, x + 1);
            }
        }
    }
    return A;
}

std::vector<std::vector<int>>& Hasse::IncidenceMatrix(int p, int k) {
    if (k < p) {
      throw std::runtime_error("k < size of node");
    }  
    std::vector<HasseNode*> k_simp, prev_simp;
    std::unordered_set<HasseNode*> checked;
    RecursiveFixedDimElements(root, k_simp, checked, k);
    checked.clear();
    RecursiveFixedDimElements(root, prev_simp, checked, p);
    std::unordered_map<HasseNode*, int> translation;
    for (size_t i = 0; i < k_simp.size(); i++) {
        auto up = k_simp[i];
        translation[up] = i;
    }
    std::vector<std::vector<int>> res(prev_simp.size(), std::vector<int>(k_simp.size()));
    for (size_t i = 0; i < prev_len; i++) {
        std::vector<HasseNode*> k_cofaces;
        checked.clear();
        RecursiveFixedDimElements(prev_simp[i], k_cofaces, checked, k);
        for (auto coface : k_cofaces) {
            auto pos = translation[coface];
            res[i][pos] = 1;
        }
    }
    return res;
  }



vector<pair<double, int>> dijkstra(int s, vector<vector<pair<int, double>>>& edges) {
    vector<pair<double, int>> dist(edges.size(), { HasseINF, 0 });
    dist[s] = { 0, 1 };
    set<std::pair<double, int>> r;
    r.insert(std::make_pair(dist[s].first, s));
    while (!r.empty()) {
        pair<double, int> v = *r.begin();
        for (auto c : edges[v.second] ) {
            if (dist[c.first].first == c.second + dist[v.second].first) {
                dist[c.first].second += dist[v.second].second;
            }
            if (dist[c.first].first > c.second + dist[v.second].first) {
                r.erase({ dist[c.first].first, c.first });
                dist[c.first].first = c.second + dist[v.second].first;
                dist[c.first].second = dist[v.second].second;
                r.insert({ dist[c.first].first, c.first });
            }
        }
        r.erase(v);
    }
    return dist;
}

double dijkstraSpecial(int s, int t, vector<vector<pair<int, double>>>& edges) {
    vector<pair<double, int>> dist(edges.size(), { HasseINF, 0 });
    dist[s] = { 0, 1 };
    set<std::pair<double, int>> r;
    r.insert(std::make_pair(dist[s].first, s));
    while (!r.empty()) {
        pair<double, int> v = *r.begin();
        if (v.second == t) {
            return v.first;
        }
        for (auto c : edges[v.second]) {
            if (dist[c.first].first == c.second + dist[v.second].first) {
                dist[c.first].second += dist[v.second].second;
            }
            if (dist[c.first].first > c.second + dist[v.second].first) {
                r.erase({ dist[c.first].first, c.first });
                dist[c.first].first = c.second + dist[v.second].first;
                dist[c.first].second = dist[v.second].second;
                r.insert({ dist[c.first].first, c.first });
            }
        }
        r.erase(v);
    }
    return HasseINF;
}


std::vector<std::pair<std::vector<int>, double>> ClosenessAll(
    int p, int q, bool weighted = false) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, finish;
    std::vector<std::vector<int>> graph = IncidenceMatrix(p, q);
    std::vector<double> cent(graph.size());
    double ans = graph.size();
    size_t n = graph.size();
    std::vector<int> rev(graph.size());
    for (int i = 0; i < translation.size(); i++) {
        if (translation[i] != -1) {
            rev[translation[i]] = i;
        }
    }
    #pragma omp parallel for 
     for (int i = 0; i < n; i++) {
        std::vector<std::pair<double, int>> dist = dijkstra(i, graph);
        double sum = 0;
        double cur = 0;
        for (auto c : dist) {
            if (c.first == HasseINF) {
                continue;
            }
            sum += c.first;
            cur++;
        }
        double res = (cur - 1) / sum;
        res *= ((cur - 1) / (ans - 1));
        cent[i] = res;
    }
    std::vector<std::pair<std::vector<int>, double>> res;
    for (int i = 0; i < graph.size(); i++) {
        int rl = rev[i];
        HasseNode* v = nums[rl];
        res.push_back(std::make_pair(v->word, cent[i]));
    }
    return res;
}

std::vector<std::pair<std::vector<int>, double>> BetweennessAll(
    int p, int q, bool weighted = false) {
    if (p < 0 || q < 0) {
        throw std::runtime_error("p and q can't be negative\n");
        return;
    }
    if (p == q) {
        throw std::runtime_error("p and q can't be equal\n");
        return;
    }
    if (f[p] == 0) {
        throw std::runtime_error("There are no p-simplices\n");
        return;
    }
    if (f[q] == 0) {
        throw std::runtime_error("There are no q-simplices\n");
        return;
    }
    std::vector<std::vector<int>> graph = IncidenceMatrix(p, q);
    std::vector<double> cent(graph.size());
    #pragma omp parallel for 
    for (int i = 0; i < graph.size(); i++) {
        int cur = i;
        std::vector<std::pair<double, int>> path = dijkstra(cur, graph);
        std::vector<std::pair<double, int>> dist;
        double ans = 0;
        for (int s = 0; s < graph.size() - 1; s++) {
            if (s == cur) {
                continue;
            }
            if (path[s].first == HasseINF) {
                continue;
            }
            dist = dijkstra(s, graph);
            for (int t = s + 1; t < graph.size(); t++) {
                if (t == cur) {
                    continue;
                }
                if (path[t].first == HasseINF || path[s].first + path[t].first > dist[t].first) {
                    continue;
                }
                double x = 0;
                double y = dist[t].second;
                if (path[s].first + path[t].first == dist[t].first) {
                    double x1 = path[s].second;
                    double x2 = path[t].second;
                    x += x1 * x2;
                }
                ans += x / y;
            }
        }
        cent[i] = ans / ((graph.size() - 1) * (graph.size() - 2) / 2);
    }
    std::vector<int> rev(graph.size());
    for (int i = 0; i < translation.size(); i++) {
        if (translation[i] != -1) {
            rev[translation[i]] = i;
        }
    }
    std::vector<std::pair<std::vector<int>, double>> res;
    for (int i = 0; i < graph.size(); i++) {
        int rl = rev[i];
        HasseNode* v = nums[rl];
        res.push_back(std::make_pair(v->word, cent[i]));
    }
    return res;
}