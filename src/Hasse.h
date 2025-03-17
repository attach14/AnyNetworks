#pragma once

#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "HasseNode.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <armadillo>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> DoubleMatrix;

typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagMatrix;

typedef Eigen::SparseMatrix<int, Eigen::RowMajor> IntMatrix;

const int HasseMx = 30;
const int HasseInf = 100000000;

class Hasse {
 public:
  Hasse();

  Hasse(int object_type);

  ~Hasse();


  void InsertNode(const std::vector<int>& data, double weight = 1);

  void RemoveNode(const std::vector<int>& data);

  void UpdateWeight(std::vector<int> node, double new_weight);

  std::vector<std::vector<int>> GetMaxElements();

  std::vector<std::vector<int>> GetAllElements();

  std::vector<std::vector<int>> GetFixedDimElements(int dim);

  int Size();

  int Dimension();

  std::vector<std::vector<int>>& IncidenceMatrix(int p, int k);

  int BettiNumber(int k);
  IntMatrix BoundaryMatrix(int k, int p);
  DoubleMatrix LaplacianMatrix(int k, int p, int q, bool weighted);

  std::pair<vec, mat> LaplacianSpectre(int k, int p = 1, int q = 1, bool weighted = false);

  std::vector<std::pair<std::vector<int>, double>> ClosenessAll(
      int p, int max_rank, bool weighted = false);
  std::vector<std::pair<std::vector<int>, double>> BetweennessAll(
      int p, int max_rank, bool weighted = false);

  std::vector<std::pair<int, int>> FVector();

  int EulerCharacteristic();

  int Orientation(const std::vector<int>& vec1, const std::vector<int>& vec2);

 private:
  HasseNode* root;
	int f[HasseMx];
	std::vector<HasseNode*> nums;
	std::vector<int>translation;
  int object_type = -1;

  bool Hasse::FirstInSecond(const std::vector<int>& vec1, const std::vector<int>& vec2);

  bool Hasse::CompareWords(const std::vector<int>& vec1, const std::vector<int>& vec2);

  std::vector<int>& MakePretty(const std::vector<int>& word);

  HasseNode* NodeFinder(const std::vector<int>& word);

  HasseNode* AddSingleNode(int curNode, double weight);

  void SoftDeleteAllCofaces(HasseNode* v);

  void RecursiveMaxElements(HasseNode* v, std::vector<HasseNode*>& ans, std::unordered_set<HasseNode*>& checked);

  void RecursiveAllElements(HasseNode* v, std::vector<HasseNode*>& ans, std::unordered_set<HasseNode*>& checked);

  void RecursiveFixedDimElements(HasseNode* v, std::vector<HasseNode*>& ans, std::unordered_set<HasseNode*>& checked, int dim);

  void Rec(int n, int cur, int pos, int sum, std::vector<int>& res, int k);

  std::vector<int> Combinations(int n, int k);

  DoubleMatrix LaplacianMatrixWeight(int k, int p = 1, int q = 1);

  DiagMatrix GetDimensionWeights(int k);

  mat Reduce(mat A, int x = 0);
};
