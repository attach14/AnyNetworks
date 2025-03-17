#pragma once

#include "Hasse.h"
#include <vector>
#include <string>
#include <functional>

class AbstractModel {
 public:
 std::vector<std::vector<int>> GetMaxElements();
 std::vector<std::vector<int>> GetAllElements();
 std::vector<std::vector<int>> GetFixedDimElements(int dim);

  std::vector<std::vector<int>> Incidence(std::vector<int> node, int k);
  int IncidenceDegree(std::vector<int> node, int k);

  std::vector<std::vector<int>> Adjacency(std::vector<int> node, int k);
  double Degree(std::vector<int> node, int k, bool weighted = false);
  std::vector<double> DegreeAll(int p, int k, bool weighted = false);

  int BettiNumber(int k);
  IntMatrix BoundaryMatrix(int k, int p);
  DoubleMatrix LaplacianMatrix(int k, int p, int q, bool weighted);

  std::pair<vec, mat> LaplacianSpectre(int k, int p = 1, int q = 1, bool weighted = false);

  double Closeness(std::vector<int> node, int max_rank, bool weighted = false);
  double Betweenness(std::vector<int> node, int max_rank,
                     bool weighted = false);

  std::vector<std::pair<std::vector<int>, double>> ClosenessAll(
      int p, int max_rank, bool weighted = false);
  std::vector<std::pair<std::vector<int>, double>> BetweennessAll(
      int p, int max_rank, bool weighted = false);

  int Size();

  int Dimension();

  std::vector<std::pair<int, int>> FVector();

  int EulerCharacteristic();

  void Clear();

 protected:
  Hasse hasse_;
};
