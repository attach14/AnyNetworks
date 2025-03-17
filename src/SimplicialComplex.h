#pragma once

#include <vector>

#include "AbstractModel.h"

class SimplicialComplex : public AbstractModel {
 public:
  void AddSimplex(std::vector<int> simplex);

  void RemoveSimplex(std::vector<int> simplex);

  std::vector<std::vector<int>> GetMaxSimplices();
};
