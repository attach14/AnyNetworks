#include "SimplicialComplex.h"
#include <algorithm>
#include <thread>
#include <stdexcept>

void SimplicialComplex::AddSimplex(std::vector<int> simplex) {
  hasse_.InsertNode(simplex);
}

void SimplicialComplex::RemoveSimplex(std::vector<int> simplex) {
  hasse_.RemoveNode(simplex);
}

std::vector<std::vector<int>> SimplicialComplex::GetMaxSimplices() {
  return this->GetMaxElements();
}
