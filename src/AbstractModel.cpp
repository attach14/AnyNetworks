#include "AbstractModel.h"
#include "Hasse.h"

int AbstractModel::HasseSize() {
  return hasse_.Size();
}

std::vector<std::vector<int>> AbstractModel::GetMaxElements() {
  return hasse_.GetMaxElements();
}

std::vector<std::vector<int>> AbstractModel::GetAllElements() {
  return hasse_.GetAllElements();
}

std::vector<std::vector<int>> GetFixedDimElements(int dim) {
  return hasse_.GetFixedDimElements(dim);
}

std::vector<std::vector<int>> AbstractModel::Incidence(std::vector<int> node,
                                                       int k) {
  return hasse_.Incidence(node, k);
}

int AbstractModel::IncidenceDegree(std::vector<int> node, int k) {
  return hasse_.IncidenceDegree(node, k);
}

std::vector<std::vector<int>> AbstractModel::Adjacency(std::vector<int> node,
                                                       int k) {
  return hasse_.Adjacency(node, k);
}

double AbstractModel::Degree(std::vector<int> node, int k, bool weighted) {
  return hasse_.Degree(node, k, weighted);
}

std::vector<double> AbstractModel::DegreeAll(int p, int k, bool weighted) {
  return hasse_.DegreeAll(p, k, weighted);
}

int AbstractModel::BettiNumber(int k) {
  return hasse_.BettiNumber(k);
}

MyMatrixInt AbstractModel::BoundaryMatrix(int k, int p) {
  return hasse_.BoundaryMatrix(k, p);
}

double AbstractModel::Closeness(std::vector<int> node, int max_rank,
                                bool weighted) {
  return hasse_.Closeness(node, max_rank, weighted);
}

double AbstractModel::Betweenness(std::vector<int> node, int max_rank,
                                  bool weighted) {
  return hasse_.Betweenness(node, max_rank, weighted);
}

std::vector<std::pair<std::vector<int>, double>> AbstractModel::ClosenessAll(
    int p, int max_rank, bool weighted) {
  return hasse_.ClosenessAll(p, max_rank, weighted);
}

std::vector<std::pair<std::vector<int>, double>> AbstractModel::BetweennessAll(
    int p, int max_rank, bool weighted) {
  return hasse_.BetweennessAll(p, max_rank, weighted);
}

int AbstractModel::Dimension() {
  return hasse_.Dimension();
}

std::vector<std::pair<int, int>> AbstractModel::FVector() {
  return hasse_.FVector();
}

int AbstractModel::Size() {
  return hasse_.Size();
}

int AbstractModel::EulerCharacteristic() {
  return hasse_.EulerCharacteristic();
}

void AbstractModel::Clear() {
  hasse_ = Hasse();
}

DoubleMatrix AbstractModel::LaplacianMatrix(int k, int p, int q,
                                              bool weighted) {
  return hasse_.LaplacianMatrix(k, p, q, weighted);
}

std::pair<vec, mat> AbstractModel::LaplacianSpectre(int k, int p = 1, int q = 1, bool weighted = false) {
  return hasse_.LaplacianSpectre(k, p, q, weighted);
}

void AbstractModel::UpdateWeight(std::vector<int> node, double new_weight) {
  hasse_.UpdateWeight(node, new_weight);
}

std::vector<std::vector<int>> AbstractModel::GetFixedDimElements(int dim) {
  return hasse_.GetFixedDimElements(dim);
}
