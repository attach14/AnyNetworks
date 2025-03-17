#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"
#include "pybind11/complex.h"
#include <pybind11/eigen.h>

#include "src/CombinatorialComplex.h"
#include "src/Graph.h"
#include "src/HyperGraph.h"
#include "src/SimplicialComplex.h"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(anynetworks, m) {
  m.doc() =
      "AnyNetworks library for TDA";

  py::class_<SimplicialComplex>(m, "SimplicialComplex")
      .def(py::init<>(), "Creating a new simplicial complex")

      .def("AddSimplex", &SimplicialComplex::AddSimplex, "simplex"_a,
           "Add a simplex to the simplicial complex")
      .def("RemoveSimplex", &SimplicialComplex::RemoveSimplex, "simplex"_a,
           "Remove a simplex from the simplicial complex")
      .def("UpdateWeight", &SimplicialComplex::UpdateWeight, "node"_a,
           "weight"_a, "Update the weight of a simplex")

      .def("BoundaryMatrix", &SimplicialComplex::BoundaryMatrix, "k"_a, "p"_a,
           "Get the boundary matrix. Should be: k > p")

      .def("LaplacianMatrix", &SimplicialComplex::LaplacianMatrix, "k"_a, "p"_a,
           "q"_a, "weighted"_a = false,
           "Get the laplacian matrix. Should be: p < k < q")
      .def("LaplacianSpectre", &SimplicialComplex::LaplacianSpectre, "k"_a, "p"_a, "q"_a,
           "weighted"_a,
           "Calculate the spectre of laplacian matrix. Should be: p < k < q")

      .def("BettiNumber", &SimplicialComplex::BettiNumber, "k"_a,
           "Get the k betti number of simplicial complex")
      .def("ClosenessAll", &SimplicialComplex::ClosenessAll, "k"_a, "q"_a,
           "weighted"_a = false,
           "Calculate closeness for all node with rank=k via nodes with rank=q")
      .def("BetweennessAll", &SimplicialComplex::BetweennessAll, "k"_a, "q"_a,
           "weighted"_a = false,
           "Calculate betweenness for all node with rank=k via nodes with "
           "rank=q")

      .def("GetMaxSimplices", &SimplicialComplex::GetMaxSimplices,
           "Get all maximum simplices of complex")
      .def("GetFixedDimElements", &SimplicialComplex::GetFixedDimElements,
           "dim"_a, "Get all simplices with given dimension")
      .def("GetAllSimplices", &SimplicialComplex::GetAllElements, "Get all simplices")
      .def("FVector", &SimplicialComplex::FVector,
           "Get the f-vector of the simplicial complex")
      .def("Dimension", &SimplicialComplex::Dimension,
           "Get the dimension of the simplicial complex - maximum dimension of "
           "simplices")
      .def("EulerCharacteristic", &SimplicialComplex::EulerCharacteristic,
           "Get the Euler characteristic of the simplicial complex")
      .def("Clear", &SimplicialComplex::Clear, "Clear the simplicial complex");

  py::class_<HyperGraph>(m, "HyperGraph")
      .def(py::init<>(), "Creating a new hypergraph")

      .def("UpdateWeight", &HyperGraph::UpdateWeight, "node"_a, "weight"_a,
           "Update the weight of a hyperedge in the hypergraph")
      .def("LaplacianMatrix", &HyperGraph::LaplacianMatrix, "k"_a, "p"_a, "q"_a,
           "weighted"_a = false,
           "Get the Laplacian matrix of the hypergraph. Should be: p < k < q")
      .def("LaplacianSpectre", &HyperGraph::LaplacianSpectre, "k"_a, "p"_a, "q"_a,
    "weighted"_a,
    "Calculate the spectre of laplacian matrix. Should be: p < k < q")
      .def("AddEdge", &HyperGraph::AddEdge, "edge"_a,
           "Add an hyperedge to the hypergraph")
      .def("RemoveEdge", &HyperGraph::RemoveEdge, "edge"_a,
           "Remove an hyperedge from the hypergraph")
      .def("BoundaryMatrix", &HyperGraph::BoundaryMatrix, "k"_a, "p"_a,
           "Get the boundary matrix of the hypergraph. Should be: k > p")
      .def("BettiNumber", &HyperGraph::BettiNumber, "k"_a,
           "Get the k-th Betti number of the hypergraph")
      .def("ClosenessAll", &HyperGraph::ClosenessAll, "k"_a, "q"_a,
           "weighted"_a = false,
           "Calculate the closeness centrality for all nodes with rank=k via "
           "nodes with rank=q")
      .def("BetweennessAll", &HyperGraph::BetweennessAll, "k"_a, "q"_a,
           "weighted"_a = false,
           "Calculate the betweenness centrality for all nodes with rank=k via "
           "nodes with rank=q")
      .def("GetEdges", &HyperGraph::GetEdges, "Get all edges in the hypergraph")
      .def("GetFixedDimElements", &HyperGraph::GetFixedDimElements, "rank"_a,
           "Get all elements of the hypergraph with the specified rank")
      .def("GetAllElements", &HyperGraph::GetAllElements, "Get all elements in the hypergraph")
      .def("FVector", &HyperGraph::FVector,
           "Get the F-vector of the hypergraph")
      .def("Dimension", &HyperGraph::Dimension,
           "Get the dimension of the hypergraph")
      .def("EulerCharacteristic", &HyperGraph::EulerCharacteristic,
           "Get the Euler characteristic of the hypergraph")

      .def("Clear", &HyperGraph::Clear, "Clear all elements in the hypergraph");

  py::class_<Graph>(m, "Graph")
      .def(py::init<>(), "Creating a new graph")

      .def("UpdateWeight", &Graph::UpdateWeight, "node"_a, "weight"_a,
           "Update the weight of a node/edge in the graph")
      .def("LaplacianMatrix", &Graph::LaplacianMatrix, "k"_a, "p"_a, "q"_a,
           "weighted"_a = false, "Get the Laplacian matrix of the graph")
      ..def("LaplacianSpectre", &Graph::LaplacianSpectre, "k"_a, "p"_a, "q"_a,
        "weighted"_a,
        "Calculate the spectre of laplacian matrix. Should be: p < k < q")
      .def("RemoveEdge", &Graph::RemoveEdge, "v"_a, "u"_a,
           "Remove an edge from the graph")
      .def("BoundaryMatrix", &Graph::BoundaryMatrix, "k"_a, "p"_a,
           "Get the boundary matrix of the graph. Should be: k > p")
      .def("BettiNumber", &Graph::BettiNumber, "k"_a,
           "Get the k-th Betti number of the graph")
      .def("ClosenessAll", &Graph::ClosenessAll, "k"_a, "q"_a,
           "weighted"_a = false,
           "Calculate the closeness centrality for all nodes with rank=k via "
           "nodes with rank=q")
      .def("BetweennessAll", &Graph::BetweennessAll, "k"_a, "q"_a,
           "weighted"_a = false,
           "Calculate the betweenness centrality for all nodes with rank=k via "
           "nodes with rank=q")
      .def("GetEdges", &Graph::GetEdges, "Get all edges in the graph")
      .def("GetFixedDimElements", &Graph::GetFixedDimElements, "rank"_a,
           "Get all elements of the graph with the specified rank")
      .def("GetAllElements", &Graph::GetAllElements, "Get all elements in the graph")
      .def("FVector", &Graph::FVector, "Get the F-vector of the graph")
      .def("Dimension", &Graph::Dimension, "Get the dimension of the graph")
      .def("EulerCharacteristic", &Graph::EulerCharacteristic,
           "Get the Euler characteristic of the graph")

      .def("Clear", &Graph::Clear, "Clear all elements in the graph");
}