use crate::graph::NodeIndex;
use crate::shortest_paths::ShortestPath;
use crate::util::NaturalOrInfinite;
use crate::Graph;
use std::collections::{HashMap, HashSet};

/// A tree represented by the set of its edges.
/// This struct should only be used to represent actual trees.
/// Its methods do **not** check this so you can have non-tree [EdgeTree]s during construction.
#[derive(Debug)]
pub struct EdgeTree {
    // edges are always ordered
    edges: HashSet<(NodeIndex, NodeIndex)>,
}

impl EdgeTree {
    /// Create a new [EdgeTree] from a path. Since [ShortestPath] does not store the start node,
    /// it has to be specified as a separate parameter.
    pub fn new(path: &ShortestPath, start: NodeIndex) -> Self {
        let edges = path.edges_on_path(start).collect::<HashSet<_>>();
        Self { edges }
    }

    /// Create an empty [EdgeTree].
    pub fn empty() -> Self {
        Self {
            edges: HashSet::new(),
        }
    }

    /// Return true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Add all edges of the `other` tree to this one.
    pub fn extend(&mut self, other: &Self) {
        self.edges.extend(other.edges.iter());
    }

    /// Add an edge to the tree.
    /// Requires that `a < b` where `edge = (a, b)`.
    pub fn insert(&mut self, edge: (NodeIndex, NodeIndex)) {
        assert!(edge.0 < edge.1);
        self.edges.insert(edge);
    }

    /// Remove an edge from the tree.
    /// Requires that `a < b` where `edge = (a, b)`.
    pub fn remove(&mut self, edge: (NodeIndex, NodeIndex)) {
        assert!(edge.0 < edge.1);
        self.edges.remove(&edge);
    }

    /// Calculate how much the tree weighs. `graph` should usually be the [Graph] from which the
    /// tree was constructed.
    pub fn weight_in(&self, graph: &Graph) -> NaturalOrInfinite {
        let mut weight = NaturalOrInfinite::from(0);
        for w in self.edges.iter().map(|&(a, b)| graph.weight(a, b)) {
            weight = weight + w;
        }
        weight
    }

    #[cfg(test)]
    pub(crate) fn nodes(&self) -> HashSet<NodeIndex> {
        self.edges
            .iter()
            .flat_map(|&(a, b)| std::iter::once(a).chain(std::iter::once(b)))
            .collect::<HashSet<_>>()
    }

    #[cfg(test)]
    pub(crate) fn neighbors(&self, node: NodeIndex) -> HashSet<NodeIndex> {
        let mut neighbors = HashSet::new();
        for &(from, to) in self.edges() {
            if from == node {
                neighbors.insert(to);
            }
            if to == node {
                neighbors.insert(from);
            }
        }
        neighbors
    }

    /// Find the leaves of the tree. Not efficient but useful for debug assertions/testing.
    #[cfg(debug_assertions)]
    pub fn find_leaves(&self) -> HashSet<NodeIndex> {
        let mut encountered: HashMap<NodeIndex, (u32, (NodeIndex, NodeIndex))> = HashMap::new();
        for &(from, to) in self.edges() {
            let edge = (from, to);
            for &n in &[from, to] {
                encountered.entry(n).or_insert((0, edge)).0 += 1;
            }
        }
        let mut leaves = HashSet::new();
        for node in encountered.keys() {
            let (count, _edge) = encountered[node];
            if count == 1 {
                leaves.insert(*node);
            }
        }
        leaves
    }

    pub(crate) fn edges(&self) -> &HashSet<(NodeIndex, NodeIndex)> {
        &self.edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_edge_tree() {
        let path = ShortestPath::new(vec![2, 3, 1], 10.into());
        let et = EdgeTree::new(&path, 4);
        assert_eq!(
            et.edges,
            [(2, 4), (2, 3), (1, 3)]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
        );
        let empty = EdgeTree::new(&ShortestPath::empty(), 0);
        assert_eq!(empty.edges, HashSet::new());
    }

    #[test]
    fn test_extend() {
        let path = ShortestPath::new(vec![2, 3], 10.into());
        let mut et = EdgeTree::new(&path, 4);
        let et2 = EdgeTree::new(&ShortestPath::new(vec![1, 5], 10.into()), 0);
        et.extend(&et2);
        assert_eq!(et.edges.len(), 4);
        assert_eq!(
            et.edges,
            [(2, 4), (2, 3), (0, 1), (1, 5)]
                .iter()
                .copied()
                .collect::<HashSet<_>>()
        );
    }
}
