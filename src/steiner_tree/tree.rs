use crate::graph::NodeIndex;
use crate::shortest_paths::ShortestPath;
use crate::util::NaturalOrInfinite;
use crate::Graph;
use std::collections::HashSet;
use std::iter;

#[derive(Debug)]
pub struct EdgeTree {
    // edges are always ordered
    edges: HashSet<(NodeIndex, NodeIndex)>,
}

impl EdgeTree {
    pub fn new(path: &ShortestPath, start: NodeIndex) -> Self {
        let edges = path.edges_on_path(start).collect::<HashSet<_>>();
        Self { edges }
    }

    pub fn empty() -> Self {
        Self {
            edges: HashSet::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    pub fn extend(&mut self, other: &Self) {
        self.edges.extend(other.edges.iter());
    }

    /// Requires that `a < b` where `edge = (a, b)`.
    pub fn insert(&mut self, edge: (NodeIndex, NodeIndex)) {
        assert!(edge.0 < edge.1);
        self.edges.insert(edge);
    }

    /// Requires that `a < b` where `edge = (a, b)`.
    pub fn remove(&mut self, edge: (NodeIndex, NodeIndex)) {
        assert!(edge.0 < edge.1);
        self.edges.remove(&edge);
    }

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
            .flat_map(|&(a, b)| iter::once(a).chain(iter::once(b)))
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
